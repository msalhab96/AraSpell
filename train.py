from pathlib import Path
from time import sleep
from typing import Union
from args import get_train_args
from callback import TermCallback, get_callback
from data import get_train_test_loaders
from interfaces import ILogger
from logger import get_logger
from loss import get_criterion
from models import get_model
from optimizer import get_optimizer
from tokenizer import get_tokenizer
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from tqdm import tqdm
import torch
import os
from utils import load_state
import socket
from torch.multiprocessing import spawn
import torch.nn as nn


class DistTrainer:
    _train_loss_key = 'train_loss'
    _acc_train_loss_key = 'acc_train_loss'
    _test_loss_key = 'test_loss'
    _acc_test_loss_key = 'acc_test_loss'

    def __init__(
            self,
            train_loader,
            test_loader,
            model,
            criterion,
            optimizer,
            epochs: int,
            callback: TermCallback,
            logger: ILogger,
            outdir: Union[str, Path],
            url: str,
            backend: str,
            world_size: int,
            rank: int,
            port: int,
            clip_grad: bool,
            grad_norm=None,
            ckpt=None
            ) -> None:
        self.callback = callback
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.rank = rank
        self.url = url
        self.backend = backend
        self.world_size = world_size
        self.model.cuda(self.rank)
        self.port = port
        self.__counter = 0
        self.outdir = outdir
        self.last_epoch = 0
        self.logger.set_rank(self.rank)
        if ckpt is not None:
            self._set_state(ckpt)
        self.init()
        self.model = DistributedDataParallel(
            self.model, device_ids=[self.rank]
            )
        self.history = dict()
        self.grad_norm = grad_norm
        self.clip_grad = clip_grad

    def _set_state(self, ckpt_path):
        model, optimizer, epoch, steps = load_state(ckpt_path)
        self.model.load_state_dict(model)
        self.optimizer.load_state_dict(optimizer, steps)
        self.last_epoch = epoch + 1

    @property
    def is_master(self):
        return self.rank == 0

    def log_results(self, epoch: int):
        if self._train_loss_key in self.history:
            self.logger.log(
                key=self._acc_train_loss_key,
                value=self.history[self._train_loss_key][-1],
                step=epoch,
                end=''
            )
        if self._test_loss_key in self.history:
            self.logger.log(
                key=self._acc_test_loss_key,
                value=self.history[self._test_loss_key][-1]
            )

    def set_train_mode(self) -> None:
        self.model = self.model.train()

    def set_test_mode(self) -> None:
        self.model = self.model.eval()

    def init(self):
        os.environ['MASTER_ADDR'] = self.url
        os.environ['MASTER_PORT'] = str(self.port)
        dist.init_process_group(
            self.backend,
            init_method=self.url,
            world_size=self.world_size,
            rank=self.rank
            )

    def _get_ckpt_state(self, epoch: int):
        return {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'steps': self.optimizer.counter
        }

    def save_ckpt(self, epoch: int) -> None:
        state = self._get_ckpt_state(epoch)
        path = os.path.join(self.outdir, f'checkpoint_{epoch}.pt')
        torch.save(state, path)
        print(f'checkpoint {path} saved!')

    def test_and_log(self, epoch):
        if self.is_master:
            self.test()
            self.log_results(epoch)
            save_ckpt, terminate = self.callback(
                self.history[self._test_loss_key][-1]
                )
            if epoch == -1:
                # When the model just loadded and no trainin introduced
                return
            if save_ckpt is True:
                self.save_ckpt(epoch)
            if terminate is True:
                print('The model is not improving any more!')
                print('terminated!')
                exit()

    def fit(self, *args, **kwargs):
        # test the model before training
        self.test_and_log(-1)
        for epoch in range(self.last_epoch, self.epochs):
            self.train_loader.sampler.set_epoch(epoch)
            self.train()
            self.test_and_log(epoch)
            dist.barrier()
        dist.destroy_process_group()

    @torch.no_grad()
    def test(self):
        total_loss = []
        self.set_test_mode()
        for batch in tqdm(self.test_loader):
            self.__counter += 1
            (enc_inp, dec_inp, enc_mask, dec_mask) = batch
            enc_inp = enc_inp.cuda(self.rank)
            dec_inp = dec_inp.cuda(self.rank)
            preds, att = self.model(enc_inp, dec_inp, enc_mask, dec_mask)
            loss = self.criterion(preds, dec_inp, dec_mask)
            total_loss.append(loss.item())
        total_loss = sum(total_loss)
        total_loss /= len(self.test_loader)
        if self._test_loss_key in self.history:
            self.history[self._test_loss_key].append(total_loss)
        else:
            self.history[self._test_loss_key] = [total_loss]
        h = att.shape[0] // dec_inp.shape[0]
        self.logger.log_img('enc_dec_att', att[:h, ...])

    def train(self):
        total_loss = 0
        self.set_train_mode()
        total = torch.tensor([0]).cuda(self.rank)
        for batch in tqdm(self.train_loader, total=len(self.train_loader)):
            (enc_inp, dec_inp, enc_mask, dec_mask) = batch
            enc_inp = enc_inp.cuda(self.rank)
            dec_inp = dec_inp.cuda(self.rank)
            self.optimizer.zero_grad()
            preds, att = self.model(enc_inp, dec_inp, enc_mask, dec_mask)
            loss = self.criterion(preds, dec_inp, dec_mask)
            loss.backward()
            if self.clip_grad is True:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_norm
                    )
            self.optimizer.step()
            self.logger.log_step(self._train_loss_key, loss.item())
            total_loss += loss.item()
        total = torch.tensor([total_loss]).cuda(self.rank)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            total_loss /= len(self.train_loader)
            if self._train_loss_key in self.history:
                self.history[self._train_loss_key].append(total_loss)
            else:
                self.history[self._train_loss_key] = [total_loss]


def get_trainer(rank: int, args):
    if rank != 0:
        sleep(2)
    callback = get_callback(args)
    logger = get_logger(args)
    tokenizer = get_tokenizer(args)
    vocab_size = tokenizer.vocab_size
    train_loader, test_loader = get_train_test_loaders(args, rank, tokenizer)
    model = get_model(
        args,
        rank,
        vocab_size,
        pad_idx=tokenizer.special_tokens.pad_id
        )
    criterion = get_criterion(args, vocab_size)
    optimizer = get_optimizer(args, model.parameters())
    url = 'tcp://{}:{}'.format(
        socket.gethostbyname(socket.gethostname()),
        args.dist_port
    )
    return DistTrainer(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        callback=callback,
        logger=logger,
        backend=args.dist_backend,
        world_size=args.n_gpus,
        outdir=args.outdir,
        rank=rank,
        port=args.dist_port,
        url=url,
        ckpt=args.pre_trained_path,
        grad_norm=args.grad_norm,
        clip_grad=args.clip_grad
    )


def run(rank, args):
    print(f'{rank} started ..')
    trainer = get_trainer(
        rank=rank,
        args=args
    )
    trainer.fit()


def main(args):
    spawn(
        run,
        nprocs=args.n_gpus,
        args=(args,)
        )


if __name__ == '__main__':
    args = get_train_args()
    print(args)
    main(args)
