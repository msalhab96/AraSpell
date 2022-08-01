from data import get_data_laoder, get_dist_data_laoder
from loss import Loss
from model import get_model
from optimizer import AdamWarmup
from processors import get_text_distorter
from tokenizer import get_tokenizer
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from tqdm import tqdm
import torch
import os 


class DistTrainer:
    _train_loss_key = 'train_loss'
    _test_loss_key = 'test_loss'
    def __init__(
            self,
            train_loader,
            test_loader,
            model,
            criterion,
            optimizer,
            epochs: int,
            url: str,
            backend: str,
            world_size: int,
            rank: int,
            ) -> None:
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
        self.init()
        
        self.model = DistributedDataParallel(
            self.model, device_ids=[self.rank]
            )
        self.history = dict()

    def log_results(self, epoch):
        result = ''
        for key, value in self.history.items():
            result += f'{key}: {str(value[-1])}, '
        print(result[:-2])

    def set_train_mode(self) -> None:
        self.model = self.model.train()

    def set_test_mode(self) -> None:
        self.model = self.model.eval()

    def init(self):
        os.environ['MASTER_ADDR'] = 'localhot'
        os.environ['MASTER_PORT'] = '8008'
        dist.init_process_group(
            self.backend,
            init_method=self.url,
            world_size=self.world_size,
            rank=self.rank
            )
        print(f'{self.rank} started!')

    def fit(self, *args, **kwargs):
        for epoch in range(self.epochs):
            self.train()
            if self.rank == 0:
                self.test()
                self.log_results(epoch)
        dist.destroy_process_group()

    def test(self):
        total_loss = []
        self.set_test_mode()
        for batch in tqdm(self.test_loader):
            (enc_inp, dec_inp, enc_mask, dec_mask) = batch
            # print(enc_inp.shape)
            enc_inp = enc_inp.cuda(self.rank)
            dec_inp = dec_inp.cuda(self.rank)
            preds, _ = self.model(enc_inp, dec_inp, enc_mask, dec_mask)
            loss = self.criterion(preds, dec_inp)
            # print(loss.item())
            total_loss.append(loss.item())
            # print(total_loss)
        # print(total_loss)
        # print(f'total is {sum(total_loss)}')
        # print(f'the length is {len(self.test_loader)}')
        total_loss = sum(total_loss)
        total_loss /= len(self.test_loader)
        if self._test_loss_key in self.history:
            self.history[self._test_loss_key].append(total_loss)
        else:
            self.history[self._test_loss_key] = [total_loss]

    def train(self):
        total_loss = 0
        self.set_train_mode()
        total = torch.tensor([0]).cuda(self.rank)
        # print(len(self.train_loader))
        for batch in tqdm(self.train_loader):
            (enc_inp, dec_inp, enc_mask, dec_mask) = batch
            
            # print(f'rank {self.rank}, batch {enc_inp.shape}')
            enc_inp = enc_inp.cuda(self.rank)
            dec_inp = dec_inp.cuda(self.rank)
            self.optimizer.zero_grad()
            preds, _ = self.model(enc_inp, dec_inp, enc_mask, dec_mask)
            loss = self.criterion(preds, dec_inp)
            # print(f'rank {self.rank}, loss: {loss.item()}')
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
        if self.rank == 0:
            print('-' * 10)
            # print(preds.shape)
            print(torch.argmax(preds, dim=-1)[: ,:50])
            print(dec_inp[:, :50])
            print('-' * 10)
            for p in self.model.parameters():
                print(p)
                break
        # total_loss /= self.world_size
        # print(f'rank {self.rank} || {total_loss}')
        total = torch.tensor([total_loss]).cuda(self.rank)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            total_loss /= len(self.train_loader)
            if self._train_loss_key in self.history:
                self.history[self._train_loss_key].append(total_loss)
            else:
                self.history[self._train_loss_key] = [total_loss]
            # print(len(self.train_loader))
            # print(total_loss)
            # print(self.history)

def get_trainer(rank: int, world_size):
    tokenizer = get_tokenizer()
    dist_proc = get_text_distorter(0.15)
    train_batch_size = 16
    test_batch_size = 8
    train_loader = get_dist_data_laoder(
        'train.txt',
        dist_proc,
        tokenizer,
        train_batch_size,
        rank=rank,
        world_size=world_size
    )
    test_loader = get_data_laoder(
        'test.txt',
        dist_proc,
        tokenizer,
        test_batch_size
    )
    model = get_model(rank, tokenizer.vocab_size)
    criterion = Loss(tokenizer.special_tokens.pad_id)
    optimizer = AdamWarmup(model.parameters(), betas=[0.9, 0.98], eps=1e-9, warmup_staps=4000, d_model=512)
    epochs = 300
    import socket
    
    url = f'tcp://{socket.gethostbyname(socket.gethostname())}:8008'
    backend = 'nccl'
    return DistTrainer(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        backend=backend,
        world_size=world_size,
        rank=rank,
        url=url
    )
    
def main(rank, world_size):
    print(f'started_rank {rank} with world_size {world_size}')
    trainer = get_trainer(
        rank=rank,
        world_size=world_size
    )
    trainer.fit()
    
    
if __name__ == '__main__':
    from torch.multiprocessing import spawn
    spawn(
        main,
        nprocs=2,
        args=(2,)
        )