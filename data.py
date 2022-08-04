import os
from pathlib import Path
from typing import Tuple, Union
from interfaces import ITokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from torch import Tensor


class ArabicData(Dataset):
    def __init__(
            self,
            data_path: Union[str, Path],
            tokenizer: ITokenizer,
            pad_idx: int,
            max_len: int,
            ratio: float,
            dist_key: str,
            clean_key: str
            ) -> None:
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len + 2
        self.dist_key = dist_key
        self.clean_key = clean_key
        self.pad_idx = pad_idx
        self.max_dist_len = int(ratio * max_len) + max_len
        self.df = pd.read_csv(data_path)

    def pad(self, line: list, max_len: int) -> Tuple[list, int]:
        length = len(line)
        diff = max_len - length
        assert diff >= 0
        return line + [self.pad_idx] * diff, diff

    def _get_clean(self, idx: int) -> Tuple[Tensor, Tensor]:
        item = self.df.iloc[idx][self.clean_key]
        item = self.tokenizer.tokenize(
            item, add_sos=True, add_eos=True
            )
        mask = [False] * len(item)
        item, diff = self.pad(item, self.max_len)
        mask += [True] * diff
        item = torch.LongTensor(item)
        mask = torch.BoolTensor(mask)
        return item, mask

    def _get_distorted(self, idx: int) -> Tuple[Tensor, Tensor]:
        item = self.df.iloc[idx][self.dist_key]
        item = self.tokenizer.tokenize(item)
        mask = [False] * len(item)
        item, diff = self.pad(item, self.max_dist_len)
        mask += [True] * diff
        item = torch.LongTensor(item)
        mask = torch.BoolTensor(mask)
        return item, mask

    def __getitem__(self, idx: int):
        clean, clean_mask = self._get_clean(idx)
        distorted, distorted_mask = self._get_distorted(idx)
        return distorted, clean, distorted_mask, clean_mask

    def __len__(self):
        return self.df.shape[0]


def get_dist_data_laoder(
        data_path,
        tokenizer,
        max_len,
        ratio,
        batch_size,
        rank,
        world_size,
        dist_key,
        clean_key
        ):
    dataset = ArabicData(
        data_path=data_path,
        tokenizer=tokenizer,
        pad_idx=tokenizer.special_tokens.pad_id,
        max_len=max_len,
        ratio=ratio,
        dist_key=dist_key,
        clean_key=clean_key
    )
    sampler = DistributedSampler(
        dataset=dataset,
        rank=rank,
        num_replicas=world_size,
        drop_last=True
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=world_size,
        drop_last=True
        )


def get_data_laoder(
        data_path,
        tokenizer,
        batch_size,
        max_len,
        ratio,
        dist_key,
        clean_key
        ):
    dataset = ArabicData(
        data_path=data_path,
        tokenizer=tokenizer,
        pad_idx=tokenizer.special_tokens.pad_id,
        max_len=max_len,
        ratio=ratio,
        dist_key=dist_key,
        clean_key=clean_key
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True
        )


def get_train_test_loaders(args, rank: int, tokenizer: ITokenizer) -> tuple:
    assert os.path.exists(args.train_path), \
        f'{args.train_path} does not exist!'
    assert os.path.exists(args.test_path), \
        f'{args.test_path} does not exist!'
    train_loader = get_dist_data_laoder(
        data_path=args.train_path,
        tokenizer=tokenizer,
        max_len=args.max_len,
        ratio=args.distortion_ratio,
        batch_size=args.batch_size,
        rank=rank,
        world_size=args.n_gpus,
        dist_key=args.dist_key,
        clean_key=args.clean_key
    )
    test_loader = get_data_laoder(
        data_path=args.test_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_len=args.max_len,
        ratio=args.distortion_ratio,
        dist_key=args.dist_key,
        clean_key=args.clean_key
    )
    return train_loader, test_loader
