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
            dist_key='distorted',
            clean_key='clean'
            ) -> None:
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len + 2
        self.dist_key = dist_key
        self.clean_key = clean_key
        self.pad_idx = pad_idx
        self.max_dist_len = int(ratio * max_len) + max_len
        self.__data = []
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
        world_size
        ):
    dataset = ArabicData(
        data_path=data_path,
        tokenizer=tokenizer,
        pad_idx=tokenizer.special_tokens.pad_id,
        max_len=max_len,
        ratio=ratio
    )
    sampler = DistributedSampler(
        dataset=dataset,
        rank=rank,
        num_replicas=world_size
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
        ):
    dataset = ArabicData(
        data_path=data_path,
        tokenizer=tokenizer,
        pad_idx=tokenizer.special_tokens.pad_id,
        max_len=max_len,
        ratio=ratio
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True
        )
