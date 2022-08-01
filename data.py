from pathlib import Path
from typing import Tuple, Union
from interfaces import IProcessor, ITokenizer
from torch.utils.data import Dataset, DataLoader
from utils import load_text_file
import torch
from torch.utils.data.distributed import DistributedSampler


class ArabicData(Dataset):
    def __init__(
            self,
            data_path: Union[str, Path],
            distortion_processor: IProcessor,
            tokenizer: ITokenizer,
            batch_size: int,
            pad_idx: int,
            hold=True
            ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data = load_text_file(data_path).split('\n')
        self.lengths = list(map(len, self.data))
        self.dist_proc = distortion_processor
        ratio = distortion_processor.ratio
        self.max_lengths = list(map(
            lambda l: l + int(ratio * l),
            self.lengths
            ))
        self.batch_size = batch_size
        self._cache = []
        self.pad_idx = pad_idx

    def pad(self, line: list, max_len: int) -> Tuple[list, int]:
        length = len(line)
        diff = max_len - length
        assert diff >= 0
        return line + [self.pad_idx] * diff, diff

    def __getitem__(self, idx: int):
        # TODO: Refactor this part of code
        if idx < len(self._cache):
            return self._cache[idx]
        item = self.data[idx]
        dist_item = self.dist_proc.run(item)
        item = self.tokenizer.tokenize(item, add_sos=True, add_eos=True)
        max_len = 800
        item, diff = self.pad(item, max_len)
        item_mask = [False] * (max_len - diff) + [True] * diff
        max_len = 900
        dist_item = self.tokenizer.tokenize(dist_item)
        dist_item, diff = self.pad(dist_item, 900)
        dist_item_mask = [False] * (max_len - diff) + [True] * diff
        item = torch.LongTensor(item)
        dist_item = torch.LongTensor(dist_item)
        item_mask = torch.BoolTensor(item_mask)
        dist_item_mask = torch.BoolTensor(dist_item_mask)
        self._cache.append((dist_item, item, dist_item_mask, item_mask))
        return dist_item, item, dist_item_mask, item_mask

    def __len__(self):
        return len(self.data)


def get_dist_data_laoder(
        data_path,
        distortion_processor,
        tokenizer,
        batch_size,
        rank,
        world_size
        ):
    dataset = ArabicData(
        data_path=data_path,
        distortion_processor=distortion_processor,
        tokenizer=tokenizer,
        batch_size=batch_size,
        pad_idx=tokenizer.special_tokens.pad_id
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
        distortion_processor,
        tokenizer,
        batch_size
        ):
    dataset = ArabicData(
        data_path=data_path,
        distortion_processor=distortion_processor,
        tokenizer=tokenizer,
        batch_size=batch_size,
        pad_idx=tokenizer.special_tokens.pad_id
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True
        )