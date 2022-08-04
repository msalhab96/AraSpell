from functools import lru_cache
import json
import math
import re
from typing import List, Union
from pathlib import Path
import torch
from torch import Tensor


def load_text_file(
        file_path: Union[Path, str],
        encoding='utf-8',
        *args, **kwargs
        ) -> str:
    with open(file_path, 'r', encoding=encoding) as f:
        data = f.read()
    return data


def save_text_file(
        file_path: Union[Path, str],
        data: str,
        encoding='utf-8'
        ) -> str:
    with open(file_path, 'w', encoding=encoding) as f:
        data = f.write(data)
    return data


def remove_long_spaces(line: str) -> str:
    return re.sub('\s{2,}', ' ', line)


@lru_cache(maxsize=2)
def get_positionals(max_length: int, d_model: int) -> Tensor:
    """Create Positionals tensor to be added to the input
    Args:
        max_length (int): The maximum length of the positionals sequence.
        d_model (int): The dimensionality of the positionals sequence.
    Returns:
        Tensor: Positional tensor
    """
    result = torch.zeros(max_length, d_model, dtype=torch.float)
    for pos in range(max_length):
        for i in range(0, d_model, 2):
            denominator = pow(10000, 2 * i / d_model)
            result[pos, i] = math.sin(pos / denominator)
            result[pos, i + 1] = math.cos(pos / denominator)
    return result


def load_json(file_path: Union[Path, str]) -> Union[dict, list]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(
        file_path: Union[Path, str], data: Union[dict, list]
        ) -> None:
    with open(file_path, 'w') as f:
        json.dump(data, f)


def get_freq_dict(data: List[str]) -> dict:
    freq = {}
    for item in data:
        for word in item.split(' '):
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
    return freq


def load_state(state_path: Union[Path, str]):
    state = torch.load(state_path)
    model = state['model']
    model = {
        key.replace('module.', ''): value
        for key, value in model.items()
        }
    optimizer = state['optimizer']
    epoch = state['epoch']
    steps = state['steps']
    return model, optimizer, epoch, steps
