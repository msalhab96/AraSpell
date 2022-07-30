from functools import lru_cache
import math
import re
from typing import Union
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
