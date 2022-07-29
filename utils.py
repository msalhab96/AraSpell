import re
from typing import Union
from pathlib import Path


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
