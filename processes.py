from random import random
from typing import Union
import re
from typing import List
from interfaces import IProcess
from utils import load_text_file, remove_long_spaces


class LoadFile(IProcess):

    def execute(self, file_path: str):
        return load_text_file(
            file_path
            )


class LinesSplitter(IProcess):
    def __init__(self, sep: str) -> None:
        super().__init__()
        self.sep = sep

    def split(self, line):
        return line.split(self.sep)

    def execute(self, data: Union[List[str], str]) -> List[str]:
        if isinstance(data, str):
            return data.split(self.sep)
        results = []
        for lines in map(self.split, data):
            results.extend(lines)
        return results


class LengthFilter(IProcess):
    def __init__(
            self, min_length: int, max_length: int
            ) -> None:
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length

    def execute(self, lines: List[str]):
        return list(filter(
            lambda x: self.min_length <= len(x) <= self.max_length, lines
            ))


class CharsRemover(IProcess):
    def __init__(self, chars: str) -> None:
        super().__init__()
        self.pat = f'[{chars}]'

    def remove(self, line: str) -> str:
        return re.sub(self.pat, '', line)

    def execute(self, lines: List[str]) -> List[str]:
        return map(self.remove, lines)


class RepeatedCharsCollapsor(IProcess):
    def __init__(self, max_repeteion: int) -> None:
        super().__init__()
        self.pat = r"(.)\1{}".format(f"{{{2},}}")

    def collaps(self, line: str) -> str:
        return re.sub(self.pat, r"\1" * 1, line)

    def execute(self, lines: List[str]) -> List[str]:
        return list(map(self.collaps, lines))


class ValidCharsKeeper(IProcess):
    def __init__(self, valid_chars: str, rep_with=' ') -> None:
        super().__init__()
        self.valid_chars = valid_chars
        self.rep_with = rep_with
        self.pat = f'[^{self.valid_chars}]'

    def __keep(self, line: str) -> str:
        return re.sub(self.pat, ' ', line)

    def execute(self, lines: List[str]) -> List[str]:
        return list(map(self.__keep, lines))


class SpacesRemover(IProcess):

    def __remove(self, line: str) -> str:
        return remove_long_spaces(line).strip()

    def execute(self, lines: List[str]):
        return list(map(self.__remove, lines))


class RandomCharsInjector(IProcess):
    def __init__(self, chars: str) -> None:
        super().__init__()
        self.chars = chars

    def get_char(self) -> str:
        return random.choice(self.chars)

    def execute(self, line: str):
        length = len(line)
        idx = random.randint(0, length - 1)
        return line[:idx] + self.get_char() + line[idx:]


class RandomCharsSwapper(IProcess):

    def execute(self, line: str) -> str:
        length = len(line)
        idx = random.randint(0, length - 2)
        return line[:idx] + line[idx + 1] + line[idx] + line[idx + 2:]


class RandomCharRemover(IProcess):

    def execute(self, line: str) -> str:
        length = len(line)
        idx = random.randint(0, length - 1)
        return line[:idx] + line[idx + 1:]


class RandomWordsCollapsor(IProcess):

    def execute(self, line: str) -> str:
        indices = [
            i for i, char in enumerate(line)
            if char == ' '
            ]
        idx = random.choice(indices)
        return line[: idx] + line[idx + 1:]
