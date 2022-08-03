import random
import re
from typing import List, Union
from interfaces import IProcess
from utils import get_freq_dict, load_text_file, remove_long_spaces


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


class WordsNumberFilter(IProcess):
    def __init__(self, min_words: int, max_words: int) -> None:
        super().__init__()
        self.min_words = min_words
        self.max_words = max_words

    def _is_valid(self, line: str) -> bool:
        return self.min_words < line.count(' ') < self.max_words

    def execute(self, lines: List[str]):
        return list(filter(self._is_valid, lines))


class WordsFilter(IProcess):
    def __init__(self, words: List[str]) -> None:
        super().__init__()
        self.words = set(words)

    def _not_contain(self, line: str) -> bool:
        return not any((
            word in line for word in self.words
            ))

    def execute(self, lines: List[str]):
        return list(filter(self._not_contain, lines))


class SoloCharFilter(IProcess):

    def _not_contain(self, line: str) -> bool:
        return re.search('^. | . | .$', line) is None

    def execute(self, lines: List[str]):
        return list(filter(self._not_contain, lines))


class NumbersFilter(IProcess):

    def _not_contain(self, line: str) -> bool:
        return re.search('[0-9]+', line) is None

    def execute(self, lines: List[str]):
        return list(filter(self._not_contain, lines))


class OOVFilter(IProcess):
    def __init__(self, max_oov: int) -> None:
        super().__init__()
        self.max_oov = max_oov
        self.__freq = {}

    def _is_valid(self, line: str):
        counter = 0
        for word in line.split(' '):
            counter += (self.__freq[word] == 1)
        return counter < self.max_oov

    def execute(self, lines: List[str]):
        self.__freq = get_freq_dict(lines)
        return list(filter(self._is_valid, lines))


class CharsRemover(IProcess):
    def __init__(self, chars: str) -> None:
        super().__init__()
        self.pat = f'[{chars}]'

    def remove(self, line: str) -> str:
        return re.sub(self.pat, '', line)

    def execute(self, lines: List[str]) -> List[str]:
        return list(map(self.remove, lines))


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
        if len(indices) == 0:
            return line
        idx = random.choice(indices)
        return line[: idx] + line[idx + 1:]
