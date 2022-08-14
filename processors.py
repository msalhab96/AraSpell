from threading import Thread
import constants
from pathlib import Path
import random
from typing import Union, Any, List
from interfaces import IProcess, IProcessor
from processes import (
    RandomCharRemover,
    RandomCharsInjector,
    RandomCharsSwapper,
    RandomNeighborReplacer,
    RandomWordsCollapsor
    )


class FilesProcessor(IProcessor):
    def __init__(
            self, processes: List[IProcess],
            n_dist: int = 32
            ) -> None:
        self.processes = processes
        self.n_dist = n_dist
        self.__dist = False
        self.__cache = []

    def file_run(self, file: Union[str, Path]) -> Any:
        result = file
        for process in self.processes:
            result = process.execute(result)
        return result

    def run(
            self,
            files: List[Union[str, Path]]
            ) -> Any:
        result = list(map(self.file_run, files))
        if self.__dist is True:
            self.__cache.append(result)
            return
        return result

    def _divde(self, data: List[Any]):
        items_per_div = len(data) // self.n_dist
        divs = []
        for i in range(items_per_div):
            start = i * items_per_div
            end = (i + 1) * items_per_div
            if i == (items_per_div - 1):
                end = len(divs)
            divs.append(data[start: end])
        return divs

    def dist_run(
            self,
            files: List[Union[str, Path]]
            ) -> Any:
        self.__dist = True
        self.__cache = []
        divs = self._divde(files)
        threads = []
        for div in divs:
            t = Thread(target=self.run, args=(div,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        self.__dist = False
        results = []
        for item in self.__cache:
            results.extend(item)
        self.__cache = []
        return results


class TextDistorter(IProcessor):
    def __init__(
            self, ratio: float, processes: List[IProcess]
            ) -> None:
        super().__init__()
        self.ratio = ratio
        self.processes = processes

    def run(self, line: str) -> str:
        length = len(line)
        n = int(self.ratio * length)
        for _ in range(n):
            line = random.choice(self.processes).execute(line)
        return line

    def dist_run(self):
        # TODO
        pass


class TextProcessor(IProcessor):
    def __init__(self, processes: List[IProcess]) -> None:
        super().__init__()
        self.processes = processes

    def  run(self, sentence: str):
        for process in self.processes:
            sentence = process.execute(sentence)
        return sentence

    def dist_run(self, sentence: str) -> str:
        return self.run(sentence)


def get_text_distorter(ratio):
    return TextDistorter(
        ratio=ratio,
        processes=[
            RandomCharsInjector(constants.VALID_CHARS),
            RandomCharsSwapper(),
            RandomCharRemover(),
            RandomWordsCollapsor(),
            RandomNeighborReplacer(
                constants.KEYBOARD_KEYS, constants.KEYBOARD_BLANK
                )
        ]
    )
