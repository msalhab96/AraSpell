import constants
from pathlib import Path
import random
from typing import Union, Any, List
from interfaces import IProcess, IProcessor
import concurrent.futures

from processes import RandomCharRemover, RandomCharsInjector, RandomCharsSwapper, RandomWordsCollapsor


class FilesProcessor(IProcessor):
    def __init__(
            self, processes: List[IProcess]
            ) -> None:
        self.processes = processes

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
        return result

    def dist_run(
            self,
            files: List[Union[str, Path]]
            ) -> Any:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = executor.map(self.file_run, files)
        return list(result)


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


def get_text_distorter(ratio):
    return TextDistorter(
        ratio=ratio,
        processes=[
            RandomCharsInjector(constants.VALID_CHARS),
            RandomCharsSwapper(),
            RandomCharRemover(),
            RandomWordsCollapsor()
        ]
    )