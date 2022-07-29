from pathlib import Path
from typing import Union, Any, List
from interfaces import IProcess, IProcessor
import concurrent.futures


class FilesProcessor(IProcessor):
    def __init__(
            self, processes: IProcess
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
