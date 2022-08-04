import os
from pathlib import Path
from typing import Any, Union
from interfaces import ILogger
from torch.utils.tensorboard import SummaryWriter


class BasicLogger(ILogger):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.__rank = 0
        self.__tag_format = '{}_{}'

    def _get_tag(self, key: str) -> str:
        return self.__tag_format.format(key, self.__rank)

    def set_rank(self, rank: int):
        self.__rank = rank

    def log_step(self, key, value):
        tag = self._get_tag(key)
        print(tag + f': {value}')

    def log(
            self,
            key: str,
            value: Any,
            step: Union[None, int] = None,
            end='\n'
            ):
        tag = self._get_tag(key)
        if step is None:
            print(f'{tag}' + f': {value} ', end=end)
            return
        print(f'{step}== ' + tag + f': {value} ', end=end)


class TBLogger(BasicLogger):

    def __init__(
            self,
            log_dir: Union[str, Path],
            *args, **kwargs
            ) -> None:
        super().__init__()
        self.writer = SummaryWriter(log_dir)
        self.steps = {}

    def get_step(self, tag: str) -> int:
        step = self.steps.get(tag, 0)
        if step == 0:
            self.steps[tag] = 0
            self.steps[tag] += 1
            return step
        self.steps[tag] += 1
        return self.steps[tag]

    def log_step(self, key: str, value: Any):
        tag = self._get_tag(key)
        self.writer.add_scalar(tag, value, self.get_step(tag))

    def log(
            self,
            key: str,
            value: Any,
            step: Union[None, int] = None,
            end='\n'
            ):
        tag = self._get_tag(key)
        super().log(
            key=key, value=value, step=step, end=end
            )
        self.writer.add_scalar(tag, value, self.get_step(tag))


def get_logger(args):
    mapper = {
        'basic': BasicLogger,
        'tensor_board': TBLogger
    }
    if os.path.exists(args.logdir) is False:
        os.makedirs(args.logdir)
    return mapper[args.logger_type](args.logdir)
