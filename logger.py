import os
from pathlib import Path
from typing import Any, Union
from interfaces import ILogger
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch import Tensor


class BasicLogger(ILogger):

    def __init__(
            self,
            log_dir: Union[str, Path],
            *args, **kwargs
            ) -> None:
        super().__init__()
        self.log_dir = log_dir
        self.__rank = 0
        self.__tag_format = '{}_{}'
        self.steps = {}

    def get_step(self, tag: str) -> int:
        step = self.steps.get(tag, 0)
        if step == 0:
            self.steps[tag] = 0
            self.steps[tag] += 1
            return step
        self.steps[tag] += 1
        return self.steps[tag]

    def _get_img_path(self, tag: str) -> Union[Path, str]:
        step = self.get_step(tag)
        path = f'{tag}_{step}.png'
        return os.path.join(self.log_dir, path), step

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
        print(f'{step} ==== ' + tag + f': {value} ', end=end)

    def log_img(self, tag: str, images: Tensor) -> None:
        n_imgs = images.shape[0]
        if n_imgs == 1:
            x = 1
            y = 1
        else:
            x = n_imgs // 2
            x += 0 if n_imgs % 2 == 0 else 1
            y = 2
        figure = plt.figure(figsize=(8, 8))
        for i in range(1, 1 + n_imgs):
            image = images[i - 1].squeeze().cpu().numpy()
            ax = figure.add_subplot(x, y, i)
            ax.pcolormesh(image)
            ax.xaxis.label.set_size(4)
            ax.yaxis.label.set_size(4)
        figure.subplots_adjust(
            left=0.1,
            bottom=0.1,
            right=0.9,
            top=0.9,
            wspace=0.4,
            hspace=0.4
            )
        path, step = self._get_img_path(tag)
        figure.savefig(path)
        return path, step


class TBLogger(BasicLogger):

    def __init__(
            self,
            log_dir: Union[str, Path],
            *args, **kwargs
            ) -> None:
        super().__init__(log_dir)
        self.writer = SummaryWriter(log_dir)

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

    def log_img(self, tag: str, images: Tensor) -> None:
        path, step = super().log_img(tag, images)
        img = plt.imread(path)[..., :3]
        plt.close()
        self.writer.add_image(tag, img, step, dataformats='HWC')


def get_logger(args):
    mapper = {
        'basic': BasicLogger,
        'tensor_board': TBLogger
    }
    if os.path.exists(args.logdir) is False:
        os.makedirs(args.logdir)
    return mapper[args.logger_type](args.logdir)
