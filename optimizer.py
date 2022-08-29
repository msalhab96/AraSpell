from torch.optim import Adam
from typing import Iterator, Tuple
import math


class AdamWarmup:
    def __init__(
            self,
            parameters: Iterator,
            betas: Tuple[float, float],
            eps: float,
            warmup_staps: int,
            d_model: int,
            *args,
            **kwargs
            ):
        self.optimizer = Adam(
            parameters,
            betas=betas,
            eps=eps
        )
        self.warmup_staps = warmup_staps
        self.d_model = d_model
        self.peak = 1 / math.sqrt(self.d_model)
        self.inv_warmup_staps = 1 / math.sqrt(self.warmup_staps ** 3)
        self.counter = 0
        self._update_lr()

    def get_lr(self, step: int) -> float:
        return self.peak * min(
            1 / math.sqrt(step),
            step * self.inv_warmup_staps
        )

    def _update_lr(self) -> None:
        self.counter += 1
        lr = self.get_lr(self.counter)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step(self) -> None:
        self.optimizer.step()
        self._update_lr()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, counter) -> None:
        self.optimizer.load_state_dict(state_dict)
        self.counter = counter


class AdamOptim(Adam):
    def __init__(self, parameters, lr=0.001, *args, **kwargs) -> None:
        super().__init__(parameters, lr=lr)
        self.counter = 0

    def load_state_dict(self, state_dict, *args, **kwargs) -> None:
        super().load_state_dict(state_dict)


class AdamExpDecay:
    def __init__(
            self,
            parameters: Iterator,
            lr: float,
            decay_rate: int,
            *args,
            **kwargs
            ):
        self.optimizer = Adam(
            parameters,
            lr=lr
        )
        self.lr = lr
        self.decay_rate = decay_rate
        self.counter = 0

    def get_lr(self, step: int) -> float:
        return self.lr * (1 - (self.decay_rate/100)) ** step

    def _update_lr(self) -> None:
        self.counter += 1
        lr = self.get_lr(self.counter)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step(self) -> None:
        self.optimizer.step()
        self._update_lr()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, counter) -> None:
        self.optimizer.load_state_dict(state_dict)
        self.counter = counter


def get_optimizer(args, parameters: Iterator) -> object:
    mapper = {
        'adam': AdamOptim,
        'adamw': AdamWarmup,
        'adamexp': AdamExpDecay
    }
    return mapper[args.optim](
        parameters=parameters,
        betas=args.opt_betas,
        eps=args.opt_eps,
        warmup_staps=args.warmup_staps,
        d_model=args.d_model,
        lr=args.lr,
        decay_rate=args.decay_rate
    )
