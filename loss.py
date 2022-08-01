import torch.nn as nn
from torch import Tensor


class Loss(nn.Module):
    def __init__(self, pad_idx: int) -> None:
        super().__init__()
        self.cel = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, results: Tensor, targets: Tensor) -> Tensor:
        # results of shape [B, M, voc_size]
        # targets of shape [B, M]
        results = results[:, :-1, :]
        targets = targets[:, 1:]
        v = results.shape[-1]
        targets = targets.reshape(-1)
        results = results.reshape(-1, v)
        return self.cel(results, targets)
