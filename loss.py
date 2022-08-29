import torch.nn as nn
from torch import Tensor
from torch.nn.functional import one_hot
from torch.nn import Module


class Loss(nn.Module):
    def __init__(
            self,
            voc_size: int,
            alpha=0.1
            ) -> None:
        super().__init__()
        self.kld = nn.KLDivLoss(reduction='batchmean')
        self.voc_size = voc_size
        self.alpha = alpha

    def process_target(self, target: Tensor, mask: Tensor) -> Tensor:
        """Processes the target tensor of shape [B, M]
        by converting the sparse target into uniform distribution
        using lable smoothing.

        Args:
            target (Tensor): The target Tensor of shape [B, M]
            mask (Tensor): The target mask tensor of shape [B, M]
            where it's True whereever there's a padding.

        Returns:
            Tensor: The processed Tensor of shape [B, M - 1, V]
        """
        target = target[:, 1:]
        target = target.reshape(-1)
        target = one_hot(
            target, num_classes=self.voc_size
            ).to(target.device)
        weight = self.alpha / self.voc_size
        target = target * (1 - self.alpha) + weight
        target = (~mask.view(-1, 1)) * target
        return target

    def procees_preds(self, preds: Tensor, mask: Tensor) -> Tensor:
        """Processes the model's results tensor of shape [B, M, V]
        by conducting masking and tensor reshaping to the original tensor

        Args:
            preds (Tensor): The model's results of shape [B, M, V].
            mask (Tensor): The masking tensor of shape [B, M-1, V].

        Returns:
            Tensor: The processed tensor of shape [B, M-1, V]
        """
        preds = preds[:, :-1, :]
        preds = preds.reshape(-1, preds.shape[-1])
        preds = (~mask.reshape(-1, 1)) * preds
        return preds

    def forward(
            self,
            preds: Tensor,
            target: Tensor,
            mask: Tensor
            ) -> Tensor:
        # results of shape [B, M, voc_size]
        # targets of shape [B, M]
        max_len = preds.shape[1]
        mask = mask[:, :max_len]
        target = target[:, :max_len]
        mask = mask[:, 1:]
        mask = mask.to(preds.device)
        target = self.process_target(target, mask)
        preds = self.procees_preds(preds, mask)
        return self.kld(preds, target)


def get_criterion(args, voc_szie: int) -> Module:
    return Loss(voc_szie, args.alpha)
