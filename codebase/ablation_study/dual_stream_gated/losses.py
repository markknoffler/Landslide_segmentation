from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        probs = torch.sigmoid(logits)
        target = target.float()

        dims = (0, 2, 3)
        tp = torch.sum(probs * target, dims)
        fp = torch.sum(probs * (1.0 - target), dims)
        fn = torch.sum((1.0 - probs) * target, dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class DualStreamLoss(nn.Module):
    """
    L = lambda0 * T(main) + lambda1 * T(aux2) + lambda2 * T(aux3) + lambda_r * sum(gate_reg)
    """

    def __init__(
        self,
        lambda0: float = 1.0,
        lambda1: float = 0.5,
        lambda2: float = 0.5,
        lambda_r: float = 1e-3,
        alpha: float = 0.3,
        beta: float = 0.7,
    ):
        super().__init__()
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_r = lambda_r
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)

    def _resize_target(self, target: torch.Tensor, pred: torch.Tensor):
        if target.shape[-2:] != pred.shape[-2:]:
            target = F.interpolate(target.float(), size=pred.shape[-2:], mode="nearest")
        return target

    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor):
        target_main = self._resize_target(target, outputs["main"])
        target_aux2 = self._resize_target(target, outputs["aux2"])
        target_aux3 = self._resize_target(target, outputs["aux3"])

        l_main = self.tversky(outputs["main"], target_main)
        l_aux2 = self.tversky(outputs["aux2"], target_aux2)
        l_aux3 = self.tversky(outputs["aux3"], target_aux3)

        reg = torch.stack(outputs.get("gate_regs", [torch.tensor(0.0, device=target.device)])).sum()
        total = self.lambda0 * l_main + self.lambda1 * l_aux2 + self.lambda2 * l_aux3 + self.lambda_r * reg
        return {
            "loss": total,
            "loss_main": l_main.detach(),
            "loss_aux2": l_aux2.detach(),
            "loss_aux3": l_aux3.detach(),
            "loss_reg": reg.detach(),
        }
