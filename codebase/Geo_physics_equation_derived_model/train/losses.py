import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)
        target = target.float()
        probs = probs.reshape(-1)
        target = target.reshape(-1)
        tp = (probs * target).sum()
        fp = ((1.0 - target) * probs).sum()
        fn = (target * (1.0 - probs)).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class GeoPhysicsLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        main_weight: float = 1.0,
        aux2_weight: float = 0.6,
        aux3_weight: float = 0.4,
    ):
        super().__init__()
        self.main_weight = main_weight
        self.aux2_weight = aux2_weight
        self.aux3_weight = aux3_weight
        self.criterion = TverskyLoss(alpha=alpha, beta=beta)

    @staticmethod
    def _resize_target(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if target.shape[-2:] != pred.shape[-2:]:
            target = F.interpolate(target.float(), size=pred.shape[-2:], mode="nearest")
        return target

    def forward(self, outputs, target: torch.Tensor):
        """Same weighted Tversky sum as dual_stream_gated.losses.DualStreamLoss."""
        main, aux2, aux3 = outputs
        t_main = self._resize_target(target, main)
        loss_main = self.criterion(main, t_main)
        total = self.main_weight * loss_main
        loss_aux2 = None
        loss_aux3 = None
        if aux2 is not None:
            loss_aux2 = self.criterion(aux2, self._resize_target(target, aux2))
            total = total + self.aux2_weight * loss_aux2
        if aux3 is not None:
            loss_aux3 = self.criterion(aux3, self._resize_target(target, aux3))
            total = total + self.aux3_weight * loss_aux3
        return {
            "loss": total,
            "loss_main": loss_main.detach(),
            "loss_aux2": loss_aux2.detach() if loss_aux2 is not None else loss_main.new_zeros(()),
            "loss_aux3": loss_aux3.detach() if loss_aux3 is not None else loss_main.new_zeros(()),
        }
