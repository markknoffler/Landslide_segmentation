from typing import Optional

import torch

from .wgan import ConvBlock, PCBlock, Scout, Critic


def compute_surprise_factor(
    real_rgb: torch.Tensor,
    pred_rgb: torch.Tensor,
    keep_channel: bool = False,
) -> torch.Tensor:
    diff = (real_rgb - pred_rgb).abs()
    if keep_channel:
        return diff
    return diff.mean(dim=1, keepdim=True)


__all__ = [
    "ConvBlock",
    "PCBlock",
    "Scout",
    "Critic",
    "compute_surprise_factor",
]
