from typing import Optional

import torch

from .wgan import ConvBlock, UpBlock, Scout, Critic


def compute_surprise_factor(
    real_rgb: torch.Tensor,
    pred_rgb: torch.Tensor,
    keep_channel: bool = False,
) -> torch.Tensor:
    """
    Per-pixel surprise = |real_rgb - pred_rgb|.
    If keep_channel=False (default), averages over RGB channels
    returning a 1-channel map. If True, returns the 3-channel
    per-channel absolute difference.
    """
    diff = (real_rgb - pred_rgb).abs()
    if keep_channel:
        return diff
    return diff.mean(dim=1, keepdim=True)


__all__ = [
    "ConvBlock",
    "UpBlock",
    "Scout",
    "Critic",
    "compute_surprise_factor",
]
