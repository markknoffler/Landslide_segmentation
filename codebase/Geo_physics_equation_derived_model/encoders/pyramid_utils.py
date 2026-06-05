"""Helpers for native EfficientNet pyramids and cross-stream alignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def group_norm_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels >= groups and channels % groups == 0:
            return groups
    return 1


def match_spatial(feat: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if feat.shape[-2:] == ref.shape[-2:]:
        return feat
    return F.interpolate(feat, size=ref.shape[-2:], mode="bilinear", align_corners=False)
