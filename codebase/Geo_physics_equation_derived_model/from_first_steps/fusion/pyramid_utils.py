"""Helpers for cross-stream pyramid alignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def match_spatial(feat: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if feat.shape[-2:] == ref.shape[-2:]:
        return feat
    return F.interpolate(feat, size=ref.shape[-2:], mode="bilinear", align_corners=False)
