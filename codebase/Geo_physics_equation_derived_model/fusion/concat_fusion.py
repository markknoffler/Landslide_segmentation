"""Simple tri-stream fusion: concat encoder features then 1x1 project to C."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcatTriStreamLevel(nn.Module):
    """Concatenate RGB-physics, DEM-physics, and FM at one pyramid level."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        t_fm: torch.Tensor,
        x_rgb: torch.Tensor,
        x_dem: torch.Tensor,
    ) -> torch.Tensor:
        return self.proj(torch.cat([x_rgb, x_dem, t_fm], dim=1))


class ConcatTriStreamSkip(nn.Module):
    """Concat fusion skip at pyramid level L."""

    def __init__(self, channels: int):
        super().__init__()
        self.level = ConcatTriStreamLevel(channels)

    def forward(
        self,
        p_rgb: list[torch.Tensor],
        p_dem: list[torch.Tensor],
        t_fm: list[torch.Tensor],
        level: int,
    ) -> torch.Tensor:
        level = max(0, min(level, len(p_rgb) - 1))
        return self.level(t_fm[level], p_rgb[level], p_dem[level])
