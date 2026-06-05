"""Simple tri-stream fusion: concat encoder features then 1x1 project."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..encoders.pyramid_utils import group_norm_groups, match_spatial


class ConcatTriStreamLevel(nn.Module):
    """Concatenate RGB-physics, DEM-physics, and FM at one pyramid level."""

    def __init__(self, physics_channels: int, fm_channels: int | None = None):
        super().__init__()
        out_channels = fm_channels if fm_channels is not None else physics_channels
        self.physics_channels = physics_channels
        self.out_channels = out_channels
        self.rgb_proj = (
            nn.Identity()
            if physics_channels == out_channels
            else nn.Conv2d(physics_channels, out_channels, kernel_size=1, bias=False)
        )
        self.dem_proj = (
            nn.Identity()
            if physics_channels == out_channels
            else nn.Conv2d(physics_channels, out_channels, kernel_size=1, bias=False)
        )
        in_concat = out_channels * 3
        self.proj = nn.Sequential(
            nn.Conv2d(in_concat, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(group_norm_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        t_fm: torch.Tensor,
        x_rgb: torch.Tensor,
        x_dem: torch.Tensor,
    ) -> torch.Tensor:
        x_rgb = match_spatial(self.rgb_proj(x_rgb), t_fm)
        x_dem = match_spatial(self.dem_proj(x_dem), t_fm)
        return self.proj(torch.cat([x_rgb, x_dem, t_fm], dim=1))


class ConcatTriStreamSkip(nn.Module):
    """Concat fusion skip at pyramid level L."""

    def __init__(self, physics_channels: int, fm_channels: int | None = None):
        super().__init__()
        self.level = ConcatTriStreamLevel(physics_channels, fm_channels)

    def forward(
        self,
        p_rgb: list[torch.Tensor],
        p_dem: list[torch.Tensor],
        t_fm: list[torch.Tensor],
        level: int,
    ) -> torch.Tensor:
        level = max(0, min(level, len(p_rgb) - 1))
        return self.level(t_fm[level], p_rgb[level], p_dem[level])
