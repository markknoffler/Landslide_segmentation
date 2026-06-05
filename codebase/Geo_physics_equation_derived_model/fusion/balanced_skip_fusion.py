"""Symmetric skip fusion across RGB-physics, DEM-physics, and FM pyramid levels."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.pyramid_utils import group_norm_groups, match_spatial
from .intra_stream import IntraStreamBlock


class BalancedTriStreamSkip(nn.Module):
    """
    Per-level skip builder: intra-stream refine each encoder, then symmetric gating.

    No cross-encoder attention here (keeps skips efficient); cross fusion happens at L3/L4.
    """

    def __init__(
        self,
        physics_channels: int,
        fm_channels: int | None = None,
        num_heads: int = 4,
        attn_spatial_max: int = 4096,
    ):
        super().__init__()
        channels = fm_channels if fm_channels is not None else physics_channels
        self.channels = channels
        self.rgb_proj = (
            nn.Identity()
            if physics_channels == channels
            else nn.Conv2d(physics_channels, channels, kernel_size=1, bias=False)
        )
        self.dem_proj = (
            nn.Identity()
            if physics_channels == channels
            else nn.Conv2d(physics_channels, channels, kernel_size=1, bias=False)
        )

        self.rgb_intra = IntraStreamBlock(channels, num_heads=num_heads, attn_spatial_max=attn_spatial_max)
        self.dem_intra = IntraStreamBlock(channels, num_heads=num_heads, attn_spatial_max=attn_spatial_max)
        self.fm_intra = IntraStreamBlock(channels, num_heads=num_heads, attn_spatial_max=attn_spatial_max)

        gn = group_norm_groups(channels)
        self.rgb_cal = nn.GroupNorm(gn, channels)
        self.dem_cal = nn.GroupNorm(gn, channels)
        self.fm_cal = nn.GroupNorm(gn, channels)

        self.stream_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, 3, kernel_size=1, bias=True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )

    def forward(
        self,
        p_rgb: list[torch.Tensor],
        p_dem: list[torch.Tensor],
        t_fm: list[torch.Tensor],
        level: int,
    ) -> torch.Tensor:
        level = max(0, min(level, len(p_rgb) - 1))
        x_rgb = match_spatial(self.rgb_proj(p_rgb[level]), t_fm[level])
        x_dem = match_spatial(self.dem_proj(p_dem[level]), t_fm[level])

        rgb = self.rgb_cal(self.rgb_intra(x_rgb))
        dem = self.dem_cal(self.dem_intra(x_dem))
        fm = self.fm_cal(self.fm_intra(t_fm[level]))

        gates = F.softmax(self.stream_gate(torch.cat([rgb, dem, fm], dim=1)), dim=1)
        mixed = (
            gates[:, 0:1] * rgb
            + gates[:, 1:2] * dem
            + gates[:, 2:3] * fm
        )
        return self.out(mixed)
