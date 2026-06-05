"""Symmetric skip fusion across RGB-physics, DEM-physics, and FM pyramid levels."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .intra_stream import IntraStreamBlock


class BalancedTriStreamSkip(nn.Module):
    """
    Per-level skip builder: intra-stream refine each encoder, then symmetric gating.

    No cross-encoder attention here (keeps skips efficient); cross fusion happens at L3/L4.
    """

    def __init__(self, channels: int, num_heads: int = 4, attn_spatial_max: int = 4096):
        super().__init__()
        self.rgb_intra = IntraStreamBlock(channels, num_heads=num_heads, attn_spatial_max=attn_spatial_max)
        self.dem_intra = IntraStreamBlock(channels, num_heads=num_heads, attn_spatial_max=attn_spatial_max)
        self.fm_intra = IntraStreamBlock(channels, num_heads=num_heads, attn_spatial_max=attn_spatial_max)

        self.rgb_cal = nn.GroupNorm(8, channels)
        self.dem_cal = nn.GroupNorm(8, channels)
        self.fm_cal = nn.GroupNorm(8, channels)

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
        rgb = self.rgb_cal(self.rgb_intra(p_rgb[level]))
        dem = self.dem_cal(self.dem_intra(p_dem[level]))
        fm = self.fm_cal(self.fm_intra(t_fm[level]))

        gates = F.softmax(self.stream_gate(torch.cat([rgb, dem, fm], dim=1)), dim=1)
        mixed = (
            gates[:, 0:1] * rgb
            + gates[:, 1:2] * dem
            + gates[:, 2:3] * fm
        )
        return self.out(mixed)
