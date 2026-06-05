"""
Balanced tri-stream fusion: intra-stream refinement, symmetric mixing, then cross-attention.

Replaces FM-heavy MAO (physics Q, FM-only K/V) with symmetric stream handling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.pyramid_utils import group_norm_groups, match_spatial
from .intra_stream import IntraStreamBlock


class SymmetricCrossAttention(nn.Module):
    """Cross-attention after symmetric stream mixing."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        rgb: torch.Tensor,
        dem: torch.Tensor,
        fm: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = query.shape
        n = h * w
        kv_source = (rgb + dem + fm) / 3.0

        q = self.q_proj(query).view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)
        k = self.k_proj(kv_source).view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)
        v = self.v_proj(kv_source).view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)

        scale = self.head_dim**-0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        context = torch.matmul(attn, v).permute(0, 1, 3, 2).reshape(b, c, h, w)
        return self.out(context)


class BalancedTriStreamFusion(nn.Module):
    """
    1) Intra-stream self-attention on RGB-physics, DEM-physics, and FM (separately).
    2) Per-stream calibration + softmax stream gates (symmetric, per-pixel).
    3) Cross-attention on the gated mix with symmetric K/V from all streams.
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
        self.cross_attn = SymmetricCrossAttention(channels, num_heads=num_heads)
        self.out_norm = nn.GroupNorm(gn, channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )

    def forward(
        self,
        t_fm: torch.Tensor,
        x_rgb: torch.Tensor,
        x_dem: torch.Tensor,
    ) -> torch.Tensor:
        x_rgb = match_spatial(self.rgb_proj(x_rgb), t_fm)
        x_dem = match_spatial(self.dem_proj(x_dem), t_fm)

        rgb = self.rgb_cal(self.rgb_intra(x_rgb))
        dem = self.dem_cal(self.dem_intra(x_dem))
        fm = self.fm_cal(self.fm_intra(t_fm))

        gates = F.softmax(self.stream_gate(torch.cat([rgb, dem, fm], dim=1)), dim=1)
        mixed = (
            gates[:, 0:1] * rgb
            + gates[:, 1:2] * dem
            + gates[:, 2:3] * fm
        )

        fused = mixed + self.cross_attn(mixed, rgb, dem, fm)
        return self.out_conv(self.out_norm(fused))
