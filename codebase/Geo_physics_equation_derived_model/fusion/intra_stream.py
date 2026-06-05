"""Within-stream refinement before cross-stream fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvFFN(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialSelfAttention(nn.Module):
    """Lightweight multi-head self-attention on spatial tokens (single stream)."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x).view(b, 3, self.num_heads, self.head_dim, n)
        q = qkv[:, 0].permute(0, 1, 3, 2)
        k = qkv[:, 1].permute(0, 1, 3, 2)
        v = qkv[:, 2].permute(0, 1, 3, 2)
        scale = self.head_dim**-0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v).permute(0, 1, 3, 2).reshape(b, c, h, w)
        return self.out(out)


class IntraStreamBlock(nn.Module):
    """
    Refine one encoder stream in isolation (conv + optional spatial self-attention).

    Self-attention is only used when H*W <= attn_spatial_max to avoid OOM on L0/L1.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        attn_spatial_max: int = 4096,
    ):
        super().__init__()
        self.attn_spatial_max = attn_spatial_max
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.norm2 = nn.GroupNorm(8, channels)
        self.self_attn = SpatialSelfAttention(channels, num_heads=num_heads)
        self.norm3 = nn.GroupNorm(8, channels)
        self.ffn = _ConvFFN(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv_refine(self.norm1(x))
        if x.shape[-2] * x.shape[-1] <= self.attn_spatial_max:
            x = x + self.self_attn(self.norm2(x))
        x = x + self.ffn(self.norm3(x))
        return x
