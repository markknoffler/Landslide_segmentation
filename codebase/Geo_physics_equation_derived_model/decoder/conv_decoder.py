"""Standard UNet-style convolution decoder (ablation alternative to PhysicsDecoder)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Up(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class _SkipFuse(nn.Module):
    """Concatenate skip connection and apply conv block (standard UNet injection)."""

    def __init__(self, channels: int):
        super().__init__()
        self.fuse = _DoubleConv(channels * 2, channels)

    def forward(self, decoder_state: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if skip.shape[-2:] != decoder_state.shape[-2:]:
            skip = F.interpolate(skip, size=decoder_state.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([decoder_state, skip], dim=1))


class ConvDecoder(nn.Module):
    """
    Convolutional decoder with the same call signature as PhysicsDecoder.

    Uses additive fusion for fused3/fused4 neck features and skip concatenation at
    each scale. Ignores physics proxies (alpha, h, m) so the flag switch is drop-in.
    """

    def __init__(self, channels: int = 64, n_classes: int = 1):
        super().__init__()
        self.stem4 = _DoubleConv(channels, channels)
        self.up3 = _Up(channels)
        self.stem3 = _DoubleConv(channels, channels)
        self.skip3 = _SkipFuse(channels)
        self.up2 = _Up(channels)
        self.skip2 = _SkipFuse(channels)
        self.up1 = _Up(channels)
        self.skip1 = _SkipFuse(channels)
        self.up0 = _Up(channels)
        self.skip0 = _SkipFuse(channels)
        self.aux3_head = nn.Conv2d(channels, n_classes, kernel_size=1)
        self.aux2_head = nn.Conv2d(channels, n_classes, kernel_size=1)
        self.head = nn.Conv2d(channels, n_classes, kernel_size=1)
        for head in (self.head, self.aux2_head, self.aux3_head):
            if head.bias is not None:
                nn.init.constant_(head.bias, -2.0)

    def forward(
        self,
        fused4: torch.Tensor,
        fused3: torch.Tensor,
        skips: list[torch.Tensor],
        alpha: torch.Tensor,
        h: torch.Tensor,
        m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del alpha, h, m

        d4 = self.stem4(fused4)
        d3 = self.stem3(self.up3(d4) + fused3)
        d3 = self.skip3(d3, skips[3])
        aux3 = self.aux3_head(d3)

        d2 = self.up2(d3)
        d2 = self.skip2(d2, skips[2])
        aux2 = self.aux2_head(d2)

        d1 = self.up1(d2)
        d1 = self.skip1(d1, skips[1])

        d0 = self.up0(d1)
        d0 = self.skip0(d0, skips[0])

        main = self.head(d0)
        return main, aux2, aux3
