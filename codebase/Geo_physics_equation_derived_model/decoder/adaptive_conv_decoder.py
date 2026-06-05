"""UNet-style decoder with per-level channel widths (native EfficientNet pyramid)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.pyramid_utils import group_norm_groups


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(group_norm_groups(channels), channels)


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class _SkipFuse(nn.Module):
    def __init__(self, dec_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.fuse = _DoubleConv(dec_channels + skip_channels, out_channels)

    def forward(self, decoder_state: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if skip.shape[-2:] != decoder_state.shape[-2:]:
            skip = F.interpolate(
                skip, size=decoder_state.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.fuse(torch.cat([decoder_state, skip], dim=1))


class AdaptiveConvDecoder(nn.Module):
    """
    Convolutional decoder for variable-width encoder pyramids.

    level_channels: [L0, L1, L2, L3, L4] channel widths (e.g. EfficientNet-B4 native).
    """

    def __init__(self, level_channels: list[int], n_classes: int = 1):
        super().__init__()
        if len(level_channels) != 5:
            raise ValueError(f"Expected 5 pyramid levels, got {len(level_channels)}")
        c0, c1, c2, c3, c4 = level_channels

        self.stem4 = _DoubleConv(c4, c4)
        self.up3 = _Up(c4, c3)
        self.stem3 = _DoubleConv(c3, c3)
        self.skip3 = _SkipFuse(c3, c3, c3)
        self.up2 = _Up(c3, c2)
        self.skip2 = _SkipFuse(c2, c2, c2)
        self.up1 = _Up(c2, c1)
        self.skip1 = _SkipFuse(c1, c1, c1)
        self.up0 = _Up(c1, c0)
        self.skip0 = _SkipFuse(c0, c0, c0)
        self.aux3_head = nn.Conv2d(c3, n_classes, kernel_size=1)
        self.aux2_head = nn.Conv2d(c2, n_classes, kernel_size=1)
        self.head = nn.Conv2d(c0, n_classes, kernel_size=1)
        for head in (self.head, self.aux2_head, self.aux3_head):
            if head.bias is not None:
                nn.init.constant_(head.bias, -2.0)

    def forward(
        self,
        fused4: torch.Tensor,
        fused3: torch.Tensor | None,
        skips: list[torch.Tensor],
        alpha: torch.Tensor,
        h: torch.Tensor,
        m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del alpha, h, m

        d4 = self.stem4(fused4)
        d3 = self.stem3(self.up3(d4))
        if fused3 is not None:
            if fused3.shape[-2:] != d3.shape[-2:]:
                fused3 = F.interpolate(
                    fused3, size=d3.shape[-2:], mode="bilinear", align_corners=False
                )
            d3 = d3 + fused3
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
