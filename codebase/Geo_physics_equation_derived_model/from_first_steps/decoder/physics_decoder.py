import torch.nn as nn
import torch.nn.functional as F

from decoder.pgdi import PhysicsGatedDecoderInjection
from physics import LatentMechanisticCell, PixelMechanisticCell


class _Up(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class PhysicsDecoder(nn.Module):
    def __init__(self, channels: int = 64, n_classes: int = 1, mechanistic_gating: bool = True):
        super().__init__()
        self.mechanistic_gating = mechanistic_gating
        self.latent4 = LatentMechanisticCell(channels)
        self.latent3 = LatentMechanisticCell(channels)
        self.latent2 = LatentMechanisticCell(channels)
        self.up3 = _Up(channels)
        self.up2 = _Up(channels)
        self.up1 = _Up(channels)
        self.up0 = _Up(channels)
        self.pgdi3 = PhysicsGatedDecoderInjection(channels)
        self.pgdi2 = PhysicsGatedDecoderInjection(channels)
        self.pgdi1 = PhysicsGatedDecoderInjection(channels)
        self.pgdi0 = PhysicsGatedDecoderInjection(channels)
        self.aux2_head = nn.Conv2d(channels, n_classes, kernel_size=1)
        self.aux3_head = nn.Conv2d(channels, n_classes, kernel_size=1)
        self.pixel_out = PixelMechanisticCell(channels, channels, mechanistic_gating=mechanistic_gating)
        self.head = nn.Conv2d(channels, n_classes, kernel_size=1)
        for head in (self.head, self.aux2_head, self.aux3_head):
            if head.bias is not None:
                nn.init.constant_(head.bias, -2.0)

    def forward(self, fused4, fused3, skips, alpha, h, m):
        d4 = self.latent4(fused4)
        d3 = self.latent3(self.up3(d4))
        if fused3 is not None:
            d3 = d3 + fused3
        d3 = self.pgdi3(d3, skips[3])
        aux3 = self.aux3_head(d3)

        d2 = self.latent2(self.up2(d3))
        d2 = self.pgdi2(d2, skips[2])
        aux2 = self.aux2_head(d2)

        d1 = self.up1(d2)
        d1 = self.pgdi1(d1, skips[1])

        d0 = self.up0(d1)
        d0 = self.pgdi0(d0, skips[0])

        if self.mechanistic_gating:
            d0 = self.pixel_out(d0, alpha, h, m)
        main = self.head(d0)
        return main, aux2, aux3
