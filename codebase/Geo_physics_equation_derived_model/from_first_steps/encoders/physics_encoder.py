import torch
import torch.nn as nn

from encoders.projector import StreamProjector
from physics import LatentMechanisticCell, PixelMechanisticCell


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PhysicsEncoder(nn.Module):
    """5-level physics pyramid with pixel cell at L0 and latent cells deeper."""

    def __init__(
        self,
        in_channels: int,
        unified_channels: int = 64,
        mechanistic_gating: bool = True,
    ):
        super().__init__()
        c0 = max(8, unified_channels // 4)
        c1 = max(8, unified_channels // 2)
        c2 = unified_channels
        c3 = unified_channels
        c4 = unified_channels
        self.pixel = PixelMechanisticCell(in_channels, c0, mechanistic_gating=mechanistic_gating)
        self.stem = nn.Sequential(
            nn.Conv2d(c0, c0, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c0),
            nn.ReLU(inplace=True),
        )
        self.down1 = _Down(c0, c1)
        self.latent1 = LatentMechanisticCell(c1)
        self.down2 = _Down(c1, c2)
        self.latent2 = LatentMechanisticCell(c2)
        self.down3 = _Down(c2, c3)
        self.latent3 = LatentMechanisticCell(c3)
        self.down4 = _Down(c3, c4)
        self.latent4 = LatentMechanisticCell(c4)

        self.proj0 = StreamProjector(c0, unified_channels)
        self.proj1 = StreamProjector(c1, unified_channels)
        self.proj2 = StreamProjector(c2, unified_channels)
        self.proj3 = StreamProjector(c3, unified_channels)
        self.proj4 = StreamProjector(c4, unified_channels)

    def forward(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        h: torch.Tensor,
        m: torch.Tensor,
    ) -> list:
        f0 = self.stem(self.pixel(x, alpha, h, m))
        f1 = self.latent1(self.down1(f0))
        f2 = self.latent2(self.down2(f1))
        f3 = self.latent3(self.down3(f2))
        f4 = self.latent4(self.down4(f3))
        return [
            self.proj0(f0),
            self.proj1(f1),
            self.proj2(f2),
            self.proj3(f3),
            self.proj4(f4),
        ]
