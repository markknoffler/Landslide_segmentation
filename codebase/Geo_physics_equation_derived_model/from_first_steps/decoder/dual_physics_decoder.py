"""Dual-stream physics decoder with GateFuse on outputs (paper strategy + physics cells)."""

from __future__ import annotations

import torch
import torch.nn as nn

from decoder.physics_decoder import PhysicsDecoder
from fusion.pyramid_utils import match_spatial


def _ln2d(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(1, channels)


class GateFuse(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.g = nn.Sequential(nn.Conv2d(ch * 2, 1, 1), nn.Sigmoid())

    def forward(self, a, b):
        alpha = self.g(torch.cat([a, b], dim=1))
        out = alpha * a + (1 - alpha) * b
        reg = (alpha * (1 - alpha)).mean()
        return out, reg


class DualPhysicsGatedDecoder(nn.Module):
    """
    Two PhysicsDecoder paths (RGB-proxy vs DEM-proxy physics variables) sharing
    tri-stream fused skips, with stream-specific CNN bottleneck residuals and
    GateFuse on main/aux outputs (same role as paper dual AdaptiveDecoder).
    """

    def __init__(
        self,
        channels: int = 64,
        n_classes: int = 1,
        bottleneck_ch: int = 448,
        mechanistic_gating: bool = True,
    ):
        super().__init__()
        self.decoder_a = PhysicsDecoder(channels, n_classes, mechanistic_gating)
        self.decoder_b = PhysicsDecoder(channels, n_classes, mechanistic_gating)
        self.bn_proj_a = nn.Sequential(
            nn.Conv2d(bottleneck_ch, channels, kernel_size=1, bias=False),
            _ln2d(channels),
            nn.ReLU(inplace=True),
        )
        self.bn_proj_b = nn.Sequential(
            nn.Conv2d(bottleneck_ch, channels, kernel_size=1, bias=False),
            _ln2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse_main = GateFuse(n_classes)
        self.fuse_aux2 = GateFuse(n_classes)
        self.fuse_aux3 = GateFuse(n_classes)

    def forward(self, f4, f3, skips, a5, b5, alpha_a, h_a, m_a, alpha_b, h_b, m_b):
        f4_a = f4 + match_spatial(self.bn_proj_a(a5), f4)
        f4_b = f4 + match_spatial(self.bn_proj_b(b5), f4)

        main_a, aux2_a, aux3_a = self.decoder_a(f4_a, f3, skips, alpha_a, h_a, m_a)
        main_b, aux2_b, aux3_b = self.decoder_b(f4_b, f3, skips, alpha_b, h_b, m_b)

        main, reg_m = self.fuse_main(main_a, main_b)
        aux2, reg_a2 = self.fuse_aux2(aux2_a, aux2_b)
        aux3, reg_a3 = self.fuse_aux3(aux3_a, aux3_b)
        return main, aux2, aux3, (reg_m, reg_a2, reg_a3)
