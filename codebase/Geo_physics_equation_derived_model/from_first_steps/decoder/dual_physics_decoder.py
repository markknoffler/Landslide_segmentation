"""Dual physics decoder with mechanistic path equilibrium fusion (MPEF)."""

from __future__ import annotations

import torch.nn as nn

from decoder.mechanistic_path_fusion import MechanisticPathEquilibriumFusion
from decoder.physics_decoder import PhysicsDecoder
from fusion.pyramid_utils import match_spatial


def _ln2d(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(1, channels)


class DualPhysicsDecoder(nn.Module):
    """
    Two PhysicsDecoder paths:
      - Path A: RGB-proxy geotechnical variables + RGB CNN bottleneck residual
      - Path B: DEM-proxy geotechnical variables + DEM CNN bottleneck residual

    Shared MAO/TTEB context; outputs merged by MechanisticPathEquilibriumFusion (MPEF).
    """

    def __init__(
        self,
        channels: int = 64,
        n_classes: int = 1,
        bottleneck_ch: int = 448,
        mechanistic_gating: bool = True,
        mpef_mode: str = "mpef",
        path_mode: str = "dual",
    ):
        super().__init__()
        if path_mode not in {"dual", "path_a"}:
            raise ValueError(f"Unknown path_mode: {path_mode}")
        self.path_mode = path_mode
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
        self.fuse_main = MechanisticPathEquilibriumFusion(n_classes, mode=mpef_mode)
        self.fuse_aux2 = MechanisticPathEquilibriumFusion(n_classes, mode=mpef_mode)
        self.fuse_aux3 = MechanisticPathEquilibriumFusion(n_classes, mode=mpef_mode)

    def forward(self, f4, f3, skips, a5, b5, alpha_a, h_a, m_a, alpha_b, h_b, m_b):
        f4_a = f4 + match_spatial(self.bn_proj_a(a5), f4)
        main_a, aux2_a, aux3_a = self.decoder_a(f4_a, f3, skips, alpha_a, h_a, m_a)

        if self.path_mode == "path_a":
            z = main_a.new_zeros(())
            return main_a, aux2_a, aux3_a, (z, z, z)

        f4_b = f4 + match_spatial(self.bn_proj_b(b5), f4)
        main_b, aux2_b, aux3_b = self.decoder_b(f4_b, f3, skips, alpha_b, h_b, m_b)

        main, reg_m = self.fuse_main(main_a, main_b, alpha_a, h_a, m_a, alpha_b, h_b, m_b)
        aux2, reg_a2 = self.fuse_aux2(aux2_a, aux2_b, alpha_a, h_a, m_a, alpha_b, h_b, m_b)
        aux3, reg_a3 = self.fuse_aux3(aux3_a, aux3_b, alpha_a, h_a, m_a, alpha_b, h_b, m_b)
        return main, aux2, aux3, (reg_m, reg_a2, reg_a3)


# Backward-compatible alias (deprecated name).
DualPhysicsGatedDecoder = DualPhysicsDecoder
