"""
Complementary Modality Bridge (CMB).

Fuses one CNN stream (EfficientNet) with one physics stream (PhysicsEncoder) per modality
before tri-stream MAO/TTEB fusion. When CNN and physics agree (high resonance), physics
features are weighted more heavily; otherwise CNN texture is preserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fusion.pyramid_utils import match_spatial


class ComplementaryModalityBridge(nn.Module):
    def __init__(self, cnn_channels: int, unified_channels: int = 64):
        super().__init__()
        self.unified_channels = unified_channels
        if cnn_channels == unified_channels:
            self.cnn_proj = nn.Identity()
        else:
            self.cnn_proj = nn.Sequential(
                nn.Conv2d(cnn_channels, unified_channels, kernel_size=1, bias=False),
                nn.GroupNorm(8, unified_channels),
                nn.ReLU(inplace=True),
            )
        gn = 8 if unified_channels >= 8 else 1
        self.cnn_cal = nn.GroupNorm(gn, unified_channels)
        self.physics_cal = nn.GroupNorm(gn, unified_channels)
        self.coherence = nn.Sequential(
            nn.Conv2d(unified_channels, max(1, unified_channels // 4), kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(1, unified_channels // 4), 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.blend = nn.Conv2d(unified_channels * 2, unified_channels, kernel_size=1, bias=False)

    def forward(self, cnn_feat: torch.Tensor, physics_feat: torch.Tensor) -> torch.Tensor:
        physics_feat = match_spatial(physics_feat, cnn_feat)
        e = self.cnn_cal(self.cnn_proj(cnn_feat))
        p = self.physics_cal(physics_feat)
        g = self.coherence(e * p)
        mixed = self.blend(torch.cat([e, p], dim=1))
        return g * p + (1.0 - g) * e + mixed
