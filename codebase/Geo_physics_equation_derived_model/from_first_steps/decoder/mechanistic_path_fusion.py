"""
Mechanistic Path Equilibrium Fusion (MPEF).

Merges dual physics-decoder path logits using per-pixel geotechnical instability
weights derived from each path's (alpha, h, m) proxies — not DiGATe-style GateFuse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.pyramid_utils import match_spatial


class MechanisticPathEquilibriumFusion(nn.Module):
    """
    Route between RGB-proxy and DEM-proxy decode paths by relative failure energy.

    Higher instability on a path increases its contribution to the merged logits.
    A learned 1x1 correction is added on top of the instability-weighted blend.
    """

    def __init__(self, n_classes: int = 1):
        super().__init__()
        self.w_c = nn.Parameter(torch.zeros(1))
        self.w_phi = nn.Parameter(torch.zeros(1))
        self.w_gamma = nn.Parameter(torch.zeros(1))
        self.w_m = nn.Parameter(torch.zeros(1))
        self.correction = nn.Conv2d(n_classes * 2, n_classes, kernel_size=1, bias=False)

    def _failure_energy(
        self,
        alpha: torch.Tensor,
        h: torch.Tensor,
        m: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        alpha = match_spatial(alpha, ref)
        h = match_spatial(h, ref)
        m = match_spatial(m, ref)

        c = torch.exp(self.w_c)
        phi = torch.exp(self.w_phi)
        gamma = torch.exp(self.w_gamma)
        w_moist = torch.exp(self.w_m)

        cos2 = torch.cos(alpha) ** 2
        sin_cos = torch.sin(alpha) * torch.cos(alpha)
        resisting = c + phi * h * cos2
        driving = gamma * h * sin_cos + w_moist * m + 1e-6
        fs = resisting / driving
        return torch.relu(1.0 - fs)

    def forward(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        alpha_a: torch.Tensor,
        h_a: torch.Tensor,
        m_a: torch.Tensor,
        alpha_b: torch.Tensor,
        h_b: torch.Tensor,
        m_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fe_a = self._failure_energy(alpha_a, h_a, m_a, logits_a)
        fe_b = self._failure_energy(alpha_b, h_b, m_b, logits_b)

        weights = torch.softmax(torch.cat([fe_a, fe_b], dim=1), dim=1)
        wa = weights[:, 0:1]
        wb = weights[:, 1:2]

        blended = wa * logits_a + wb * logits_b
        out = blended + self.correction(torch.cat([logits_a, logits_b], dim=1))

        # Encourage decisive routing (low entropy), distinct from DiGATe gate regularizer.
        reg = -(wa * torch.log(wa + 1e-6) + wb * torch.log(wb + 1e-6)).mean()
        return out, reg
