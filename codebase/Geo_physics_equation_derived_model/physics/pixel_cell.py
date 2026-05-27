import math

import torch
import torch.nn as nn


class PixelMechanisticCell(nn.Module):
    """Pixel-level infinite-slope stability gate (Taylor-stabilized ratio)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.feature_map = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.w_c = nn.Parameter(torch.randn(1, out_channels, 1, 1) * 0.02)
        self.w_phi = nn.Parameter(torch.randn(1, out_channels, 1, 1) * 0.02)
        self.w_gamma = nn.Parameter(torch.randn(1, out_channels, 1, 1) * 0.02)
        self.w_m = nn.Parameter(torch.randn(1, out_channels, 1, 1) * 0.02)
        self.psi = nn.Parameter(torch.ones(1, out_channels, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        h: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        c = torch.exp(self.w_c)
        phi = torch.exp(self.w_phi)
        gamma = torch.exp(self.w_gamma)
        w_moist = torch.exp(self.w_m)

        cos2 = torch.cos(alpha) ** 2
        sin_cos = torch.sin(alpha) * torch.cos(alpha)

        resisting = c + phi * h * cos2
        driving = gamma * h * sin_cos + w_moist * m + 1e-6
        fs = resisting / driving
        failure_energy = self.psi - fs
        gate = torch.sigmoid(failure_energy)
        return self.feature_map(x) * gate
