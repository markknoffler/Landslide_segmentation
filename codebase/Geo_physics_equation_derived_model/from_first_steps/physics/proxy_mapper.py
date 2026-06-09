import math

import torch
import torch.nn as nn


class PhysicsProxyMapper(nn.Module):
    """Map normalized slope/dem/ndvi proxies to alpha, h, m."""

    def __init__(self, alpha_max: float = math.pi / 2):
        super().__init__()
        self.alpha_max = alpha_max
        self.w_alpha = nn.Parameter(torch.ones(1))
        self.b_alpha = nn.Parameter(torch.zeros(1))
        self.w_h = nn.Parameter(torch.ones(1))
        self.b_h = nn.Parameter(torch.zeros(1))
        self.w_m = nn.Parameter(torch.ones(1))
        self.b_m = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        slope_norm: torch.Tensor,
        dem_norm: torch.Tensor,
        ndvi_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if slope_norm.dim() == 3:
            slope_norm = slope_norm.unsqueeze(1)
        if dem_norm.dim() == 3:
            dem_norm = dem_norm.unsqueeze(1)
        if ndvi_norm.dim() == 3:
            ndvi_norm = ndvi_norm.unsqueeze(1)

        alpha = self.alpha_max * torch.sigmoid(self.w_alpha * slope_norm + self.b_alpha)
        h = torch.nn.functional.softplus(self.w_h * dem_norm + self.b_h) + 1e-4
        m = torch.sigmoid(self.w_m * ndvi_norm + self.b_m)
        return alpha, h, m
