import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentMechanisticCell(nn.Module):
    """Latent continuum factor-of-safety cell (no trigonometry)."""

    def __init__(self, channels: int):
        super().__init__()
        self.to_resisting = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.to_driving = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.psi = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.out_layer = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, h_tensor: torch.Tensor) -> torch.Tensor:
        resisting = F.softplus(self.to_resisting(h_tensor))
        driving = F.softplus(self.to_driving(h_tensor))
        latent_fs = resisting / (driving + 1e-6)
        latent_stress = self.psi - latent_fs
        return self.out_layer(F.leaky_relu(latent_stress, 0.2))
