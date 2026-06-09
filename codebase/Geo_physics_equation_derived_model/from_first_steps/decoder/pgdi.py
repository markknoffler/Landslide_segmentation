import torch
import torch.nn as nn

from fusion.pyramid_utils import match_spatial
from physics import LatentMechanisticCell


class PhysicsGatedDecoderInjection(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.latent = LatentMechanisticCell(channels)

    def forward(self, decoder_state, skip):
        skip = match_spatial(skip, decoder_state)
        beta = self.gate(torch.cat([decoder_state, skip], dim=1))
        injected = self.latent(skip)
        return decoder_state + beta * injected
