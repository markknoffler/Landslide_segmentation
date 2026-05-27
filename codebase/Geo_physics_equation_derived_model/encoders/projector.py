import torch.nn as nn


class StreamProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)
