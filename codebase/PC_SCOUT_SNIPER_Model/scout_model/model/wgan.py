from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PCBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alignment = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        pred = self.alignment(dec)
        error = enc - pred
        return torch.cat([error, dec], dim=1)


class Scout(nn.Module):
    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.enc1 = ConvBlock(1, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)

        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.pc4 = PCBlock(base_ch * 8)
        self.conv4 = ConvBlock(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.pc3 = PCBlock(base_ch * 4)
        self.conv3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.pc2 = PCBlock(base_ch * 2)
        self.conv2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.pc1 = PCBlock(base_ch)
        self.conv1 = ConvBlock(base_ch * 2, base_ch)

        self.head = nn.Sequential(
            nn.Conv2d(base_ch, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        if d4.shape[-2:] != e4.shape[-2:]:
            d4 = F.interpolate(d4, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.pc4(e4, d4)
        d4 = self.conv4(d4)

        d3 = self.up3(d4)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.pc3(e3, d3)
        d3 = self.conv3(d3)

        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.pc2(e2, d2)
        d2 = self.conv2(d2)

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.pc1(e1, d1)
        d1 = self.conv1(d1)

        return self.head(d1)


class Critic(nn.Module):
    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 8, 1, 4, stride=1, padding=0),
        )

    def forward(self, dem: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([dem, rgb], dim=1)
        x = self.net(x)
        return x.view(x.size(0), -1).mean(dim=1, keepdim=True)
