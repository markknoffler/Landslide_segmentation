from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bridge import TriTemporalTriStreamBridge
from .decoder import PhysicsDecoder
from .encoders import PhysicsEncoder, PrithviFoundationEncoder
from .fusion import MAOGeoEGCA
from .physics import PhysicsProxyMapper


class GeoPhysicsLandslideNet(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        n_classes: int = 1,
        lora_rank: int = 8,
        prithvi_snapshot: str | None = None,
        prithvi_input_normalization: str = "observed_rasters",
    ):
        super().__init__()
        self.channels = channels
        self.proxy_rgb = PhysicsProxyMapper()
        self.proxy_dem = PhysicsProxyMapper()
        self.enc_rgb = PhysicsEncoder(in_channels=3, unified_channels=channels)
        self.enc_dem = PhysicsEncoder(in_channels=1, unified_channels=channels)
        self.enc_fm = PrithviFoundationEncoder(
            unified_channels=channels,
            lora_rank=lora_rank,
            snapshot_dir=prithvi_snapshot,
            input_normalization=prithvi_input_normalization,
        )
        self.mao3 = MAOGeoEGCA(channels)
        self.mao4 = MAOGeoEGCA(channels)
        self.tteb = nn.ModuleList([TriTemporalTriStreamBridge(channels) for _ in range(4)])
        self.fuse3 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.decoder = PhysicsDecoder(channels=channels, n_classes=n_classes)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = batch["stream_a"]
        dem = batch["dem"]
        prithvi = batch["prithvi_input"]
        slope = batch["slope_norm"]
        dem_norm = batch["dem_norm"]
        ndvi = batch["ndvi_norm"]

        alpha_r, h_r, m_r = self.proxy_rgb(slope, dem_norm, ndvi)
        alpha_d, h_d, m_d = self.proxy_dem(slope, dem_norm, ndvi)

        p_rgb = self.enc_rgb(rgb, alpha_r, h_r, m_r)
        p_dem = self.enc_dem(dem, alpha_d, h_d, m_d)
        t_fm = self.enc_fm(prithvi)

        f3 = self.mao3(t_fm[3], p_rgb[3], p_dem[3])
        f4 = self.mao4(t_fm[4], p_rgb[4], p_dem[4])

        skips = [self.tteb[i](p_rgb, p_dem, t_fm, level=i) for i in range(4)]

        alpha = 0.5 * (alpha_r + alpha_d)
        h = 0.5 * (h_r + h_d)
        m = 0.5 * (m_r + m_d)

        main, aux2, aux3 = self.decoder(f4, self.fuse3(f3), skips, alpha, h, m)
        target_size = batch["mask"].shape[-2:]
        if aux2.shape[-2:] != target_size:
            aux2 = F.interpolate(aux2, size=target_size, mode="bilinear", align_corners=False)
        if aux3.shape[-2:] != target_size:
            aux3 = F.interpolate(aux3, size=target_size, mode="bilinear", align_corners=False)
        return main, aux2, aux3
