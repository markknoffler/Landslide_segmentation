from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bridge import TriTemporalTriStreamBridge
from .decoder import AdaptiveConvDecoder, ConvDecoder, PhysicsDecoder
from .encoders import EfficientNetFoundationEncoder, PhysicsEncoder, PrithviFoundationEncoder
from .encoders.pyramid_utils import match_spatial
from .fusion import (
    BalancedTriStreamFusion,
    BalancedTriStreamSkip,
    ConcatTriStreamLevel,
    ConcatTriStreamSkip,
    MAOGeoEGCA,
)
from .physics import PhysicsProxyMapper

FmBackbone = Literal["efficientnet", "prithvi"]
DecoderType = Literal["physics", "conv"]
FusionType = Literal["balanced", "mao", "concat"]
FmPyramid = Literal["native", "legacy"]


class DualStreamPhysicsNet(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        n_classes: int = 1,
        lora_rank: int = 8,
        prithvi_snapshot: str | None = None,
        prithvi_input_normalization: str = "observed_rasters",
        fm_backbone: FmBackbone | str = "efficientnet",
        efficientnet_name: str = "tf_efficientnet_b4",
        efficientnet_pretrained: bool = True,
        freeze_efficientnet: bool = True,
        fm_pyramid: FmPyramid | str = "native",
        decoder_type: DecoderType | str = "physics",
        fusion_type: FusionType | str = "balanced",
        tteb_attn_chunk: int = 1024,
        tteb_attn_low_res_max: int = 4096,
        mechanistic_gating: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.fm_backbone = str(fm_backbone).lower()
        self.decoder_type = str(decoder_type).lower()
        self.fusion_type = str(fusion_type).lower()
        self.mechanistic_gating = mechanistic_gating
        self.fm_pyramid = str(fm_pyramid).lower()
        if self.fm_backbone != "efficientnet":
            self.fm_pyramid = "legacy"

        # Physics proxy mappers for both streams
        self.proxy_rgb = PhysicsProxyMapper()
        self.proxy_dem = PhysicsProxyMapper()

        # Physics encoders for both streams
        self.enc_rgb = PhysicsEncoder(
            in_channels=3, unified_channels=channels, mechanistic_gating=mechanistic_gating
        )
        self.enc_dem = PhysicsEncoder(
            in_channels=1, unified_channels=channels, mechanistic_gating=mechanistic_gating
        )

        # Feature map backbone - using the existing foundation encoders
        if self.fm_backbone == "prithvi":
            self.enc_fm = PrithviFoundationEncoder(
                unified_channels=channels,
                lora_rank=lora_rank,
                snapshot_dir=prithvi_snapshot,
                input_normalization=prithvi_input_normalization,
            )
            self.native_fm = False
            self.fusion_level_channels = [channels] * 5
        elif self.fm_backbone == "efficientnet":
            self.enc_fm = EfficientNetFoundationEncoder(
                unified_channels=channels,
                backbone=efficientnet_name,
                pretrained=efficientnet_pretrained,
                freeze_backbone=freeze_efficientnet,
                pyramid_mode=self.fm_pyramid,
            )
            self.native_fm = self.enc_fm.uses_native_pyramid
            self.fusion_level_channels = (
                list(self.enc_fm.level_channels)
                if self.native_fm
                else [channels] * 5
            )
        else:
            raise ValueError(
                f"Unknown fm_backbone={fm_backbone!r}. Choose 'efficientnet' or 'prithvi'."
            )

        self._build_fusion(
            tteb_attn_chunk=tteb_attn_chunk,
            tteb_attn_low_res_max=tteb_attn_low_res_max,
        )
        self._build_decoder(n_classes=n_classes)

    def _build_fusion(self, *, tteb_attn_chunk: int, tteb_attn_low_res_max: int) -> None:
        phys = self.channels
        fm_levels = self.fusion_level_channels

        if self.fusion_type == "concat":
            self.fuse3 = None
            self.fuse4 = ConcatTriStreamLevel(phys, fm_levels[4])
            self.skips = nn.ModuleList(
                [ConcatTriStreamSkip(phys, fm_levels[i]) for i in range(4)]
            )
        elif self.fusion_type == "balanced":
            self.fuse3 = BalancedTriStreamFusion(phys, fm_levels[3])
            self.fuse4 = BalancedTriStreamFusion(phys, fm_levels[4])
            self.skips = nn.ModuleList(
                [BalancedTriStreamSkip(phys, fm_levels[i]) for i in range(4)]
            )
        elif self.fusion_type == "mao":
            self.fuse3 = MAOGeoEGCA(fm_levels[3])
            self.fuse4 = MAOGeoEGCA(fm_levels[4])
            self.skips = nn.ModuleList(
                [
                    TriTemporalTriStreamBridge(
                        fm_levels[i],
                        attn_chunk_size=tteb_attn_chunk,
                        attn_low_res_max=tteb_attn_low_res_max,
                    )
                    for i in range(4)
                ]
            )
            self.post_fuse3 = nn.Conv2d(fm_levels[3], fm_levels[3], kernel_size=1, bias=False)
            if self.native_fm:
                self.mao_rgb_proj = nn.ModuleList(
                    [
                        nn.Conv2d(phys, fm_levels[i], kernel_size=1, bias=False)
                        if phys != fm_levels[i]
                        else nn.Identity()
                        for i in range(5)
                    ]
                )
                self.mao_dem_proj = nn.ModuleList(
                    [
                        nn.Conv2d(phys, fm_levels[i], kernel_size=1, bias=False)
                        if phys != fm_levels[i]
                        else nn.Identity()
                        for i in range(5)
                    ]
                )
            else:
                self.mao_rgb_proj = None
                self.mao_dem_proj = None
        else:
            raise ValueError(
                f"Unknown fusion_type={self.fusion_type!r}. Choose 'balanced', 'concat', or 'mao'."
            )

    def _build_decoder(self, *, n_classes: int) -> None:
        if self.native_fm:
            if self.decoder_type == "physics":
                raise ValueError(
                    "Native EfficientNet pyramid requires --decoder conv "
                    "(448-ch bottleneck is not compatible with PhysicsDecoder)."
                )
            self.decoder = AdaptiveConvDecoder(
                level_channels=self.fusion_level_channels,
                n_classes=n_classes,
            )
            return

        if self.decoder_type == "physics":
            self.decoder = PhysicsDecoder(
                channels=self.channels,
                n_classes=n_classes,
                mechanistic_gating=self.mechanistic_gating,
            )
        elif self.decoder_type == "conv":
            self.decoder = ConvDecoder(channels=self.channels, n_classes=n_classes)
        else:
            raise ValueError(
                f"Unknown decoder_type={self.decoder_type!r}. Choose 'physics' or 'conv'."
            )

    def _fm_input(self, batch: dict) -> torch.Tensor:
        if "fm_input" in batch:
            return batch["fm_input"]
        if "prithvi_input" in batch:
            return batch["prithvi_input"]
        raise KeyError("Batch must contain 'fm_input' (or legacy 'prithvi_input').")

    def _adapted_physics_pyramids(
        self,
        p_rgb: list[torch.Tensor],
        p_dem: list[torch.Tensor],
        t_fm: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if getattr(self, "mao_rgb_proj", None) is None:
            return p_rgb, p_dem
        rgb_adapted = [
            match_spatial(self.mao_rgb_proj[i](p_rgb[i]), t_fm[i]) for i in range(len(t_fm))
        ]
        dem_adapted = [
            match_spatial(self.mao_dem_proj[i](p_dem[i]), t_fm[i]) for i in range(len(t_fm))
        ]
        return rgb_adapted, dem_adapted

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = batch["stream_a"]
        dem = batch["dem"]
        fm_in = self._fm_input(batch)
        slope = batch["slope_norm"]
        dem_norm = batch["dem_norm"]
        ndvi = batch["ndvi_norm"]

        alpha_r, h_r, m_r = self.proxy_rgb(slope, dem_norm, ndvi)
        alpha_d, h_d, m_d = self.proxy_dem(slope, dem_norm, ndvi)

        p_rgb = self.enc_rgb(rgb, alpha_r, h_r, m_r)
        p_dem = self.enc_dem(dem, alpha_d, h_d, m_d)
        t_fm = self.enc_fm(fm_in)

        if self.fusion_type == "concat":
            f3 = None
            f4 = self.fuse4(t_fm[4], p_rgb[4], p_dem[4])
            skip_feats = [self.skips[i](p_rgb, p_dem, t_fm, level=i) for i in range(4)]
        elif self.fusion_type == "balanced":
            f3 = self.fuse3(t_fm[3], p_rgb[3], p_dem[3])
            f4 = self.fuse4(t_fm[4], p_rgb[4], p_dem[4])
            skip_feats = [self.skips[i](p_rgb, p_dem, t_fm, level=i) for i in range(4)]
        else:
            p_rgb_mao, p_dem_mao = self._adapted_physics_pyramids(p_rgb, p_dem, t_fm)
            f3 = self.post_fuse3(self.fuse3(t_fm[3], p_rgb_mao[3], p_dem_mao[3]))
            f4 = self.fuse4(t_fm[4], p_rgb_mao[4], p_dem_mao[4])
            skip_feats = [
                self.skips[i](p_rgb_mao, p_dem_mao, t_fm, level=i) for i in range(4)
            ]

        alpha = 0.5 * (alpha_r + alpha_d)
        h = 0.5 * (h_r + h_d)
        m = 0.5 * (m_r + m_d)

        main, aux2, aux3 = self.decoder(f4, f3, skip_feats, alpha, h, m)
        target_size = batch["mask"].shape[-2:]
        if main.shape[-2:] != target_size:
            main = F.interpolate(main, size=target_size, mode="bilinear", align_corners=False)
        if aux2.shape[-2:] != target_size:
            aux2 = F.interpolate(aux2, size=target_size, mode="bilinear", align_corners=False)
        if aux3.shape[-2:] != target_size:
            aux3 = F.interpolate(aux3, size=target_size, mode="bilinear", align_corners=False)
        return main, aux2, aux3


# Keep the original GeoPhysicsLandslideNet as an alias for backward compatibility
GeoPhysicsLandslideNet = DualStreamPhysicsNet