from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import DualPhysicsDecoder
from encoders import PhysicsEncoder
from fusion import ComplementaryModalityBridge, MAOGeoEGCA, TriTemporalTriStreamBridge
from fusion.pyramid_utils import match_spatial
from physics import PhysicsProxyMapper
from prithvi_encoder import PrithviFoundationEncoder


def LN2d(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(1, channels)


@dataclass
class EncoderSpec:
    name: str = "tf_efficientnet_b4"
    n_channels: int = 3
    out_indices: Tuple[int, ...] = (0, 1, 2, 3, 4)
    pretrained: bool = True
    pretrained_path: Optional[str] = None
    use_input_adapter: bool = False
    freeze: bool = False


def _adapt_conv1_weight(state_dict: Dict[str, torch.Tensor], n_channels: int) -> Dict[str, torch.Tensor]:
    conv1_keys = [k for k in state_dict.keys() if k.endswith("conv1.weight")]
    if not conv1_keys:
        return state_dict
    key = conv1_keys[0]
    weight = state_dict[key]
    cin_src = weight.shape[1]
    if cin_src == n_channels:
        return state_dict
    avg = weight.mean(1, keepdim=True)
    state_dict[key] = avg.repeat(1, n_channels, 1, 1) * (cin_src / n_channels)
    return state_dict


class InputAdapter(nn.Module):
    def __init__(self, in_ch: int, mid_norm: bool = True):
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(in_ch, 3, kernel_size=1, bias=False)]
        if mid_norm:
            layers.append(nn.BatchNorm2d(3, affine=True))
        layers.append(nn.ReLU(inplace=True))
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TimmEncoder(nn.Module):
    def __init__(self, spec: EncoderSpec):
        super().__init__()
        self.spec = spec

        target_in = 3 if spec.use_input_adapter else spec.n_channels
        self.input_adapter = (
            InputAdapter(spec.n_channels) if (spec.use_input_adapter and spec.n_channels != 3) else nn.Identity()
        )

        net = timm.create_model(
            spec.name,
            pretrained=spec.pretrained and spec.pretrained_path is None,
            features_only=True,
            out_indices=spec.out_indices,
            in_chans=target_in,
        )

        if spec.pretrained_path is not None:
            state_dict = torch.load(spec.pretrained_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith(("fc.", "classifier.", "head."))}
            if not spec.use_input_adapter and target_in != 3:
                state_dict = _adapt_conv1_weight(state_dict, target_in)
            missing, unexpected = net.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                warnings.warn(
                    f"[TimmEncoder] Loaded with missing={len(missing)}, unexpected={len(unexpected)}",
                    stacklevel=2,
                )

        self.net = net
        self.feature_info = self.net.feature_info
        self.channels = list(self.feature_info.channels())
        try:
            self.strides = list(self.feature_info.reduction())
        except Exception:
            self.strides = [
                feature["reduction"] if "reduction" in feature else 2 ** (i + 1)
                for i, feature in enumerate(self.feature_info)
            ]

        if spec.freeze:
            for param in self.net.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.input_adapter(x)
        return tuple(self.net(x))


def build_encoder(
    name: str = "tf_efficientnet_b4",
    n_channels: int = 3,
    out_indices: Tuple[int, ...] = (0, 1, 2, 3, 4),
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    use_input_adapter: bool = False,
    freeze: bool = False,
) -> TimmEncoder:
    spec = EncoderSpec(
        name=name,
        n_channels=n_channels,
        out_indices=out_indices,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        use_input_adapter=use_input_adapter,
        freeze=freeze,
    )
    return TimmEncoder(spec)


def _sobel_slope_norm(dem: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample normalized slope in [0, 1] from a single-channel DEM tensor [B, 1, H, W]."""
    b, _, h, w = dem.shape
    out = []
    for i in range(b):
        d = dem[i, 0]
        gy, gx = torch.gradient(d, spacing=(1.0, 1.0))
        slope = torch.sqrt(gx * gx + gy * gy)
        mn = slope.min()
        mx = slope.max()
        if float(mx - mn) > eps:
            slope = (slope - mn) / (mx - mn + eps)
        out.append(torch.clamp(slope, 0.0, 1.0))
    return torch.stack(out, dim=0).unsqueeze(1)


def _vegetation_index_from_rgb(rgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Green-red vegetation proxy from RGB [B, 3, H, W], normalized per sample to [0, 1]."""
    r = rgb[:, 0]
    g = rgb[:, 1]
    vi = (g - r) / (g + r + eps)
    vi = torch.clamp(vi, 0.0, 1.0)
    b = vi.shape[0]
    out = []
    for i in range(b):
        channel = vi[i]
        mn = channel.min()
        mx = channel.max()
        if float(mx - mn) > eps:
            channel = (channel - mn) / (mx - mn + eps)
        out.append(torch.clamp(channel, 0.0, 1.0))
    return torch.stack(out, dim=0).unsqueeze(1)


def _minmax_per_channel_batch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    out = x.clone()
    b, c, _, _ = out.shape
    for bi in range(b):
        for ci in range(c):
            channel = out[bi, ci]
            mn = channel.min()
            mx = channel.max()
            if float(mx - mn) > eps:
                channel = (channel - mn) / (mx - mn + eps)
            out[bi, ci] = torch.clamp(channel, 0.0, 1.0)
    return out


def _is_replicated_dem_stream(stream_b: torch.Tensor, eps: float = 1e-3) -> bool:
    """True when stream_b carries the same raster in all channels (Bijie DEM x3 layout)."""
    diff01 = (stream_b[:, 0] - stream_b[:, 1]).abs().mean()
    diff12 = (stream_b[:, 1] - stream_b[:, 2]).abs().mean()
    return bool(diff01 < eps and diff12 < eps)


def observed_stack_from_streams(stream_a: torch.Tensor, stream_b: torch.Tensor) -> torch.Tensor:
    """
    Build the 6-channel Prithvi input from the existing dual-stream batch tensors.

    Landslide4Sense layout (stream_b = NDVI, slope, DEM):
      [R, G, B, NDVI, slope, DEM]

    Bijie layout (stream_b = DEM replicated x3):
      [R, G, B, DEM, slope(DEM), vegetation_index(RGB)]
    """
    if stream_a.shape[1] < 3 or stream_b.shape[1] < 3:
        raise ValueError(
            f"Expected 3+ channels in each stream, got {stream_a.shape} and {stream_b.shape}"
        )

    r, g, b = stream_a[:, 0], stream_a[:, 1], stream_a[:, 2]
    if _is_replicated_dem_stream(stream_b):
        dem = stream_b[:, 0:1]
        slope = _sobel_slope_norm(dem)
        veg = _vegetation_index_from_rgb(stream_a)
        stack = torch.cat([r.unsqueeze(1), g.unsqueeze(1), b.unsqueeze(1), dem, slope, veg], dim=1)
    else:
        ndvi, slope, dem = stream_b[:, 0], stream_b[:, 1], stream_b[:, 2]
        stack = torch.stack([r, g, b, ndvi, slope, dem], dim=1)
    return _minmax_per_channel_batch(stack)


def _extract_dem_channel(stream_b: torch.Tensor) -> torch.Tensor:
    if _is_replicated_dem_stream(stream_b):
        return stream_b[:, 0:1]
    return stream_b[:, 2:3]


def physics_proxies_from_streams(
    stream_a: torch.Tensor, stream_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slope, DEM, and vegetation proxies for PhysicsProxyMapper (no dataset changes)."""
    dem = _extract_dem_channel(stream_b)
    if _is_replicated_dem_stream(stream_b):
        slope = _sobel_slope_norm(dem)
        ndvi = _vegetation_index_from_rgb(stream_a)
    else:
        ndvi = stream_b[:, 0:1]
        slope = stream_b[:, 1:2]
    return slope, dem, ndvi


class GeoPhysicsLandslideNet(nn.Module):
    """
    PS-GPLNet: pentastream physics-closed landslide segmentation.

      - EfficientNet RGB + PhysicsEncoder RGB  -> CMB -> H_rgb
      - EfficientNet DEM + PhysicsEncoder DEM  -> CMB -> H_dem
      - Prithvi FM                             -> T_fm
      - MAO-GeoEGCA + TTEB on hybrid tri-stream pyramids
      - DualPhysicsDecoder + MechanisticPathEquilibriumFusion (MPEF)
    """

    LEGACY_LEVEL_FOR_EFFNET = (1, 2, 3, 4, 4)
    MAO_LEVELS = (2, 3)
    TTEB_LEVELS = (0, 1, 2, 3)

    def __init__(
        self,
        n_classes: int = 1,
        backbone: str = "tf_efficientnet_b4",
        n_channels: int = 3,
        n_channels_b: Optional[int] = None,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        use_input_adapter: bool = False,
        freeze_backbone: bool = True,
        share_backbone: bool = False,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3, 4),
        prithvi_snapshot: Optional[str | Path] = None,
        lora_rank: int = 8,
        fusion_channels: int = 64,
        shared_physics_decoder: bool = False,
        fp16_frozen_encoders: bool = False,
        enable_prithvi: bool = True,
        enable_physics_encoders: bool = True,
        mechanistic_gating: bool = True,
        tteb_attn_chunk: int = 1024,
        tteb_attn_low_res_max: int = 4096,
    ):
        super().__init__()

        if share_backbone:
            warnings.warn(
                "share_backbone is ignored (RGB, DEM, and Prithvi use separate encoders).",
                stacklevel=2,
            )
        if not enable_prithvi:
            raise ValueError("GeoPhysicsLandslideNet requires Prithvi (enable_prithvi=True).")
        if not enable_physics_encoders:
            raise ValueError("GeoPhysicsLandslideNet requires physics encoders (enable_physics_encoders=True).")

        self.enable_prithvi = True
        self.enable_physics_encoders = True
        self.mechanistic_gating = mechanistic_gating
        self.fp16_frozen_encoders = fp16_frozen_encoders
        fusion_ch = int(fusion_channels)
        if fusion_ch < 16:
            raise ValueError("fusion_channels must be >= 16")

        self.encoder_rgb = build_encoder(
            name=backbone,
            n_channels=n_channels,
            out_indices=out_indices,
            pretrained=pretrained if pretrained_path is None else False,
            pretrained_path=pretrained_path,
            use_input_adapter=use_input_adapter,
            freeze=freeze_backbone,
        )
        self.encoder_dem = build_encoder(
            name=backbone,
            n_channels=1,
            out_indices=out_indices,
            pretrained=pretrained if pretrained_path is None else False,
            pretrained_path=pretrained_path,
            use_input_adapter=False,
            freeze=freeze_backbone,
        )

        if tuple(self.encoder_rgb.channels) != tuple(self.encoder_dem.channels):
            raise ValueError(
                f"RGB/DEM encoder channel lists differ: "
                f"{self.encoder_rgb.channels} vs {self.encoder_dem.channels}"
            )
        ch_list = list(self.encoder_rgb.channels)

        self.proxy_rgb = PhysicsProxyMapper()
        self.proxy_dem = PhysicsProxyMapper()
        self.enc_phys_rgb = PhysicsEncoder(
            in_channels=3,
            unified_channels=fusion_ch,
            mechanistic_gating=mechanistic_gating,
        )
        self.enc_phys_dem = PhysicsEncoder(
            in_channels=1,
            unified_channels=fusion_ch,
            mechanistic_gating=mechanistic_gating,
        )
        self.hybrid_rgb = nn.ModuleList(
            [ComplementaryModalityBridge(ch, fusion_ch) for ch in ch_list]
        )
        self.hybrid_dem = nn.ModuleList(
            [ComplementaryModalityBridge(ch, fusion_ch) for ch in ch_list]
        )

        def _to_fusion(in_ch: int) -> nn.Module:
            if in_ch == fusion_ch:
                return nn.Identity()
            return nn.Sequential(
                nn.Conv2d(in_ch, fusion_ch, kernel_size=1, bias=False),
                LN2d(fusion_ch),
                nn.ReLU(inplace=True),
            )

        self.rgb_to_fusion = nn.ModuleList([_to_fusion(ch) for ch in ch_list])
        self.dem_to_fusion = nn.ModuleList([_to_fusion(ch) for ch in ch_list])

        self.encoder_fm = PrithviFoundationEncoder(
            unified_channels=fusion_ch,
            lora_rank=lora_rank,
            snapshot_dir=prithvi_snapshot,
            input_normalization="observed_rasters",
        )
        self.fm_align = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fusion_ch, fusion_ch, kernel_size=1, bias=False),
                    LN2d(fusion_ch),
                    nn.ReLU(inplace=True),
                )
                for _ in ch_list
            ]
        )

        self.fuse3 = MAOGeoEGCA(fusion_ch)
        self.fuse4 = MAOGeoEGCA(fusion_ch)
        self.post_fuse3 = nn.Conv2d(fusion_ch, fusion_ch, kernel_size=1, bias=False)
        self.skips = nn.ModuleList(
            [
                TriTemporalTriStreamBridge(
                    fusion_ch,
                    attn_chunk_size=tteb_attn_chunk,
                    attn_low_res_max=tteb_attn_low_res_max,
                )
                for _ in self.TTEB_LEVELS
            ]
        )

        self.physics_decoder = DualPhysicsDecoder(
            channels=fusion_ch,
            n_classes=n_classes,
            bottleneck_ch=ch_list[4],
            mechanistic_gating=mechanistic_gating,
            shared_decoder=shared_physics_decoder,
        )
        self._fp16_frozen_applied = False

    def apply_fp16_frozen_encoders(self) -> None:
        """Halve GPU memory for frozen EfficientNet + Prithvi backbones (trainable heads stay fp32)."""
        if self._fp16_frozen_applied or not self.fp16_frozen_encoders:
            return
        self.encoder_rgb.net.half()
        self.encoder_dem.net.half()
        self.encoder_fm.model.half()
        self._fp16_frozen_applied = True

    def _align_legacy_level(self, legacy_pyramid: list[torch.Tensor], ref: torch.Tensor, level_i: int) -> torch.Tensor:
        legacy_idx = self.LEGACY_LEVEL_FOR_EFFNET[level_i]
        return match_spatial(legacy_pyramid[legacy_idx], ref)

    def _hybrid_pyramids(
        self,
        rgb_feats: Tuple[torch.Tensor, ...],
        dem_feats: Tuple[torch.Tensor, ...],
        phys_rgb: list[torch.Tensor],
        phys_dem: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        h_rgb, h_dem = [], []
        for i, (rgb, dem) in enumerate(zip(rgb_feats, dem_feats)):
            pr = self._align_legacy_level(phys_rgb, rgb, i)
            pd = self._align_legacy_level(phys_dem, dem, i)
            h_rgb.append(self.hybrid_rgb[i](rgb, pr))
            h_dem.append(self.hybrid_dem[i](dem, pd))
        return h_rgb, h_dem

    def _fusion_pyramids(
        self,
        rgb_feats: Tuple[torch.Tensor, ...],
        dem_feats: Tuple[torch.Tensor, ...],
        fm_feats: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Project rgb/dem/fm native pyramids to unified C=64 for MAO/TTEB."""
        if self.fm_align is None:
            raise RuntimeError("Fusion pyramids requested but Prithvi encoder is disabled.")
        p_rgb, p_dem, t_fm = [], [], []
        for i, rgb in enumerate(rgb_feats):
            dem = dem_feats[i]
            fm_idx = self.LEGACY_LEVEL_FOR_EFFNET[i]
            fm = match_spatial(fm_feats[fm_idx], rgb)
            p_rgb.append(self.rgb_to_fusion[i](rgb))
            p_dem.append(self.dem_to_fusion[i](dem))
            t_fm.append(self.fm_align[i](fm))
        return p_rgb, p_dem, t_fm

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        dem = _extract_dem_channel(x2)
        slope, dem_norm, ndvi = physics_proxies_from_streams(x1, x2)

        enc_dtype = torch.float16 if self._fp16_frozen_applied else x1.dtype
        x1_enc = x1.to(enc_dtype) if enc_dtype != x1.dtype else x1
        dem_enc = dem.to(enc_dtype) if enc_dtype != dem.dtype else dem

        rgb_feats = self.encoder_rgb(x1_enc)
        dem_feats = self.encoder_dem(dem_enc)
        rgb_feats = tuple(f.float() for f in rgb_feats)
        dem_feats = tuple(f.float() for f in dem_feats)
        _, _, _, _, a5 = rgb_feats
        _, _, _, _, b5 = dem_feats

        fm_in = observed_stack_from_streams(x1, x2)
        fm_in_enc = fm_in.to(enc_dtype) if enc_dtype != fm_in.dtype else fm_in
        fm_raw = self.encoder_fm(fm_in_enc)
        fm_raw = [f.float() for f in fm_raw]
        _, _, t_fm = self._fusion_pyramids(rgb_feats, dem_feats, fm_raw)

        alpha_r, h_r, m_r = self.proxy_rgb(slope, dem_norm, ndvi)
        alpha_d, h_d, m_d = self.proxy_dem(slope, dem_norm, ndvi)
        phys_rgb = self.enc_phys_rgb(x1, alpha_r, h_r, m_r)
        phys_dem = self.enc_phys_dem(dem, alpha_d, h_d, m_d)
        p_rgb, p_dem = self._hybrid_pyramids(rgb_feats, dem_feats, phys_rgb, phys_dem)

        tteb_skips = [self.skips[i](p_rgb, p_dem, t_fm, level=i) for i in self.TTEB_LEVELS]

        # MAO at EffNet pyramid indices 2/3 (32² neck, 16² bottleneck).
        f3_fused = self.post_fuse3(self.fuse3(t_fm[2], p_rgb[2], p_dem[2]))
        f4_fused = self.fuse4(t_fm[3], p_rgb[3], p_dem[3])

        main, aux2, aux3, dec_reg = self.physics_decoder(
            f4_fused,
            f3_fused,
            tteb_skips,
            a5,
            b5,
            alpha_r,
            h_r,
            m_r,
            alpha_d,
            h_d,
            m_d,
        )
        target_size = x1.shape[-2:]
        if main.shape[-2:] != target_size:
            main = F.interpolate(main, size=target_size, mode="bilinear", align_corners=False)
        if aux2.shape[-2:] != target_size:
            aux2 = F.interpolate(aux2, size=target_size, mode="bilinear", align_corners=False)
        if aux3.shape[-2:] != target_size:
            aux3 = F.interpolate(aux3, size=target_size, mode="bilinear", align_corners=False)
        return main, aux2, aux3, dec_reg


# Backward-compatible aliases (use GeoPhysicsLandslideNet in new work).
DualStreamGateNet = GeoPhysicsLandslideNet
DiGATe_Unet = GeoPhysicsLandslideNet
