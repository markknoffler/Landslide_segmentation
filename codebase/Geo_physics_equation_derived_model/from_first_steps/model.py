from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from decoder import DualPhysicsGatedDecoder
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
            pretrained=False,
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
        elif spec.pretrained:
            net = timm.create_model(
                spec.name,
                pretrained=True,
                features_only=True,
                out_indices=spec.out_indices,
                in_chans=target_in,
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


class SubPixelUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 1, bias=False)
        self.norm = LN2d(out_ch * 4)
        self.act = nn.ReLU(inplace=True)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.act(self.norm(self.conv(x))))


class DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, mid_c: Optional[int] = None):
        super().__init__()
        mid_c = out_c if mid_c is None else mid_c
        self.block = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, 1, 1, bias=False),
            LN2d(mid_c),
            nn.ReLU(True),
            nn.Conv2d(mid_c, out_c, 3, 1, 1, bias=False),
            LN2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, g_ch: int, x_ch: int, inter: int):
        super().__init__()
        inter = max(1, inter)
        self.Wg = nn.Sequential(nn.Conv2d(g_ch, inter, 1, bias=False), LN2d(inter))
        self.Wx = nn.Sequential(nn.Conv2d(x_ch, inter, 1, bias=False), LN2d(inter))
        self.psi = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(inter, 1, 1, bias=False),
            nn.BatchNorm2d(1, affine=False),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(self.Wg(g) + self.Wx(x))
        return alpha * x


class UpFlex(nn.Module):
    def __init__(self, dec_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = SubPixelUp(dec_ch, dec_ch // 2)
        self.attn = AttentionGate(dec_ch // 2, skip_ch, inter=min(dec_ch // 2, skip_ch) // 4)
        self.conv = DoubleConv(dec_ch // 2 + skip_ch, out_ch)

    def forward(self, d: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        d = self.up(d)
        dy, dx = s.size(2) - d.size(2), s.size(3) - d.size(3)
        d = F.pad(d, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        s = self.attn(d, s)
        return self.conv(torch.cat([s, d], 1))


class XAttn(nn.Module):
    def __init__(self, dim: int, heads: int = 2, mlp: float = 2.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.attn(self.ln1(q), self.ln1(k), self.ln1(v))
        q = q + hidden
        return q + self.mlp(self.ln2(q))


class TransUp(nn.Module):
    def __init__(self, dec_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = SubPixelUp(dec_ch, dec_ch // 2)
        embed_dim = out_ch
        self.proj_q = nn.Conv2d(dec_ch // 2, embed_dim, 1, bias=False)
        self.proj_k = nn.Conv2d(skip_ch, embed_dim, 1, bias=False)
        self.proj_v = nn.Conv2d(skip_ch, embed_dim, 1, bias=False)
        self.xattn = XAttn(embed_dim)
        self.post = nn.Sequential(nn.Conv2d(embed_dim, out_ch, 3, 1, 1, bias=False), LN2d(out_ch), nn.ReLU(True))

    def forward(self, d: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        d = self.up(d)
        dy, dx = s.size(2) - d.size(2), s.size(3) - d.size(3)
        d = F.pad(d, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        batch, _, height, width = d.shape
        q = self.proj_q(d).flatten(2).transpose(1, 2)
        k = self.proj_k(s).flatten(2).transpose(1, 2)
        v = self.proj_v(s).flatten(2).transpose(1, 2)
        q = checkpoint(self.xattn, q, k, v) if q.requires_grad else self.xattn(q, k, v)
        q = q.transpose(1, 2).reshape(batch, -1, height, width)
        return self.post(q)


class OutConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AdaptiveDecoder(nn.Module):
    def __init__(self, ch_list: List[int]):
        super().__init__()
        c1, c2, c3, c4, c5 = ch_list
        self.up1 = TransUp(c5, c4, c4 // 2)
        self.up2 = UpFlex(c4 // 2, c3, c3 // 2)
        self.up3 = UpFlex(c3 // 2, c2, c2 // 2)
        self.up4 = UpFlex(c2 // 2, c1, c1 // 2)

        self.ch_x1 = c4 // 2
        self.ch_x2 = c3 // 2
        self.ch_x3 = c2 // 2
        self.ch_x4 = c1 // 2
        self.final_ch = self.ch_x4

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
        f5: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.up1(f5, f4)
        x2 = self.up2(x1, f3)
        x3 = self.up3(x2, f2)
        x4 = self.up4(x3, f1)
        return x1, x2, x3, x4


class GateFuse(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.g = nn.Sequential(nn.Conv2d(ch * 2, 1, 1), nn.Sigmoid())

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = self.g(torch.cat([a, b], dim=1))
        out = alpha * a + (1 - alpha) * b
        reg = torch.mean(alpha * (1 - alpha))
        return out, reg


class DualStreamGateNet(nn.Module):
    """
    Penta-stream model (Step 3 default):
      - EfficientNet RGB + PhysicsEncoder RGB  -> ComplementaryModalityBridge -> H_rgb
      - EfficientNet DEM + PhysicsEncoder DEM    -> ComplementaryModalityBridge -> H_dem
      - Prithvi FM                             -> T_fm
      - MAO-GeoEGCA @ c3/c4 + TTEB skips @ L0-L3 on (H_rgb, H_dem, T_fm)
      - DualPhysicsGatedDecoder (physics cells + PGDI + dual GateFuse)

    fusion_type='gate' + decoder_type='paper' reproduces Step 1/2 ablations.
    """

    PRITHVI_CHANNELS = 64
    FUSION_CHANNELS = 64
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
        enable_prithvi: bool = True,
        enable_physics_encoders: bool = True,
        fusion_type: str = "mao",
        decoder_type: str = "physics",
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

        self.enable_prithvi = enable_prithvi
        self.enable_physics_encoders = enable_physics_encoders
        self.fusion_type = str(fusion_type).lower()
        self.decoder_type = str(decoder_type).lower()
        self.mechanistic_gating = mechanistic_gating
        if self.fusion_type not in {"mao", "gate"}:
            raise ValueError(f"Unknown fusion_type={fusion_type!r}. Choose 'mao' or 'gate'.")
        if self.decoder_type not in {"physics", "paper"}:
            raise ValueError(f"Unknown decoder_type={decoder_type!r}. Choose 'physics' or 'paper'.")
        if self.decoder_type == "physics":
            if self.fusion_type != "mao":
                raise ValueError("decoder_type='physics' requires fusion_type='mao'.")
            if not enable_physics_encoders:
                raise ValueError("decoder_type='physics' requires enable_physics_encoders=True.")
            if not enable_prithvi:
                raise ValueError("decoder_type='physics' requires Prithvi enabled.")

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
        fusion_ch = self.FUSION_CHANNELS

        if self.enable_physics_encoders:
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
        else:
            self.proxy_rgb = None
            self.proxy_dem = None
            self.enc_phys_rgb = None
            self.enc_phys_dem = None
            self.hybrid_rgb = None
            self.hybrid_dem = None

        def _to_fusion(in_ch: int) -> nn.Module:
            if in_ch == fusion_ch:
                return nn.Identity()
            return nn.Sequential(
                nn.Conv2d(in_ch, fusion_ch, kernel_size=1, bias=False),
                LN2d(fusion_ch),
                nn.ReLU(inplace=True),
            )

        def _from_fusion(out_ch: int) -> nn.Module:
            if out_ch == fusion_ch:
                return nn.Identity()
            return nn.Sequential(
                nn.Conv2d(fusion_ch, out_ch, kernel_size=1, bias=False),
                LN2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.rgb_to_fusion = nn.ModuleList([_to_fusion(ch) for ch in ch_list])
        self.dem_to_fusion = nn.ModuleList([_to_fusion(ch) for ch in ch_list])
        self.fusion_to_decoder = nn.ModuleList([_from_fusion(ch) for ch in ch_list])

        if self.enable_prithvi:
            self.encoder_fm = PrithviFoundationEncoder(
                unified_channels=self.PRITHVI_CHANNELS,
                lora_rank=lora_rank,
                snapshot_dir=prithvi_snapshot,
                input_normalization="observed_rasters",
            )
            self.fm_align = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(self.PRITHVI_CHANNELS, fusion_ch, kernel_size=1, bias=False),
                        LN2d(fusion_ch),
                        nn.ReLU(inplace=True),
                    )
                    for _ in ch_list
                ]
            )
        else:
            self.encoder_fm = None
            self.fm_align = None

        if self.fusion_type == "mao" and self.enable_prithvi:
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
                    for i in self.TTEB_LEVELS
                ]
            )
            self.efuse_c4_ab = None
            self.efuse_c4_fm = None
            self.efuse_c3_ab = None
            self.efuse_c3_fm = None
        else:
            self.fuse3 = None
            self.fuse4 = None
            self.post_fuse3 = None
            self.skips = None
            self.efuse_c4_ab = GateFuse(ch_list[3])
            self.efuse_c4_fm = GateFuse(ch_list[3]) if self.enable_prithvi else None
            self.efuse_c3_ab = GateFuse(ch_list[2])
            self.efuse_c3_fm = GateFuse(ch_list[2]) if self.enable_prithvi else None

        if self.decoder_type == "physics":
            self.physics_decoder = DualPhysicsGatedDecoder(
                channels=fusion_ch,
                n_classes=n_classes,
                bottleneck_ch=ch_list[4],
                mechanistic_gating=mechanistic_gating,
            )
            self.decoderA = None
            self.decoderB = None
            self.fuse_x3 = None
            self.fuse_x4 = None
            self.up_final = None
            self.head = None
            self.aux2 = None
            self.aux3 = None
        else:
            self.physics_decoder = None
            self.decoderA = AdaptiveDecoder(ch_list)
            self.decoderB = AdaptiveDecoder(ch_list)
            self.fuse_x3 = GateFuse(self.decoderA.ch_x3)
            self.fuse_x4 = GateFuse(self.decoderA.ch_x4)
            final_ch = self.decoderA.final_ch
            self.up_final = SubPixelUp(final_ch, final_ch // 2)
            self.head = OutConv(final_ch // 2, n_classes)
            self.aux2 = OutConv(self.decoderA.ch_x3, n_classes)
            self.aux3 = OutConv(self.decoderA.ch_x4, n_classes)

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

    def _decode_from_fusion(self, level: int, fused: torch.Tensor) -> torch.Tensor:
        return self.fusion_to_decoder[level](fused)

    def _fuse_encoder_level_gate(
        self,
        rgb_feat: torch.Tensor,
        dem_feat: torch.Tensor,
        fm_feat: Optional[torch.Tensor],
        fuse_ab: GateFuse,
        fuse_fm: Optional[GateFuse],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        fused_ab, reg_ab = fuse_ab(rgb_feat, dem_feat)
        if fm_feat is None or fuse_fm is None:
            return fused_ab, (reg_ab,)
        fused, reg_fm = fuse_fm(fused_ab, fm_feat)
        return fused, (reg_ab, reg_fm)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        dem = _extract_dem_channel(x2)
        slope, dem_norm, ndvi = physics_proxies_from_streams(x1, x2)

        rgb_feats = self.encoder_rgb(x1)
        dem_feats = self.encoder_dem(dem)
        a1, a2, a3, a4, a5 = rgb_feats
        b1, b2, b3, b4, b5 = dem_feats

        reg_encoder: Tuple[torch.Tensor, ...] = ()

        if self.fusion_type == "mao" and self.enable_prithvi and self.encoder_fm is not None:
            fm_in = observed_stack_from_streams(x1, x2)
            fm_raw = self.encoder_fm(fm_in)
            _, _, t_fm = self._fusion_pyramids(rgb_feats, dem_feats, fm_raw)

            if self.enable_physics_encoders:
                alpha_r, h_r, m_r = self.proxy_rgb(slope, dem_norm, ndvi)
                alpha_d, h_d, m_d = self.proxy_dem(slope, dem_norm, ndvi)
                phys_rgb = self.enc_phys_rgb(x1, alpha_r, h_r, m_r)
                phys_dem = self.enc_phys_dem(dem, alpha_d, h_d, m_d)
                p_rgb, p_dem = self._hybrid_pyramids(rgb_feats, dem_feats, phys_rgb, phys_dem)
            else:
                alpha_r = h_r = m_r = alpha_d = h_d = m_d = None
                p_rgb, p_dem, _ = self._fusion_pyramids(rgb_feats, dem_feats, fm_raw)

            tteb_skips = [self.skips[i](p_rgb, p_dem, t_fm, level=i) for i in self.TTEB_LEVELS]

            # MAO at EffNet pyramid indices 2/3 (32² neck, 16² bottleneck).
            # Index 4 is 8² for B4@256 — too coarse for the 4-stage physics decoder to reach 256².
            f3_fused = self.post_fuse3(self.fuse3(t_fm[2], p_rgb[2], p_dem[2]))
            f4_fused = self.fuse4(t_fm[3], p_rgb[3], p_dem[3])

            if self.decoder_type == "physics" and self.physics_decoder is not None:
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

            f3 = self._decode_from_fusion(2, f3_fused)
            f4 = self._decode_from_fusion(3, f4_fused)
            s0 = self._decode_from_fusion(0, tteb_skips[0])
            s1 = self._decode_from_fusion(1, tteb_skips[1])

            _, _, x3a, x4a = self.decoderA(s0, s1, f3, f4, a5)
            _, _, x3b, x4b = self.decoderB(s0, s1, f3, f4, b5)
        else:
            fm3 = fm4 = None
            if self.enable_prithvi and self.encoder_fm is not None and self.fm_align is not None:
                fm_in = observed_stack_from_streams(x1, x2)
                fm_raw = self.encoder_fm(fm_in)
                _, _, t_fm = self._fusion_pyramids(rgb_feats, dem_feats, fm_raw)
                fm3 = self._decode_from_fusion(2, t_fm[2])
                fm4 = self._decode_from_fusion(3, t_fm[3])

            f4, reg_c4 = self._fuse_encoder_level_gate(a4, b4, fm4, self.efuse_c4_ab, self.efuse_c4_fm)
            f3, reg_c3 = self._fuse_encoder_level_gate(a3, b3, fm3, self.efuse_c3_ab, self.efuse_c3_fm)
            reg_encoder = (*reg_c4, *reg_c3)

            _, _, x3a, x4a = self.decoderA(a1, a2, f3, f4, a5)
            _, _, x3b, x4b = self.decoderB(b1, b2, f3, f4, b5)

        x3, reg_x3 = self.fuse_x3(x3a, x3b)
        x4, reg_x4 = self.fuse_x4(x4a, x4b)

        main = self.head(self.up_final(x4))
        aux2 = F.interpolate(self.aux2(x3), size=main.shape[2:], mode="bilinear", align_corners=True)
        aux3 = F.interpolate(self.aux3(x4), size=main.shape[2:], mode="bilinear", align_corners=True)
        reg = (*reg_encoder, reg_x3, reg_x4)
        return main, aux2, aux3, reg


DiGATe_Unet = DualStreamGateNet
