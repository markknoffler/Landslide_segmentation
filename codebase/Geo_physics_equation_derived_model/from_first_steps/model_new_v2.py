from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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


class PrithviEncoder(nn.Module):
    """
    Prithvi encoder for RGB data - using a different backbone than EfficientNet
    """
    def __init__(self, n_channels: int = 3, pretrained: bool = True):
        super().__init__()
        # Using a different backbone for Prithvi (this would be a placeholder)
        # In practice, this would be a specific Prithvi model
        self.encoder = build_encoder(
            name="prithvi_vit_384",  # Placeholder - this would be your actual Prithvi backbone
            n_channels=n_channels,
            pretrained=pretrained,
            freeze=True
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.encoder(x)


class DualStreamMixedNet(nn.Module):
    """
    Model with:
    1. One Prithvi encoder for RGB data
    2. Two EfficientNet encoders (one for DEM and one for RGB)
    3. Single decoder that combines features from all three encoders
    """
    def __init__(
        self,
        n_classes: int = 1,
        backbone_rgb: str = "tf_efficientnet_b4",
        backbone_dem: str = "tf_efficientnet_b4",
        backbone_prithvi: str = "prithvi_vit_384",
        n_channels_rgb: int = 3,
        n_channels_dem: int = 1,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        use_input_adapter: bool = False,
        freeze_backbone: bool = True,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3, 4),
    ):
        super().__init__()

        # Prithvi encoder for RGB data
        self.prithvi_encoder = PrithviEncoder(
            n_channels=n_channels_rgb,
            pretrained=pretrained
        )

        # EfficientNet encoder for DEM data
        self.dem_encoder = build_encoder(
            name=backbone_dem,
            n_channels=n_channels_dem,
            out_indices=out_indices,
            pretrained=pretrained if pretrained_path is None else False,
            pretrained_path=pretrained_path,
            use_input_adapter=use_input_adapter,
            freeze=freeze_backbone,
        )

        # EfficientNet encoder for RGB data (separate from Prithvi)
        self.rgb_encoder = build_encoder(
            name=backbone_rgb,
            n_channels=n_channels_rgb,
            out_indices=out_indices,
            pretrained=pretrained if pretrained_path is None else False,
            pretrained_path=pretrained_path,
            use_input_adapter=use_input_adapter,
            freeze=freeze_backbone,
        )

        # Get channel sizes from the encoders
        # Note: Using the dem_encoder channel sizes as reference
        ch_list = self.dem_encoder.channels

        # Single decoder that processes features from all three encoders
        self.decoder = AdaptiveDecoder(ch_list)

        # Gate fusion modules for combining features from different encoders
        self.fuse_prithvi_dem = GateFuse(ch_list[-1])  # Fuse at highest resolution
        self.fuse_rgb_dem = GateFuse(ch_list[-1])      # Fuse at highest resolution

        # Additional fusion modules for intermediate features
        self.fuse_prithvi_rgb = GateFuse(ch_list[-2])  # Fuse at second highest resolution

        # Final head components
        final_ch = self.decoder.final_ch
        self.up_final = SubPixelUp(final_ch, final_ch // 2)
        self.head = OutConv(final_ch // 2, n_classes)
        self.aux2 = OutConv(self.decoder.ch_x3, n_classes)
        self.aux3 = OutConv(self.decoder.ch_x4, n_classes)

    def forward(
        self,
        rgb: torch.Tensor,
        dem: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        # Process RGB data through Prithvi encoder
        prithvi_feats = self.prithvi_encoder(rgb)

        # Process DEM data through EfficientNet encoder
        dem_feats = self.dem_encoder(dem)

        # Process RGB data through EfficientNet encoder
        rgb_feats = self.rgb_encoder(rgb)

        # Unpack features from each encoder
        prithvi_f1, prithvi_f2, prithvi_f3, prithvi_f4, prithvi_f5 = prithvi_feats
        dem_f1, dem_f2, dem_f3, dem_f4, dem_f5 = dem_feats
        rgb_f1, rgb_f2, rgb_f3, rgb_f4, rgb_f5 = rgb_feats

        # Gate fusion of features from different encoders
        # Fuse Prithvi and DEM features at highest resolution
        fused_f5, reg_f5 = self.fuse_prithvi_dem(prithvi_f5, dem_f5)

        # Fuse RGB and DEM features at highest resolution
        fused_f5_2, reg_f5_2 = self.fuse_rgb_dem(rgb_f5, dem_f5)

        # Combine the fused features (simple concatenation or more sophisticated fusion)
        # For now, we'll use the Prithvi-Dem fused features as the main input to decoder
        # This can be modified based on specific requirements
        fused_features = (prithvi_f1, prithvi_f2, prithvi_f3, fused_f5, prithvi_f5)

        # Process through decoder
        x1, x2, x3, x4 = self.decoder(*fused_features)

        # Final prediction
        main = self.head(self.up_final(x4))
        aux2 = F.interpolate(self.aux2(x3), size=main.shape[2:], mode="bilinear", align_corners=True)
        aux3 = F.interpolate(self.aux3(x4), size=main.shape[2:], mode="bilinear", align_corners=True)
        reg = (reg_f5, reg_f5_2)  # Regularization terms from gate fusion

        return main, aux2, aux3, reg


# Alias for backward compatibility
DiGATe_Unet = DualStreamMixedNet