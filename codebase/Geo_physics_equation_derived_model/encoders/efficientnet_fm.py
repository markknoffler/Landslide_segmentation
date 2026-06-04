"""ImageNet-pretrained EfficientNet backbone for the foundation-model stream (dual-stream parity)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .projector import StreamProjector

# Align with GeoPhysicsLandslideNet pyramid (L0..L4).
TARGET_SIZES: Tuple[Tuple[int, int], ...] = ((256, 256), (128, 128), (64, 64), (32, 32), (16, 16))


class EfficientNetFoundationEncoder(nn.Module):
    """
    Foundation encoder using timm EfficientNet-B4 (same default as dual_stream_gated).

    Input: B×3×H×W RGB in [0, 1] per channel (Bijie stream_a).
    Output: five maps [L0..L4] with unified channel width C.
    """

    def __init__(
        self,
        unified_channels: int = 64,
        backbone: str = "tf_efficientnet_b4",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3, 4),
    ):
        super().__init__()
        self.backbone_name = backbone
        self.out_indices = tuple(out_indices)

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
            in_chans=3,
        )
        in_channels: List[int] = list(self.backbone.feature_info.channels())

        self.spatial_projs = nn.ModuleList(
            [nn.Conv2d(in_ch, unified_channels, kernel_size=1, bias=False) for in_ch in in_channels]
        )
        self.proj0 = StreamProjector(unified_channels, unified_channels)
        self.proj1 = StreamProjector(unified_channels, unified_channels)
        self.proj2 = StreamProjector(unified_channels, unified_channels)
        self.proj3 = StreamProjector(unified_channels, unified_channels)
        self.proj4 = StreamProjector(unified_channels, unified_channels)
        self.projectors = nn.ModuleList(
            [self.proj0, self.proj1, self.proj2, self.proj3, self.proj4]
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.shape[1] != 3:
            raise ValueError(f"EfficientNet FM expects 3-channel RGB, got {x.shape[1]} channels")

        feats = self.backbone(x)
        if len(feats) != len(TARGET_SIZES):
            raise RuntimeError(
                f"Expected {len(TARGET_SIZES)} feature maps from {self.backbone_name}, got {len(feats)}"
            )

        outputs: list[torch.Tensor] = []
        for feat, proj_layer, spatial_proj, target_size in zip(
            feats, self.projectors, self.spatial_projs, TARGET_SIZES
        ):
            m = spatial_proj(feat)
            if m.shape[-2:] != target_size:
                m = F.interpolate(m, size=target_size, mode="bilinear", align_corners=False)
            outputs.append(proj_layer(m))
        return outputs
