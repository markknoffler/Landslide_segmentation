"""ImageNet-pretrained EfficientNet backbone for the foundation-model stream."""

from __future__ import annotations

from typing import List, Literal, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .projector import StreamProjector
from .pyramid_utils import group_norm_groups

PyramidMode = Literal["native", "legacy"]

# Legacy Geo pyramid (physics-aligned fixed grid).
LEGACY_TARGET_SIZES = ((256, 256), (128, 128), (64, 64), (32, 32), (16, 16))


class EfficientNetFoundationEncoder(nn.Module):
    """
    Foundation encoder using timm EfficientNet (default: tf_efficientnet_b4).

    pyramid_mode:
      native  - keep timm spatial sizes and channel widths (dual-stream parity).
                Example for 256x256 input: sizes [128,64,32,16,8], channels [24,32,56,160,448].
      legacy  - force [256,128,64,32,16] and project every level to unified_channels.
    """

    def __init__(
        self,
        unified_channels: int = 64,
        backbone: str = "tf_efficientnet_b4",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3, 4),
        pyramid_mode: PyramidMode | str = "native",
    ):
        super().__init__()
        self.backbone_name = backbone
        self.out_indices = tuple(out_indices)
        self.pyramid_mode = str(pyramid_mode).lower()
        self.unified_channels = unified_channels

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
            in_chans=3,
        )
        self.level_channels: List[int] = list(self.backbone.feature_info.channels())

        if self.pyramid_mode == "native":
            self.refiners = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(ch, ch, kernel_size=1, bias=False),
                        nn.GroupNorm(group_norm_groups(ch), ch),
                        nn.ReLU(inplace=True),
                    )
                    for ch in self.level_channels
                ]
            )
            self.spatial_projs = None
            self.projectors = None
        elif self.pyramid_mode == "legacy":
            self.spatial_projs = nn.ModuleList(
                [
                    nn.Conv2d(in_ch, unified_channels, kernel_size=1, bias=False)
                    for in_ch in self.level_channels
                ]
            )
            self.proj0 = StreamProjector(unified_channels, unified_channels)
            self.proj1 = StreamProjector(unified_channels, unified_channels)
            self.proj2 = StreamProjector(unified_channels, unified_channels)
            self.proj3 = StreamProjector(unified_channels, unified_channels)
            self.proj4 = StreamProjector(unified_channels, unified_channels)
            self.projectors = nn.ModuleList(
                [self.proj0, self.proj1, self.proj2, self.proj3, self.proj4]
            )
            self.refiners = None
        else:
            raise ValueError(
                f"Unknown pyramid_mode={pyramid_mode!r}. Choose 'native' or 'legacy'."
            )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    @property
    def uses_native_pyramid(self) -> bool:
        return self.pyramid_mode == "native"

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.shape[1] != 3:
            raise ValueError(f"EfficientNet FM expects 3-channel RGB, got {x.shape[1]} channels")

        feats = self.backbone(x)
        if len(feats) != len(self.level_channels):
            raise RuntimeError(
                f"Expected {len(self.level_channels)} feature maps from {self.backbone_name}, "
                f"got {len(feats)}"
            )

        if self.pyramid_mode == "native":
            return [refiner(feat) for refiner, feat in zip(self.refiners, feats)]

        outputs: list[torch.Tensor] = []
        for feat, proj_layer, spatial_proj, target_size in zip(
            feats, self.projectors, self.spatial_projs, LEGACY_TARGET_SIZES
        ):
            m = spatial_proj(feat)
            if m.shape[-2:] != target_size:
                m = F.interpolate(m, size=target_size, mode="bilinear", align_corners=False)
            outputs.append(proj_layer(m))
        return outputs
