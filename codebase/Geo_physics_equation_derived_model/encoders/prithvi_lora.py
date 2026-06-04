from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projector import StreamProjector

DEFAULT_WEIGHTS_ROOT = Path("/home/user/Desktop/Deep_learning_projects/4PI/prithvi_weights")
DEFAULT_SNAPSHOT = (
    DEFAULT_WEIGHTS_ROOT
    / "models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL"
    / "snapshots"
    / "2c84e383194986040f883cc43d7869002c425e1b"
)

PRITHVI_MEAN = torch.tensor([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]).view(1, 6, 1, 1)
PRITHVI_STD = torch.tensor([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]).view(1, 6, 1, 1)


def _load_prithvi_class(snapshot_dir: Path):
    module_path = snapshot_dir / "prithvi_mae.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Prithvi source not found at {module_path}. Run scripts/download_prithvi.sh first."
        )
    spec = importlib.util.spec_from_file_location("prithvi_mae_local", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["prithvi_mae_local"] = module
    spec.loader.exec_module(module)
    return module.PrithviMAE


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


def _inject_lora_into_block(block, rank: int):
    attn = block.attn
    if not hasattr(attn, "qkv") or not isinstance(attn.qkv, nn.Linear):
        return
    attn.qkv = LoRALinear(attn.qkv, rank=rank)
    if hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
        attn.proj = LoRALinear(attn.proj, rank=rank)


class PrithviFoundationEncoder(nn.Module):
    """Prithvi-EO-2.0-100M-TL encoder with LoRA and multi-scale feature maps."""

    def __init__(
        self,
        unified_channels: int = 64,
        lora_rank: int = 8,
        snapshot_dir: Optional[str | Path] = None,
        block_indices: Optional[List[int]] = None,
        input_normalization: Literal["eo_multispectral", "observed_rasters"] = "observed_rasters",
    ):
        super().__init__()
        snapshot_dir = Path(snapshot_dir or DEFAULT_SNAPSHOT)
        PrithviMAE = _load_prithvi_class(snapshot_dir)

        with open(snapshot_dir / "config.json") as f:
            cfg = json.load(f)["pretrained_cfg"]

        self.img_size = int(cfg.get("img_size", 224))
        self.num_frames = int(cfg.get("num_frames", 4))
        self.model = PrithviMAE(
            img_size=self.img_size,
            num_frames=self.num_frames,
            patch_size=tuple(cfg["patch_size"]),
            in_chans=cfg["in_chans"],
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            decoder_embed_dim=cfg["decoder_embed_dim"],
            decoder_depth=cfg["decoder_depth"],
            decoder_num_heads=cfg["decoder_num_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            coords_encoding=cfg.get("coords_encoding"),
            coords_scale_learn=cfg.get("coords_scale_learn", True),
            mask_ratio=0.0,
        )

        ckpt_path = snapshot_dir / "Prithvi_EO_V2_100M_TL.pt"
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)

        for p in self.model.parameters():
            p.requires_grad = False
        for block in self.model.encoder.blocks:
            _inject_lora_into_block(block, rank=lora_rank)

        self.block_indices = block_indices or [2, 5, 8, 11]
        embed_dim = cfg["embed_dim"]
        self.spatial_projs = nn.ModuleList(
            [nn.Conv2d(embed_dim, unified_channels, kernel_size=1, bias=False) for _ in self.block_indices]
        )
        self.proj0 = StreamProjector(unified_channels, unified_channels)
        self.proj1 = StreamProjector(unified_channels, unified_channels)
        self.proj2 = StreamProjector(unified_channels, unified_channels)
        self.proj3 = StreamProjector(unified_channels, unified_channels)
        self.proj4 = StreamProjector(unified_channels, unified_channels)

        self.input_normalization = input_normalization
        self.register_buffer("band_mean", PRITHVI_MEAN.clone())
        self.register_buffer("band_std", PRITHVI_STD.clone())

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize 6-channel input before the ViT backbone.

        observed_rasters: channels are measured/derived rasters in [0, 1] (Bijie RGB+DEM,
          L4S RGB+NDVI+slope+DEM). Do not apply satellite EO band mean/std.
        eo_multispectral: true Prithvi EO stack (B, G, R, NIR, SWIR1, SWIR2 reflectance).
        """
        if self.input_normalization == "eo_multispectral":
            x = x * 10000.0
            mean = self.band_mean.to(dtype=x.dtype, device=x.device)
            std = self.band_std.to(dtype=x.dtype, device=x.device)
            return (x - mean) / std
        return (x - 0.5) / 0.25

    def _tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        prepared = self.model.encoder.prepare_features_for_image_model([tokens])[0]
        b, c, h, w = prepared.shape
        t = max(1, self.num_frames)
        if c % t == 0:
            e = c // t
            prepared = prepared.view(b, t, e, h, w).mean(dim=1)
        return prepared

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.normalize_input(x)
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        if x.dim() == 4:
            x = x.unsqueeze(2)
        if x.shape[2] == 1 and self.num_frames > 1:
            x = x.repeat(1, 1, self.num_frames, 1, 1)

        feats = self.model.forward_features(x, temporal_coords=None, location_coords=None)
        target_sizes = [(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)]
        maps = []
        for idx, block_i in enumerate(self.block_indices):
            tok = feats[block_i]
            m = self._tokens_to_map(tok)
            m = self.spatial_projs[idx](m)
            m = F.interpolate(m, size=target_sizes[idx + 1], mode="bilinear", align_corners=False)
            maps.append(m)

        l1, l2, l3, l4 = maps
        l0 = F.interpolate(l1, size=target_sizes[0], mode="bilinear", align_corners=False)
        return [
            self.proj0(l0),
            self.proj1(l1),
            self.proj2(l2),
            self.proj3(l3),
            self.proj4(l4),
        ]
