"""Prithvi-EO-2.0 foundation encoder with LoRA (adapted from Geo_physics_equation_derived_model)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_WEIGHT_NAMES = ("Prithvi_EO_V2_100M_TL.pt", "pytorch_model.bin")
_REPO_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL"


def _has_weights(path: Path) -> bool:
    return any((path / name).is_file() for name in _WEIGHT_NAMES)


def _is_prithvi_snapshot(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and _has_weights(path)


def _hf_cache_root(candidate: Path) -> Optional[Path]:
    for parent in [candidate, *candidate.parents]:
        if parent.name == "models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL":
            return parent
    return None


def _hf_hub_fetch(filename: str, snapshot_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    cache_root = _hf_cache_root(snapshot_dir)
    kwargs = {"repo_id": _REPO_ID, "filename": filename}
    if cache_root is not None:
        kwargs["cache_dir"] = str(cache_root.parent)
    downloaded = Path(hf_hub_download(**kwargs))
    target = snapshot_dir / filename
    if not target.is_file() and downloaded.is_file():
        target.write_bytes(downloaded.read_bytes())
    if not target.is_file():
        raise FileNotFoundError(f"Failed to place {filename} under {snapshot_dir}")
    return target


def _ensure_prithvi_files(snapshot_dir: Path) -> None:
    if not (snapshot_dir / "config.json").is_file():
        _hf_hub_fetch("config.json", snapshot_dir)
    if not _has_weights(snapshot_dir):
        _hf_hub_fetch("Prithvi_EO_V2_100M_TL.pt", snapshot_dir)
    if not (snapshot_dir / "prithvi_mae.py").is_file():
        _hf_hub_fetch("prithvi_mae.py", snapshot_dir)


def resolve_prithvi_snapshot(path: Optional[str | Path] = None) -> Path:
    """
    Resolve a HuggingFace cache root or snapshot directory to a valid Prithvi snapshot.

    Accepts:
      - snapshot dir containing config.json + Prithvi weights (+ prithvi_mae.py, fetched if missing)
      - HF cache root .../models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL
      - path to Prithvi_EO_V2_100M_TL.pt (parent snapshot dir is used)
      - env PRITHVI_SNAPSHOT or PRITHVI_WEIGHTS_ROOT when path is None
    """
    if path is None:
        env_snapshot = os.environ.get("PRITHVI_SNAPSHOT")
        if env_snapshot:
            path = env_snapshot
        elif os.environ.get("PRITHVI_WEIGHTS_ROOT"):
            path = Path(os.environ["PRITHVI_WEIGHTS_ROOT"]) / "models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL"
        elif DEFAULT_SNAPSHOT.is_dir() and _is_prithvi_snapshot(DEFAULT_SNAPSHOT):
            path = DEFAULT_SNAPSHOT
        else:
            raise FileNotFoundError(
                "Prithvi snapshot path is required. Pass --prithvi_snapshot "
                "/scratch/.../models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL "
                "or set env PRITHVI_SNAPSHOT."
            )

    candidate = Path(path).expanduser().resolve()
    if candidate.is_file() and candidate.suffix == ".pt":
        candidate = candidate.parent

    resolved: Optional[Path] = None
    if _is_prithvi_snapshot(candidate):
        resolved = candidate
    else:
        snapshots_dir = candidate / "snapshots"
        if snapshots_dir.is_dir():
            options = sorted(
                (p for p in snapshots_dir.iterdir() if p.is_dir() and _is_prithvi_snapshot(p)),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if options:
                resolved = options[0]

    if resolved is None:
        raise FileNotFoundError(
            "Could not find a valid Prithvi snapshot at "
            f"{candidate}. Expected config.json and Prithvi_EO_V2_100M_TL.pt "
            "either directly in that directory or under snapshots/<hash>/."
        )

    _ensure_prithvi_files(resolved)
    return resolved


DEFAULT_WEIGHTS_ROOT = Path("/home/user/Desktop/Deep_learning_projects/4PI/prithvi_weights")
DEFAULT_SNAPSHOT = (
    DEFAULT_WEIGHTS_ROOT
    / "models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL"
    / "snapshots"
    / "2c84e383194986040f883cc43d7869002c425e1b"
)

PRITHVI_MEAN = torch.tensor([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]).view(1, 6, 1, 1)
PRITHVI_STD = torch.tensor([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]).view(1, 6, 1, 1)

LEGACY_TARGET_SIZES = ((256, 256), (128, 128), (64, 64), (32, 32), (16, 16))


def _load_prithvi_class(snapshot_dir: Path):
    module_path = snapshot_dir / "prithvi_mae.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Prithvi source not found at {module_path}. "
            "Run codebase/Geo_physics_equation_derived_model/scripts/download_prithvi.sh first."
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


class StreamProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


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
        snapshot_dir = resolve_prithvi_snapshot(snapshot_dir)
        print(f"Using Prithvi snapshot: {snapshot_dir}")
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

        # Decoder weights are unused (encoder-only features); drop to save GPU memory.
        if hasattr(self.model, "decoder"):
            del self.model.decoder

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
        if x.shape[1] != 6:
            raise ValueError(f"PrithviFoundationEncoder expects 6 channels, got {x.shape[1]}")

        x = self.normalize_input(x)
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        if x.dim() == 4:
            x = x.unsqueeze(2)
        if x.shape[2] == 1 and self.num_frames > 1:
            x = x.repeat(1, 1, self.num_frames, 1, 1)

        feats = self.model.forward_features(x, temporal_coords=None, location_coords=None)
        maps = []
        for idx, block_i in enumerate(self.block_indices):
            tok = feats[block_i]
            m = self._tokens_to_map(tok)
            m = self.spatial_projs[idx](m)
            m = F.interpolate(m, size=LEGACY_TARGET_SIZES[idx + 1], mode="bilinear", align_corners=False)
            maps.append(m)

        l1, l2, l3, l4 = maps
        l0 = F.interpolate(l1, size=LEGACY_TARGET_SIZES[0], mode="bilinear", align_corners=False)
        return [
            self.proj0(l0),
            self.proj1(l1),
            self.proj2(l2),
            self.proj3(l3),
            self.proj4(l4),
        ]
