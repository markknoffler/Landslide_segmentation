from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .terrain import (
    build_bijie_observed_stack,
    build_l4s_observed_stack,
    sobel_slope_norm,
    vegetation_index_from_rgb,
)

_ABLATION_COMMON = (
    Path(__file__).resolve().parents[2]
    / "ablation_study"
    / "baseline_models"
)
if str(_ABLATION_COMMON) not in sys.path:
    sys.path.insert(0, str(_ABLATION_COMMON))

from common.datasets import (  # noqa: E402
    AugmentDual2D,
    BijieTwoComposites,
    L4SDualStreamDataset,
    build_bijie_split,
    build_l4s_split,
)


def _fm_rgb(stream_a: torch.Tensor) -> torch.Tensor:
    """Measured RGB (3, H, W) for EfficientNet FM — same tensor as stream_a."""
    return stream_a[:3].contiguous()


def _bijie_physics_and_fm(stream_a: torch.Tensor, dem: torch.Tensor, fm_backbone: str) -> dict[str, torch.Tensor]:
    """Physics proxies from measured RGB + DEM; FM input depends on backbone flag."""
    rgb = stream_a[:3]
    dem_ch = dem if dem.dim() == 3 else dem.unsqueeze(0)
    slope = torch.from_numpy(sobel_slope_norm(dem_ch[0].numpy())).unsqueeze(0)
    ndvi = torch.from_numpy(vegetation_index_from_rgb(rgb.numpy())).unsqueeze(0)

    if fm_backbone == "efficientnet":
        fm_input = _fm_rgb(stream_a)
    elif fm_backbone == "prithvi":
        fm_input = build_bijie_observed_stack(rgb.numpy(), dem_ch.numpy())
    else:
        raise ValueError(f"Unknown fm_backbone={fm_backbone!r}")

    return {
        "fm_input": fm_input,
        "slope_norm": slope,
        "dem_norm": dem_ch,
        "ndvi_norm": ndvi,
    }


class GeoPhysicsDataset(Dataset):
    def __init__(self, base: Dataset, dataset_name: str, fm_backbone: str = "efficientnet"):
        self.base = base
        self.dataset_name = dataset_name
        self.fm_backbone = fm_backbone.lower()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        stream_a = sample["stream_a"]
        stream_b = sample["stream_b"]
        mask = sample["mask"]

        if self.dataset_name == "landslide4sense":
            dem = stream_b[2:3]
            slope = stream_b[1:2]
            ndvi = stream_b[0:1]
            if self.fm_backbone == "efficientnet":
                fm_input = _fm_rgb(stream_a)
            else:
                fm_input = build_l4s_observed_stack(stream_a, stream_b)
        else:
            dem = stream_b[0:1]
            derived = _bijie_physics_and_fm(stream_a, dem, self.fm_backbone)
            slope = derived["slope_norm"]
            ndvi = derived["ndvi_norm"]
            fm_input = derived["fm_input"]

        return {
            "stream_a": stream_a,
            "stream_b": stream_b,
            "dem": dem,
            "mask": mask,
            "fm_input": fm_input,
            "slope_norm": slope,
            "dem_norm": dem,
            "ndvi_norm": ndvi,
        }


class _AugmentWrapper(Dataset):
    def __init__(self, base: GeoPhysicsDataset, transform: AugmentDual2D):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        xa, xb, y = self.transform(item["stream_a"], item["stream_b"], item["mask"])
        item = dict(item)
        item["stream_a"] = xa
        item["stream_b"] = xb
        item["mask"] = y

        if self.base.dataset_name == "landslide4sense":
            item["dem"] = xb[2:3]
            item["dem_norm"] = xb[2:3]
            item["slope_norm"] = xb[1:2]
            item["ndvi_norm"] = xb[0:1]
            if self.base.fm_backbone == "efficientnet":
                item["fm_input"] = _fm_rgb(xa)
            else:
                item["fm_input"] = build_l4s_observed_stack(xa, xb)
        else:
            dem = xb[0:1]
            derived = _bijie_physics_and_fm(xa, dem, self.base.fm_backbone)
            item["dem"] = dem
            item["dem_norm"] = derived["dem_norm"]
            item["slope_norm"] = derived["slope_norm"]
            item["ndvi_norm"] = derived["ndvi_norm"]
            item["fm_input"] = derived["fm_input"]
        return item


def build_l4s_dataloaders(
    dataset_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 8,
    val_ratio: float = 0.1,
    seed: int = 42,
    resize_to: int = 256,
    distributed: bool = False,
    fm_backbone: str = "efficientnet",
):
    train_ids, val_ids = build_l4s_split(dataset_root, val_ratio=val_ratio, seed=seed)
    train_base = L4SDualStreamDataset(dataset_root, ids=train_ids, resize_to=resize_to, transform=None)
    val_base = L4SDualStreamDataset(dataset_root, ids=val_ids, resize_to=resize_to, transform=None)
    train_ds = _AugmentWrapper(
        GeoPhysicsDataset(train_base, "landslide4sense", fm_backbone=fm_backbone),
        AugmentDual2D(p=0.5),
    )
    val_ds = GeoPhysicsDataset(val_base, "landslide4sense", fm_backbone=fm_backbone)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=distributed,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler, val_sampler


def build_bijie_dataloaders(
    dataset_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 8,
    seed: int = 42,
    resize_to: int = 256,
    distributed: bool = False,
    fm_backbone: str = "efficientnet",
):
    train_raw, val_raw, _ = build_bijie_split(dataset_root, seed=seed)
    train_base = BijieTwoComposites(train_raw, resize_to=resize_to, transform=None)
    val_base = BijieTwoComposites(val_raw, resize_to=resize_to, transform=None)
    train_ds = _AugmentWrapper(
        GeoPhysicsDataset(train_base, "bijie", fm_backbone=fm_backbone),
        AugmentDual2D(p=0.5),
    )
    val_ds = GeoPhysicsDataset(val_base, "bijie", fm_backbone=fm_backbone)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=distributed,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler, val_sampler
