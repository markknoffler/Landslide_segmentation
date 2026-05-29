from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .terrain import minmax_per_channel, sobel_slope_norm

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


def _prithvi_from_rgb(rgb_chw: np.ndarray) -> torch.Tensor:
    r, g, b = rgb_chw[0], rgb_chw[1], rgb_chw[2]
    nir = (r + g) / 2.0
    swir1 = (r + b) / 2.0
    swir2 = g
    stack = np.stack([b, g, r, nir, swir1, swir2], axis=0).astype(np.float32)
    return minmax_per_channel(torch.from_numpy(stack).float())


def _prithvi_l4s(stream_a: torch.Tensor, stream_b: torch.Tensor) -> torch.Tensor:
    prithvi = _prithvi_from_rgb(stream_a.numpy())
    prithvi[3] = stream_b[0]
    prithvi[4] = stream_b[1]
    prithvi[5] = stream_b[2] if stream_b.shape[0] > 2 else stream_b[1]
    return minmax_per_channel(prithvi)


class GeoPhysicsDataset(Dataset):
    def __init__(self, base: Dataset, dataset_name: str):
        self.base = base
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        stream_a = sample["stream_a"]
        stream_b = sample["stream_b"]
        mask = sample["mask"]

        if self.dataset_name == "landslide4sense":
            ndvi = stream_b[0:1]
            slope = stream_b[1:2]
            dem = stream_b[2:3]
            prithvi = _prithvi_l4s(stream_a, stream_b)
        else:
            dem = stream_b[0:1]
            slope = torch.from_numpy(sobel_slope_norm(dem[0].numpy())).unsqueeze(0)
            ndvi = torch.zeros_like(dem)
            prithvi = _prithvi_from_rgb(stream_a.numpy())

        return {
            "stream_a": stream_a,
            "stream_b": stream_b,
            "dem": dem,
            "mask": mask,
            "prithvi_input": prithvi,
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
            item["prithvi_input"] = _prithvi_l4s(xa, xb)
        else:
            item["dem"] = xb[0:1]
            item["dem_norm"] = xb[0:1]
            item["slope_norm"] = torch.from_numpy(sobel_slope_norm(xb[0].numpy())).unsqueeze(0)
            item["ndvi_norm"] = torch.zeros_like(item["dem_norm"])
            item["prithvi_input"] = _prithvi_from_rgb(xa.numpy())
        return item


def build_l4s_dataloaders(
    dataset_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 8,
    val_ratio: float = 0.1,
    seed: int = 42,
    resize_to: int = 256,
    distributed: bool = False,
):
    train_ids, val_ids = build_l4s_split(dataset_root, val_ratio=val_ratio, seed=seed)
    train_base = L4SDualStreamDataset(dataset_root, ids=train_ids, resize_to=resize_to, transform=None)
    val_base = L4SDualStreamDataset(dataset_root, ids=val_ids, resize_to=resize_to, transform=None)
    train_ds = _AugmentWrapper(GeoPhysicsDataset(train_base, "landslide4sense"), AugmentDual2D(p=0.5))
    val_ds = GeoPhysicsDataset(val_base, "landslide4sense")
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
):
    train_raw, val_raw, _ = build_bijie_split(dataset_root, seed=seed)
    train_base = BijieTwoComposites(train_raw, resize_to=resize_to, transform=None)
    val_base = BijieTwoComposites(val_raw, resize_to=resize_to, transform=None)
    train_ds = _AugmentWrapper(GeoPhysicsDataset(train_base, "bijie"), AugmentDual2D(p=0.5))
    val_ds = GeoPhysicsDataset(val_base, "bijie")
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
