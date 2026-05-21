from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

EPSILON = 1e-6


def _minmax_per_channel(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    for c in range(out.shape[0]):
        mn = float(out[c].min())
        mx = float(out[c].max())
        if mx > mn:
            out[c] = (out[c] - mn) / (mx - mn + EPSILON)
        out[c] = torch.clamp(out[c], 0.0, 1.0)
    return out


class ScoutAugment2D:
    """Consistent random flips and Gaussian noise on dem + rgb + mask."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, dem: torch.Tensor, rgb: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            dem = TF.hflip(dem)
            rgb = TF.hflip(rgb)
            mask = TF.hflip(mask)
        if random.random() < self.p:
            dem = TF.vflip(dem)
            rgb = TF.vflip(rgb)
            mask = TF.vflip(mask)
        if random.random() < self.p:
            dem = dem + torch.randn_like(dem) * 0.05
            rgb = rgb + torch.randn_like(rgb) * 0.05
        return dem, rgb, mask


class ScoutBijieDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        resize_to: int = 256,
        transform: Optional[ScoutAugment2D] = None,
    ):
        super().__init__()
        self.ds = base_dataset
        self.resize_to = resize_to
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.ds[idx]
        img = sample["image"]
        dem = sample["dem"]
        mask = sample["mask"]

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]

        if dem.ndim == 3:
            dem = dem[:, :, 0]

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        rgb = np.transpose(img, (2, 0, 1)).astype(np.float32)
        dem = dem.astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        rgb_hwc = np.transpose(rgb, (1, 2, 0))
        rgb_hwc = cv2.resize(
            rgb_hwc, (self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR
        )
        rgb = np.transpose(rgb_hwc, (2, 0, 1))

        dem = cv2.resize(
            dem, (self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask, (self.resize_to, self.resize_to), interpolation=cv2.INTER_NEAREST
        )

        dem = _minmax_per_channel(torch.from_numpy(dem[None, ...]).float())
        rgb = _minmax_per_channel(torch.from_numpy(rgb).float())
        mask = torch.from_numpy(mask[None, ...]).float()

        if self.transform is not None:
            dem, rgb, mask = self.transform(dem, rgb, mask)

        return {"dem": dem, "rgb": rgb, "mask": mask}


class ScoutL4SDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        ids: Sequence[int],
        resize_to: int = 256,
        transform: Optional[ScoutAugment2D] = None,
    ):
        super().__init__()
        self.root = Path(dataset_root)
        self.resize_to = resize_to
        self.transform = transform

        img_dir = self.root / "TrainData" / "img"
        mask_dir = self.root / "TrainData" / "mask"
        files = sorted(img_dir.glob("*.h5"))
        self.img_paths = [files[i] for i in ids]
        self.mask_paths = [
            mask_dir / p.name.replace("image_", "mask_") for p in self.img_paths
        ]

    def __len__(self) -> int:
        return len(self.img_paths)

    @staticmethod
    def _read_h5(path: Path) -> np.ndarray:
        with h5py.File(str(path), "r") as f:
            for key in ("img", "image", "data", "arr"):
                if key in f:
                    return np.asarray(f[key], dtype=np.float32)
            for k in f.keys():
                arr = np.asarray(f[k], dtype=np.floa/home/samreedh/Desktop/samreedh/deeplearning_projects/Landslide_segmentation/codebase/PC_SCOUT_SNIPER_Model/scout_modelt32)
                if arr.ndim >= 2:
                    return arr
        raise ValueError(f"No valid array found in {path}")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = self._read_h5(self.img_paths[idx])
        mask = self._read_h5(self.mask_paths[idx])

        if image.ndim == 3 and image.shape[-1] in (1, 3, 4, 7, 14, 20):
            image = image.transpose(2, 0, 1)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        elif mask.ndim == 3:
            mask = mask[..., 0]

        B2 = image[1]
        B3 = image[2]
        B4 = image[3]
        B13 = image[12]
        dem = B13
        rgb = np.stack([B4, B3, B2], axis=0)

        mask = (mask > 0).astype(np.float32)

        rgb_hwc = np.transpose(rgb, (1, 2, 0))
        rgb_hwc = cv2.resize(
            rgb_hwc, (self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR
        )
        rgb = np.transpose(rgb_hwc, (2, 0, 1))

        dem = cv2.resize(
            dem, (self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask, (self.resize_to, self.resize_to), interpolation=cv2.INTER_NEAREST
        )

        dem = _minmax_per_channel(torch.from_numpy(dem[None, ...]).float())
        rgb = _minmax_per_channel(torch.from_numpy(rgb).float())
        mask = torch.from_numpy(mask[None, ...]).float()

        if self.transform is not None:
            dem, rgb, mask = self.transform(dem, rgb, mask)

        return {"dem": dem, "rgb": rgb, "mask": mask}
