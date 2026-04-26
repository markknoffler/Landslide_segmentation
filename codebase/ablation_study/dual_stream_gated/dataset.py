from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_h5_array(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        for key in f.keys():
            data = np.asarray(f[key])
            if data.ndim >= 2:
                return data
    raise ValueError(f"No readable array found in {path}")


def _to_chw(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image[None, ...]
    if image.ndim == 3:
        if image.shape[0] <= 20:
            return image
        return np.transpose(image, (2, 0, 1))
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _minmax_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn + eps)


class Landslide4SenseDualStream(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        rgb_indices: Tuple[int, int, int] = (3, 2, 1),
        nir_index: int = 7,
        slope_index: int = 12,
        dem_index: int = 13,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        split_map = {"train": "TrainData", "valid": "ValidData", "val": "ValidData", "test": "TestData"}
        if split.lower() not in split_map:
            raise ValueError(f"Unknown split: {split}")
        self.split = split.lower()
        self.split_dir = self.data_root / split_map[self.split]
        self.img_dir = self.split_dir / "img"
        self.mask_dir = self.split_dir / "mask"
        self.has_mask = self.mask_dir.exists()

        self.rgb_indices = rgb_indices
        self.nir_index = nir_index
        self.slope_index = slope_index
        self.dem_index = dem_index
        self.samples = sorted(self.img_dir.glob("*.h5"))

    def __len__(self):
        return len(self.samples)

    def _load_mask(self, image_path: Path) -> Optional[np.ndarray]:
        if not self.has_mask:
            return None
        image_id = image_path.stem.replace("image_", "")
        mask_path = self.mask_dir / f"mask_{image_id}.h5"
        if not mask_path.exists():
            return None
        mask = _read_h5_array(mask_path)
        if mask.ndim == 3:
            mask = mask[0] if mask.shape[0] == 1 else mask[..., 0]
        mask = (mask > 0).astype(np.float32)
        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.samples[idx]
        img = _to_chw(_read_h5_array(image_path)).astype(np.float32)
        c, h, w = img.shape

        needed = [*self.rgb_indices, self.nir_index, self.slope_index, self.dem_index]
        if max(needed) >= c:
            raise ValueError(
                f"Insufficient channels in {image_path}. Needed up to index {max(needed)} but got {c} channels."
            )

        red = img[self.rgb_indices[0]]
        nir = img[self.nir_index]
        ndvi = (nir - red) / (nir + red + 1e-6)

        stream_a = img[list(self.rgb_indices)]
        stream_b = np.stack([ndvi, img[self.slope_index], img[self.dem_index]], axis=0)

        stream_a = _minmax_norm(stream_a)
        stream_b = _minmax_norm(stream_b)

        mask = self._load_mask(image_path)
        if mask is None:
            mask = np.zeros((h, w), dtype=np.float32)

        return {
            "stream_a": torch.from_numpy(stream_a).float(),
            "stream_b": torch.from_numpy(stream_b).float(),
            "mask": torch.from_numpy(mask[None, ...]).float(),
            "id": image_path.stem,
        }
