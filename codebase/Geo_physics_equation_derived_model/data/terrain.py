"""Terrain and spectral indices computed only from observed rasters (no placeholder zeros)."""

from __future__ import annotations

import numpy as np
import torch


def sobel_slope_norm(dem: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalized slope magnitude in [0, 1] from measured DEM."""
    if dem.ndim == 3:
        dem = dem[0]
    dem = dem.astype(np.float32)
    gy, gx = np.gradient(dem)
    slope = np.sqrt(gx * gx + gy * gy)
    mn, mx = float(slope.min()), float(slope.max())
    if mx > mn:
        slope = (slope - mn) / (mx - mn + eps)
    return np.clip(slope, 0.0, 1.0).astype(np.float32)


def vegetation_index_from_rgb(rgb_chw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Green-red vegetation index from measured RGB only (no NIR band in Bijie).

    Uses G and R channels: (G - R) / (G + R + eps), clipped to [0, 1].
    This is a standard surrogate when multispectral NIR is unavailable.
    """
    if rgb_chw.shape[0] < 3:
        raise ValueError(f"RGB stack must have 3 channels, got shape {rgb_chw.shape}")
    g = rgb_chw[1].astype(np.float32)
    r = rgb_chw[2].astype(np.float32)
    vi = (g - r) / (g + r + eps)
    vi = np.clip(vi, 0.0, 1.0)
    mn, mx = float(vi.min()), float(vi.max())
    if mx > mn:
        vi = (vi - mn) / (mx - mn + eps)
    return np.clip(vi, 0.0, 1.0).astype(np.float32)


def build_bijie_observed_stack(rgb_chw: np.ndarray, dem_chw: np.ndarray) -> torch.Tensor:
    """
    Six-channel stack for Bijie using only measured RGB + DEM and derived terrain indices.

    Channels (all from real PNG rasters on disk):
      0 Blue, 1 Green, 2 Red, 3 DEM, 4 slope(DEM), 5 vegetation_index(RGB)
    """
    if rgb_chw.shape[0] != 3:
        raise ValueError(f"Expected RGB with 3 channels, got {rgb_chw.shape}")
    if dem_chw.shape[0] != 1:
        raise ValueError(f"Expected DEM with 1 channel, got {dem_chw.shape}")

    dem_np = dem_chw[0].astype(np.float32) if dem_chw.ndim == 3 else dem_chw.astype(np.float32)
    slope = sobel_slope_norm(dem_np)
    veg = vegetation_index_from_rgb(rgb_chw)

    stack = np.stack(
        [
            rgb_chw[0].astype(np.float32),
            rgb_chw[1].astype(np.float32),
            rgb_chw[2].astype(np.float32),
            dem_np,
            slope,
            veg,
        ],
        axis=0,
    )
    return minmax_per_channel(torch.from_numpy(stack).float())


def build_l4s_observed_stack(stream_a: np.ndarray, stream_b: np.ndarray) -> torch.Tensor:
    """
    Six-channel stack for Landslide4Sense from measured stream bands.

    stream_a: RGB (3, H, W)
    stream_b: NDVI, slope, DEM (3, H, W) per L4SDualStreamDataset layout
    """
    if stream_a.shape[0] < 3 or stream_b.shape[0] < 3:
        raise ValueError(f"Expected 3+ channels in each stream, got {stream_a.shape}, {stream_b.shape}")

    b, g, r = stream_a[0], stream_a[1], stream_a[2]
    ndvi = stream_b[0]
    slope = stream_b[1]
    dem = stream_b[2]

    stack = torch.stack([b, g, r, ndvi, slope, dem], dim=0).float()
    return minmax_per_channel(stack)


def minmax_per_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    out = x.clone()
    for c in range(out.shape[0]):
        mn = float(out[c].min())
        mx = float(out[c].max())
        if mx > mn:
            out[c] = (out[c] - mn) / (mx - mn + eps)
        out[c] = torch.clamp(out[c], 0.0, 1.0)
    return out
