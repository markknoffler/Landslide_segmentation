import numpy as np
import torch


def sobel_slope_norm(dem: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute normalized slope magnitude in [0, 1] from DEM."""
    if dem.ndim == 3:
        dem = dem[0]
    dem = dem.astype(np.float32)
    gy, gx = np.gradient(dem)
    slope = np.sqrt(gx * gx + gy * gy)
    mn, mx = float(slope.min()), float(slope.max())
    if mx > mn:
        slope = (slope - mn) / (mx - mn + eps)
    return np.clip(slope, 0.0, 1.0).astype(np.float32)


def minmax_per_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    out = x.clone()
    for c in range(out.shape[0]):
        mn = float(out[c].min())
        mx = float(out[c].max())
        if mx > mn:
            out[c] = (out[c] - mn) / (mx - mn + eps)
        out[c] = torch.clamp(out[c], 0.0, 1.0)
    return out
