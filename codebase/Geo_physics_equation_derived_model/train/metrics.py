"""Metrics wrappers (cast FSDP bf16 logits to float32 for numpy/sklearn paths)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ABLATION_COMMON = (
    Path(__file__).resolve().parents[2]
    / "ablation_study"
    / "baseline_models"
)
if str(_ABLATION_COMMON) not in sys.path:
    sys.path.insert(0, str(_ABLATION_COMMON))

from common.metrics import (  # noqa: E402
    image_level_metrics_from_logits as _image_level_metrics_from_logits,
    pixel_metrics_from_logits as _pixel_metrics_from_logits,
)


def _to_float32(t: torch.Tensor) -> torch.Tensor:
    if t.is_floating_point() and t.dtype != torch.float32:
        return t.float()
    return t


def pixel_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    return _pixel_metrics_from_logits(_to_float32(logits), _to_float32(target), threshold=threshold)


def image_level_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    prob_thr_for_instances: float = 0.5,
    min_area: int = 20,
):
    return _image_level_metrics_from_logits(
        _to_float32(logits),
        _to_float32(target),
        prob_thr_for_instances=prob_thr_for_instances,
        min_area=min_area,
    )


__all__ = ["pixel_metrics_from_logits", "image_level_metrics_from_logits"]
