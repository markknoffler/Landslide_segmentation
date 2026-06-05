"""
Segmentation metrics — same definitions as ablation_study/baseline_models/common/metrics.py
and dual_stream_gated (per-image TP/FP/FN/TN, mean over batch; epoch = mean over batches).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ABLATION_COMMON = (
    Path(__file__).resolve().parents[2]
    / "ablation_study"
    / "baseline_models"
)
if str(_ABLATION_COMMON) not in sys.path:
    sys.path.insert(0, str(_ABLATION_COMMON))

from common.metrics import (  # noqa: E402
    image_level_metrics_from_logits,
    pixel_metrics_from_logits,
)

__all__ = ["pixel_metrics_from_logits", "image_level_metrics_from_logits"]
