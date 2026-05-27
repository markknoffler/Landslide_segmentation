import sys
from pathlib import Path

_ABLATION_COMMON = (
    Path(__file__).resolve().parents[2]
    / "ablation_study"
    / "baseline_models"
)
if str(_ABLATION_COMMON) not in sys.path:
    sys.path.insert(0, str(_ABLATION_COMMON))

from common.metrics import image_level_metrics_from_logits, pixel_metrics_from_logits  # noqa: F401

__all__ = ["pixel_metrics_from_logits", "image_level_metrics_from_logits"]
