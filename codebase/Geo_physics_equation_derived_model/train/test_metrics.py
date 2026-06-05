"""Unit tests for segmentation metrics (run: python -m codebase.Geo_physics_equation_derived_model.train.test_metrics)."""

from __future__ import annotations

import numpy as np
import torch

from .metrics import pixel_metrics_from_logits


def _assert_close(a: float, b: float, tol: float = 1e-4):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_perfect_prediction():
    target = torch.zeros(2, 1, 32, 32)
    target[0, :, 8:12, 8:12] = 1.0
    logits = torch.full_like(target, -6.0)
    logits[0, :, 8:12, 8:12] = 6.0
    m = pixel_metrics_from_logits(logits, target, 0.5)
    _assert_close(m["f1"], 1.0)
    _assert_close(m["iou"], 1.0)
    _assert_close(m["precision"], 1.0)
    _assert_close(m["recall"], 1.0)


def test_matches_per_image_mean_over_batch():
    """Same formula as ablation baseline_models/common/metrics.py."""
    logits = torch.randn(4, 1, 16, 16)
    target = (torch.rand(4, 1, 16, 16) > 0.85).float()
    m = pixel_metrics_from_logits(logits, target, 0.5)

    per_image = []
    for i in range(4):
        per_image.append(pixel_metrics_from_logits(logits[i : i + 1], target[i : i + 1], 0.5))
    exp_f1 = float(np.mean([x["f1"] for x in per_image]))
    _assert_close(m["f1"], exp_f1)


def test_epoch_style_mean_of_batch_metrics():
    """Trainer epoch F1 = mean of per-batch pixel_metrics (dual_stream_gated style)."""
    B, H, W = 8, 32, 32
    logits = torch.full((B, 1, H, W), -4.0)
    target = torch.zeros(B, 1, H, W)
    for i in (6, 7):
        logits[i] = 2.0
        target[i, :, 12:14, 12:14] = 1.0

    batch_f1s = []
    for i in range(B):
        batch_f1s.append(
            pixel_metrics_from_logits(logits[i : i + 1], target[i : i + 1], 0.5)["f1"]
        )
    epoch_f1 = float(np.mean(batch_f1s))
    full_batch_f1 = pixel_metrics_from_logits(logits, target, 0.5)["f1"]
    assert abs(epoch_f1 - full_batch_f1) < 1e-5


def test_f1_iou_consistency():
    torch.manual_seed(1)
    logits = torch.randn(3, 1, 16, 16)
    target = (torch.rand(3, 1, 16, 16) > 0.8).float()
    m = pixel_metrics_from_logits(logits, target, 0.55)
    p, r, f, i = m["precision"], m["recall"], m["f1"], m["iou"]
    f_from_pr = (2 * p * r) / (p + r + 1e-6)
    _assert_close(f, f_from_pr, tol=1e-3)
    i_from_pr = (p * r) / (p + r - p * r + 1e-6)
    _assert_close(i, i_from_pr, tol=1e-3)


def run_all():
    test_perfect_prediction()
    test_matches_per_image_mean_over_batch()
    test_epoch_style_mean_of_batch_metrics()
    test_f1_iou_consistency()
    print("All metric tests passed.")


if __name__ == "__main__":
    run_all()
