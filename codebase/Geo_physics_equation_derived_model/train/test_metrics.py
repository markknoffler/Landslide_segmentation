"""Unit tests for segmentation metrics (run: python -m codebase.Geo_physics_equation_derived_model.train.test_metrics)."""

from __future__ import annotations

import torch

from .metrics import MetricAccumulator, pixel_metrics_from_logits


def _manual_counts(logits, target, thr=0.5):
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).long()
    tgt = (target > 0).long()
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if tgt.dim() == 3:
        tgt = tgt.unsqueeze(1)
    tp = int((pred * tgt).sum())
    fp = int((pred * (1 - tgt)).sum())
    fn = int(((1 - pred) * tgt).sum())
    tn = int(((1 - pred) * (1 - tgt)).sum())
    return tp, fp, fn, tn


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


def test_accumulator_matches_manual_micro():
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 48, 48)
    target = (torch.rand(4, 1, 48, 48) > 0.92).float()

    acc = MetricAccumulator()
    for i in range(logits.shape[0]):
        acc.update(logits[i : i + 1], target[i : i + 1], threshold=0.6)
    got = acc.compute()

    tp, fp, fn, tn = _manual_counts(logits, target, 0.6)
    eps = 1e-6
    exp_f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    exp_iou = (tp + eps) / (tp + fp + fn + eps)
    _assert_close(got["f1"], float(exp_f1))
    _assert_close(got["iou"], float(exp_iou))


def test_accumulator_beats_mean_of_batch_metrics():
    """Mean of per-batch micro F1 must not be used as epoch F1."""
    B, H, W = 8, 32, 32
    logits = torch.full((B, 1, H, W), -4.0)
    target = torch.zeros(B, 1, H, W)
    for i in (6, 7):
        logits[i] = 2.0
        target[i, :, 12:14, 12:14] = 1.0

    acc = MetricAccumulator()
    batch_f1s = []
    for i in range(B):
        acc.update(logits[i : i + 1], target[i : i + 1], 0.5)
        batch_f1s.append(pixel_metrics_from_logits(logits[i : i + 1], target[i : i + 1], 0.5)["f1"])

    epoch = acc.compute()
    wrong_epoch_f1 = sum(batch_f1s) / len(batch_f1s)
    assert abs(epoch["f1"] - wrong_epoch_f1) > 0.01, "test should expose batch-mean vs global gap"


def test_landslide_subset_only_counts_positive_gt():
    logits = torch.full((2, 1, 4, 4), 2.0)
    target = torch.zeros(2, 1, 4, 4)
    target[1, :, 0, 0] = 1.0  # only second tile has landslide GT

    m = pixel_metrics_from_logits(logits, target, 0.5)
    assert m["n_with_gt"] == 1
    assert m["n_total"] == 2
    _assert_close(m["landslide_recall"], 1.0)
    assert m["landslide_precision"] < 0.25


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
    test_accumulator_matches_manual_micro()
    test_accumulator_beats_mean_of_batch_metrics()
    test_landslide_subset_only_counts_positive_gt()
    test_f1_iou_consistency()
    print("All metric tests passed.")


if __name__ == "__main__":
    run_all()
