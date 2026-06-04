"""Segmentation metrics with correct micro aggregation and empty-mask handling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
)


def _to_float32(t: torch.Tensor) -> torch.Tensor:
    if t.is_floating_point() and t.dtype != torch.float32:
        return t.float()
    return t


def _confusion_counts(logits: torch.Tensor, target: torch.Tensor, threshold: float):
    probs = torch.sigmoid(_to_float32(logits))
    pred = (probs >= threshold).to(torch.int64)
    tgt = (_to_float32(target) > 0).to(torch.int64)
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if tgt.dim() == 3:
        tgt = tgt.unsqueeze(1)

    dims = (1, 2, 3)
    tp = (pred * tgt).sum(dim=dims).float()
    fp = (pred * (1 - tgt)).sum(dim=dims).float()
    fn = ((1 - pred) * tgt).sum(dim=dims).float()
    tn = ((1 - pred) * (1 - tgt)).sum(dim=dims).float()
    gt_pos = tgt.sum(dim=dims).float()
    return tp, fp, fn, tn, gt_pos


def _rates_from_counts(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    eps = 1e-6
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


class MetricAccumulator:
    """Accumulate TP/FP/FN/TN across batches (and distributed ranks) for exact epoch metrics."""

    def __init__(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.ls_tp = 0.0
        self.ls_fp = 0.0
        self.ls_fn = 0.0
        self.ls_tn = 0.0
        self.n_with_gt = 0
        self.n_total = 0

    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
        min_gt_positive_pixels: int = 1,
    ) -> None:
        tp, fp, fn, tn, gt_pos = _confusion_counts(logits, target, threshold)
        self.tp += float(tp.sum())
        self.fp += float(fp.sum())
        self.fn += float(fn.sum())
        self.tn += float(tn.sum())
        self.n_total += int(tp.shape[0])

        has_gt = gt_pos >= min_gt_positive_pixels
        self.n_with_gt += int(has_gt.sum().item())
        if bool(has_gt.any()):
            self.ls_tp += float(tp[has_gt].sum())
            self.ls_fp += float(fp[has_gt].sum())
            self.ls_fn += float(fn[has_gt].sum())
            self.ls_tn += float(tn[has_gt].sum())

    def reduce_distributed(self, device: torch.device) -> None:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return
        vec = torch.tensor(
            [
                self.tp,
                self.fp,
                self.fn,
                self.tn,
                self.ls_tp,
                self.ls_fp,
                self.ls_fn,
                self.ls_tn,
                float(self.n_with_gt),
                float(self.n_total),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        (
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            self.ls_tp,
            self.ls_fp,
            self.ls_fn,
            self.ls_tn,
            self.n_with_gt,
            self.n_total,
        ) = (float(v) for v in vec.tolist())

    def compute(self) -> dict[str, float]:
        micro = _rates_from_counts(self.tp, self.fp, self.fn, self.tn)
        if self.n_with_gt > 0:
            landslide = _rates_from_counts(self.ls_tp, self.ls_fp, self.ls_fn, self.ls_tn)
        else:
            landslide = {k: 0.0 for k in micro}
        return {
            **micro,
            "landslide_precision": landslide["precision"],
            "landslide_recall": landslide["recall"],
            "landslide_f1": landslide["f1"],
            "landslide_iou": landslide["iou"],
            "n_with_gt": int(self.n_with_gt),
            "n_total": int(self.n_total),
        }


def pixel_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    min_gt_positive_pixels: int = 1,
):
    """
    Pixel metrics for binary landslide segmentation.

    Primary numbers use **micro** aggregation (sum TP/FP/FN/TN over the batch).
    Per-image means inflate accuracy/recall when many tiles have empty masks
    (recall becomes 1.0 whenever fn=0, even with massive false positives).

    Also returns landslide-only rates computed only on images with GT positives.
    """
    tp, fp, fn, tn, gt_pos = _confusion_counts(logits, target, threshold)
    micro = _rates_from_counts(
        float(tp.sum()),
        float(fp.sum()),
        float(fn.sum()),
        float(tn.sum()),
    )

    has_gt = gt_pos >= min_gt_positive_pixels
    if bool(has_gt.any()):
        ls_tp = tp[has_gt].sum()
        ls_fp = fp[has_gt].sum()
        ls_fn = fn[has_gt].sum()
        ls_tn = tn[has_gt].sum()
        landslide = _rates_from_counts(
            float(ls_tp),
            float(ls_fp),
            float(ls_fn),
            float(ls_tn),
        )
    else:
        landslide = {k: 0.0 for k in micro}

    return {
        **micro,
        "landslide_precision": landslide["precision"],
        "landslide_recall": landslide["recall"],
        "landslide_f1": landslide["f1"],
        "landslide_iou": landslide["iou"],
        "n_with_gt": int(has_gt.sum().item()),
        "n_total": int(tp.shape[0]),
    }


def best_pixel_threshold(
    logits: torch.Tensor,
    target: torch.Tensor,
    thresholds: tuple[float, ...] = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7),
) -> tuple[float, dict[str, float]]:
    """Pick threshold that maximizes micro F1 on a batch or epoch aggregate."""
    best_thr = thresholds[0]
    best_metrics = pixel_metrics_from_logits(logits, target, threshold=best_thr)
    best_f1 = best_metrics["f1"]
    for thr in thresholds[1:]:
        m = pixel_metrics_from_logits(logits, target, threshold=thr)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = thr
            best_metrics = m
    best_metrics["best_threshold"] = best_thr
    return best_thr, best_metrics


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


__all__ = [
    "MetricAccumulator",
    "pixel_metrics_from_logits",
    "image_level_metrics_from_logits",
    "best_pixel_threshold",
]
