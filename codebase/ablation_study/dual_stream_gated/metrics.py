from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def pixel_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    gt = target.detach().cpu().numpy().reshape(-1).astype(np.uint8)
    pred = (probs >= threshold).astype(np.uint8)

    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def image_level_detection_metrics(image_probs: List[float], image_labels: List[int]) -> Dict[str, float]:
    y_score = np.asarray(image_probs, dtype=np.float32)
    y_true = np.asarray(image_labels, dtype=np.uint8)

    if len(np.unique(y_true)) < 2:
        return {"auroc": 0.0, "auprc": 0.0, "best_f1": 0.0, "best_threshold": 0.5}

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = (2 * precision * recall) / np.clip(precision + recall, 1e-8, None)
    best_idx = int(np.nanargmax(f1_scores))
    best_f1 = float(f1_scores[best_idx])

    if best_idx >= len(thresholds):
        best_threshold = 1.0
    else:
        best_threshold = float(thresholds[best_idx])

    return {"auroc": auroc, "auprc": auprc, "best_f1": best_f1, "best_threshold": best_threshold}
