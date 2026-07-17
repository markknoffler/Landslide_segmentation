#!/usr/bin/env python3
"""Regenerate comparative figures from comparison_metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DATA = ROOT / "assets" / "comparison_metrics.json"

DISPLAY = {
    "linknet": "LinkNet",
    "dep_unet": "DEP-UNet",
    "gmnet": "GMNet",
    "rmau_net": "RMAU-Net",
    "transunet": "TransUNet",
    "emr_hrnet": "EMR-HRNet",
    "shapeformer": "ShapeFormer",
    "dual_stream_unet": "Dual-Stream UNet",
    "dual_stream_gated": "dual_stream_gated",
    "unet": "U-Net",
    "deeplabv3plus": "DeepLabV3+",
    "PS-GPLNet": "PS-GPLNet",
}


def load():
    with open(DATA) as f:
        return json.load(f)


def bar_chart(split: str, out: Path) -> None:
    rows = load()[split]
    models = sorted(rows.keys(), key=lambda k: rows[k]["val_f1"])
    labels = [DISPLAY.get(m, m) for m in models]
    f1 = [rows[m]["val_f1"] for m in models]
    iou = [rows[m]["val_iou"] for m in models]
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, f1, w, label="F1", color="#2a6fbb")
    ax.bar(x + w / 2, iou, w, label="IoU", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"{split}: best-epoch validation F1 / IoU")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def heatmap(split: str, out: Path) -> None:
    rows = load()[split]
    models = sorted(rows.keys(), key=lambda k: rows[k]["val_f1"], reverse=True)
    metrics = ["val_precision", "val_recall", "val_f1", "val_iou"]
    mat = np.array([[rows[m][k] for k in metrics] for m in models])
    fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(models))))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0.35, vmax=1.0)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["Prec", "Rec", "F1", "IoU"])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([DISPLAY.get(m, m) for m in models], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"{split}: metric heatmap (best epoch)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def pr_proxy(split: str, out: Path) -> None:
    rows = load()[split]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, m in sorted(rows.items(), key=lambda x: x[1]["val_f1"]):
        p, r = m["val_precision"], m["val_recall"]
        ax.scatter(r, p, s=50, label=DISPLAY.get(name, name))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_title(f"{split}: precision--recall (best epoch)")
    ax.legend(fontsize=6, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def iou_rank(split: str, out: Path) -> None:
    rows = load()[split]
    models = sorted(rows.keys(), key=lambda k: rows[k]["val_iou"])
    ious = [rows[m]["val_iou"] for m in models]
    colors = ["#d62728" if m == "PS-GPLNet" else "#4c72b0" for m in models]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([DISPLAY.get(m, m) for m in models], ious, color=colors)
    ax.set_xlabel("IoU")
    ax.set_xlim(0.3, 1.0)
    ax.set_title(f"{split}: IoU ranking")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def delta_vs_deeplab(split: str, out: Path) -> None:
    rows = load()[split]
    if "deeplabv3plus" not in rows or "PS-GPLNet" not in rows:
        return
    base = rows["deeplabv3plus"]
    ours = rows["PS-GPLNet"]
    keys = ["val_precision", "val_recall", "val_f1", "val_iou"]
    labels = ["Prec", "Rec", "F1", "IoU"]
    delta = [ours[k] - base[k] for k in keys]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta]
    ax.bar(labels, delta, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("PS-GPLNet $-$ DeepLabV3+")
    ax.set_title(f"{split}: gain over DeepLabV3+")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for split in ("landslide4sense", "bijie"):
        tag = "l4s" if split == "landslide4sense" else "bijie"
        bar_chart(split, FIG / f"fig_{tag}_bars.png")
        heatmap(split, FIG / f"fig_{tag}_heatmap.png")
        pr_proxy(split, FIG / f"fig_{tag}_pr.png")
        iou_rank(split, FIG / f"fig_{tag}_iou_rank.png")
        delta_vs_deeplab(split, FIG / f"fig_{tag}_delta_deeplab.png")
    print("Wrote figures to", FIG)


if __name__ == "__main__":
    main()
