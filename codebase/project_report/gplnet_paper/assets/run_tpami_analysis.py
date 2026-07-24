#!/usr/bin/env python3
"""TPAMI analysis + figure generation (GPU 0). Does not upgrade packages."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3] / "Geo_physics_equation_derived_model" / "from_first_steps"
PAPER = Path(__file__).resolve().parents[1]
FIG = PAPER / "figures" / "tpami"
FIG.mkdir(parents=True, exist_ok=True)


def positive_scale(p: torch.Tensor, max_log: float = 8.0) -> torch.Tensor:
    return torch.exp(torch.clamp(p, -8.0, max_log))


def best_from_csv(p: Path) -> dict:
    rows = list(csv.DictReader(open(p)))
    best = max(rows, key=lambda r: float(r["val_f1"]))
    return {
        k: float(best[k]) if k != "epoch" else int(float(best[k]))
        for k in ["epoch", "val_f1", "val_iou", "val_precision", "val_recall", "val_acc"]
    }


def plot_curves(csv_path: Path, out: Path, title: str) -> None:
    rows = list(csv.DictReader(open(csv_path)))
    ep = [int(float(r["epoch"])) for r in rows]

    def col(name: str):
        return [float(r[name]) if r[name] != "nan" else np.nan for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].plot(ep, col("train_loss"), label="train")
    axes[0].plot(ep, col("val_loss"), label="val")
    axes[0].set_title("Loss")
    axes[0].legend(fontsize=7)
    axes[1].plot(ep, col("val_f1"), color="#2a6fbb")
    axes[1].set_title("Val F1")
    axes[2].plot(ep, col("val_iou"), color="#c44e52")
    axes[2].set_title("Val IoU")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlabel("epoch")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close()


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)
    metrics = json.loads((PAPER / "assets" / "comparison_metrics.json").read_text())

    ckpt_path = ROOT / "prev_legacy_results" / "outputs_bijie" / "checkpoint" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    info: dict = {"ckpt_type": str(type(ckpt)), "path": str(ckpt_path)}
    if isinstance(ckpt, dict):
        info["keys"] = list(ckpt.keys())[:40]
        for k in ("epoch", "best_f1", "best_iou", "val_f1", "val_iou"):
            if k in ckpt:
                info[k] = ckpt[k]
        sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt.get("model_state_dict")
        if sd is None and any(torch.is_tensor(v) for v in list(ckpt.values())[:5]):
            # maybe raw state dict
            if all(isinstance(k, str) for k in list(ckpt.keys())[:5]):
                maybe = True
                for v in list(ckpt.values())[:5]:
                    if not (torch.is_tensor(v) or isinstance(v, dict)):
                        maybe = False
                if maybe and "epoch" not in ckpt:
                    sd = ckpt
        if isinstance(sd, dict):
            info["n_params_in_sd"] = int(sum(p.numel() for p in sd.values() if torch.is_tensor(p)))
            info["n_tensors"] = len(sd)

    results = {
        "bijie_final": best_from_csv(ROOT / "outputs_final_bijie" / "results" / "epoch_metrics.csv"),
        "l4s_final": best_from_csv(ROOT / "outputs_final_l4s" / "results" / "epoch_metrics.csv"),
        "checkpoint": info,
        "device": str(device),
    }

    # compactness
    ws = torch.linspace(-20, 20, 1001, device=device)
    ps = positive_scale(ws)
    results["material_bounds"] = {"min": float(ps.min()), "max": float(ps.max())}

    # MPEF softmax dominates random simplex
    torch.manual_seed(0)
    FE = torch.randn(5000, 2, device=device)
    w = torch.softmax(FE, dim=-1)
    H = -(w * torch.log(w.clamp_min(1e-8))).sum(-1)
    obj = (w * FE).sum(-1) + H
    u = torch.rand(5000, 2, device=device)
    u = u / u.sum(-1, keepdim=True)
    H2 = -(u * torch.log(u.clamp_min(1e-8))).sum(-1)
    obj2 = (u * FE).sum(-1) + H2
    results["mpef_softmax_dominates_random"] = float((obj >= obj2 - 1e-5).float().mean())

    # local Lip FS gate
    N = 4096
    alpha = 0.05 + 1.1 * torch.rand(N, 1, 1, 1, device=device)
    h = 0.2 + 2.5 * torch.rand(N, 1, 1, 1, device=device)
    m = torch.rand(N, 1, 1, 1, device=device)
    c = positive_scale(torch.zeros(1, device=device))
    phi = positive_scale(torch.zeros(1, device=device))
    g = positive_scale(torch.zeros(1, device=device))
    wm = positive_scale(torch.zeros(1, device=device))

    def gate(a, hh, mm):
        fs = (c + phi * hh * torch.cos(a) ** 2) / (
            g * hh * torch.sin(a) * torch.cos(a) + wm * mm + 1e-6
        )
        return torch.sigmoid(1.0 - fs)

    eps = 1e-3
    da = eps * torch.randn_like(alpha)
    r = ((gate(alpha + da, h, m) - gate(alpha, h, m)).abs() / (da.abs() + 1e-12)).flatten()
    results["fs_gate_local_lip"] = {
        "median": float(r.median()),
        "p95": float(r.quantile(0.95)),
        "max": float(r.max()),
        "mean": float(r.mean()),
    }

    results["deltas"] = {}
    for split, key in [("bijie", "bijie"), ("l4s", "landslide4sense")]:
        ours = metrics[key]["PS-GPLNet"]
        dl = metrics[key]["deeplabv3plus"]
        dg = metrics[key]["dual_stream_gated"]
        results["deltas"][split] = {
            "vs_deeplab_f1": ours["val_f1"] - dl["val_f1"],
            "vs_deeplab_iou": ours["val_iou"] - dl["val_iou"],
            "vs_dualgated_f1": ours["val_f1"] - dg["val_f1"],
            "vs_dualgated_iou": ours["val_iou"] - dg["val_iou"],
        }

    out_json = PAPER / "assets" / "tpami_analysis_log.json"
    out_json.write_text(json.dumps(results, indent=2))
    (ROOT / "tpami_assets" / "analysis_log.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "tpami_assets" / "analysis_log.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    for src in (ROOT / "tpami_assets" / "figures").glob("*.png"):
        shutil.copy(src, FIG / src.name)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for ax, split, title in zip(axes, ["bijie", "l4s"], ["Bijie", "Landslide4Sense"]):
        d = results["deltas"][split]
        labels = ["ΔF1 vs\nDeepLab", "ΔIoU vs\nDeepLab", "ΔF1 vs\ndual-gated", "ΔIoU vs\ndual-gated"]
        vals = [d["vs_deeplab_f1"], d["vs_deeplab_iou"], d["vs_dualgated_f1"], d["vs_dualgated_iou"]]
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
        ax.bar(labels, vals, color=colors)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel("PS-GPLNet − baseline")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_tpami_deltas.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in zip(axes, ["bijie", "landslide4sense"], ["Bijie IoU", "L4S IoU"]):
        items = sorted(metrics[key].items(), key=lambda kv: kv[1]["val_iou"])
        names = [k for k, _ in items]
        vals = [v["val_iou"] for _, v in items]
        cols = ["#d62728" if n == "PS-GPLNet" else "#4c72b0" for n in names]
        ax.barh(names, vals, color=cols)
        ax.set_xlim(0.3, 1.0)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_tpami_iou_rank.png", dpi=180)
    plt.close()

    plot_curves(
        ROOT / "outputs_final_bijie" / "results" / "epoch_metrics.csv",
        FIG / "fig_tpami_bijie_curves.png",
        "Bijie final run",
    )
    plot_curves(
        ROOT / "outputs_final_l4s" / "results" / "epoch_metrics.csv",
        FIG / "fig_tpami_l4s_curves.png",
        "Landslide4Sense final run",
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    stages = ["Inlet\nFS cell", "CMB\nhybrid", "MAO/TTEB", "PGDI\ndecoder", "MPEF\nFE routing"]
    scores = [1.0, 0.92, 0.88, 0.95, 1.0]
    ax.plot(stages, scores, "o-", color="#2a6fbb", lw=2)
    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel("Shared FS currency (schematic)")
    ax.set_title("Latent Continuum Equivalency across stages")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_tpami_lce_schematic.png", dpi=180)
    plt.close()

    # Lip histogram already in theory figs; regenerate into tpami folder explicitly
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(np.clip(r.detach().cpu().numpy(), 0, float(np.percentile(r.detach().cpu().numpy(), 99))), bins=40, color="#2a6fbb", alpha=0.85)
    ax.axvline(float(r.median()), color="#c44e52", ls="--", label=f"median={float(r.median()):.2f}")
    ax.set_xlabel(r"local ratio $\|\Delta gate\|/\|\Delta\alpha\|$")
    ax.set_ylabel("count")
    ax.set_title("Empirical local sensitivity of FS gate (GPU-0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_fs_local_lipschitz_hist.png", dpi=180)
    plt.close()

    print("wrote figures to", FIG)
    print("n figures", len(list(FIG.glob('*.png'))))
    if device.type == "cuda":
        print("cuda allocated MiB", torch.cuda.memory_allocated() / 1e6)


if __name__ == "__main__":
    main()
