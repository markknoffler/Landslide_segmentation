#!/usr/bin/env python3
"""
TPAMI-grade inference studies for PS-GPLNet (GeoPhysicsLandslideNet).

Runs on the real Bijie dual-stream loaders and the real model forward API:
  main, aux2, aux3, reg = model(stream_a, stream_b)

Studies:
  1) Latent manifold t-SNE (hook on MPEF fuse_main / bottleneck)
  2) Boundary F-score + Hausdorff on validation predictions
  3) RGB/DEM Gaussian noise robustness decay

Also optionally evaluates RGB baselines (U-Net, DeepLabV3+) that share the
same Bijie split seed for comparative decay / boundary tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import ConcatDataset, DataLoader, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bijie_dataset import BijieRawDataset, BijieTwoComposites  # noqa: E402
from device_utils import resolve_device  # noqa: E402
from model import GeoPhysicsLandslideNet  # noqa: E402
from tpami_eval_utils import compute_boundary_fscore, compute_hausdorff_distance  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TPAMI PS-GPLNet evaluation studies")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="PS-GPLNet best.pt / epoch_XXXX.pt")
    p.add_argument("--prithvi_snapshot", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./tpami_study_outputs")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--auto_gpu", action="store_true", default=True)
    p.add_argument("--no-auto_gpu", dest="auto_gpu", action="store_false")
    p.add_argument("--min_free_gb", type=float, default=2.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric_threshold", type=float, default=0.5)
    p.add_argument("--backbone", type=str, default="tf_efficientnet_b4")
    p.add_argument("--fusion_channels", type=int, default=64)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--compact", action="store_true")
    p.add_argument("--max_val_batches", type=int, default=0, help="0 = full val set")
    p.add_argument("--tsne_max_samples", type=int, default=400)
    p.add_argument("--noise_levels", type=str, default="0,0.05,0.1,0.2,0.3")
    p.add_argument(
        "--baseline_unet_ckpt",
        type=str,
        default="",
        help="Optional U-Net best.pt for comparative plots",
    )
    p.add_argument(
        "--baseline_deeplab_ckpt",
        type=str,
        default="",
        help="Optional DeepLabV3+ best.pt for comparative plots",
    )
    p.add_argument("--skip_tsne", action="store_true")
    p.add_argument("--skip_boundary", action="store_true")
    p.add_argument("--skip_robustness", action="store_true")
    return p.parse_args()


def build_val_loader(dataset_root: Path, seed: int, batch_size: int, num_workers: int) -> DataLoader:
    landslide_raw = BijieRawDataset(dataset_root / "landslide", phase="landslide")
    nonlandslide_raw = BijieRawDataset(dataset_root / "non-landslide", phase="non-landslide")
    generator = torch.Generator().manual_seed(seed)

    def split(ds):
        n = len(ds)
        ratios = (0.7, 0.2, 0.1)
        sizes = [int(r * n) for r in ratios]
        sizes[2] = n - sum(sizes[:2])
        return random_split(ds, sizes, generator=generator)

    _, vl, _ = split(landslide_raw)
    _, vn, _ = split(nonlandslide_raw)
    val_ds = BijieTwoComposites(ConcatDataset([vl, vn]), resize_to=256, transform=None)
    return DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def prep_batch(batch, device: torch.device):
    x1 = batch["stream_a"].float().to(device, non_blocking=True)
    x2 = batch["stream_b"].float().to(device, non_blocking=True)
    y = batch["mask"].float()
    if y.dtype.is_floating_point:
        y = y.round()
    y = y.to(device, non_blocking=True)
    if y.dim() == 3:
        y = y.unsqueeze(1)
    return x1, x2, y


def load_ps_gplnet(args, device: torch.device) -> nn.Module:
    backbone = "tf_efficientnet_b0" if args.compact else args.backbone
    fusion_ch = 32 if args.compact else args.fusion_channels
    lora = 4 if args.compact else args.lora_rank
    model = GeoPhysicsLandslideNet(
        n_classes=1,
        backbone=backbone,
        n_channels=3,
        pretrained=False,
        freeze_backbone=True,
        prithvi_snapshot=args.prithvi_snapshot,
        lora_rank=lora,
        fusion_channels=fusion_ch,
        tteb_attn_chunk=512 if args.compact else 1024,
        tteb_attn_low_res_max=1024 if args.compact else 4096,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:8]}...")
    if unexpected:
        print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:8]}...")
    meta = {
        "ckpt_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "ckpt_best_f1": ckpt.get("best_f1") if isinstance(ckpt, dict) else None,
        "checkpoint": str(args.checkpoint),
        "compact": bool(args.compact),
        "backbone": backbone,
        "fusion_channels": fusion_ch,
    }
    print(f"Loaded PS-GPLNet | epoch={meta['ckpt_epoch']} best_f1={meta['ckpt_best_f1']}")
    model.to(device).eval()
    model._tpami_meta = meta  # type: ignore[attr-defined]
    return model


def _main_logits(out) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


@torch.no_grad()
def predict_masks(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    max_batches: int = 0,
    noise_sigma: float = 0.0,
    noise_target: str = "both",
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
    preds, gts = [], []
    tp = fp = fn = 0
    n_batches = 0
    for batch in tqdm(loader, desc=f"infer σ={noise_sigma}", leave=False):
        x1, x2, y = prep_batch(batch, device)
        if noise_sigma > 0:
            if noise_target in ("both", "rgb"):
                x1 = x1 + torch.randn_like(x1) * noise_sigma
            if noise_target in ("both", "dem"):
                x2 = x2 + torch.randn_like(x2) * noise_sigma
        logits = _main_logits(model(x1, x2))
        prob = torch.sigmoid(logits)
        pred = (prob >= threshold).to(torch.int64)
        y_i = y.to(torch.int64)
        tp += int((pred * y_i).sum().item())
        fp += int((pred * (1 - y_i)).sum().item())
        fn += int(((1 - pred) * y_i).sum().item())
        for i in range(pred.size(0)):
            preds.append(pred[i, 0].cpu().numpy().astype(np.uint8))
            gts.append(y_i[i, 0].cpu().numpy().astype(np.uint8))
        n_batches += 1
        if max_batches > 0 and n_batches >= max_batches:
            break
    eps = 1e-8
    metrics = {
        "iou": tp / (tp + fp + fn + eps),
        "f1": (2 * tp) / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "n_images": float(len(preds)),
    }
    return preds, gts, metrics


def run_boundary_study(preds, gts, out_dir: Path) -> Dict[str, float]:
    bfs, hds = [], []
    bfs_ls, hds_ls = [], []
    for p, g in zip(preds, gts):
        bf = compute_boundary_fscore(p, g, theta=2.0)
        hd = compute_hausdorff_distance(p, g)
        bfs.append(bf)
        hds.append(hd)
        # Landslide-present subset (avoids trivial empty-empty zeros)
        if g.sum() > 0:
            bfs_ls.append(bf)
            hds_ls.append(hd)
    summary = {
        "bf_score_mean": float(np.mean(bfs)) if bfs else 0.0,
        "bf_score_std": float(np.std(bfs)) if bfs else 0.0,
        "hausdorff_mean": float(np.mean(hds)) if hds else 0.0,
        "hausdorff_median": float(np.median(hds)) if hds else 0.0,
        "hausdorff_std": float(np.std(hds)) if hds else 0.0,
        "n": len(bfs),
        "bf_score_mean_landslide": float(np.mean(bfs_ls)) if bfs_ls else 0.0,
        "hausdorff_mean_landslide": float(np.mean(hds_ls)) if hds_ls else 0.0,
        "hausdorff_median_landslide": float(np.median(hds_ls)) if hds_ls else 0.0,
        "n_landslide": len(bfs_ls),
    }
    fig_path = out_dir / "boundary_metrics_hist.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    axes[0].hist(bfs, bins=30, color="#1f4e79", alpha=0.85)
    axes[0].set_title(f"Boundary F-score (mean={summary['bf_score_mean']:.3f})")
    axes[0].set_xlabel("BF-score")
    axes[1].hist(hds, bins=30, color="#8b1e1e", alpha=0.85)
    axes[1].set_title(f"Hausdorff (mean={summary['hausdorff_mean']:.2f})")
    axes[1].set_xlabel("Hausdorff distance (px)")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[boundary] {summary} -> {fig_path}")
    return summary


def run_tsne_study(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    max_samples: int,
) -> Dict[str, Any]:
    feats: List[np.ndarray] = []
    labels: List[int] = []
    fs_means: List[float] = []
    captured: List[np.ndarray] = []

    def hook_fn(_module, _inp, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.dim() == 4:
            pooled = tensor.mean(dim=(2, 3))
        elif tensor.dim() == 2:
            pooled = tensor
        else:
            pooled = tensor.flatten(1)
        captured.append(pooled.detach().float().cpu().numpy())

    # Prefer MAO fuse3 (C-dim physics-semantic bottleneck). Do NOT hook MPEF:
    # fuse_main emits 1-channel logits -> pooled dim=1 breaks t-SNE/PCA.
    if hasattr(model, "fuse3"):
        handle = model.fuse3.register_forward_hook(hook_fn)
        hook_name = "fuse3"
    elif hasattr(model, "physics_decoder") and hasattr(model.physics_decoder, "bn_proj_a"):
        handle = model.physics_decoder.bn_proj_a.register_forward_hook(hook_fn)
        hook_name = "physics_decoder.bn_proj_a"
    else:
        handle = model.register_forward_hook(hook_fn)
        hook_name = "model"

    collected = 0
    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc="t-SNE feats", leave=False):
                x1, x2, y = prep_batch(batch, device)
                _ = model(x1, x2)
                if not captured:
                    continue
                feat = captured.pop(0)
                flat_y = y.view(y.size(0), -1).cpu().numpy()
                lab = (flat_y.sum(axis=1) > flat_y.shape[1] * 0.05).astype(int)
                # Proxy FS instability from DEM channel variance as weak topographic cue
                dem = x2[:, :1]
                slope_proxy = dem.std(dim=(2, 3)).squeeze(1).cpu().numpy()
                feats.append(feat)
                labels.append(lab)
                fs_means.append(slope_proxy)
                collected += x1.size(0)
                if collected >= max_samples:
                    break
    finally:
        handle.remove()

    if not feats:
        raise RuntimeError("t-SNE hook captured no features")

    X = np.concatenate(feats, axis=0)[:max_samples]
    y_lab = np.concatenate(labels, axis=0)[:max_samples]
    topo = np.concatenate(fs_means, axis=0)[:max_samples]

    if X.ndim != 2 or X.shape[1] < 2:
        raise RuntimeError(
            f"t-SNE needs feature dim>=2, got shape {X.shape} from hook={hook_name}. "
            "Hook a C-channel bottleneck (e.g. fuse3), not 1-channel logits."
        )
    print(f"[t-SNE] running on {X.shape} from hook={hook_name}")
    perplexity = max(5, min(30, X.shape[0] // 4))
    init = "pca" if X.shape[1] >= 2 else "random"
    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init=init,
        learning_rate="auto",
    ).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    sc0 = axes[0].scatter(emb[:, 0], emb[:, 1], c=y_lab, cmap="coolwarm", s=18, alpha=0.85, edgecolors="none")
    axes[0].set_title("t-SNE colored by landslide presence")
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, label="landslide (>5% area)")
    sc1 = axes[1].scatter(emb[:, 0], emb[:, 1], c=topo, cmap="viridis", s=18, alpha=0.85, edgecolors="none")
    axes[1].set_title("t-SNE colored by DEM texture proxy")
    plt.colorbar(sc1, ax=axes[1], fraction=0.046, label="DEM channel std")
    for ax in axes:
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")
        ax.grid(True, linestyle="--", alpha=0.35)
    fig.suptitle("PS-GPLNet latent manifold (MPEF / physics-routed)", fontsize=12)
    fig.tight_layout()
    fig_path = out_dir / "latent_manifold_tsne.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # Simple cluster separation score: mean distance between class centroids / pooled std
    if len(np.unique(y_lab)) == 2:
        c0, c1 = emb[y_lab == 0], emb[y_lab == 1]
        sep = float(np.linalg.norm(c0.mean(0) - c1.mean(0)) / (emb.std() + 1e-8))
    else:
        sep = float("nan")
    meta = {"hook": hook_name, "n": int(X.shape[0]), "centroid_separation": sep, "figure": str(fig_path)}
    print(f"[t-SNE] saved {fig_path} | centroid_separation={sep:.3f}")
    return meta


def run_robustness_study(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    noise_levels: List[float],
    threshold: float,
    max_batches: int,
    baseline_curves: Optional[Dict[str, Dict[str, List[float]]]] = None,
) -> Dict[str, Any]:
    f1s, ious = [], []
    for sigma in noise_levels:
        _, _, m = predict_masks(
            model, loader, device, threshold, max_batches=max_batches, noise_sigma=sigma, noise_target="both"
        )
        f1s.append(m["f1"])
        ious.append(m["iou"])
        print(f"  [PS-GPLNet] σ={sigma:.2f} F1={m['f1']:.4f} IoU={m['iou']:.4f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    ax.plot(noise_levels, f1s, marker="o", color="#8b1e1e", linewidth=2.0, label="PS-GPLNet F1")
    ax.plot(noise_levels, ious, marker="s", color="#1f4e79", linewidth=2.0, label="PS-GPLNet IoU")
    if baseline_curves:
        for name, curves in baseline_curves.items():
            ax.plot(noise_levels, curves["f1"], marker="^", linestyle="--", linewidth=1.6, label=f"{name} F1")
    ax.set_xlabel(r"Gaussian noise $\sigma$ on RGB+DEM streams")
    ax.set_ylabel("Metric")
    ax.set_title("Perturbation robustness decay")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = out_dir / "robustness_decay.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # Decay slope (F1 drop per unit sigma) via linear fit
    if len(noise_levels) >= 2:
        slope = float(np.polyfit(noise_levels, f1s, 1)[0])
    else:
        slope = float("nan")
    out = {
        "noise_levels": noise_levels,
        "f1": f1s,
        "iou": ious,
        "f1_decay_slope": slope,
        "figure": str(fig_path),
    }
    print(f"[robustness] F1 decay slope={slope:.4f} -> {fig_path}")
    return out


def load_baseline_rgb(model_name: str, ckpt_path: str, device: torch.device) -> nn.Module:
    abl_root = Path(__file__).resolve().parents[2] / "ablation_study" / "baseline_models"
    sys.path.insert(0, str(abl_root))
    from common.architectures import build_model  # type: ignore

    model = build_model(model_name, in_channels=3, n_classes=1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    print(f"Loaded baseline {model_name} | epoch={ckpt.get('epoch')} best_f1={ckpt.get('best_f1')}")
    return model


@torch.no_grad()
def baseline_robustness(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    noise_levels: List[float],
    threshold: float,
    max_batches: int,
) -> Dict[str, List[float]]:
    """Baselines are single-stream RGB; use stream_a only."""
    f1s, ious = [], []
    for sigma in noise_levels:
        tp = fp = fn = 0
        n_batches = 0
        for batch in loader:
            x1 = batch["stream_a"].float().to(device)
            y = batch["mask"].float().to(device)
            if y.dim() == 3:
                y = y.unsqueeze(1)
            if sigma > 0:
                x1 = x1 + torch.randn_like(x1) * sigma
            logits = model(x1)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = (torch.sigmoid(logits) >= threshold).to(torch.int64)
            y_i = y.round().to(torch.int64)
            tp += int((pred * y_i).sum().item())
            fp += int((pred * (1 - y_i)).sum().item())
            fn += int(((1 - pred) * y_i).sum().item())
            n_batches += 1
            if max_batches > 0 and n_batches >= max_batches:
                break
        eps = 1e-8
        f1s.append((2 * tp) / (2 * tp + fp + fn + eps))
        ious.append(tp / (tp + fp + fn + eps))
        print(f"  [baseline] σ={sigma:.2f} F1={f1s[-1]:.4f}")
    return {"f1": f1s, "iou": ious}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip() != ""]
    device = resolve_device(args.device, auto_select=args.auto_gpu, min_free_gb=args.min_free_gb)
    print(f"device={device}")

    loader = build_val_loader(Path(args.dataset_root), args.seed, args.batch_size, args.num_workers)
    print(f"val batches≈{len(loader)} images≈{len(loader.dataset)}")

    model = load_ps_gplnet(args, device)
    report: Dict[str, Any] = {"meta": getattr(model, "_tpami_meta", {}), "device": str(device)}

    # Clean validation metrics + predictions
    preds, gts, clean_metrics = predict_masks(
        model, loader, device, args.metric_threshold, max_batches=args.max_val_batches, noise_sigma=0.0
    )
    report["clean_metrics"] = clean_metrics
    print(f"[clean] {clean_metrics}")

    if not args.skip_boundary:
        report["boundary"] = run_boundary_study(preds, gts, fig_dir)

    if not args.skip_tsne:
        report["tsne"] = run_tsne_study(model, loader, device, fig_dir, args.tsne_max_samples)

    baseline_curves = None
    if args.baseline_unet_ckpt or args.baseline_deeplab_ckpt:
        baseline_curves = {}
        if args.baseline_unet_ckpt:
            unet = load_baseline_rgb("unet", args.baseline_unet_ckpt, device)
            baseline_curves["U-Net"] = baseline_robustness(
                unet, loader, device, noise_levels, args.metric_threshold, args.max_val_batches
            )
            del unet
            torch.cuda.empty_cache()
        if args.baseline_deeplab_ckpt:
            dl = load_baseline_rgb("deeplabv3plus", args.baseline_deeplab_ckpt, device)
            baseline_curves["DeepLabV3+"] = baseline_robustness(
                dl, loader, device, noise_levels, args.metric_threshold, args.max_val_batches
            )
            del dl
            torch.cuda.empty_cache()
        report["baseline_robustness"] = baseline_curves

    if not args.skip_robustness:
        report["robustness"] = run_robustness_study(
            model,
            loader,
            device,
            fig_dir,
            noise_levels,
            args.metric_threshold,
            args.max_val_batches,
            baseline_curves=baseline_curves,
        )

    json_path = out_dir / "tpami_study_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
