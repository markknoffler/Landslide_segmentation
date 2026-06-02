from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from .distributed import all_reduce_sum_and_count, barrier, is_main_process
from .logging_utils import log_dict, log_main, tqdm_file
from .losses import GeoPhysicsLoss
from .metrics import image_level_metrics_from_logits, pixel_metrics_from_logits


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _epoch_metrics_line(row: dict) -> str:
    return (
        f"epoch={row['epoch']:03d} | "
        f"train loss={_fmt(row['train_loss'])} acc={_fmt(row['train_acc'])} "
        f"prec={_fmt(row['train_precision'])} rec={_fmt(row['train_recall'])} "
        f"f1={_fmt(row['train_f1'])} iou={_fmt(row['train_iou'])} | "
        f"val loss={_fmt(row['val_loss'])} acc={_fmt(row['val_acc'])} "
        f"prec={_fmt(row['val_precision'])} rec={_fmt(row['val_recall'])} "
        f"f1={_fmt(row['val_f1'])} iou={_fmt(row['val_iou'])}"
    )


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    threshold: float,
    training: bool,
    optimizer=None,
    *,
    rank: int = 0,
    distributed: bool = False,
    epoch: int | None = None,
    train_sampler=None,
    log_interval: int = 10,
    total_epochs: int | None = None,
):
    if training and train_sampler is not None and epoch is not None:
        train_sampler.set_epoch(epoch)

    model.train() if training else model.eval()
    losses = []
    pix_hist = {"acc": [], "precision": [], "recall": [], "f1": [], "iou": []}
    img_hist = {"auroc": [], "auprc": [], "best_f1": [], "best_threshold": []}

    phase = "Train" if training else "Val"
    if epoch is not None and total_epochs is not None:
        desc = f"E{epoch:03d}/{total_epochs:03d} {phase}"
    elif epoch is not None:
        desc = f"E{epoch:03d} {phase}"
    else:
        desc = phase

    show_pbar = is_main_process(rank)
    num_batches = len(loader)
    t0 = time.time()

    pbar = tqdm(
        loader,
        desc=desc,
        total=num_batches,
        leave=True,
        disable=not show_pbar,
        file=tqdm_file(),
        dynamic_ncols=True,
        mininterval=0.5,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    for batch in pbar:
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device, non_blocking=True)
        y = batch["mask"].float()
        if y.dim() == 3:
            y = y.unsqueeze(1)

        with torch.set_grad_enabled(training):
            main, aux2, aux3 = model(batch)
            loss = criterion((main, aux2, aux3), y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        pix = pixel_metrics_from_logits(main, y, threshold=threshold)
        for k in pix_hist:
            pix_hist[k].append(float(pix[k]))
        img = image_level_metrics_from_logits(main, y, prob_thr_for_instances=threshold, min_area=20)
        for k in img_hist:
            img_hist[k].append(float(img[k]))

        if show_pbar:
            pbar.set_postfix(
                loss=f"{losses[-1]:.4f}",
                f1=f"{pix_hist['f1'][-1]:.4f}",
                iou=f"{pix_hist['iou'][-1]:.4f}",
                refresh=False,
            )

    def _mean(vals: list[float]) -> float:
        return all_reduce_sum_and_count(vals, device) if distributed else (float(np.mean(vals)) if vals else 0.0)

    metrics = {
        "loss": _mean(losses),
        **{k: _mean(v) for k, v in pix_hist.items()},
        "auroc": _mean(img_hist["auroc"]),
        "auprc": _mean(img_hist["auprc"]),
        "image_best_f1": _mean(img_hist["best_f1"]),
        "image_best_threshold": _mean(img_hist["best_threshold"]) if img_hist["best_threshold"] else threshold,
    }

    if show_pbar:
        log_main(
            rank,
            f"{desc} done in {time.time() - t0:.0f}s | "
            f"loss={metrics['loss']:.4f} f1={metrics['f1']:.4f} iou={metrics['iou']:.4f}",
        )

    return metrics


def train_model(
    model,
    train_loader,
    val_loader,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    device_str: str = "cuda",
    metric_threshold: float = 0.5,
    alpha: float = 0.3,
    beta: float = 0.7,
    main_weight: float = 1.0,
    aux2_weight: float = 0.6,
    aux3_weight: float = 0.4,
    extra_final: Optional[Dict] = None,
    *,
    distributed: bool = False,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
    use_fsdp: bool = False,
    train_sampler=None,
    val_sampler=None,
    log_interval: int = 10,
    metrics_suffix: str = "",
):
    if distributed and torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    if not use_fsdp:
        model = model.to(device)

    criterion = GeoPhysicsLoss(
        alpha=alpha,
        beta=beta,
        main_weight=main_weight,
        aux2_weight=aux2_weight,
        aux3_weight=aux3_weight,
    )
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    results_dir = output_dir / "results"
    suffix = metrics_suffix.strip()
    epoch_csv_name = f"epoch_metrics{suffix}.csv" if suffix else "epoch_metrics.csv"
    final_csv_name = f"final_metrics{suffix}.csv" if suffix else "final_metrics.csv"
    epoch_csv = results_dir / epoch_csv_name
    final_csv = results_dir / final_csv_name

    best_f1 = 0.0

    log_dict(
        rank,
        "Training setup",
        {
            "output_dir": str(output_dir),
            "epochs": f"1..{epochs}",
            "train_batches_per_epoch": len(train_loader),
            "val_batches_per_epoch": len(val_loader),
            "per_gpu_batch": batch_size,
            "global_batch": batch_size * world_size,
            "world_size": world_size,
            "fsdp": use_fsdp,
            "device": str(device),
            "metrics_csv": str(epoch_csv),
            "model_checkpoints": "disabled (metrics-only to avoid GlusterFS stalls)",
        },
    )
    log_main(rank, "Metrics append to results/epoch_metrics.csv (no .pt checkpoints written).")

    if distributed:
        barrier()

    for epoch in range(1, epochs + 1):
        if is_main_process(rank):
            log_main(rank, f"\n===== Epoch {epoch}/{epochs} =====")
        train_m = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            threshold=metric_threshold,
            training=True,
            optimizer=optimizer,
            rank=rank,
            distributed=distributed,
            epoch=epoch,
            train_sampler=train_sampler,
            log_interval=log_interval,
            total_epochs=epochs,
        )
        val_m = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            threshold=metric_threshold,
            training=False,
            optimizer=None,
            rank=rank,
            distributed=distributed,
            epoch=epoch,
            log_interval=log_interval,
            total_epochs=epochs,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_acc": train_m["acc"],
            "train_precision": train_m["precision"],
            "train_recall": train_m["recall"],
            "train_f1": train_m["f1"],
            "train_iou": train_m["iou"],
            "train_auroc": train_m["auroc"],
            "train_auprc": train_m["auprc"],
            "train_image_best_f1": train_m["image_best_f1"],
            "train_image_best_threshold": train_m["image_best_threshold"],
            "val_loss": val_m["loss"],
            "val_acc": val_m["acc"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_f1": val_m["f1"],
            "val_iou": val_m["iou"],
            "val_auroc": val_m["auroc"],
            "val_auprc": val_m["auprc"],
            "val_image_best_f1": val_m["image_best_f1"],
            "val_image_best_threshold": val_m["image_best_threshold"],
        }
        if is_main_process(rank):
            append_csv(epoch_csv, row)
            log_main(rank, _epoch_metrics_line(row))

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            log_main(rank, f"New best val F1={best_f1:.4f} (metrics only; no checkpoint saved)")

    log_main(rank, f"Training finished. Best val F1={best_f1:.4f}")

    if is_main_process(rank):
        final = {
            "best_val_f1": best_f1,
            "epochs": epochs,
            "batch_size": batch_size,
            "per_gpu_batch_size": batch_size,
            "world_size": world_size,
            "global_batch_size": batch_size * world_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "metric_threshold": metric_threshold,
            "tversky_alpha": alpha,
            "tversky_beta": beta,
            "main_weight": main_weight,
            "aux2_weight": aux2_weight,
            "aux3_weight": aux3_weight,
            "use_fsdp": use_fsdp,
            "distributed": distributed,
        }
        if extra_final:
            final.update(extra_final)
        append_csv(final_csv, final)
        log_main(rank, f"Final metrics written to {final_csv}")
