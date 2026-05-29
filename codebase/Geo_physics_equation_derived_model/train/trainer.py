from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.optim import Adam
from tqdm import tqdm

from .distributed import all_reduce_sum_and_count, barrier, is_main_process
from .logging_utils import log_dict, log_main, tqdm_file
from .losses import GeoPhysicsLoss
from .metrics import image_level_metrics_from_logits, pixel_metrics_from_logits


def latest_checkpoint(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def save_checkpoint(path: Path, state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _gather_model_state(model) -> Dict[str, torch.Tensor]:
    if isinstance(model, FSDP):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()
    return model.state_dict()


def _load_model_state(model, state_dict: Dict[str, torch.Tensor]) -> None:
    if isinstance(model, FSDP):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


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

    for batch_idx, batch in enumerate(pbar):
        if batch_idx == 0 and show_pbar and epoch == 1 and training:
            log_main(rank, f"{desc}: starting first batch (cold start can take several minutes)...")

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

        if batch_idx == 0 and show_pbar and epoch == 1 and training:
            log_main(rank, f"{desc}: first batch done in {time.time() - t0:.1f}s")

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

        if show_pbar and log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - t0
            log_main(
                rank,
                f"  {desc} step {batch_idx + 1}/{num_batches} "
                f"loss={losses[-1]:.4f} f1={pix_hist['f1'][-1]:.4f} "
                f"elapsed={elapsed:.0f}s",
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
    save_every: int = 5,
    resume: bool = False,
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

    ckpt_dir = output_dir / "checkpoint"
    results_dir = output_dir / "results"
    epoch_csv = results_dir / "epoch_metrics.csv"
    final_csv = results_dir / "final_metrics.csv"

    start_epoch = 1
    best_f1 = 0.0
    if resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            if is_main_process(rank):
                log_main(rank, f"Loading checkpoint {ckpt} ...")
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            _load_model_state(model, state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state["epoch"]) + 1
            best_f1 = float(state.get("best_f1", 0.0))
            log_main(rank, f"Resumed from epoch {start_epoch - 1}, best_val_f1={best_f1:.4f}")

    log_dict(
        rank,
        "Training setup",
        {
            "output_dir": str(output_dir),
            "epochs": f"{start_epoch}..{epochs}",
            "train_batches_per_epoch": len(train_loader),
            "val_batches_per_epoch": len(val_loader),
            "per_gpu_batch": batch_size,
            "global_batch": batch_size * world_size,
            "world_size": world_size,
            "fsdp": use_fsdp,
            "device": str(device),
            "metrics_csv": str(epoch_csv),
            "log_interval": log_interval,
        },
    )
    log_main(rank, "Epoch progress bars print below (rank 0 only). Metrics append to results/epoch_metrics.csv")

    if distributed:
        barrier()

    epoch_pbar = tqdm(
        range(start_epoch, epochs + 1),
        desc="Epochs",
        total=epochs - start_epoch + 1,
        disable=not is_main_process(rank),
        file=tqdm_file(),
        dynamic_ncols=True,
        initial=0,
    )

    for epoch in epoch_pbar:
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
            epoch_pbar.set_postfix(
                train_loss=f"{train_m['loss']:.3f}",
                val_f1=f"{val_m['f1']:.3f}",
                best_f1=f"{max(best_f1, val_m['f1']):.3f}",
                refresh=True,
            )
            log_dict(rank, f"Epoch {epoch} summary", row)

        improved = val_m["f1"] > best_f1
        if improved:
            best_f1 = val_m["f1"]
            log_main(rank, f"New best val F1={best_f1:.4f} — saving best.pt")

        if epoch % save_every == 0 or improved:
            barrier()
            if is_main_process(rank):
                log_main(rank, f"Saving checkpoint(s) for epoch {epoch}...")
                payload = {
                    "epoch": epoch,
                    "model": _gather_model_state(model),
                    "optimizer": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "world_size": world_size,
                    "use_fsdp": use_fsdp,
                }
                if epoch % save_every == 0:
                    path = ckpt_dir / f"epoch_{epoch:04d}.pt"
                    save_checkpoint(path, payload)
                    log_main(rank, f"  wrote {path}")
                if improved:
                    path = ckpt_dir / "best.pt"
                    save_checkpoint(path, payload)
                    log_main(rank, f"  wrote {path}")
            barrier()

    epoch_pbar.close()
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
