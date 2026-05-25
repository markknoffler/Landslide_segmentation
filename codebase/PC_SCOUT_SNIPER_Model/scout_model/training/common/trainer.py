from __future__ import annotations

import os
import random

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils
from tqdm import tqdm

from common.datasets import ScoutAugment2D

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def latest_checkpoint(directory: str | Path, pattern: str = "*.pt") -> Optional[Path]:
    d = Path(directory)
    ckpts = list(d.glob(pattern))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(p.stem.split("_")[-1]) if "_" in p.stem else 0)
    return ckpts[-1]


def save_checkpoint(
    path: str | Path,
    generator: nn.Module,
    critic: nn.Module,
    g_optim: torch.optim.Optimizer,
    c_optim: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "g_optimizer_state_dict": g_optim.state_dict(),
            "c_optimizer_state_dict": c_optim.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def append_csv(path: str | Path, row: dict[str, float]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(row.keys())
    if not p.exists():
        p.write_text(",".join(keys) + "\n")
    with open(p, "a") as f:
        f.write(",".join(str(row[k]) for k in keys) + "\n")



# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_scout_model(
    generator: nn.Module,
    critic: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    g_optimizer: torch.optim.Optimizer,
    c_optimizer: torch.optim.Optimizer,
    recon_loss_fn: nn.Module,
    gradient_penalty_fn,
    critic_loss_fn,
    generator_loss_fn,
    device: torch.device,
    num_epochs: int,
    n_critic: int = 5,
    recon_weight: float = 100.0,
    adv_weight: float = 1.0,
    lambda_gp: float = 10.0,
    metric_threshold: float = 0.5,
    output_dir: str | Path = "./output",
    run_id: str = "scout",
    save_every: int = 5,
    resume: Optional[str] = None,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / run_id / "checkpoint"
    results_dir = output_dir / run_id / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_loss: float = float("inf")
    best_metrics: dict[str, float] = {}

    # Resume from checkpoint ---------------------------------------------------
    if resume:
        ckpt_path = Path(resume)
        if not ckpt_path.is_file():
            latest = latest_checkpoint(ckpt_dir)
            if latest:
                ckpt_path = latest
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location=device)
            generator.load_state_dict(ckpt["generator_state_dict"])
            critic.load_state_dict(ckpt["critic_state_dict"])
            g_optimizer.load_state_dict(ckpt["g_optimizer_state_dict"])
            c_optimizer.load_state_dict(ckpt["c_optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("metrics", {}).get("val/recon_loss", float("inf"))
            tqdm.write(f"Resumed from epoch {start_epoch - 1}")

    # CSV headers ---------------------------------------------------------------
    csv_path = results_dir / "epoch_metrics.csv"
    val_csv_path = results_dir / "val_epoch_metrics.csv"
    append_csv(
        csv_path,
        {
            "epoch": 0,
            "g_loss": 0,
            "c_loss": 0,
            "recon_loss": 0,
        },
    )
    append_csv(
        val_csv_path,
        {
            "epoch": 0,
            "val_recon_loss": 0,
            "val_surprise_mean": 0,
        },
    )

    # Train loop ----------------------------------------------------------------
    for epoch in range(start_epoch, num_epochs + 1):
        generator.train()
        critic.train()
        epoch_g_loss: list[float] = []
        epoch_c_loss: list[float] = []
        epoch_recon_loss: list[float] = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for batch in pbar:
            dem = batch["dem"].to(device)       # B×1×H×W
            real_rgb = batch["rgb"].to(device)  # B×3×H×W

            # ---- Critic updates (n_critic per generator step) ----------------
            for _ in range(n_critic):
                fake_rgb = generator(dem).detach()
                c_loss = critic_loss_fn(critic, dem, real_rgb, fake_rgb, lambda_gp)
                c_optimizer.zero_grad()
                c_loss.backward()
                c_optimizer.step()
                epoch_c_loss.append(float(c_loss))

            # ---- Generator update --------------------------------------------
            fake_rgb = generator(dem)
            adv_loss = generator_loss_fn(critic, dem, fake_rgb) * adv_weight
            recon_loss = recon_loss_fn(fake_rgb, real_rgb) * recon_weight
            g_loss = adv_loss + recon_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss.append(float(g_loss))
            epoch_recon_loss.append(float(recon_loss))

            pbar.set_postfix(
                g_loss=float(g_loss), c_loss=float(c_loss), recon=float(recon_loss)
            )

        # Log train metrics ----------------------------------------------------
        avg = {
            "epoch": epoch,
            "g_loss": float(np.mean(epoch_g_loss)),
            "c_loss": float(np.mean(epoch_c_loss)),
            "recon_loss": float(np.mean(epoch_recon_loss)),
        }
        append_csv(csv_path, avg)

        # ---- Validation ------------------------------------------------------
        val_metrics = _validate(
            generator,
            val_loader,
            device,
        )
        append_csv(val_csv_path, {"epoch": epoch, **val_metrics})
        tqdm.write(
            f"Epoch {epoch}: g_loss={avg['g_loss']:.4f}  c_loss={avg['c_loss']:.4f}  "
            f"val_recon_loss={val_metrics['val_recon_loss']:.4f}  val_surprise={val_metrics['val_surprise_mean']:.4f}"
        )

        # Best checkpoint tracking ---------------------------------------------
        current_val_loss = val_metrics.get("val_recon_loss", float("inf"))
        is_best = current_val_loss < best_val_loss
        if is_best:
            best_val_loss = current_val_loss
            best_metrics = val_metrics

        # Save checkpoint ------------------------------------------------------
        if epoch % save_every == 0 or is_best or epoch == num_epochs:
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                ckpt_path,
                generator,
                critic,
                g_optimizer,
                c_optimizer,
                epoch,
                {"val/recon_loss": current_val_loss, **val_metrics},
            )
            if is_best:
                best_path = ckpt_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "generator_state_dict": generator.state_dict(),
                        "critic_state_dict": critic.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    best_path,
                )

        # Save sample predictions every save_every epochs --------------------
        if epoch % save_every == 0:
            sample_dir = ckpt_dir / f"epoch_{epoch:04d}" / "test_data"
            _save_epoch_predictions(
                generator,
                val_loader,
                device,
                sample_dir,
                num_samples=5,
            )

    # Final metrics ------------------------------------------------------------
    final = {"val_recon_loss": best_val_loss, "val_surprise": best_metrics.get("val_surprise_mean", 0.0)}
    append_csv(results_dir / "final_metrics.csv", final)

    # Save 5 predicted images from validation set ------------------------------
    _save_predictions(
        generator,
        val_loader,
        device,
        output_dir / run_id / "predictions",
        num_samples=5,
    )
    return final


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(
    generator: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    generator.eval()
    recon_losses: list[float] = []
    surprise_vals: list[float] = []

    for batch in val_loader:
        dem = batch["dem"].to(device)
        real_rgb = batch["rgb"].to(device)

        fake_rgb = generator(dem)
        recon_loss = F.mse_loss(fake_rgb, real_rgb)
        recon_losses.append(float(recon_loss.item()))

        surprise = ((real_rgb - fake_rgb).abs()).mean(dim=1, keepdim=True)
        surprise_vals.append(float(surprise.mean().item()))

    return {
        "val_recon_loss": float(np.mean(recon_losses)),
        "val_surprise_mean": float(np.mean(surprise_vals)),
    }


# ---------------------------------------------------------------------------
# Landslide test evaluation  –  surprise + segmentation metrics
# ---------------------------------------------------------------------------

EPS = 1e-7


@torch.no_grad()
def evaluate_on_landslide(
    generator: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    generator.eval()
    recon_losses: list[float] = []
    surprise_vals: list[float] = []
    tp = fp = fn = tn = 0.0

    for batch in test_loader:
        dem = batch["dem"].to(device)
        real_rgb = batch["rgb"].to(device)
        mask = batch["mask"].to(device)

        fake_rgb = generator(dem)
        recon_losses.append(float(F.mse_loss(fake_rgb, real_rgb).item()))

        # surprise factor  S = |real - pred|.mean(dim=1)   →  B×1×H×W
        surprise = (real_rgb - fake_rgb).abs().mean(dim=1, keepdim=True)
        surprise_vals.append(float(surprise.mean().item()))

        # binarise surprise at threshold vs ground-truth mask
        pred_bin = (surprise > threshold).float()
        mask_bin = mask.float()
        tp += (pred_bin * mask_bin).sum().item()
        fp += (pred_bin * (1 - mask_bin)).sum().item()
        fn += ((1 - pred_bin) * mask_bin).sum().item()
        tn += ((1 - pred_bin) * (1 - mask_bin)).sum().item()

    acc = (tp + tn) / (tp + fp + fn + tn + EPS)
    prec = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * prec * recall / (prec + recall + EPS)
    iou = tp / (tp + fp + fn + EPS)

    return {
        "test_recon_loss": float(np.mean(recon_losses)),
        "test_surprise_mean": float(np.mean(surprise_vals)),
        "test_acc": acc,
        "test_prec": prec,
        "test_recall": recall,
        "test_f1": f1,
        "test_iou": iou,
    }


# ---------------------------------------------------------------------------
# Prediction image saver
# ---------------------------------------------------------------------------

@torch.no_grad()
def _save_predictions(
    generator: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: str | Path,
    num_samples: int = 5,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    saved = 0

    for batch in val_loader:
        dem = batch["dem"].to(device)
        fake_rgb = generator(dem)
        b = min(fake_rgb.size(0), num_samples - saved)
        for i in range(b):
            torchvision.utils.save_image(
                fake_rgb[i],
                save_dir / f"pred_{saved:04d}.png",
            )
            saved += 1
        if saved >= num_samples:
            break

    tqdm.write(f"Saved {saved} prediction images to {save_dir}")


@torch.no_grad()
def _save_epoch_predictions(
    generator: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: str | Path,
    num_samples: int = 5,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    saved = 0

    for batch in val_loader:
        dem = batch["dem"].to(device)          # B×1×H×W
        real_rgb = batch["rgb"].to(device)     # B×3×H×W
        fake_rgb = generator(dem)              # B×3×H×W

        b = min(fake_rgb.size(0), num_samples - saved)
        for i in range(b):
            dem_single = dem[i]                # 1×H×W
            real_single = real_rgb[i]          # 3×H×W
            fake_single = fake_rgb[i]          # 3×H×W
            grid = torchvision.utils.make_grid(
                [dem_single.repeat(3, 1, 1), real_single, fake_single],
                nrow=3,
            )
            torchvision.utils.save_image(
                grid,
                save_dir / f"sample_{saved:04d}.png",
            )
            saved += 1
        if saved >= num_samples:
            break

    tqdm.write(f"Saved {saved} epoch-sample grids to {save_dir}")
