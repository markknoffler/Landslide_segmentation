import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Landslide4SenseDualStream
from losses import DualStreamLoss
from metrics import image_level_detection_metrics, pixel_metrics_from_logits
from model import DualStreamGateNet


def parse_args():
    p = argparse.ArgumentParser(description="Train dual-stream gated landslide segmentation model.")
    p.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root containing TrainData/ValidData/TestData.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in checkpoint directory.")
    p.add_argument("--output_dir", type=str, default=".", help="Directory where checkpoint/ and results/ will be created.")
    p.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs.")
    p.add_argument("--rgb_indices", type=int, nargs=3, default=[3, 2, 1])
    p.add_argument("--nir_index", type=int, default=7)
    p.add_argument("--slope_index", type=int, default=12)
    p.add_argument("--dem_index", type=int, default=13)
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device):
    model.eval()
    loss_meter = 0.0
    n_batches = 0

    sum_tp = sum_fp = sum_fn = sum_tn = 0
    image_scores = []
    image_labels = []

    for batch in tqdm(loader, desc="Valid", leave=False):
        xa = batch["stream_a"].to(device)
        xb = batch["stream_b"].to(device)
        y = batch["mask"].to(device)

        out = model(xa, xb)
        loss_dict = criterion(out, y)
        loss_meter += float(loss_dict["loss"].item())
        n_batches += 1

        pix = pixel_metrics_from_logits(out["main"], y, threshold=0.5)
        sum_tp += pix["tp"]
        sum_fp += pix["fp"]
        sum_fn += pix["fn"]
        sum_tn += pix["tn"]

        probs = torch.sigmoid(out["main"]).amax(dim=(-2, -1)).squeeze(1).detach().cpu().numpy()
        labels = (y.amax(dim=(-2, -1)).squeeze(1) > 0.5).detach().cpu().numpy().astype(int)
        image_scores.extend(probs.tolist())
        image_labels.extend(labels.tolist())

    precision = (sum_tp / (sum_tp + sum_fp)) if (sum_tp + sum_fp) > 0 else 0.0
    recall = (sum_tp / (sum_tp + sum_fn)) if (sum_tp + sum_fn) > 0 else 0.0
    f1 = (2 * sum_tp / (2 * sum_tp + sum_fp + sum_fn)) if (2 * sum_tp + sum_fp + sum_fn) > 0 else 0.0
    iou = (sum_tp / (sum_tp + sum_fp + sum_fn)) if (sum_tp + sum_fp + sum_fn) > 0 else 0.0
    acc = ((sum_tp + sum_tn) / (sum_tp + sum_tn + sum_fp + sum_fn)) if (sum_tp + sum_tn + sum_fp + sum_fn) > 0 else 0.0
    img_metrics = image_level_detection_metrics(image_scores, image_labels)

    return {
        "loss": loss_meter / max(n_batches, 1),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "auroc": img_metrics["auroc"],
        "auprc": img_metrics["auprc"],
        "best_f1": img_metrics["best_f1"],
        "best_threshold": img_metrics["best_threshold"],
    }


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        xa = batch["stream_a"].to(device)
        xb = batch["stream_b"].to(device)
        y = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(xa, xb)
        loss_dict = criterion(out, y)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    ckpt_dir = output_dir / "checkpoint"
    results_dir = output_dir / "results"
    epoch_csv = results_dir / "epoch_metrics.csv"
    final_csv = results_dir / "final_metrics.csv"

    train_ds = Landslide4SenseDualStream(
        data_root=args.dataset_root,
        split="train",
        rgb_indices=tuple(args.rgb_indices),
        nir_index=args.nir_index,
        slope_index=args.slope_index,
        dem_index=args.dem_index,
    )
    valid_ds = Landslide4SenseDualStream(
        data_root=args.dataset_root,
        split="valid",
        rgb_indices=tuple(args.rgb_indices),
        nir_index=args.nir_index,
        slope_index=args.slope_index,
        dem_index=args.dem_index,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = DualStreamGateNet(in_channels_a=3, in_channels_b=3, base_channels=args.base_channels).to(device)
    criterion = DualStreamLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_f1 = 0.0

    if args.resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state["epoch"]) + 1
            best_f1 = float(state.get("best_f1", 0.0))
            print(f"Resumed from {ckpt} at epoch {start_epoch}.")
        else:
            print("Resume requested, but no checkpoint found. Starting from scratch.")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, valid_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_f1": val["f1"],
            "val_iou": val["iou"],
            "val_auroc": val["auroc"],
            "val_auprc": val["auprc"],
            "val_image_best_f1": val["best_f1"],
            "val_image_best_threshold": val["best_threshold"],
        }
        append_csv(epoch_csv, row)
        print(row)

        if epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:04d}.pt",
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "args": vars(args),
                },
            )

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            save_checkpoint(
                ckpt_dir / "best.pt",
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "args": vars(args),
                },
            )

    append_csv(
        final_csv,
        {
            "best_val_f1": best_f1,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dataset_root": args.dataset_root,
        },
    )


if __name__ == "__main__":
    main()
