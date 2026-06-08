import argparse
import csv
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import DualStreamTransform, Landslide4SenseDualStream
from losses import DualStreamLoss
from metrics import pixel_metrics_from_logits
from model_new_v2 import DualStreamMixedNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train paper-aligned DiGATe-UNet on Landslide4Sense.")
    parser.add_argument("--dataset_root", type=str, default="/home/user/Desktop/Deep_learning_projects/4PI/dataset")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize_to", type=int, default=256)
    parser.add_argument("--bands", type=str, default="RGB-NDVI-SLOPE-DEM")
    parser.add_argument("--backbone", type=str, default="tf_efficientnet_b4")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze_backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--share_backbone", action="store_true", default=False)
    parser.add_argument("--no-share_backbone", dest="share_backbone", action="store_false")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--use_input_adapter", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--val_split_ratio", type=float, default=0.1)
    parser.add_argument("--val_split_seed", type=int, default=42)
    parser.add_argument("--tversky_alpha", type=float, default=0.6)
    parser.add_argument("--tversky_beta", type=float, default=0.4)
    parser.add_argument("--main_weight", type=float, default=1.0)
    parser.add_argument("--aux2_weight", type=float, default=0.6)
    parser.add_argument("--aux3_weight", type=float, default=0.4)
    parser.add_argument("--reg_weight", type=float, default=1e-3)
    parser.add_argument("--metric_threshold", type=float, default=0.6)

    return parser.parse_args()


def main():
    args = parse_args()
    print("Args:", args)

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Create model
    model = DualStreamMixedNet(
        n_classes=1,
        backbone_rgb=args.backbone,
        backbone_dem=args.backbone,
        backbone_prithvi="prithvi_vit_384",  # Placeholder - you would specify the actual Prithvi backbone
        n_channels_rgb=3,
        n_channels_dem=1,
        pretrained=args.pretrained,
        pretrained_path=args.pretrained_path,
        use_input_adapter=args.use_input_adapter,
        freeze_backbone=args.freeze_backbone,
    )

    # Move model to device
    device = torch.device(args.device)
    model = model.to(device)

    # Create dataset
    transform = DualStreamTransform(
        resize_to=args.resize_to,
        bands=args.bands,
        augment=True,
    )
    dataset = Landslide4SenseDualStream(
        root=args.dataset_root,
        transform=transform,
        bands=args.bands,
    )

    # Split dataset
    n_val = int(len(dataset) * args.val_split_ratio)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.val_split_seed),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create loss function
    loss_fn = DualStreamLoss(
        alpha=args.tversky_alpha,
        beta=args.tversky_beta,
        main_weight=args.main_weight,
        aux2_weight=args.aux2_weight,
        aux3_weight=args.aux3_weight,
        reg_weight=args.reg_weight,
    )

    # Resume training if requested
    start_epoch = 0
    if args.resume:
        latest_checkpoint = checkpoint_dir / "latest.pt"
        if latest_checkpoint.exists():
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1

    # Training loop
    best_f1 = 0.0
    train_metrics = []
    val_metrics = []

    for epoch in range(start_epoch, args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            rgb = batch["rgb"].to(device)
            dem = batch["dem"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            main, aux2, aux3, reg = model(rgb, dem)
            loss = loss_fn(main, aux2, aux3, mask, reg)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_metrics = []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in pbar:
                rgb = batch["rgb"].to(device)
                dem = batch["dem"].to(device)
                mask = batch["mask"].to(device)

                main, aux2, aux3, reg = model(rgb, dem)
                loss = loss_fn(main, aux2, aux3, mask, reg)
                val_loss += loss.item()
                val_batches += 1

                # Compute metrics
                metrics = pixel_metrics_from_logits(main, mask, threshold=args.metric_threshold)
                all_metrics.append(metrics)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss /= val_batches
        avg_metrics = {k: torch.stack([m[k] for m in all_metrics]).mean() for k in all_metrics[0].keys()}
        f1 = avg_metrics["f1"]

        # Save metrics
        train_metrics.append({"epoch": epoch, "loss": train_loss})
        val_metrics.append({"epoch": epoch, "loss": val_loss, **{k: v.item() for k, v in avg_metrics.items()}})

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch+1:04d}.pt")
            torch.save(checkpoint, checkpoint_dir / "latest.pt")

            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                torch.save(checkpoint, checkpoint_dir / "best.pt")

        # Print epoch results
        print(
            f"Epoch {epoch+1:03d}: Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"F1: {f1:.4f}, "
            f"Precision: {avg_metrics['precision'].item():.4f}, "
            f"Recall: {avg_metrics['recall'].item():.4f}"
        )

    # Save final metrics
    with open(results_dir / "epoch_metrics.csv", "w", newline="") as f:
        if train_metrics:
            fieldnames = list(train_metrics[0].keys()) + list(val_metrics[0].keys())[1:]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(train_metrics)):
                row = train_metrics[i].copy()
                row.update(val_metrics[i])
                writer.writerow(row)

    # Save final summary
    final_metrics = {
        "best_f1": best_f1.item(),
        "final_epoch": args.epochs,
    }
    with open(results_dir / "final_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_metrics.keys()))
        writer.writeheader()
        writer.writerow(final_metrics)

    print(f"Training completed. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()