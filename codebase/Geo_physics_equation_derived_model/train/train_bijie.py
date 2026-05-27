#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from ..data import build_bijie_dataloaders
from ..model import GeoPhysicsLandslideNet
from .trainer import train_model


def parse_args():
    p = argparse.ArgumentParser(description="Train GeoPhysicsLandslideNet on Bijie dataset.")
    p.add_argument(
        "--dataset_root",
        type=str,
        default="/home/user/Desktop/Deep_learning_projects/4PI/dataset_bijie_landslide",
    )
    p.add_argument("--output_dir", type=str, default="./outputs_bijie")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resize_to", type=int, default=256)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--metric_threshold", type=float, default=0.5)
    p.add_argument("--tversky_alpha", type=float, default=0.3)
    p.add_argument("--tversky_beta", type=float, default=0.7)
    p.add_argument("--main_weight", type=float, default=1.0)
    p.add_argument("--aux2_weight", type=float, default=0.6)
    p.add_argument("--aux3_weight", type=float, default=0.4)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--prithvi_snapshot", type=str, default=None)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    train_loader, val_loader = build_bijie_dataloaders(
        args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        resize_to=args.resize_to,
    )

    model = GeoPhysicsLandslideNet(
        channels=64,
        n_classes=1,
        lora_rank=args.lora_rank,
        prithvi_snapshot=args.prithvi_snapshot,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=Path(args.output_dir).resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device_str=args.device,
        metric_threshold=args.metric_threshold,
        save_every=args.save_every,
        resume=args.resume,
        alpha=args.tversky_alpha,
        beta=args.tversky_beta,
        main_weight=args.main_weight,
        aux2_weight=args.aux2_weight,
        aux3_weight=args.aux3_weight,
        extra_final={"dataset": "bijie", "dataset_root": args.dataset_root},
    )


if __name__ == "__main__":
    main()
