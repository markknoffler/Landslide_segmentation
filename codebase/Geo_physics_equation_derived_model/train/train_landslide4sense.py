#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from ..data import build_l4s_dataloaders
from ..model import GeoPhysicsLandslideNet
from .distributed import cleanup_distributed, init_distributed, is_main_process
from .fsdp_utils import wrap_geo_physics_fsdp
from .logging_utils import log_main
from .trainer import train_model


def parse_args():
    p = argparse.ArgumentParser(description="Train GeoPhysicsLandslideNet on Landslide4Sense.")
    p.add_argument(
        "--dataset_root",
        type=str,
        default="/home/user/Desktop/Deep_learning_projects/4PI/dataset",
    )
    p.add_argument("--output_dir", type=str, default="./outputs_l4s")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32, help="Per-GPU batch size.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resize_to", type=int, default=256)
    p.add_argument("--val_split_ratio", type=float, default=0.1)
    p.add_argument("--metric_threshold", type=float, default=0.6)
    p.add_argument("--tversky_alpha", type=float, default=0.7)
    p.add_argument("--tversky_beta", type=float, default=0.3)
    p.add_argument("--main_weight", type=float, default=1.0)
    p.add_argument("--aux2_weight", type=float, default=0.6)
    p.add_argument("--aux3_weight", type=float, default=0.4)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--prithvi_snapshot", type=str, default=None)
    p.add_argument(
        "--high_dim_256",
        action="store_true",
        help="Use unified feature width C=256 across the full model.",
    )
    p.add_argument(
        "--full_precision",
        action="store_true",
        help="Force full-float (FP32) training (disables FSDP bf16 mixed precision).",
    )
    p.add_argument("--fsdp", action="store_true")
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--no_activation_checkpointing", action="store_true")
    p.add_argument("--log_interval", type=int, default=10)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank = init_distributed()
    use_fsdp = args.fsdp or distributed
    channels = 256 if args.high_dim_256 else 64
    if distributed and not use_fsdp and is_main_process(rank):
        print("Warning: multi-GPU job without --fsdp; enabling FSDP automatically.")
        use_fsdp = True

    set_seed(args.seed + rank)

    log_main(rank, f"Rank {rank}/{world_size} — building dataloaders...")
    train_loader, val_loader, train_sampler, val_sampler = build_l4s_dataloaders(
        args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_split_ratio,
        seed=args.seed,
        resize_to=args.resize_to,
        distributed=distributed,
    )

    log_main(rank, "Building GeoPhysicsLandslideNet...")
    model = GeoPhysicsLandslideNet(
        channels=channels,
        n_classes=1,
        lora_rank=args.lora_rank,
        prithvi_snapshot=args.prithvi_snapshot,
    )

    if use_fsdp:
        if not distributed:
            raise RuntimeError("FSDP requires launch via torchrun (multiple processes).")
        model = wrap_geo_physics_fsdp(
            model,
            device_id=local_rank,
            use_bf16=not (args.no_bf16 or args.full_precision),
            activation_checkpointing=not args.no_activation_checkpointing,
        )

    log_main(
        rank,
        f"Ready: world_size={world_size} channels={channels} full_precision={args.full_precision} "
        f"train_steps={len(train_loader)} val_steps={len(val_loader)}",
    )

    try:
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
            alpha=args.tversky_alpha,
            beta=args.tversky_beta,
            main_weight=args.main_weight,
            aux2_weight=args.aux2_weight,
            aux3_weight=args.aux3_weight,
            extra_final={"dataset": "landslide4sense", "dataset_root": args.dataset_root},
            distributed=distributed,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            use_fsdp=use_fsdp,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            log_interval=args.log_interval,
            metrics_suffix="_w256" if args.high_dim_256 else "",
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
