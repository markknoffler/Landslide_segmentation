#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from ..data import build_bijie_dataloaders
from ..model import GeoPhysicsLandslideNet
from .distributed import cleanup_distributed, init_distributed, is_main_process
from .fsdp_utils import wrap_geo_physics_fsdp
from .logging_utils import log_main
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
    p.add_argument("--batch_size", type=int, default=32, help="Per-GPU batch size.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resize_to", type=int, default=256)
    p.add_argument(
        "--metric_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for pixel metrics (same as dual_stream_gated/train_bijie.py).",
    )
    p.add_argument(
        "--tversky_alpha",
        type=float,
        default=0.3,
        help="Tversky FP weight (same default as dual_stream_gated Bijie).",
    )
    p.add_argument(
        "--tversky_beta",
        type=float,
        default=0.7,
        help="Tversky FN weight (same default as dual_stream_gated Bijie).",
    )
    p.add_argument("--main_weight", type=float, default=1.0)
    p.add_argument("--aux2_weight", type=float, default=0.6)
    p.add_argument("--aux3_weight", type=float, default=0.4)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument(
        "--fm_backbone",
        type=str,
        choices=("efficientnet", "prithvi"),
        default="efficientnet",
        help="Foundation stream: timm EfficientNet-B4 (RGB) or Prithvi-EO ViT (6-ch stack).",
    )
    p.add_argument("--prithvi_snapshot", type=str, default=None)
    p.add_argument(
        "--efficientnet_name",
        type=str,
        default="tf_efficientnet_b4",
        help="timm model name when --fm_backbone=efficientnet.",
    )
    p.add_argument(
        "--no_efficientnet_pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights for EfficientNet.",
    )
    p.add_argument(
        "--unfreeze_efficientnet",
        action="store_true",
        help="Unfreeze and train the full EfficientNet backbone (default: frozen).",
    )
    p.add_argument(
        "--decoder",
        type=str,
        choices=("physics", "conv"),
        default="physics",
        help="Decoder head: physics (default) or conv (standard UNet-style ablation).",
    )
    p.add_argument(
        "--fusion",
        type=str,
        choices=("balanced", "mao"),
        default="balanced",
        help="Fusion: balanced (intra-stream + symmetric mix + cross-attn) or mao (legacy).",
    )
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
    p.add_argument(
        "--fsdp",
        action="store_true",
        help="Wrap model in FSDP (recommended when using torchrun with multiple GPUs).",
    )
    p.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bfloat16 mixed precision under FSDP.",
    )
    p.add_argument(
        "--no_activation_checkpointing",
        action="store_true",
        help="Disable activation checkpointing on heavy blocks.",
    )
    p.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Print a plain-text line every N train/val steps on rank 0 (0=disable).",
    )
    p.add_argument(
        "--tteb_attn_chunk",
        type=int,
        default=1024,
        help="Max query tokens per TTEB attention chunk (lower = less VRAM).",
    )
    p.add_argument(
        "--tteb_attn_low_res_max",
        type=int,
        default=4096,
        help="If H*W exceeds this, TTEB attention runs at 64x64 then upsamples (4096=64^2).",
    )
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

    log_main(rank, f"Rank {rank}/{world_size} local_rank={local_rank} — building dataloaders...")
    train_loader, val_loader, train_sampler, val_sampler = build_bijie_dataloaders(
        args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        resize_to=args.resize_to,
        distributed=distributed,
        fm_backbone=args.fm_backbone,
    )

    log_main(
        rank,
        f"Building GeoPhysicsLandslideNet (fm_backbone={args.fm_backbone}, "
        f"decoder={args.decoder}, fusion={args.fusion})...",
    )
    model = GeoPhysicsLandslideNet(
        channels=channels,
        n_classes=1,
        lora_rank=args.lora_rank,
        prithvi_snapshot=args.prithvi_snapshot,
        fm_backbone=args.fm_backbone,
        efficientnet_name=args.efficientnet_name,
        efficientnet_pretrained=not args.no_efficientnet_pretrained,
        freeze_efficientnet=not args.unfreeze_efficientnet,
        decoder_type=args.decoder,
        fusion_type=args.fusion,
        tteb_attn_chunk=args.tteb_attn_chunk,
        tteb_attn_low_res_max=args.tteb_attn_low_res_max,
    )
    log_main(rank, "Base model constructed.")

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
        f"Ready: distributed={distributed} world_size={world_size} "
        f"per_gpu_batch={args.batch_size} global_batch={args.batch_size * world_size} "
        f"fm_backbone={args.fm_backbone} decoder={args.decoder} fusion={args.fusion} "
        f"freeze_efficientnet={not args.unfreeze_efficientnet} "
        f"fsdp={use_fsdp} resize_to={args.resize_to} "
        f"channels={channels} "
        f"full_precision={args.full_precision} "
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
            extra_final={
                "dataset": "bijie",
                "dataset_root": args.dataset_root,
                "fm_backbone": args.fm_backbone,
                "efficientnet_name": args.efficientnet_name,
                "decoder": args.decoder,
                "fusion": args.fusion,
                "freeze_efficientnet": not args.unfreeze_efficientnet,
            },
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
