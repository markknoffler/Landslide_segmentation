"""GPU-friendly model size presets (all modules stay on GPU)."""

from __future__ import annotations

import argparse


def add_compact_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Smaller GPU footprint: EfficientNet-B0, fusion C=32, LoRA r=4, "
            "shared physics decoder, fp16 frozen encoders, lighter TTEB."
        ),
    )
    parser.add_argument("--fusion_channels", type=int, default=64)
    parser.add_argument("--shared_physics_decoder", action="store_true", default=False)
    parser.add_argument("--fp16_frozen_encoders", action="store_true", default=False)


def apply_compact_preset(args: argparse.Namespace) -> argparse.Namespace:
    if not args.compact:
        return args
    args.backbone = "tf_efficientnet_b0"
    args.fusion_channels = 32
    args.lora_rank = min(args.lora_rank, 4)
    args.shared_physics_decoder = True
    args.fp16_frozen_encoders = True
    args.tteb_attn_chunk = min(args.tteb_attn_chunk, 512)
    args.tteb_attn_low_res_max = min(args.tteb_attn_low_res_max, 1024)
    return args
