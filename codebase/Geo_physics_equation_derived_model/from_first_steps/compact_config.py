"""GPU-friendly width/backbone presets — architecture unchanged (dual decoder kept)."""

from __future__ import annotations

import argparse


def add_compact_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Smaller footprint without changing architecture: EfficientNet-B0, "
            "fusion C=32, LoRA r=4, lighter TTEB. Dual physics decoders unchanged."
        ),
    )
    parser.add_argument("--fusion_channels", type=int, default=64)


def apply_compact_preset(args: argparse.Namespace) -> argparse.Namespace:
    if not args.compact:
        return args
    args.backbone = "tf_efficientnet_b0"
    args.fusion_channels = 32
    args.lora_rank = min(args.lora_rank, 4)
    args.tteb_attn_chunk = min(args.tteb_attn_chunk, 512)
    args.tteb_attn_low_res_max = min(args.tteb_attn_low_res_max, 1024)
    return args
