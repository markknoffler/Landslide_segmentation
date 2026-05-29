"""Rank-0 console logging with flush (works under torchrun / SLURM)."""

from __future__ import annotations

import sys
from typing import Any


def log_main(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def log_dict(rank: int, title: str, data: dict[str, Any]) -> None:
    if rank != 0:
        return
    print(f"\n=== {title} ===", flush=True)
    for key, value in data.items():
        print(f"  {key}: {value}", flush=True)
    print("", flush=True)


def tqdm_file():
    return sys.stderr
