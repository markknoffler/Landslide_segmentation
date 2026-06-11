"""CUDA device selection and memory diagnostics for single-GPU training."""

from __future__ import annotations

import subprocess
from typing import List, Optional, Tuple

import torch


def gpu_memory_table() -> List[Tuple[int, float, float]]:
    if not torch.cuda.is_available():
        return []
    rows: List[Tuple[int, float, float]] = []
    for idx in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(idx)
        rows.append((idx, free_b / (1024**3), total_b / (1024**3)))
    return rows


def nvidia_smi_processes() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,gpu_bus_id,pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return out.strip() or "(no compute processes reported)"
    except Exception as exc:
        return f"(nvidia-smi unavailable: {exc})"


def log_gpu_memory(prefix: str = "") -> None:
    if not torch.cuda.is_available():
        print(f"{prefix}CUDA not available")
        return
    print(f"{prefix}Visible CUDA devices: {torch.cuda.device_count()}")
    for idx, free_gb, total_gb in gpu_memory_table():
        alloc_mb = torch.cuda.memory_allocated(idx) / (1024**2)
        print(
            f"{prefix}  cuda:{idx} free={free_gb:.2f} GiB / total={total_gb:.2f} GiB "
            f"| torch allocated={alloc_mb:.1f} MiB"
        )
    print(f"{prefix}nvidia-smi compute apps:\n{nvidia_smi_processes()}")


def select_best_cuda_device(min_free_gb: float = 4.0) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_idx: Optional[int] = None
    best_free = -1.0
    for idx, free_gb, _ in gpu_memory_table():
        if free_gb > best_free:
            best_free = free_gb
            best_idx = idx

    if best_idx is None:
        return torch.device("cpu")

    if best_free < min_free_gb:
        log_gpu_memory(prefix="[GPU] ")
        raise RuntimeError(
            f"No CUDA device has >= {min_free_gb:.1f} GiB free. "
            f"Best device cuda:{best_idx} has only {best_free:.2f} GiB free. "
            f"Kill stale GPU jobs (`nvidia-smi` then `kill <pid>`) or request a fresh node."
        )

    if best_idx != 0:
        print(f"Auto-selected cuda:{best_idx} ({best_free:.2f} GiB free)")
    return torch.device(f"cuda:{best_idx}")


def resolve_device(requested: str, *, auto_select: bool, min_free_gb: float) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    if requested == "cuda" and auto_select:
        return select_best_cuda_device(min_free_gb=min_free_gb)

    device = torch.device(requested if requested != "cuda" else "cuda:0")
    if device.type != "cuda":
        return device

    idx = device.index if device.index is not None else torch.cuda.current_device()
    free_b, total_b = torch.cuda.mem_get_info(idx)
    free_gb = free_b / (1024**3)
    total_gb = total_b / (1024**3)
    if free_gb < min_free_gb:
        log_gpu_memory(prefix="[GPU] ")
        raise RuntimeError(
            f"CUDA device cuda:{idx} has only {free_gb:.2f} GiB free of {total_gb:.2f} GiB total. "
            f"GeoPhysicsLandslideNet needs ~4+ GiB free before loading (~0.5 GiB weights + activations). "
            f"Kill stale jobs or enable --auto_gpu if another visible GPU is free."
        )
    return device
