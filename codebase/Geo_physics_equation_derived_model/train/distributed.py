"""Distributed training helpers (NCCL / torchrun)."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed(backend: str = "nccl") -> tuple[bool, int, int, int]:
    """Initialize process group when launched via torchrun. Returns (enabled, rank, world_size, local_rank)."""
    if not is_distributed():
        return False, 0, 1, 0

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    return rank == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / dist.get_world_size())


def all_reduce_sum_and_count(values: list[float], device: torch.device) -> float:
    if not values:
        return 0.0
    if not dist.is_available() or not dist.is_initialized():
        return float(sum(values) / len(values))
    total = float(sum(values))
    count = float(len(values))
    tensor = torch.tensor([total, count], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor[0] / tensor[1]) if tensor[1] > 0 else 0.0
