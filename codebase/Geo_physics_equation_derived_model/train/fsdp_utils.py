"""FSDP wrapping for GeoPhysicsLandslideNet."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from ..bridge.tteb import TriTemporalTriStreamBridge
from ..decoder.physics_decoder import PhysicsDecoder
from ..encoders.physics_encoder import PhysicsEncoder
from ..encoders.efficientnet_fm import EfficientNetFoundationEncoder
from ..encoders.prithvi_lora import PrithviFoundationEncoder
from ..fusion.mao_geo_egca import MAOGeoEGCA
from ..model import GeoPhysicsLandslideNet


def _checkpoint_policy(module: nn.Module) -> bool:
    return isinstance(
        module,
        (TriTemporalTriStreamBridge, MAOGeoEGCA, PrithviFoundationEncoder, EfficientNetFoundationEncoder),
    )


def wrap_geo_physics_fsdp(
    model: GeoPhysicsLandslideNet,
    device_id: int,
    *,
    use_bf16: bool = True,
    activation_checkpointing: bool = True,
) -> FSDP:
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print("[FSDP] Sharding model across GPUs (may take 1–3 min)...", flush=True)

    auto_wrap_policy = ModuleWrapPolicy(
        {
            TriTemporalTriStreamBridge,
            MAOGeoEGCA,
            PhysicsEncoder,
            PrithviFoundationEncoder,
            EfficientNetFoundationEncoder,
            PhysicsDecoder,
        }
    )
    mixed_precision = None
    if use_bf16:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    fsdp_model = FSDP(
        model,
        device_id=device_id,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        use_orig_params=True,
        sync_module_states=True,
    )

    if activation_checkpointing:
        if rank == 0:
            print("[FSDP] Applying activation checkpointing on heavy blocks...", flush=True)
        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=_checkpoint_policy,
        )
    if rank == 0:
        print("[FSDP] Model ready.", flush=True)
    return fsdp_model
