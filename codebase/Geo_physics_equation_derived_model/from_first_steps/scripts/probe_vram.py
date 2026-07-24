#!/usr/bin/env python3
"""Probe compact PS-GPLNet VRAM for forward/backward on the freest GPU."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from losses import DualStreamLoss  # noqa: E402
from model import GeoPhysicsLandslideNet  # noqa: E402

PRITHVI = (
    "/home/user/Desktop/Deep_learning_projects/4PI/prithvi_weights/"
    "models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL"
)


def main() -> None:
    best_i, best_free = 0, -1.0
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"cuda:{i} free={free/1e9:.2f}GB / {total/1e9:.2f}GB")
        if free > best_free:
            best_free, best_i = free, i

    device = torch.device(f"cuda:{best_i}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    print("using", device)

    print("Building compact model on CPU...")
    model = GeoPhysicsLandslideNet(
        backbone="tf_efficientnet_b0",
        pretrained=True,
        freeze_backbone=True,
        prithvi_snapshot=PRITHVI,
        lora_rank=4,
        fusion_channels=32,
        tteb_attn_chunk=512,
        tteb_attn_low_res_max=1024,
    )
    print("Moving to GPU...")
    model = model.to(device)
    torch.cuda.synchronize()
    print(f"weights allocated {torch.cuda.memory_allocated(device)/1e9:.2f} GB")

    x1 = torch.randn(1, 3, 256, 256, device=device)
    x2 = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        out = model(x1, x2)
    print("forward ok", [t.shape for t in out[:3]])
    print(
        f"after fwd alloc={torch.cuda.memory_allocated(device)/1e9:.2f} "
        f"peak={torch.cuda.max_memory_allocated(device)/1e9:.2f}"
    )

    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    crit = DualStreamLoss()
    for bs in (1, 2, 4):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        try:
            x1 = torch.randn(bs, 3, 256, 256, device=device)
            x2 = torch.randn(bs, 3, 256, 256, device=device)
            y = torch.randint(0, 2, (bs, 1, 256, 256), device=device).float()
            opt.zero_grad(set_to_none=True)
            main, a2, a3, reg = model(x1, x2)
            loss = crit(main, a2, a3, reg, y)["loss"]
            loss.backward()
            opt.step()
            print(
                f"train bs={bs} ok loss={float(loss):.4f} "
                f"peak={torch.cuda.max_memory_allocated(device)/1e9:.2f}GB"
            )
        except RuntimeError as exc:
            print(f"train bs={bs} FAILED: {exc}")
            break


if __name__ == "__main__":
    main()
