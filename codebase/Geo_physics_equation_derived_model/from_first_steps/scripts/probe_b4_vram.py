#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from losses import DualStreamLoss
from model import GeoPhysicsLandslideNet

PRITHVI = (
    "/home/user/Desktop/Deep_learning_projects/4PI/prithvi_weights/"
    "models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL"
)


def main() -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(0)
    print(f"free={free/1e9:.2f}/{total/1e9:.2f}")
    model = GeoPhysicsLandslideNet(
        backbone="tf_efficientnet_b4",
        pretrained=True,
        freeze_backbone=True,
        prithvi_snapshot=PRITHVI,
        lora_rank=8,
        fusion_channels=64,
    ).to(device)
    print(f"weights={torch.cuda.memory_allocated()/1e9:.2f}")
    crit = DualStreamLoss()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    for bs in (1, 2):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        try:
            x1 = torch.randn(bs, 3, 256, 256, device=device)
            x2 = torch.randn(bs, 3, 256, 256, device=device)
            y = torch.randint(0, 2, (bs, 1, 256, 256), device=device).float()
            main, a2, a3, reg = model(x1, x2)
            loss = crit(main, a2, a3, reg, y)["loss"]
            loss.backward()
            opt.step()
            print(f"bs={bs} peak={torch.cuda.max_memory_allocated()/1e9:.2f}")
        except Exception as exc:
            print(f"bs={bs} FAIL {exc}")
            break


if __name__ == "__main__":
    main()
