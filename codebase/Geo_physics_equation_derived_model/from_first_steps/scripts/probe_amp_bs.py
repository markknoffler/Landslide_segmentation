#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from losses import DualStreamLoss
from model import GeoPhysicsLandslideNet
PRITHVI = "/home/user/Desktop/Deep_learning_projects/4PI/prithvi_weights/models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL"
device = torch.device("cuda:0")
torch.cuda.empty_cache()
m = GeoPhysicsLandslideNet(backbone="tf_efficientnet_b4", pretrained=True, freeze_backbone=True,
    prithvi_snapshot=PRITHVI, lora_rank=8, fusion_channels=64).to(device)
crit = DualStreamLoss()
opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-4)
scaler = torch.amp.GradScaler("cuda")
for bs in (1, 2, 4):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    try:
        x1 = torch.randn(bs, 3, 256, 256, device=device)
        x2 = torch.randn(bs, 3, 256, 256, device=device)
        y = torch.randint(0, 2, (bs, 1, 256, 256), device=device).float()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=True):
            main, a2, a3, reg = m(x1, x2)
            loss = crit(main, a2, a3, reg, y)["loss"]
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        print(f"AMP bs={bs} OK peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    except RuntimeError as e:
        print(f"AMP bs={bs} FAIL {str(e).split(chr(10))[0]}")
        break
