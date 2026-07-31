import argparse
import csv
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from compact_config import add_compact_args, apply_compact_preset
from dataset import DualStreamTransform, Landslide4SenseDualStream
from device_utils import log_gpu_memory, resolve_device
from losses import DualStreamLoss
from metrics import pixel_metrics_from_logits
from model import GeoPhysicsLandslideNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PS-GPLNet (penta-stream + dual physics decoder) on Landslide4Sense."
    )
    parser.add_argument("--dataset_root", type=str, default="/home/user/Desktop/Deep_learning_projects/4PI/dataset")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--auto_gpu", action="store_true", default=True)
    parser.add_argument("--no-auto_gpu", dest="auto_gpu", action="store_false")
    parser.add_argument("--min_free_gb", type=float, default=4.0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    add_compact_args(parser)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize_to", type=int, default=256)
    parser.add_argument("--bands", type=str, default="RGB-NDVI-SLOPE-DEM")
    parser.add_argument("--backbone", type=str, default="tf_efficientnet_b4")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze_backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--share_backbone", action="store_true", default=False)
    parser.add_argument("--no-share_backbone", dest="share_backbone", action="store_false")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Optional local EfficientNet checkpoint. Omit to use timm download with --pretrained.",
    )
    parser.add_argument(
        "--prithvi_snapshot",
        type=str,
        required=True,
        help=(
            "Required. Prithvi HF cache root "
            "(.../models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL) or snapshots/<hash>/."
        ),
    )
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--tteb_attn_chunk", type=int, default=1024)
    parser.add_argument("--tteb_attn_low_res_max", type=int, default=4096)
    parser.add_argument(
        "--no_mechanistic_gating",
        action="store_true",
        help="Disable FS-based gating inside physics cells (ablation).",
    )
    parser.add_argument("--no_cmb", action="store_true")
    parser.add_argument("--no_mao", action="store_true")
    parser.add_argument("--no_tteb", action="store_true")
    parser.add_argument("--no_prithvi", action="store_true")
    parser.add_argument("--mpef_mode", type=str, default="mpef", choices=["mpef", "mean"])
    parser.add_argument("--path_mode", type=str, default="dual", choices=["dual", "path_a"])
    parser.add_argument("--use_input_adapter", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--val_split_ratio", type=float, default=0.1)
    parser.add_argument("--val_split_seed", type=int, default=42)
    parser.add_argument("--tversky_alpha", type=float, default=0.3)
    parser.add_argument("--tversky_beta", type=float, default=0.7)
    parser.add_argument("--main_weight", type=float, default=1.0)
    parser.add_argument("--aux2_weight", type=float, default=0.6)
    parser.add_argument("--aux3_weight", type=float, default=0.4)
    parser.add_argument("--reg_weight", type=float, default=1e-3)
    parser.add_argument("--metric_threshold", type=float, default=0.5)
    return parser.parse_args()


def set_seed(seed: int, *, cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataloader_worker_init_fn(base_seed: int):
    def _init(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init


def _loader_kwargs(args) -> Dict:
    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs["worker_init_fn"] = dataloader_worker_init_fn(args.seed)
    return kwargs


def latest_checkpoint(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def save_checkpoint(path: Path, state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def prep_batch(batch, device: torch.device):
    x1 = batch["stream_a"].float().to(device, non_blocking=True)
    x2 = batch["stream_b"].float().to(device, non_blocking=True)
    y = batch["mask"].float()
    if y.dtype.is_floating_point:
        y = y.round().long()
    y = y.to(device, non_blocking=True)
    if y.dim() == 3:
        y = y.unsqueeze(1)
    return x1, x2, y


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    threshold: float,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
    grad_clip: float,
):
    model.train()
    losses = []
    loss_parts = {"main": [], "aux2": [], "aux3": [], "reg": []}
    metrics = {"acc": [], "precision": [], "recall": [], "f1": [], "iou": []}

    for batch in tqdm(loader, desc="Train", leave=False):
        x1, x2, y = prep_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            main, aux2, aux3, reg_tuple = model(x1, x2)
            loss_dict = criterion(main, aux2, aux3, reg_tuple, y)
            loss = loss_dict["loss"]

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss — skipping optimizer step for this batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses.append(float(loss.item()))
        loss_parts["main"].append(float(loss_dict["loss_main"].item()))
        loss_parts["aux2"].append(float(loss_dict["loss_aux2"].item()))
        loss_parts["aux3"].append(float(loss_dict["loss_aux3"].item()))
        loss_parts["reg"].append(float(loss_dict["loss_reg"].item()))

        pix = pixel_metrics_from_logits(main, y, threshold=threshold)
        for key in metrics:
            metrics[key].append(float(pix[key]))

    out = {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}
    out["loss"] = float(np.mean(losses)) if losses else 0.0
    for key, values in loss_parts.items():
        out[f"loss_{key}"] = float(np.mean(values)) if values else 0.0
    return out


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device, threshold: float, use_amp: bool):
    model.eval()
    losses = []
    loss_parts = {"main": [], "aux2": [], "aux3": [], "reg": []}
    metrics = {"acc": [], "precision": [], "recall": [], "f1": [], "iou": []}

    for batch in tqdm(loader, desc="Val", leave=False):
        x1, x2, y = prep_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            main, aux2, aux3, reg_tuple = model(x1, x2)
            loss_dict = criterion(main, aux2, aux3, reg_tuple, y)

        losses.append(float(loss_dict["loss"].item()))
        loss_parts["main"].append(float(loss_dict["loss_main"].item()))
        loss_parts["aux2"].append(float(loss_dict["loss_aux2"].item()))
        loss_parts["aux3"].append(float(loss_dict["loss_aux3"].item()))
        loss_parts["reg"].append(float(loss_dict["loss_reg"].item()))

        pix = pixel_metrics_from_logits(main, y, threshold=threshold)
        for key in metrics:
            metrics[key].append(float(pix[key]))

    out = {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}
    out["loss"] = float(np.mean(losses)) if losses else 0.0
    for key, values in loss_parts.items():
        out[f"loss_{key}"] = float(np.mean(values)) if values else 0.0
    return out


def build_dataloaders(args):
    train_ds = Landslide4SenseDualStream(
        data_root=args.dataset_root,
        split="train",
        bands=args.bands,
        resize_to=args.resize_to,
        transform=DualStreamTransform(p=0.5),
    )
    valid_ds = Landslide4SenseDualStream(
        data_root=args.dataset_root,
        split="valid",
        bands=args.bands,
        resize_to=args.resize_to,
        transform=None,
    )

    if valid_ds.has_mask:
        loader_kwargs = _loader_kwargs(args)
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kwargs)
        return train_loader, valid_loader

    val_ratio = float(args.val_split_ratio)
    n_total = len(train_ds)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_val = min(n_val, n_total - 1)
    generator = torch.Generator().manual_seed(args.val_split_seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    train_subset = Subset(train_ds, train_indices)
    valid_subset = Subset(
        Landslide4SenseDualStream(
            data_root=args.dataset_root,
            split="train",
            bands=args.bands,
            resize_to=args.resize_to,
            transform=None,
        ),
        val_indices,
    )

    loader_kwargs = _loader_kwargs(args)
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_subset, shuffle=False, **loader_kwargs)
    return train_loader, valid_loader


def main():
    args = apply_compact_preset(parse_args())
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    ckpt_dir = output_dir / "checkpoint"
    results_dir = output_dir / "results"
    epoch_csv = results_dir / "epoch_metrics.csv"
    final_csv = results_dir / "final_metrics.csv"

    train_loader, valid_loader = build_dataloaders(args)
    print(f"Landslide4Sense | train={len(train_loader.dataset)} val={len(valid_loader.dataset)} | PS-GPLNet")

    pretrained_path = args.pretrained_path
    if pretrained_path in (None, "", "None", "none"):
        pretrained_path = None

    if args.compact:
        print("Compact mode: B0 backbone, fusion C=32, LoRA r=4 (dual decoders unchanged).")

    print("Building GeoPhysicsLandslideNet on CPU...")
    model = GeoPhysicsLandslideNet(
        n_classes=1,
        backbone=args.backbone,
        n_channels=3,
        n_channels_b=3,
        pretrained=args.pretrained,
        pretrained_path=pretrained_path,
        use_input_adapter=args.use_input_adapter,
        freeze_backbone=args.freeze_backbone,
        share_backbone=args.share_backbone,
        prithvi_snapshot=args.prithvi_snapshot,
        lora_rank=args.lora_rank,
        fusion_channels=args.fusion_channels,
        mechanistic_gating=not args.no_mechanistic_gating,
        tteb_attn_chunk=args.tteb_attn_chunk,
        tteb_attn_low_res_max=args.tteb_attn_low_res_max,
        use_cmb=not args.no_cmb,
        use_mao=not args.no_mao,
        use_tteb=not args.no_tteb,
        use_prithvi=not args.no_prithvi,
        mpef_mode=args.mpef_mode,
        path_mode=args.path_mode,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model params: trainable={trainable/1e6:.2f}M total={total/1e6:.2f}M")

    min_free = args.min_free_gb
    device = resolve_device(
        args.device if torch.cuda.is_available() else "cpu",
        auto_select=args.auto_gpu,
        min_free_gb=min_free,
    )
    print(f"Training pid={os.getpid()} on {device} | amp={args.amp}")
    log_gpu_memory(prefix="[GPU] ")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Moving model to GPU...")
    model = model.to(device)
    set_seed(args.seed, cuda=True)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    criterion = DualStreamLoss(
        alpha=args.tversky_alpha,
        beta=args.tversky_beta,
        main_weight=args.main_weight,
        aux2_weight=args.aux2_weight,
        aux3_weight=args.aux3_weight,
        reg_weight=args.reg_weight,
    )
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_f1 = 0.0

    if args.resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state["epoch"]) + 1
            best_f1 = float(state.get("best_f1", 0.0))

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.metric_threshold,
            use_amp,
            scaler,
            args.grad_clip,
        )
        val_metrics = evaluate(model, valid_loader, criterion, device, args.metric_threshold, use_amp)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_loss_main": train_metrics.get("loss_main", 0.0),
            "train_loss_aux2": train_metrics.get("loss_aux2", 0.0),
            "train_loss_aux3": train_metrics.get("loss_aux3", 0.0),
            "train_loss_reg": train_metrics.get("loss_reg", 0.0),
            "train_acc": train_metrics["acc"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_loss_main": val_metrics.get("loss_main", 0.0),
            "val_loss_aux2": val_metrics.get("loss_aux2", 0.0),
            "val_loss_aux3": val_metrics.get("loss_aux3", 0.0),
            "val_loss_reg": val_metrics.get("loss_reg", 0.0),
            "val_acc": val_metrics["acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_iou": val_metrics["iou"],
        }
        append_csv(epoch_csv, row)
        print(row)
        print(
            f"  val loss parts: main={row['val_loss_main']:.3f} "
            f"aux2={row['val_loss_aux2']:.3f} aux3={row['val_loss_aux3']:.3f} "
            f"reg={row['val_loss_reg']:.5f} "
            f"(weighted sum={row['val_loss']:.3f})"
        )

        if epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:04d}.pt",
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_f1": best_f1},
            )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(
                ckpt_dir / "best.pt",
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_f1": best_f1},
            )

    append_csv(
        final_csv,
        {
            "best_val_f1": best_f1,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "backbone": args.backbone,
            "pretrained": args.pretrained,
            "freeze_backbone": args.freeze_backbone,
            "resize_to": args.resize_to,
            "bands": args.bands,
            "mechanistic_gating": not args.no_mechanistic_gating,
            "tversky_alpha": args.tversky_alpha,
            "tversky_beta": args.tversky_beta,
            "main_weight": args.main_weight,
            "aux2_weight": args.aux2_weight,
            "aux3_weight": args.aux3_weight,
            "reg_weight": args.reg_weight,
            "metric_threshold": args.metric_threshold,
            "dataset_root": args.dataset_root,
        },
    )


if __name__ == "__main__":
    main()
