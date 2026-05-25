from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from tqdm import tqdm

import torch

# ---------------------------------------------------------------------------
# Locate codebase root from this file's location
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent          # common/
_TRAINING_DIR = _FILE_DIR.parent                     # training/
_SCOUT_MODEL_DIR = _TRAINING_DIR.parent              # scout_model/
CODEBASE = _SCOUT_MODEL_DIR.parent.parent            # codebase/
assert CODEBASE.name == "codebase", (
    f"Expected 'codebase' as parent of scout_model, got {CODEBASE.name}"
)

# Baseline dataset split functions
BASELINE_DATASETS = str(CODEBASE / "ablation_study" / "baseline_models" / "common" / "datasets.py")


# ---------------------------------------------------------------------------
# Imports that depend on sys.path manipulation
# ---------------------------------------------------------------------------
def _import_scout_modules():
    """Insert scout_model/ into sys.path and import Scout/Critic/losses."""
    sm = str(_SCOUT_MODEL_DIR)
    if sm not in sys.path:
        sys.path.insert(0, sm)

    from model.wgan import Scout, Critic       # noqa: F811
    from losses.losses import (
        ReconstructionLoss,
        gradient_penalty,
        critic_loss,
        generator_loss,
    )
    return Scout, Critic, ReconstructionLoss, gradient_penalty, critic_loss, generator_loss


def _import_baseline_splits():
    """Import build_bijie_split and build_l4s_split from baseline datasets.py."""
    spec = importlib.util.spec_from_file_location("baseline_datasets", BASELINE_DATASETS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_bijie_split, mod.build_l4s_split


def _build_bijie_scout_split(dataset_root: str | Path, seed: int = 42, val_ratio: float = 0.2):
    """
    SCOUT+ split for Bijie:
      - Train on non-landslide only (normal-mountain reconstruction).
      - Validate on non-landslide only (reconstruction quality on normal terrain).
      - Test on all landslide (surprise factor vs ground-truth masks during forward pass).
    """
    spec = importlib.util.spec_from_file_location("baseline_datasets", BASELINE_DATASETS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    BijieRawDataset = mod.BijieRawDataset

    root = Path(dataset_root)
    if not (root / "landslide").exists() and (root / "Bijie-landslide-dataset").exists():
        root = root / "Bijie-landslide-dataset"

    nonlandslide = BijieRawDataset(root / "non-landslide", phase="non-landslide")
    landslide = BijieRawDataset(root / "landslide", phase="landslide")
    if len(landslide) == 0 or len(nonlandslide) == 0:
        raise ValueError(f"Empty Bijie dataset at {root}")

    g = torch.Generator().manual_seed(seed)
    n = len(nonlandslide)
    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1)
    perm = torch.randperm(n, generator=g).tolist()

    train_raw = torch.utils.data.Subset(nonlandslide, perm[n_val:])
    val_raw = torch.utils.data.Subset(nonlandslide, perm[:n_val])

    return train_raw, val_raw, landslide


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scout WGAN-GP training")
    parser.add_argument("--dataset", type=str, default="bijie", choices=["bijie", "l4s"])
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--run_id", type=str, default="scout")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resize_to", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--n_critic", type=int, default=5, help="critic steps per generator step")
    parser.add_argument("--recon_weight", type=float, default=100.0)
    parser.add_argument("--adv_weight", type=float, default=1.0)
    parser.add_argument("--lambda_gp", type=float, default=20.0)
    parser.add_argument("--metric_threshold", type=float, default=0.5)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path or auto-detect")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_scout(args: argparse.Namespace) -> dict[str, float]:
    from common.trainer import set_seed, train_scout_model
    from common.datasets import ScoutAugment2D, ScoutBijieDataset, ScoutL4SDataset

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Device: {device}")

    # ---- Datasets ----------------------------------------------------------
    _, build_l4s_split = _import_baseline_splits()

    transform = ScoutAugment2D(p=0.5)

    if args.dataset == "bijie":
        base_train, base_val, _ = _build_bijie_scout_split(args.dataset_root, seed=args.seed)
        train_ds = ScoutBijieDataset(base_train, resize_to=args.resize_to, transform=transform)
        val_ds = ScoutBijieDataset(base_val, resize_to=args.resize_to, transform=None)
    elif args.dataset == "l4s":
        train_ids, val_ids = build_l4s_split(args.dataset_root, args.val_ratio, seed=args.seed)
        train_ds = ScoutL4SDataset(args.dataset_root, train_ids, resize_to=args.resize_to, transform=transform)
        val_ds = ScoutL4SDataset(args.dataset_root, val_ids, resize_to=args.resize_to, transform=None)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Models ------------------------------------------------------------
    Scout, Critic, ReconstructionLoss, gradient_penalty, critic_loss, generator_loss = (
        _import_scout_modules()
    )

    generator = Scout(base_ch=args.base_ch).to(device)
    critic = Critic(base_ch=args.base_ch).to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    c_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.999))

    recon_loss_fn = ReconstructionLoss()

    tqdm.write(
        f"Generator: {sum(p.numel() for p in generator.parameters()):,} params\n"
        f"Critic:    {sum(p.numel() for p in critic.parameters()):,} params"
    )

    # ---- Train -------------------------------------------------------------
    final = train_scout_model(
        generator=generator,
        critic=critic,
        train_loader=train_loader,
        val_loader=val_loader,
        g_optimizer=g_optimizer,
        c_optimizer=c_optimizer,
        recon_loss_fn=recon_loss_fn,
        gradient_penalty_fn=gradient_penalty,
        critic_loss_fn=critic_loss,
        generator_loss_fn=generator_loss,
        device=device,
        num_epochs=args.epochs,
        n_critic=args.n_critic,
        recon_weight=args.recon_weight,
        adv_weight=args.adv_weight,
        lambda_gp=args.lambda_gp,
        metric_threshold=args.metric_threshold,
        output_dir=args.output_dir,
        run_id=args.run_id,
        save_every=args.save_every,
        resume=args.resume,
    )

    # ---- Landslide evaluation (only for bijie) -----------------------------
    landslide_metrics: dict[str, float] = {}
    if args.dataset == "bijie":
        _, _, landslide_ds = _build_bijie_scout_split(args.dataset_root, seed=args.seed)
        test_loader = torch.utils.data.DataLoader(
            ScoutBijieDataset(landslide_ds, resize_to=args.resize_to, transform=None),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        from common.trainer import evaluate_on_landslide

        landslide_metrics = evaluate_on_landslide(
            generator, test_loader, device, threshold=args.metric_threshold
        )
        for k, v in landslide_metrics.items():
            tqdm.write(f"  {k}: {v:.4f}")

    tqdm.write(f"Training complete. Best val F1: {final['val_f1']:.4f}")
    return {**final, **landslide_metrics}
