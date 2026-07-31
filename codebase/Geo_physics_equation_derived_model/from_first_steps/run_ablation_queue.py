#!/usr/bin/env python3
"""Sequential ablation queue for PS-GPLNet on GPU 1 (12GB), compact mode.

Runs variants from ABLATION_PLAN.md, writes status + summary CSVs, and is safe to resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "outputs_ablation"
STATUS = OUT_ROOT / "ABLATION_STATUS.md"
SUMMARY = OUT_ROOT / "ABLATION_SUMMARY.csv"

PRITHVI = (
    "/home/user/Desktop/Deep_learning_projects/4PI/prithvi_weights/"
    "models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL"
)
BIJIE = "/home/user/Desktop/Deep_learning_projects/4PI/dataset_bijie_landslide/Bijie-landslide-dataset"
L4S = "/home/user/Desktop/Deep_learning_projects/4PI/dataset"

# Expected ballparks for FULL compact runs (monitor health)
ANCHORS = {
    "bijie": {"f1_min_ok": 0.85, "f1_target": 0.91},
    "l4s": {"f1_min_ok": 0.60, "f1_target": 0.70},
}

VARIANTS = [
    ("full", {}),
    ("no_fs_gate", {"no_mechanistic_gating": True}),
    ("no_mpef", {"mpef_mode": "mean"}),
    ("no_cmb", {"no_cmb": True}),
    ("no_mao", {"no_mao": True}),
    ("no_tteb", {"no_tteb": True}),
    ("no_prithvi", {"no_prithvi": True}),
    ("path_a_only", {"path_mode": "path_a"}),
]


def best_f1_from_csv(path: Path) -> float | None:
    if not path.exists():
        return None
    best = None
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                v = float(row.get("val_f1") or row.get("f1") or "nan")
            except ValueError:
                continue
            if v != v:  # NaN
                continue
            best = v if best is None else max(best, v)
    return best


def write_status(lines: list[str]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text("\n".join(lines) + "\n")


def append_summary(row: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    new = not SUMMARY.exists()
    with open(SUMMARY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def build_cmd(dataset: str, variant: str, flags: dict, epochs: int, batch: int) -> list[str]:
    out = OUT_ROOT / dataset / variant
    common = [
        "conda",
        "run",
        "-n",
        "deeplearning",
        "--no-capture-output",
        "python",
        "-u",
    ]
    if dataset == "bijie":
        script = str(ROOT / "train_bijie.py")
        cmd = common + [
            script,
            "--dataset_root",
            BIJIE,
            "--output_dir",
            str(out),
            "--prithvi_snapshot",
            PRITHVI,
            "--compact",
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch),
            "--num_workers",
            "4",
            "--device",
            "cuda:1",
            "--no-auto_gpu",
            "--min_free_gb",
            "0.5",
            "--seed",
            "42",
            "--tversky_alpha",
            "0.7",
            "--tversky_beta",
            "0.3",
            "--save_every",
            "5",
            "--resume",
        ]
    else:
        script = str(ROOT / "training.py")
        cmd = common + [
            script,
            "--dataset_root",
            L4S,
            "--output_dir",
            str(out),
            "--prithvi_snapshot",
            PRITHVI,
            "--compact",
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch),
            "--num_workers",
            "4",
            "--device",
            "cuda:1",
            "--no-auto_gpu",
            "--min_free_gb",
            "0.5",
            "--seed",
            "42",
            "--save_every",
            "5",
            "--resume",
        ]

    if flags.get("no_mechanistic_gating"):
        cmd.append("--no_mechanistic_gating")
    if flags.get("no_cmb"):
        cmd.append("--no_cmb")
    if flags.get("no_mao"):
        cmd.append("--no_mao")
    if flags.get("no_tteb"):
        cmd.append("--no_tteb")
    if flags.get("no_prithvi"):
        cmd.append("--no_prithvi")
    if "mpef_mode" in flags:
        cmd += ["--mpef_mode", flags["mpef_mode"]]
    if "path_mode" in flags:
        cmd += ["--path_mode", flags["path_mode"]]
    return cmd


def variant_done(dataset: str, variant: str, epochs: int) -> bool:
    csv_path = OUT_ROOT / dataset / variant / "results" / "epoch_metrics.csv"
    final = OUT_ROOT / dataset / variant / "results" / "final_metrics.csv"
    if final.exists():
        return True
    if not csv_path.exists():
        return False
    # Count finished epochs
    with open(csv_path) as f:
        n = sum(1 for _ in csv.DictReader(f))
    return n >= epochs


def run_one(dataset: str, variant: str, flags: dict, epochs: int, batch: int) -> int:
    out = OUT_ROOT / dataset / variant
    out.mkdir(parents=True, exist_ok=True)
    log = out / "train.log"
    cmd = build_cmd(dataset, variant, flags, epochs, batch)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    # When CUDA_VISIBLE_DEVICES=1, the only visible GPU is cuda:0 inside the process.
    # Rewrite device to cuda:0 for the child.
    cmd = [c if c != "cuda:1" else "cuda:0" for c in cmd]

    with open(log, "a") as lf:
        lf.write(f"\n==== START {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        lf.write("CMD: " + " ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def refresh_summary(datasets: list[str]) -> None:
    rows = []
    for ds in datasets:
        for name, _ in VARIANTS:
            csv_path = OUT_ROOT / ds / name / "results" / "epoch_metrics.csv"
            bf = best_f1_from_csv(csv_path)
            rows.append(
                {
                    "dataset": ds,
                    "variant": name,
                    "best_val_f1": "" if bf is None else f"{bf:.6f}",
                    "status": "done" if (OUT_ROOT / ds / name / "results" / "final_metrics.csv").exists() else (
                        "running_or_partial" if bf is not None else "pending"
                    ),
                }
            )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "variant", "best_val_f1", "status"])
        w.writeheader()
        w.writerows(rows)

    lines = [
        f"# Ablation status ({time.strftime('%Y-%m-%d %H:%M:%S')})",
        "",
        "Anchors: Bijie full F1≳0.91 (ok>0.85); L4S full F1≳0.70 (ok>0.60).",
        "",
        "| Dataset | Variant | Best val F1 | Status |",
        "|---------|---------|-------------|--------|",
    ]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['variant']} | {r['best_val_f1'] or '—'} | {r['status']} |")
    write_status(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default="bijie", help="Comma list: bijie,l4s")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--only", type=str, default="", help="Comma list of variant names")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None

    refresh_summary(datasets)
    for ds in datasets:
        for name, flags in VARIANTS:
            if only is not None and name not in only:
                continue
            if variant_done(ds, name, args.epochs):
                print(f"[skip] {ds}/{name} already finished ≥{args.epochs} epochs")
                continue
            print(f"[run] {ds}/{name}")
            if args.dry_run:
                print(" ", " ".join(build_cmd(ds, name, flags, args.epochs, args.batch_size)))
                continue
            rc = run_one(ds, name, flags, args.epochs, args.batch_size)
            refresh_summary(datasets)
            bf = best_f1_from_csv(OUT_ROOT / ds / name / "results" / "epoch_metrics.csv")
            print(f"[done] {ds}/{name} rc={rc} best_f1={bf}")
            if name == "full" and bf is not None:
                mn = ANCHORS[ds]["f1_min_ok"]
                if bf < mn:
                    print(
                        f"WARNING: full {ds} F1={bf:.3f} < {mn} — check training health before trusting ablations."
                    )
                    # Continue anyway; user can inspect logs.
            if rc != 0:
                print(f"ERROR: {ds}/{name} failed with code {rc}; see train.log")
                return rc
    refresh_summary(datasets)
    print("Ablation queue finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
