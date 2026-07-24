# Final results & checkpoints index (`from_first_steps/`)

## Canonical final **metrics** (cluster training — CSVs only, no `.pt` synced)

These are the official numbers used in the paper tables (F1 / IoU).

| Role | Path | Dataset | Best epoch | Best val F1 | Best val IoU |
|------|------|---------|------------|-------------|--------------|
| **FINAL Bijie** | `outputs_absolute_final_fully_novel_complete/` | Bijie | **92** | **0.933** | **0.907** |
| alias | `outputs_final_bijie` → symlink to above | same | same | same | same |
| **FINAL L4S** | `outputs_step3_l4s/` | Landslide4Sense | **70** | **0.736** | **0.660** |
| alias | `outputs_final_l4s` → symlink to above | same | same | same | same |

Notes:
- Bijie CSV has **200** rows = failed NaN run (epochs 1–100 stuck F1≈0.723) **concatenated with** the successful run (best at epoch 92). Use the finite-loss rows / `final_metrics.csv` second line (`0.933…`).
- L4S `final_metrics.csv` records full PS-GPLNet flags: `fusion=mao`, `decoder=physics`, `physics_encoders=True`, `prithvi=True`, backbone **B4**.
- Bijie `final_metrics.csv` lists backbone **B0** (compact) for the 0.933 run — that is what was logged on the cluster for that job.
- **Weights (`.pt`) for these runs were never copied off the cluster.** Only `results/*.csv` exist locally.

## Local retrain (weights present — for TPAMI inference studies only)

| Path | Purpose | Best epoch | Best val F1 | Best val IoU |
|------|---------|------------|-------------|--------------|
| `outputs_tpami_local_bijie/` | Architecture-faithful local B4+Prithvi retrain so t-SNE / BF / robustness could run | **98** | **0.912** | **0.886** |

- Checkpoints: `outputs_tpami_local_bijie/checkpoint/best.pt` (+ `epoch_*.pt`).
- **Do not** replace paper main-table Bijie 0.933 with 0.912. Table = cluster CSV; studies = local weights.

## Active top-level dirs (keep clean)

| Path | Role |
|------|------|
| `outputs_absolute_final_fully_novel_complete/` | **FINAL Bijie metrics** |
| `outputs_final_bijie` | symlink → absolute_final… |
| `outputs_step3_l4s/` | **FINAL L4S metrics** |
| `outputs_final_l4s` | symlink → step3_l4s |
| `outputs_tpami_local_bijie/` | local retrain weights for studies |
| `outputs_tpami_studies_final/` | current t-SNE / boundary / robustness |
| `tpami_assets/` | theory probe figures/logs |
| `prev_legacy_results/` | everything else (older steps, legacy ckpts, interim studies) |

## TPAMI study outputs

| Path | Checkpoint used | Contents |
|------|-----------------|----------|
| `outputs_tpami_studies_final/` | local `best.pt` (epoch 98, F1≈0.912) | **current** study JSON + figures |
| `prev_legacy_results/outputs_tpami_studies_interim/` | early local (epoch 4, F1≈0.861) | superseded; stashed |
| Paper figures | `../../project_report/gplnet_paper/figures/tpami/` | used by `paper_tpami_v2.pdf` |

## What not to use

| Path | Why |
|------|-----|
| `prev_legacy_results/**/best.pt` | Older DualStreamGateNet (no physics/MPEF), F1≈0.897 — wrong architecture |
| First 100 rows of Bijie absolute_final CSV | Dead NaN run (F1 stuck ≈0.723) |
| `prev_legacy_results/outputs_tpami_studies_interim/` | Early study run; use `outputs_tpami_studies_final/` |

## Paper mapping

- Main results table (0.933 / 0.736): from **canonical CSVs** above.
- Theory probes: `tpami_assets/` + `gplnet_paper/assets/tpami_analysis_log.json`.
- Latent / boundary / robustness: `outputs_tpami_studies_final/` + `paper_tpami.tex` §Experiments (local checkpoint, disclosed).
- Manuscript PDF: `../../project_report/gplnet_paper/paper_tpami_v2.pdf` (does not overwrite old `paper.pdf`).
