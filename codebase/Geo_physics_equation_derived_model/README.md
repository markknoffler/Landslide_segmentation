# Geo-Physics Equation-Derived Landslide Segmentation

Three-stream architecture: physics-encoded RGB/DEM, foundation-model stream (EfficientNet-B4 or Prithvi), MAO-GeoEGCA fusion, TTEB skip bridge, and physics-gated decoder.

See [model_architecture.md](model_architecture.md) for full specification.

## Data contract (no synthetic bands)

**Bijie** on disk is RGB PNG + DEM PNG + mask only.

| Field | Source |
|-------|--------|
| stream_a | Measured RGB |
| dem / dem_norm | Measured DEM |
| slope_norm | Sobel on measured DEM |
| ndvi_norm | Green–red index from measured RGB (no zero-filled NDVI) |
| fm_input (`efficientnet`) | **Same measured RGB** as stream_a (3 channels) |
| fm_input (`prithvi`) | B, G, R, DEM, slope(DEM), veg_index(RGB) — 6 observed/derived channels |

Physics RGB/DEM encoders, MAO, TTEB, and decoder are **unchanged** regardless of `--fm_backbone`.

## Foundation-model backbone (`--fm_backbone`)

| Value | FM encoder | Batch `fm_input` | Extra setup on HPC |
|-------|------------|------------------|---------------------|
| **`efficientnet`** (default) | `timm` **tf_efficientnet_b4** (ImageNet), frozen by default | RGB `[0,1]` | `pip install timm` if missing |
| **`prithvi`** | Prithvi-EO ViT + LoRA | 6-channel stack | Prithvi snapshot + `download_prithvi.sh` |

## HPC: sync code and environment

On the cluster (from repo root):

```bash
cd /path/to/Landslide_segmentation
git pull   # or rsync your updated tree

conda activate deeplearning   # or your env name

# Required for default EfficientNet FM (same as dual_stream_gated)
pip install timm

# Optional: only if you use --fm_backbone prithvi
pip install huggingface_hub
bash codebase/Geo_physics_equation_derived_model/scripts/download_prithvi.sh
```

Ensure the gitignored `data/` package exists on the server (copy from local or use the same layout as dual-stream baselines).

## HPC: Bijie training (EfficientNet — recommended)

Single GPU:

```bash
cd /path/to/Landslide_segmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m codebase.Geo_physics_equation_derived_model.train.train_bijie \
  --dataset_root /scratch/earnest/samreedh/landslide_segmentation/dataset_bijie_landslide \
  --output_dir codebase/Geo_physics_equation_derived_model/outputs_bijie_effnet \
  --fm_backbone efficientnet \
  --resize_to 256 \
  --batch_size 4 \
  --num_workers 4 \
  --metric_threshold 0.6 \
  --tversky_alpha 0.7 \
  --tversky_beta 0.3
```

Multi-GPU FSDP (3–4× A100 example):

```bash
cd /path/to/Landslide_segmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

export NUM_GPUS=3
export DATASET_ROOT=/scratch/earnest/samreedh/landslide_segmentation/dataset_bijie_landslide
export OUTPUT_DIR=codebase/Geo_physics_equation_derived_model/outputs_bijie_effnet
export FM_BACKBONE=efficientnet

bash codebase/Geo_physics_equation_derived_model/scripts/run_train_bijie_fsdp.sh
```

Or explicit `torchrun`:

```bash
torchrun --standalone --nproc_per_node=3 \
  -m codebase.Geo_physics_equation_derived_model.train.train_bijie \
  --dataset_root "${DATASET_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --fm_backbone efficientnet \
  --resize_to 256 \
  --batch_size 2 \
  --num_workers 2 \
  --fsdp \
  --high_dim_256
```

**Notes:**

- `--batch_size` is per GPU. EfficientNet FM uses much less VRAM than Prithvi+TTEB at 256-d; try `batch_size 4–8` per GPU before `--high_dim_256`.
- First run downloads ImageNet weights for EfficientNet via `timm` (needs network once per node/cache).
- Metrics: `outputs_*/results/epoch_metrics.csv` — use **`val_landslide_f1`** / **`val_landslide_iou`** (micro metrics over full val set).
- No `.pt` checkpoints are saved (GlusterFS-safe); metrics-only CSV.

## HPC: Bijie with Prithvi (legacy path)

```bash
torchrun --standalone --nproc_per_node=3 \
  -m codebase.Geo_physics_equation_derived_model.train.train_bijie \
  --dataset_root /scratch/.../dataset_bijie_landslide \
  --output_dir codebase/Geo_physics_equation_derived_model/outputs_bijie_prithvi \
  --fm_backbone prithvi \
  --prithvi_snapshot /scratch/.../snapshots/2c84e383194986040f883cc43d7869002c425e1b \
  --resize_to 256 \
  --batch_size 1 \
  --num_workers 2 \
  --fsdp
```

## Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--fm_backbone` | `efficientnet` | `efficientnet` or `prithvi` |
| `--efficientnet_name` | `tf_efficientnet_b4` | Any `timm` features-only model |
| `--unfreeze_efficientnet` | off | Train full EfficientNet (not only 1×1 projectors) |
| `--no_efficientnet_pretrained` | off | Random-init EfficientNet |
| `--high_dim_256` | off | C=256 everywhere (heavier) |
| `--fsdp` | auto on multi-GPU | Shard model across GPUs |
| `--metric_threshold` | `0.6` | Aligned with dual-stream Bijie |
| `--tversky_alpha` / `--tversky_beta` | `0.7` / `0.3` | Penalize false positives |

## Landslide4Sense

Same `--fm_backbone` flag:

```bash
python -m codebase.Geo_physics_equation_derived_model.train.train_landslide4sense \
  --dataset_root /path/to/Landslide4Sense \
  --fm_backbone efficientnet \
  --output_dir codebase/Geo_physics_equation_derived_model/outputs_l4s_effnet
```

## Layout

- `encoders/efficientnet_fm.py` — EfficientNet FM pyramid (L0–L4)
- `encoders/prithvi_lora.py` — Prithvi FM (optional)
- `physics/` — mechanistic cells, proxy mapper
- `fusion/` — MAO-GeoEGCA
- `bridge/` — TTEB
- `decoder/` — physics decoder
- `data/` — Bijie/L4S loaders + real proxies
- `train/` — trainer, fixed micro metrics, FSDP helpers
