# Geo-Physics Equation-Derived Landslide Segmentation

Novel three-stream architecture: physics-encoded RGB/DEM encoders, Prithvi-EO-2.0-100M-TL (LoRA), MAO-GeoEGCA fusion, TTEB skip bridge, and physics-gated decoder.

See [model_architecture.md](model_architecture.md) for full specification.

## Setup

Download Prithvi weights (outside git repo):

```bash
bash scripts/download_prithvi.sh
```

## Training

From repository root (single GPU):

```bash
conda run -n deeplearning python -m codebase.Geo_physics_equation_derived_model.train.train_bijie \
  --dataset_root /home/user/Desktop/Deep_learning_projects/4PI/dataset_bijie_landslide \
  --output_dir codebase/Geo_physics_equation_derived_model/outputs_bijie \
  --prithvi_snapshot /path/to/prithvi/snapshots/<hash> \
  --batch_size 2
```

### Multi-GPU (FSDP + NCCL, e.g. 4× A100 80GB on one node)

Launch with `torchrun` (not plain `python`). NCCL is used automatically as the process-group backend; you still write normal CUDA PyTorch code.

```bash
cd /path/to/Landslide_segmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=4 \
  -m codebase.Geo_physics_equation_derived_model.train.train_bijie \
  --dataset_root /scratch/.../dataset_bijie_landslide \
  --output_dir codebase/Geo_physics_equation_derived_model/outputs_bijie \
  --prithvi_snapshot /scratch/.../snapshots/<hash> \
  --resize_to 256 \
  --batch_size 2 \
  --num_workers 4 \
  --fsdp
```

Or use the helper script (edit paths inside or via env vars):

```bash
bash codebase/Geo_physics_equation_derived_model/scripts/run_train_bijie_fsdp.sh
```

- `--batch_size` is **per GPU** (global batch = `batch_size × num_gpus`).
- FSDP shards weights across GPUs; TTEB uses **chunked attention** so 256×256 training fits in memory.
- **No model checkpoints** (`.pt`) are written during training — only small CSV metrics — to avoid GlusterFS stalls and NCCL timeouts on multi-GPU jobs.

Landslide4Sense: same pattern with `train_landslide4sense`.

Metrics CSVs are written under `results/` (`epoch_metrics.csv`, `final_metrics.csv`).

## Layout

- `physics/` — pixel & latent mechanistic cells, proxy mapper
- `encoders/` — physics encoders, Prithvi+LoRA
- `fusion/` — MAO-GeoEGCA
- `bridge/` — TTEB
- `decoder/` — physics decoder + PGDI
- `data/` — datasets aligned with ablation study splits
- `train/` — trainer, losses, metrics
