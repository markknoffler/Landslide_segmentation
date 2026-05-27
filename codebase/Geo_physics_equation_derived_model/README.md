# Geo-Physics Equation-Derived Landslide Segmentation

Novel three-stream architecture: physics-encoded RGB/DEM encoders, Prithvi-EO-2.0-100M-TL (LoRA), MAO-GeoEGCA fusion, TTEB skip bridge, and physics-gated decoder.

See [model_architecture.md](model_architecture.md) for full specification.

## Setup

Download Prithvi weights (outside git repo):

```bash
bash scripts/download_prithvi.sh
```

## Training

From repository root:

```bash
conda run -n deeplearning python -m codebase.Geo_physics_equation_derived_model.train.train_bijie \
  --dataset_root /home/user/Desktop/Deep_learning_projects/4PI/dataset_bijie_landslide \
  --output_dir codebase/Geo_physics_equation_derived_model/outputs_bijie

conda run -n deeplearning python -m codebase.Geo_physics_equation_derived_model.train.train_landslide4sense \
  --dataset_root /home/user/Desktop/Deep_learning_projects/4PI/dataset \
  --output_dir codebase/Geo_physics_equation_derived_model/outputs_l4s
```

Use `--resume` to continue from the latest checkpoint. Metrics CSVs are written under `results/`.

## Layout

- `physics/` — pixel & latent mechanistic cells, proxy mapper
- `encoders/` — physics encoders, Prithvi+LoRA
- `fusion/` — MAO-GeoEGCA
- `bridge/` — TTEB
- `decoder/` — physics decoder + PGDI
- `data/` — datasets aligned with ablation study splits
- `train/` — trainer, losses, metrics
