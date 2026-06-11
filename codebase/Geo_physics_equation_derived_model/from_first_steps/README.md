# Penta-Stream Geo-Physics Landslide Segmentation (`from_first_steps/`)

Incremental graft of **PS-GPLNet** (5 encoders + MAO/TTEB + dual physics decoder) onto the working DiGATe dual-stream baseline.

## Architecture (Step 3 default)

- **Encoders:** EfficientNet RGB + Physics RGB + EfficientNet DEM + Physics DEM + Prithvi+LoRA
- **Fusion:** Complementary Modality Bridge → MAO-GeoEGCA + TTEB
- **Decoder:** Dual physics decoder + MPEF path fusion
- **Baseline comparison:** use `model_backup.py` (DiGATe) separately in ablation studies

See `model_architecture.md` and `step_by_step_implementation.md` for full design.

## Dataset Layout Expected

```
dataset/
  TrainData/
    img/image_*.h5
    mask/mask_*.h5
  ValidData/
    img/image_*.h5
    mask/mask_*.h5   # optional
  TestData/
    img/image_*.h5
```

## Train — Landslide4Sense (Step 3)

```bash
python training.py \
  --dataset_root /path/to/Landslide4Sense/dataset \
  --output_dir ./outputs_step3_l4s \
  --prithvi_snapshot /path/to/models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL \
  --backbone tf_efficientnet_b4 \
  --pretrained \
  --freeze_backbone \
  --epochs 100 \
  --batch_size 32 \
  --lr 3e-4 \
  --tversky_alpha 0.3 \
  --tversky_beta 0.7 \
  --metric_threshold 0.5 \
  --save_every 5
```

Do **not** pass `--pretrained_path None` (shell passes the literal string `None`).

## Resume from Last Checkpoint

```bash
python training.py \
  --dataset_root /path/to/Landslide4Sense/dataset \
  --output_dir ./outputs_step3_l4s \
  --prithvi_snapshot /path/to/models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL \
  --resume
```

## Outputs

- `checkpoint/epoch_XXXX.pt` saved every 5 epochs.
- `checkpoint/best.pt` best validation F1.
- `results/epoch_metrics.csv` epoch-wise training/validation metrics.
- `results/final_metrics.csv` final run summary.

## Optional Dataset Check

```bash
python data_processing.py \
  --dataset_root /home/user/Desktop/Deep_learning_projects/4PI/dataset \
  --output_json ./results/dataset_summary.json
```

## Train — Bijie (Step 3)

Use `train_bijie.py` (PNG loader, 70/20/10 split):

```bash
python train_bijie.py \
  --dataset_root /path/to/Bijie-landslide-dataset \
  --output_dir ./outputs_step3_bijie \
  --prithvi_snapshot /path/to/models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL \
  --backbone tf_efficientnet_b4 \
  --pretrained \
  --freeze_backbone \
  --epochs 100 \
  --batch_size 32 \
  --lr 3e-4 \
  --tversky_alpha 0.3 \
  --tversky_beta 0.7 \
  --metric_threshold 0.5 \
  --save_every 5
```

## Notes

- **Landslide4Sense streams:** `stream_a` = RGB; `stream_b` = NDVI + slope + DEM (`--bands RGB-NDVI-SLOPE-DEM`)
- **Bijie streams:** `stream_a` = RGB; `stream_b` = DEM ×3 (proxies derived inside `forward()`)
- Default hyperparameters match Bijie Step 3: Tversky 0.3/0.7, threshold 0.5
- Reproduce Step 2 on either dataset: add `--decoder paper --no_physics_encoders`
- Required packages: `timm`, `h5py`, `opencv-python`, Prithvi snapshot on disk or `PRITHVI_SNAPSHOT` env var
