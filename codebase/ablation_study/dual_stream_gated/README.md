# Dual-Stream Gated Landslide Segmentation (Ablation Study)

Implementation of the paper **"A Dual-Stream Framework for Landslide Segmentation with Cross-Attention Enhancement and Gated Multimodal Fusion"** adapted for **Landslide4Sense** with your provided dataset directory format.

## Files

- `model.py`: Dual-stream model with:
  - shared siamese encoder
  - `TransUp` cross-attention upsampling block
  - `UpFlex` attention-gated skip fusion
  - early (stage 3/4) and late gated multimodal fusion
  - deep-supervision outputs (`main`, `aux2`, `aux3`)
- `losses.py`: Tversky + deep supervision + gate regularization composite loss.
- `dataset.py`: Landslide4Sense `.h5` dataset loader.
  - Stream A: RGB
  - Stream B: NDVI + Slope + DEM
- `metrics.py`: Pixel-level metrics (Acc/Precision/Recall/F1/IoU) and image-level metrics (AUROC/AUPRC/Best-F1 threshold).
- `training.py`: End-to-end training script with checkpointing and CSV logging.
- `data_processing.py`: Dataset structure inspector for `.h5` channel/shape checks.

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

## Train

```bash
python training.py \
  --dataset_root /home/user/Desktop/Deep_learning_projects/4PI/dataset \
  --output_dir . \
  --epochs 100 \
  --batch_size 32 \
  --save_every 5
```

## Resume from Last Checkpoint

```bash
python training.py \
  --dataset_root /home/user/Desktop/Deep_learning_projects/4PI/dataset \
  --output_dir . \
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

## Notes

- Default Landslide4Sense channel assumptions:
  - RGB indices: `3 2 1`
  - NIR index: `7`
  - Slope index: `12`
  - DEM index: `13`
- Override with:
  - `--rgb_indices`
  - `--nir_index`
  - `--slope_index`
  - `--dem_index`
