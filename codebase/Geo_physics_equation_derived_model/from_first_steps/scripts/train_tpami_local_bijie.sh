#!/usr/bin/env bash
# Local architecture-faithful PS-GPLNet (B4) Bijie train — cluster final .pt not synced.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=./outputs_tpami_local_bijie
PRITHVI=/home/user/Desktop/Deep_learning_projects/4PI/prithvi_weights/models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL
DATA=/home/user/Desktop/Deep_learning_projects/4PI/dataset_bijie_landslide/Bijie-landslide-dataset

mkdir -p "$OUT"
exec conda run -n deeplearning --no-capture-output python train_bijie.py \
  --dataset_root "$DATA" \
  --output_dir "$OUT" \
  --prithvi_snapshot "$PRITHVI" \
  --backbone tf_efficientnet_b4 \
  --pretrained \
  --freeze_backbone \
  --epochs 100 \
  --batch_size 4 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --device cuda:0 \
  --no-auto_gpu \
  --min_free_gb 4.0 \
  --amp \
  --save_every 1 \
  --seed 42 \
  --tversky_alpha 0.3 \
  --tversky_beta 0.7 \
  --main_weight 1.0 \
  --aux2_weight 0.6 \
  --aux3_weight 0.4 \
  "$@"
