#!/usr/bin/env bash
# Multi-GPU FSDP training on one node (e.g. 4x A100 80GB, NVLink).
# Usage (interactive):
#   bash codebase/Geo_physics_equation_derived_model/scripts/run_train_bijie_fsdp.sh
# SLURM example at bottom of this file.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

DATASET_ROOT="${DATASET_ROOT:-/scratch/earnest/samreedh/landslide_segmentation/dataset_bijie_landslide}"
OUTPUT_DIR="${OUTPUT_DIR:-codebase/Geo_physics_equation_derived_model/outputs_bijie}"
PRITHVI_SNAPSHOT="${PRITHVI_SNAPSHOT:-/scratch/earnest/samreedh/landslide_segmentation/models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL/snapshots/2c84e383194986040f883cc43d7869002c425e1b}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  -m codebase.Geo_physics_equation_derived_model.train.train_bijie \
  --dataset_root "${DATASET_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --prithvi_snapshot "${PRITHVI_SNAPSHOT}" \
  --resize_to 256 \
  --batch_size 2 \
  --num_workers 4 \
  --fsdp \
  "$@"

# --- SLURM (single node, 4 GPUs) ---
# #SBATCH --gres=gpu:4
# #SBATCH --cpus-per-task=32
# #SBATCH --time=48:00:00
# module load cuda/12.1  # site-specific
# source activate deeplearning
# export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
# export MASTER_PORT=29500
# srun bash codebase/Geo_physics_equation_derived_model/scripts/run_train_bijie_fsdp.sh
