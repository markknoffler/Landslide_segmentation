#!/usr/bin/env bash
set -euo pipefail
CACHE_DIR="/home/user/Desktop/Deep_learning_projects/4PI/prithvi_weights"
mkdir -p "${CACHE_DIR}"
conda run -n deeplearning python - <<'PY'
from huggingface_hub import hf_hub_download
repo = "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL"
cache = "/home/user/Desktop/Deep_learning_projects/4PI/prithvi_weights"
for name in ["config.json", "prithvi_mae.py", "Prithvi_EO_V2_100M_TL.pt"]:
    path = hf_hub_download(repo, name, cache_dir=cache)
    print("Downloaded:", path)
PY
