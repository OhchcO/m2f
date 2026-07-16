#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-m2f}"
PROJECT_DIR="${PROJECT_DIR:-/data/m2f}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_DIR}/temp_data/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/temp_data/multiview_feature_dataset_train2k_val100_512}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [ ! -d "${SOURCE_DIR}/labels" ] || [ ! -d "${SOURCE_DIR}/steps" ]; then
  echo "[ERROR] SOURCE_DIR must contain labels/ and steps/: ${SOURCE_DIR}"
  echo "        Usage example:"
  echo "        SOURCE_DIR=/path/to/data bash $0"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

python "${PROJECT_DIR}/new_add/MFRInstSeg_to_mask2former.py" \
  --input-dir "${SOURCE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-train 2000 \
  --num-val 100 \
  --width 512 \
  --height 512 \
  --num-workers "${NUM_WORKERS}"
