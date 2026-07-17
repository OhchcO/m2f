#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-m2f}"
PROJECT_DIR="${PROJECT_DIR:-/data/m2f}"
MASK2FORMER_DIR="${MASK2FORMER_DIR:-${PROJECT_DIR}/Mask2Former}"
DATASET_DIR="${DATASET_DIR:-/mnt/e/wsl/datasets/MFRInstSegM2F_2100}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/e/wsl/result/eval_mfr_multiview_model_0059999}"
WEIGHTS="${WEIGHTS:-/mnt/e/wsl/tmp/model_0059999.pth}"

if [ ! -s "${WEIGHTS}" ]; then
  echo "[ERROR] Trained weights not found: ${WEIGHTS}"
  exit 1
fi
if [ ! -s "${DATASET_DIR}/val/models.json" ]; then
  echo "[ERROR] Dataset val/models.json not found under: ${DATASET_DIR}"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${MASK2FORMER_DIR}"

export MFR_MULTIVIEW_DATASET="${DATASET_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-0}}"

python "${PROJECT_DIR}/new_add/eval_mfr_multiview.py" \
  --config_file "${MASK2FORMER_DIR}/configs/mfr_multiview/video_maskformer2_R50_bs1_14view.yaml" \
  --weights "${WEIGHTS}" \
  --val_dir "${DATASET_DIR}/val" \
  --output_dir "${OUTPUT_DIR}" \
  --score_threshold 0.3 \
  --min_size_test 512 \
  --num_views 14 \
  --eval_class_mode both
