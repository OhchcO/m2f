#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-m2f}"
PROJECT_DIR="${PROJECT_DIR:-/data/m2f}"
MASK2FORMER_DIR="${MASK2FORMER_DIR:-${PROJECT_DIR}/Mask2Former}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/temp_data/multiview_feature_dataset_train2k_val100_512}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/temp_data/eval_mfr_multiview_server_bs1_512_train2k}"
WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/temp_data/mfr_multiview_server_bs1_512_train2k_output/model_final.pth}"

if [ ! -s "${WEIGHTS}" ]; then
  echo "[ERROR] Trained weights not found: ${WEIGHTS}"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${MASK2FORMER_DIR}"

export MFR_MULTIVIEW_DATASET="${DATASET_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-0}}"

python "${PROJECT_DIR}/new_add/eval_mfr_multiview.py" \
  --config-file configs/mfr_multiview/video_maskformer2_R50_bs1_14view.yaml \
  --weights "${WEIGHTS}" \
  --dataset-dir "${DATASET_DIR}" \
  --split val \
  --output-dir "${OUTPUT_DIR}" \
  --score-threshold 0.3 \
  --eval-class-mode both \
  --opts \
  INPUT.MIN_SIZE_TEST 512 \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100
