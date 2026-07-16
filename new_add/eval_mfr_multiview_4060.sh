#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-m2f}"
PROJECT_DIR="/data/m2f"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/temp_data/mfr_multiview_4060_8g_output/model_final.pth}"
VAL_DIR="${VAL_DIR:-${PROJECT_DIR}/temp_data/multiview_feature_dataset/val}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/Mask2Former/configs/mfr_multiview/video_maskformer2_R50_bs1_14view.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/temp_data/eval_mfr_multiview_4060}"

if [ ! -s "${WEIGHTS}" ]; then
  echo "[ERROR] Weights not found or empty: ${WEIGHTS}"
  exit 1
fi

python "${PROJECT_DIR}/new_add/eval_mfr_multiview.py" \
  --val_dir "${VAL_DIR}" \
  --config_file "${CONFIG_FILE}" \
  --weights "${WEIGHTS}" \
  --output_dir "${OUTPUT_DIR}" \
  --min_size_test 256 \
  --eval_class_mode both


#  跑其他权重的评估结果：WEIGHTS、OUTPUT_DIR、CONFIG_FILE
#  WEIGHTS=/data/m2f/temp_data/mfr_multiview_server_bs1_512_output/model_final.pth OUTPUT_DIR=/data/m2f/temp_data/eval_mfr_multiview_server_bs1_512 ./new_add/eval_mfr_multiview_4060.sh