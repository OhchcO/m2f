#!/bin/bash
# Run face-level area evaluation.
# Edit the variables in this block when paths or evaluation mode change.

set -e

PROJECT_DIR="/data/m2f"
CONDA_ENV="m2f"

VAL_DIR="${PROJECT_DIR}/temp_data/dataset_24class/val"
CONFIG_FILE="${PROJECT_DIR}/Mask2Former/configs/dataset_24class/Base-24Class-InstanceSegmentation.yaml"
WEIGHTS="${PROJECT_DIR}/temp_data/24class_output/model_final.pth"
OUTPUT_DIR="${PROJECT_DIR}/temp_data/eval_area_metrics_24class"

THRESHOLD="0.5"
LABEL_SET="24"
EVAL_CLASS_MODE="both"  # fine, coarse, or both

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

if [ ! -d "${VAL_DIR}" ]; then
    echo "[ERROR] VAL_DIR does not exist: ${VAL_DIR}"
    exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "[ERROR] CONFIG_FILE does not exist: ${CONFIG_FILE}"
    exit 1
fi

if [ ! -f "${WEIGHTS}" ]; then
    echo "[ERROR] WEIGHTS does not exist: ${WEIGHTS}"
    echo "Edit WEIGHTS in this script before running evaluation."
    exit 1
fi

cd "${PROJECT_DIR}"

python "${PROJECT_DIR}/new_add/eval_area_metrics.py" \
    --val_dir "${VAL_DIR}" \
    --config_file "${CONFIG_FILE}" \
    --weights "${WEIGHTS}" \
    --output_dir "${OUTPUT_DIR}" \
    --threshold "${THRESHOLD}" \
    --label_set "${LABEL_SET}" \
    --eval_class_mode "${EVAL_CLASS_MODE}"
