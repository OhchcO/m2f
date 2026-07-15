#!/bin/bash
# Mask2Former 7-class instance segmentation training script.

set -e

eval "$(conda shell.bash hook)"
conda activate mask2former

PROJECT_DIR="/data/m2f"
MASK2FORMER_DIR="${PROJECT_DIR}/Mask2Former"
CONFIG="${MASK2FORMER_DIR}/configs/dataset_7class/Base-7Class-InstanceSegmentation.yaml"
PRETRAIN_WEIGHTS="${PROJECT_DIR}/pretrained/maskformer2_R50_coco_instance.pkl"
NUM_GPUS=1
OUTPUT_DIR="${PROJECT_DIR}/temp_data/7class_output"
IMS_PER_BATCH=2
BASE_LR=0.0001
MAX_ITER=10000
CHECKPOINT_PERIOD=2000
EVAL_PERIOD=1000

echo "=========================================="
echo "  Mask2Former 7-class instance training"
echo "  Config: ${CONFIG}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPUs: ${NUM_GPUS}"
echo "=========================================="

if [ ! -s "${PRETRAIN_WEIGHTS}" ]; then
    echo "[ERROR] Pretrained weights not found or empty: ${PRETRAIN_WEIGHTS}"
    echo "Run: bash ${PROJECT_DIR}/new_add/download_pretrained_weights.sh"
    exit 1
fi

cd "${MASK2FORMER_DIR}"

python train_net.py \
    --config-file "${CONFIG}" \
    --num-gpus "${NUM_GPUS}" \
    SOLVER.IMS_PER_BATCH "${IMS_PER_BATCH}" \
    SOLVER.BASE_LR "${BASE_LR}" \
    SOLVER.MAX_ITER "${MAX_ITER}" \
    SOLVER.CHECKPOINT_PERIOD "${CHECKPOINT_PERIOD}" \
    TEST.EVAL_PERIOD "${EVAL_PERIOD}" \
    MODEL.WEIGHTS "${PRETRAIN_WEIGHTS}" \
    OUTPUT_DIR "${OUTPUT_DIR}"
