#!/bin/bash
# Mask2Former 24类实例分割训练脚本
# GPU: RTX 4060 Laptop (8GB)

set -e

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate m2f

# 项目路径
PROJECT_DIR="/data/m2f"
MASK2FORMER_DIR="${PROJECT_DIR}/Mask2Former"
CONFIG="${MASK2FORMER_DIR}/configs/dataset_24class/Base-24Class-InstanceSegmentation.yaml"
PRETRAIN_WEIGHTS="${PROJECT_DIR}/pretrained/maskformer2_R50_coco_instance.pkl"
NUM_GPUS=1
OUTPUT_DIR="${PROJECT_DIR}/temp_data/24class_output"
IMS_PER_BATCH=2
BASE_LR=0.0001
MAX_ITER=10000
CHECKPOINT_PERIOD=2000
EVAL_PERIOD=1000

echo "=========================================="
echo "  Mask2Former 24类实例分割训练"
echo "  配置: ${CONFIG}"
echo "  GPU数量: ${NUM_GPUS}"
echo "=========================================="

if [ ! -s "${PRETRAIN_WEIGHTS}" ]; then
    echo "[ERROR] Pretrained weights not found or empty: ${PRETRAIN_WEIGHTS}"
    echo "Run: bash ${PROJECT_DIR}/new_add/download_pretrained_weights.sh"
    exit 1
fi

cd "${MASK2FORMER_DIR}"

# 启动训练
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
