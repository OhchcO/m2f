#!/bin/bash
# Mask2Former 24类实例分割训练脚本
# GPU: RTX 4060 Laptop (8GB)

set -e

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate mask2former

# 项目路径
PROJECT_DIR="/data/m2f"
MASK2FORMER_DIR="${PROJECT_DIR}/Mask2Former"
CONFIG="${MASK2FORMER_DIR}/configs/dataset_24class/Base-24Class-InstanceSegmentation.yaml"
NUM_GPUS=1

echo "=========================================="
echo "  Mask2Former 24类实例分割训练"
echo "  配置: ${CONFIG}"
echo "  GPU数量: ${NUM_GPUS}"
echo "=========================================="

cd "${MASK2FORMER_DIR}"

# 启动训练
python train_net.py \
    --config-file "${CONFIG}" \
    --num-gpus "${NUM_GPUS}" \
    SOLVER.IMS_PER_BATCH 2 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.MAX_ITER 10000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    TEST.EVAL_PERIOD 1000 \
    OUTPUT_DIR "${PROJECT_DIR}/temp_data/24class_output"
