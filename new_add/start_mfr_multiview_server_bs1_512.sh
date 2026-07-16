#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-m2f}"
PROJECT_DIR="${PROJECT_DIR:-/data/m2f}"
MASK2FORMER_DIR="${MASK2FORMER_DIR:-${PROJECT_DIR}/Mask2Former}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${MASK2FORMER_DIR}"

export MFR_MULTIVIEW_DATASET="${MFR_MULTIVIEW_DATASET:-${PROJECT_DIR}/temp_data/multiview_feature_dataset}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-0}}"

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

PRETRAIN_WEIGHTS="${PRETRAIN_WEIGHTS:-${PROJECT_DIR}/pretrained/maskformer2_R50_coco_instance.pkl}"
if [ ! -s "${PRETRAIN_WEIGHTS}" ]; then
  echo "[ERROR] Pretrained weights not found or empty: ${PRETRAIN_WEIGHTS}"
  exit 1
fi

python train_net_video.py \
  --config-file configs/mfr_multiview/video_maskformer2_R50_bs1_14view.yaml \
  --num-gpus 1 \
  SOLVER.IMS_PER_BATCH 1 \
  INPUT.MIN_SIZE_TRAIN "(512,)" \
  INPUT.MIN_SIZE_TEST 512 \
  DATALOADER.NUM_WORKERS 4 \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 \
  MODEL.WEIGHTS "${PRETRAIN_WEIGHTS}" \
  OUTPUT_DIR "${PROJECT_DIR}/temp_data/mfr_multiview_server_bs1_512_output"
