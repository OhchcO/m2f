#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-m2f}"
PROJECT_DIR="${PROJECT_DIR:-/data/m2f}"
MASK2FORMER_DIR="${MASK2FORMER_DIR:-${PROJECT_DIR}/Mask2Former}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${MASK2FORMER_DIR}"

export MFR_MULTIVIEW_DATASET="${MFR_MULTIVIEW_DATASET:-${PROJECT_DIR}/temp_data/multiview_feature_dataset}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PRETRAIN_WEIGHTS="${PRETRAIN_WEIGHTS:-}"

python train_net_video.py \
  --config-file configs/mfr_multiview/video_maskformer2_R50_bs1_14view.yaml \
  --num-gpus 1 \
  SOLVER.IMS_PER_BATCH 1 \
  INPUT.MIN_SIZE_TRAIN "(256,)" \
  INPUT.MIN_SIZE_TEST 256 \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 50 \
  MODEL.MASK_FORMER.TRAIN_NUM_POINTS 2048 \
  MODEL.MASK_FORMER.DEC_LAYERS 6 \
  DATALOADER.NUM_WORKERS 0 \
  MODEL.WEIGHTS "${PRETRAIN_WEIGHTS}" \
  OUTPUT_DIR "${PROJECT_DIR}/temp_data/mfr_multiview_4060_8g_output"
