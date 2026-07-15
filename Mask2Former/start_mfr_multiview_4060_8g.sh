#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-mask2former}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "$(dirname "$0")"

export MFR_MULTIVIEW_DATASET="${MFR_MULTIVIEW_DATASET:-/data/m2f/temp_data/multiview_feature_dataset}"
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
  OUTPUT_DIR /data/m2f/temp_data/mfr_multiview_4060_8g_output
