#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-mask2former}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "$(dirname "$0")"

export MFR_MULTIVIEW_DATASET="${MFR_MULTIVIEW_DATASET:-/data/m2f/temp_data/multiview_feature_dataset}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-0}}"

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

PRETRAIN_WEIGHTS="${PRETRAIN_WEIGHTS:-/data/m2f/pretrained/maskformer2_R50_coco_instance.pkl}"
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
  OUTPUT_DIR /data/m2f/temp_data/mfr_multiview_server_bs1_512_output
