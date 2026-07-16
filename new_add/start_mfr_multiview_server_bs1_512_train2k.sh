#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-m2f}"
PROJECT_DIR="${PROJECT_DIR:-/data/m2f}"
MASK2FORMER_DIR="${MASK2FORMER_DIR:-${PROJECT_DIR}/Mask2Former}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/temp_data/multiview_feature_dataset_train2k_val100_512}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/temp_data/mfr_multiview_server_bs1_512_train2k_output}"
MAX_ITER="${MAX_ITER:-8000}"
EVAL_PERIOD="${EVAL_PERIOD:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ ! -s "${DATASET_DIR}/train/models.json" ] || [ ! -s "${DATASET_DIR}/val/models.json" ]; then
  echo "[ERROR] Dataset models.json not found under: ${DATASET_DIR}"
  echo "        Run convert_mfr_multiview_train2k_val100_512.sh first,"
  echo "        or set DATASET_DIR=/path/to/multiview_feature_dataset."
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${MASK2FORMER_DIR}"

export MFR_MULTIVIEW_DATASET="${DATASET_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-0}}"

echo "Using dataset: ${MFR_MULTIVIEW_DATASET}"
echo "Using output:  ${OUTPUT_DIR}"
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
  SOLVER.MAX_ITER "${MAX_ITER}" \
  TEST.EVAL_PERIOD "${EVAL_PERIOD}" \
  INPUT.MIN_SIZE_TRAIN "(512,)" \
  INPUT.MIN_SIZE_TEST 512 \
  DATALOADER.NUM_WORKERS "${NUM_WORKERS}" \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 \
  MODEL.WEIGHTS "${PRETRAIN_WEIGHTS}" \
  OUTPUT_DIR "${OUTPUT_DIR}"
