#!/usr/bin/env bash
# Quick validation fine-tune for VideoMaskFormer + Face-ID mean fusion.
# This intentionally loads model weights without --resume: the new fusion
# gates start at zero and the old optimizer state must not be reused.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-m2f}"
PROJECT_DIR="${PROJECT_DIR:-/data/m2f}"
MASK2FORMER_DIR="${MASK2FORMER_DIR:-${PROJECT_DIR}/Mask2Former}"
CONFIG_FILE="${CONFIG_FILE:-${MASK2FORMER_DIR}/configs/mfr_multiview/video_maskformer2_R50_bs1_14view.yaml}"
DATASET_DIR="${DATASET_DIR:-/hy-tmp/datasets/MFRInstSegM2F_2100}"
DATASET_NAME="${DATASET_NAME:-$(basename "${DATASET_DIR}")}"
BASE_WEIGHTS="${BASE_WEIGHTS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/hy-tmp/mfr_multiview_face_mean_fusion_512_output}"
INPUT_SIZE="${INPUT_SIZE:-512}"
MAX_ITER="${MAX_ITER:-20000}"
BASE_LR="${BASE_LR:-0.00001}"
STEPS="${STEPS:-(15000,)}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-2000}"
EVAL_PERIOD="${EVAL_PERIOD:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ -z "${BASE_WEIGHTS}" || ! -s "${BASE_WEIGHTS}" ]]; then
  echo "[ERROR] Set BASE_WEIGHTS to an existing multi-view .pth checkpoint."
  echo "Example: BASE_WEIGHTS=/path/to/mfr_multiview/model_final.pth $0"
  exit 1
fi

if [[ ! -s "${DATASET_DIR}/train/models.json" || ! -s "${DATASET_DIR}/val/models.json" ]]; then
  echo "[ERROR] Dataset models.json not found under: ${DATASET_DIR}"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
cd "${MASK2FORMER_DIR}"

export MFR_MULTIVIEW_DATASET="${DATASET_DIR}"
export MFR_DATASET_NAME="${DATASET_NAME}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-0}}"

echo "Fine-tuning Face-ID mean fusion"
echo "  dataset:      ${MFR_MULTIVIEW_DATASET}"
echo "  config:       ${CONFIG_FILE}"
echo "  base weights: ${BASE_WEIGHTS}"
echo "  output:       ${OUTPUT_DIR}"
echo "  max iter:     ${MAX_ITER}"
echo "  base lr:      ${BASE_LR}"
echo "  GPU:          ${CUDA_VISIBLE_DEVICES}"

python train_net_video.py \
  --config-file "${CONFIG_FILE}" \
  --num-gpus 1 \
  SOLVER.IMS_PER_BATCH 1 \
  SOLVER.MAX_ITER "${MAX_ITER}" \
  SOLVER.BASE_LR "${BASE_LR}" \
  SOLVER.STEPS "${STEPS}" \
  SOLVER.CHECKPOINT_PERIOD "${CHECKPOINT_PERIOD}" \
  TEST.EVAL_PERIOD "${EVAL_PERIOD}" \
  DATASETS.TRAIN "('${DATASET_NAME}_multiview_train',)" \
  DATASETS.TEST "('${DATASET_NAME}_multiview_val',)" \
  INPUT.MIN_SIZE_TRAIN "(${INPUT_SIZE},)" \
  INPUT.MIN_SIZE_TEST "${INPUT_SIZE}" \
  DATALOADER.NUM_WORKERS "${NUM_WORKERS}" \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 \
  MODEL.WEIGHTS "${BASE_WEIGHTS}" \
  MODEL.FACE_FUSION.ENABLED True \
  MODEL.FACE_FUSION.INIT_GAMMA 0.0 \
  OUTPUT_DIR "${OUTPUT_DIR}"
