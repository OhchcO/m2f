#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-m2f}"
PROJECT_DIR="${PROJECT_DIR:-/data/m2f}"
MASK2FORMER_DIR="${MASK2FORMER_DIR:-${PROJECT_DIR}/Mask2Former}"
DATASET_DIR="${DATASET_DIR:-/hy-tmp/datasets/SimpleM2F_200}"
DATASET_NAME="${DATASET_NAME:-$(basename "${DATASET_DIR}")}"
OUTPUT_DIR="${OUTPUT_DIR:-/hy-tmp/mfr_multiview_SimpleM2F_200_F_512_ep50_output}"
INPUT_SIZE="${INPUT_SIZE:-512}"
EPOCHS="${EPOCHS:-50}"
MAX_ITER="${MAX_ITER:-}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-}"
EVAL_PERIOD="${EVAL_PERIOD:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ ! -s "${DATASET_DIR}/train/models.json" ] || [ ! -s "${DATASET_DIR}/val/models.json" ]; then
  echo "[ERROR] Dataset models.json not found under: ${DATASET_DIR}"
  exit 1
fi

TRAIN_SAMPLES="$(python - "${DATASET_DIR}/train/models.json" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(len(json.load(f).get("models", [])))
PY
)"
ITER_PER_EPOCH="${TRAIN_SAMPLES}"
MAX_ITER="${MAX_ITER:-$((ITER_PER_EPOCH * EPOCHS))}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-$((ITER_PER_EPOCH * 10))}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${MASK2FORMER_DIR}"

export MFR_MULTIVIEW_DATASET="${DATASET_DIR}"
export MFR_DATASET_NAME="${DATASET_NAME}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-0}}"

echo "Using multi-view dataset: ${MFR_MULTIVIEW_DATASET}"
echo "Using dataset name: ${MFR_DATASET_NAME}"
echo "Using output: ${OUTPUT_DIR}"
echo "Using epochs: ${EPOCHS}, train samples: ${TRAIN_SAMPLES}, iter/epoch: ${ITER_PER_EPOCH}, max_iter: ${MAX_ITER}"
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
  SOLVER.CHECKPOINT_PERIOD "${CHECKPOINT_PERIOD}" \
  TEST.EVAL_PERIOD "${EVAL_PERIOD}" \
  DATASETS.TRAIN "('${DATASET_NAME}_multiview_train',)" \
  DATASETS.TEST "('${DATASET_NAME}_multiview_val',)" \
  INPUT.MIN_SIZE_TRAIN "(${INPUT_SIZE},)" \
  INPUT.MIN_SIZE_TEST "${INPUT_SIZE}" \
  DATALOADER.NUM_WORKERS "${NUM_WORKERS}" \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 \
  MODEL.WEIGHTS "${PRETRAIN_WEIGHTS}" \
  OUTPUT_DIR "${OUTPUT_DIR}"
