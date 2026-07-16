#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-m2f_new}"
PROJECT_DIR="${PROJECT_DIR:-/data/m2f}"
MASK2FORMER_DIR="${MASK2FORMER_DIR:-${PROJECT_DIR}/Mask2Former}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7860}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/inference:${MASK2FORMER_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-0}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-m2f}"

if [ -z "${DISPLAY:-}" ] && [ -d /mnt/wslg ]; then
  export DISPLAY=:0
  export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
  export PULSE_SERVER="${PULSE_SERVER:-/mnt/wslg/PulseServer}"
fi

python "${PROJECT_DIR}/inference/web_inference_server.py" --host "${HOST}" --port "${PORT}"
