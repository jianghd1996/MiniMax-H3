#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/DataPart/jianghongda/checkpoint}"
MODEL_VARIANT="${MODEL_VARIANT:-fl2va}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
PORT="${PORT:-30010}"
HOST="${HOST:-0.0.0.0}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-speed}"

if [[ "${MODEL_VARIANT}" != "fl2va" && "${MODEL_VARIANT}" != "ref2va" ]]; then
  echo "MODEL_VARIANT must be fl2va or ref2va" >&2
  exit 2
fi

resolve_model_path() {
  if [[ -n "${MODEL_PATH:-}" ]]; then
    printf '%s\n' "${MODEL_PATH}"
    return
  fi

  local candidate
  for candidate in     "${CHECKPOINT_ROOT}/MiniMax-H3"     "${CHECKPOINT_ROOT}/minimax-h3"     "${CHECKPOINT_ROOT}"; do
    if [[ -f "${candidate}/model_index.json" && -f "${candidate}/${MODEL_VARIANT^^}/model_index.json" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done

  local index_file
  index_file="$(find "${CHECKPOINT_ROOT}" -maxdepth 3 -type f -path "*/${MODEL_VARIANT^^}/model_index.json" -print -quit 2>/dev/null || true)"
  if [[ -n "${index_file}" ]]; then
    dirname "$(dirname "${index_file}")"
    return
  fi

  echo "Could not find a MiniMax-H3 checkpoint below ${CHECKPOINT_ROOT}." >&2
  echo "Set MODEL_PATH to the directory containing model_index.json and FL2VA/ or Ref2VA/." >&2
  exit 1
}

MODEL_PATH="$(resolve_model_path)"
TASK_DIR="${MODEL_PATH}/${MODEL_VARIANT^^}"

if [[ ! -f "${MODEL_PATH}/model_index.json" || ! -f "${TASK_DIR}/model_index.json" ]]; then
  echo "Incomplete checkpoint: ${MODEL_PATH}" >&2
  echo "Expected model_index.json and ${MODEL_VARIANT^^}/model_index.json." >&2
  exit 1
fi

if ! command -v sglang >/dev/null 2>&1; then
  echo "sglang is not installed in the active environment." >&2
  echo "Install the SGLang diffusion environment, then rerun this script." >&2
  exit 1
fi

visible_count="$(awk -F',' '{print NF}' <<<"${GPU_IDS}")"
if [[ "${visible_count}" -ne "${NUM_GPUS}" ]]; then
  echo "GPU_IDS exposes ${visible_count} GPUs, but NUM_GPUS=${NUM_GPUS}." >&2
  exit 2
fi

echo "Model path : ${MODEL_PATH}"
echo "Variant    : ${MODEL_VARIANT}"
echo "GPUs       : ${GPU_IDS}"
echo "Endpoint   : http://${HOST}:${PORT}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec sglang serve   --model-path "${MODEL_PATH}"   --num-gpus "${NUM_GPUS}"   --ulysses-degree "${NUM_GPUS}"   --performance-mode "${PERFORMANCE_MODE}"   --host "${HOST}"   --port "${PORT}"   --model-variant "${MODEL_VARIANT}"
