#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-single}"
MODEL_PATH="${MODEL_PATH:-/mnt/DataPart/jianghongda/checkpoint/MiniMax-H3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TASK_TYPE="${TASK_TYPE:-fl2va}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VENV_DIR="${VENV_DIR:-.venv-h3}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${MODEL_PATH}" ]] || fail "model directory not found: ${MODEL_PATH}"
[[ -f "${MODEL_PATH}/model_index.json" ]] || fail "missing ${MODEL_PATH}/model_index.json"
[[ -d "${MODEL_PATH}/FL2VA" ]] || fail "missing ${MODEL_PATH}/FL2VA; download the complete H3 checkpoint"
[[ "${TASK_TYPE}" == "fl2va" || "${TASK_TYPE}" == "ref2va" ]] ||
  fail "TASK_TYPE must be fl2va or ref2va"
if [[ "${TASK_TYPE}" == "ref2va" && ! -d "${MODEL_PATH}/Ref2VA" ]]; then
  fail "missing ${MODEL_PATH}/Ref2VA"
fi

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi
command -v vllm >/dev/null || fail "vllm not found; run scripts/local/setup_vllm_omni.sh first"
command -v ffmpeg >/dev/null || fail "ffmpeg not found"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-14400}"

common=(
  serve "${MODEL_PATH}"
  --omni
  --task-type "${TASK_TYPE}"
  --host "${HOST}"
  --port "${PORT}"
  --trust-remote-code
)

case "${PROFILE}" in
  single)
    # Accuracy-oriented single-GPU path. Requires substantial system RAM.
    exec vllm "${common[@]}"       --num-gpus 1       --enable-cpu-offload       --diffusion-attention-backend "${ATTENTION_BACKEND:-FLASH_ATTN}"
    ;;
  2x4090)
    # Validated low-memory starting shape: 1024x576, 5 seconds.
    exec vllm "${common[@]}"       --num-gpus 2       --tensor-parallel-size 2       --usp 1 --ring 1       --text-encoder-tp-size 2       --vae-patch-parallel-size 2       --vae-parallel-mode tile       --vae-use-tiling       --enable-distributed-layerwise-offload       --dlo-no-use-allgather       --dlo-resident-layers "${DLO_RESIDENT_LAYERS:-12}"       --enforce-eager       --diffusion-attention-backend "${ATTENTION_BACKEND:-CUDNN_ATTN}"
    ;;
  2x5090)
    exec vllm "${common[@]}"       --num-gpus 2       --tensor-parallel-size 2       --usp 1 --ring 1       --text-encoder-tp-size 2       --vae-patch-parallel-size 2       --vae-parallel-mode tile       --vae-use-tiling       --enable-distributed-layerwise-offload       --dlo-no-use-allgather       --dlo-resident-layers "${DLO_RESIDENT_LAYERS:-20}"       --enforce-eager       --diffusion-attention-backend "${ATTENTION_BACKEND:-CUDNN_ATTN}"
    ;;
  4gpu)
    exec vllm "${common[@]}"       --num-gpus 4       --usp 4 --ring 1       --text-encoder-tp-size 4       --vae-patch-parallel-size 4       --vae-parallel-mode tile       --vae-use-tiling
    ;;
  *)
    fail "unknown profile '${PROFILE}'. Use: single, 2x4090, 2x5090, or 4gpu"
    ;;
esac
