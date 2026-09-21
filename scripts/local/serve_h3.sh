#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-a100-1}"
MODEL_PATH="${MODEL_PATH:-/mnt/DataPart/jianghongda/checkpoint/MiniMax-H3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TASK_TYPE="${TASK_TYPE:-fl2va}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VENV_DIR="${VENV_DIR:-.venv-h3}"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "${MODEL_PATH}" ]] || fail "model directory not found: ${MODEL_PATH}"
[[ -f "${MODEL_PATH}/model_index.json" ]] || fail "missing ${MODEL_PATH}/model_index.json"
[[ -d "${MODEL_PATH}/FL2VA" ]] || fail "missing ${MODEL_PATH}/FL2VA"
[[ "${TASK_TYPE}" == "fl2va" || "${TASK_TYPE}" == "ref2va" ]] ||
  fail "TASK_TYPE must be fl2va or ref2va"
[[ "${TASK_TYPE}" != "ref2va" || -d "${MODEL_PATH}/Ref2VA" ]] ||
  fail "missing ${MODEL_PATH}/Ref2VA"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi
command -v vllm >/dev/null || fail "vllm not found in the offline environment"
command -v ffmpeg >/dev/null || fail "ffmpeg not found"

# Force local-only loading. Any missing file fails immediately instead of
# silently trying Hugging Face.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-14400}"

common=(
  serve "${MODEL_PATH}" --omni --task-type "${TASK_TYPE}"
  --host "${HOST}" --port "${PORT}" --trust-remote-code
)

case "${PROFILE}" in
  a100-1|single)
    # Works through CPU offload; plan for roughly 160+ GiB available host RAM.
    exec vllm "${common[@]}"       --num-gpus 1       --enable-cpu-offload       --diffusion-attention-backend FLASH_ATTN
    ;;
  a100-2)
    # Conservative two-card path for both A100 40GB and 80GB.
    exec vllm "${common[@]}"       --num-gpus 2 --tensor-parallel-size 2       --usp 1 --ring 1 --text-encoder-tp-size 2       --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling       --enable-distributed-layerwise-offload --dlo-no-use-allgather       --dlo-resident-layers "${DLO_RESIDENT_LAYERS:-20}"       --enforce-eager --diffusion-attention-backend FLASH_ATTN
    ;;
  a100-4|4gpu)
    exec vllm "${common[@]}"       --num-gpus 4 --usp 4 --ring 1       --text-encoder-tp-size 4       --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling       --diffusion-attention-backend FLASH_ATTN
    ;;
  a100-8)
    exec vllm "${common[@]}"       --num-gpus 8 --usp 8 --ring 1       --text-encoder-tp-size 8       --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling       --diffusion-attention-backend FLASH_ATTN
    ;;
  2x4090)
    exec vllm "${common[@]}"       --num-gpus 2 --tensor-parallel-size 2 --usp 1 --ring 1       --text-encoder-tp-size 2       --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling       --enable-distributed-layerwise-offload --dlo-no-use-allgather       --dlo-resident-layers "${DLO_RESIDENT_LAYERS:-12}"       --enforce-eager --diffusion-attention-backend CUDNN_ATTN
    ;;
  2x5090)
    exec vllm "${common[@]}"       --num-gpus 2 --tensor-parallel-size 2 --usp 1 --ring 1       --text-encoder-tp-size 2       --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling       --enable-distributed-layerwise-offload --dlo-no-use-allgather       --dlo-resident-layers "${DLO_RESIDENT_LAYERS:-20}"       --enforce-eager --diffusion-attention-backend CUDNN_ATTN
    ;;
  *)
    fail "unknown profile '${PROFILE}'. Use a100-1, a100-2, a100-4, a100-8, 2x4090, or 2x5090"
    ;;
esac
