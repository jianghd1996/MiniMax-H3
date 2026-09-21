#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/DataPart/jianghongda/checkpoint/MiniMax-H3}"
PORT="${PORT:-8000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-576}"
DURATION="${DURATION:-5}"
STEPS="${STEPS:-50}"
SEED="${SEED:-42}"
VENV_DIR="${VENV_DIR:-.venv-h3}"
SERVER_LOG="${SERVER_LOG:-${OUTPUT_DIR}/h3_server.log}"

fail() { echo "ERROR: $*" >&2; exit 1; }

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
command -v curl >/dev/null || fail "curl not found"
command -v python >/dev/null || fail "python not found"
[[ -d "${MODEL_PATH}/FL2VA" ]] || fail "missing ${MODEL_PATH}/FL2VA"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  gpu_count="${#gpu_ids[@]}"
else
  gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c 'A100' || true)"
  (( gpu_count > 0 )) || fail "no A100 detected"
  ids=()
  for ((i=0; i<gpu_count; i++)); do ids+=("$i"); done
  CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${ids[*]}")"
  export CUDA_VISIBLE_DEVICES
fi

case "${gpu_count}" in
  1|2|4|8) profile="a100-${gpu_count}" ;;
  *) fail "detected ${gpu_count} visible GPUs; set CUDA_VISIBLE_DEVICES to 1, 2, 4, or 8 A100s" ;;
esac

mkdir -p "${OUTPUT_DIR}"
server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    echo "Stopping local H3 server (PID ${server_pid})..."
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "Reusing H3 server on port ${PORT}."
else
  echo "Starting offline H3 server: ${profile}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "Server log: ${SERVER_LOG}"
  MODEL_PATH="${MODEL_PATH}" PORT="${PORT}"     bash scripts/local/serve_h3.sh "${profile}" >"${SERVER_LOG}" 2>&1 &
  server_pid=$!

  echo "Loading weights. This can take a long time..."
  ready=0
  for _ in $(seq 1 720); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "H3 server exited during startup. Last log lines:" >&2
      tail -n 80 "${SERVER_LOG}" >&2 || true
      exit 1
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 10
  done
  (( ready == 1 )) || fail "server was not ready after 2 hours; inspect ${SERVER_LOG}"
fi

echo
echo "H3 is ready. Paste one text prompt per generation."
echo "Press Enter on an empty line (or Ctrl-D) to stop."
echo

index=1
while true; do
  if ! IFS= read -r -p "Prompt> " prompt; then
    echo
    break
  fi
  [[ -n "${prompt//[[:space:]]/}" ]] || break

  timestamp="$(date +%Y%m%d_%H%M%S)"
  output="${OUTPUT_DIR}/h3_${timestamp}_${index}.mp4"
  python scripts/local/generate_t2va.py     --server "http://127.0.0.1:${PORT}"     --prompt "${prompt}"     --output "${output}"     --width "${WIDTH}" --height "${HEIGHT}"     --duration "${DURATION}" --steps "${STEPS}"     --seed "$((SEED + index - 1))"
  index=$((index + 1))
  echo
done
