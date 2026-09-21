#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv-h3}"
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-third_party/vllm-omni}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

command -v ffmpeg >/dev/null || {
  echo "ERROR: ffmpeg is required. Install ffmpeg and retry." >&2
  exit 1
}
command -v git >/dev/null || {
  echo "ERROR: git is required." >&2
  exit 1
}

if ! command -v uv >/dev/null; then
  echo "ERROR: uv is required. Install it from https://docs.astral.sh/uv/ and retry." >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
uv pip install "vllm==0.26.0" requests

if [[ ! -d "${VLLM_OMNI_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${VLLM_OMNI_DIR}")"
  git clone --depth 1 https://github.com/vllm-project/vllm-omni.git "${VLLM_OMNI_DIR}"
else
  git -C "${VLLM_OMNI_DIR}" pull --ff-only
fi

uv pip install -e "${VLLM_OMNI_DIR}"

echo
echo "Environment ready."
echo "Activate it with: source ${VENV_DIR}/bin/activate"
echo "Then start H3 with: bash scripts/local/serve_h3.sh single"
