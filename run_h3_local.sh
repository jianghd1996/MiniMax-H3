#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/DataPart/jianghongda/checkpoint/MiniMax-H3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-576}"
DURATION="${DURATION:-5}"
STEPS="${STEPS:-50}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" infer_h3_diffusers.py   --model-path "${MODEL_PATH}"   --output-dir "${OUTPUT_DIR}"   --width "${WIDTH}"   --height "${HEIGHT}"   --duration "${DURATION}"   --steps "${STEPS}"   --seed "${SEED}"   "$@"
