#!/usr/bin/env bash
# ============================================================
# vLLM model serving script for Battery Data Standardizer
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Isolate from broken system packages
unset PYTHONPATH
export PYTHONNOUSERSITE=1

MODEL="${MODEL:-LGAI-EXAONE/EXAONE-4.0-32B-FP8}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

echo "============================================"
echo "Battery Data Standardizer — LLM Server"
echo "============================================"
echo "Model:    ${MODEL}"
echo "Port:     ${PORT}"
echo "Max len:  ${MAX_MODEL_LEN}"
echo "GPU util: ${GPU_MEMORY_UTILIZATION}"
echo "============================================"

exec uv run vllm serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype auto \
    --trust-remote-code
