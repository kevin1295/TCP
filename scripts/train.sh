#!/bin/bash
# ============================================================
# ② TCP 训练脚本
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 默认参数
GPU="${1:-1}"
BATCH_SIZE="${2:-32}"
EPOCHS="${3:-60}"
LR="${4:-0.0001}"
ID="${5:-TCP}"

echo "============================================================"
echo "  TCP 训练"
echo "  GPUs: $GPU | Batch: $BATCH_SIZE | Epochs: $EPOCHS | LR: $LR"
echo "============================================================"

python TCP/train.py \
    --gpus "$GPU" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --id "$ID"
