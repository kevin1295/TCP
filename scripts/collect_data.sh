#!/bin/bash
# ============================================================
# ① Roach 采集训练集
# 启动 CARLA 后用 Roach RL agent 在 Leaderboard 上采集数据
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 加载环境变量
source "${SCRIPT_DIR}/env.sh" 2>/dev/null || true

cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Roach 数据采集"
echo "  确保 CARLA 服务已在运行 (scripts/start_carla.sh)"
echo "============================================================"

# 转发到 leaderboard 的数据采集脚本
exec bash leaderboard/scripts/data_collection.sh "$@"
