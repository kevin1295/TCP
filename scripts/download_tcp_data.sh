#!/bin/bash
# ============================================================
# 下载 tcp_carla_data 数据集 (123 GB, 3 个分卷)
# 来源: https://huggingface.co/datasets/craigwu/tcp_carla_data
# 镜像: hf-mirror.com (国内加速)
# 依赖: aria2 或 wget, sha256sum
# ============================================================
set -euo pipefail

# ---- 配置 ----
MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO="datasets/craigwu/tcp_carla_data"
OUTPUT_DIR="${1:-./tcp_carla_data}"

# ---- 文件列表 & 预期 SHA-256 ----
declare -A FILES=(
  ["tcp_carla_data_part_aa"]="9f5e0adfe3a27e231a655a2503e1c5e28fb165d27484b896a6e5ce8f3147952b"
  ["tcp_carla_data_part_ab"]="ce79c6ad299d78884c37467125ba0413a0451b95e808afd365732bbb43fbdfdd"
  ["tcp_carla_data_part_ac"]="09295b608791cde259f0d0bcc0fe812116382a0a039334e38461a209187af146"
)

# ---- 颜色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "============================================================"
echo "  TCP CARLA 数据集下载"
echo "  镜像: $MIRROR"
echo "  保存到: $(pwd)"
echo "============================================================"
echo ""

# ---- 检测下载工具 ----
DOWNLOADER=""
if command -v aria2c &>/dev/null; then
  DOWNLOADER="aria2c"
  echo -e "${GREEN}[OK]${NC} 检测到 aria2c (多线程+断点续传)"
elif command -v wget &>/dev/null; then
  DOWNLOADER="wget"
  echo -e "${YELLOW}[OK]${NC} 检测到 wget (单线程+断点续传)"
elif command -v curl &>/dev/null; then
  DOWNLOADER="curl"
  echo -e "${YELLOW}[OK]${NC} 检测到 curl (不支持断点续传，不推荐)"
  echo "建议先安装 aria2: sudo apt install aria2  或  brew install aria2"
else
  echo -e "${RED}[ERROR]${NC} 未找到下载工具，请安装 aria2, wget 或 curl"
  exit 1
fi

# ---- 逐文件下载 ----
FAILED=()
for FILE in "${!FILES[@]}"; do
  EXPECTED_HASH="${FILES[$FILE]}"
  URL="${MIRROR}/${REPO}/resolve/main/${FILE}?download=true"
  echo ""
  echo "------------------------------------------------------------"
  echo "  📥 $FILE"
  echo "  SHA-256: $EXPECTED_HASH"
  echo "------------------------------------------------------------"

  # ---- 检查是否已存在且完整 ----
  if [ -f "$FILE" ]; then
    echo -n "  文件已存在，校验中... "
    ACTUAL_HASH=$(sha256sum "$FILE" | awk '{print $1}')
    if [ "$ACTUAL_HASH" = "$EXPECTED_HASH" ]; then
      echo -e "${GREEN}✅ 已完整，跳过下载${NC}"
      continue
    else
      echo -e "${YELLOW}⚠️  不完整/已损坏，重新下载${NC}"
      rm -f "$FILE" "$FILE.aria2"
    fi
  fi

  # ---- 下载 ----
  RETRY=0
  MAX_RETRIES=3
  while [ $RETRY -lt $MAX_RETRIES ]; do
    echo "  下载中 (尝试 $((RETRY + 1))/$MAX_RETRIES)..."
    case "$DOWNLOADER" in
      aria2c)
        # aria2: 16 线程, 断点续传, 重试 5 次
        aria2c \
          --continue=true \
          --max-connection-per-server=16 \
          --split=16 \
          --min-split-size=1M \
          --max-tries=5 \
          --retry-wait=10 \
          --connect-timeout=30 \
          --timeout=120 \
          --allow-overwrite=false \
          --auto-file-renaming=false \
          --summary-interval=30 \
          "$URL" \
          --out="$FILE"
        ;;
      wget)
        # wget: 断点续传, 重试
        wget \
          --continue \
          --tries=5 \
          --timeout=120 \
          --progress=bar:force \
          -O "$FILE" \
          "$URL"
        ;;
      curl)
        curl \
          --retry 5 \
          --retry-delay 10 \
          --connect-timeout 30 \
          --max-time 0 \
          -C - \
          -L \
          -o "$FILE" \
          "$URL"
        ;;
    esac

    # ---- 校验 ----
    echo -n "  校验 SHA-256... "
    ACTUAL_HASH=$(sha256sum "$FILE" | awk '{print $1}')
    if [ "$ACTUAL_HASH" = "$EXPECTED_HASH" ]; then
      echo -e "${GREEN}✅ 通过${NC}"
      break
    else
      echo -e "${RED}❌ 不匹配${NC}"
      echo "    预期: $EXPECTED_HASH"
      echo "    实际: $ACTUAL_HASH"
      RETRY=$((RETRY + 1))
      if [ $RETRY -lt $MAX_RETRIES ]; then
        echo "    将在 5 秒后重试..."
        sleep 5
      else
        echo -e "${RED}    已达最大重试次数，标记为失败${NC}"
        FAILED+=("$FILE")
      fi
    fi
  done
  echo ""
done

# ---- 结果汇总 ----
echo ""
echo "============================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo -e "${GREEN}  ✅ 全部下载完成并校验通过！${NC}"
  echo ""
  echo "  文件清单:"
  for FILE in "${!FILES[@]}"; do
    SIZE=$(du -h "$FILE" | cut -f1)
    echo "    $FILE  ($SIZE)"
  done
else
  echo -e "${RED}  ❌ 以下文件下载失败:${NC}"
  for FILE in "${FAILED[@]}"; do
    echo "    - $FILE"
  done
  echo ""
  echo "  请重新运行本脚本继续下载（已下载的部分不会丢失）"
  exit 1
fi
echo "============================================================"
