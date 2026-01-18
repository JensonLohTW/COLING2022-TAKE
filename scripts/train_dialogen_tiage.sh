#!/bin/bash
# ==============================================================================
# TAKE 對話生成訓練腳本 (dialogen + tiage)
# 數據集: tiage
# 功能: 使用 GPT-2 進行對話生成，基於 knowSelect 的知識選擇結果
#
# 完整流程:
#   1. knowSelect → 話題轉移檢測 + 知識選擇
#   2. dialogen → GPT-2 對話生成
# ==============================================================================

set -e

# 專案根目錄
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 虛擬環境路徑
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

# 參數配置
DATASET="tiage"
NAME="TAKE_GEN_tiage"
GPT2_DATA_PATH="datasets/tiage/"
MODE="train"
GPU=0

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}          TAKE 對話生成訓練 (GPT-2 + tiage)                  ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "${YELLOW}數據集: ${DATASET}${NC}"
echo -e "${YELLOW}模式: ${MODE}${NC}"
echo ""

# 檢查虛擬環境
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}錯誤: 找不到虛擬環境 Python: $VENV_PYTHON${NC}"
    exit 1
fi

# 創建輸出目錄
mkdir -p "dialogen/output/${NAME}/gen_model"
mkdir -p "dialogen/output/${NAME}/result_gen"

# 初始化 checkpoints.json
if [ ! -f "dialogen/output/${NAME}/gen_model/checkpoints.json" ]; then
    echo '{"time": []}' > "dialogen/output/${NAME}/gen_model/checkpoints.json"
fi

# 初始化 finished_inference.json
if [ ! -f "dialogen/output/${NAME}/gen_model/finished_inference.json" ]; then
    echo '{"time": []}' > "dialogen/output/${NAME}/gen_model/finished_inference.json"
fi

echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 開始 GPT-2 對話生成訓練...${NC}"

cd dialogen

$VENV_PYTHON TAKE/Run.py \
    --mode "$MODE" \
    --dataset "$DATASET" \
    --name "$NAME" \
    --gpt2_data_path "$GPT2_DATA_PATH" \
    --GPU "$GPU" \
    --gen_epoches 10 \
    --train_batch_size 4 \
    --accumulation_steps 16

if [ $? -eq 0 ]; then
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] GPT-2 對話生成訓練完成!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo -e "${CYAN}模型保存位置: dialogen/output/${NAME}/gen_model/${NC}"
else
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] 訓練失敗!${NC}"
    exit 1
fi
