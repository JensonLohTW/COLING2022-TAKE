#!/bin/bash
# ==============================================================================
# TAKE + DGCN3 完整流程腳本 (tiage 數據集)
# 數據集: tiage (對話網絡數據集)
# 功能: 一鍵執行 DGCN3 中心性導出 → TAKE 訓練 → 推論評估
# ==============================================================================

set -e

# 專案根目錄
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 虛擬環境路徑
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

# 參數配置
DATASET="tiage"
EXPERIMENT_NAME="TAKE_tiage"
CENTRALITY_ALPHA=1.5
CENTRALITY_FEATURE_SET="all"
CENTRALITY_WINDOW=2
NODE_ID_JSON="knowSelect/datasets/tiage/node_id.json"
DGCN_DATASET_NAME="tiage"
DGCN_ALPHAS="1.5"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}          TAKE + DGCN3 完整訓練流程                          ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "${YELLOW}數據集: ${DATASET}${NC}"
echo -e "${YELLOW}實驗名稱: ${EXPERIMENT_NAME}${NC}"
echo -e "${YELLOW}中心性 Alpha: ${CENTRALITY_ALPHA}${NC}"
echo ""

# 檢查虛擬環境
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}錯誤: 找不到虛擬環境 Python: $VENV_PYTHON${NC}"
    echo "請先創建虛擬環境: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# 創建必要目錄
mkdir -p "knowSelect/output/${EXPERIMENT_NAME}/model"
mkdir -p "knowSelect/output/${EXPERIMENT_NAME}/ks_pred"
mkdir -p "knowSelect/output/${EXPERIMENT_NAME}/logs"

# 初始化 checkpoints.json
if [ ! -f "knowSelect/output/${EXPERIMENT_NAME}/model/checkpoints.json" ]; then
    echo '{"time": []}' > "knowSelect/output/${EXPERIMENT_NAME}/model/checkpoints.json"
fi

echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 開始完整流程...${NC}"

# 執行 pipeline
$VENV_PYTHON main.py pipeline \
    --dataset "$DATASET" \
    --name "$EXPERIMENT_NAME" \
    --use-centrality \
    --centrality-alpha "$CENTRALITY_ALPHA" \
    --centrality-feature-set "$CENTRALITY_FEATURE_SET" \
    --centrality-window "$CENTRALITY_WINDOW" \
    --node-id-json "$NODE_ID_JSON" \
    --dataset-name "$DGCN_DATASET_NAME" \
    --alphas "$DGCN_ALPHAS"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 完整流程執行成功!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${YELLOW}輸出位置:${NC}"
    echo -e "  模型: knowSelect/output/${EXPERIMENT_NAME}/model/"
    echo -e "  預測: knowSelect/output/${EXPERIMENT_NAME}/ks_pred/"
    echo -e "  日誌: knowSelect/output/${EXPERIMENT_NAME}/logs/"
else
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] 流程執行失敗!${NC}"
    exit 1
fi
