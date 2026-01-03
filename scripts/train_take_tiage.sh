#!/bin/bash
# ==============================================================================
# TAKE + DGCN3 模型訓練腳本 (tiage 數據集)
# 數據集: tiage (對話網絡數據集)
# 功能: 訓練啟用中心性特徵的 TAKE 模型
# ==============================================================================

set -e  # 遇到錯誤立即退出

# 專案根目錄 (根據實際情況修改)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 虛擬環境路徑
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

# 訓練參數配置
DATASET="tiage"
EXPERIMENT_NAME="TAKE_tiage"
CENTRALITY_ALPHA=1.5
CENTRALITY_FEATURE_SET="all"  # 可選: none, imp_pct, all
CENTRALITY_WINDOW=2
NODE_ID_JSON="knowSelect/datasets/tiage/node_id.json"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 開始 TAKE 模型訓練...${NC}"
echo -e "${YELLOW}數據集: ${DATASET}${NC}"
echo -e "${YELLOW}實驗名稱: ${EXPERIMENT_NAME}${NC}"
echo -e "${YELLOW}中心性 Alpha: ${CENTRALITY_ALPHA}${NC}"
echo -e "${YELLOW}特徵集: ${CENTRALITY_FEATURE_SET}${NC}"

# 檢查虛擬環境
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}錯誤: 找不到虛擬環境 Python: $VENV_PYTHON${NC}"
    echo "請先創建虛擬環境: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# 檢查數據集
if [ ! -d "knowSelect/datasets/${DATASET}" ]; then
    echo -e "${RED}錯誤: 找不到數據集目錄: knowSelect/datasets/${DATASET}${NC}"
    exit 1
fi

# 創建輸出目錄
mkdir -p "knowSelect/output/${EXPERIMENT_NAME}/model"
mkdir -p "knowSelect/output/${EXPERIMENT_NAME}/ks_pred"
mkdir -p "knowSelect/output/${EXPERIMENT_NAME}/logs"

# 初始化 checkpoints.json (如果不存在)
if [ ! -f "knowSelect/output/${EXPERIMENT_NAME}/model/checkpoints.json" ]; then
    echo '{"time": []}' > "knowSelect/output/${EXPERIMENT_NAME}/model/checkpoints.json"
fi

# 執行訓練
$VENV_PYTHON main.py train-take \
    --dataset "$DATASET" \
    --name "$EXPERIMENT_NAME" \
    --use-centrality \
    --centrality-alpha "$CENTRALITY_ALPHA" \
    --centrality-feature-set "$CENTRALITY_FEATURE_SET" \
    --centrality-window "$CENTRALITY_WINDOW" \
    --node-id-json "$NODE_ID_JSON"

# 檢查訓練結果
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 訓練完成!${NC}"
    echo -e "${GREEN}模型輸出: knowSelect/output/${EXPERIMENT_NAME}/model/${NC}"
    echo -e "${GREEN}訓練日誌: knowSelect/output/${EXPERIMENT_NAME}/logs/${NC}"
else
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] 訓練失敗!${NC}"
    exit 1
fi
