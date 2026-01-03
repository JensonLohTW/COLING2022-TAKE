#!/bin/bash
# ==============================================================================
# TAKE + DGCN3 模型推論腳本 (tiage 數據集)
# 數據集: tiage (對話網絡數據集)
# 功能: 使用訓練好的模型進行推論和評估
# ==============================================================================

set -e

# 專案根目錄
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 虛擬環境路徑
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

# 推論參數配置
DATASET="tiage"
EXPERIMENT_NAME="TAKE_tiage"
CENTRALITY_ALPHA=1.5
CENTRALITY_FEATURE_SET="all"
CENTRALITY_WINDOW=2
NODE_ID_JSON="knowSelect/datasets/tiage/node_id.json"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 開始 TAKE 模型推論...${NC}"
echo -e "${YELLOW}數據集: ${DATASET}${NC}"
echo -e "${YELLOW}實驗名稱: ${EXPERIMENT_NAME}${NC}"

# 檢查虛擬環境
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}錯誤: 找不到虛擬環境 Python: $VENV_PYTHON${NC}"
    exit 1
fi

# 檢查模型是否存在
if [ ! -d "knowSelect/output/${EXPERIMENT_NAME}/model" ]; then
    echo -e "${RED}錯誤: 找不到訓練好的模型: knowSelect/output/${EXPERIMENT_NAME}/model${NC}"
    echo "請先運行訓練腳本: ./scripts/train_take_tiage.sh"
    exit 1
fi

# 執行推論
$VENV_PYTHON main.py infer-take \
    --dataset "$DATASET" \
    --name "$EXPERIMENT_NAME" \
    --use-centrality \
    --centrality-alpha "$CENTRALITY_ALPHA" \
    --centrality-feature-set "$CENTRALITY_FEATURE_SET" \
    --centrality-window "$CENTRALITY_WINDOW" \
    --node-id-json "$NODE_ID_JSON"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 推論完成!${NC}"
    echo -e "${GREEN}預測結果: knowSelect/output/${EXPERIMENT_NAME}/ks_pred/${NC}"
else
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] 推論失敗!${NC}"
    exit 1
fi
