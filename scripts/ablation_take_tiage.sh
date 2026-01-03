#!/bin/bash
# ==============================================================================
# TAKE 消融實驗腳本 (tiage 數據集)
# 數據集: tiage (對話網絡數據集)
# 功能: 運行三組消融實驗對比不同特徵配置的效果
#   - A1: 純文本基線 (不使用中心性)
#   - A2: 僅使用 imp_pct 特徵
#   - A3: 使用全部 6 維結構特徵
#
# 輸出結果:
#   - 話題轉移預測結果 (0=不轉移, 1=轉移)
#   - Top-K 高中心性句子及內容
#   - Precision、Recall、F1 評價指標
#   - 消融實驗對比報告
# ==============================================================================

set -e

# 專案根目錄
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 虛擬環境路徑
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

# 參數配置
DATASET="tiage"
CENTRALITY_ALPHA=1.5
CENTRALITY_WINDOW=2
NODE_ID_JSON="knowSelect/datasets/tiage/node_id.json"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}          TAKE 消融實驗 + 話題轉移檢測評估                  ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "${YELLOW}數據集: ${DATASET}${NC}"
echo -e "${YELLOW}評價指標: Precision / Recall / F1${NC}"
echo ""

# 檢查虛擬環境
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}錯誤: 找不到虛擬環境 Python: $VENV_PYTHON${NC}"
    exit 1
fi

echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 開始消融實驗...${NC}"
echo ""

# ==============================================================================
# 實驗配置
# ==============================================================================
declare -a EXPERIMENTS=(
    "TAKE_tiage_text_only:none:false:純文本基線"
    "TAKE_tiage_imp_pct:imp_pct:true:文本+imp_pct"
    "TAKE_tiage_all_feats:all:true:文本+6維結構特徵"
)

# ==============================================================================
# 運行消融實驗
# ==============================================================================
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r NAME FEATURE_SET USE_CENTRALITY DESC <<< "$exp"
    
    echo -e "${CYAN}------------------------------------------------------------${NC}"
    echo -e "${CYAN}實驗: ${DESC} (${NAME})${NC}"
    echo -e "${CYAN}特徵集: ${FEATURE_SET}${NC}"
    echo -e "${CYAN}------------------------------------------------------------${NC}"
    
    # 創建輸出目錄
    mkdir -p "knowSelect/output/${NAME}/model"
    mkdir -p "knowSelect/output/${NAME}/ks_pred"
    mkdir -p "knowSelect/output/${NAME}/logs"
    mkdir -p "knowSelect/output/${NAME}/metrics"
    
    # 初始化 checkpoints.json
    if [ ! -f "knowSelect/output/${NAME}/model/checkpoints.json" ]; then
        echo '{"time": []}' > "knowSelect/output/${NAME}/model/checkpoints.json"
    fi
    
    # 構建命令
    if [ "$USE_CENTRALITY" == "true" ]; then
        CENTRALITY_FLAG="--use-centrality --centrality-alpha ${CENTRALITY_ALPHA} --centrality-feature-set ${FEATURE_SET} --centrality-window ${CENTRALITY_WINDOW} --node-id-json ${NODE_ID_JSON}"
    else
        CENTRALITY_FLAG=""
    fi
    
    # 訓練
    echo -e "${YELLOW}[訓練] ${NAME}${NC}"
    $VENV_PYTHON main.py train-take \
        --dataset "$DATASET" \
        --name "$NAME" \
        $CENTRALITY_FLAG || {
        echo -e "${RED}訓練失敗: ${NAME}${NC}"
        continue
    }
    
    # 推論
    echo -e "${YELLOW}[推論] ${NAME}${NC}"
    $VENV_PYTHON main.py infer-take \
        --dataset "$DATASET" \
        --name "$NAME" \
        $CENTRALITY_FLAG || {
        echo -e "${RED}推論失敗: ${NAME}${NC}"
        continue
    }
    
    echo -e "${GREEN}[完成] ${NAME}${NC}"
    echo ""
done

# ==============================================================================
# 生成彙總報告
# ==============================================================================
echo -e "${BLUE}------------------------------------------------------------${NC}"
echo -e "${BLUE}生成消融實驗彙總報告${NC}"
echo -e "${BLUE}------------------------------------------------------------${NC}"

cd knowSelect
$VENV_PYTHON TAKE/summarize_ablation.py --output_dir output/
cd ..

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 消融實驗完成!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "${CYAN}結果文件位置:${NC}"
echo -e "  Precision/Recall/F1: knowSelect/output/*/metrics/shift_metrics.json"
echo -e "  Top-K 高中心性句子: knowSelect/output/*/metrics/shift_top3.jsonl"
echo -e "  消融實驗 CSV:       knowSelect/output/*/metrics/ablation_results.csv"
echo -e "  彙總報告:           knowSelect/output/ablation_summary.md"
echo ""
echo -e "${CYAN}評價指標說明:${NC}"
echo -e "  Precision (精確率): 預測為話題轉移的樣本中，真正是話題轉移的比例"
echo -e "  Recall (召回率):    所有真實話題轉移樣本中，被正確預測的比例"
echo -e "  F1:                 Precision 和 Recall 的調和平均數"

