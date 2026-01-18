#!/usr/bin/env bash
# 一鍵流程：匯出中心性 → 訓練 → 推論（含 6 維結構特徵）
# 建議先確認 DGCN3 模型檔已存在
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
  echo "錯誤：找不到虛擬環境 Python：$VENV_PYTHON"
  exit 1
fi

# 先依 dialog_id 產生/更新 tiage.split（按時間片切分）
bash scripts/generate_tiage_split.sh

$VENV_PYTHON main.py pipeline \
  --dataset tiage \
  --name TAKE_tiage_all_feats \
  --use-centrality \
  --centrality-alpha 1.5 \
  --centrality-feature-set all \
  --centrality-window 2 \
  --node-id-json datasets/tiage/node_id.json \
  --dataset-name tiage \
  --alphas 1.5
