#!/usr/bin/env bash
# 匯出 tiage 各時間片中心性預測（DGCN3）
# 需先有已訓練的模型檔（demo/DGCN3/model_registry/node_importance_tiage.pkl）
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
  echo "錯誤：找不到虛擬環境 Python：$VENV_PYTHON"
  exit 1
fi

$VENV_PYTHON main.py export-centrality --dataset-name tiage --alphas 1.5
