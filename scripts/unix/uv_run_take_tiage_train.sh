#!/usr/bin/env bash
# 使用 uv 訓練 tiage 的 TAKE（knowSelect，含中心性/社團/6維特徵）
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "錯誤：找不到 uv。請先執行：bash scripts/unix/uv_setup.sh"
  exit 1
fi

# 若切分規則或 tiage.split 更新，需刪除舊 pkl 讓其重新建構資料。
DATA_DIR="knowSelect/datasets/tiage"
rm -f \
  "${DATA_DIR}/train_TAKE.pkl" \
  "${DATA_DIR}/test_TAKE.pkl" \
  "${DATA_DIR}/query_TAKE.pkl" \
  "${DATA_DIR}/passage_TAKE.pkl"

uv run python main.py train-take \
  --dataset tiage \
  --name TAKE_tiage_all_feats \
  --use-centrality \
  --centrality-alpha 1.5 \
  --centrality-feature-set all \
  --centrality-window 2 \
  --node-id-json datasets/tiage/node_id.json

