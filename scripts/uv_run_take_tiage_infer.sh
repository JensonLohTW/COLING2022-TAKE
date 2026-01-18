#!/usr/bin/env bash
# 使用 uv 推論 tiage 的 TAKE（knowSelect，test = slice >= 8）
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "錯誤：找不到 uv。請先執行：bash scripts/uv_setup.sh"
  exit 1
fi

# 若 tiage.split 更新，需刪除舊 test_TAKE.pkl 讓其重新建構測試集 episodes。
DATA_DIR="knowSelect/datasets/tiage"
rm -f "${DATA_DIR}/test_TAKE.pkl"

uv run python main.py infer-take \
  --dataset tiage \
  --name TAKE_tiage_all_feats \
  --use-centrality \
  --centrality-alpha 1.5 \
  --centrality-feature-set all \
  --centrality-window 2 \
  --node-id-json datasets/tiage/node_id.json

