#!/usr/bin/env bash
# 依 dialog_id（數值排序）每 50 dialogs 分箱產生 tiage.split（TAKE：train=0..7，test>=8）
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "錯誤：找不到 uv。請先執行：bash scripts/unix/uv_setup.sh"
  exit 1
fi

ANNO_CSV="${PROJECT_ROOT}/demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv"
OUT_SPLIT="${PROJECT_ROOT}/knowSelect/datasets/tiage/tiage.split"

uv run python tools/generate_tiage_split_by_dialog_slices.py \
  --anno-csv "$ANNO_CSV" \
  --out-split "$OUT_SPLIT" \
  --dialogs-per-slice 50 \
  --train-max-slice 7

echo "[OK] 已更新：$OUT_SPLIT"

