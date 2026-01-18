#!/usr/bin/env bash
# Tiage 完整訓練 + 測試流程（uv 版）
# - 產生 tiage.split（時間片切分）
# - DGCN3 匯出中心性（所有 slices）
# - TAKE 訓練（train）
# - TAKE 測試推論（test）
# - 生成 shift 事件 GPT-2 回答
# - Smoke Check
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "錯誤：找不到 uv。請先安裝 uv。"
  exit 1
fi

bash scripts/uv_setup.sh
bash scripts/uv_generate_tiage_split.sh
bash scripts/uv_run_export_centrality_tiage.sh
bash scripts/uv_run_take_tiage_train.sh
bash scripts/uv_run_take_tiage_infer.sh
bash scripts/uv_run_generate_shift_answers_tiage.sh
bash scripts/uv_smoke_check_tiage_outputs.sh

echo "[OK] Tiage 完整訓練/測試流程完成"

