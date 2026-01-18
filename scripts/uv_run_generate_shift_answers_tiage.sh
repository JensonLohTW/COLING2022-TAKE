#!/usr/bin/env bash
# 使用 uv 生成「每個 shift 事件」的 GPT-2 回答文字檔（tiage）
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "錯誤：找不到 uv。請先執行：bash scripts/uv_setup.sh"
  exit 1
fi

DATASET="tiage"
EXPERIMENT_NAME="TAKE_tiage_all_feats"
SPLIT="test"
EPOCH="all"
GPT2_MODEL="gpt2"

uv run python main.py generate-shift-answers \
  --dataset "$DATASET" \
  --name "$EXPERIMENT_NAME" \
  --split "$SPLIT" \
  --epoch "$EPOCH" \
  --gpt2-model "$GPT2_MODEL"

echo "完成：已生成 shift 回答文字檔。"

