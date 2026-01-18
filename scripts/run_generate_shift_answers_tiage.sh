#!/usr/bin/env bash
# 生成「每個 shift 事件」的 GPT-2 回答文字檔（tiage）
# 輸入：knowSelect/output/<name>/metrics/shift_top3.jsonl
# 輸出：knowSelect/output/<name>/metrics/shift_answers_<epoch>_<dataset>.txt
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
  echo "錯誤：找不到虛擬環境 Python：$VENV_PYTHON"
  exit 1
fi

DATASET="tiage"
EXPERIMENT_NAME="TAKE_tiage_all_feats"
# 注意：shift_top3.jsonl 內的 dataset 欄位其實是 split（例如 test）
SPLIT="test"
# all 表示不過濾（把所有 epoch 都生成到同一份 txt 裡）
EPOCH="all"

# GPT-2 模型來源：可改成本地 fine-tuned checkpoint 目錄
GPT2_MODEL="gpt2"

$VENV_PYTHON main.py generate-shift-answers \
  --dataset "$DATASET" \
  --name "$EXPERIMENT_NAME" \
  --split "$SPLIT" \
  --epoch "$EPOCH" \
  --gpt2-model "$GPT2_MODEL"

echo "完成：已生成 shift 回答文字檔。"

