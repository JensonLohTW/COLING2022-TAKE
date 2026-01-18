#!/usr/bin/env bash
# 使用 uv 匯出 tiage 各時間片中心性預測（DGCN3，對所有 slices）
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "錯誤：找不到 uv。請先執行：bash scripts/unix/uv_setup.sh"
  exit 1
fi

uv run python main.py export-centrality --dataset-name tiage --alphas 1.5

