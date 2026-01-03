#!/usr/bin/env bash
# 依序執行消融實驗：純文本 / 文本+imp_pct / 文本+全部特徵
# 結果會寫入 output/*/metrics/ablation_results.csv
set -e

python main.py ablation \
  --dataset tiage \
  --centrality-alpha 1.5 \
  --centrality-window 2 \
  --node-id-json datasets/tiage/node_id.json
