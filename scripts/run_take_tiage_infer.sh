#!/usr/bin/env bash
# 以 6 維結構特徵推論 tiage（含中心性）
# 會輸出 shift precision/recall/F1 與 shift_top3.jsonl
set -e

python main.py infer-take \
  --dataset tiage \
  --name TAKE_tiage_all_feats \
  --use-centrality \
  --centrality-alpha 1.5 \
  --centrality-feature-set all \
  --centrality-window 2 \
  --node-id-json datasets/tiage/node_id.json
