#!/usr/bin/env bash
# 以 6 維結構特徵訓練 tiage 的 TAKE（含中心性）
# 若已有舊的 *_TAKE.pkl 且缺少 node_id，請先移除再重建
set -e

python main.py train-take \
  --dataset tiage \
  --name TAKE_tiage_all_feats \
  --use-centrality \
  --centrality-alpha 1.5 \
  --centrality-feature-set all \
  --centrality-window 2 \
  --node-id-json datasets/tiage/node_id.json
