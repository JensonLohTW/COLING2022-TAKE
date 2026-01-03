#!/usr/bin/env bash
# 一鍵流程：匯出中心性 → 訓練 → 推論（含 6 維結構特徵）
# 建議先確認 DGCN3 模型檔已存在
set -e

python main.py pipeline \
  --dataset tiage \
  --name TAKE_tiage_all_feats \
  --use-centrality \
  --centrality-alpha 1.5 \
  --centrality-feature-set all \
  --centrality-window 2 \
  --node-id-json datasets/tiage/node_id.json \
  --dataset-name tiage \
  --alphas 1.5
