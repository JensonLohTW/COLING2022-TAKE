#!/usr/bin/env bash
# 匯出 tiage 各時間片中心性預測（DGCN3）
# 需先有已訓練的模型檔（demo/DGCN3/model_registry/node_importance_tiage.pkl）
set -e

python main.py export-centrality --dataset-name tiage --alphas 1.5
