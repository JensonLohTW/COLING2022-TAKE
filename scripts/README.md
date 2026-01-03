# Scripts 說明

此資料夾集中管理 tiage 相關的啟動腳本。請在專案根目錄執行這些腳本。

## 先決條件

- 需先完成 DGCN3 模型訓練，並產生：
  - `demo/DGCN3/model_registry/node_importance_tiage.pkl`
- `datasets/tiage/node_id.json` 必須存在
- 若已有舊的 `*_TAKE.pkl` 且缺少 `node_id`，請先移除再重建

## 腳本列表

- `run_export_centrality_tiage.sh`
  - 匯出 0~9 時間片中心性預測到 `demo/DGCN3/Centrality/alpha_1.5/`
- `run_take_tiage_train.sh`
  - 使用 6 維結構特徵訓練 tiage
- `run_take_tiage_infer.sh`
  - 推論並輸出 shift 指標與 top3
- `run_take_tiage_ablation.sh`
  - 依序跑三組消融設定，輸出 `ablation_results.csv`
- `run_tiage_pipeline.sh`
  - 一鍵流程（匯出中心性 → 訓練 → 推論）

## 範例

```bash
bash scripts/run_export_centrality_tiage.sh
bash scripts/run_take_tiage_train.sh
bash scripts/run_take_tiage_infer.sh
```

消融實驗：

```bash
bash scripts/run_take_tiage_ablation.sh
```

一鍵流程：

```bash
bash scripts/run_tiage_pipeline.sh
```
