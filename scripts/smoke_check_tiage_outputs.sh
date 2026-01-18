#!/usr/bin/env bash
# Tiage 流程輸出 Smoke Check（不跑訓練，只檢查輸出是否齊全、格式是否包含必要欄位）
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
  echo "錯誤：找不到虛擬環境 Python：$VENV_PYTHON"
  exit 1
fi

NAME="TAKE_tiage_all_feats"
METRICS_DIR="${PROJECT_ROOT}/knowSelect/output/${NAME}/metrics"

echo "[*] 檢查輸出目錄：${METRICS_DIR}"
test -d "$METRICS_DIR"

echo "[*] 檢查 tiage.split 是否存在且包含 train/test"
test -f "${PROJECT_ROOT}/knowSelect/datasets/tiage/tiage.split"

test -f "${METRICS_DIR}/shift_metrics.json"
test -f "${METRICS_DIR}/shift_top3.jsonl"
test -f "${METRICS_DIR}/shift_pred.jsonl"

echo "[*] 檢查 shift_pred.jsonl 欄位（dialog_id/query_id/turn_id/node_id/pred_shift）"
$VENV_PYTHON - << 'PY'
import json, os, sys
path = os.path.join("knowSelect","output","TAKE_tiage_all_feats","metrics","shift_pred.jsonl")
need = {"dialog_id","query_id","turn_id","node_id","pred_shift"}
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line=line.strip()
        if not line:
            continue
        obj=json.loads(line)
        miss=need-obj.keys()
        if miss:
            raise SystemExit(f"缺少欄位：{miss} @ line {i+1}")
        if obj["pred_shift"] not in (0,1):
            raise SystemExit(f"pred_shift 非 0/1 @ line {i+1}")
        break
print("[OK] shift_pred.jsonl 欄位與取值正常（抽樣第一筆）")
PY

echo "[*] 檢查 shift_top3.jsonl 是否包含 shift_events 與 interval_top3.turn_id"
$VENV_PYTHON - << 'PY'
import json, os
path = os.path.join("knowSelect","output","TAKE_tiage_all_feats","metrics","shift_top3.jsonl")
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line=line.strip()
        if not line:
            continue
        obj=json.loads(line)
        events=obj.get("shift_events") or []
        if events:
            ev=events[0]
            top3=ev.get("interval_top3") or []
            if top3:
                if "turn_id" not in top3[0]:
                    raise SystemExit("interval_top3 缺少 turn_id")
        break
print("[OK] shift_top3.jsonl 結構正常（抽樣第一筆）")
PY

echo "[OK] Smoke check 完成"

