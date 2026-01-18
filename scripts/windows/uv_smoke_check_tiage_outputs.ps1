<#
使用 uv 執行 Tiage 輸出 Smoke Check（不跑訓練，只檢查輸出是否齊全、欄位是否存在）
#>
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "找不到 uv。請先執行 scripts/windows/uv_setup.ps1"
}

$name = "TAKE_tiage_all_feats"
$metricsDir = Join-Path $projectRoot "knowSelect\output\$name\metrics"
$splitPath = Join-Path $projectRoot "knowSelect\datasets\tiage\tiage.split"

Write-Host "[*] 檢查輸出目錄：$metricsDir"
if (-not (Test-Path $metricsDir)) { throw "找不到 metrics 目錄：$metricsDir" }

Write-Host "[*] 檢查 tiage.split 是否存在"
if (-not (Test-Path $splitPath)) { throw "找不到 split：$splitPath" }

foreach ($f in @("shift_metrics.json","shift_top3.jsonl","shift_pred.jsonl")) {
  $p = Join-Path $metricsDir $f
  if (-not (Test-Path $p)) { throw "缺少輸出檔：$p" }
}

Write-Host "[*] 檢查 shift_pred.jsonl 欄位"
uv run python - << 'PY'
import json, os
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

Write-Host "[*] 檢查 shift_top3.jsonl 結構"
uv run python - << 'PY'
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
            if top3 and "turn_id" not in top3[0]:
                raise SystemExit("interval_top3 缺少 turn_id")
        break
print("[OK] shift_top3.jsonl 結構正常（抽樣第一筆）")
PY

Write-Host "[OK] Smoke check 完成"

