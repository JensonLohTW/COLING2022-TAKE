<#
依 dialog_id（數值排序）每 50 dialogs 分箱產生 tiage.split（TAKE：train=0..7，test>=8）
#>
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "找不到 uv。請先執行 scripts/windows/uv_setup.ps1"
}

$annoCsv = Join-Path $projectRoot "demo\tiage-1\outputs_nodes\tiage_anno_nodes_all.csv"
$outSplit = Join-Path $projectRoot "knowSelect\datasets\tiage\tiage.split"

uv run python tools/generate_tiage_split_by_dialog_slices.py `
  --anno-csv "$annoCsv" `
  --out-split "$outSplit" `
  --dialogs-per-slice 50 `
  --train-max-slice 7

Write-Host "[OK] 已更新：$outSplit"

