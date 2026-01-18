<#
使用 uv 推論 tiage 的 TAKE（knowSelect，test = slice >= 8）
#>
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "找不到 uv。請先執行 scripts/windows/uv_setup.ps1"
}

# 若 tiage.split 更新，需刪除舊 test_TAKE.pkl
$dataDir = Join-Path $projectRoot "knowSelect\datasets\tiage"
Remove-Item -ErrorAction SilentlyContinue -Force (Join-Path $dataDir "test_TAKE.pkl")

uv run python main.py infer-take `
  --dataset tiage `
  --name TAKE_tiage_all_feats `
  --use-centrality `
  --centrality-alpha 1.5 `
  --centrality-feature-set all `
  --centrality-window 2 `
  --node-id-json datasets/tiage/node_id.json

