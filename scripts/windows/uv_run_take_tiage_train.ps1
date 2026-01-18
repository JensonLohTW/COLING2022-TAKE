<#
使用 uv 訓練 tiage 的 TAKE（knowSelect，含中心性/社團/6維特徵）
#>
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "找不到 uv。請先執行 scripts/windows/uv_setup.ps1"
}

# 清理舊的 *_TAKE.pkl，確保切分生效
$dataDir = Join-Path $projectRoot "knowSelect\datasets\tiage"
Remove-Item -ErrorAction SilentlyContinue -Force `
  (Join-Path $dataDir "train_TAKE.pkl"), `
  (Join-Path $dataDir "test_TAKE.pkl"), `
  (Join-Path $dataDir "query_TAKE.pkl"), `
  (Join-Path $dataDir "passage_TAKE.pkl")

uv run python main.py train-take `
  --dataset tiage `
  --name TAKE_tiage_all_feats `
  --use-centrality `
  --centrality-alpha 1.5 `
  --centrality-feature-set all `
  --centrality-window 2 `
  --node-id-json datasets/tiage/node_id.json

