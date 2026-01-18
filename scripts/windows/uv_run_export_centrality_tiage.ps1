<#
使用 uv 匯出 tiage 各時間片中心性預測（DGCN3，對所有 slices）
#>
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "找不到 uv。請先執行 scripts/windows/uv_setup.ps1"
}

uv run python main.py export-centrality --dataset-name tiage --alphas 1.5

