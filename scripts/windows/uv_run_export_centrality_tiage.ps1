<#
使用 uv 匯出 tiage 各時間片中心性預測（DGCN3，對所有 slices）
#>
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "找不到 uv。請先執行 scripts/windows/uv_setup.ps1"
}

$logFile = $null
if ($env:ORCHESTRATED -ne "1") {
  if (-not $env:RUN_ID) { $env:RUN_ID = Get-Date -Format "yyyy-MM-dd_HH-mm-ss" }
  if (-not $env:RUN_LOG_DIR) { $env:RUN_LOG_DIR = Join-Path $projectRoot (Join-Path "logs" $env:RUN_ID) }
  if (-not $env:RUN_OUTPUT_DIR) { $env:RUN_OUTPUT_DIR = Join-Path $projectRoot (Join-Path "outputs" $env:RUN_ID) }
  New-Item -ItemType Directory -Force -Path $env:RUN_LOG_DIR | Out-Null
  New-Item -ItemType Directory -Force -Path $env:RUN_OUTPUT_DIR | Out-Null
  $logFile = Join-Path $env:RUN_LOG_DIR "02_export_centrality.log"
}

$outDir = Join-Path $projectRoot "demo\DGCN3\Centrality"
if ($env:RUN_OUTPUT_DIR) {
  $outDir = Join-Path $env:RUN_OUTPUT_DIR "dgcn3\Centrality"
}
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
if ($logFile) {
  uv run python main.py export-centrality --dataset-name tiage --alphas 1.5 --output-dir "$outDir" 2>&1 | Tee-Object -FilePath $logFile -Append
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
  uv run python main.py export-centrality --dataset-name tiage --alphas 1.5 --output-dir "$outDir"
}

if ($logFile) {
  Write-Host ("[OK] logs:    {0}" -f $env:RUN_LOG_DIR)
  Write-Host ("[OK] outputs: {0}" -f $env:RUN_OUTPUT_DIR)
}

