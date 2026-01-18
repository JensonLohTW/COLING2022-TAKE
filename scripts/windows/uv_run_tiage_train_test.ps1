<#
Tiage 完整訓練 + 測試流程（uv 版）
  1) uv sync
  2) 生成 tiage.split
  3) DGCN3 匯出中心性（所有 slices）
  4) TAKE 訓練（train）
  5) TAKE 測試推論（test）
  6) GPT-2 生成 shift 事件回答
  7) Smoke Check
#>
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

& (Join-Path $PSScriptRoot "uv_setup.ps1")
& (Join-Path $PSScriptRoot "uv_generate_tiage_split.ps1")
& (Join-Path $PSScriptRoot "uv_run_export_centrality_tiage.ps1")
& (Join-Path $PSScriptRoot "uv_run_take_tiage_train.ps1")
& (Join-Path $PSScriptRoot "uv_run_take_tiage_infer.ps1")
& (Join-Path $PSScriptRoot "uv_run_generate_shift_answers_tiage.ps1")
& (Join-Path $PSScriptRoot "uv_smoke_check_tiage_outputs.ps1")

Write-Host "[OK] Tiage 完整訓練/測試流程完成"

