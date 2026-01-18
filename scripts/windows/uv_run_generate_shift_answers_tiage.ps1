<#
使用 uv 生成「每個 shift 事件」的 GPT-2 回答文字檔（tiage）
#>
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "找不到 uv。請先執行 scripts/windows/uv_setup.ps1"
}

$dataset = "tiage"
$name = "TAKE_tiage_all_feats"
$split = "test"
$epoch = "all"
$gpt2Model = "gpt2"

uv run python main.py generate-shift-answers `
  --dataset $dataset `
  --name $name `
  --split $split `
  --epoch $epoch `
  --gpt2-model $gpt2Model

Write-Host "完成：已生成 shift 回答文字檔。"

