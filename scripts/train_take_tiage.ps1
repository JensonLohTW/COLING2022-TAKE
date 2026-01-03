# ==============================================================================
# TAKE + DGCN3 模型訓練腳本 (tiage 數據集) - PowerShell 版
# 數據集: tiage (對話網絡數據集)
# 功能: 訓練啟用中心性特徵的 TAKE 模型
# ==============================================================================

$ErrorActionPreference = "Stop"

# 專案根目錄
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $PROJECT_ROOT

# 虛擬環境 Python 路徑
$VENV_PYTHON = "$PROJECT_ROOT\.venv\Scripts\python.exe"

# 訓練參數配置
$DATASET = "tiage"
$EXPERIMENT_NAME = "TAKE_tiage"
$CENTRALITY_ALPHA = 1.5
$CENTRALITY_FEATURE_SET = "all"
$CENTRALITY_WINDOW = 2
$NODE_ID_JSON = "knowSelect\datasets\tiage\node_id.json"

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 開始 TAKE 模型訓練..." -ForegroundColor Green
Write-Host "數據集: $DATASET" -ForegroundColor Yellow
Write-Host "實驗名稱: $EXPERIMENT_NAME" -ForegroundColor Yellow
Write-Host "中心性 Alpha: $CENTRALITY_ALPHA" -ForegroundColor Yellow
Write-Host "特徵集: $CENTRALITY_FEATURE_SET" -ForegroundColor Yellow
Write-Host ""

# 檢查虛擬環境
if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "錯誤: 找不到虛擬環境 Python: $VENV_PYTHON" -ForegroundColor Red
    Write-Host "請先創建虛擬環境: python -m venv .venv; .venv\Scripts\Activate.ps1; pip install -r requirements.txt"
    exit 1
}

# 檢查數據集
if (-not (Test-Path "knowSelect\datasets\$DATASET")) {
    Write-Host "錯誤: 找不到數據集目錄: knowSelect\datasets\$DATASET" -ForegroundColor Red
    exit 1
}

# 創建輸出目錄
New-Item -ItemType Directory -Force -Path "knowSelect\output\$EXPERIMENT_NAME\model" | Out-Null
New-Item -ItemType Directory -Force -Path "knowSelect\output\$EXPERIMENT_NAME\ks_pred" | Out-Null
New-Item -ItemType Directory -Force -Path "knowSelect\output\$EXPERIMENT_NAME\logs" | Out-Null

# 初始化 checkpoints.json (如果不存在)
$checkpointsPath = "knowSelect\output\$EXPERIMENT_NAME\model\checkpoints.json"
if (-not (Test-Path $checkpointsPath)) {
    Set-Content -Path $checkpointsPath -Value '{"time": []}'
}

# 執行訓練
& $VENV_PYTHON main.py train-take `
    --dataset $DATASET `
    --name $EXPERIMENT_NAME `
    --use-centrality `
    --centrality-alpha $CENTRALITY_ALPHA `
    --centrality-feature-set $CENTRALITY_FEATURE_SET `
    --centrality-window $CENTRALITY_WINDOW `
    --node-id-json $NODE_ID_JSON

if ($LASTEXITCODE -ne 0) {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 訓練失敗! 錯誤碼: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 訓練完成!" -ForegroundColor Green
Write-Host "模型輸出: knowSelect\output\$EXPERIMENT_NAME\model\" -ForegroundColor Green
Write-Host "訓練日誌: knowSelect\output\$EXPERIMENT_NAME\logs\" -ForegroundColor Green
