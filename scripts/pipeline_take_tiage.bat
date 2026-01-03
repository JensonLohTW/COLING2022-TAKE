@echo off
REM ==============================================================================
REM TAKE + DGCN3 完整流程腳本 (tiage 數據集) - Windows 版
REM 數據集: tiage (對話網絡數據集)
REM 功能: 一鍵執行 DGCN3 中心性導出 → TAKE 訓練 → 推論評估
REM ==============================================================================

setlocal EnableDelayedExpansion

REM 專案根目錄
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

REM 虛擬環境 Python 路徑
set VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe

REM 參數配置
set DATASET=tiage
set EXPERIMENT_NAME=TAKE_tiage
set CENTRALITY_ALPHA=1.5
set CENTRALITY_FEATURE_SET=all
set CENTRALITY_WINDOW=2
set NODE_ID_JSON=knowSelect\datasets\tiage\node_id.json
set DGCN_DATASET_NAME=tiage
set DGCN_ALPHAS=1.5

echo ============================================================
echo           TAKE + DGCN3 完整訓練流程
echo ============================================================
echo 數據集: %DATASET%
echo 實驗名稱: %EXPERIMENT_NAME%
echo 中心性 Alpha: %CENTRALITY_ALPHA%
echo.

REM 檢查虛擬環境
if not exist "%VENV_PYTHON%" (
    echo 錯誤: 找不到虛擬環境 Python: %VENV_PYTHON%
    echo 請先創建虛擬環境: python -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -r requirements.txt
    pause
    exit /b 1
)

REM 創建必要目錄
if not exist "knowSelect\output\%EXPERIMENT_NAME%\model" mkdir "knowSelect\output\%EXPERIMENT_NAME%\model"
if not exist "knowSelect\output\%EXPERIMENT_NAME%\ks_pred" mkdir "knowSelect\output\%EXPERIMENT_NAME%\ks_pred"
if not exist "knowSelect\output\%EXPERIMENT_NAME%\logs" mkdir "knowSelect\output\%EXPERIMENT_NAME%\logs"

REM 初始化 checkpoints.json
if not exist "knowSelect\output\%EXPERIMENT_NAME%\model\checkpoints.json" (
    echo {"time": []} > "knowSelect\output\%EXPERIMENT_NAME%\model\checkpoints.json"
)

echo [%date% %time%] 開始完整流程...

REM 執行 pipeline
"%VENV_PYTHON%" main.py pipeline ^
    --dataset %DATASET% ^
    --name %EXPERIMENT_NAME% ^
    --use-centrality ^
    --centrality-alpha %CENTRALITY_ALPHA% ^
    --centrality-feature-set %CENTRALITY_FEATURE_SET% ^
    --centrality-window %CENTRALITY_WINDOW% ^
    --node-id-json %NODE_ID_JSON% ^
    --dataset-name %DGCN_DATASET_NAME% ^
    --alphas %DGCN_ALPHAS%

if %ERRORLEVEL% neq 0 (
    echo [%date% %time%] 流程執行失敗! 錯誤碼: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo [%date% %time%] 完整流程執行成功!
echo ============================================================
echo 輸出位置:
echo   模型: knowSelect\output\%EXPERIMENT_NAME%\model\
echo   預測: knowSelect\output\%EXPERIMENT_NAME%\ks_pred\
echo   日誌: knowSelect\output\%EXPERIMENT_NAME%\logs\
pause
