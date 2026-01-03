@echo off
REM ==============================================================================
REM TAKE + DGCN3 模型訓練腳本 (tiage 數據集) - Windows 版
REM 數據集: tiage (對話網絡數據集)
REM 功能: 訓練啟用中心性特徵的 TAKE 模型
REM ==============================================================================

setlocal EnableDelayedExpansion

REM 專案根目錄 (根據實際情況修改)
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

REM 虛擬環境 Python 路徑
set VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe

REM 訓練參數配置
set DATASET=tiage
set EXPERIMENT_NAME=TAKE_tiage
set CENTRALITY_ALPHA=1.5
set CENTRALITY_FEATURE_SET=all
set CENTRALITY_WINDOW=2
set NODE_ID_JSON=knowSelect\datasets\tiage\node_id.json

echo [%date% %time%] 開始 TAKE 模型訓練...
echo 數據集: %DATASET%
echo 實驗名稱: %EXPERIMENT_NAME%
echo 中心性 Alpha: %CENTRALITY_ALPHA%
echo 特徵集: %CENTRALITY_FEATURE_SET%
echo.

REM 檢查虛擬環境
if not exist "%VENV_PYTHON%" (
    echo 錯誤: 找不到虛擬環境 Python: %VENV_PYTHON%
    echo 請先創建虛擬環境: python -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -r requirements.txt
    pause
    exit /b 1
)

REM 檢查數據集
if not exist "knowSelect\datasets\%DATASET%" (
    echo 錯誤: 找不到數據集目錄: knowSelect\datasets\%DATASET%
    pause
    exit /b 1
)

REM 創建輸出目錄
if not exist "knowSelect\output\%EXPERIMENT_NAME%\model" mkdir "knowSelect\output\%EXPERIMENT_NAME%\model"
if not exist "knowSelect\output\%EXPERIMENT_NAME%\ks_pred" mkdir "knowSelect\output\%EXPERIMENT_NAME%\ks_pred"
if not exist "knowSelect\output\%EXPERIMENT_NAME%\logs" mkdir "knowSelect\output\%EXPERIMENT_NAME%\logs"

REM 初始化 checkpoints.json (如果不存在)
if not exist "knowSelect\output\%EXPERIMENT_NAME%\model\checkpoints.json" (
    echo {"time": []} > "knowSelect\output\%EXPERIMENT_NAME%\model\checkpoints.json"
)

REM 執行訓練
"%VENV_PYTHON%" main.py train-take ^
    --dataset %DATASET% ^
    --name %EXPERIMENT_NAME% ^
    --use-centrality ^
    --centrality-alpha %CENTRALITY_ALPHA% ^
    --centrality-feature-set %CENTRALITY_FEATURE_SET% ^
    --centrality-window %CENTRALITY_WINDOW% ^
    --node-id-json %NODE_ID_JSON%

if %ERRORLEVEL% neq 0 (
    echo [%date% %time%] 訓練失敗! 錯誤碼: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [%date% %time%] 訓練完成!
echo 模型輸出: knowSelect\output\%EXPERIMENT_NAME%\model\
echo 訓練日誌: knowSelect\output\%EXPERIMENT_NAME%\logs\
pause
