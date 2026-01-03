@echo off
REM ==============================================================================
REM TAKE + DGCN3 模型推論腳本 (tiage 數據集) - Windows 版
REM 數據集: tiage (對話網絡數據集)
REM 功能: 使用訓練好的模型進行推論和評估
REM ==============================================================================

setlocal EnableDelayedExpansion

REM 專案根目錄
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

REM 虛擬環境 Python 路徑
set VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe

REM 推論參數配置
set DATASET=tiage
set EXPERIMENT_NAME=TAKE_tiage
set CENTRALITY_ALPHA=1.5
set CENTRALITY_FEATURE_SET=all
set CENTRALITY_WINDOW=2
set NODE_ID_JSON=knowSelect\datasets\tiage\node_id.json

echo [%date% %time%] 開始 TAKE 模型推論...
echo 數據集: %DATASET%
echo 實驗名稱: %EXPERIMENT_NAME%
echo.

REM 檢查虛擬環境
if not exist "%VENV_PYTHON%" (
    echo 錯誤: 找不到虛擬環境 Python: %VENV_PYTHON%
    pause
    exit /b 1
)

REM 檢查模型是否存在
if not exist "knowSelect\output\%EXPERIMENT_NAME%\model" (
    echo 錯誤: 找不到訓練好的模型: knowSelect\output\%EXPERIMENT_NAME%\model
    echo 請先運行訓練腳本: scripts\train_take_tiage.bat
    pause
    exit /b 1
)

REM 執行推論
"%VENV_PYTHON%" main.py infer-take ^
    --dataset %DATASET% ^
    --name %EXPERIMENT_NAME% ^
    --use-centrality ^
    --centrality-alpha %CENTRALITY_ALPHA% ^
    --centrality-feature-set %CENTRALITY_FEATURE_SET% ^
    --centrality-window %CENTRALITY_WINDOW% ^
    --node-id-json %NODE_ID_JSON%

if %ERRORLEVEL% neq 0 (
    echo [%date% %time%] 推論失敗! 錯誤碼: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [%date% %time%] 推論完成!
echo 預測結果: knowSelect\output\%EXPERIMENT_NAME%\ks_pred\
pause
