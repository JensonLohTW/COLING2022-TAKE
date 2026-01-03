@echo off
REM ==============================================================================
REM TAKE 對話生成訓練腳本 (dialogen + tiage) - Windows 版
REM 數據集: tiage
REM 功能: 使用 GPT-2 進行對話生成，基於 knowSelect 的知識選擇結果
REM
REM 完整流程:
REM   1. knowSelect → 話題轉移檢測 + 知識選擇
REM   2. dialogen → GPT-2 對話生成
REM ==============================================================================

setlocal EnableDelayedExpansion

REM 專案根目錄
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

REM 虛擬環境 Python 路徑
set VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe

REM 參數配置
set DATASET=tiage
set NAME=TAKE_GEN_tiage
set GPT2_DATA_PATH=datasets/tiage/
set MODE=train
set GPU=0

echo ============================================================
echo           TAKE 對話生成訓練 (GPT-2 + tiage)
echo ============================================================
echo 數據集: %DATASET%
echo 模式: %MODE%
echo.

REM 檢查虛擬環境
if not exist "%VENV_PYTHON%" (
    echo 錯誤: 找不到虛擬環境 Python: %VENV_PYTHON%
    pause
    exit /b 1
)

REM 創建輸出目錄
if not exist "dialogen\output\%NAME%\gen_model" mkdir "dialogen\output\%NAME%\gen_model"
if not exist "dialogen\output\%NAME%\result_gen" mkdir "dialogen\output\%NAME%\result_gen"

REM 初始化 checkpoints.json
if not exist "dialogen\output\%NAME%\gen_model\checkpoints.json" (
    echo {"time": []} > "dialogen\output\%NAME%\gen_model\checkpoints.json"
)

REM 初始化 finished_inference.json
if not exist "dialogen\output\%NAME%\gen_model\finished_inference.json" (
    echo {"time": []} > "dialogen\output\%NAME%\gen_model\finished_inference.json"
)

echo [%date% %time%] 開始 GPT-2 對話生成訓練...

cd dialogen

"%VENV_PYTHON%" TAKE\Run.py ^
    --mode %MODE% ^
    --dataset %DATASET% ^
    --name %NAME% ^
    --gpt2_data_path %GPT2_DATA_PATH% ^
    --GPU %GPU% ^
    --gen_epoches 10 ^
    --train_batch_size 4 ^
    --accumulation_steps 16

if %ERRORLEVEL% neq 0 (
    echo [%date% %time%] 訓練失敗! 錯誤碼: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo [%date% %time%] GPT-2 對話生成訓練完成!
echo ============================================================
echo.
echo 模型保存位置: dialogen\output\%NAME%\gen_model\
pause
