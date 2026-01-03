@echo off
REM ==============================================================================
REM 環境設置腳本 - Windows 版
REM 功能: 創建虛擬環境並安裝依賴
REM ==============================================================================

setlocal EnableDelayedExpansion

REM 專案根目錄
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

echo [%date% %time%] 開始環境設置...

REM 檢查 Python 是否安裝
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 錯誤: 找不到 Python，請先安裝 Python 3.9+
    pause
    exit /b 1
)

REM 顯示 Python 版本
for /f "tokens=*" %%i in ('python --version') do echo Python 版本: %%i

REM 創建虛擬環境
if not exist ".venv" (
    echo 創建虛擬環境...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo 錯誤: 虛擬環境創建失敗
        pause
        exit /b 1
    )
    echo 虛擬環境創建成功!
) else (
    echo 虛擬環境已存在
)

REM 激活虛擬環境
call .venv\Scripts\activate.bat

REM 升級 pip
echo 升級 pip...
python -m pip install --upgrade pip

REM 安裝依賴
echo 安裝依賴...
pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo 錯誤: 依賴安裝失敗
    pause
    exit /b 1
)

REM 驗證安裝
echo 驗證核心依賴...
python -c "import torch; import transformers; import nltk; print('核心依賴安裝成功!')"

REM 檢查 PyTorch 設備
echo PyTorch 設備信息:
python -c "import torch; print(f'  PyTorch 版本: {torch.__version__}'); print(f'  CUDA 可用: {torch.cuda.is_available()}')"

REM 創建必要目錄
echo 創建輸出目錄...
if not exist "knowSelect\output\TAKE_tiage\model" mkdir "knowSelect\output\TAKE_tiage\model"
if not exist "knowSelect\output\TAKE_tiage\ks_pred" mkdir "knowSelect\output\TAKE_tiage\ks_pred"
if not exist "knowSelect\output\TAKE_tiage\logs" mkdir "knowSelect\output\TAKE_tiage\logs"

REM 初始化 checkpoints.json
if not exist "knowSelect\output\TAKE_tiage\model\checkpoints.json" (
    echo {"time": []} > "knowSelect\output\TAKE_tiage\model\checkpoints.json"
)

echo.
echo ============================================================
echo [%date% %time%] 環境設置完成!
echo ============================================================
echo.
echo 使用方法:
echo   1. 激活環境: .venv\Scripts\activate.bat
echo   2. 訓練模型: scripts\train_take_tiage.bat
echo   3. 推論評估: scripts\infer_take_tiage.bat
echo   4. 完整流程: scripts\pipeline_take_tiage.bat
echo   5. 消融實驗: scripts\ablation_take_tiage.bat
pause
