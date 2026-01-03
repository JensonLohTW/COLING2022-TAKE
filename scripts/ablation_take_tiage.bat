@echo off
REM ==============================================================================
REM TAKE 消融實驗腳本 (tiage 數據集) - Windows 版
REM 數據集: tiage (對話網絡數據集)
REM 功能: 運行三組消融實驗對比不同特徵配置的效果
REM   - A1: 純文本基線 (不使用中心性)
REM   - A2: 僅使用 imp_pct 特徵
REM   - A3: 使用全部 6 維結構特徵
REM
REM 輸出結果:
REM   - 話題轉移預測結果 (0=不轉移, 1=轉移)
REM   - Top-K 高中心性句子及內容
REM   - Precision、Recall、F1 評價指標
REM   - 消融實驗對比報告
REM ==============================================================================

setlocal EnableDelayedExpansion

REM 專案根目錄
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

REM 虛擬環境 Python 路徑
set VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe

REM 參數配置
set DATASET=tiage
set CENTRALITY_ALPHA=1.5
set CENTRALITY_WINDOW=2
set NODE_ID_JSON=knowSelect\datasets\tiage\node_id.json

echo ============================================================
echo           TAKE 消融實驗 + 話題轉移檢測評估
echo ============================================================
echo 數據集: %DATASET%
echo 評價指標: Precision / Recall / F1
echo.

REM 檢查虛擬環境
if not exist "%VENV_PYTHON%" (
    echo 錯誤: 找不到虛擬環境 Python: %VENV_PYTHON%
    pause
    exit /b 1
)

echo [%date% %time%] 開始消融實驗...
echo.

REM ==============================================================================
REM 實驗 A1: 純文本基線
REM ==============================================================================
echo ------------------------------------------------------------
echo 實驗 A1: 純文本基線 (TAKE_tiage_text_only)
echo ------------------------------------------------------------

set NAME=TAKE_tiage_text_only
if not exist "knowSelect\output\%NAME%\model" mkdir "knowSelect\output\%NAME%\model"
if not exist "knowSelect\output\%NAME%\ks_pred" mkdir "knowSelect\output\%NAME%\ks_pred"
if not exist "knowSelect\output\%NAME%\logs" mkdir "knowSelect\output\%NAME%\logs"
if not exist "knowSelect\output\%NAME%\metrics" mkdir "knowSelect\output\%NAME%\metrics"
if not exist "knowSelect\output\%NAME%\model\checkpoints.json" (
    echo {"time": []} > "knowSelect\output\%NAME%\model\checkpoints.json"
)

echo [訓練] %NAME%
"%VENV_PYTHON%" main.py train-take --dataset %DATASET% --name %NAME%
if %ERRORLEVEL% neq 0 echo 訓練失敗: %NAME% & goto :exp2

echo [推論] %NAME%
"%VENV_PYTHON%" main.py infer-take --dataset %DATASET% --name %NAME%
if %ERRORLEVEL% neq 0 echo 推論失敗: %NAME%

:exp2
REM ==============================================================================
REM 實驗 A2: 文本 + imp_pct
REM ==============================================================================
echo ------------------------------------------------------------
echo 實驗 A2: 文本 + imp_pct (TAKE_tiage_imp_pct)
echo ------------------------------------------------------------

set NAME=TAKE_tiage_imp_pct
if not exist "knowSelect\output\%NAME%\model" mkdir "knowSelect\output\%NAME%\model"
if not exist "knowSelect\output\%NAME%\ks_pred" mkdir "knowSelect\output\%NAME%\ks_pred"
if not exist "knowSelect\output\%NAME%\logs" mkdir "knowSelect\output\%NAME%\logs"
if not exist "knowSelect\output\%NAME%\metrics" mkdir "knowSelect\output\%NAME%\metrics"
if not exist "knowSelect\output\%NAME%\model\checkpoints.json" (
    echo {"time": []} > "knowSelect\output\%NAME%\model\checkpoints.json"
)

echo [訓練] %NAME%
"%VENV_PYTHON%" main.py train-take --dataset %DATASET% --name %NAME% ^
    --use-centrality --centrality-alpha %CENTRALITY_ALPHA% ^
    --centrality-feature-set imp_pct --centrality-window %CENTRALITY_WINDOW% ^
    --node-id-json %NODE_ID_JSON%
if %ERRORLEVEL% neq 0 echo 訓練失敗: %NAME% & goto :exp3

echo [推論] %NAME%
"%VENV_PYTHON%" main.py infer-take --dataset %DATASET% --name %NAME% ^
    --use-centrality --centrality-alpha %CENTRALITY_ALPHA% ^
    --centrality-feature-set imp_pct --centrality-window %CENTRALITY_WINDOW% ^
    --node-id-json %NODE_ID_JSON%
if %ERRORLEVEL% neq 0 echo 推論失敗: %NAME%

:exp3
REM ==============================================================================
REM 實驗 A3: 文本 + 6維結構特徵
REM ==============================================================================
echo ------------------------------------------------------------
echo 實驗 A3: 文本 + 6維結構特徵 (TAKE_tiage_all_feats)
echo ------------------------------------------------------------

set NAME=TAKE_tiage_all_feats
if not exist "knowSelect\output\%NAME%\model" mkdir "knowSelect\output\%NAME%\model"
if not exist "knowSelect\output\%NAME%\ks_pred" mkdir "knowSelect\output\%NAME%\ks_pred"
if not exist "knowSelect\output\%NAME%\logs" mkdir "knowSelect\output\%NAME%\logs"
if not exist "knowSelect\output\%NAME%\metrics" mkdir "knowSelect\output\%NAME%\metrics"
if not exist "knowSelect\output\%NAME%\model\checkpoints.json" (
    echo {"time": []} > "knowSelect\output\%NAME%\model\checkpoints.json"
)

echo [訓練] %NAME%
"%VENV_PYTHON%" main.py train-take --dataset %DATASET% --name %NAME% ^
    --use-centrality --centrality-alpha %CENTRALITY_ALPHA% ^
    --centrality-feature-set all --centrality-window %CENTRALITY_WINDOW% ^
    --node-id-json %NODE_ID_JSON%
if %ERRORLEVEL% neq 0 echo 訓練失敗: %NAME% & goto :summary

echo [推論] %NAME%
"%VENV_PYTHON%" main.py infer-take --dataset %DATASET% --name %NAME% ^
    --use-centrality --centrality-alpha %CENTRALITY_ALPHA% ^
    --centrality-feature-set all --centrality-window %CENTRALITY_WINDOW% ^
    --node-id-json %NODE_ID_JSON%
if %ERRORLEVEL% neq 0 echo 推論失敗: %NAME%

:summary
REM ==============================================================================
REM 生成彙總報告
REM ==============================================================================
echo ------------------------------------------------------------
echo 生成消融實驗彙總報告
echo ------------------------------------------------------------

cd knowSelect
"%VENV_PYTHON%" TAKE\summarize_ablation.py --output_dir output\
cd ..

echo.
echo ============================================================
echo [%date% %time%] 消融實驗完成!
echo ============================================================
echo.
echo 結果文件位置:
echo   Precision/Recall/F1: knowSelect\output\*\metrics\shift_metrics.json
echo   Top-K 高中心性句子: knowSelect\output\*\metrics\shift_top3.jsonl
echo   消融實驗 CSV:       knowSelect\output\*\metrics\ablation_results.csv
echo   彙總報告:           knowSelect\output\ablation_summary.md
echo.
echo 評價指標說明:
echo   Precision (精確率): 預測為話題轉移的樣本中，真正是話題轉移的比例
echo   Recall (召回率):    所有真實話題轉移樣本中，被正確預測的比例
echo   F1:                 Precision 和 Recall 的調和平均數
pause
