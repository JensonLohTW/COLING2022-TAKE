# TAKE + DGCN3 训练指南（Windows 版）

> 本文档专为 Windows 环境设计，说明如何在 Windows 10/11 上运行 TAKE 模型训练、查看日志以及常见问题排查。

---

## 目录

1. [环境准备](#一环境准备)
2. [数据准备](#二数据准备)
3. [训练命令](#三训练命令)
4. [日志系统](#四日志系统)
5. [推论与评估](#五推论与评估)
6. [常见问题](#六常见问题)

---

## 一、环境准备

### 1.1 Python 环境

本项目使用 Python 3.9，建议使用虚拟环境：

#### 方法 A：使用 uv（推荐）

```powershell
# 打开 PowerShell，导航到项目目录
cd C:\Users\20190827\Downloads\COLING2022-TAKE-main

# 创建虚拟环境
uv venv .venv

# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 安装依赖
uv pip install -r requirements.txt
```

#### 方法 B：使用 pip

```powershell
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

**注意**：如果遇到 PowerShell 执行策略错误，运行：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 1.2 依赖检查

```powershell
# 检查核心依赖
python -c "import torch; import transformers; import nltk; print('All OK')"

# 检查 PyTorch 版本和设备
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 1.3 目录结构

确认以下目录已存在：

```
COLING2022-TAKE-main\
├── .venv\                          # 虚拟环境
├── demo\
│   ├── DGCN3\
│   │   ├── Centrality\             # DGCN3 预测输出
│   │   │   └── alpha_1.5\
│   │   │       └── tiage_0~9.csv
│   │   └── datasets\raw_data\tiage\
│   └── tiage-1\
│       └── outputs_nodes\
│           └── tiage_anno_nodes_all.csv
├── knowSelect\
│   ├── datasets\tiage\             # TAKE 数据集
│   ├── output\TAKE_tiage\          # 训练输出
│   │   ├── model\
│   │   ├── ks_pred\
│   │   └── logs\
│   └── TAKE\                       # 模型代码
└── docs\
```

---

## 二、数据准备

### 2.1 检查现有数据

根据您的项目结构，以下数据已准备完成：

✅ **DGCN3 预测**：`demo\DGCN3\Centrality\alpha_1.5\tiage_0~9.csv`（已存在）
✅ **TAKE 数据集**：`knowSelect\datasets\tiage\`（已存在）
✅ **输出目录**：`knowSelect\output\TAKE_tiage\`（已存在）

### 2.2 如需重新生成数据

#### 生成 DGCN3 预测

```powershell
cd demo\DGCN3
python main.py --dataset_name tiage
cd ..\..
```

#### 生成 TAKE 数据集

```powershell
cd demo\tiage-1
python export_take_dataset.py --out ..\..\knowSelect\datasets\tiage
cd ..\..
```

### 2.3 创建输出目录（如不存在）

```powershell
# 创建目录
New-Item -ItemType Directory -Force -Path knowSelect\output\TAKE_tiage\model
New-Item -ItemType Directory -Force -Path knowSelect\output\TAKE_tiage\ks_pred
New-Item -ItemType Directory -Force -Path knowSelect\output\TAKE_tiage\logs

# 初始化 checkpoints.json
Set-Content -Path knowSelect\output\TAKE_tiage\model\checkpoints.json -Value '{"time": []}'
```

---

## 三、训练命令

### 3.1 使用统一入口（推荐）

项目提供了 `main.py` 统一入口，简化命令调用：

#### 训练 TAKE 模型

```powershell
# 确保在项目根目录
cd C:\Users\20190827\Downloads\COLING2022-TAKE-main

# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 训练模型
python main.py train-take `
    --dataset tiage `
    --name TAKE_tiage `
    --use-centrality `
    --centrality-alpha 1.5 `
    --centrality-feature-set all `
    --centrality-window 2 `
    --node-id-json knowSelect\datasets\tiage\node_id.json
```

**注意**：PowerShell 中使用反引号 `` ` `` 进行换行。

#### 一键完整流程（导出中心性 + 训练 + 推论）

```powershell
python main.py pipeline `
    --dataset tiage `
    --name TAKE_tiage `
    --use-centrality `
    --centrality-alpha 1.5 `
    --centrality-feature-set all `
    --centrality-window 2 `
    --node-id-json knowSelect\datasets\tiage\node_id.json `
    --dataset-name tiage `
    --alphas 1.5
```

### 3.2 直接调用训练脚本

```powershell
cd knowSelect

# 训练命令
python -u .\TAKE\Run.py `
    --name TAKE_tiage `
    --dataset tiage `
    --mode train `
    --use_centrality `
    --centrality_alpha 1.5 `
    --centrality_feature_set all `
    --centrality_window 2 `
    --node_id_json datasets\tiage\node_id.json
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--name` | 实验名称 | 必填 |
| `--dataset` | 数据集名称 | 必填 |
| `--mode` | `train` 或 `inference` | 必填 |
| `--use_centrality` | 启用中心性特征 | False |
| `--centrality_alpha` | SIR alpha 参数 | 1.5 |
| `--centrality_feature_set` | 特征集（`none`/`imp_pct`/`all`） | `all` |
| `--centrality_window` | 中心性窗口大小 | 2 |
| `--node_id_json` | 节点 ID 映射文件 | 必填（使用中心性时） |
| `--GPU` | GPU 设备 ID | 0 |
| `--epoches` | 训练轮数 | 10 |
| `--train_batch_size` | 批次大小 | 2 |

### 3.3 后台运行

#### 使用 Start-Job（PowerShell）

```powershell
# 后台运行训练
$job = Start-Job -ScriptBlock {
    cd C:\Users\20190827\Downloads\COLING2022-TAKE-main\knowSelect
    & C:\Users\20190827\Downloads\COLING2022-TAKE-main\.venv\Scripts\python.exe -u .\TAKE\Run.py `
        --name TAKE_tiage `
        --dataset tiage `
        --mode train `
        --use_centrality `
        --centrality_alpha 1.5
}

# 查看任务状态
Get-Job

# 查看任务输出
Receive-Job $job -Keep

# 等待任务完成
Wait-Job $job
```

#### 使用输出重定向

```powershell
cd knowSelect

# 将输出重定向到文件
python -u .\TAKE\Run.py `
    --name TAKE_tiage `
    --dataset tiage `
    --mode train `
    --use_centrality `
    --centrality_alpha 1.5 `
    > ..\train.log 2>&1
```

### 3.4 消融实验

```powershell
# 使用 main.py 运行消融实验
python main.py ablation `
    --dataset tiage `
    --centrality-alpha 1.5 `
    --centrality-window 2 `
    --node-id-json knowSelect\datasets\tiage\node_id.json
```

这将自动运行三个实验：
1. **纯文本基线**（不使用中心性）
2. **仅使用 imp_pct 特征**
3. **使用全部结构特征**

### 3.5 从检查点恢复

```powershell
python -u .\TAKE\Run.py `
    --name TAKE_tiage `
    --dataset tiage `
    --mode train `
    --use_centrality `
    --resume
```

---

## 四、日志系统

### 4.1 日志文件位置

训练日志自动保存到：

```
knowSelect\output\{name}\logs\train_{timestamp}.log
```

例如：`knowSelect\output\TAKE_tiage\logs\train_20260103_165408.log`

### 4.2 日志格式

```
[2026-01-03 16:54:08] === Training session started: TAKE_tiage ===
[2026-01-03 16:54:08] Log file: output/TAKE_tiage/logs/train_20260103_165408.log
[2026-01-03 16:54:08] Using CPU
[2026-01-03 16:54:08] ============================================================
[2026-01-03 16:54:08] Starting Epoch 0 | Total batches: 150 | Batch size: 2
[2026-01-03 16:54:08] ============================================================
[2026-01-03 16:54:47] [Epoch 0] Batch 1/150 (0.7%) | loss_ks: 0.0000 | loss_distill: 0.9490 | loss_ID: 0.4478 | ks_acc: 1.0000 | ID_acc: 0.5000 | elapsed: 38.8s | LR: 0.00e+00
```

### 4.3 实时查看日志

#### 方法 A：使用 PowerShell Get-Content

```powershell
# 实时跟踪日志（类似 tail -f）
Get-Content -Path knowSelect\output\TAKE_tiage\logs\train_*.log -Wait -Tail 20
```

#### 方法 B：使用文本编辑器

打开日志文件，使用支持自动刷新的编辑器（如 Notepad++、VS Code）。

#### 方法 C：查看最新日志

```powershell
# 列出最新的日志文件
Get-ChildItem knowSelect\output\TAKE_tiage\logs\train_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# 显示最新日志内容
Get-Content (Get-ChildItem knowSelect\output\TAKE_tiage\logs\train_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
```

#### 方法 D：使用批处理脚本

创建 `view_log.bat`：

```batch
@echo off
:loop
cls
type knowSelect\output\TAKE_tiage\logs\train_*.log
timeout /t 5 >nul
goto loop
```

运行：`.\view_log.bat`

### 4.4 日志指标说明

| 指标 | 说明 | 理想趋势 |
|------|------|----------|
| `loss_ks` | 知识选择损失 | 下降 |
| `loss_distill` | 蒸馏损失 | 下降 |
| `loss_ID` | 话题判别损失 | 下降 |
| `ks_acc` | 知识选择准确率 | 上升 |
| `ID_acc` | 话题判别准确率 | 上升 |
| `elapsed` | 已用时间（秒） | - |
| `LR` | 学习率 | 先升后降（warmup） |

### 4.5 调整日志频率

在 `knowSelect\TAKE\CumulativeTrainer.py` 中修改：

```python
# 约第 199 行
log_interval = 10  # 改为更小的值如 5 或 1
```

---

## 五、推论与评估

### 5.1 运行推论

#### 使用 main.py

```powershell
python main.py infer-take `
    --dataset tiage `
    --name TAKE_tiage `
    --use-centrality `
    --centrality-alpha 1.5 `
    --centrality-feature-set all `
    --centrality-window 2 `
    --node-id-json knowSelect\datasets\tiage\node_id.json
```

#### 直接调用

```powershell
cd knowSelect

python -u .\TAKE\Run.py `
    --name TAKE_tiage `
    --dataset tiage `
    --mode inference `
    --use_centrality `
    --centrality_alpha 1.5 `
    --centrality_feature_set all `
    --centrality_window 2 `
    --node_id_json datasets\tiage\node_id.json
```

### 5.2 推论输出

输出文件位于：`knowSelect\output\TAKE_tiage\ks_pred\`

```
ks_pred\
├── 0_test.json      # Epoch 0 的测试集预测
├── 1_test.json      # Epoch 1 的测试集预测
└── ...
```

### 5.3 评估指标

推论完成后，终端会输出：

- `final_ks_acc`：最终知识选择准确率
- `shifted_ks_acc`：话题转移时的准确率
- `inherited_ks_acc`：话题继承时的准确率
- `ID_acc`：话题转移判别准确率

---

## 六、常见问题

### 6.1 CUDA 不可用

**问题**：`AssertionError: Torch not compiled with CUDA enabled`

**解决**：代码已适配 CPU 模式，会自动回退。CPU 训练较慢，每 batch 约 40 秒。

如需使用 GPU，请安装 CUDA 版本的 PyTorch：

```powershell
# 卸载 CPU 版本
pip uninstall torch torchvision torchaudio

# 安装 CUDA 版本（以 CUDA 11.3 为例）
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### 6.2 PowerShell 执行策略错误

**问题**：`无法加载文件 Activate.ps1，因为在此系统上禁止运行脚本`

**解决**：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 6.3 路径包含空格或中文

**问题**：路径中的空格或中文字符导致错误

**解决**：使用引号包裹路径：

```powershell
python -u ".\TAKE\Run.py" --name TAKE_tiage --dataset tiage
```

### 6.4 目录不存在

**问题**：`FileNotFoundError: ... checkpoints.json`

**解决**：

```powershell
New-Item -ItemType Directory -Force -Path knowSelect\output\TAKE_tiage\model
New-Item -ItemType Directory -Force -Path knowSelect\output\TAKE_tiage\ks_pred
New-Item -ItemType Directory -Force -Path knowSelect\output\TAKE_tiage\logs
Set-Content -Path knowSelect\output\TAKE_tiage\model\checkpoints.json -Value '{"time": []}'
```

### 6.5 内存不足

**问题**：`RuntimeError: out of memory`

**解决**：减小批次大小

```powershell
--train_batch_size 1
```

### 6.6 日志不更新

**问题**：日志文件没有实时更新

**解决**：
1. 确保使用 `-u` 参数运行 Python（禁用缓冲）
2. 代码已使用 `FlushFileHandler` 即时刷新
3. 使用 `Get-Content -Wait` 实时查看

### 6.7 中心性特征加载失败

**问题**：找不到中心性预测文件

**解决**：确保 DGCN3 预测已生成

```powershell
Get-ChildItem demo\DGCN3\Centrality\alpha_1.5\
# 应该有 tiage_0.csv ~ tiage_9.csv
```

### 6.8 终止训练

#### 使用 Ctrl+C（前台运行时）

直接按 `Ctrl+C`

#### 使用任务管理器

1. 打开任务管理器（`Ctrl+Shift+Esc`）
2. 找到 `python.exe` 进程
3. 右键 → 结束任务

#### 使用 PowerShell 停止后台任务

```powershell
# 查看任务
Get-Job

# 停止任务
Stop-Job -Name <JobName>

# 或强制停止所有 Python 进程
Get-Process python | Stop-Process -Force
```

---

## 七、批处理脚本

### 7.1 训练脚本（train_take_tiage.bat）

创建 `train_take_tiage.bat`：

```batch
@echo off
setlocal

set PROJECT_ROOT=C:\Users\20190827\Downloads\COLING2022-TAKE-main
set VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe

echo [%date% %time%] Starting training...

cd %PROJECT_ROOT%

%VENV_PYTHON% main.py train-take ^
    --dataset tiage ^
    --name TAKE_tiage ^
    --use-centrality ^
    --centrality-alpha 1.5 ^
    --centrality-feature-set all ^
    --centrality-window 2 ^
    --node-id-json knowSelect\datasets\tiage\node_id.json

if %ERRORLEVEL% neq 0 (
    echo [%date% %time%] Training failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo [%date% %time%] Training completed successfully!
pause
```

### 7.2 推论脚本（infer_take_tiage.bat）

创建 `infer_take_tiage.bat`：

```batch
@echo off
setlocal

set PROJECT_ROOT=C:\Users\20190827\Downloads\COLING2022-TAKE-main
set VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe

echo [%date% %time%] Starting inference...

cd %PROJECT_ROOT%

%VENV_PYTHON% main.py infer-take ^
    --dataset tiage ^
    --name TAKE_tiage ^
    --use-centrality ^
    --centrality-alpha 1.5 ^
    --centrality-feature-set all ^
    --centrality-window 2 ^
    --node-id-json knowSelect\datasets\tiage\node_id.json

if %ERRORLEVEL% neq 0 (
    echo [%date% %time%] Inference failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo [%date% %time%] Inference completed successfully!
pause
```

### 7.3 完整流程脚本（pipeline_take_tiage.bat）

创建 `pipeline_take_tiage.bat`：

```batch
@echo off
setlocal

set PROJECT_ROOT=C:\Users\20190827\Downloads\COLING2022-TAKE-main
set VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe

echo [%date% %time%] Starting TAKE pipeline...

cd %PROJECT_ROOT%

%VENV_PYTHON% main.py pipeline ^
    --dataset tiage ^
    --name TAKE_tiage ^
    --use-centrality ^
    --centrality-alpha 1.5 ^
    --centrality-feature-set all ^
    --centrality-window 2 ^
    --node-id-json knowSelect\datasets\tiage\node_id.json ^
    --dataset-name tiage ^
    --alphas 1.5

if %ERRORLEVEL% neq 0 (
    echo [%date% %time%] Pipeline failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo [%date% %time%] Pipeline completed successfully!
pause
```

---

## 八、PowerShell 脚本

### 8.1 训练脚本（train_take_tiage.ps1）

创建 `train_take_tiage.ps1`：

```powershell
$ErrorActionPreference = "Stop"

$PROJECT_ROOT = "C:\Users\20190827\Downloads\COLING2022-TAKE-main"
$VENV_PYTHON = "$PROJECT_ROOT\.venv\Scripts\python.exe"

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting training..."

Set-Location $PROJECT_ROOT

& $VENV_PYTHON main.py train-take `
    --dataset tiage `
    --name TAKE_tiage `
    --use-centrality `
    --centrality-alpha 1.5 `
    --centrality-feature-set all `
    --centrality-window 2 `
    --node-id-json knowSelect\datasets\tiage\node_id.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Training failed with error code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Training completed successfully!" -ForegroundColor Green
```

运行：`.\train_take_tiage.ps1`

---

## 快速参考

### 快速启动命令

```powershell
# 1. 激活虚拟环境
cd C:\Users\20190827\Downloads\COLING2022-TAKE-main
.\.venv\Scripts\Activate.ps1

# 2. 训练
python main.py train-take --dataset tiage --name TAKE_tiage --use-centrality

# 3. 查看日志
Get-Content -Path knowSelect\output\TAKE_tiage\logs\train_*.log -Wait -Tail 20

# 4. 推论
python main.py infer-take --dataset tiage --name TAKE_tiage --use-centrality

# 5. 一键完整流程
python main.py pipeline --dataset tiage --name TAKE_tiage --use-centrality --dataset-name tiage --alphas 1.5
```

### 常用 PowerShell 命令

```powershell
# 查看 Python 进程
Get-Process python

# 停止所有 Python 进程
Get-Process python | Stop-Process -Force

# 查看日志文件列表
Get-ChildItem knowSelect\output\TAKE_tiage\logs\

# 查看最新日志
Get-Content (Get-ChildItem knowSelect\output\TAKE_tiage\logs\train_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName -Tail 50

# 监控 CPU/内存使用
Get-Process python | Format-Table Name, CPU, WorkingSet -AutoSize
```

---

## 附录：Windows 与 Linux 命令对照表

| Linux (Bash) | Windows (PowerShell) | 说明 |
|--------------|----------------------|------|
| `cd /path` | `cd C:\path` | 切换目录 |
| `ls` | `Get-ChildItem` / `ls` | 列出文件 |
| `cat file.txt` | `Get-Content file.txt` / `type file.txt` | 查看文件 |
| `tail -f file.log` | `Get-Content file.log -Wait -Tail 20` | 实时查看日志 |
| `mkdir -p dir` | `New-Item -ItemType Directory -Force dir` | 创建目录 |
| `rm -rf dir` | `Remove-Item -Recurse -Force dir` | 删除目录 |
| `ps aux \| grep python` | `Get-Process python` | 查看进程 |
| `kill -9 <pid>` | `Stop-Process -Id <pid> -Force` | 终止进程 |
| `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` | 激活虚拟环境 |
| `python -u script.py` | `python -u script.py` | 运行 Python（相同） |
| `\` （换行） | `` ` `` （换行） | 命令换行符 |

---

## 总结

本文档提供了在 Windows 环境下运行 TAKE 模型的完整指南。关键要点：

1. **统一入口**：使用 `main.py` 简化命令调用
2. **路径格式**：使用反斜杠 `\` 而非正斜杠 `/`
3. **换行符**：PowerShell 使用反引号 `` ` ``
4. **虚拟环境**：激活命令为 `.\.venv\Scripts\Activate.ps1`
5. **日志查看**：使用 `Get-Content -Wait` 实时跟踪

如有其他问题，请参考常见问题章节或联系项目维护者。


