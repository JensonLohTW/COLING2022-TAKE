#!/bin/bash
# ==============================================================================
# 環境設置腳本
# 功能: 創建虛擬環境並安裝依賴
# ==============================================================================

set -e

# 專案根目錄
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 開始環境設置...${NC}"

# 檢查 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${YELLOW}Python 版本: $PYTHON_VERSION${NC}"

# 創建虛擬環境
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}創建虛擬環境...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}虛擬環境創建成功!${NC}"
else
    echo -e "${YELLOW}虛擬環境已存在${NC}"
fi

# 激活虛擬環境
source .venv/bin/activate

# 升級 pip
echo -e "${YELLOW}升級 pip...${NC}"
pip install --upgrade pip

# 安裝依賴
echo -e "${YELLOW}安裝依賴...${NC}"
pip install -r requirements.txt

# 驗證安裝
echo -e "${YELLOW}驗證核心依賴...${NC}"
python -c "import torch; import transformers; import nltk; print('核心依賴安裝成功!')"

# 檢查 PyTorch 設備
echo -e "${YELLOW}PyTorch 設備信息:${NC}"
python -c "import torch; print(f'  PyTorch 版本: {torch.__version__}'); print(f'  CUDA 可用: {torch.cuda.is_available()}')"

# 創建必要目錄
echo -e "${YELLOW}創建輸出目錄...${NC}"
mkdir -p knowSelect/output/TAKE_tiage/model
mkdir -p knowSelect/output/TAKE_tiage/ks_pred
mkdir -p knowSelect/output/TAKE_tiage/logs

# 初始化 checkpoints.json
if [ ! -f "knowSelect/output/TAKE_tiage/model/checkpoints.json" ]; then
    echo '{"time": []}' > knowSelect/output/TAKE_tiage/model/checkpoints.json
fi

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 環境設置完成!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "${YELLOW}使用方法:${NC}"
echo "  1. 激活環境: source .venv/bin/activate"
echo "  2. 訓練模型: ./scripts/train_take_tiage.sh"
echo "  3. 推論評估: ./scripts/infer_take_tiage.sh"
echo "  4. 完整流程: ./scripts/pipeline_take_tiage.sh"
echo "  5. 消融實驗: ./scripts/ablation_take_tiage.sh"
