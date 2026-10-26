#!/usr/bin/env bash
set -e

QUANT=${1:-"int8"}
COMFY_DIR="/content/ComfyUI"

echo "=== [1/5] 檢查系統記憶體 (System RAM) 與 GPU ==="
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo "偵測到系統記憶體: ${TOTAL_RAM_GB} GB"
if [ "$TOTAL_RAM_GB" -lt 30 ]; then
    echo "❌ 錯誤: 系統記憶體不足 (${TOTAL_RAM_GB} GB < 30 GB)！"
    echo "請在 Colab [執行階段] -> [變更執行階段類型] 切換至 L4 GPU 或開啟「高記憶體 (High-RAM)」！"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [2/5] 安裝系統工具與 ComfyUI ==="
apt-get update -qq && apt-get install -y -qq aria2 ffmpeg
if [ ! -d "$COMFY_DIR" ]; then
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
fi
cd "$COMFY_DIR"
pip install -q -r requirements.txt
pip install -q huggingface_hub[hf_transfer]

echo "=== [3/5] 安裝 ComfyUI-Manager 與 ComfyUI-Drama3 ==="
mkdir -p "$COMFY_DIR/custom_nodes"
cd "$COMFY_DIR/custom_nodes"
if [ ! -d "ComfyUI-Manager" ]; then
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git
fi
if [ ! -d "ComfyUI-Drama3" ]; then
    git clone --depth 1 https://github.com/PedroMarinhoDev/ComfyUI-Drama3.git
fi
if [ -f "ComfyUI-Drama3/requirements.txt" ]; then
    pip install -q -r ComfyUI-Drama3/requirements.txt
fi

echo "=== [4/5] 建立模型目錄並高速下載 Drama 3 權重 ==="
mkdir -p "$COMFY_DIR/models/diffusion_models"
mkdir -p "$COMFY_DIR/models/vae"
mkdir -p "$COMFY_DIR/models/asr"

export HF_HUB_ENABLE_HF_TRANSFER=1
MODEL_FILE="drama3_${QUANT}experts_bf16.safetensors"

echo "正在下載 Codec: drama3_codec.safetensors (~1.57 GB)..."
hf download PedroMarinhoDev/Drama-3-ComfyUI drama3_codec.safetensors \
    --local-dir "$COMFY_DIR/models/vae"

echo "正在下載 ASR: qwen3_asr_1.7b_bf16.safetensors (~4.1 GB)..."
hf download PedroMarinhoDev/Qwen3-ASR-ComfyUI qwen3_asr_1.7b_bf16.safetensors \
    --local-dir "$COMFY_DIR/models/asr"

echo "正在下載主模型: ${MODEL_FILE}..."
hf download PedroMarinhoDev/Drama-3-ComfyUI "$MODEL_FILE" \
    --local-dir "$COMFY_DIR/models/diffusion_models"

echo "=== [5/5] 安裝完成！ ==="
