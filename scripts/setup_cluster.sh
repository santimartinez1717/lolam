#!/bin/bash
# Prepara el entorno en el cluster (una vez). Ejecutar en el nodo con GPU (p. ej. desde
# la sesión de VS Code del job), desde la raíz del repo:
#   bash scripts/setup_cluster.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(uname -s)" != "Linux" ] || ! command -v nvidia-smi >/dev/null; then
  echo "Este script es para el nodo con GPU del cluster (Linux + NVIDIA). Aquí no hay GPU." >&2
  echo "En local basta con: uv sync && uv run pytest" >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync --extra gpu
source .venv/bin/activate

python - <<'EOF'
import torch, vllm, transformers
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0))
print("vllm", vllm.__version__, "| transformers", transformers.__version__)
free, total = torch.cuda.mem_get_info()
print(f"VRAM libre {free / 2**30:.1f} / {total / 2**30:.1f} GB")
EOF

# Descarga previa de modelos y datos (evita perder tiempo de GPU dentro de los jobs).
python - <<'EOF'
from huggingface_hub import snapshot_download
from datasets import load_dataset
for repo in ["Qwen/Qwen3.5-0.8B", "RikiyaT/Qwen3.5-0.8B-weight-space-programming"]:
    print("descargado:", snapshot_download(repo))
load_dataset("HuggingFaceH4/MATH-500", split="test")
load_dataset("openai/gsm8k", "main", split="test")
print("datasets listos")
EOF
mkdir -p logs runs reports
echo "Listo. Siguiente: bash scripts/phase0.sh smoke"
