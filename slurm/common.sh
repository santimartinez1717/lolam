# Entorno común para todos los jobs. Se carga con `source slurm/common.sh`.
set -euo pipefail

# Raíz del repo a partir de la ubicación de este fichero (vale con sbatch y en la sesión
# interactiva de VS Code, donde SLURM_SUBMIT_DIR apunta a otro sitio).
export LOLAM_ROOT="${LOLAM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$LOLAM_ROOT"

# Cachés en el RAID grande (no en /tmp, que se borra al acabar el job).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"

# Solo 4 CPUs y 16 GB de RAM: no lanzar más hilos de los que hay.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false

# Entorno creado por scripts/setup_cluster.sh (no usamos `uv run`: re-sincronizaría
# sin el extra `gpu` y desinstalaría vLLM).
source .venv/bin/activate

echo "[lolam] job ${SLURM_JOB_ID:-local} en $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
