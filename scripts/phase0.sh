#!/bin/bash
# Fase 0: lanza en orden todo lo necesario en el cluster.
#   bash scripts/phase0.sh smoke      # prueba rápida (~10 min): 5 problemas, 1 muestra, 4k tokens
#   bash scripts/phase0.sh throughput # rendimiento real de la porción de GPU
#   bash scripts/phase0.sh diff       # comprobar la edición WSP (CPU, sin GPU)
#   bash scripts/phase0.sh baselines  # líneas base: thinking / sin thinking / WSP
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=Qwen/Qwen3.5-0.8B
WSP=RikiyaT/Qwen3.5-0.8B-weight-space-programming
LIMIT_MATH=200
LIMIT_GSM=300
N=4

case "${1:-}" in
  smoke)
    sbatch slurm/job.sbatch lolam-eval --model "$BASE" --task math500 \
      --limit 5 --n 1 --max-tokens 4096 --max-model-len 6144 --out runs/smoke/math500
    ;;
  throughput)
    sbatch slurm/job.sbatch lolam-throughput --model "$BASE" --out reports/throughput.json
    ;;
  diff)
    # Solo CPU y ~4 GB de RAM: se puede ejecutar directamente, sin sbatch.
    .venv/bin/lolam-wsp-diff "$BASE" "$WSP" --out reports/wsp_diff
    ;;
  baselines)
    for task in math500 gsm8k; do
      limit=$([ "$task" = math500 ] && echo $LIMIT_MATH || echo $LIMIT_GSM)
      common=(--task "$task" --limit "$limit" --n "$N")
      bash slurm/chain.sh 3 lolam-eval --model "$BASE" "${common[@]}" --out "runs/base-think/$task"
      bash slurm/chain.sh 1 lolam-eval --model "$BASE" "${common[@]}" --no-thinking --out "runs/base-nothink/$task"
      bash slurm/chain.sh 2 lolam-eval --model "$WSP" "${common[@]}" --out "runs/wsp-think/$task"
    done
    ;;
  *)
    sed -n '2,7p' "$0"; exit 1 ;;
esac
