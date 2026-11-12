#!/bin/bash
# Encadena N jobs iguales: cada uno empieza cuando acaba el anterior (bien o por tiempo).
# Como la evaluación es reanudable, cada job continúa el trabajo del anterior.
# Si el trabajo termina antes, los jobs restantes acaban en segundos sin hacer nada.
#
#   bash slurm/chain.sh 3 lolam-eval --model ... --out runs/base-think/math500
set -euo pipefail
n="$1"; shift
mkdir -p logs
prev=""
for i in $(seq 1 "$n"); do
  dep=${prev:+--dependency=afterany:$prev}
  prev=$(sbatch --parsable $dep slurm/job.sbatch "$@")
  echo "job $i/$n: $prev"
done
