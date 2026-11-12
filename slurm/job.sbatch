#!/bin/bash
# Job genérico: ejecuta el comando que se le pase dentro del entorno de lolam.
#
#   sbatch slurm/job.sbatch lolam-throughput
#   sbatch slurm/job.sbatch lolam-eval --model Qwen/Qwen3.5-0.8B --task math500 ...
#
# Todo lo que se lanza aquí debe ser reanudable: si el job llega a 8 h, se mata,
# y el siguiente de la cadena (slurm/chain.sh) continúa donde se quedó.
#
#SBATCH --job-name=lolam
#SBATCH --partition=dgx
#SBATCH --account=dgx_estudiante
#SBATCH --qos=dgx_estudiante
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
# Si el cluster pide un perfil MIG concreto, cambia la línea --gres
# (mira los disponibles con: sinfo -p dgx -o "%G").

source slurm/common.sh
echo "[lolam] $*"
exec "$@"
