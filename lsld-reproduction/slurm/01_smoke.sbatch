#!/usr/bin/env bash
#SBATCH --job-name=lsld-smoke
#SBATCH --nodelist=ariel-v8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=02:00:00
#SBATCH --partition=batch_ugrad
#SBATCH --chdir=/nas2/data/janthonio03/robust_medqa/med_prm/lsld
#SBATCH --output=logs/%j.out
#SBATCH --export=NONE

set -euo pipefail

PROJECT_ROOT="/nas2/data/janthonio03/robust_medqa/med_prm/lsld"
export HF_HOME="${PROJECT_ROOT}/hf_cache"

exec "${PROJECT_ROOT}/env/bin/lsld" smoke \
  --model meta-llama/Llama-3.1-8B \
  --output-dir outputs/smoke
