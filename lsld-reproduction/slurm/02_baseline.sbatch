#!/usr/bin/env bash
#SBATCH --job-name=lsld-baseline
#SBATCH --nodelist=ariel-v8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1-0
#SBATCH --partition=batch_ugrad
#SBATCH --chdir=/nas2/data/janthonio03/robust_medqa/med_prm/lsld
#SBATCH --output=logs/%j.out
#SBATCH --export=NONE

set -euo pipefail

PROJECT_ROOT="/nas2/data/janthonio03/robust_medqa/med_prm/lsld"
export HF_HOME="${PROJECT_ROOT}/hf_cache"

exec "${PROJECT_ROOT}/env/bin/lsld" baseline \
  --model meta-llama/Llama-3.1-8B \
  --lre-dir data/lre_dataset \
  --max-examples 1000 \
  --batch-size 4 \
  --output-dir outputs/baseline_all
