#!/usr/bin/env bash
#SBATCH --job-name=lsld-heads-repeat
#SBATCH --array=1-4%1
#SBATCH --nodelist=ariel-v8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1-0
#SBATCH --partition=batch_ugrad
#SBATCH --chdir=/nas2/data/janthonio03/robust_medqa/med_prm/lsld
#SBATCH --output=logs/%A_%a.out
#SBATCH --export=NONE

set -euo pipefail

PROJECT_ROOT="/nas2/data/janthonio03/robust_medqa/med_prm/lsld"
export HF_HOME="${PROJECT_ROOT}/hf_cache"

exec "${PROJECT_ROOT}/env/bin/lsld" head-search \
  --model meta-llama/Llama-3.1-8B \
  --lre-dir data/lre_dataset \
  --relation country_capital_city \
  --head-data-protocol official-code \
  --max-pairs-per-split 100 \
  --epochs 500 \
  --batch-size 16 \
  --learning-rate 1.0 \
  --sparsity-lambda 1.0 \
  --temperature 1.0 \
  --head-count-weight 0.1 \
  --checkpoint-selection released-code-last-batch \
  --seed "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "outputs/head_search_capital_v040_seed${SLURM_ARRAY_TASK_ID}"
