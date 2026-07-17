#!/bin/bash
# Sequentially regenerate all six Qwen3 runs {1.7B,8B} x {trivia_qa,nq,squad}
# with the fixed hidden-state indexing. Each job uses all 8 GPUs, so they run
# one after another. Writes into the repo-local sep_scratch to match the
# existing transfer-study layout.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATASETS=(trivia_qa nq squad)
MODELS=(Qwen3-1.7B Qwen3-8B)

for ds in "${DATASETS[@]}"; do
  for mdl in "${MODELS[@]}"; do
    echo "############################################################"
    echo "### REGEN $mdl $ds  ($(date))"
    echo "############################################################"
    SHARDS_PARENT="$REPO_ROOT/sep_scratch/shards/${mdl}_${ds}_$(date +%Y%m%d_%H%M%S)" \
    NUM_GPUS=8 MODEL_NAME="$mdl" DATASET="$ds" NUM_SAMPLES=2000 \
    NUM_GENERATIONS=10 RANDOM_SEED=20 \
      bash slurm/run_multigpu.sh
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "### FAILED $mdl $ds (rc=$rc). Continuing to next job." >&2
    else
      echo "### DONE $mdl $ds ($(date))"
    fi
  done
done
echo "ALL REGEN JOBS FINISHED ($(date))"
