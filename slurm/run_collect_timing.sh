#!/bin/bash
# Collect T1 (answer sampling, 11 generations/question) and T2 (semantic clustering)
# timing for each model x dataset, on 100 questions.  This is step 1 of the timing
# pipeline; see src/sep/transfer/README_timing.md.
#
# Usage:
#   nohup bash slurm/run_collect_timing.sh \
#     > /proj/MR_dataset/mtk53728/UQ/sep_scratch/collect_timing.log 2>&1 &
#
#   # one or more datasets (default: all three)
#   bash slurm/run_collect_timing.sh trivia_qa
#   bash slurm/run_collect_timing.sh squad nq
#
#   # keep the ~1-2 GB/dataset generation dir instead of deleting it at the end
#   KEEP_RUN_DIR=1 bash slurm/run_collect_timing.sh
#
# Outputs, which is all step 2 needs:
#   $TIMING/data_generation_timing/<ds>_<model>.json   T1/T2 timestamps, all datasets
#   $TIMING/collect_timing_logs_<timestamp>/           per-model generation logs
# where $TIMING = <scratch>/transfer_v2/timing
#
# The generation run itself (answers, hidden states, wandb dirs) is deleted once every
# expected timing.json has been harvested -- it is ~1.9 GB for squad+nq and ~1 GB for
# trivia_qa, and nothing downstream reads it.
#
# This script does NOT build the timing tables: T4/T5 come from a separate controlled
# re-timing.  Run slurm/run_time_fits.sh afterwards for the CSVs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE=offline
export WANDB_ENT="${WANDB_ENT:-offline}"

SCRATCH_BASE="/proj/MR_dataset/mtk53728/UQ/sep_scratch"
TIMING="$SCRATCH_BASE/transfer_v2/timing"   # every timing artefact lives here
NUM_SAMPLES=100
KEEP_RUN_DIR="${KEEP_RUN_DIR:-0}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_BASE="$SCRATCH_BASE/collect_timing_${TIMESTAMP}"
LOG_DIR="$RUN_BASE/logs"
mkdir -p "$LOG_DIR"

DATASETS=("$@")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(squad nq trivia_qa)

# GPUs available
SMALL_GPUS=(0 1 2 3)
LARGE_GPU_PAIRS=("0,1" "2,3")

# Which models to time, per dataset.  llama-2-7b / mistral-7b have no trivia_qa
# generations (see slurm/inputs/model_paths.txt) and no pair uses them there.
models_for() {
    local dataset="$1" which="$2"
    if [[ "$which" == small ]]; then
        if [[ "$dataset" == trivia_qa ]]; then
            echo configs/model/llama-3.2-1b.yaml \
                 configs/model/llama-3.1-8b.yaml \
                 configs/model/qwen3-8b.yaml
        else
            echo configs/model/llama-2-7b.yaml \
                 configs/model/mistral-7b.yaml \
                 configs/model/llama-3.2-1b.yaml \
                 configs/model/llama-3.1-8b.yaml \
                 configs/model/qwen3-8b.yaml
        fi
    else
        echo configs/model/phi-4.yaml \
             configs/model/gemma-4-12b.yaml \
             configs/model/mistral-nemo.yaml
    fi
}

declare -A gpu_pid

wait_for_gpu() {
    local gpu="$1"
    if [[ -n "${gpu_pid[$gpu]+_}" ]]; then
        wait "${gpu_pid[$gpu]}" 2>/dev/null || true
        unset gpu_pid[$gpu]
    fi
}

launch_small() {
    local cfg="$1" dataset="$2" gpu="$3"
    local name
    name=$(basename "$cfg" .yaml)
    local scratch_dir="$RUN_BASE/${dataset}/${name}"
    local log="$LOG_DIR/${dataset}_${name}.log"
    echo "[$(date '+%H:%M:%S')] Launching $name ($dataset) on GPU $gpu"
    (
        export SCRATCH_DIR="$scratch_dir"
        CUDA_VISIBLE_DEVICES=$gpu python -m sep.generate_answers \
            --config "$cfg" \
            --dataset "$dataset" \
            --num_samples "$NUM_SAMPLES" \
            --compute_uncertainties
    ) >"$log" 2>&1 &
    gpu_pid[$gpu]=$!
}

launch_large() {
    local cfg="$1" dataset="$2" pair_idx="$3"
    local name
    name=$(basename "$cfg" .yaml)
    local gpus="${LARGE_GPU_PAIRS[$pair_idx]}"
    local scratch_dir="$RUN_BASE/${dataset}/${name}"
    local log="$LOG_DIR/${dataset}_${name}.log"
    echo "[$(date '+%H:%M:%S')] Launching $name ($dataset) on GPUs $gpus"
    (
        export SCRATCH_DIR="$scratch_dir"
        CUDA_VISIBLE_DEVICES=$gpus python -m sep.generate_answers \
            --config "$cfg" \
            --dataset "$dataset" \
            --num_samples "$NUM_SAMPLES" \
            --compute_uncertainties \
            --multi_gpu
    ) >"$log" 2>&1 &
    gpu_pid[$gpus]=$!
}

for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Dataset: $dataset | samples: $NUM_SAMPLES"
    echo "=========================================="

    echo "--- Small models ---"
    g_idx=0
    for cfg in $(models_for "$dataset" small); do
        g=${SMALL_GPUS[$((g_idx % ${#SMALL_GPUS[@]}))]}
        wait_for_gpu "$g"
        launch_small "$cfg" "$dataset" "$g"
        (( g_idx++ )) || true
    done
    wait
    echo "Small models done."

    echo "--- Large models ---"
    pair_idx=0
    for cfg in $(models_for "$dataset" large); do
        idx=$((pair_idx % ${#LARGE_GPU_PAIRS[@]}))
        gpus="${LARGE_GPU_PAIRS[$idx]}"
        if [[ -n "${gpu_pid[$gpus]+_}" ]]; then
            wait "${gpu_pid[$gpus]}" 2>/dev/null || true
        fi
        launch_large "$cfg" "$dataset" "$idx"
        (( pair_idx++ )) || true
    done
    wait
    echo "Large models done."
done

wait
echo ""
echo "All generation done. Harvesting timing.json files..."

# generate_answers writes its timings to <SCRATCH_DIR>/wandb/run-*/files/timing.json --
# a path that differs every run and carries no model/dataset in the name.  Copy each one
# to the flat <dataset>_<model>.json layout make_timing_table parses.
DEST="$TIMING/data_generation_timing"
mkdir -p "$DEST"
harvested=0
missing=0
for dataset in "${DATASETS[@]}"; do
    for model_dir in "$RUN_BASE/${dataset}"/*/; do
        [[ -d "$model_dir" ]] || continue
        model=$(basename "$model_dir")
        tf=$(find "$model_dir" -path "*/wandb/*/files/timing.json" | head -1)
        if [[ -n "$tf" ]]; then
            cp "$tf" "$DEST/${dataset}_${model}.json"
            echo "  saved $DEST/${dataset}_${model}.json"
            (( harvested++ )) || true
        else
            echo "  WARNING: no timing.json for ${dataset}/${model}"
            (( missing++ )) || true
        fi
    done
done
echo "Harvested $harvested timing file(s), $missing missing."

# Keep the logs, they are the only part of the run dir worth having afterwards.
LOG_KEEP="$TIMING/collect_timing_logs_${TIMESTAMP}"
mkdir -p "$LOG_KEEP"
cp "$LOG_DIR"/*.log "$LOG_KEEP"/ 2>/dev/null || true
echo "Logs kept at $LOG_KEEP"

# Delete the generation run: answers, hidden states and wandb dirs, ~1-2 GB per dataset.
# Nothing downstream reads them -- the timings are now in data_generation_timing*/.
# Refuse to delete if anything is missing, so a partial run can be debugged.
echo ""
RUN_SIZE=$(du -sh "$RUN_BASE" 2>/dev/null | cut -f1)
if [[ "$KEEP_RUN_DIR" == 1 ]]; then
    echo "KEEP_RUN_DIR=1, keeping $RUN_BASE ($RUN_SIZE)"
elif [[ "$missing" -gt 0 ]]; then
    echo "WARNING: $missing timing file(s) missing, keeping $RUN_BASE ($RUN_SIZE) for debugging."
    echo "         Delete it yourself once you are done: rm -rf $RUN_BASE"
elif [[ "$harvested" -eq 0 ]]; then
    echo "WARNING: nothing harvested, keeping $RUN_BASE ($RUN_SIZE)."
else
    echo "Deleting generation run dir $RUN_BASE ($RUN_SIZE)..."
    rm -rf "$RUN_BASE"
    echo "Deleted."
fi

echo ""
echo "Done.  T1/T2 for: ${DATASETS[*]}  ->  $DEST/<ds>_<model>.json"
echo ""
echo "Next: bash slurm/run_time_fits.sh   (T4/T5 + the timing_table_*.csv)"
