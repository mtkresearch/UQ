#!/bin/bash
# Collect T1 (inference 11x) and T2 (semantic clustering) timing for each
# model x dataset combination, using 100 samples.
# Results saved to: $OUT/timing_collect_data.json
#
# Usage:
#   nohup bash slurm/run_collect_timing.sh \
#     > /proj/MR_dataset/mtk53728/UQ/sep_scratch/collect_timing.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE=offline
export WANDB_ENT="${WANDB_ENT:-offline}"

SCRATCH_BASE="/proj/MR_dataset/mtk53728/UQ/sep_scratch"
OUT_JSON="$SCRATCH_BASE/transfer_v2/timing_collect_data.json"
NUM_SAMPLES=100
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="$SCRATCH_BASE/collect_timing_${TIMESTAMP}/logs"
mkdir -p "$LOG_DIR"

DATASETS=(squad nq)

SMALL_MODELS=(
    configs/model/llama-2-7b.yaml
    configs/model/mistral-7b.yaml
    configs/model/llama-3.2-1b.yaml
    configs/model/llama-3.1-8b.yaml
    configs/model/qwen3-8b.yaml
)

LARGE_MODELS=(
    configs/model/phi-4.yaml
    configs/model/gemma-4-12b.yaml
    configs/model/mistral-nemo.yaml
)

# GPUs available
SMALL_GPUS=(0 1 2 3)
LARGE_GPU_PAIRS=("0,1" "2,3")

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
    local scratch_dir="$SCRATCH_BASE/collect_timing_${TIMESTAMP}/${dataset}/${name}"
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
    local scratch_dir="$SCRATCH_BASE/collect_timing_${TIMESTAMP}/${dataset}/${name}"
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
    for cfg in "${SMALL_MODELS[@]}"; do
        g=${SMALL_GPUS[$((g_idx % ${#SMALL_GPUS[@]}))]}
        wait_for_gpu "$g"
        launch_small "$cfg" "$dataset" "$g"
        (( g_idx++ )) || true
    done
    wait
    echo "Small models done."

    echo "--- Large models ---"
    pair_idx=0
    for cfg in "${LARGE_MODELS[@]}"; do
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
echo "All generation done. Collecting timing.json files..."

DEST="$SCRATCH_BASE/transfer_v2/data_generation_timing"
mkdir -p "$DEST"

for dataset in "${DATASETS[@]}"; do
    for model_dir in "$SCRATCH_BASE/collect_timing_${TIMESTAMP}/${dataset}"/*/; do
        model=$(basename "$model_dir")
        tf=$(find "$model_dir" -path "*/wandb/*/files/timing.json" | head -1)
        if [[ -n "$tf" ]]; then
            cp "$tf" "$DEST/${dataset}_${model}.json"
            echo "  Saved: ${dataset}_${model}.json"
        else
            echo "  WARNING: no timing.json found for ${dataset}/${model}"
        fi
    done
done

echo "T1/T2 timing saved to: $DEST"

echo ""
echo "Generating timing tables..."
python -m sep.transfer.make_timing_table \
    --collect-timing-base "$DEST" \
    --transfer-v2-timing  "$SCRATCH_BASE/transfer_v2/timing.json" \
    --pair-list           "$REPO_ROOT/slurm/inputs/pair_list.txt" \
    --out-json            "$SCRATCH_BASE/transfer_v2/timing_collect_data.json" \
    --out-csv-dir         "$SCRATCH_BASE/transfer_v2" \
    --datasets            "${DATASETS[@]}"

echo "Done."
