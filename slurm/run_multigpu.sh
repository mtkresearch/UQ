#!/bin/bash
# Data-parallel generation across multiple GPUs, followed by a single merged
# compute pass. Each GPU runs one shard (a disjoint stride of the same seeded
# sample), writes to its own scratch dir, then merge_shards concatenates them
# and computes uncertainty measures once.
#
# The experiment (model, dataset, num_samples, num_generations, random_seed,
# experiment_lot, ...) is driven by an omegaconf YAML config. This launcher
# passes that config straight through and only adds the per-shard mechanics:
# --num_shards and --shard_index. The --no-* flags below are sharding
# *orchestration* (not experiment config): they keep each shard from
# duplicating training generations / running the per-shard compute stage, so
# they live here rather than in the config (a standalone `sep-generate --config
# X` should still do training-gens + uncertainties normally).
#
# Usage:
#   bash slurm/run_multigpu.sh --config configs/model/qwen3-8b.yaml --num_gpus 8
#   bash slurm/run_multigpu.sh --config configs/model/qwen3-8b.yaml --num_gpus 4 --dataset squad --num_samples 5000
#
#   --config <path>        omegaconf YAML path layered on base.yaml (repeatable). Required.
#   --num_gpus <N>         number of data-parallel shards / GPUs (default 8).
#   <any other args>       forwarded verbatim to `sep.generate_answers`,
#                          overriding the config (CLI > YAML). These generate
#                          overrides are NOT forwarded to the merge step, whose
#                          compute config comes from the YAML.
set -euo pipefail

# --- Parse launcher args ----------------------------------------------------
NUM_GPUS="${NUM_GPUS:-8}"   # env still honoured as the default; --num_gpus wins
CONFIG_ARGS=()              # --config X ... (consumed by omegaconf + argparse)
CONFIG_TAG=""               # sanitised config names, for the scratch dir tag
PASSTHROUGH_ARGS=()         # everything else -> generation-stage overrides

add_config() {
    CONFIG_ARGS+=(--config "$1")
    local base; base="$(basename "$1")"; base="${base%.yaml}"
    CONFIG_TAG="${CONFIG_TAG:+${CONFIG_TAG}_}${base}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)             add_config "$2"; shift 2 ;;
        --config=*)           add_config "${1#*=}"; shift ;;
        --num_gpus|--num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --num_gpus=*|--num-gpus=*) NUM_GPUS="${1#*=}"; shift ;;
        *)                    PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#CONFIG_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: pass --config <path>, e.g. --config configs/model/qwen3-8b.yaml" >&2
    echo "       (available: $(ls configs/model/*.yaml 2>/dev/null | paste -sd' '))" >&2
    exit 1
fi

# --- Resolve which physical GPUs to use -------------------------------------
# Honour an outer CUDA_VISIBLE_DEVICES (e.g. 4,5,6,7): shard i runs on the i-th
# entry. Fall back to 0..NUM_GPUS-1. Then UNSET it so the per-shard export is
# the only value CUDA sees (CUDA_VISIBLE_DEVICES does not compose).
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=($(seq 0 $((NUM_GPUS - 1))))
fi
unset CUDA_VISIBLE_DEVICES
if [[ ${#GPU_LIST[@]} -lt $NUM_GPUS ]]; then
    echo "ERROR: --num_gpus=$NUM_GPUS but only ${#GPU_LIST[@]} GPU(s) available: ${GPU_LIST[*]}" >&2
    exit 1
fi

# --- TLS / offline env (corporate proxy + no wandb account) ----------------
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export WANDB_MODE=offline
export WANDB_ENT="${WANDB_ENT:-offline}"
export WANDB_SEM_UNC_ENTITY="${WANDB_SEM_UNC_ENTITY:-offline}"

# --- Paths ------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# Resolve Python interpreter and PYTHONPATH so background subshells inherit them.
# Priority: PYTHON env var > .venv in repo > sep_intra_env fallback.
# Use PYTHON=/path/to/python bash slurm/run_multigpu.sh ... to override explicitly.
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -f "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    elif [[ -f /build_bak/UQ/sep_intra_env/bin/python ]]; then
        # Resolve symlink so the path is valid on any machine, not just the one
        # where /proj is mounted.
        PYTHON="$(readlink -f /build_bak/UQ/sep_intra_env/bin/python)"
    else
        PYTHON="$(which python3 || which python)"
    fi
fi
export PYTHON
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT/src}"
echo "Using Python: $PYTHON"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${CONFIG_TAG}_${TIMESTAMP}"
SHARDS_PARENT="${SHARDS_PARENT:-$REPO_ROOT/sep_scratch/${CONFIG_TAG}/${TIMESTAMP}/shards}"
mkdir -p "$SHARDS_PARENT"
LOG_DIR="$SHARDS_PARENT/logs"
mkdir -p "$LOG_DIR"

echo "Launching $NUM_GPUS shards for config '${CONFIG_ARGS[*]}' -> $SHARDS_PARENT"
if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    echo "  generate overrides: ${PASSTHROUGH_ARGS[*]}"
fi

# --- Launch one generation shard per GPU -----------------------------------
pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    shard_dir="$SHARDS_PARENT/shard_$(printf '%02d' "$i")"
    mkdir -p "$shard_dir"
    gpu="${GPU_LIST[$i]}"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        # Each shard isolates its wandb output under its own scratch dir so the
        # merge step can locate exactly one set of pkls per shard.
        export SCRATCH_DIR="$shard_dir"
        "$PYTHON" -m sep.generate_answers \
            "${CONFIG_ARGS[@]}" \
            "${PASSTHROUGH_ARGS[@]}" \
            --num_shards="$NUM_GPUS" \
            --shard_index="$i" \
            --no-get_training_set_generations \
            --no-compute_p_ik \
            --no-compute_p_ik_answerable \
            --no-compute_uncertainties
    ) >"$LOG_DIR/shard_$(printf '%02d' "$i").log" 2>&1 &
    pids+=($!)
    echo "  shard $i -> GPU $gpu (pid ${pids[-1]}, log $LOG_DIR/shard_$(printf '%02d' "$i").log)"
done

# --- Wait for all shards ----------------------------------------------------
fail=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        echo "ERROR: shard $idx (pid ${pids[$idx]}) failed. See $LOG_DIR/shard_$(printf '%02d' "$idx").log" >&2
        fail=1
    fi
done
if [[ $fail -ne 0 ]]; then
    echo "One or more shards failed; not merging." >&2
    exit 1
fi
echo "All shards finished. Merging + computing uncertainty measures."

# --- Merge + single compute pass (first GPU for the DeBERTa entailment model) ---
export CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}"
export SCRATCH_DIR="$SHARDS_PARENT/merged"
mkdir -p "$SCRATCH_DIR"
"$PYTHON" -m sep.merge_shards \
    "${CONFIG_ARGS[@]}" \
    --shards_parent="$SHARDS_PARENT" \
    2>&1 | tee "$LOG_DIR/merge.log"

echo "Done. Merged run + measures under $SCRATCH_DIR"
echo "Shards parent: $SHARDS_PARENT"
