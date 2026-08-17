#!/bin/bash
# Alpha ablation sweep for ridge alignment.
# Phase 1 (probe_cache) must already be complete; it is alpha-independent.
#
# Per alpha:  Phase 2 (align_cache) -> Phase 3 (evaluate) -> Phase 4 (summary)
#   Phase 4 emits figures for that alpha only, tagged with it in the filename.
# Once, at the end: sep.transfer.sweep_summary
#   compares all alphas on disk -> best-alpha tables + best-alpha figure.
#
# Usage:
#   nohup bash slurm/run_transfer_alpha_sweep.sh \
#     > /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2/run_alpha_sweep.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXPECTED_BRANCH="exp/intra-family-transfer"
CURRENT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]; then
    echo "ERROR: expected branch '$EXPECTED_BRANCH' but on '$CURRENT_BRANCH'. Refusing to run (checking out mid-sweep deletes untracked-on-that-branch files)." >&2
    exit 1
fi

export WANDB_MODE=offline

check_branch() {
    local branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
    if [ "$branch" != "$EXPECTED_BRANCH" ]; then
        echo "ERROR: branch changed to '$branch' mid-sweep (expected '$EXPECTED_BRANCH'). Aborting." >&2
        exit 1
    fi
}

OUT=/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2
PAIR_LIST=$REPO_ROOT/slurm/inputs/pair_list.txt
MODEL_PATHS=$REPO_ROOT/slurm/inputs/model_paths.txt
BASE_ARGS="--out-dir $OUT --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 --seed 0"
PAIR_ARGS="--datasets squad nq --cross-align-dataset squad:nq nq:squad"

echo "=========================================="
echo "Phase 2: align_cache (all alphas)  $(date '+%H:%M:%S')"
echo "=========================================="
for ALPHA in 1e4 1e5 1e3 1e2 1e1; do
    check_branch
    echo "--- align_cache  alpha=$ALPHA  $(date '+%H:%M:%S') ---"
    python -m sep.transfer.transfer2 align_cache \
        $BASE_ARGS --alpha $ALPHA \
        $PAIR_ARGS
done

for ALPHA in 1e4 1e5 1e3 1e2 1e1; do
    check_branch
    COMMON="$BASE_ARGS --alpha $ALPHA"

    echo "=========================================="
    echo "Phase 3: evaluate  alpha=$ALPHA  $(date '+%H:%M:%S')"
    echo "=========================================="
    python -m sep.transfer.transfer2 evaluate \
        $COMMON \
        $PAIR_ARGS

    echo "=========================================="
    echo "Phase 4: summary (this alpha only)  alpha=$ALPHA  $(date '+%H:%M:%S')"
    echo "=========================================="
    python -m sep.transfer.transfer2 summary \
        $COMMON $PAIR_ARGS

    echo "--- Done alpha=$ALPHA  $(date '+%H:%M:%S') ---"
    echo ""
done

check_branch
echo "=========================================="
echo "Sweep summary: cross-alpha comparison  $(date '+%H:%M:%S')"
echo "=========================================="
python -m sep.transfer.sweep_summary \
    --out-dir "$OUT" \
    $PAIR_ARGS

echo ""
echo "All alphas complete.  $(date '+%H:%M:%S')"
echo "Results:            $OUT/results/"
echo "Per-alpha figures:  $OUT/summary_plots/"
echo "Cross-alpha sweep:  $OUT/sweep_summary/"
