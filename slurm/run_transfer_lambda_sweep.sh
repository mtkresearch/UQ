#!/bin/bash
# Lambda ablation sweep for the E2-R* aligner. Self-contained: runs phases 1-3.
#
# NQ only: source probes are fit on nq, alignment is fit on nq (no cross-align).
#
# Writes to the leak-free dir transfer_v2_pooled/ (alignment fit on `pool`, disjoint
# from the eval rows). The old leaky results stay under transfer_v2/. Override OUT=...
#
# Phase 1 (probe_cache) is lambda-independent and runs once up front. The bootstrap step
# copies the existing probe caches in, so this call is a no-op that prints "skip" per
# model -- see slurm/bootstrap_pooled_outdir.sh for why they must be copied, not refit.
# Set SKIP_PROBE_CACHE=1 to drop the (already no-op) Phase 1 call entirely.
#
# Then, per lambda:  Phase 2 (align_cache, --aligners e2_rstar) -> Phase 3 (evaluate)
#   Each lambda is carried end-to-end before the next starts, so partial results
#   are usable as soon as that lambda finishes.
#   Alignment caches are tagged e2_rstar_l<lambda> (e.g. e2_rstar_l1e3), so the
#   lambdas do not overwrite each other and evaluate picks up the matching one.
#
# Once, at the end: sep.transfer.sweep_summary --hparam lambda --datasets nq
#   compares all lambdas on disk -> best-lambda tables + per-lambda curve figures,
#   under $OUT/sweep_summary/ (hparam_lambda_*, lambda_curves_nq_align_nq.*).
#
# Phase 4 (transfer2 summary) is NOT run: it tags every output by the ridge alpha
# (transfer2.py:1193) and would mislabel an e2_rstar-only run.
#
# Usage:
#   nohup bash slurm/run_transfer_lambda_sweep.sh \
#     > /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled/run_lambda_sweep.log 2>&1 &
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

OUT=${OUT:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled}
BASE_ARGS="--out-dir $OUT --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 --seed 0"
PAIR_ARGS="--datasets nq"

OUT_NEW="$OUT" bash slurm/bootstrap_pooled_outdir.sh

if [ "${SKIP_PROBE_CACHE:-0}" != "1" ]; then
    echo "=========================================="
    echo "Phase 1: probe_cache (lambda-independent, nq; no-op if cached)  $(date '+%H:%M:%S')"
    echo "=========================================="
    check_branch
    python -m sep.transfer.transfer2 probe_cache \
        $BASE_ARGS \
        $PAIR_ARGS
fi

for LAM in 1e3 1e4 1e5 1e6 1e7; do
    check_branch
    COMMON="$BASE_ARGS --lambda-val $LAM"

    echo "=========================================="
    echo "Phase 2: align_cache  e2_rstar lambda=$LAM  $(date '+%H:%M:%S')"
    echo "=========================================="
    python -m sep.transfer.transfer2 align_cache \
        $COMMON \
        $PAIR_ARGS \
        --aligners e2_rstar

    echo "=========================================="
    echo "Phase 3: evaluate  lambda=$LAM  $(date '+%H:%M:%S')"
    echo "=========================================="
    python -m sep.transfer.transfer2 evaluate \
        $COMMON \
        $PAIR_ARGS

    echo "--- Done lambda=$LAM  $(date '+%H:%M:%S') ---"
    echo ""
done

check_branch
echo "=========================================="
echo "Sweep summary: cross-lambda comparison  $(date '+%H:%M:%S')"
echo "=========================================="
python -m sep.transfer.sweep_summary \
    --out-dir "$OUT" --hparam lambda --datasets nq

echo ""
echo "All lambdas complete.  $(date '+%H:%M:%S')"
echo "Alignment caches:  $OUT/alignments/nq/<pair>/align_nq_e2_rstar_l*.pkl"
echo "Curves:            $OUT/results/nq/<pair>/probe_grid_align_nq_*e2_rstar_l*.json"
echo "Cross-lambda:      $OUT/sweep_summary/  (hparam_lambda_table_*, lambda_curves_nq_align_nq.*)"
