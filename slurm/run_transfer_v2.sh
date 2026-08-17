#!/bin/bash
# Efficient SE probe transfer pipeline (transfer2.py).
# Eliminates redundant probe training via caching.
#
# Phase 1: probe_cache  — fit probes for all 16 (model, dataset) combinations
# Phase 2: align_cache  — fit alignment matrices for all 24 (pair, eval_ds, align_ds)
# Phase 3: evaluate     — compute AUROC curves (alignment_grid + probe_grid)
# Phase 4: summary      — plot 4 summary figures
#
# Usage:
#   mkdir -p /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2
#   nohup bash slurm/run_transfer_v2.sh \
#     > /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2/run.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE=offline

OUT=/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2
COMMON="--out-dir $OUT --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 --seed 0 --alpha 1e3"
PAIR_ARGS="--datasets squad nq --cross-align-dataset squad:nq nq:squad"

mkdir -p "$OUT"

echo "=========================================="
echo "Phase 1: probe cache  $(date '+%H:%M:%S')"
echo "=========================================="
python -m sep.transfer.transfer2 probe_cache \
    $COMMON $PAIR_ARGS

echo "=========================================="
echo "Phase 2: alignment cache  $(date '+%H:%M:%S')"
echo "=========================================="
python -m sep.transfer.transfer2 align_cache \
    $COMMON $PAIR_ARGS

echo "=========================================="
echo "Phase 3: evaluate  $(date '+%H:%M:%S')"
echo "=========================================="
python -m sep.transfer.transfer2 evaluate \
    $COMMON $PAIR_ARGS

echo "=========================================="
echo "Phase 4: summary plots  $(date '+%H:%M:%S')"
echo "=========================================="
python -m sep.transfer.transfer2 summary \
    $COMMON $PAIR_ARGS

echo ""
echo "All done.  $(date '+%H:%M:%S')"
echo "Results:      $OUT/results/"
echo "Summary plots: $OUT/summary_plots/"
