#!/bin/bash
# SE probe transfer pipeline (transfer2.py) — TriviaQA, alpha=1e4, no cross-alignment.
#
# Phase 1: probe_cache  — fit probes for all 6 (model, trivia_qa) combinations
# Phase 2: align_cache  — fit alignment matrices for all 21 pairs
# Phase 3: evaluate     — compute AUROC curves
# Phase 4: summary      — summary figures + Venn diagrams
#
# Usage:
#   mkdir -p /build_bak/UQ/UQ-transfer/sep_scratch/transfer_v2_trivia_qa
#   nohup bash slurm/run_transfer_v2_trivia_qa.sh \
#     > /build_bak/UQ/UQ-transfer/sep_scratch/transfer_v2_trivia_qa/run.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE=offline
export PYTHONPATH=/build_bak/UQ/UQ-transfer/src:/build_bak/UQ/python_packages
PYTHON=/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python

OUT=/build_bak/UQ/UQ-transfer/sep_scratch/transfer_v2_trivia_qa
COMMON="--out-dir $OUT --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 --seed 0 --alpha 1e4"
PAIR_ARGS="--datasets trivia_qa"

mkdir -p "$OUT"

echo "=========================================="
echo "Phase 1: probe cache  $(date '+%H:%M:%S')"
echo "=========================================="
$PYTHON -m sep.transfer.transfer2 probe_cache \
    $COMMON $PAIR_ARGS

echo "=========================================="
echo "Phase 2: alignment cache  $(date '+%H:%M:%S')"
echo "=========================================="
$PYTHON -m sep.transfer.transfer2 align_cache \
    $COMMON $PAIR_ARGS

echo "=========================================="
echo "Phase 3: evaluate  $(date '+%H:%M:%S')"
echo "=========================================="
$PYTHON -m sep.transfer.transfer2 evaluate \
    $COMMON $PAIR_ARGS

echo "=========================================="
echo "Phase 4: summary plots + Venn  $(date '+%H:%M:%S')"
echo "=========================================="
$PYTHON -m sep.transfer.transfer2 summary \
    $COMMON $PAIR_ARGS

echo ""
echo "All done.  $(date '+%H:%M:%S')"
echo "Results:       $OUT/results/"
echo "Summary plots: $OUT/summary_plots/"
