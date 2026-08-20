#!/bin/bash
# SE probe transfer pipeline (transfer2.py) — SQuAD/NQ probe, cross-alignment with TriviaQA.
#
# Complements run_transfer_v2.sh (which covers squad:nq and nq:squad).
# This script adds the two TriviaQA cross-alignment directions:
#   train (squad), align (trivia_qa)
#   train (nq),    align (trivia_qa)
#
# Prerequisites:
#   - probes/squad/ and probes/nq/ already cached by run_transfer_v2.sh
#   - probes/trivia_qa/ symlinked from transfer_v2_trivia_qa (already done)
#
# Phase 1: probe_cache  — skipped (all probes already on disk)
# Phase 2: align_cache  — fits alignment matrices for squad:trivia_qa and nq:trivia_qa
# Phase 3: evaluate     — AUROC curves for the new cross-align pairs
# Phase 4: summary      — regenerates summary figures with the new dashed curves
#
# Usage:
#   nohup bash slurm/run_transfer_v2_cross_align_triviaqa.sh \
#     >> /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2/run_cross_align_triviaqa.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE=offline
export PYTHONPATH=/build_bak/UQ/UQ-transfer/src:/build_bak/UQ/python_packages
PYTHON=/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python

OUT=/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2
COMMON="--out-dir $OUT --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 --seed 0 --alpha 1e4"
PAIR_ARGS="--datasets squad nq --cross-align-dataset squad:trivia_qa nq:trivia_qa"

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
