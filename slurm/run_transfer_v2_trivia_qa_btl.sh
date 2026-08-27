#!/bin/bash
# SE probe transfer — TriviaQA, best-to-last (BTL) mode, ridge alpha=1e4.
#
# Source probe:  trained at each model's best layer (Phase 1, unchanged).
# Alignment:     best layer of source  →  last layer of target.
# Native target: probe trained and evaluated at target's last layer.
# Eval dataset:  trivia_qa, same-align + cross-align with NQ and SQuAD.
#
# All output goes to a clean new directory (separate from the best-to-best run).
#
# Usage:
#   nohup bash slurm/run_transfer_v2_trivia_qa_btl.sh \
#     > /build_bak/UQ/UQ-transfer/sep_scratch/transfer_v2_trivia_qa_btl/run.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE=offline
export PYTHONPATH=/build_bak/UQ/UQ-transfer/src:/build_bak/UQ/python_packages
PYTHON=/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python

OUT=/build_bak/UQ/UQ-transfer/sep_scratch/transfer_v2_trivia_qa_btl

COMMON="--out-dir $OUT --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 --seed 0 \
        --alpha 1e4 --transfer-mode best-to-last"
PAIR_ARGS="--datasets trivia_qa --cross-align-dataset trivia_qa:nq trivia_qa:squad"

mkdir -p "$OUT"

echo "=========================================="
echo "Phase 1: probe cache  $(date '+%H:%M:%S')"
echo "=========================================="
# Fits probes at each model's best layer (same as best-to-best; transfer-mode is ignored here).
$PYTHON -m sep.transfer.transfer2 probe_cache \
    $COMMON $PAIR_ARGS

echo "=========================================="
echo "Phase 2: alignment cache  $(date '+%H:%M:%S')"
echo "=========================================="
# Fits ridge alignment matrices: source best layer → target last layer.
# Output: alignments/<eval_ds>/<pair>/align_<ds>_ridge_a1e4_btl.pkl
$PYTHON -m sep.transfer.transfer2 align_cache \
    $COMMON $PAIR_ARGS --aligners ridge

echo "=========================================="
echo "Phase 3: evaluate  $(date '+%H:%M:%S')"
echo "=========================================="
# Computes AUROC curves and saves:
#   native_curves_btl_trivia_qa.json  (native target probe at last layer)
#   probe_grid_btl_align_<ds>_ridge_a1e4.json
#   predictions_btl_align_<ds>_ridge_a1e4.json
$PYTHON -m sep.transfer.transfer2 evaluate \
    $COMMON $PAIR_ARGS --aligners ridge

echo "=========================================="
echo "Phase 4: summary plots + Venn  $(date '+%H:%M:%S')"
echo "=========================================="
# Summary figures: summary_trivia_qa_probe_grid_btl_ridge_a1e4.{pdf,png}
# Venn diagrams:   summary_plots/venn_ridge_a1e4/
$PYTHON -m sep.transfer.transfer2 summary \
    $COMMON $PAIR_ARGS

echo ""
echo "All done.  $(date '+%H:%M:%S')"
echo "Results:       $OUT/results/"
echo "Summary plots: $OUT/summary_plots/"
