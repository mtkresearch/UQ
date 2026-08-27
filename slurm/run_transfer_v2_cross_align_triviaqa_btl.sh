#!/bin/bash
# SE probe transfer — NQ + SQuAD eval, TriviaQA alignment, best-to-last (BTL) mode.
#
# BTL counterpart of run_transfer_v2_cross_align_triviaqa.sh:
#   Source probe:  trained at each model's best layer on NQ / SQuAD.
#   Alignment:     best layer of source  →  last layer of target,
#                  using TriviaQA hidden states as the alignment dataset.
#   Native target: probe trained and evaluated at target's last layer.
#   Eval datasets: nq, squad.
#
# Phase 1 is skipped — probe caches for nq, squad, and trivia_qa already
# exist in transfer_v2/probes/ and are transfer-mode-independent.
#
# Usage:
#   nohup bash slurm/run_transfer_v2_cross_align_triviaqa_btl.sh \
#     >> /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_btl/run.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE=offline
export PYTHONPATH=/build_bak/UQ/UQ-transfer/src:/build_bak/UQ/python_packages
PYTHON=/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python

OUT=/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_btl

COMMON="--out-dir $OUT --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 --seed 0 \
        --alpha 1e4 --transfer-mode best-to-last"
PAIR_ARGS="--datasets squad nq --cross-align-dataset squad:trivia_qa nq:trivia_qa"

mkdir -p "$OUT"

# Phase 1 is skipped: probe caches are identical across transfer modes.
# Symlink from the existing run which already has nq/, squad/, and trivia_qa/.
BTB=/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2
if [ ! -e "$OUT/probes" ]; then
    ln -s "$BTB/probes" "$OUT/probes"
    echo "Symlinked probes/ from $BTB"
else
    echo "probes/ already present, skipping symlink"
fi

echo "=========================================="
echo "Phase 2: alignment cache  $(date '+%H:%M:%S')"
echo "=========================================="
# Fits ridge alignment matrices: source best layer → target last layer.
# Output: alignments/<eval_ds>/<pair>/align_trivia_qa_ridge_a1e4_btl.pkl
$PYTHON -m sep.transfer.transfer2 align_cache \
    $COMMON $PAIR_ARGS --aligners ridge

echo "=========================================="
echo "Phase 3: evaluate  $(date '+%H:%M:%S')"
echo "=========================================="
# Computes AUROC curves and saves:
#   native_curves_btl_{nq,squad}.json  (native target probe at last layer)
#   probe_grid_btl_align_trivia_qa_ridge_a1e4.json
#   predictions_btl_align_trivia_qa_ridge_a1e4.json
$PYTHON -m sep.transfer.transfer2 evaluate \
    $COMMON $PAIR_ARGS --aligners ridge

echo "=========================================="
echo "Phase 4: summary plots + Venn  $(date '+%H:%M:%S')"
echo "=========================================="
# Summary figures: summary_{nq,squad}_probe_grid_btl_ridge_a1e4.{pdf,png}
# Venn diagrams:   summary_plots/venn_ridge_a1e4/
$PYTHON -m sep.transfer.transfer2 summary \
    $COMMON $PAIR_ARGS

echo ""
echo "All done.  $(date '+%H:%M:%S')"
echo "Results:       $OUT/results/"
echo "Summary plots: $OUT/summary_plots/"
