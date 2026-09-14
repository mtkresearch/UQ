#!/bin/bash
# Efficient SE probe transfer pipeline (transfer2.py).
# Eliminates redundant probe training via caching.
#
# Phase 1: probe_cache  — fit probes for all 16 (model, dataset) combinations
# Phase 2: align_cache  — fit alignment matrices for every (pair, eval_ds, align_ds)
# Phase 3: evaluate     — compute AUROC curves (alignment_grid + probe_grid)
# Phase 4: summary      — plot 4 summary figures
#
#
# Writes to the leak-free dir transfer_v2_pooled/ (alignment fit on `pool`, disjoint from
# the eval rows). The old leaky results stay under transfer_v2/. Override OUT=...
#
# Overridable: OUT, ALPHA (ridge alpha, default 1e4), DATASETS, SUMMARY_DATASETS.
#
# Usage:
#   nohup bash slurm/run_transfer_v2.sh \
#     > /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled/run.log 2>&1 &
#
#   # squad only, ridge alpha=1e4:
#   ALPHA=1e4 DATASETS=squad nohup bash slurm/run_transfer_v2.sh \
#     > /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled/run_squad_a1e4.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE=offline

OUT=${OUT:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled}
ALPHA=${ALPHA:-1e4}
COMMON="--out-dir $OUT --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 --seed 0 --alpha $ALPHA"
# Same-align only. The cross-align combos are unaffected by the align_all fix, so
# recomputing them here would only be a consistency check -- add back
# `--cross-align-dataset squad:nq nq:squad` if you want them in this out dir.
#
# DATASETS narrows phases 1-3 (e.g. DATASETS=squad when the alpha sweep has already
# produced nq at this alpha). SUMMARY_DATASETS lets Phase 4 still plot both, since it
# only reads the grid JSONs phases 2-3 wrote, whoever wrote them.
DATASETS=${DATASETS:-"squad nq"}
SUMMARY_DATASETS=${SUMMARY_DATASETS:-"$DATASETS"}
PAIR_ARGS="--datasets $DATASETS"

mkdir -p "$OUT"
# copies the probe caches over so Phase 1 does not redraw the eval/pool splits
OUT_NEW="$OUT" bash slurm/bootstrap_pooled_outdir.sh

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
    $COMMON --datasets $SUMMARY_DATASETS

echo ""
echo "All done.  $(date '+%H:%M:%S')"
echo "Results:      $OUT/results/"
echo "Summary plots: $OUT/summary_plots/"
