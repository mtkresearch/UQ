#!/bin/bash
# E2-R* alignment only, lambda=1e5, on nq + squad, with cross-alignment.
#
# Four (eval_ds, align_ds) combos per model pair -- transfer2 always emits the same-align
# combo, so `--datasets nq squad --cross-align-dataset nq:squad squad:nq` gives exactly:
#     (nq, nq)      (nq, squad)      (squad, squad)      (squad, nq)
#
# Phases: 1 (no-op, probes cached) -> 2 align_cache --aligners e2_rstar -> 3 evaluate.
# Phase 4 (transfer2 summary) is skipped: it tags every output by the ridge alpha
# (transfer2.py:1254) and would mislabel an e2_rstar-only run.
#
# Phase 3 also gets --aligners e2_rstar, so it ignores the ridge caches sitting in this
# dir and all four combos are named uniformly:
#     probe_grid_align_<align_ds>_e2_rstar_l1e5.json
# Without that flag Phase 3 appends the tag of every aligner cache it finds, which here
# would give probe_grid_align_<ds>_ridge_a1e3_e2_rstar_l1e5.json for the same-align combos
# and the bare e2_rstar name for the cross-align ones -- inconsistent, and a second copy
# of numbers already stored under the bare name.
#
# Usage:
#   nohup bash slurm/run_transfer_e2rstar_l1e5.sh \
#     > /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled/run_e2rstar_l1e5.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXPECTED_BRANCH="exp/intra-family-transfer"
CURRENT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]; then
    echo "ERROR: expected branch '$EXPECTED_BRANCH' but on '$CURRENT_BRANCH'. Refusing to run (checking out mid-run deletes untracked-on-that-branch files)." >&2
    exit 1
fi

export WANDB_MODE=offline

OUT=${OUT:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled}
LAM=${LAM:-1e5}
BASE_ARGS="--out-dir $OUT --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 --seed 0"
COMMON="$BASE_ARGS --lambda-val $LAM"
PAIR_ARGS="--datasets nq squad --cross-align-dataset nq:squad squad:nq"

mkdir -p "$OUT"
# copies the probe caches over so Phase 1 does not redraw the eval/pool splits
OUT_NEW="$OUT" bash slurm/bootstrap_pooled_outdir.sh

echo "=========================================="
echo "Phase 1: probe_cache (no-op if cached)  $(date '+%H:%M:%S')"
echo "=========================================="
python -m sep.transfer.transfer2 probe_cache \
    $BASE_ARGS $PAIR_ARGS

echo "=========================================="
echo "Phase 2: align_cache  e2_rstar lambda=$LAM  $(date '+%H:%M:%S')"
echo "=========================================="
# Existing align_<ds>_e2_rstar_l<LAM>.pkl files are skipped unless --force, so the
# (nq, nq) combo the lambda sweep already produced costs nothing here.
python -m sep.transfer.transfer2 align_cache \
    $COMMON $PAIR_ARGS \
    --aligners e2_rstar

echo "=========================================="
echo "Phase 3: evaluate  lambda=$LAM  $(date '+%H:%M:%S')"
echo "=========================================="
python -m sep.transfer.transfer2 evaluate \
    $COMMON $PAIR_ARGS \
    --aligners e2_rstar

echo ""
echo "Done.  $(date '+%H:%M:%S')"
echo "Alignments:  $OUT/alignments/{nq,squad}/<pair>/align_{nq,squad}_e2_rstar_l$LAM.pkl"
echo "Curves:      $OUT/results/{nq,squad}/<pair>/probe_grid_align_{nq,squad}_e2_rstar_l$LAM.json"
