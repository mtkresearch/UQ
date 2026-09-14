#!/bin/bash
# SE probe transfer — best-to-best-SUB (BBS) mode, FULL 3x3 dataset grid, ridge alpha=1e4.
#
#   Source probe:  each model's best layer, chosen with all 1500 pool labels
#                  (read straight from the copied probe caches -- unchanged).
#   Target layer:  the target's best layer, chosen with only 150 target SE labels
#                  via 5-fold CV on pool[:150]  (transfer2.py:_sub_layer_search).
#   Everything after the layer choice -- alignment fit, the probe-size grid, the
#   500-row eval -- is IDENTICAL to best-to-best.  Only Lt differs.
#
# Point of the run: best-to-best spends 1500 target labels just to pick a layer,
# which undercuts the "transfer needs few target labels" claim; best-to-last
# spends none but gives up ~0.06 AUROC.  BBS asks whether 150 labels buy back
# most of that gap.
#
# Covers all 9 (eval_ds, align_ds) combinations over {trivia_qa, squad, nq}:
#
#                 align=trivia_qa   align=squad   align=nq
#   eval=trivia_qa      same           cross        cross
#   eval=squad          cross          same         cross
#   eval=nq             cross          cross        same
#
# same-align combos are added automatically by _resolve_pairs() for every entry
# in --datasets, so --cross-align-dataset lists only the 6 off-diagonal cells.
# 21 model pairs x 9 combos = 189 alignment fits, ~2 min each => ~6.5 h Phase 2.
# Phase 2/3 skip any (pair, eval_ds, align_ds) whose _bbs pkl already exists, so
# an interrupted run resumes for free.
#
# No alpha sweep here: a single ridge alpha=1e4, matching the published runs.
# The sweep is a separate script (slurm/run_transfer_alpha_sweep.sh).
#
# ---------------------------------------------------------------------------
# Probe caches: deep-copied from transfer_v2_pooled (the leak-free run), so the
# 500/1500 eval/pool splits match the best-to-best results this is compared
# against, and Phase 1 never redraws them.  They MUST be real writable files,
# not symlinks: Phase 1 writes sub_best_layer back INTO each pkl, which through
# a symlink would mutate the other run's cache -- and transfer_v2_pooled's
# trivia_qa used to point into another account's read-only tree, where that
# write fails outright.  The preflight below enforces this.
# ---------------------------------------------------------------------------
#
# After Phase 1 the script prints and saves a sub-layer report
# ($OUT/sub_layer_report.csv) comparing the 150-label layer choice against the
# 1500-label one.  Inspect it and kill the job if the regret is not worth the
# remaining ~7 h.  STOP_AFTER_PHASE1=1 stops there on purpose.
#
# Usage:
#   nohup bash slurm/run_transfer_v2_bbs_all.sh \
#     >> /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_bbs/run_all.log 2>&1 &
#
#   # just the layer report, no alignment:
#   STOP_AFTER_PHASE1=1 bash slurm/run_transfer_v2_bbs_all.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE=offline
# This repo's src, NOT /build_bak/UQ/UQ-transfer/src -- that copy is owned by
# another account, is read-only to us, and predates best-to-best-sub entirely.
export PYTHONPATH="$REPO_ROOT/src":/build_bak/UQ/python_packages
PYTHON=${PYTHON:-/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python}

OUT=${OUT:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_bbs}
PROBE_SRC=${PROBE_SRC:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled}
SUB_N=${SUB_N:-150}
SUB_FOLDS=${SUB_FOLDS:-5}
STOP_AFTER_PHASE1=${STOP_AFTER_PHASE1:-0}

COMMON="--out-dir $OUT --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 --seed 0 \
        --alpha 1e4 --transfer-mode best-to-best-sub --sub-n $SUB_N --sub-folds $SUB_FOLDS"

# 3 same-align (implicit) + 6 cross-align (explicit) = 9 combos.
PAIR_ARGS="--datasets trivia_qa squad nq \
           --cross-align-dataset trivia_qa:squad trivia_qa:nq \
                                 squad:trivia_qa squad:nq \
                                 nq:trivia_qa nq:squad"

mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# Preflight: seed probes/ from PROBE_SRC, dereferencing symlinks, and verify
# every pkl is writable before Phase 1 tries to add sub_best_layer to it.
# cp -L follows symlinks; -n never clobbers a cache already here.
# ---------------------------------------------------------------------------
if [ -L "$OUT/probes" ]; then
    echo "ERROR: $OUT/probes is a symlink -> $(readlink "$OUT/probes")" >&2
    echo "       Phase 1 would write sub_best_layer into that run's cache." >&2
    exit 1
fi
cp -rLn "$PROBE_SRC/probes" "$OUT/" 2>/dev/null || true
chmod -R u+w "$OUT/probes"

echo "=== probe cache inventory before Phase 1"
fail=0
for ds in trivia_qa squad nq; do
    have=$(find "$OUT/probes/$ds" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l)
    rw=$(find "$OUT/probes/$ds" -maxdepth 1 -name '*.pkl' -writable 2>/dev/null | wc -l)
    want=$(awk -v d="$ds" '!/^[[:space:]]*(#|$)/ && $1==d' slurm/inputs/model_paths.txt | wc -l)
    printf "  %-10s %s / %s present, %s writable\n" "$ds" "$have" "$want" "$rw"
    [ "$have" -eq "$rw" ] || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "ERROR: some probe caches are read-only; Phase 1 cannot store sub_best_layer" >&2
    exit 1
fi
if find "$OUT/probes" -type l | grep -q .; then
    echo "ERROR: symlinks remain under $OUT/probes" >&2
    exit 1
fi

echo "=========================================="
echo "Phase 1: probe cache + ${SUB_N}-label layer search  $(date '+%H:%M:%S')"
echo "=========================================="
# Probes themselves are all pre-copied, so this prints "skip <ds>/<model>
# (exists)" for each and then runs ONLY the sub-budget layer search, backfilling
# sub_best_layer into the pkl.  22 (dataset, model) entries; the cost is loading
# each one's hidden states once (~30 min total), the search itself is seconds.
$PYTHON -m sep.transfer.transfer2 probe_cache \
    $COMMON --datasets trivia_qa squad nq

echo "=========================================="
echo "Phase 1 report: ${SUB_N}-label vs 1500-label layer choice  $(date '+%H:%M:%S')"
echo "=========================================="
# Cache-only, seconds.  Kill the job here if regret is not clearly below last_reg.
$PYTHON -m sep.transfer.sub_layer_report --out-dir "$OUT"

if [ "$STOP_AFTER_PHASE1" != "0" ]; then
    echo "STOP_AFTER_PHASE1 set -- stopping before alignment."
    echo "Report: $OUT/sub_layer_report.csv"
    exit 0
fi

echo "=========================================="
echo "Phase 2: alignment cache  $(date '+%H:%M:%S')"
echo "=========================================="
# Output: alignments/<eval_ds>/<pair>/align_<align_ds>_ridge_a1e4_bbs.pkl
$PYTHON -m sep.transfer.transfer2 align_cache \
    $COMMON $PAIR_ARGS --aligners ridge

echo "=========================================="
echo "Phase 3: evaluate  $(date '+%H:%M:%S')"
echo "=========================================="
# results/<eval_ds>/<pair>/
#   native_curves_bbs_<eval_ds>.json
#   probe_grid_bbs_align_<align_ds>_ridge_a1e4.json
#   predictions_bbs_align_<align_ds>_ridge_a1e4.json
$PYTHON -m sep.transfer.transfer2 evaluate \
    $COMMON $PAIR_ARGS --aligners ridge

echo "=========================================="
echo "Phase 4: summary plots + Venn  $(date '+%H:%M:%S')"
echo "=========================================="
# summary_plots/summary_{trivia_qa,squad,nq}_probe_grid_bbs_ridge_a1e4.{pdf,png}
$PYTHON -m sep.transfer.transfer2 summary \
    $COMMON $PAIR_ARGS

echo ""
echo "All done.  $(date '+%H:%M:%S')"
echo "Layer report:  $OUT/sub_layer_report.csv"
echo "Results:       $OUT/results/"
echo "Summary plots: $OUT/summary_plots/"
