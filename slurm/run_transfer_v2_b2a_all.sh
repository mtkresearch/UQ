#!/bin/bash
# SE probe transfer — best-to-align (B2A) mode, FULL 3x3 dataset grid, ridge alpha=1e4.
#
#   Source probe:  trained at each model's best layer on the eval dataset.
#   Target layer:  chosen WITHOUT ANY TARGET SE LABEL -- one alignment is fitted
#                  per candidate target layer L and the winner minimises the
#                  held-out ||R_L(h_t) - h_s||^2.  See Phase 1.5 below.
#   Alignment:     best layer of source -> that layer, refitted per n on the
#                  alignment dataset's hidden states (identical to every mode).
#
# This is the label-budget-zero counterpart to best-to-best-sub (150 labels) and
# the accuracy counterpart to best-to-last (0 labels but a fixed, usually poor
# layer).  The comparison to make after Phase 1.5 is regret vs last_reg in the
# align_layer_report.
#
# Covers all 9 (eval_ds, align_ds) combinations over {trivia_qa, squad, nq}:
# same-align combos are added automatically by _resolve_pairs() for every entry
# in --datasets, so --cross-align-dataset lists only the 6 off-diagonal cells.
# 21 model pairs x 9 combos = 189 (pair, eval_ds, align_ds) experiments.
#
# ---------------------------------------------------------------------------
# COST -- Phase 1.5 is the expensive new part.
# Unlike the other modes, this layer choice depends on (pair, eval_ds, align_ds),
# so it runs 189 times, each scoring ~33 target layers with one ridge fit per
# layer (the same d^3 solve as a real alignment fit).  Wall-clock therefore
# tracks machine load: this box has recorded anywhere from 0.7 s to 13 s for that
# identical solve, i.e. ~1.7 h to ~13 h for the full grid.  STRIDE=k scores every
# k-th layer if you need to cut it; the residual curve is smooth in L.
# STOP_AFTER_PHASE15=1 stops after the layer report.
# ---------------------------------------------------------------------------
#
# Probe caches: seeded from PROBE_SRC (transfer_v2_pooled, the leak-free run) so
# the 500/1500 eval/pool splits match the best-to-best results this is compared
# against.  Phase 1 is still required -- phase 3 needs the source probes and the
# target EVAL labels from those caches; what B2A avoids is spending labels on the
# layer CHOICE.  Unlike the bbs run, nothing here writes back into probes/, so a
# symlinked probes/ is safe (and cheaper) than a copy.  NOTE: that symlink means
# you must materialise probes/ into a real directory before running bbs in the
# same OUT -- bbs writes sub_best_layer back into each pkl.
#
# Usage:
#   nohup bash slurm/run_transfer_v2_b2a_all.sh \
#     >> /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_b2a/run_all.log 2>&1 &
#
#   # just the label-free layer choices + report, no alignment:
#   STOP_AFTER_PHASE15=1 bash slurm/run_transfer_v2_b2a_all.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE=offline
# This repo's src, NOT /build_bak/UQ/UQ-transfer/src -- that copy is owned by
# another account, is read-only to us, and predates best-to-align entirely.
export PYTHONPATH="$REPO_ROOT/src":/build_bak/UQ/python_packages
PYTHON=${PYTHON:-/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python}

OUT=${OUT:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_b2a}
PROBE_SRC=${PROBE_SRC:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled}
STRIDE=${STRIDE:-1}
HOLDOUT=${HOLDOUT:-0.2}
SEL_ALIGNER=${SEL_ALIGNER:-ridge}
STOP_AFTER_PHASE15=${STOP_AFTER_PHASE15:-0}

COMMON="--out-dir $OUT --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 --seed 0 \
        --alpha 1e4 --transfer-mode best-to-align \
        --align-sel-aligner $SEL_ALIGNER --align-sel-holdout $HOLDOUT \
        --align-sel-stride $STRIDE"

# 3 same-align (implicit) + 6 cross-align (explicit) = 9 combos.
PAIR_ARGS="--datasets trivia_qa squad nq \
           --cross-align-dataset trivia_qa:squad trivia_qa:nq \
                                 squad:trivia_qa squad:nq \
                                 nq:trivia_qa nq:squad"

mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# Preflight: make probes/ available.  Read-only is fine for this mode, so a
# symlink to PROBE_SRC is the cheapest correct option; an existing real
# directory is left alone.
# ---------------------------------------------------------------------------
if [ ! -e "$OUT/probes" ]; then
    ln -s "$PROBE_SRC/probes" "$OUT/probes"
    echo "probes/ -> $PROBE_SRC/probes (read-only in this mode)"
fi

echo "=== probe cache inventory"
for ds in trivia_qa squad nq; do
    # find -L, not `ls dir/*.pkl`: an unmatched glob exits 2 and `set -o pipefail`
    # would abort the script from inside this command substitution.  -L follows
    # the probes/ symlink.
    have=$(find -L "$OUT/probes/$ds" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l)
    want=$(awk -v d="$ds" '!/^[[:space:]]*(#|$)/ && $1==d' slurm/inputs/model_paths.txt | wc -l)
    printf "  %-10s %s / %s\n" "$ds" "$have" "$want"
done

echo "=========================================="
echo "Phase 1: probe cache  $(date '+%H:%M:%S')"
echo "=========================================="
# Idempotent: prints "skip <ds>/<model> (exists)" for every seeded probe and
# fits only genuinely missing ones.  Probe caches are transfer-mode independent.
$PYTHON -m sep.transfer.transfer2 probe_cache \
    $COMMON --datasets trivia_qa squad nq

echo "=========================================="
echo "Phase 1.5: label-free target layer choice  $(date '+%H:%M:%S')"
echo "=========================================="
# Output: tgt_layers/<eval_ds>/<pair>/<align_ds>_b2a.json  (idempotent per combo,
# so an interrupted run resumes without recomputing finished combos)
$PYTHON -m sep.transfer.transfer2 tgt_layer_cache \
    $COMMON $PAIR_ARGS

echo "=========================================="
echo "Phase 1.5 report: label-free vs 1500-label layer choice  $(date '+%H:%M:%S')"
echo "=========================================="
# Cache-only, seconds.  Kill the job here if regret is not clearly below last_reg.
$PYTHON -m sep.transfer.align_layer_report --out-dir "$OUT"

if [ "$STOP_AFTER_PHASE15" != "0" ]; then
    echo "STOP_AFTER_PHASE15 set -- stopping before alignment."
    echo "Report: $OUT/align_layer_report.csv"
    exit 0
fi

echo "=========================================="
echo "Phase 2: alignment cache  $(date '+%H:%M:%S')"
echo "=========================================="
# Output: alignments/<eval_ds>/<pair>/align_<align_ds>_ridge_a1e4_b2a.pkl
$PYTHON -m sep.transfer.transfer2 align_cache \
    $COMMON $PAIR_ARGS --aligners ridge

echo "=========================================="
echo "Phase 3: evaluate  $(date '+%H:%M:%S')"
echo "=========================================="
# results/<eval_ds>/<pair>/
#   native_curves_b2a_<eval_ds>.json
#   probe_grid_b2a_align_<align_ds>_ridge_a1e4.json
#   predictions_b2a_align_<align_ds>_ridge_a1e4.json
$PYTHON -m sep.transfer.transfer2 evaluate \
    $COMMON $PAIR_ARGS --aligners ridge

echo "=========================================="
echo "Phase 4: summary plots + Venn  $(date '+%H:%M:%S')"
echo "=========================================="
# summary_plots/summary_{trivia_qa,squad,nq}_probe_grid_b2a_ridge_a1e4.{pdf,png}
$PYTHON -m sep.transfer.transfer2 summary \
    $COMMON $PAIR_ARGS

echo ""
echo "All done.  $(date '+%H:%M:%S')"
echo "Results:       $OUT/results/"
echo "Summary plots: $OUT/summary_plots/"
