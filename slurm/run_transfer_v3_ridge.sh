#!/bin/bash
# SE probe transfer, normalisation convention D -- ONE transfer mode, ridge only.
#
# This is the parametrised worker.  To run all four modes use the driver:
#   slurm/run_transfer_v3_all.sh
#
# WHY A NEW RUN AT ALL
# --------------------
# transfer2.py used to normalise the SOURCE alignment features with the align
# dataset's own best-layer mu/sd instead of the statistics at Ls, and it fitted
# the map in align_ds target coordinates while applying it in eval_ds ones.  Both
# are fixed; the code is now convention D (see the transfer2.py module docstring).
# Every cross-align artefact written before that fix is incomparable with these,
# so this run shares NOTHING with the older output trees -- no symlinks, no
# copied probe caches.  Cross-align caches from an older convention are rejected
# at load time by _check_norm_convention(), so an accidental mix fails loudly
# rather than silently mixing conventions.
#
# Probe caches ARE shared between the four modes of THIS run, via PROBE_SRC (the
# driver builds one set first).  They are transfer-mode independent -- phase 1
# fits the same probes at the same layers off the same 500/1500 split regardless
# of --transfer-mode -- and sharing them is what makes the four modes comparable
# cell by cell rather than merely on average.  The one asymmetry:
# best-to-best-sub writes sub_best_layer back into each pkl, so it gets a real
# COPY while the read-only modes get a symlink.
#
# SCOPE
# -----
#   aligner:      ridge only, single alpha=1e4.  No procrustes, no e2_rstar, no
#                 alpha sweep.
#   modes:        one of best-to-best | best-to-last | best-to-best-sub |
#                 best-to-align, via MODE (see driver).
#   datasets:     3 same-align + 6 cross-align = 9 (eval_ds, align_ds) combos
#                 over {trivia_qa, squad, nq}; same-align combos are added
#                 implicitly by _resolve_pairs() for every --datasets entry.
#   pairs:        21 model pairs from slurm/inputs/pair_list.txt.
#                 21 x 9 = 189 (pair, eval_ds, align_ds) cells per mode.
#
# Usage:
#   MODE=best-to-best bash slurm/run_transfer_v3_ridge.sh
#   OUT_ROOT=/somewhere MODE=best-to-align bash slurm/run_transfer_v3_ridge.sh
#
# Env knobs:
#   MODE                  required, transfer mode
#   OUT_ROOT              parent of the per-mode output dirs
#   OUT                   override the output dir outright (ignores OUT_ROOT/MODE)
#   PROBE_SRC             a convention-D run dir to take probes/ from; symlinked
#                         for read-only modes, copied for best-to-best-sub
#   PHASE1_ONLY=1         stop after phase 1 (how the driver builds PROBE_SRC)
#   STRIDE / HOLDOUT      best-to-align layer search only (stride k scores every
#                         k-th candidate layer; the residual curve is smooth in L)
#   SKIP_SUMMARY=1        stop after phase 3

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE=offline
# This repo's src, NOT /build_bak/UQ/UQ-transfer/src -- that copy is owned by
# another account and predates the convention-D fix entirely.
export PYTHONPATH="$REPO_ROOT/src":/build_bak/UQ/python_packages
PYTHON=${PYTHON:-/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python}

MODE=${MODE:?set MODE to one of best-to-best best-to-last best-to-best-sub best-to-align}
case "$MODE" in
    best-to-best)     TAG=b2b ;;
    best-to-last)     TAG=btl ;;
    best-to-best-sub) TAG=bbs ;;
    best-to-align)    TAG=b2a ;;
    *) echo "unknown MODE '$MODE'" >&2; exit 2 ;;
esac

OUT_ROOT=${OUT_ROOT:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v3}
OUT=${OUT:-"$OUT_ROOT/transfer_v3_$TAG"}
PROBE_SRC=${PROBE_SRC:-}
PHASE1_ONLY=${PHASE1_ONLY:-0}
STRIDE=${STRIDE:-1}
HOLDOUT=${HOLDOUT:-0.2}
SKIP_SUMMARY=${SKIP_SUMMARY:-0}

COMMON="--out-dir $OUT --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 \
        --seed 0 --alpha 1e4 --transfer-mode $MODE \
        --align-sel-aligner ridge --align-sel-holdout $HOLDOUT \
        --align-sel-stride $STRIDE"

PAIR_ARGS="--datasets trivia_qa squad nq \
           --cross-align-dataset trivia_qa:squad trivia_qa:nq \
                                 squad:trivia_qa squad:nq \
                                 nq:trivia_qa nq:squad"

mkdir -p "$OUT"

echo "=========================================="
echo "mode $MODE  ->  $OUT   $(date '+%F %H:%M:%S')"
echo "convention: $($PYTHON -c 'import sep.transfer.transfer2 as t; print(t.NORM_CONVENTION)')"
echo "=========================================="

# --- seed probes/ from PROBE_SRC.  A symlink is enough for the read-only modes;
# best-to-best-sub writes sub_best_layer back into each pkl, so it must own real
# files or it would mutate the shared set (and, if the shared set were a symlink
# into a third tree, someone else's).
if [ -n "$PROBE_SRC" ] && [ ! -e "$OUT/probes" ]; then
    if [ ! -d "$PROBE_SRC" ]; then
        echo "ERROR: PROBE_SRC=$PROBE_SRC is not a directory" >&2; exit 1
    fi
    if [ "$MODE" = best-to-best-sub ]; then
        cp -rL "$PROBE_SRC" "$OUT/probes"          # -L: materialise, never chain
        echo "probes/ copied from $PROBE_SRC (this mode writes sub_best_layer)"
    else
        ln -s "$PROBE_SRC" "$OUT/probes"
        echo "probes/ -> $PROBE_SRC (read-only in this mode)"
    fi
fi
if [ "$MODE" = best-to-best-sub ] && [ -L "$OUT/probes" ]; then
    echo "ERROR: $OUT/probes is a symlink and $MODE writes sub_best_layer into" >&2
    echo "       each pkl.  Materialise it first:  cp -rL --remove-destination" >&2
    exit 1
fi

echo "=== Phase 1: probe cache  $(date '+%H:%M:%S')"
# Idempotent: prints "skip <ds>/<model> (exists)" for anything PROBE_SRC already
# provided and fits only genuinely missing entries.  For best-to-best-sub it also
# fills in sub_best_layer, which is the one field the shared caches lack.
$PYTHON -m sep.transfer.transfer2 probe_cache \
    $COMMON --datasets trivia_qa squad nq

if [ "$PHASE1_ONLY" != "0" ]; then
    echo "PHASE1_ONLY set -- probe caches ready at $OUT/probes"
    exit 0
fi

if [ "$MODE" = best-to-align ]; then
    echo "=== Phase 1.5: label-free target layer choice  $(date '+%H:%M:%S')"
    # tgt_layers/<eval_ds>/<pair>/<align_ds>_b2a.json, idempotent per combo.
    # 189 combos x ~33 candidate layers, one ridge solve each -- the expensive
    # phase of this mode (hours, and load-dependent).
    $PYTHON -m sep.transfer.transfer2 tgt_layer_cache $COMMON $PAIR_ARGS

    echo "=== Phase 1.5 report  $(date '+%H:%M:%S')"
    # Cache-only, seconds.  Kill the job here if regret is not clearly below
    # last_reg -- there is no point aligning to a layer choice that lost.
    $PYTHON -m sep.transfer.align_layer_report --out-dir "$OUT"
fi

echo "=== Phase 2: alignment cache  $(date '+%H:%M:%S')"
# alignments/<eval_ds>/<pair>/align_<align_ds>_ridge_a1e4[_<tag>].pkl
$PYTHON -m sep.transfer.transfer2 align_cache $COMMON $PAIR_ARGS --aligners ridge

echo "=== Phase 3: evaluate  $(date '+%H:%M:%S')"
# --aligners ridge is load-bearing: without it phase 3 also scans for
# procrustes/e2_rstar caches and appends their tags to the output filenames.
$PYTHON -m sep.transfer.transfer2 evaluate $COMMON $PAIR_ARGS --aligners ridge

if [ "$SKIP_SUMMARY" != "0" ]; then
    echo "SKIP_SUMMARY set -- stopping after phase 3."
    exit 0
fi

echo "=== Phase 4: summary plots + Venn  $(date '+%H:%M:%S')"
$PYTHON -m sep.transfer.transfer2 summary $COMMON $PAIR_ARGS

echo ""
echo "mode $MODE done.  $(date '+%F %H:%M:%S')"
echo "Results:       $OUT/results/"
echo "Summary plots: $OUT/summary_plots/"
