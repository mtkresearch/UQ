#!/bin/bash
# Driver: all four ridge transfer modes under normalisation convention D and the
# per-dataset eval/pool split.
#
# Output tree.  Nothing is shared with transfer_v2* or transfer_v3* -- v4 changes
# the split, so every artefact in those trees is on different eval rows and none of
# it is reusable.  There is deliberately no seed script; see
# run_transfer_v4_ridge.sh for the full reasoning.
#   $OUT_ROOT/_probes_shared/probes   phase 1, built once
#   $OUT_ROOT/transfer_v4_b2b         best-to-best
#   $OUT_ROOT/transfer_v4_btl         best-to-last
#   $OUT_ROOT/transfer_v4_bbs         best-to-best-sub
#   $OUT_ROOT/transfer_v4_b2a         best-to-align
# each with probes/ alignments/ results/ summary_plots/ timing.json.
#
# Phase 1 (probe caches) runs ONCE into _probes_shared and every mode picks it up
# via PROBE_SRC: probes are transfer-mode independent, and sharing them means the
# four modes differ only in the thing being compared.  b2b/btl/b2a symlink it;
# bbs gets a real copy because it writes sub_best_layer back into each pkl.
#
# Order is cheapest-first so a failure surfaces early and the expensive mode
# (b2a, whose phase 1.5 is 189 x ~33 ridge solves) runs last.
#
# Modes are run SEQUENTIALLY.  They are independent and could be parallel, but
# each one is already numpy-BLAS-bound on all cores; running them concurrently
# mostly trades wall-clock for cache thrash.  Set MODES to a subset to split the
# work across machines instead.
#
# Usage:
#   nohup bash slurm/run_transfer_v4_all.sh \
#     >> /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v4/run_all.log 2>&1 &
#
#   # one mode only, or a different order:
#   MODES="best-to-align" bash slurm/run_transfer_v4_all.sh
#
#   # everything except the plots:
#   SKIP_SUMMARY=1 bash slurm/run_transfer_v4_all.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT=${OUT_ROOT:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v4}
# Cheapest-first (see the note above).  In v3 this was reordered to put
# best-to-align first because the cheap trees were seeded and only cross-align was
# outstanding; in v4 nothing is seeded, so putting the multi-hour phase 1.5 last
# means three complete trees land before it starts.  Reorder freely.
MODES=${MODES:-"best-to-best best-to-last best-to-best-sub best-to-align"}
SHARED="$OUT_ROOT/_probes_shared"
export OUT_ROOT

mkdir -p "$OUT_ROOT"

echo "##########################################"
echo "transfer v4 (convention D), ridge alpha=1e4"
echo "modes:    $MODES"
echo "out root: $OUT_ROOT"
echo "start:    $(date '+%F %H:%M:%S')"
echo "##########################################"

echo ""
echo ">>> shared phase 1: probe caches  ($(date '+%H:%M:%S'))"
# MODE here only picks the code path, not the probes: phase 1 is transfer-mode
# independent, so best-to-best is an arbitrary (and the cheapest) choice.  It is
# idempotent, so re-running the driver skips this in seconds.
MODE=best-to-best OUT="$SHARED" PHASE1_ONLY=1 \
    bash slurm/run_transfer_v4_ridge.sh 2>&1 | tee "$OUT_ROOT/_probes_shared.log"
export PROBE_SRC="$SHARED/probes"
echo ">>> shared probes: $PROBE_SRC ($(find -L "$PROBE_SRC" -name '*.pkl' | wc -l) pkl, expect 22)"

# Per-mode logs in addition to this driver's stdout, so one mode's failure is
# readable without scrolling past the others.
for mode in $MODES; do
    log="$OUT_ROOT/$(printf '%s' "$mode").log"
    echo ""
    echo ">>> $mode  ($(date '+%H:%M:%S'))  log: $log"
    if MODE="$mode" bash slurm/run_transfer_v4_ridge.sh 2>&1 | tee "$log"; then
        echo ">>> $mode OK  ($(date '+%H:%M:%S'))"
    else
        # Do not abort the remaining modes: they share no state, so a b2a
        # failure should not cost a finished best-to-best tree.
        echo ">>> $mode FAILED -- see $log; continuing with the rest" >&2
    fi
done

echo ""
echo "##########################################"
echo "all requested modes finished  $(date '+%F %H:%M:%S')"
for mode in $MODES; do
    case "$mode" in
        best-to-best) t=b2b ;; best-to-last) t=btl ;;
        best-to-best-sub) t=bbs ;; best-to-align) t=b2a ;;
    esac
    d="$OUT_ROOT/transfer_v4_$t"
    n=$(find "$d/results" -name 'probe_grid*ridge_a1e4*.json' 2>/dev/null | wc -l)
    printf "  %-18s %s  (%s probe_grid files, expect 189)\n" "$mode" "$d" "$n"
done
echo "##########################################"
