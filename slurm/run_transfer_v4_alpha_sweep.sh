#!/bin/bash
# Ridge alpha ablation on the v4 tree (normalisation convention D, per-dataset split).
#
# SCOPE -- deliberately the smallest cell set that answers "which alpha":
#   dataset:   NQ only            (--datasets nq)
#   alignment: same-align only    (no --cross-align-dataset)
#   pairs:     21 from slurm/inputs/pair_list.txt
#   => 21 (pair, eval_ds, align_ds) cells per alpha, vs 189 in the full v4 run.
# Matches the scope of the v2 sweep (slurm/run_transfer_alpha_sweep.sh) and of the
# E2-R* lambda sweep, so the three ablations stay comparable.
#
# WHERE IT WRITES
# ---------------
# Into the EXISTING best-to-best v4 tree, $OUT_ROOT/transfer_v4_b2b, not a private
# directory.  Two reasons:
#   * sweep_summary compares the alphas already on disk under ONE --out-dir; split
#     them across directories and there is nothing to compare.
#   * alpha=1e4 is already there from the main v4 run (21 nq same-align cells), so
#     that value costs nothing -- phases 2 and 3 skip every cell.
# Alpha appears in every artefact name (align_nq_ridge_a1e<exp>.pkl,
# probe_grid_align_nq_ridge_a1e<exp>.json), so the added values sit beside the
# canonical 1e4 ones without shadowing them; the main run always passes --alpha 1e4
# explicitly and keeps reading its own files.
#
# WHY best-to-best ONLY
# ---------------------
# Sweeping another mode in one directory is not currently sound, and the script
# refuses unless FORCE_MODE=1:
#   * sweep_summary matches '^probe_grid_align_<ds>_ridge_a1e(-?\d+)' -- the
#     non-default modes emit probe_grid_<btl|bbs|b2a>_align_..., which that regex
#     never matches, so the sweep tables would come out empty.
#   * best-to-align is worse than empty: its layer choice DEPENDS on alpha
#     (_align_layer_search is handed args.alpha) but is cached at
#     tgt_layers/<eval_ds>/<pair>/<align_ds>_b2a.json with no alpha in the path,
#     and phase 1.5 skips on mere file existence without checking the sel_alpha it
#     recorded.  The first alpha's layers would be silently reused for all the
#     others, making the comparison meaningless rather than merely unreported.
# Fixing either is small (a mode infix in the regex; an alpha tag in the
# tgt_layers path) but neither is done, so do not pass FORCE_MODE blind.
#
# COST
# ----
# 4 new alphas x 21 cells.  Phase 2 dominates at ~2 min/cell => ~45 min per alpha,
# ~3-4 h total.  Every phase skips completed cells, so an interrupted sweep resumes
# for free and re-running the whole script is cheap.
#
# Usage:
#   nohup bash slurm/run_transfer_v4_alpha_sweep.sh \
#     >> "$SEP_SCRATCH/transfer_v4/alpha_sweep.log" 2>&1 &
#
#   # narrower / different grid:
#   ALPHAS="1e3 1e4 1e5" bash slurm/run_transfer_v4_alpha_sweep.sh
#
#   # tables only, once the alphas are already on disk:
#   SUMMARY_ONLY=1 bash slurm/run_transfer_v4_alpha_sweep.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

[ -f "$REPO_ROOT/.env" ] && source "$REPO_ROOT/.env"

# The sweep runs for hours off the working tree; a checkout mid-run would swap the
# code under it and delete files untracked on the other branch.  Re-checked before
# every alpha, as in the v2 sweep.
EXPECTED_BRANCH=${EXPECTED_BRANCH:-clean/v4-pipeline}
check_branch() {
    local branch
    branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
    if [ "$branch" != "$EXPECTED_BRANCH" ]; then
        echo "ERROR: on branch '$branch', expected '$EXPECTED_BRANCH'. Aborting." >&2
        exit 1
    fi
}
check_branch

export WANDB_MODE=offline
export PYTHONPATH="$REPO_ROOT/src"
PYTHON=${PYTHON:-${SEP_PYTHON:-python}}

OUT_ROOT=${OUT_ROOT:-${SEP_SCRATCH:?set SEP_SCRATCH in .env}/transfer_v4}
MODE=${MODE:-best-to-best}
FORCE_MODE=${FORCE_MODE:-0}
SUMMARY_ONLY=${SUMMARY_ONLY:-0}
# 1e4 first: it is already on disk, so the whole pipeline is exercised in seconds
# and a plumbing error surfaces before any real compute. Then outward from it.
ALPHAS=${ALPHAS:-"1e4 1e5 1e3 1e2 1e1"}

case "$MODE" in
    best-to-best) TAG=b2b ;;
    best-to-last) TAG=btl ;;
    best-to-best-sub) TAG=bbs ;;
    best-to-align) TAG=b2a ;;
    *) echo "unknown MODE '$MODE'" >&2; exit 2 ;;
esac
if [ "$MODE" != best-to-best ] && [ "$FORCE_MODE" = 0 ]; then
    echo "ERROR: MODE=$MODE cannot be swept as-is -- sweep_summary's filename regex" >&2
    echo "       ignores the _$TAG infix, and best-to-align additionally caches its" >&2
    echo "       alpha-dependent layer choice without alpha in the path.  See the" >&2
    echo "       header.  Set FORCE_MODE=1 only after fixing those." >&2
    exit 2
fi

OUT=${OUT:-"$OUT_ROOT/transfer_v4_$TAG"}
BASE_ARGS="--out-dir $OUT --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 \
           --seed 0 --transfer-mode $MODE"
# NQ, same-align only: _resolve_pairs() adds the (nq, nq) combo implicitly for every
# --datasets entry, and with no --cross-align-dataset that is the only one.
PAIR_ARGS="--datasets nq"

if [ ! -d "$OUT" ]; then
    echo "ERROR: $OUT does not exist -- run slurm/run_transfer_v4_all.sh first." >&2
    echo "       This sweep extends an existing v4 tree, it does not build one." >&2
    exit 1
fi
if [ ! -e "$OUT/probes/nq" ]; then
    echo "ERROR: no probe caches at $OUT/probes/nq" >&2
    exit 1
fi

echo "##########################################"
echo "v4 ridge alpha sweep -- NQ, same-align only"
echo "mode:       $MODE  ($TAG)"
echo "out:        $OUT"
echo "alphas:     $ALPHAS"
echo "convention: $($PYTHON -c 'import sep.transfer.transfer2 as t; print(t.NORM_CONVENTION)')"
echo "start:      $(date '+%F %H:%M:%S')"
echo "##########################################"

if [ "$SUMMARY_ONLY" = 0 ]; then
    echo ""
    echo "=== Phase 1: probe cache (alpha-independent)  $(date '+%H:%M:%S')"
    # Idempotent and normally a pure no-op: probes/ is the shared symlink the v4
    # driver created, so this prints "skip nq/<model> (exists)" for all 8 models.
    # No --alpha: phase 1 never sees one.
    $PYTHON -m sep.transfer.transfer2 probe_cache $BASE_ARGS $PAIR_ARGS

    for ALPHA in $ALPHAS; do
        check_branch
        COMMON="$BASE_ARGS --alpha $ALPHA"

        echo ""
        echo "=========================================="
        echo "alpha=$ALPHA  $(date '+%F %H:%M:%S')"
        echo "=========================================="

        # alignments/nq/<pair>/align_nq_ridge_a1e<exp>.pkl -- skipped if present.
        echo "--- Phase 2: align_cache  alpha=$ALPHA  $(date '+%H:%M:%S')"
        $PYTHON -m sep.transfer.transfer2 align_cache \
            $COMMON $PAIR_ARGS --aligners ridge

        # --aligners ridge is load-bearing in phase 3 too: without it the phase also
        # scans for procrustes/e2_rstar caches and appends their tags to the output
        # filenames, which would then miss sweep_summary's regex.
        echo "--- Phase 3: evaluate  alpha=$ALPHA  $(date '+%H:%M:%S')"
        $PYTHON -m sep.transfer.transfer2 evaluate \
            $COMMON $PAIR_ARGS --aligners ridge

        # Per-alpha figures, tagged with the alpha, so they accumulate rather than
        # overwrite.  Carrying each alpha end-to-end means its figures are usable
        # before the next one starts.
        echo "--- Phase 4: summary (this alpha only)  alpha=$ALPHA  $(date '+%H:%M:%S')"
        $PYTHON -m sep.transfer.transfer2 summary $COMMON $PAIR_ARGS

        echo "--- done alpha=$ALPHA  $(date '+%H:%M:%S')"
    done
fi

check_branch
echo ""
echo "=========================================="
echo "Cross-alpha comparison  $(date '+%H:%M:%S')"
echo "=========================================="
# sweep_summary has its own flags (--hparam/--datasets/--pair-list/--n-target), so
# $PAIR_ARGS must NOT be passed here.  --datasets nq keeps it to the (nq, nq) cell.
$PYTHON -m sep.transfer.sweep_summary \
    --out-dir "$OUT" --hparam alpha --datasets nq

echo ""
echo "Sweep complete.  $(date '+%F %H:%M:%S')"
echo "Per-alpha results:  $OUT/results/nq/"
echo "Per-alpha figures:  $OUT/summary_plots/"
echo "Cross-alpha tables: $OUT/sweep_summary/"
