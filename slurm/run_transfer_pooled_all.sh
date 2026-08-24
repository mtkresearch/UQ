#!/bin/bash
# One command to run the whole leak-free rerun, in this order:
#
#   3. run_transfer_lambda_sweep.sh  E2-R* lambda sweep, nq only        (first: wanted soonest)
#   2. run_transfer_alpha_sweep.sh   ridge alpha sweep, nq only, 5 alphas incl. 1e3
#   1. run_transfer_v2.sh            ridge alpha=1e3 -- SQUAD ONLY, then a nq+squad Phase 4
#
# Step 1 is squad-only (DATASETS=squad) because step 2's alpha=1e3 pass already produces
# every nq artifact step 1 would: same aligner, same alpha, same phases 2-4. Its Phase 4
# still gets SUMMARY_DATASETS="squad nq" so the combined figures cover both datasets --
# Phase 4 only reads the grid JSONs, so it does not care which step wrote nq's.
#
# SEQUENTIAL on purpose: all three write the same $OUT, and step 1 + step 2 both touch
# align_nq_ridge_a1e3.pkl if step 1 is not narrowed -- concurrent runs would clobber each
# other's caches. Step 1 must also come after step 2 so nq's grid JSON exists for Phase 4.
#
# Running step 3 first is safe for filenames: steps 1-2 do not pass --lambda-val, so they
# look for the DEFAULT tag e2_rstar_l1 (transfer2.py:1344, default 1.0), which the sweep
# (1e3..1e7) never writes. So their grid files stay ridge-only rather than picking up an
# e2_rstar tag, and sweep_summary's two regexes tolerate either spelling anyway.
#
# Each child script has its own branch guard and calls bootstrap_pooled_outdir.sh, so
# Phase 1 (probe_cache) is a no-op everywhere: the probe caches are COPIED from the old
# out dir rather than refit, to keep the eval/pool splits identical (see that script).
#
# Per-step logs go to $OUT/run.log, run_alpha_sweep.log, run_lambda_sweep.log; this
# script's own progress lines go to stdout. A failing step aborts the rest.
#
# Usage:
#   nohup bash slurm/run_transfer_pooled_all.sh \
#     > /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled/run_all.log 2>&1 &
#
#   OUT=/some/other/dir bash slurm/run_transfer_pooled_all.sh   # override out dir
#   STEPS="2 1"        bash slurm/run_transfer_pooled_all.sh   # skip the lambda sweep
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Exported so the child scripts pick it up instead of their own default.
export OUT=${OUT:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled}
STEPS=${STEPS:-"3 2 1"}

mkdir -p "$OUT"

# run_step <n> <script> <log> [VAR=val ...]   -- trailing args are env for the child
run_step() {
    local n="$1" script="$2" log="$3"
    shift 3
    echo ""
    echo "##########################################################"
    echo "# STEP $n/3: $script   start $(date '+%F %H:%M:%S')"
    echo "#   env: ${*:-(none)}"
    echo "#   log: $OUT/$log"
    echo "##########################################################"
    local t0=$SECONDS
    # Child stdout/stderr go to its own log; tee would interleave three noisy phases.
    if env "$@" bash "slurm/$script" > "$OUT/$log" 2>&1; then
        echo "# STEP $n OK    $(( (SECONDS - t0) / 60 )) min   $(date '+%F %H:%M:%S')"
    else
        local rc=$?
        echo "# STEP $n FAILED (exit $rc) after $(( (SECONDS - t0) / 60 )) min" >&2
        echo "#   last 30 lines of $OUT/$log:" >&2
        tail -n 30 "$OUT/$log" >&2 || true
        exit $rc
    fi
}

echo "Leak-free transfer2 rerun"
echo "  OUT   = $OUT"
echo "  STEPS = $STEPS"
echo "  start   $(date '+%F %H:%M:%S')"

for s in $STEPS; do
    case "$s" in
        # squad only -- step 2 covers nq at this same alpha; Phase 4 still plots both
        1) run_step 1 run_transfer_v2.sh           run.log \
               DATASETS=squad SUMMARY_DATASETS="squad nq" ;;
        2) run_step 2 run_transfer_alpha_sweep.sh  run_alpha_sweep.log ;;
        3) run_step 3 run_transfer_lambda_sweep.sh run_lambda_sweep.log ;;
        *) echo "ERROR: unknown step '$s' (expected 1, 2 or 3)" >&2; exit 1 ;;
    esac
done

echo ""
echo "=========================================================="
echo "All steps complete.  $(date '+%F %H:%M:%S')"
echo "  curves:        $OUT/results/"
echo "  per-hp figs:   $OUT/summary_plots/"
echo "  alpha sweep:   $OUT/sweep_summary/hparam_alpha_*"
echo "  lambda sweep:  $OUT/sweep_summary/hparam_lambda_*"
echo "=========================================================="
