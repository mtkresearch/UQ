#!/bin/bash
# Detached entry point for seeding: clean up after any previous kill, then copy.
#
# Meant to survive logout.  setsid detaches it from the terminal's session so it
# does not get SIGHUP, and everything lands in one log.
#
#   bash slurm/seed_transfer_v3_bg.sh
#
# then check on it later with:
#   tail -f  /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v3/seed.log
#   pgrep -af seed_transfer_v3
#
# Killing this at ANY point is safe and resumable: cp_one() copies to *.part and
# renames, so a destination file is either absent or complete, and the verify step
# below sweeps up the *.part leftovers on the next run.  Re-running after a kill
# picks up exactly where it stopped.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT=${OUT_ROOT:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v3}
export OUT_ROOT
LOG="$OUT_ROOT/seed.log"
mkdir -p "$OUT_ROOT"

# Refuse to start a second copier: two cp_one loops racing on the same *.part
# name would interleave writes into one file and then rename it into place.
if pgrep -f 'slurm/seed_transfer_v3\.sh' >/dev/null; then
    echo "ERROR: a seed_transfer_v3.sh is already running:" >&2
    pgrep -af 'slurm/seed_transfer_v3\.sh' >&2
    echo "kill it first, or wait for it to finish." >&2
    exit 1
fi

run() {
    echo "=========================================="
    echo "seed transfer_v3  start $(date '+%F %H:%M:%S')  pid $$"
    echo "out root: $OUT_ROOT"
    echo "=========================================="

    echo ""
    echo ">>> step 0: clean up anything a previous kill left half-written"
    bash slurm/verify_transfer_v3.sh

    echo ""
    echo ">>> step 1: copy + migrate"
    bash slurm/seed_transfer_v3.sh

    echo ""
    echo ">>> step 2: re-verify (every copied file vs its v2 source)"
    bash slurm/verify_transfer_v3.sh

    echo ""
    echo "=========================================="
    echo "seed done $(date '+%F %H:%M:%S')"
    du -sh "$OUT_ROOT"/* 2>/dev/null || true
    echo ""
    echo "Next:  nohup bash slurm/run_transfer_v3_all.sh \\"
    echo "         >> $OUT_ROOT/run_all.log 2>&1 &"
    echo "=========================================="
}

setsid bash -c "$(declare -f run); run" >>"$LOG" 2>&1 < /dev/null &
disown || true

echo "seeding in background, pid $!"
echo "log: $LOG"
echo "watch:  tail -f $LOG"
