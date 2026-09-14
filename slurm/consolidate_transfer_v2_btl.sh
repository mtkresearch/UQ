#!/bin/bash
# Consolidate every BTL artefact into ONE real directory, no symlinks.
#
#   /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_btl/
#       probes/{nq,squad,trivia_qa}/<model>.pkl    <- COPIED  (source stays put)
#       alignments/{nq,squad,trivia_qa}/<pair>/*_btl.pkl
#       results/{nq,squad,trivia_qa}/<pair>/*btl*.json
#       timing.json                                <- merged, not overwritten
#
# The eval=nq / eval=squad half already lives here (run_transfer_v2_cross_align_
# triviaqa_btl.sh writes straight into this directory).  The eval=trivia_qa half
# sits on the other storage root and is MOVED here.  Probe caches are COPIED,
# because the best-to-best runs still need them.
#
#   COPY : transfer_v2/probes/{nq,squad,trivia_qa}          ~5 MB
#   MOVE : transfer_v2_trivia_qa_btl/alignments/trivia_qa    ~62 GB  (63 pkl)
#   MOVE : transfer_v2_trivia_qa_btl/results/trivia_qa       ~5 MB
#   MOVE : transfer_v2_trivia_qa_btl/summary_plots/*         trivia_qa pdf/png + venn
#   MOVE : transfer_v2_trivia_qa_btl/run.log -> run_trivia_qa_btl.log
#   MERGE: transfer_v2_trivia_qa_btl/timing.json into $OUT/timing.json
#
# NOTE: "MOVE" degrades to a plain copy whenever the source is owned by another
# account (see the DEL_SRC probe below) — that is the normal case here, so
# /build_bak keeps its 62 GB until the owner removes it.
#
# The 62 GB move crosses filesystems (/build_bak -> /proj/MR_dataset), so it is
# a copy+delete either way.  rsync --remove-source-files is used instead of mv:
# it is resumable and only unlinks a source file after that file transferred.
#
# Usage:
#   bash slurm/consolidate_transfer_v2_btl.sh --dry-run   # show the plan only
#   bash slurm/consolidate_transfer_v2_btl.sh             # do it

set -euo pipefail

OUT=/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_btl
PROBE_SRC=/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2/probes
TQ_BTL=/build_bak/UQ/UQ-transfer/sep_scratch/transfer_v2_trivia_qa_btl

# PROBE_SRC/nq and PROBE_SRC/squad are real directories, but PROBE_SRC/trivia_qa
# is itself a symlink into /build_bak — the trivia_qa probes only ever existed
# there.  Step 1b materialises that one too, so neither this directory nor
# transfer_v2/probes depends on the other storage root any more.
TQ_PROBE_REAL=/build_bak/UQ/UQ-transfer/sep_scratch/transfer_v2_trivia_qa/probes/trivia_qa

DATASETS=(nq squad trivia_qa)

# find, not `ls dir/*.pkl`: an unmatched glob makes ls exit 2, and under
# `set -o pipefail` that aborts the script from inside a command substitution.
count_pkl() { find "$1" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l; }

DRY=0
case "${1:-}" in
    --dry-run) DRY=1 ;;
    "")        DRY=0 ;;
    # Anything else is a typo. Silently treating e.g. `--dryrun` as a real run
    # would start a 62 GB copy the caller did not ask for.
    *) echo "unknown argument: $1  (expected --dry-run or nothing)" >&2; exit 2 ;;
esac
[ "$DRY" = 1 ] && echo "*** DRY RUN — nothing will be modified ***"

# ---------------------------------------------------------------------------
# Can we delete from the trivia_qa source?  It is owned by another user
# (gpu_mtk53658, mode drwxrwxr-x), so a different account can read and copy but
# not unlink.  In that case --remove-source-files would transfer all 62 GB and
# then fail on every unlink, so degrade the "move" to a plain copy and print
# the cleanup command for the owner to run.
# ---------------------------------------------------------------------------
DEL_SRC=0
if [ -d "$TQ_BTL/alignments" ] && touch "$TQ_BTL/alignments/.wtest" 2>/dev/null; then
    rm -f "$TQ_BTL/alignments/.wtest"
    DEL_SRC=1
fi

# -a implies -p -g -o.  When the transfer root is a directory that already exists
# and is owned by ANOTHER account, rsync tries to chmod/chown it, gets EPERM and
# exits 23 — even though every file transferred fine.  Under `set -e` that aborts
# the script mid-way.  Drop those three: we only care about the bytes, and new
# files get sensible modes from the umask.  -rlt keeps recursion, symlinks and
# mtimes, which is what makes re-runs skip already-copied files.
RSYNC_BASE=(rsync -rlt --no-perms --no-group --no-owner)

if [ "$DEL_SRC" = 1 ]; then
    RSYNC_MOVE=("${RSYNC_BASE[@]}" --remove-source-files)
    echo "source is writable: artefacts will be MOVED (source files removed)"
else
    RSYNC_MOVE=("${RSYNC_BASE[@]}")
    echo "source is NOT writable by $(whoami): artefacts will be COPIED."
    echo "  Owner ($(stat -c %U "$TQ_BTL" 2>/dev/null || echo '?')) can reclaim the space afterwards with:"
    echo "      rm -rf $TQ_BTL"
fi

# run <cmd...> : execute, or just print under --dry-run.
run() {
    if [ "$DRY" = 1 ]; then
        echo "  [dry] $*"
    else
        echo "  \$ $*"
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# 0. Guard — never touch a directory a transfer2 job is still using.
#    Phase 3/4 of the trivia_qa run reads alignments/ that we are about to move.
# ---------------------------------------------------------------------------
echo "=== checking for running jobs"
busy=0
for d in "$OUT" "$TQ_BTL"; do
    if pgrep -f "sep\.transfer\.transfer2 .*--out-dir $d" > /dev/null; then
        echo "  BUSY: $d"
        pgrep -af "sep\.transfer\.transfer2 .*--out-dir $d" | sed 's/^/        /'
        busy=1
    else
        echo "  idle: $d"
    fi
done
if [ "$busy" = 1 ]; then
    echo ""
    echo "ERROR: at least one transfer2 job is still running." >&2
    echo "       Moving alignments/ out from under it would break Phase 3/4." >&2
    echo "       Wait for both jobs to finish, then re-run." >&2
    [ "$DRY" = 1 ] && echo "       (dry run: continuing to show the plan anyway)" || exit 1
fi

# ---------------------------------------------------------------------------
# 1. probes/  — COPY.  Must delete the symlink first: copying into a symlinked
#    directory writes through to the best-to-best run's cache.
# ---------------------------------------------------------------------------
echo "=== 1. probes/ (copy, source untouched)"
# Already-materialised case: skip entirely.  Re-running the cp would be a no-op
# (-n never clobbers) but it would pull from the OTHER run's tree, re-coupling
# two directories that are now deliberately independent physical copies.
have_probes=$(find "$OUT/probes" -name '*.pkl' 2>/dev/null | wc -l)
want_probes=$(find "$PROBE_SRC" -name '*.pkl' 2>/dev/null | wc -l)
if [ ! -L "$OUT/probes" ] && [ -d "$OUT/probes" ] && [ "$have_probes" -ge "$want_probes" ]; then
    echo "  already a real, complete directory ($have_probes pkl >= $want_probes) — nothing to do"
else
    if [ -L "$OUT/probes" ]; then
        echo "  removing symlink -> $(readlink "$OUT/probes")  (link only, target data kept)"
        run rm -f "$OUT/probes"
    fi
    run mkdir -p "$OUT/probes"
    for ds in "${DATASETS[@]}"; do
        run mkdir -p "$OUT/probes/$ds"
        # -L dereferences: PROBE_SRC/trivia_qa may be a symlink, we want real files.
        # -n so a re-run never clobbers probes already in place.
        run cp -rnL "$PROBE_SRC/$ds/." "$OUT/probes/$ds/"
    done
fi

# ---------------------------------------------------------------------------
# 1b. Also de-symlink transfer_v2/probes/trivia_qa itself.  It points into
#     /build_bak, so transfer_v2/ silently depends on the other storage root.
#     Replacing it with real files is content-neutral (same bytes) and lets
#     /build_bak be reclaimed later without breaking the best-to-best run.
#     Staged copy + swap keeps the window where the path is absent near zero.
# ---------------------------------------------------------------------------
echo "=== 1b. de-symlink $PROBE_SRC/trivia_qa"
if [ -L "$PROBE_SRC/trivia_qa" ]; then
    # Follow the link that is actually there.  TQ_PROBE_REAL is only a sanity
    # reference: hardcoding the copy source would silently substitute different
    # data if the link ever pointed somewhere else.
    real=$(readlink -f "$PROBE_SRC/trivia_qa")
    echo "  currently -> $real"
    [ "$real" = "$TQ_PROBE_REAL" ] || echo "  NOTE: differs from expected $TQ_PROBE_REAL"
    if [ "$DRY" = 1 ]; then
        echo "  [dry] cp -rL $real $PROBE_SRC/trivia_qa.staged"
        echo "  [dry] rm -f $PROBE_SRC/trivia_qa && mv $PROBE_SRC/trivia_qa.staged $PROBE_SRC/trivia_qa"
    else
        rm -rf "$PROBE_SRC/trivia_qa.staged"
        cp -rL "$real" "$PROBE_SRC/trivia_qa.staged"
        n=$(count_pkl "$PROBE_SRC/trivia_qa.staged")
        echo "  staged $n pkl"
        # Without this guard an empty staged dir (stale mount, moved target)
        # would replace the only working path to the trivia_qa probes.
        if [ "$n" -eq 0 ]; then
            rm -rf "$PROBE_SRC/trivia_qa.staged"
            echo "ERROR: staged copy is empty, leaving the symlink alone." >&2
            exit 1
        fi
        rm -f "$PROBE_SRC/trivia_qa"      # the link only
        mv "$PROBE_SRC/trivia_qa.staged" "$PROBE_SRC/trivia_qa"
        echo "  now a real directory with $(count_pkl "$PROBE_SRC/trivia_qa") pkl"
    fi
elif [ -d "$PROBE_SRC/trivia_qa" ]; then
    echo "  already a real directory, nothing to do"
else
    echo "  not present, skipping"
fi

# ---------------------------------------------------------------------------
# 2. alignments/trivia_qa  — MOVE (62 GB, 10-30 min over NFS).
# ---------------------------------------------------------------------------
echo "=== 2. alignments/trivia_qa (move)"
SRC_ALIGN_N=0
if [ -d "$TQ_BTL/alignments/trivia_qa" ]; then
    sz=$(du -sh "$TQ_BTL/alignments/trivia_qa" | cut -f1)
    n=$(find "$TQ_BTL/alignments/trivia_qa" -name '*.pkl' | wc -l)
    SRC_ALIGN_N=$n
    echo "  source: $n pkl, $sz"
    run mkdir -p "$OUT/alignments"
    run "${RSYNC_MOVE[@]}" --info=progress2 \
        "$TQ_BTL/alignments/trivia_qa" "$OUT/alignments/"
    # --remove-source-files leaves the (now empty) directory tree behind.
    if [ "$DRY" = 0 ] && [ "$DEL_SRC" = 1 ]; then
        find "$TQ_BTL/alignments/trivia_qa" -type d -empty -delete || true
    fi
else
    echo "  not present, skipping"
fi

# ---------------------------------------------------------------------------
# 3. results/trivia_qa  — MOVE.
# ---------------------------------------------------------------------------
echo "=== 3. results/trivia_qa (move)"
SRC_RESULT_N=0
if [ -d "$TQ_BTL/results/trivia_qa" ]; then
    n=$(find "$TQ_BTL/results/trivia_qa" -type f | wc -l)
    SRC_RESULT_N=$n
    echo "  source: $n files, $(du -sh "$TQ_BTL/results/trivia_qa" | cut -f1)"
    run mkdir -p "$OUT/results"
    run "${RSYNC_MOVE[@]}" "$TQ_BTL/results/trivia_qa" "$OUT/results/"
    if [ "$DRY" = 0 ] && [ "$DEL_SRC" = 1 ]; then
        find "$TQ_BTL/results/trivia_qa" -type d -empty -delete || true
    fi
else
    echo "  not present, skipping"
fi

# ---------------------------------------------------------------------------
# 3b. summary_plots/  — MOVE.  Phase 4 regenerates these, but they are the only
#     copy until it is re-run, and they would otherwise be stranded on the
#     storage root that gets reclaimed.  Filenames carry the eval dataset
#     (summary_trivia_qa_* vs summary_{nq,squad}_*) and the venn subdirectory is
#     keyed per pair+align_ds, so the two runs' files do not collide.  No
#     trailing slash on the source and a "/" destination would nest it, so copy
#     the CONTENTS with "/." into the existing directory.
# ---------------------------------------------------------------------------
echo "=== 3b. summary_plots (move, merged into existing)"
SRC_PLOT_N=0
SRC_PLOT_LIST=""
if [ -d "$TQ_BTL/summary_plots" ]; then
    n=$(find "$TQ_BTL/summary_plots" -type f | wc -l)
    SRC_PLOT_N=$n
    echo "  source: $n files"
    have=$(find "$OUT/summary_plots" -type f 2>/dev/null | wc -l)
    echo "  destination already has: $have files"
    # Remember the source listing NOW: this step merges into a non-empty
    # directory, so a plain file count cannot tell "copied" from "already there",
    # and --remove-source-files may empty the source before step 6b runs.
    # Counting have+src is wrong — it is not idempotent, a second run would
    # expect have(=already merged)+src and always fail.
    SRC_PLOT_LIST=$( cd "$TQ_BTL/summary_plots" && find . -type f | sort )
    run mkdir -p "$OUT/summary_plots"
    run "${RSYNC_MOVE[@]}" "$TQ_BTL/summary_plots/." "$OUT/summary_plots/"
    if [ "$DRY" = 0 ] && [ "$DEL_SRC" = 1 ]; then
        find "$TQ_BTL/summary_plots" -type d -empty -delete || true
    fi
else
    echo "  not present, skipping"
fi

# ---------------------------------------------------------------------------
# 4. timing.json  — MERGE, never overwrite.  Structure is
#    {pair: {"<eval_ds>_probe": {"<align_ds>_align": {...}}}} so the two runs
#    populate disjoint leaves; a plain cp would drop one run's timings.
# ---------------------------------------------------------------------------
echo "=== 4. timing.json (deep merge)"
if [ -f "$TQ_BTL/timing.json" ]; then
    if [ "$DRY" = 1 ]; then
        echo "  [dry] merge $TQ_BTL/timing.json into $OUT/timing.json"
    else
        python3 - "$TQ_BTL/timing.json" "$OUT/timing.json" <<'PY'
import json, os, shutil, sys
src_p, dst_p = sys.argv[1], sys.argv[2]
src = json.load(open(src_p))
dst = json.load(open(dst_p)) if os.path.exists(dst_p) else {}
conflicts = []

def merge(a, b, path=""):
    """Merge b (source) into a (destination). Destination wins on conflict."""
    for k, v in b.items():
        p = f"{path}/{k}"
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            merge(a[k], v, p)
        elif k in a and a[k] != v:
            conflicts.append(p)          # keep destination value
        else:
            a[k] = v

merge(dst, src)
if os.path.exists(dst_p):
    shutil.copy2(dst_p, dst_p + ".bak")
    print(f"  backup: {dst_p}.bak")
with open(dst_p, "w") as f:
    json.dump(dst, f, indent=2)
print(f"  merged {len(src)} source keys -> {len(dst)} keys in {dst_p}")
if conflicts:
    print(f"  {len(conflicts)} conflicting leaves kept destination value, e.g.:")
    for c in conflicts[:5]:
        print(f"    {c}")
PY
        [ "$DEL_SRC" = 1 ] && run rm -f "$TQ_BTL/timing.json" || true
    fi
else
    echo "  not present, skipping"
fi

# ---------------------------------------------------------------------------
# 5. run.log  — MOVE, renamed to avoid colliding with this directory's own log.
# ---------------------------------------------------------------------------
echo "=== 5. run.log (move + rename)"
if [ -f "$TQ_BTL/run.log" ]; then
    run "${RSYNC_MOVE[@]}" "$TQ_BTL/run.log" "$OUT/run_trivia_qa_btl.log"
else
    echo "  not present, skipping"
fi

# ---------------------------------------------------------------------------
# 6. Verify.
# ---------------------------------------------------------------------------
echo "=== 6. final state of $OUT"
if [ "$DRY" = 1 ]; then
    echo "  [dry] skipped"
    exit 0
fi
for ds in "${DATASETS[@]}"; do
    printf "  probes/%-10s %s pkl\n" "$ds" "$(count_pkl "$OUT/probes/$ds")"
done
for e in "$OUT"/alignments/*/; do
    [ -d "$e" ] || continue
    printf "  alignments/%-10s %s pkl in %s pair dirs\n" "$(basename "$e")" \
        "$(find "$e" -name '*.pkl' | wc -l)" \
        "$(find "$e" -mindepth 1 -maxdepth 1 -type d | wc -l)"
done
for e in "$OUT"/results/*/; do
    [ -d "$e" ] || continue
    printf "  results/%-10s %s files\n" "$(basename "$e")" "$(find "$e" -type f | wc -l)"
done
echo "  total: $(du -sh "$OUT" | cut -f1)"

for d in "$OUT" "$PROBE_SRC"; do
    links=$(find "$d" -type l | wc -l)
    if [ "$links" -eq 0 ]; then
        echo "  symlinks under $d: none — all real files"
    else
        echo "  WARNING: $links symlink(s) remain under $d:"
        find "$d" -type l -printf '    %p -> %l\n'
    fi
done

# --- hard assertions.  Without these a partial rsync (NFS drop, quota) still
# --- exits 0, and the next Phase 2 silently re-fits whatever failed to land.
echo "=== 6b. verifying counts"
fail=0
check() {  # check <label> <expected> <actual>
    if [ "$3" -ge "$2" ] && [ "$2" -gt 0 ]; then
        printf "  OK   %-24s %s >= %s\n" "$1" "$3" "$2"
    elif [ "$2" -eq 0 ]; then
        printf "  skip %-24s (nothing in source)\n" "$1"
    else
        printf "  FAIL %-24s %s < %s\n" "$1" "$3" "$2"
        fail=1
    fi
}
check "alignments/trivia_qa pkl" "$SRC_ALIGN_N"  "$(find "$OUT/alignments/trivia_qa" -name '*.pkl' 2>/dev/null | wc -l)"
check "results/trivia_qa files"  "$SRC_RESULT_N" "$(find "$OUT/results/trivia_qa" -type f 2>/dev/null | wc -l)"
# summary_plots merges into a shared directory, so check presence per file
# instead of a count.  Idempotent: passes on every re-run once the union is there.
if [ -n "$SRC_PLOT_LIST" ]; then
    missing=0
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        [ -e "$OUT/summary_plots/$f" ] || { missing=$((missing + 1)); echo "    missing: $f"; }
    done <<< "$SRC_PLOT_LIST"
    if [ "$missing" -eq 0 ]; then
        printf "  OK   %-24s all %s source files present (dir total: %s)\n" \
            "summary_plots" "$SRC_PLOT_N" "$(find "$OUT/summary_plots" -type f | wc -l)"
    else
        printf "  FAIL %-24s %s of %s source files missing\n" "summary_plots" "$missing" "$SRC_PLOT_N"
        fail=1
    fi
else
    printf "  skip %-24s (nothing in source)\n" "summary_plots"
fi
if [ "$fail" = 1 ]; then
    echo ""
    echo "ERROR: consolidation is INCOMPLETE — do not start a new run yet." >&2
    echo "       Re-run this script; rsync is resumable and will fill the gaps." >&2
    exit 1
fi

echo ""
echo "Leftovers in $TQ_BTL (should be empty dirs / the probes symlink only):"
ls -la "$TQ_BTL" 2>/dev/null || echo "  (gone)"
echo ""
echo "Next: bash slurm/run_transfer_v2_btl_all.sh"
echo "  Phase 1 skips all 22 copied probes; Phase 2 skips the 147 alignments"
echo "  already present and fits only the missing nq:squad / squad:nq (42)."
