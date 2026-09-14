#!/bin/bash
# Seed the convention-D run tree with the results from transfer_v2 that the
# convention change cannot have touched, so the run only pays for cross-align.
#
# WHAT IS REUSABLE AND WHY
# ------------------------
# probe caches (phase 1)
#     Transfer-mode and align-dataset independent, and phase 1 never normalised
#     anything with the wrong layer -- the bug lived in phases 1.5/2/3.  What the
#     old caches lack is the per-layer mu_all/sd_all that _layer_stats() now
#     needs, and that is recoverable from the hidden states without refitting a
#     single probe: see migrate_probe_stats.py, which asserts
#     mu_all[best_layer] == mu bitwise before writing.  The three v2 trees carry
#     byte-identical splits and statistics, so which one we copy from is
#     irrelevant EXCEPT for bbs: only transfer_v2_bbs has sub_best_layer filled
#     in, so bbs is seeded from its own tree to avoid redoing that 5-fold search.
#
# same-align results (phases 1.5, 2, 3, where align_ds == eval_ds)
#     align_ds == eval_ds makes every normalisation convention pick the SAME
#     statistics: source eval_ds@Ls, target eval_ds@Lt, fit and apply alike.  The
#     layer bug also could not fire (src_align_cache IS src_cache, so its "mu"
#     genuinely is Ls's).  These files are therefore bit-for-bit what convention
#     D would produce, which is why _check_norm_convention() exempts same-align
#     rather than rejecting unstamped files.
#
# native_curves / native_preds
#     Aligner- and align_ds-independent (a property of the target model alone),
#     so one copy per (pair, eval_ds) serves all nine combos.
#
# NOT reusable: everything with align_ds != eval_ds.  Those carry both the layer
# bug and the fit/apply mismatch, and are rejected at load time if left in place.
#
# best-to-best is deliberately NOT seeded.  Its same-align results exist, but
# spread over three unrelated roots (see plot_final_figures.py: transfer_v2_pooled,
# transfer_v2, and a third tree under another account's /build_bak).  Splicing
# three provenances into one tree is what made the v2 results hard to audit; its
# same-align third is 63 cells, cheap enough to just recompute.
#
# HOW THE SKIPPING WORKS
# ----------------------
# Nothing here is special-cased in the pipeline: every phase skips work whose
# output file already exists (phase 3's skip was added for exactly this), so the
# run scripts need no --resume flag.  Seed first, then launch
# run_transfer_v3_all.sh normally.
#
# Usage:
#   bash slurm/seed_transfer_v3.sh              # copy + migrate
#   DRY_RUN=1 bash slurm/seed_transfer_v3.sh    # print what would be copied

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src":/build_bak/UQ/python_packages
PYTHON=${PYTHON:-/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python}

V2=${V2:-/proj/MR_dataset/mtk53728/UQ/sep_scratch}
OUT_ROOT=${OUT_ROOT:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v3}
SHARED="$OUT_ROOT/_probes_shared"
DRY_RUN=${DRY_RUN:-0}
DATASETS="trivia_qa squad nq"

say() { printf '%s\n' "$*"; }
cp_one() {   # cp_one <src> <dst>   -- never overwrites; counts into $N_COPIED
    local s=$1 d=$2
    [ -e "$s" ] || return 0
    [ -e "$d" ] && return 0
    if [ "$DRY_RUN" != "0" ]; then
        say "    would copy ${s#$V2/} -> ${d#$OUT_ROOT/}"
    else
        mkdir -p "$(dirname "$d")"
        # Copy to a sidecar and rename.  The alignment matrices run to 1.1 GB
        # each, so an interrupted copy is likely, and a truncated file at the
        # final name would then look complete to the "never overwrite" check
        # above -- silently poisoning the tree on the next seed.
        cp -p "$s" "$d.part"
        mv -f "$d.part" "$d"
    fi
    N_COPIED=$((N_COPIED + 1))
}

# ---------------------------------------------------------------------------
# 1. shared probe caches (used by b2b / btl / b2a via PROBE_SRC; bbs copies its
#    own below because it needs sub_best_layer)
# ---------------------------------------------------------------------------
say "=== shared probes  <- $V2/transfer_v2_btl/probes"
if [ -L "$SHARED/probes" ]; then
    say "ERROR: $SHARED/probes is a symlink; migration would write through it" >&2
    exit 1
fi
N_COPIED=0
for ds in $DATASETS; do
    for f in "$V2/transfer_v2_btl/probes/$ds"/*.pkl; do
        [ -e "$f" ] || continue
        cp_one "$f" "$SHARED/probes/$ds/$(basename "$f")"
    done
done
say "    $N_COPIED probe caches copied"

# ---------------------------------------------------------------------------
# 2. per-mode: probes (bbs only), same-align alignments / tgt_layers / results
# ---------------------------------------------------------------------------
for spec in "btl btl" "bbs bbs" "b2a b2a"; do
    set -- $spec; tag=$1; v2tag=$2
    SRC="$V2/transfer_v2_$v2tag"
    DST="$OUT_ROOT/transfer_v3_$tag"
    say ""
    say "=== $tag  <- ${SRC#$V2/}"
    [ -d "$SRC" ] || { say "    ERROR: $SRC missing" >&2; exit 1; }
    N_COPIED=0

    # bbs owns real probe files: phase 1 writes sub_best_layer back into them,
    # and its v2 tree already has that field computed.
    if [ "$tag" = bbs ]; then
        for ds in $DATASETS; do
            for f in "$SRC/probes/$ds"/*.pkl; do
                [ -e "$f" ] || continue
                cp_one "$f" "$DST/probes/$ds/$(basename "$f")"
            done
        done
        say "    probes: $N_COPIED (with sub_best_layer)"
        N_COPIED=0
    fi

    for ds in $DATASETS; do
        # -- same-align alignment matrices: align_<ds>_*_<tag>.pkl under eval_ds=<ds>
        for d in "$SRC/alignments/$ds"/*/; do
            [ -d "$d" ] || continue
            pair=$(basename "$d")
            for f in "$d"align_"$ds"_*.pkl; do
                cp_one "$f" "$DST/alignments/$ds/$pair/$(basename "$f")"
            done
        done
        # -- b2a's same-align target-layer selections
        for d in "$SRC/tgt_layers/$ds"/*/; do
            [ -d "$d" ] || continue
            pair=$(basename "$d")
            cp_one "${d}${ds}_b2a.json" "$DST/tgt_layers/$ds/$pair/${ds}_b2a.json"
        done
        # -- results: native (align_ds-independent) + same-align grids/predictions
        for d in "$SRC/results/$ds"/*/; do
            [ -d "$d" ] || continue
            pair=$(basename "$d")
            for f in "$d"native_curves_*"$ds".json "$d"native_preds_*"$ds".json \
                     "$d"probe_grid_*_align_"$ds"_*.json \
                     "$d"predictions_*_align_"$ds"_*.json; do
                cp_one "$f" "$DST/results/$ds/$pair/$(basename "$f")"
            done
        done
    done
    say "    same-align + native files: $N_COPIED"
done

# ---------------------------------------------------------------------------
# 3. migrate every copied probe cache to carry mu_all/sd_all
# ---------------------------------------------------------------------------
say ""
say "=== migrate probe caches (adds mu_all/sd_all; refits nothing)"
MIG_FLAGS=""
[ "$DRY_RUN" != "0" ] && MIG_FLAGS="--dry-run"
for d in "$SHARED/probes" "$OUT_ROOT/transfer_v3_bbs/probes"; do
    [ -d "$d" ] || continue
    say "--- $d"
    $PYTHON -m sep.transfer.migrate_probe_stats --probes-dir "$d" $MIG_FLAGS
done

say ""
say "=== seeded.  Next:"
say "    bash slurm/run_transfer_v3_all.sh          # skips everything seeded above"
say ""
say "Recommended one-cell check before the long run -- recompute a SEEDED"
say "same-align cell with --force and diff it against the copy.  If convention D"
say "really is a no-op for same-align, the json is identical:"
say "    cp \$OUT/results/squad/<pair>/probe_grid_btl_align_squad_ridge_a1e4.json /tmp/before.json"
say "    ... evaluate --force --datasets squad ...   # then diff /tmp/before.json against it"
