#!/bin/bash
# Merge the still-valid parts of transfer_v2 into transfer_v2_pooled, so pooled
# becomes one clean, self-contained directory.
#
# Background — why only part of transfer_v2 is valid.  Commit 707f7ae fixed
# align_cache to fit on `pool` (the complement of Phase 3's eval_idx) instead of
# the first n rows in file order.  The fix is conditional:
#
#     if align_ds == eval_ds:  align_all = src_cache["pool"]     # changed
#     else:                    align_all = np.arange(N_rows)     # unchanged
#
# so ONLY same-align was ever affected.  Verified on disk: same-align N_align
# went 2000 -> 1500, cross-align stayed 2000 in both directories.  The probe
# caches are byte-identical between the two runs (gen_path, best_layer, eval_idx,
# pool, probe coefficients all equal), so old cross-align numbers sit on exactly
# the same split as pooled's and are directly comparable.
#
# Therefore:
#   COPY  cross-align (align_ds != eval_ds) alignments + results + figures
#   SKIP  same-align  (align_ds == eval_ds) — leaky, pooled already has reruns
#   COPY  all timing artefacts (the paper's numbers come from transfer_v2's
#         timing, not pooled's rerun timing), keeping the originals in place
#
# Nothing in transfer_v2 is deleted unless --purge-moved is given, and even then
# only the 252 cross-align pkl that were successfully copied.
#
# Usage:
#   bash slurm/merge_transfer_v2_into_pooled.sh --dry-run
#   bash slurm/merge_transfer_v2_into_pooled.sh
#   bash slurm/merge_transfer_v2_into_pooled.sh --purge-moved   # + reclaim 244 GB

set -euo pipefail

SRC=/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2
DST=/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled

DRY=0
PURGE=0
for a in "$@"; do
    case "$a" in
        --dry-run)     DRY=1 ;;
        --purge-moved) PURGE=1 ;;
        *) echo "unknown flag: $a" >&2; exit 2 ;;
    esac
done
[ "$DRY" = 1 ] && echo "*** DRY RUN — nothing will be modified ***"

run() {
    if [ "$DRY" = 1 ]; then echo "  [dry] $*"; else echo "  \$ $*"; "$@"; fi
}

# Cross-align combinations: eval_ds -> align_ds that are NOT eval_ds.
# Everything this script copies is keyed off these; same-align never matches.
CROSS_NQ=(squad trivia_qa)
CROSS_SQUAD=(nq trivia_qa)

echo "=== checking for running jobs"
busy=0
for d in "$SRC" "$DST"; do
    # The trailing "( |$)" matters: without it, "--out-dir .../transfer_v2" also
    # matches the unrelated .../transfer_v2_btl job, since that path is a prefix
    # extension.  Anchor on the argument boundary instead.
    if pgrep -f "sep\.transfer\.transfer2 .*--out-dir $d( |\$)" > /dev/null; then
        echo "  BUSY: $d"
        pgrep -af "sep\.transfer\.transfer2 .*--out-dir $d( |\$)" \
            | sed -E 's/(--out-dir [^ ]+).*/\1/;s/^/        /'
        busy=1
    else
        echo "  idle: $d"
    fi
done
if [ "$busy" = 1 ]; then
    echo "ERROR: a transfer2 job is running against one of these — refusing to touch it." >&2
    [ "$DRY" = 1 ] || exit 1
fi

# ---------------------------------------------------------------------------
# 1. alignments — cross-align pkl only.  252 files, ~244 GiB.
#    --ignore-existing so a re-run never rewrites 244 GiB, and so a pkl pooled
#    produced itself always wins over transfer_v2's.
# ---------------------------------------------------------------------------
echo "=== 1. alignments (cross-align only, ~244 GiB)"
for eval_ds in nq squad; do
    eval "cross=(\"\${CROSS_${eval_ds^^}[@]}\")"
    for align_ds in "${cross[@]}"; do
        pat="align_${align_ds}_*.pkl"
        n=$(find "$SRC/alignments/$eval_ds" -name "$pat" 2>/dev/null | wc -l)
        [ "$n" -eq 0 ] && continue
        echo "  $eval_ds <- align=$align_ds : $n pkl"
        run rsync -a --ignore-existing \
            --include='*/' --include="$pat" --exclude='*' \
            "$SRC/alignments/$eval_ds/" "$DST/alignments/$eval_ds/"
    done
done

# ---------------------------------------------------------------------------
# 2. results — cross-align probe_grid / predictions / venn JSON.
#    native_curves_* / native_preds_* are NOT copied: they describe the target's
#    own probe, are unaffected by the alignment fix, and pooled already has them.
# ---------------------------------------------------------------------------
echo "=== 2. results (cross-align JSON only)"
for eval_ds in nq squad; do
    eval "cross=(\"\${CROSS_${eval_ds^^}[@]}\")"
    for align_ds in "${cross[@]}"; do
        inc=()
        for kind in probe_grid predictions venn; do
            inc+=(--include="${kind}_align_${align_ds}_*.json")
        done
        n=$(find "$SRC/results/$eval_ds" -name "probe_grid_align_${align_ds}_*.json" 2>/dev/null | wc -l)
        echo "  $eval_ds <- align=$align_ds : ${n} probe_grid (+ predictions, venn)"
        run rsync -a --ignore-existing \
            --include='*/' "${inc[@]}" --exclude='*' \
            "$SRC/results/$eval_ds/" "$DST/results/$eval_ds/"
    done
done

# ---------------------------------------------------------------------------
# 3. Figures — only the ones a filename proves are pure cross-align.
#      venn_<eval>_align_<align>_n<N>.png   per (eval, align, n)  -> clean
#      alpha_curves_<eval>_align_<align>.*  per (eval, align)     -> clean
#    Everything else aggregates same-align and cross-align in one artefact
#    (panel plots draw same-align solid + cross-align dashed; the venn table_*.csv
#    and hparam_* tables sum over align datasets), so it cannot be salvaged —
#    rerun phase_summary / sweep_summary in pooled to regenerate it cleanly.
# ---------------------------------------------------------------------------
echo "=== 3. figures (pure cross-align only)"
for tagdir in "$SRC"/summary_plots/venn_*; do
    [ -d "$tagdir" ] || continue
    tag=$(basename "$tagdir")
    # find, not ls: an unmatched glob makes ls exit 2, and under `set -o pipefail`
    # that aborts the whole script from inside the command substitution.
    n=$(find "$tagdir" -maxdepth 1 \( -name 'venn_nq_align_squad_n*.png' \
        -o -name 'venn_squad_align_nq_n*.png' \
        -o -name 'venn_*_align_trivia_qa_n*.png' \) | wc -l)
    [ "$n" -eq 0 ] && continue
    echo "  summary_plots/$tag: $n clean venn png"
    run mkdir -p "$DST/summary_plots/$tag"
    run rsync -a --ignore-existing \
        --include='venn_nq_align_squad_n*.png' \
        --include='venn_squad_align_nq_n*.png' \
        --include='venn_nq_align_trivia_qa_n*.png' \
        --include='venn_squad_align_trivia_qa_n*.png' \
        --exclude='*' "$tagdir/" "$DST/summary_plots/$tag/"
done
echo "  sweep_summary: cross-align hyperparam curves"
run mkdir -p "$DST/sweep_summary"
run rsync -a --ignore-existing \
    --include='alpha_curves_nq_align_squad.*' \
    --include='alpha_curves_squad_align_nq.*' \
    --include='alpha_curves_nq_align_trivia_qa.*' \
    --include='alpha_curves_squad_align_trivia_qa.*' \
    --include='lambda_curves_nq_align_squad.*' \
    --include='lambda_curves_squad_align_nq.*' \
    --exclude='*' "$SRC/sweep_summary/" "$DST/sweep_summary/"

# ---------------------------------------------------------------------------
# 4. timing — COPY, originals stay in transfer_v2.
#    The two timing.json disagree on 1466 shared leaves (pooled re-timed the same
#    ridge/e2_rstar fits), so they cannot be merged; the paper uses transfer_v2's.
#    transfer_v2's becomes timing.json here and pooled's is kept alongside as
#    timing_pooled_rerun.json.
# ---------------------------------------------------------------------------
echo "=== 4. timing (copy; transfer_v2 keeps its originals)"
if [ -f "$DST/timing.json" ] && [ ! -f "$DST/timing_pooled_rerun.json" ]; then
    echo "  setting aside pooled's own timing.json -> timing_pooled_rerun.json"
    run mv "$DST/timing.json" "$DST/timing_pooled_rerun.json"
fi
run rsync -a "$SRC/timing.json" "$DST/timing.json"
# Everything else timing-related lives in transfer_v2/timing/ (see README_timing.md).
[ -d "$SRC/timing" ] && run rsync -a --ignore-existing "$SRC/timing/" "$DST/timing/"
for d in timing_backup_20260818 timing_backup_20260820; do
    [ -d "$SRC/$d" ] && run rsync -a --ignore-existing "$SRC/$d" "$DST/"
done

# ---------------------------------------------------------------------------
# 5. Logs — suffixed so they cannot collide with pooled's own logs.
# ---------------------------------------------------------------------------
echo "=== 5. logs"
for f in run_alpha_sweep.log run_lambda_sweep.log run_cross_align_triviaqa.log; do
    [ -f "$SRC/$f" ] && run rsync -a --ignore-existing \
        "$SRC/$f" "$DST/${f%.log}_transfer_v2.log"
done

# ---------------------------------------------------------------------------
# 6. Optional purge — only the cross-align pkl we just copied, and only after
#    verifying each destination file exists with a matching size.
# ---------------------------------------------------------------------------
if [ "$PURGE" = 1 ]; then
    echo "=== 6. purging copied cross-align pkl from $SRC"
    freed=0; kept=0
    for eval_ds in nq squad; do
        eval "cross=(\"\${CROSS_${eval_ds^^}[@]}\")"
        for align_ds in "${cross[@]}"; do
            while IFS= read -r s; do
                d="$DST/${s#$SRC/}"
                if [ -f "$d" ] && [ "$(stat -c %s "$s")" = "$(stat -c %s "$d")" ]; then
                    freed=$((freed + $(stat -c %s "$s")))
                    if [ "$DRY" = 1 ]; then echo "  [dry] rm $s"; else rm -f "$s"; fi
                else
                    echo "  KEEP (not verified at destination): $s"
                    kept=$((kept+1))
                fi
            done < <(find "$SRC/alignments/$eval_ds" -name "align_${align_ds}_*.pkl")
        done
    done
    echo "  freed $((freed/1024/1024/1024)) GiB, kept $kept unverified file(s)"
else
    echo "=== 6. purge skipped (pass --purge-moved to reclaim ~244 GiB in $SRC)"
fi

# ---------------------------------------------------------------------------
# 7. Verify.
# ---------------------------------------------------------------------------
echo "=== 7. final state of $DST"
if [ "$DRY" = 1 ]; then echo "  [dry] skipped"; exit 0; fi
for eval_ds in nq squad; do
    echo "  alignments/$eval_ds:"
    find "$DST/alignments/$eval_ds" -name '*.pkl' \
        | sed -E "s|.*/([^/]+)$|\1|" | sort | uniq -c | sed 's/^/    /'
done
echo "  results: $(find "$DST/results" -name '*.json' | wc -l) json"
echo "  total: $(du -sh "$DST" | cut -f1)"
echo ""
echo "Regenerate the aggregate figures that could not be salvaged:"
echo "  python -m sep.transfer.transfer2 summary --out-dir $DST \\"
echo "      --datasets nq squad --cross-align-dataset nq:squad squad:nq --alpha 1e4 ..."
echo "  python -m sep.transfer.sweep_summary --out-dir $DST --hparam alpha ..."
echo "NOTE: squad same-align only exists at a1e3/a1e4 in pooled, so a regenerated"
echo "      squad alpha sweep will have just those two points until a1e1/a1e2/a1e5"
echo "      are refitted on pool."
