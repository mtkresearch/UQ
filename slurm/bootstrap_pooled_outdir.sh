#!/bin/bash
# Seed the leak-free output dir ($OUT_NEW) with the probe caches from the old one.
#
# Background: the alignment fix (transfer2.py, `align_all = src_cache["pool"]` for
# same-align) changes every same-align result, so the reruns go to a NEW out dir and the
# old leaky results stay intact under transfer_v2/ for comparison.
#
# Why COPY the probe caches instead of letting Phase 1 refit them:
#   Phase 1 draws each (dataset, model)'s eval/pool split from ONE rng that is advanced
#   once per entry, in `--datasets` order (transfer2.py:313, :336). So the split a model
#   gets depends on how many datasets -- and in what order -- that invocation was given.
#   Refitting with `--datasets nq` alone would hand every model a DIFFERENT eval_idx than
#   the original `--datasets squad nq` run, which would (a) confound old-vs-new ridge
#   comparisons with a split change and (b) make the nq-only sweeps inconsistent with the
#   squad+nq ridge run sharing this dir (whichever ran Phase 1 first would win).
#   Probe caches are provably unaffected by the alignment fix -- Phase 1 never touches
#   align_all -- so copying is exact, not an approximation.
#
# Idempotent: skips any probe file already present. ~5 MB total.
set -euo pipefail

OUT_OLD=${OUT_OLD:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2}
OUT_NEW=${OUT_NEW:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled}

if [ ! -d "$OUT_OLD/probes" ]; then
    echo "ERROR: no probe caches at $OUT_OLD/probes" >&2
    exit 1
fi

mkdir -p "$OUT_NEW"
# -n: never overwrite an existing probe cache in the destination
cp -rn "$OUT_OLD/probes" "$OUT_NEW/"

echo "[bootstrap] probe caches in $OUT_NEW/probes:"
for ds in "$OUT_NEW"/probes/*/; do
    echo "  $(basename "$ds"): $(ls "$ds" | wc -l) models"
done
