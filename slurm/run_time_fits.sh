#!/bin/bash
# Re-time "probe training" and "alignment fitting" for all three datasets under ONE
# controlled environment (single conda env, pinned BLAS threads), as a single fit at
# n=1500 rather than a sum over the n-grid.
#
# Reads the existing probe/alignment caches -- nothing is refit for real, and no
# existing artefact is overwritten.
#
# Outputs:
#   $OUT/timing_fits.json                     the timings, in the same schema as
#                                             transfer_v2/timing.json
#   $OUT/timing_table_{squad,nq,trivia_qa}.csv  the final per-dataset tables
#   $OUT/timing_collect_data_all.json         T1/T2 per dataset+model (table by-product)
#   $OUT/fit_timing/z/<dataset>__<model>.npz  cached best-layer features (~65 MB each,
#                                             ~1.2 GB total; safe to delete afterwards)
#   $OUT/timing_table_prev_<timestamp>/       backup of the CSVs being replaced
#   $OUT/time_fits.log                        full log, if invoked as suggested below
#
# Usage:
#   nohup bash slurm/run_time_fits.sh \
#     > /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2/time_fits.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Pin BLAS threads.  Without this the shared 64-core host's load swings the numbers
# by several-fold, which is exactly what made the original timings incomparable.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Pin the interpreter too, rather than relying on whichever env happens to be active.
# `sep` is editable-installed only in se_probes, and comparability across the three
# datasets is exactly what this script exists to establish -- numpy 1.26 + MKL here.
PYTHON="${PYTHON:-/proj/gpu_mtk53728/miniconda3/envs/se_probes/bin/python}"
if ! "$PYTHON" -c "import sep" 2>/dev/null; then
    echo "ERROR: $PYTHON cannot import sep; set PYTHON=<interpreter with sep installed>"
    exit 1
fi
echo "Interpreter: $PYTHON"
"$PYTHON" -c "import numpy; print('numpy', numpy.__version__)"

OUT=/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2
WORK="$OUT/fit_timing"
DATASETS=(squad nq trivia_qa)

# transfer2 --out-dir per dataset (TriviaQA was run under a different out-dir).
SQUAD_NQ_ROOT=/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2
TRIVIA_ROOT=/build_bak/UQ/UQ-transfer/sep_scratch/transfer_v2_trivia_qa

echo "=========================================="
echo "Phase 1: extract best-layer features  $(date '+%H:%M:%S')"
echo "=========================================="
"$PYTHON" -m sep.transfer.time_fits extract \
    --cache-root "squad=$SQUAD_NQ_ROOT" \
    --cache-root "nq=$SQUAD_NQ_ROOT" \
    --cache-root "trivia_qa=$TRIVIA_ROOT" \
    --work-dir "$WORK" \
    --token slt

echo ""
echo "=========================================="
echo "Phase 2: time the fits  $(date '+%H:%M:%S')"
echo "=========================================="
"$PYTHON" -m sep.transfer.time_fits time \
    --work-dir "$WORK" \
    --pair-list "$REPO_ROOT/slurm/inputs/pair_list.txt" \
    --datasets "${DATASETS[@]}" \
    --n 1500 \
    --alpha 1e4 \
    --repeats 3 \
    --out-json "$OUT/timing_fits.json"

echo ""
echo "=========================================="
echo "Phase 3: timing tables  $(date '+%H:%M:%S')"
echo "=========================================="

# T1/T2 live in two directories (squad+nq vs trivia_qa); make_timing_table reads one
# flat dir of <dataset>_<model>.json, so link them together.
T1T2="$WORK/t1t2"
rm -rf "$T1T2"; mkdir -p "$T1T2"
for src in "$OUT/data_generation_timing" "$OUT/data_generation_timing_trivia_qa"; do
    if [[ -d "$src" ]]; then
        ln -sf "$src"/*.json "$T1T2"/ 2>/dev/null || true
    else
        echo "WARNING: missing T1/T2 dir $src"
    fi
done
echo "T1/T2 files: $(ls -1 "$T1T2" | wc -l)"

# Keep the CSVs we are about to replace -- they are the numbers currently in the paper.
BACKUP="$OUT/timing_table_prev_$(date '+%Y%m%d_%H%M%S')"
if compgen -G "$OUT/timing_table_*.csv" > /dev/null; then
    mkdir -p "$BACKUP"
    cp "$OUT"/timing_table_*.csv "$BACKUP"/
    echo "Previous CSVs backed up to $BACKUP"
fi

"$PYTHON" -m sep.transfer.make_timing_table \
    --collect-timing-base "$T1T2" \
    --transfer-v2-timing  "$OUT/timing_fits.json" \
    --pair-list           "$REPO_ROOT/slurm/inputs/pair_list.txt" \
    --out-json            "$OUT/timing_collect_data_all.json" \
    --out-csv-dir         "$OUT" \
    --datasets            "${DATASETS[@]}"

echo ""
echo "Done.  $(date '+%H:%M:%S')"
echo "Timings:        $OUT/timing_fits.json"
for ds in "${DATASETS[@]}"; do
    echo "Table:          $OUT/timing_table_${ds}.csv"
done
echo "Feature cache:  $WORK/z/   (deletable)"
