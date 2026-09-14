#!/bin/bash
# Re-time "probe training" and "alignment fitting" for all three datasets under ONE
# controlled environment (single conda env, pinned BLAS threads), as a single fit at
# n=1500 rather than a sum over the n-grid.
#
# Reads the existing probe/alignment caches -- nothing is refit for real, and no
# existing artefact is overwritten.
#
# Outputs, all under $TIMING = <scratch>/transfer_v2/timing:
#   timing_fits.json                     T4 (probe_cache) and T5 (align_cache) + _meta
#   timing_table_{squad,nq,trivia_qa}.csv  the final per-dataset tables
#   timing_collect_data_all.json         T1/T2 per dataset+model (table by-product)
#   fit_timing/z/<dataset>__<model>.npz  cached best-layer features (~65 MB each,
#                                        ~1.2 GB total; safe to delete afterwards)
#   timing_table_prev_<timestamp>/       backup of the CSVs being replaced
#   time_fits.log                        full log, if invoked as suggested below
#
# Usage:
#   nohup bash slurm/run_time_fits.sh \
#     > /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2/timing/time_fits.log 2>&1 &
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
TIMING="$OUT/timing"          # every timing artefact lives here
WORK="$TIMING/fit_timing"
mkdir -p "$TIMING"
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
    --out-json "$TIMING/timing_fits.json"

echo ""
echo "=========================================="
echo "Phase 3: timing tables  $(date '+%H:%M:%S')"
echo "=========================================="

# T1/T2 for all datasets live in one flat dir of <dataset>_<model>.json, written by
# slurm/run_collect_timing.sh.
T1T2="$TIMING/data_generation_timing"
[[ -d "$T1T2" ]] || { echo "ERROR: missing T1/T2 dir $T1T2 (run slurm/run_collect_timing.sh)"; exit 1; }
echo "T1/T2 files: $(ls -1 "$T1T2"/*.json | wc -l)  <- $T1T2"

# Keep the CSVs we are about to replace -- they are the numbers currently in the paper.
BACKUP="$TIMING/timing_table_prev_$(date '+%Y%m%d_%H%M%S')"
if compgen -G "$TIMING/timing_table_*.csv" > /dev/null; then
    mkdir -p "$BACKUP"
    cp "$TIMING"/timing_table_*.csv "$BACKUP"/
    echo "Previous CSVs backed up to $BACKUP"
fi

"$PYTHON" -m sep.transfer.make_timing_table \
    --collect-timing-base "$T1T2" \
    --transfer-v2-timing  "$TIMING/timing_fits.json" \
    --pair-list           "$REPO_ROOT/slurm/inputs/pair_list.txt" \
    --out-json            "$TIMING/timing_collect_data_all.json" \
    --out-csv-dir         "$TIMING" \
    --datasets            "${DATASETS[@]}"

echo ""
echo "Done.  $(date '+%H:%M:%S')"
echo "Timings:        $TIMING/timing_fits.json"
for ds in "${DATASETS[@]}"; do
    echo "Table:          $TIMING/timing_table_${ds}.csv"
done
echo "Feature cache:  $WORK/z/   (deletable)"
