#!/bin/bash
# SE probe transfer — best-to-last (BTL) mode, FULL 3x3 dataset grid, ridge alpha=1e4.
#
#   Source probe:  trained at each model's best layer on the eval dataset.
#   Alignment:     best layer of source  ->  last layer of target,
#                  fitted on the alignment dataset's hidden states.
#   Native target: probe trained and evaluated at the target's last layer.
#
# Covers all 9 (eval_ds, align_ds) combinations over {trivia_qa, squad, nq}:
#
#                 align=trivia_qa   align=squad   align=nq
#   eval=trivia_qa      same           cross        cross
#   eval=squad          cross          same         cross
#   eval=nq             cross          cross        same
#
# same-align combos are added automatically by _resolve_pairs() for every entry
# in --datasets, so --cross-align-dataset lists only the 6 off-diagonal cells.
#
# 21 model pairs x 9 combos = 189 alignment fits.  At ~2 min each that is ~6.5 h
# for Phase 2; Phase 2 skips any (pair, eval_ds, align_ds) whose pkl already
# exists, so results migrated from the earlier per-dataset BTL runs are reused.
#
# Phase 1 (probe_cache) IS run, but it is idempotent: phase_probe_cache() skips
# any probes/<ds>/<model>.pkl that already exists, so pre-copied caches (see
# slurm/copy_probes_to_transfer_v2_btl.sh) cost nothing and only genuinely
# missing (dataset, model) probes get fitted.  Probe caches are transfer-mode
# independent, so copying from a best-to-best run is always valid.
#
# Usage:
#   mkdir -p /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_btl
#   nohup bash slurm/run_transfer_v2_btl_all.sh \
#     >> /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_btl/run_all.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE=offline
# This repo's src, NOT /build_bak/UQ/UQ-transfer/src.  That copy is owned by
# another account (read-only to us) and lags behind: it still has the
# _make_summary_fig() bug that made every best-to-last summary figure come out
# blank ("(missing)" in all 21 cells).  The two trees are otherwise identical.
export PYTHONPATH="$REPO_ROOT/src":/build_bak/UQ/python_packages
PYTHON=/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python

OUT=/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_btl

COMMON="--out-dir $OUT --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 --seed 0 \
        --alpha 1e4 --transfer-mode best-to-last"

# 3 same-align (implicit) + 6 cross-align (explicit) = 9 combos.
PAIR_ARGS="--datasets trivia_qa squad nq \
           --cross-align-dataset trivia_qa:squad trivia_qa:nq \
                                 squad:trivia_qa squad:nq \
                                 nq:trivia_qa nq:squad"

mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# Preflight: probes/ must be a REAL directory, not a leftover symlink into one
# of the best-to-best runs.  Phase 1 writes into probes/, and through a symlink
# those writes would land in the other run's cache.  Missing probes are fine
# (Phase 1 fits them); a symlink is not.
# ---------------------------------------------------------------------------
if [ -L "$OUT/probes" ]; then
    echo "ERROR: $OUT/probes is a symlink -> $(readlink "$OUT/probes")" >&2
    echo "       Phase 1 would write into that other run's cache." >&2
    echo "       Run slurm/copy_probes_to_transfer_v2_btl.sh to materialise it." >&2
    exit 1
fi

echo "=== probe cache inventory before Phase 1"
for ds in trivia_qa squad nq; do
    # find, not `ls dir/*.pkl`: an unmatched glob exits 2 and `set -o pipefail`
    # would abort the script from inside this command substitution.
    have=$(find "$OUT/probes/$ds" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l)
    want=$(awk -v d="$ds" '!/^[[:space:]]*(#|$)/ && $1==d' slurm/inputs/model_paths.txt | wc -l)
    printf "  %-10s %s / %s\n" "$ds" "$have" "$want"
done

echo "=========================================="
echo "Phase 1: probe cache  $(date '+%H:%M:%S')"
echo "=========================================="
# Idempotent: prints "skip <ds>/<model> (exists)" for every pre-copied probe and
# only fits the missing ones.  Note it iterates over ALL models in
# model_paths.txt for each dataset (nq/squad: 8, trivia_qa: 6), which is a
# superset of the 6 models used by pair_list.txt.
$PYTHON -m sep.transfer.transfer2 probe_cache \
    $COMMON --datasets trivia_qa squad nq

echo "=========================================="
echo "Phase 2: alignment cache  $(date '+%H:%M:%S')"
echo "=========================================="
# Output: alignments/<eval_ds>/<pair>/align_<align_ds>_ridge_a1e4_btl.pkl
$PYTHON -m sep.transfer.transfer2 align_cache \
    $COMMON $PAIR_ARGS --aligners ridge

echo "=========================================="
echo "Phase 3: evaluate  $(date '+%H:%M:%S')"
echo "=========================================="
# results/<eval_ds>/<pair>/
#   native_curves_btl_<eval_ds>.json
#   probe_grid_btl_align_<align_ds>_ridge_a1e4.json
#   predictions_btl_align_<align_ds>_ridge_a1e4.json
$PYTHON -m sep.transfer.transfer2 evaluate \
    $COMMON $PAIR_ARGS --aligners ridge

echo "=========================================="
echo "Phase 4: summary plots + Venn  $(date '+%H:%M:%S')"
echo "=========================================="
# summary_plots/summary_{trivia_qa,squad,nq}_probe_grid_btl_ridge_a1e4.{pdf,png}
# summary_plots/venn_ridge_a1e4/
$PYTHON -m sep.transfer.transfer2 summary \
    $COMMON $PAIR_ARGS

echo ""
echo "All done.  $(date '+%H:%M:%S')"
echo "Results:       $OUT/results/"
echo "Summary plots: $OUT/summary_plots/"
