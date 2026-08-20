#!/bin/bash
set -e

PYTHON=/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python
export PYTHONPATH=/build_bak/UQ/UQ-transfer/src:/build_bak/UQ/python_packages

SCRATCH=/build_bak/UQ_project/sep_scratch
UQT_SCRATCH=/build_bak/UQ/UQ-transfer/sep_scratch
OUT_DIR=sep_scratch/transfer/e2_compare

# Parallel arrays — one entry per model pair
NAMES=(
    llama2_to_mistral
    llama32-1b_to_llama31-8b
    llama31-8b_to_qwen3-8b
    llama31-8b_to_phi4
    llama31-8b_to_gemma
    llama31-8b_to_nemo
)

SRCS=(
    "$UQT_SCRATCH/llama-2-7b/20260727_223927/shards/merged/gpu_mtk53658/uncertainty/wandb/offline-run-20260727_225923-m88txqoz/files/validation_generations.pkl"
    "$UQT_SCRATCH/llama-3.2-1b/20260727_232644/shards/merged/gpu_mtk53658/uncertainty/wandb/offline-run-20260727_233553-derway32/files/validation_generations.pkl"
    "$SCRATCH/llama-3.1-8b/20260722_183618/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_212914-3na5fqgj/files/validation_generations.pkl"
    "$SCRATCH/llama-3.1-8b/20260722_183618/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_212914-3na5fqgj/files/validation_generations.pkl"
    "$SCRATCH/llama-3.1-8b/20260722_183618/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_212914-3na5fqgj/files/validation_generations.pkl"
    "$SCRATCH/llama-3.1-8b/20260722_183618/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_212914-3na5fqgj/files/validation_generations.pkl"
)

TGTS=(
    "$UQT_SCRATCH/mistral-7b/20260727_230620/shards/merged/gpu_mtk53658/uncertainty/wandb/offline-run-20260727_233032-g9dio8gc/files/validation_generations.pkl"
    "$SCRATCH/llama-3.1-8b/20260722_183618/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_212914-3na5fqgj/files/validation_generations.pkl"
    "$SCRATCH/qwen3-8b/20260722_225812/shards/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_230742-72ju7z96/files/validation_generations.pkl"
    "$SCRATCH/phi-4/20260722_202423/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_212315-r9j3f3bx/files/validation_generations.pkl"
    "$SCRATCH/gemma-4-12b/20260722_195243/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_202149-26xjcjwe/files/validation_generations.pkl"
    "$SCRATCH/mistral-nemo/20260722_201421/shards/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_203045-84l9fd2d/files/validation_generations.pkl"
)

PAIR_NAMES="${NAMES[*]}"
PAIR_LABELS="Llama2-7B->Mistral-7B Llama3.2-1B->Llama3.1-8B Llama3.1-8B->Qwen3-8B Llama3.1-8B->Phi-4 Llama3.1-8B->Gemma-12B Llama3.1-8B->Mistral-Nemo"
PAIR_TYPES="cross-family cross-scale cross-family cross-family cross-family cross-family"

N_PAIRS=${#NAMES[@]}

# ------------------------------------------------------------------ #
# Part 1: 6 pairs, lam=1e4 — generates map-budget + probe-budget
# ------------------------------------------------------------------ #
echo "=== Part 1/3: map-budget + probe-budget runs (lam=1e4) ==="

for (( i=0; i<N_PAIRS; i++ )); do
    name="${NAMES[$i]}"
    src="${SRCS[$i]}"
    tgt="${TGTS[$i]}"
    echo "  $name"
    $PYTHON -m sep.transfer.transfer \
        --source-gen "$src" \
        --target-gen "$tgt" \
        --token slt --alpha 1e4 --lam-e2-map 1e4 \
        --n-eval 500 --n-grid 50 100 200 400 800 1500 \
        --curves target_probe source_probe ridge e2_minimised e2_map \
        --metrics auroc \
        --out-dir "$OUT_DIR/$name"
done

# ------------------------------------------------------------------ #
# Part 2: 6 pairs x 4 remaining lambdas — probe-budget only
# ------------------------------------------------------------------ #
echo ""
echo "=== Part 2/3: probe-budget lambda sweep (lam = 1e1 1e2 1e3 1e5) ==="

for lam in 1e1 1e2 1e3 1e5; do
    echo "  --- lambda=$lam ---"
    for (( i=0; i<N_PAIRS; i++ )); do
        name="${NAMES[$i]}"
        src="${SRCS[$i]}"
        tgt="${TGTS[$i]}"
        echo "    $name"
        $PYTHON -m sep.transfer.transfer \
            --source-gen "$src" \
            --target-gen "$tgt" \
            --token slt --alpha 1e4 --lam-e2-map "$lam" \
            --n-eval 500 --n-grid 50 100 200 400 800 1500 \
            --curves target_probe source_probe ridge e2_minimised e2_map \
            --metrics auroc \
            --out-suffix "_lam${lam}" \
            --out-dir "$OUT_DIR/$name"
    done
done

# ------------------------------------------------------------------ #
# Part 3: aggregate figures
# ------------------------------------------------------------------ #
echo ""
echo "=== Part 3/3: aggregate figures ==="

# Map-budget figure (single, lam=1e4, no suffix)
echo "  map-budget combined figure"
$PYTHON -m sep.transfer.plot_transfer_comparison_probe_aligned \
    --pairs-dir "$OUT_DIR" \
    --pair-names $PAIR_NAMES \
    --pair-labels $PAIR_LABELS \
    --pair-types  $PAIR_TYPES \
    --token slt --alpha 1e4 --lam-e2-map 1e4 \
    --out-dir "$OUT_DIR"

# Probe-budget figures: one per lambda (lam=1e4 has no suffix, others have _lam<X>)
for lam in 1e1 1e2 1e3 1e4 1e5; do
    if [ "$lam" = "1e4" ]; then
        suffix=""
    else
        suffix="_lam${lam}"
    fi
    echo "  probe-budget combined figure  lam=$lam"
    $PYTHON -m sep.transfer.plot_transfer_comparison_probe_aligned \
        --pairs-dir "$OUT_DIR" \
        --pair-names $PAIR_NAMES \
        --pair-labels $PAIR_LABELS \
        --pair-types  $PAIR_TYPES \
        --token slt --alpha 1e4 --lam-e2-map "$lam" \
        --out-suffix "$suffix" \
        --probe-budget-only \
        --out-dir "$OUT_DIR"
done

echo ""
echo "=== All done. Figures written to $OUT_DIR ==="
echo ""
echo "Output figures:"
echo "  ALL_pairs_mapbudget_slt_a1e4.png            (map-budget,   lam=1e4)"
echo "  ALL_pairs_probebudget_slt_a1e4.png          (probe-budget, lam=1e4)"
echo "  ALL_pairs_probebudget_slt_a1e4_lam1e1.png   (probe-budget, lam=1e1)"
echo "  ALL_pairs_probebudget_slt_a1e4_lam1e2.png   (probe-budget, lam=1e2)"
echo "  ALL_pairs_probebudget_slt_a1e4_lam1e3.png   (probe-budget, lam=1e3)"
echo "  ALL_pairs_probebudget_slt_a1e4_lam1e5.png   (probe-budget, lam=1e5)"
