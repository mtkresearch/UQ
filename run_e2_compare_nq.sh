#!/bin/bash
set -e

PYTHON=/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python
export PYTHONPATH=/build_bak/UQ/UQ-transfer/src:/build_bak/UQ/python_packages

NQ_SCRATCH=/proj/MR_dataset/mtk53728/UQ/sep_scratch/nq_all_20260724_030801
OUT_DIR=sep_scratch/transfer/e2_compare_nq

NAMES=(
    llama2_to_mistral
    llama32-1b_to_llama31-8b
    llama31-8b_to_qwen3-8b
    llama31-8b_to_phi4
    llama31-8b_to_gemma
    llama31-8b_to_nemo
)

SRCS=(
    "$NQ_SCRATCH/llama-2-7b/gpu_mtk53728/uncertainty/wandb/offline-run-20260724_030812-ywwuqhs4/files/validation_generations.pkl"
    "$NQ_SCRATCH/llama-3.2-1b/gpu_mtk53728/uncertainty/wandb/offline-run-20260724_030809-6tiwcjgy/files/validation_generations.pkl"
    "$NQ_SCRATCH/llama-3.1-8b/gpu_mtk53728/uncertainty/wandb/offline-run-20260724_030809-b6p5avs1/files/validation_generations.pkl"
    "$NQ_SCRATCH/llama-3.1-8b/gpu_mtk53728/uncertainty/wandb/offline-run-20260724_030809-b6p5avs1/files/validation_generations.pkl"
    "$NQ_SCRATCH/llama-3.1-8b/gpu_mtk53728/uncertainty/wandb/offline-run-20260724_030809-b6p5avs1/files/validation_generations.pkl"
    "$NQ_SCRATCH/llama-3.1-8b/gpu_mtk53728/uncertainty/wandb/offline-run-20260724_030809-b6p5avs1/files/validation_generations.pkl"
)

TGTS=(
    "$NQ_SCRATCH/mistral-7b/gpu_mtk53728/uncertainty/wandb/offline-run-20260724_035751-9pib3jl5/files/validation_generations.pkl"
    "$NQ_SCRATCH/llama-3.1-8b/gpu_mtk53728/uncertainty/wandb/offline-run-20260724_030809-b6p5avs1/files/validation_generations.pkl"
    "$NQ_SCRATCH/qwen3-8b/gpu_mtk53728/uncertainty/wandb/offline-run-20260724_042641-pbl4x2wk/files/validation_generations.pkl"
    "$NQ_SCRATCH/phi-4/gpu_mtk53728/uncertainty/wandb/offline-run-20260724_073914-6stzhhpn/files/validation_generations.pkl"
    "$NQ_SCRATCH/gemma-4-12b/gpu_mtk53728/uncertainty/wandb/offline-run-20260724_054641-3malc07q/files/validation_generations.pkl"
    "$NQ_SCRATCH/mistral-nemo/gpu_mtk53728/uncertainty/wandb/offline-run-20260724_054641-5q62o7yr/files/validation_generations.pkl"
)

PAIR_NAMES="${NAMES[*]}"
PAIR_LABELS="Llama2-7B->Mistral-7B Llama3.2-1B->Llama3.1-8B Llama3.1-8B->Qwen3-8B Llama3.1-8B->Phi-4 Llama3.1-8B->Gemma-12B Llama3.1-8B->Mistral-Nemo"
PAIR_TYPES="cross-family cross-scale cross-family cross-family cross-family cross-family"

N_PAIRS=${#NAMES[@]}

# ------------------------------------------------------------------ #
# Part 1: 6 pairs, alpha=1e4, lam=1e1 — probe-budget only
# ------------------------------------------------------------------ #
echo "=== Part 1/2: transfer runs (NQ, alpha=1e4, lam=1e1) ==="

for (( i=0; i<N_PAIRS; i++ )); do
    name="${NAMES[$i]}"
    src="${SRCS[$i]}"
    tgt="${TGTS[$i]}"
    echo "  $name"
    $PYTHON -m sep.transfer.transfer \
        --source-gen "$src" \
        --target-gen "$tgt" \
        --token slt --alpha 1e4 --lam-e2-map 1e1 \
        --n-eval 500 --n-grid 50 100 200 400 800 1500 \
        --curves target_probe source_probe ridge e2_map \
        --metrics auroc \
        --out-suffix "_lam1e1" \
        --out-dir "$OUT_DIR/$name"
done

# ------------------------------------------------------------------ #
# Part 2: aggregate probe-budget figure
# ------------------------------------------------------------------ #
echo ""
echo "=== Part 2/2: probe-budget combined figure (NQ) ==="

$PYTHON -m sep.transfer.plot_transfer_comparison_probe_aligned \
    --pairs-dir "$OUT_DIR" \
    --pair-names $PAIR_NAMES \
    --pair-labels $PAIR_LABELS \
    --pair-types  $PAIR_TYPES \
    --token slt --alpha 1e4 --lam-e2-map 1e1 \
    --out-suffix "_lam1e1" \
    --probe-budget-only \
    --out-dir "$OUT_DIR"

echo ""
echo "=== All done. Output: $OUT_DIR ==="
echo "  ALL_pairs_probebudget_slt_a1e+04_lam1e1.png"
