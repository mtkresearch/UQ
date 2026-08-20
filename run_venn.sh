#!/bin/bash
# Venn error decomposition runs.
# Curves: ridge (alpha=1e4), e2_map, e2_r0, e2_rstar (lam=10, alpha_eff≈0.47).
# --save-venn writes venn_slt_a1e+04_venn.json per pair.
# plot_venn.py then produces one 6x6 grid PNG per (axis, aligner) = 8 PNGs.
set -e

PYTHON=/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python
export PYTHONPATH=/build_bak/UQ/UQ-transfer/src:/build_bak/UQ/python_packages

SCRATCH=/build_bak/UQ_project/sep_scratch
UQT_SCRATCH=/build_bak/UQ/UQ-transfer/sep_scratch
OUT_DIR=sep_scratch/transfer/venn

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
PAIR_LABELS=(
    "Llama2-7B->Mistral-7B"
    "Llama3.2-1B->Llama3.1-8B"
    "Llama3.1-8B->Qwen3-8B"
    "Llama3.1-8B->Phi-4"
    "Llama3.1-8B->Gemma-12B"
    "Llama3.1-8B->Mistral-Nemo"
)

N_PAIRS=${#NAMES[@]}

# ------------------------------------------------------------------ #
# Part 1: transfer runs with --save-venn
# ridge uses --alpha 1e4; e2_map/e2_r0/e2_rstar use --lam-e2-map 10
# (alpha_eff = 10/||w_s||^2 ~ 0.47 for e2_r0/e2_rstar)
# ------------------------------------------------------------------ #
echo "=== Venn transfer runs: 6 pairs ==="
echo "    curves: ridge e2_map e2_r0 e2_rstar"
echo "    alpha=1e4  lam=10 (alpha_eff~0.47 for e2_r0/e2_rstar)"

for (( i=0; i<N_PAIRS; i++ )); do
    echo "  ${NAMES[$i]}"
    $PYTHON -m sep.transfer.transfer \
        --source-gen "${SRCS[$i]}" \
        --target-gen "${TGTS[$i]}" \
        --token slt \
        --alpha 1e4 \
        --lam-e2-map 10 \
        --n-eval 500 --n-grid 50 100 200 400 800 1500 \
        --curves ridge e2_map e2_r0 e2_rstar \
        --metrics auroc error_rate \
        --save-venn \
        --out-suffix "_venn" \
        --out-dir "$OUT_DIR/${NAMES[$i]}"
done

# ------------------------------------------------------------------ #
# Part 2: Venn grid plots — one PNG per (axis, aligner) = 8 PNGs
# ------------------------------------------------------------------ #
echo ""
echo "=== Plotting Venn grids ==="

$PYTHON -m sep.transfer.plot_venn \
    --pairs-dir "$OUT_DIR" \
    --pair-names "${NAMES[@]}" \
    --pair-labels "${PAIR_LABELS[@]}" \
    --token slt \
    --alpha 1e4 \
    --in-suffix "_venn" \
    --axes map_budget probe_budget \
    --n-grid 50 100 200 400 800 1500 \
    --out-dir "$OUT_DIR"

echo ""
echo "=== Done. Output in $OUT_DIR ==="
echo ""
echo "  Per pair:  venn_slt_a1e+04_venn.json"
echo ""
echo "  Figures (map-budget axis):"
echo "    venn_map_budget_ridge_slt_a1e+04_venn.png"
echo "    venn_map_budget_e2_map_slt_a1e+04_venn.png"
echo "    venn_map_budget_e2_r0_slt_a1e+04_venn.png"
echo "    venn_map_budget_e2_rstar_slt_a1e+04_venn.png"
echo ""
echo "  Figures (probe-budget axis):"
echo "    venn_probe_budget_ridge_slt_a1e+04_venn.png"
echo "    venn_probe_budget_e2_map_slt_a1e+04_venn.png"
echo "    venn_probe_budget_e2_r0_slt_a1e+04_venn.png"
echo "    venn_probe_budget_e2_rstar_slt_a1e+04_venn.png"
