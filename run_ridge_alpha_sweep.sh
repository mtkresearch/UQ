#!/bin/bash
# Ridge alpha sweep — ridge curve only, no source/target probe (run_baselines.sh handles those).
# Output per pair: transfer_slt_a<alpha>_rdg.json for each alpha in {1e1..1e5}.
set -e

PYTHON=/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python
export PYTHONPATH=/build_bak/UQ/UQ-transfer/src

SCRATCH=/build_bak/UQ_project/sep_scratch
UQT_SCRATCH=/build_bak/UQ/UQ-transfer/sep_scratch
OUT_DIR=sep_scratch/transfer/hyperparam_sweep_v2

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

N_PAIRS=${#NAMES[@]}

echo "=== Ridge alpha sweep: 6 pairs x 5 alpha values ==="

for alpha in 1e1 1e2 1e3 1e4 1e5; do
    echo "  --- alpha=$alpha ---"
    for (( i=0; i<N_PAIRS; i++ )); do
        echo "    ${NAMES[$i]}"
        $PYTHON -m sep.transfer.transfer \
            --source-gen "${SRCS[$i]}" \
            --target-gen "${TGTS[$i]}" \
            --token slt --alpha "$alpha" \
            --n-eval 500 --n-grid 50 100 200 400 800 1500 \
            --curves ridge \
            --metrics auroc error_rate \
            --out-suffix "_rdg" \
            --out-dir "$OUT_DIR/${NAMES[$i]}"
    done
done

echo ""
echo "=== Done. Output per pair: transfer_slt_a<alpha>_rdg.json ==="
echo "    e.g. transfer_slt_a1e+01_rdg.json ... transfer_slt_a1e+05_rdg.json"
