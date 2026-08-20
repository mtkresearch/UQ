#!/bin/bash
set -e  # stop on first error

PYTHON=/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python
export PYTHONPATH=/build_bak/UQ/UQ-transfer/src:/build_bak/UQ/python_packages

SCRATCH=/build_bak/UQ_project/sep_scratch
UQT_SCRATCH=/build_bak/UQ/UQ-transfer/sep_scratch
COMMON="--token slt --alpha 1e4 --n-eval 500 --n-grid 50 100 200 400 800 1500"

# Verify we are loading the right sep source
echo "sep loaded from: $($PYTHON -c 'import sep; print(sep.__file__)')"

echo "=== 1/6 Llama2-7B -> Mistral-7B ==="
$PYTHON -m sep.transfer.transfer \
  --source-gen $UQT_SCRATCH/llama-2-7b/20260727_223927/shards/merged/gpu_mtk53658/uncertainty/wandb/offline-run-20260727_225923-m88txqoz/files/validation_generations.pkl \
  --target-gen $UQT_SCRATCH/mistral-7b/20260727_230620/shards/merged/gpu_mtk53658/uncertainty/wandb/offline-run-20260727_233032-g9dio8gc/files/validation_generations.pkl \
  $COMMON --out-dir sep_scratch/transfer/probe_aligned/llama2_to_mistral

echo "=== 2/6 Llama3.2-1B -> Llama3.1-8B ==="
$PYTHON -m sep.transfer.transfer \
  --source-gen $UQT_SCRATCH/llama-3.2-1b/20260727_232644/shards/merged/gpu_mtk53658/uncertainty/wandb/offline-run-20260727_233553-derway32/files/validation_generations.pkl \
  --target-gen $SCRATCH/llama-3.1-8b/20260722_183618/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_212914-3na5fqgj/files/validation_generations.pkl \
  $COMMON --out-dir sep_scratch/transfer/probe_aligned/llama32-1b_to_llama31-8b

echo "=== 3/6 Llama3.1-8B -> Qwen3-8B ==="
$PYTHON -m sep.transfer.transfer \
  --source-gen $SCRATCH/llama-3.1-8b/20260722_183618/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_212914-3na5fqgj/files/validation_generations.pkl \
  --target-gen $SCRATCH/qwen3-8b/20260722_225812/shards/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_230742-72ju7z96/files/validation_generations.pkl \
  $COMMON --out-dir sep_scratch/transfer/probe_aligned/llama31-8b_to_qwen3-8b

echo "=== 4/6 Llama3.1-8B -> Phi-4 ==="
$PYTHON -m sep.transfer.transfer \
  --source-gen $SCRATCH/llama-3.1-8b/20260722_183618/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_212914-3na5fqgj/files/validation_generations.pkl \
  --target-gen $SCRATCH/phi-4/20260722_202423/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_212315-r9j3f3bx/files/validation_generations.pkl \
  $COMMON --out-dir sep_scratch/transfer/probe_aligned/llama31-8b_to_phi4

echo "=== 5/6 Llama3.1-8B -> Gemma-4-12B ==="
$PYTHON -m sep.transfer.transfer \
  --source-gen $SCRATCH/llama-3.1-8b/20260722_183618/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_212914-3na5fqgj/files/validation_generations.pkl \
  --target-gen $SCRATCH/gemma-4-12b/20260722_195243/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_202149-26xjcjwe/files/validation_generations.pkl \
  $COMMON --out-dir sep_scratch/transfer/probe_aligned/llama31-8b_to_gemma

echo "=== 6/6 Llama3.1-8B -> Mistral-Nemo ==="
$PYTHON -m sep.transfer.transfer \
  --source-gen $SCRATCH/llama-3.1-8b/20260722_183618/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_212914-3na5fqgj/files/validation_generations.pkl \
  --target-gen $SCRATCH/mistral-nemo/20260722_201421/shards/merged/gpu_mtk53686/uncertainty/wandb/offline-run-20260722_203045-84l9fd2d/files/validation_generations.pkl \
  $COMMON --out-dir sep_scratch/transfer/probe_aligned/llama31-8b_to_nemo

echo "=== All done ==="
