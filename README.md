# Semantic Uncertainty Probes Across Language Models

Semantic entropy is a useful measure of language-model uncertainty, but computing
it requires generating and comparing multiple answers. Semantic entropy probes
instead predict this uncertainty from a single forward pass, but they are normally
trained separately for each model using expensive semantic entropy labels. We study
whether a probe trained on one model can be reused on another. Our method learns a
lightweight linear map between paired source and target hidden states, without using
target-side semantic entropy labels, and then applies the fixed source probe in the
aligned target space. Across 21 ordered source–target pairs spanning different model
families and scales, transferred probes recover much of the performance of probes
trained directly on the target model and sometimes match or outperform them. To
understand these results, we partition prediction outcomes according to source-probe
errors, transfer-induced changes, and disagreement between source and target
uncertainty labels. On average, transfer-induced changes are more often corrective
than harmful, while large source–target disagreement limits performance. These
findings support the practical reuse of semantic entropy probes and suggest that
language models encode partially shared and linearly recoverable uncertainty signals.

## Installation

```bash
conda env update -f sep_enviroment.yaml
conda activate se_probes
pip install -e src/
```

Set environment variables before running:

```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_ENT=offline
```

## Step 1: Generate data

Each model needs its own generation run. Source and target must use the same
`--dataset`, `--num_samples`, and `--random_seed` so their rows align.

**Single GPU:**

```bash
python -m sep.generate_answers --config configs/model/qwen3-1.7b.yaml
python -m sep.generate_answers --config configs/model/qwen3-8b.yaml
```

Output lands under `sep_scratch/<model>/<timestamp>/` and contains
`validation_generations.pkl`, `train_generations.pkl`, `uncertainty_measures.pkl`.

**Multiple GPUs (recommended for large models):**

```bash
bash slurm/run_multigpu.sh --config configs/model/qwen3-8b.yaml --num_gpus 4
```

Shards the run across GPUs, merges, then runs clustering in one pass.
Merged output is printed at the end as *"Merged run + measures under …"*.

To regenerate all models in batch:

```bash
bash slurm/regen_all.sh
```

Available model configs: `ls configs/model/`.

## Step 2: Run transfer (v4)

The v4 pipeline uses normalisation convention D and a per-dataset eval/pool
split. It runs four transfer modes: `best-to-best`, `best-to-last`,
`best-to-best-sub`, `best-to-align`.

**Run all modes:**

```bash
nohup bash slurm/run_transfer_v4_all.sh \
  >> /path/to/sep_scratch/transfer_v4/run_all.log 2>&1 &
```

Output tree under `$OUT_ROOT` (default: `sep_scratch/transfer_v4/`):

```
_probes_shared/probes/     phase 1, shared across all modes
transfer_v4_b2b/           best-to-best
transfer_v4_btl/           best-to-last
transfer_v4_bbs/           best-to-best-sub
transfer_v4_b2a/           best-to-align
```

Each mode directory contains `probes/`, `alignments/`, `results/`,
`summary_plots/`, `timing.json`.

**Run a single mode:**

```bash
MODES="best-to-align" bash slurm/run_transfer_v4_all.sh
```

**Alpha sweep (hyperparameter search):**

```bash
bash slurm/run_transfer_v4_alpha_sweep.sh
```

## Model pairs

Transfer pairs are defined in `slurm/inputs/pair_list.txt`.
Model paths are in `slurm/inputs/model_paths.txt`.

## Documentation

See `docs/` for detailed notes on the pipeline, normalisation convention,
layer selection modes, and timing methodology.
