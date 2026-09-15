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

Each model needs its own generation run. All models on the same dataset must use
the same `--num_samples` and `--random_seed` so their rows align.

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
Merged output path is printed at the end as *"Merged run + measures under …"*.

**All models in batch:**

```bash
bash slurm/regen_all.sh
```

Available model configs: `ls configs/model/`.

## Step 2: Register model paths

Edit `slurm/inputs/model_paths.txt` to point each `(dataset, model)` at its
`validation_generations.pkl`:

```
# dataset  model       path
squad       qwen3-1.7b  /path/to/.../validation_generations.pkl
squad       qwen3-8b    /path/to/.../validation_generations.pkl
nq          qwen3-1.7b  /path/to/.../validation_generations.pkl
nq          qwen3-8b    /path/to/.../validation_generations.pkl
```

Transfer pairs are defined in `slurm/inputs/pair_list.txt` (21 pairs).

## Step 3: Run transfer (v4)

```bash
nohup bash slurm/run_transfer_v4_all.sh \
  >> /path/to/sep_scratch/transfer_v4/run_all.log 2>&1 &
```

Override the output root:

```bash
OUT_ROOT=/your/path bash slurm/run_transfer_v4_all.sh
```

Output tree under `$OUT_ROOT`:

```
_probes_shared/probes/
  <dataset>/<model>.pkl              SE probe per (model, dataset): best layer, fitted probes, labels

transfer_v4_b2b/
  alignments/<dataset>/<src>_to_<tgt>/
    ridge_a1e4_n<N>.pkl              alignment matrix M and bias b at each sample size N

  results/<dataset>/<src>_to_<tgt>/
    probe_grid_align_<dataset>_ridge_a1e4.json   AUROC vs N curve (transferred probe)
    predictions_align_<dataset>_ridge_a1e4.json  per-sample predictions at each N
    native_curves_<dataset>.json                 AUROC of source/target native probes (baselines)
    native_preds_<dataset>.json                  per-sample predictions of native probes

  summary_plots/
    summary_<dataset>_probe_grid_ridge_a1e4.{pdf,png}   main figure: AUROC vs N for all pairs
    venn_ridge_a1e4/                                     Venn error decomposition figures

  timing.json                        wall-clock time per phase
```

The pipeline has four phases, each caching its outputs to disk. Any phase can be
re-run individually without redoing earlier work:

| Phase | Subcommand | What it does |
|---|---|---|
| 1 | `probe_cache` | Layer search + fit SE probes for every (model, dataset) |
| 2 | `align_cache` | Fit target→source linear map for every pair |
| 3 | `evaluate` | Apply source probe to aligned target features → AUROC |
| 4 | `summary` | Produce all figures |

Example — re-run only figures after a plot change:

```bash
python -m sep.transfer.transfer2 evaluate --out-dir .../transfer_v4_b2b --token slt --datasets squad nq --alpha 1e4
python -m sep.transfer.transfer2 summary  --out-dir .../transfer_v4_b2b --token slt --datasets squad nq --alpha 1e4
```

## Other experiments

**Layer selection modes** — compare four strategies for choosing the target layer
(`best-to-best`, `best-to-last`, `best-to-best-sub`, `best-to-align`):

```bash
MODES="best-to-best best-to-last best-to-best-sub best-to-align" \
  bash slurm/run_transfer_v4_all.sh
```

Each mode writes to its own subdirectory (`transfer_v4_b2b/`, `transfer_v4_btl/`,
`transfer_v4_bbs/`, `transfer_v4_b2a/`) and never overwrites another.

**Alpha sweep** — search over ridge regularisation strength:

```bash
bash slurm/run_transfer_v4_alpha_sweep.sh
```

## Documentation

See `docs/` for detailed notes on transfer modes, normalisation convention,
timing methodology, and Venn error decomposition.
