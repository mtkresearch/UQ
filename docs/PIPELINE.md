# Cross-Scale SE Probe Transfer — End-to-End Pipeline

This walks the full pipeline for the cross-scale study, from generation to
figures, using a **source → target** model pair. The canonical pair is
`Qwen3-1.7B` (source) → `Qwen3-8B` (target): a probe trained on the small model
is transferred to the large model with zero target labels via a linear map fit
on paired hidden states.

Every stage is driven by an omegaconf YAML config (`configs/`). `--config` takes
a YAML **path** (e.g. `configs/model/qwen3-8b.yaml`, relative to the repo root
or absolute) that is layered on `configs/base.yaml`. Any CLI flag overrides the
YAML (CLI > YAML > base).

Available model presets: `ls configs/model/`.

> **Environment.** Run everything from the GPU worktree
> (`/build_bak/mtk53686/semantic-entropy-probes`) with its `.venv` activated.
> Edits are made in `/proj/MR_dataset/mtk53686/semantic-entropy-probes` and
> `cp`-synced. The slurm launcher exports the TLS/offline env for you; for
> manual runs first:
> ```bash
> export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
>        REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
>        CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
>        HF_HUB_OFFLINE=1 WANDB_MODE=offline WANDB_ENT=offline
> source .venv/bin/activate
> ```

---

## 1. Generate the data (two models)

Generation samples answers, caches hidden states (SLT + TBG token positions) and
log-likelihoods, then **clusters** the samples with the DeBERTa entailment model
to compute semantic entropy — this clustering is the compute stage, run inline
(single-GPU) or once after merging (multi-GPU). You need one run **per model**;
both models must use the **same `--dataset`, `--num_samples`, `--random_seed`**
so their example ids align row-for-row (the transfer scripts assert this).

### 1a. Single GPU

`generate_answers` runs generation → clustering/compute in one process
(`--compute_uncertainties` defaults on):

```bash
# Source model
CUDA_VISIBLE_DEVICES=0 python -m sep.generate_answers --config configs/model/qwen3-1.7b.yaml
# Target model
CUDA_VISIBLE_DEVICES=1 python -m sep.generate_answers --config configs/model/qwen3-8b.yaml
```

Output lands in a wandb run dir under `$SCRATCH_DIR/$USER/uncertainty/` and
contains `validation_generations.pkl`, `train_generations.pkl`, and
`uncertainty_measures.pkl`. Override any config value inline, e.g.
`--dataset squad --num_samples 5000`.

### 1b. Multiple GPUs (data-parallel)

`slurm/run_multigpu.sh` shards the seeded sample across GPUs (one model per GPU),
waits for all shards, then runs a **single merged clustering/compute pass** over
the concatenated shards. Only `--num_shards` / `--shard_index` are injected; the
config drives everything else. Extra flags after `--config`/`--num_gpus` are
forwarded to each shard as generation overrides.

```bash
# Source model, 4 GPUs
bash slurm/run_multigpu.sh --config configs/model/qwen3-1.7b.yaml --num_gpus 4
# Target model, 8 GPUs, with an ad-hoc dataset/size override
bash slurm/run_multigpu.sh --config configs/model/qwen3-8b.yaml --num_gpus 8 --dataset squad --num_samples 5000
```

Merged output (the file the later stages consume) is written under
`$SHARDS_PARENT/merged/…/`, printed at the end as *"Merged run + measures under …"*.
Default `$SHARDS_PARENT` is `$REPO_ROOT/sep_scratch/<CONFIG_TAG>/<TIMESTAMP>/shards`,
so a run's tree looks like `sep_scratch/<model>/<timestamp>/{shards,shards/merged}`.

**Where the pickles are.** Note the merged run dir for each model — later stages
reference its `validation_generations.pkl` and `uncertainty_measures.pkl`. The
merged wandb run dir is nested a few levels down under `merged/`; below we use
these placeholders (real example paths, one `offline-run-*/files` dir per model):

```bash
SRC_GEN=sep_scratch/qwen3-1.7b/<timestamp>/merged/$USER/uncertainty/wandb/offline-run-*/files/validation_generations.pkl
TGT_GEN=sep_scratch/qwen3-8b/<timestamp>/merged/$USER/uncertainty/wandb/offline-run-*/files/validation_generations.pkl
```

---

## 2. Train the cross-scale mapping (source → target)

Fit the affine map `f(z_t) = z_t @ M + b` (target space → source space) on
**unlabeled** paired examples, then transfer the fixed source probe in closed
form and score target examples. Produces a sample-efficiency curve: transferred
(zero target labels) vs native (labeled) AUROC across alignment-set sizes.

```bash
python -m sep.transfer.transfer \
    --source-gen "$SRC_GEN" \
    --target-gen "$TGT_GEN" \
    --token slt \
    --out-dir results/transfer_1p7b_to_8b \
    --alpha 1e4          # ridge regularization (1e4 is the tuned optimum)
```

`--source-gen` is the model whose probe is trained (with SE labels); `--target-gen`
is where the probe is transferred. The two runs only need aligned example ids
(same `--dataset`/`--num_samples`/`--random_seed`) — unequal hidden dims are fine,
the map is `(d_target, d_source)`. A concrete cross-family run (Llama-3.1-8B →
Gemma-4-12B, both trivia_qa / 2000 / seed 20):

```bash
python -m sep.transfer.transfer \
    --source-gen sep_scratch/llama-3.1-8b/20260722_183618/merged/$USER/uncertainty/wandb/offline-run-20260722_212914-3na5fqgj/files/validation_generations.pkl \
    --target-gen sep_scratch/gemma-4-12b/20260722_195243/merged/$USER/uncertainty/wandb/offline-run-20260722_202149-26xjcjwe/files/validation_generations.pkl \
    --token slt --alpha 1e4 \
    --out-dir sep_scratch/mappings/llama-3.1-8b_gemma-4-12b/
```

> **Cross-scale vs cross-family.** The canonical study is *cross-scale* (same
> family, different size: Qwen3-1.7B → Qwen3-8B), where a linear map between
> hidden spaces is well-motivated. A *cross-family* pair (different tokenizers,
> training data, architectures — e.g. Llama → Gemma) runs mechanically but tests
> a much stronger assumption; Curve B may sit near chance. That's a useful null:
> if cross-family transfer fails where cross-scale succeeds, it supports the
> "shared subspace within a family" story.

Optional deeper analyses on the same paired data:

```bash
# Nonlinear (shallow MLP) map — does nonlinearity beat ridge?
python -m sep.transfer.step3_nonlinear \
    --source-gen "$SRC_GEN" --target-gen "$TGT_GEN" \
    --token slt --out-dir results/step3_1p7b_to_8b

# Baselines + layer-selection variants (ridge/procrustes/random/identity)
python -m sep.transfer.controls \
    --source-gen "$SRC_GEN" --target-gen "$TGT_GEN" \
    --token slt --out-dir results/controls_1p7b_to_8b \
    --layer-variant supervised

# Cross-distribution transfer (fit on dataset A, eval on B) — needs one
# SRC/TGT pair per dataset:
python -m sep.transfer.cross_dataset \
    --source-gens SRC_trivia SRC_squad SRC_nq \
    --target-gens TGT_trivia TGT_squad TGT_nq \
    --names trivia_qa squad nq \
    --token slt --out-dir results/cross_dataset_1p7b_to_8b
```

Each writes a curve/matrix (PDF) + a JSON of the numbers under `--out-dir`.

---

