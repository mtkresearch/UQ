# transfer2.py — SE probe transfer

Train an SE probe on a **source** model, learn a linear map from the target model's
representation space to the source's, and measure how well the source probe transfers
to the **target** model — AUROC as a function of the number of alignment samples `n`.

`transfer2.py` splits the pipeline into four phases, each caching its outputs to disk.
Probes are fitted once per model and reused across all pairs.  Any phase can be
re-run in isolation without redoing earlier work, and phase 4 automatically produces
all summary figures and timing tables.

## Run the full pipeline

```bash
bash slurm/run_transfer_v2.sh          # all four phases, one config, end to end
```

That script is just the four phases in a row, sharing one set of arguments:

```bash
OUT_DIR=/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2
COMMON="--out-dir $OUT_DIR --token slt --n-grid 50 100 200 400 800 1500 --n-eval 500 --seed 0 --alpha 1e3"
PAIR_ARGS="--datasets squad nq --cross-align-dataset squad:nq nq:squad"

python -m sep.transfer.transfer2 probe_cache $COMMON $PAIR_ARGS
python -m sep.transfer.transfer2 align_cache $COMMON $PAIR_ARGS
python -m sep.transfer.transfer2 evaluate    $COMMON $PAIR_ARGS
python -m sep.transfer.transfer2 summary     $COMMON
```

Paths to `validation_generations.pkl` files are looked up automatically from
`slurm/inputs/model_paths.txt`; transfer pairs come from `slurm/inputs/pair_list.txt`.
Both default to those paths and do not need to be passed explicitly.

| Arg | Default | Meaning |
|---|---|---|
| `--out-dir` | *required* | Root directory for every cache, result and figure. |
| `--datasets` | *required* | Which datasets to run, e.g. `squad nq trivia_qa`. |
| `--cross-align-dataset` | none | Cross-alignment pairs as `eval_ds:align_ds`, e.g. `squad:nq nq:squad`. |
| `--model-paths` | `slurm/inputs/model_paths.txt` | Maps `(dataset, model)` to its `validation_generations.pkl`. |
| `--pair-list` | `slurm/inputs/pair_list.txt` | Two-column file listing `src_model  tgt_model` pairs. |
| `--token` | `slt` | Which token's hidden state to probe: `slt` (second-last) or `tbg` (before generation). |
| `--n-grid` | `50 100 200 400 800 1500` | Alignment sample sizes to sweep. |
| `--n-eval` | `500` | Held-out evaluation set size per model; the rest is the training pool. |
| `--seed` | `0` | Seed for the eval/pool split. |
| `--alpha` | `1e3` | Ridge regularization strength (the `ridge` aligner). |
| `--lambda-val` | `1.0` | Lambda for the `e2` aligner. |
| `--aligners` | `ridge` | Phase 2 only: which of `ridge procrustes e2` to fit. |
| `--skip-venn` | off | Phase 4 only: skip the Venn step (the slow part). |
| `--force` | off | Recompute instead of reusing existing cache files. |

---

## The four phases

Each is a subcommand of `python -m sep.transfer.transfer2`; run them individually to
iterate on part of the pipeline.

| Phase | What it does |
|---|---|
| `probe_cache` | Layer search, then fit SE probes for every (model, dataset) at every `n`. |
| `align_cache` | Fit the target→source linear map for every pair, at every `n`. |
| `evaluate` | Apply the source probe to aligned target features → AUROC curves + per-sample predictions. |
| `summary` | Plot; reads only what the phases above wrote, computes no new models. |

Phase 1 is by far the most expensive (loads hidden states, searches all layers) and does
not depend on the aligner or its hyperparameters — run it once, then iterate on 2–4
freely. Every phase skips work whose output already exists, so re-running is cheap and
an interrupted run can simply be restarted.

The Venn step is part of Phase 4 but also runs standalone:

```bash
python -m sep.transfer.compute_venn --out-dir $OUT_DIR --aligner-suffix ridge_a1e3
python -m sep.transfer.plot_venn_grid    --out-dir $OUT_DIR --alpha 1e3
```

---

## Configuring your experiment

### 1. Which models / datasets — `slurm/inputs/`

Two plain text files drive everything; no code changes needed.

**`pair_list.txt`** — the 21 model pairs for the experiment. Do not modify.

**`model_paths.txt`** — maps each `(dataset, model)` to its `validation_generations.pkl`:

```
# dataset  model  path
squad  llama-3.1-8b  /.../validation_generations.pkl
nq     llama-3.1-8b  /.../validation_generations.pkl
```

To run on **TriviaQA**, add its pkl paths to `model_paths.txt` and pass `--datasets trivia_qa`.

### 2. Cross-alignment — via `--cross-align-dataset`

Pass `--cross-align-dataset eval_ds:align_ds` to add cross-dataset alignment pairs.
For example:

```bash
--cross-align-dataset squad:nq nq:squad
```

The main figure draws same-align curves solid and cross-align dashed automatically.
Omit the flag to skip cross-alignment entirely.

### 3. Aligner and its hyperparameters

Three aligners are implemented: `ridge` (default), `procrustes`, `e2`.

```bash
python -m sep.transfer.transfer2 align_cache $COMMON --aligners ridge procrustes e2
```

Hyperparameters are `--alpha` (ridge) and `--lambda-val` (e2); `procrustes` has none.
They must be passed identically to `align_cache`, `evaluate` and `summary`, because
they are encoded into the output filenames (`ridge_a1e3`, `procrustes`, `e2_l1.0`) and
that tag is how the later phases find the right files. A useful consequence: different
hyperparameter settings never overwrite each other, so you can accumulate several
alphas on disk and compare them (see `sweep_summary` below).

To add a **new aligner**: write `_fit_<name>(Zt, Zs) -> (M, b)` next to
`_fit_procrustes`, then register it in four places — the branch in `phase_align_cache`,
the `--aligners` choices, `_build_hyperparams` + `_aligner_tag` (so its files carry its
hyperparameters), and the aligner list in `phase_evaluate`.

---

## Where the output goes

Everything lands under `--out-dir`, which the script calls `$OUT_DIR`.

```
$OUT_DIR/
  probes/<ds>/<model>.pkl                       fitted probes: best layer, one probe per n, labels, splits
  alignments/<eval_ds>/<src>_to_<tgt>/          alignment matrix + bias per n, per aligner
  results/<eval_ds>/<src>_to_<tgt>/             AUROC curves + per-sample predictions (JSON)
  summary_plots/                                figures for one config
  sweep_summary/                                figures/tables comparing hyperparameters
```

## The result figures

**Main figure** — `summary_plots/summary_{nq,squad}_probe_grid_<tag>.{pdf,png}`.
A grid of model pairs; each panel is AUROC vs `n`, with same-align (solid) and
cross-align (dashed) curves, plus the source/target native probe baselines.

**Venn figures** — `summary_plots/venn_<tag>/venn_{eval_ds}_align_{align_ds}_n<n>.png`
(plus a matching `.csv`). Decomposes transfer error into A = source probe error,
B = alignment changed the prediction, C = source/target label mismatch, and
D = transfer error on the target.

**Timing tables** — comparing train-from-scratch vs transfer cost, normalised to 1500
samples.  Not produced automatically; requires a separate data-collection run first:

```bash
# Run once, after the main pipeline has finished (avoid running concurrently
# with the main pipeline, as it competes for GPU and would skew timing).
nohup bash slurm/run_collect_timing.sh \
  > /proj/MR_dataset/mtk53728/UQ/sep_scratch/collect_timing.log 2>&1 &
```

The script runs `sep.generate_answers --compute_uncertainties` on 100 samples per
(model, dataset) to measure:

- **T1** — wall time for 11× inference (1 low-temp + 10 high-temp) on 100 samples
- **T2** — wall time for DeBERTa entailment + semantic clustering on 100 samples

Results are copied to `$OUT_DIR/data_generation_timing/{dataset}_{model}.json`.

Once that directory exists, **Phase 4** (`summary`) automatically reads it together
with `timing.json` (which holds T4/T5/T6/T7 from the main pipeline) and writes:

```
$OUT_DIR/
  data_generation_timing/{dataset}_{model}.json   T1/T2 per model
  timing_collect_data.json                        merged T1–T7, normalised to 1500 samples
  timing_table_squad.csv                          per-pair comparison table (SQuAD)
  timing_table_nq.csv                             per-pair comparison table (NQ)
```

Each table has columns: model pair | T1×10/11 | T2 | T4 | T6 | **Train total** |
T5 | T7 | **Transfer total** | **Speedup ×**.

**Hyperparameter comparison** — run once *after* several hyperparameter settings are on
disk (this is deliberately not part of Phase 4, which only ever reports the single
config it was given):

```bash
python -m sep.transfer.sweep_summary --out-dir $OUT_DIR --pair-list slurm/inputs/pair_list.txt
```

writes to `sweep_summary/`: `hparam_alpha_table_rank.{csv,md}` (best alpha per pair,
by average rank across `n` and at `n=1500`), `hparam_alpha_table_per_n.{csv,md}` (best
alpha at every `n`, unaggregated), and
`summary_{nq,squad}_probe_grid_alpha_best.{pdf,png}` (main figure with each pair using
its own best alpha).
