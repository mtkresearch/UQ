# Timing pipeline

Builds the paper's target-trained vs. transfer wall-clock table.

Every timing artefact lives in
**`/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2/timing`** (called `$TIMING` below).

## The two commands

```bash
TIMING=/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2/timing

# 1. T1/T2 — GPU, hours.  Only needed if the T1/T2 numbers must change.
nohup bash slurm/run_collect_timing.sh > $TIMING/collect_timing.log 2>&1 &
#    subset:            bash slurm/run_collect_timing.sh trivia_qa
#    keep generations:  KEEP_RUN_DIR=1 bash slurm/run_collect_timing.sh

# 2. T4/T5 + the tables — CPU, minutes.  This is the one you normally rerun.
nohup bash slurm/run_time_fits.sh > $TIMING/time_fits.log 2>&1 &
```

T1/T2 are already collected, so **command 2 alone regenerates the tables.**

## Command 1 → files

| File | Contents |
| --- | --- |
| `$TIMING/data_generation_timing/<ds>_<model>.json` | 22 files, 282 B each. Four timestamps: `t1_inference_{start,end}` → **T1**, `t2a_clustering_se_{start,end}` → **T2**. All three datasets in this one dir. |
| `$TIMING/collect_timing_logs_<ts>/` | per-model generation logs |

The generation run (`$SCRATCH/collect_timing_<ts>/`, ~1–2 GB per dataset) is deleted
automatically once every `timing.json` has been harvested.

## Command 2 → files

| File | Contents |
| --- | --- |
| `$TIMING/timing_fits.json` | **T4** (`probe_cache`) and **T5** (`align_cache`), plus `_meta` with host/numpy/BLAS/threads |
| `$TIMING/timing_table_{nq,squad,trivia_qa}.csv` | the tables, one row per source→target pair (21 rows) |
| `$TIMING/timing_collect_data_all.json` | T1/T2 rollup per `<ds>/<model>` |
| `$TIMING/time_fits.log` | log, ends with a per-target summary |
| `$TIMING/fit_timing/z/*.npz` | feature cache, 1.4 GB, deletable |
| `$TIMING/timing_table_prev_<ts>/` | backup of the CSVs it replaced |

## T1–T5

`Total(target-trained) = T1 + T2 + T4`, `Total(transfer) = T3 + T5`, `Speedup` = their ratio.

| | Column | Measured how |
| --- | --- | --- |
| T1 | Answer sampling | 11 generations/question on 100 questions, **×15** to 1500 |
| T2 | Semantic clustering | NLI entailment, same run, ×15 |
| T3 | Hidden-state extraction | derived: **T1 / 11** — transfer needs only the 1 low-temp pass |
| T4 | Probe training | one LogisticRegression at n=1500 |
| T5 | Alignment fitting | one ridge solve at n=1500, α=1e4 |

Excluded: eval-time probe inference (<0.05 s, identical on both paths).
If you change `NUM_SAMPLES=100` in command 1, change `GEN_SAMPLES` in
`make_timing_table.py` — the ×15 comes from it.

## CSV → paper table

The CSVs are per pair; the paper table averages over each target's sources:

```python
import csv, statistics as st
rows = list(csv.DictReader(open(f"{TIMING}/timing_table_nq.csv")))
for tgt in ["llama-3.1-8b", "qwen3-8b", "phi-4", "gemma-4-12b", "mistral-nemo"]:
    r = [x for x in rows if x["Source -> Target"].split(" -> ")[1] == tgt]
    print(tgt, st.mean(float(x["Alignment fitting (s)"]) for x in r), len(r))
```

The caption must state: mean-row speedup is the **ratio of the mean totals**, not the mean
of the per-row speedups (NQ: 5722.49/482.67 = 11.86, vs 11.91); and `llama-3.1-8b`
averages over 5 sources (includes `llama-3.2-1b`), the others over 4.

## `timing.json` vs `timing_fits.json`

Same schema, different files, **disjoint contents**:

| | `timing.json` | `timing_fits.json` |
| --- | --- | --- |
| written by | `transfer2.py`, incrementally, every run | `run_time_fits.sh` (command 2) |
| `probe_cache` | **empty** | 6 models × 3 datasets → T4 |
| `align_cache` | only E2R λ-sweep timings (`e2_rstar_l1e3…l1e7`) | `ridge_a1e4` per pair → T5 |
| used by the table | **no** | **yes** |

Command 2 produces `$TIMING/timing_fits.json`. `timing.json` is the one timing file that
stays **outside** `$TIMING`, at `.../transfer_v2/timing.json` — it is `transfer2.py`'s
runtime cache, still read/written by it and by the consolidate/merge scripts.

A plain `transfer2.py` run also auto-writes tables from its own `timing.json`. Those go to
`$TIMING/transfer2_auto/` so they cannot overwrite the paper ones; their T4/T5 columns come
from an uncontrolled single cold fit (and are blank if phase 3 did not run). Ignore them.

Why command 2 re-measures instead of reusing `transfer2.py`'s timings: the original runs
weren't comparable (SQuAD/NQ under numpy 1.26+MKL, TriviaQA under numpy 2.5+OpenBLAS —
26.4 s vs 43.6 s for the same ridge grid on the same host — neither with pinned BLAS
threads on a shared 64-core box). Command 2 pins threads to 8 and the interpreter to
`se_probes`, and reports the min of 3 repeats.
