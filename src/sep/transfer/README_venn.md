# Venn error decomposition

Decomposes the transferred probe's error on the target model into three
overlapping causes, and draws it as an area-proportional Venn diagram.

## Install

Only one extra dependency, needed for the plotting step only:

```bash
pip install matplotlib_venn        # tested with 1.1.2
```

The `transfer.py` step needs nothing new.

## Usage

Two steps. First add `--save-venn` to a normal `transfer.py` run:

```bash
python -m sep.transfer.transfer \
    --source-gen <src>/validation_generations.pkl \
    --target-gen <tgt>/validation_generations.pkl \
    --token slt --n-eval 500 --n-grid 50 100 200 400 800 1500 \
    --curves target_probe source_probe ridge procrustes \
    --alpha 1e4 \
    --save-venn \
    --out-dir sep_scratch/transfer/<pair-name>
```

This writes `venn_slt_a1e+04.json` next to the usual `transfer_slt_a1e+04.json`
(counts only, a few KB). Without `--save-venn` nothing changes, so existing
runs and artifacts are unaffected.

Then plot, passing the pairs the same way as `plot_transfer_comparison.py`:

```bash
python -m sep.transfer.plot_venn \
    --pairs-dir sep_scratch/transfer \
    --pair-names llama32-1b_to_llama31-8b llama2_to_mistral \
    --pair-labels "Llama3.2-1B->Llama3.1-8B" "Llama2-7B->Mistral-7B" \
    --token slt --alpha 1e4
```

`--alpha`, `--token`, `--seed` and `--in-suffix` must match the run, since they
determine the filename. One PNG per (budget axis, aligner):
`venn_<axis>_<aligner>_<token><suffix>.png`, with one row per pair and one
column per `n`. Restrict the output with `--axes`, `--aligners`, `--n-grid`.

To sanity-check the whole thing quickly, run both steps with a single budget
point:

```bash
OUT=sep_scratch/transfer/tmp
PAIR=llama32-1b_to_llama31-8b

python -m sep.transfer.transfer \
    --source-gen <src>/validation_generations.pkl \
    --target-gen <tgt>/validation_generations.pkl \
    --token slt --n-eval 500 --n-grid 200 \
    --curves target_probe source_probe ridge procrustes \
    --alpha 1e4 --save-venn \
    --out-dir "$OUT/$PAIR"

python -m sep.transfer.plot_venn \
    --pairs-dir "$OUT" --pair-names "$PAIR" \
    --token slt --alpha 1e4 --out-dir "$OUT"
```

## What the regions mean

Per eval example, three binary events:

| | meaning |
|---|---|
| **A** | the source probe disagrees with the source label — *source probe error* |
| **B** | transfer flips the prediction relative to the source probe |
| **C** | source and target SE labels disagree — *label mismatch* |
| **D** | the transferred probe disagrees with the target label — *what we care about* |

Over GF(2), `D = A xor B xor C` exactly, so

```
D = A_only + B_only + C_only + ABC
```

The four odd-parity regions are the error; the three even-parity regions
(`AB`, `AC`, `BC`) are examples where two effects cancel and the transferred
probe is right for the wrong reason. `transfer.py` asserts this identity on
every record, so a failure means a bookkeeping bug, not a modelling result.

Reading the colors: red `B_only` is pure transfer harm (a correct source
prediction broken by transfer), dark blue `AB` is transfer *fixing* a source
error, pale blue `BC` is transfer adapting to a label mismatch, grey regions do
not involve transfer at all.

## Two budget axes

`transfer.py` sweeps `n` along two different axes, and both are recorded:

- **`map_budget`** — source probe fixed on the full pool, map fit on `n`
  unlabeled pairs. `A` is therefore constant across `n`; only `B` moves.
- **`probe_budget`** — source probe fit on `n` labeled source examples, map
  fixed on the full pool. Both `A` and `B` move with `n`.

## Caveats

- No three-circle layout can match all seven region areas at once.
  `matplotlib_venn` matches the three set sizes and three pairwise
  intersections exactly; the triple region follows from that geometry, so a
  small `ABC` sliver is not to scale. The printed counts are always exact.
- Circle sizes vary between subplots, so areas are not comparable across cells
  by eye — compare the numbers.
- Predictions use the `scores > 0` threshold, matching the `error_rate` metric
  in `transfer.py`.
