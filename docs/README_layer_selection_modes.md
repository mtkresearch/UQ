# Target-layer selection for SE-probe transfer: four modes

Setting: a semantic-entropy probe is trained on a **source** model at its best layer,
then transferred to a **target** model through a ridge alignment map fitted on the
target's hidden states. The open question this document answers is: **which target
layer do we align into, and how many target SE labels does that choice cost?**

Everything below is at `n = 1500` source-probe training rows, ridge alignment with
`alpha = 1e4`, scored on the held-out **500-row phase-3 eval set**. Grid: 3 eval
datasets x 3 alignment datasets x 21 model pairs = **189 configurations**.

The 500/1500 eval/selection split is a function of the dataset alone: **every model on
a dataset holds out the same 500 questions**, so the rows a target layer is selected on
are disjoint from the rows every cell is scored on, for both the source and the target
side of a pair. All numbers below come from a single run (`transfer_v4`) under
normalisation convention D, so cells are comparable one by one and not merely on
average.

All four modes are complete (189 cells each).

## The four modes

| mode | code name | target SE labels needed | how the target layer is chosen | granularity of the choice |
|---|---|---|---|---|
| best-to-best | `b2b` (no filename infix) | **1500** | layer with max AUROC on the full 1500-label selection split | per (eval_ds, target model) |
| best-to-best-sub | `bbs` | **150** | same search, but on a 150-label subsample with cross-validation | per (eval_ds, target model) |
| best-to-last | `btl` | **0** | always the last layer | fixed |
| best-to-align | `b2a` | **0** | fit one alignment per candidate target layer; pick the layer minimising held-out `\|\|R_L(h_t) - h_s\|\|^2` | per (source model, target model, eval_ds, align_ds) |

`b2b` is the oracle-ish upper reference (it spends the label budget we are trying to
avoid). `btl` is the free-but-dumb baseline. `bbs` and `b2a` are the two candidate
cheap methods; `b2a` is the only one that needs **zero** target labels for the layer
choice.

Note `b2a` differs structurally from the others: because the criterion is alignment
reconstruction error, the chosen layer depends on *which source model and which
alignment dataset* you are aligning from. The other three modes have one layer per
(eval_ds, target model).

## Result 1 — transferred AUROC @ n=1500, 500-row eval set

Each cell is the mean over the 21 model pairs.

| eval_ds | align_ds | | b2b | btl | bbs | b2a |
|---|---|---|---|---|---|---|
| nq | nq | same | 0.7862 | 0.7339 | 0.7830 | 0.7762 |
| nq | squad | cross | 0.7597 | 0.6800 | 0.7504 | 0.7456 |
| nq | trivia_qa | cross | 0.7093 | 0.6301 | 0.6986 | 0.6905 |
| squad | nq | cross | 0.7896 | 0.7388 | 0.7829 | 0.7846 |
| squad | squad | same | 0.8216 | 0.7820 | 0.8138 | 0.8175 |
| squad | trivia_qa | cross | 0.7493 | 0.6652 | 0.7331 | 0.7413 |
| trivia_qa | nq | cross | 0.7716 | 0.6994 | 0.7732 | 0.7651 |
| trivia_qa | squad | cross | 0.7782 | 0.6857 | 0.7781 | 0.7609 |
| trivia_qa | trivia_qa | same | 0.8251 | 0.7864 | 0.8242 | 0.8228 |
| **all 189** | | | **0.7767** | **0.7113** | **0.7708** | **0.7672** |
| **same-align only (63)** | | | **0.8110** | **0.7674** | **0.8070** | **0.8055** |
| **cross-align only (126)** | | | **0.7596** | **0.6832** | **0.7527** | **0.7480** |

Per-configuration deltas over the 189 configs:

| delta | mean | median | p5 | p95 | win / tie / loss |
|---|---|---|---|---|---|
| bbs − btl | **+0.0596** | +0.0568 | +0.0115 | +0.1178 | 188 / 0 / 1 |
| b2b − btl | **+0.0655** | +0.0608 | +0.0206 | +0.1235 | 188 / 0 / 1 |
| bbs − b2b | −0.0059 | −0.0015 | −0.0385 | +0.0182 | 60 / 27 / 102 |
| b2a − btl | **+0.0559** | +0.0542 | +0.0088 | +0.1072 | 183 / 1 / 5 |
| b2a − b2b | −0.0096 | −0.0031 | −0.0478 | +0.0153 | 49 / 31 / 109 |
| b2a − bbs | −0.0037 | 0.0000 | −0.0458 | +0.0312 | 82 / 26 / 81 |



## Result 2 — how much do the chosen layers actually differ?

| eval_ds | target model | #layers | L_b2b | L_bbs | L_btl | L_b2a (same-align; distinct layers over source models) |
|---|---|---|---|---|---|---|
| nq | gemma-4-12b | 49 | 30 | 31 | 48 | {33, 35} |
| nq | llama-3.1-8b | 33 | 12 | 14 | 32 | {14, 16, 18} |
| nq | mistral-nemo | 41 | 18 | 20 | 40 | {18, 20, 21} |
| nq | phi-4 | 41 | 24 | 15 | 40 | {17, 19, 23} |
| nq | qwen3-8b | 37 | 21 | 22 | 36 | {21, 24, 28} |
| squad | gemma-4-12b | 49 | 28 | 34 | 48 | {33, 35} |
| squad | llama-3.1-8b | 33 | 13 | 30 | 32 | {15, 16, 18} |
| squad | mistral-nemo | 41 | 18 | 17 | 40 | {18, 21} |
| squad | phi-4 | 41 | 22 | 24 | 40 | {19, 21, 23, 24} |
| squad | qwen3-8b | 37 | 24 | 20 | 36 | {21, 24} |
| trivia_qa | gemma-4-12b | 49 | 31 | 31 | 48 | {33, 35} |
| trivia_qa | llama-3.1-8b | 33 | 15 | 15 | 32 | {15, 16, 26} |
| trivia_qa | mistral-nemo | 41 | 18 | 19 | 40 | {18, 19, 21} |
| trivia_qa | phi-4 | 41 | 21 | 22 | 40 | {22, 23} |
| trivia_qa | qwen3-8b | 37 | 23 | 19 | 36 | {21, 22, 24} |

`L_b2a` shows the set of layers chosen when align_ds = eval_ds, across the 4 source
models. A singleton means all source models agreed on the same layer.

**Reading:**

- Both label-based searches land in the **middle** of the stack (llama 12–15,
  qwen3 19–24, gemma 28–34), never near the last layer — which is what the +0.06 over
  `btl` is buying.
- `b2b` and `bbs` agree to within a couple of layers in 10 of 15 (eval_ds, target)
  cells, and where they diverge the divergence is large: squad/llama 13 vs 30,
  nq/phi-4 24 vs 15, squad/gemma 28 vs 34. The AUROC-vs-layer curve is flat enough
  near the top that those jumps cost little — the median native-AUROC gap between the
  1500-label and the 150-label layer is 0.013, worst case 0.046 — which is why cheap
  selection works at all.
- The two divergent squad cells are also where `bbs` gives up most of its transferred
  AUROC, so the flatness argument has limits: a 150-label search can miss a basin
  rather than just wobble inside one.

## Where the numbers come from

No hidden states are re-loaded; everything is read from stored artifacts.

| quantity | file |
|---|---|
| L_b2b, #layers | `_probes_shared/probes/<eval_ds>/<model>.pkl` → `best_layer`, `layer_aucs` |
| L_bbs | `transfer_v4_bbs/probes/<eval_ds>/<model>.pkl` → `sub_best_layer` |
| L_b2a | `transfer_v4_b2a/tgt_layers/<eval_ds>/<pair>/<align_ds>_b2a.json` → `tgt_best_layer` |
| transferred AUROC | `<root>/results/<eval_ds>/<pair>/probe_grid<sfx>_align_<align_ds>_ridge_a1e4.json` → `curveB_ridge` at `n_grid == 1500` |

Everything is under `$SEP_SCRATCH/transfer_v4/`. Result
roots: `b2b` = `transfer_v4_b2b`, `btl` = `transfer_v4_btl`, `bbs` = `transfer_v4_bbs`,
`b2a` = `transfer_v4_b2a`; the four modes share one phase-1 probe set in
`_probes_shared/probes` (`bbs` holds a copy of it, since it writes `sub_best_layer`
back). Filename infixes: `b2b` none, `btl` `_btl`, `bbs` `_bbs`, `b2a` `_b2a`.
Run scripts: `slurm/run_transfer_v4_all.sh` (driver) and
`slurm/run_transfer_v4_ridge.sh` (one mode).

Companion diagnostics: `sub_layer_report.py` (bbs) and `align_layer_report.py` (b2a)
report per-config `regret` against the full-budget layer — but on the 450-row
*selection* split, not the 500-row eval set, so use them as a fast proxy only.
