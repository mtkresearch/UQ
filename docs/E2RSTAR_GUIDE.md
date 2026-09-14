# Testing E2-R* (Supervised Probe-Aware Transfer)

## What is E2-R*?

E2-R* learns a linear map M from target to source hidden-state space by solving:

```
min_M  sum_i [ w_s^T * M * h_t_i  -  u_t_i ]^2  +  (lam / ||w_s||^2) * ||M||^2_F
```

where:
- `h_t_i` = target model hidden states (layer SLT)
- `w_s`   = source probe (trained on source model)
- `u_t_i` = **true semantic entropy of the target model** (required — supervised)
- `lam`   = regularization hyperparameter

The closed-form rank-1 solution is `M = outer(a, w_s) / ||w_s||^2`, where `a` is the
ridge-regression solution regressing `u_t` onto `h_t`.

**Key difference from E2-R0**: the regression targets are `u_t` (target labels) instead of
`w_s^T h_s` (source probe scores). Everything else is identical.

**Key difference from E2-map**: E2-map matches sigmoid probabilities via L-BFGS.
E2-R* matches entropy values linearly — closed form, no gradient descent.

---

## What inputs does E2-R* need?

E2-R* is **supervised on the target side**. It needs a `.pkl` file of target model
generations that contains semantic entropy values `u_t`. These are computed automatically
by the SE pipeline (same `.pkl` files used for the native target probe).

Source and target `.pkl` paths are the same as for all other methods — no extra data collection.

---

## How to run the lambda sweep

```bash
cd /build_bak/UQ/UQ-transfer
bash run_e2rstar_lam_sweep.sh
```

This runs 6 pairs × 5 lambda values (`10, 100, 1000, 10000, 100000`) and writes:
```
sep_scratch/transfer/hyperparam_sweep_v3/<pair>/transfer_slt_lam<lam>_e2rstar.json
```

Each JSON contains both `auroc` and `error_rate` curves for map-budget and probe-budget axes.

To run just one pair for a quick test, extract the relevant block from the script
(the inner loop body) and run it directly.

---

## How to plot results

**All lambda values (hyperparam sweep):**
```bash
PYTHON=/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python
export PYTHONPATH=/build_bak/UQ/UQ-transfer/src:/build_bak/UQ/python_packages

$PYTHON -m sep.transfer.plot_hyperparam_sweep \
    --pairs-dir sep_scratch/transfer/hyperparam_sweep_v3 \
    --pair-names llama2_to_mistral llama32-1b_to_llama31-8b llama31-8b_to_qwen3-8b \
                 llama31-8b_to_phi4 llama31-8b_to_gemma llama31-8b_to_nemo \
    --token slt --e2rstar-lams 10 100 1000 10000 100000 \
    --metric auroc --axis map_budget \
    --out-dir sep_scratch/transfer/hyperparam_sweep_v3
```

**Best lambda per pair:**
```bash
$PYTHON -m sep.transfer.plot_best_hyperparams \
    --pairs-dir sep_scratch/transfer/hyperparam_sweep_v3 \
    --e2rstar-lams 10 100 1000 10000 100000 \
    --metric auroc --axis map_budget
```

Or run `bash run_plots_v3.sh` to produce all 8 plots at once.

---

## Known limitations and caveats

### 1. E2-R* is supervised — unfair comparison with other methods
Ridge, E2-map, and E2-R0 are **unsupervised on the target side** (they only use
unlabeled target hidden states). E2-R* uses target semantic entropy labels, so
it has access to the same information as the native target probe.
Treat it as an **oracle upper bound**, not a direct competitor to the unsupervised methods.

### 2. Probe-budget curve is flat due to a known bug
In `transfer.py` line 712, the probe-budget loop for E2-R* reuses `Me2rs_full`
(the map fitted with the full-data source probe `w_s`) instead of re-fitting M
with the budget probe `w_n`. Because `M = outer(a, w_s)/||w_s||^2` and
`transfer_probe` computes `a_t = M @ w_n = a * (w_s . w_n / ||w_s||^2)`, the
direction of `a_t` never changes — only its scale does — so AUROC is constant.
**Do not interpret the probe-budget E2-R* curve.**

### 3. Error-rate metric is unreliable for E2-R*
E2-R* outputs scores in entropy space (values ~ 0.5–1.5 nats), not logit space.
The error-rate computation thresholds at `score > 0`, which almost always predicts
"uncertain" since entropy >= 0. The resulting error rate equals the fraction of
certain examples in the eval set — a meaningless constant.
**Only trust the AUROC metric and the map-budget axis for E2-R*.**

---

## Summary: which plots to trust

| Plot | Trust E2-R*? |
|------|-------------|
| map_budget AUROC  | Yes (supervised oracle upper bound) |
| map_budget error_rate | No (threshold bug) |
| probe_budget AUROC | No (flat-line bug) |
| probe_budget error_rate | No (both bugs) |
