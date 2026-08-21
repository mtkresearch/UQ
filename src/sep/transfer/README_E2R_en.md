# `E2-R*` is a single standard ridge regression

Covers `fit_e2_rstar_map` only ([transfer.py:196](transfer.py#L196)).

---

## 1. Standard ridge regression

Data $X \in \mathbb{R}^{n \times d}$, target $y \in \mathbb{R}^n$, weights
$\theta \in \mathbb{R}^d$, intercept $\beta \in \mathbb{R}$ (intercept unpenalised):

$$
\min_{\theta,\ \beta}\ \ \bigl\| X\theta + \beta \mathbf{1} - y \bigr\|^2
\ +\ \alpha \|\theta\|^2
\tag{1}
$$

With $\bar X = X - \operatorname{mean}(X)$ and
$\bar y = y - \operatorname{mean}(y)$, the solution is

$$
\theta = \bigl(\bar X^{\top}\bar X + \alpha I\bigr)^{-1}\bar X^{\top}\bar y,
\qquad
\beta = \operatorname{mean}(y) - \operatorname{mean}(X)^{\top}\theta
\tag{2}
$$

---

## 2. The `E2-R*` objective

| Quantity | Shape | Role |
|----------|-------|------|
| $h_t$ / $Z_t$ | $d_t$ / $n \times d_t$ | target hidden state (single point / row-stacked), input |
| $w_s$ | $d_s$ | pre-trained source probe weights, **frozen constant** |
| $\mathrm{ent}_t$ | $n$ | ground-truth continuous target entropy, supervision |
| $M$, $b$ | $d_t \times d_s$, $d_s$ | alignment map, **optimised** |

$$
\min_{M,\ b}\ \ \sum_{i=1}^{n}
\Bigl( w_s^{\top}\bigl(M^{\top} h_t^{(i)} + b\bigr) - \mathrm{ent}_t^{(i)} \Bigr)^2
\ +\ \lambda \|M\|_F^2
\tag{3}
$$

**Per-point prediction:** $\ \hat u(h_t) = w_s^{\top}\bigl(M^{\top} h_t + b\bigr)$
— map $h_t$ into the source space, read out a scalar with the frozen probe, and
make that scalar approximate the true target entropy.

---

## 3. Change of variables

$w_s$ is a constant, so it can be moved onto the $h_t$ side:
$w_s^{\top}(M^{\top}h_t) = (M w_s)^{\top}h_t$. Define

$$
a := M w_s \in \mathbb{R}^{d_t},
\qquad
\beta := w_s^{\top} b \in \mathbb{R}
\tag{4}
$$

**Per-point prediction:**

$$
\hat u(h_t) = a^{\top} h_t + \beta
\tag{5}
$$

**Per-point objective:**

$$
\min_{M,\ b}\ \ \sum_{i=1}^{n}
\bigl( a^{\top} h_t^{(i)} + \beta - \mathrm{ent}_t^{(i)} \bigr)^2
\ +\ \lambda \|M\|_F^2 ,
\qquad a = M w_s,\ \beta = w_s^{\top}b
\tag{6}
$$

The penalty is still a function of $M$. But the data term depends on $M$ only
through $a$, so the optimal $M$ is the minimum-Frobenius-norm solution of
$M w_s = a$, which is rank-1:

$$
M = \frac{a\, w_s^{\top}}{\|w_s\|^2}
\quad\Longrightarrow\quad
\|M\|_F^2 = \frac{\|a\|^2}{\|w_s\|^2}
\tag{7}
$$

**Matrix form** (row $i$ of $Z_t$ is $h_t^{(i)\top}$):

$$
\min_{a,\ \beta}\ \ \bigl\| Z_t a + \beta \mathbf{1} - \mathrm{ent}_t \bigr\|^2
\ +\ \alpha_{\text{eff}} \|a\|^2 ,
\qquad
\alpha_{\text{eff}} = \frac{\lambda}{\|w_s\|^2}
\tag{8}
$$

$$
\hat u = Z_t a + \beta \mathbf{1}
\tag{9}
$$

(8) matches (1) term by term: $X \leftrightarrow Z_t$,
$\theta \leftrightarrow a$, $y \leftrightarrow \mathrm{ent}_t$,
$\alpha \leftrightarrow \alpha_{\text{eff}}$.

---

## 4. The solution, and its dependence on $w_s$

Apply (2), with $\bar Z_t = Z_t - \mu_t$,
$\overline{\mathrm{ent}_t} = \mathrm{ent}_t - \bar u$,
$\mu_t = \operatorname{mean}(Z_t)$, $\bar u = \operatorname{mean}(\mathrm{ent}_t)$:

$$
a = \Bigl(\bar Z_t^{\top}\bar Z_t + \frac{\lambda}{\|w_s\|^2} I\Bigr)^{-1}
      \bar Z_t^{\top}\,\overline{\mathrm{ent}_t},
\qquad
\beta = \bar u - \mu_t^{\top} a
\tag{10}
$$

These are exactly [transfer.py:210](transfer.py#L210) and
[:213](transfer.py#L213); $M$ and $b$ are then lifted back via (7) and
$b = \beta w_s/\|w_s\|^2$ ([:212](transfer.py#L212), [:214](transfer.py#L214)).

**$w_s$ enters the solution only through
$\alpha_{\text{eff}} = \lambda/\|w_s\|^2$**, i.e. the effective ridge
coefficient is inversely proportional to $\|w_s\|^2$. Hence, across different
probes: if their $\|w_s\|$ is unchanged, then $\alpha_{\text{eff}}$ is unchanged
$\Rightarrow$ $a$ is unchanged $\Rightarrow$
$\beta = \bar u - \mu_t^{\top}a$ is unchanged $\Rightarrow$ the prediction (9)
is pointwise identical. (The *direction* of $w_s$ does not enter (10) at all; it
only affects the coordinate representation of $M$ and $b$.)

### But $\|w_s\|$ actually varies with $n$

Probes are fit with `LogisticRegression(max_iter=1000)`
([transfer.py:633](transfer.py#L633), defaults `penalty='l2'`, `C=1.0`), whose
objective is

$$
\min_w\ \tfrac12\|w\|^2 + C\sum_{i=1}^{n}\ell\bigl(y_i,\ w^{\top}z_i + c\bigr)
$$

The data term is a **sum, not a mean**, so the larger $n$ is, the more it
dominates the fixed $\tfrac12\|w\|^2$, and $\|w_n\|$ grows with $n$. Nothing
constrains probes fit at different $n$ to have equal norm.

### Fixed bug: R\* used to ignore the matching $w_n$ on the probe-budget axis

The probe-budget axis sweeps "how many labeled source examples $n$ were used to
train the source probe", and the loop refits the probe (`src_native`, `w_n`,
[transfer.py:633-634](transfer.py#L633-L634)).

**Before the fix**, the R\* branch reused `Me2rs_full`, fit once **outside** the
loop from the full-pool `w_s`:

```python
if "e2_rstar" in curves:
    a_te2rs_n, c_te2rs_n = transfer_probe(src_native, Me2rs_full, be2rs_full)
```

So $a$, $\beta$ were the same numbers at every $n$. `w_n` appeared only in the
final `transfer_probe` step, as the pushed-through $a_s$ — and since $M$ is
rank-1:

$$
a_t = M a_s = \kappa\, a,\quad \kappa = \frac{w_s^{\top}w_n}{\|w_s\|^2};
\qquad
\text{score} = \kappa\bigl(a^{\top}h_t + \beta\bigr) + c_n
$$

a positive rescaling plus a shift of a fixed prediction, leaving AUROC /
Spearman / Kendall invariant. **That is why the R\* AUROC curve was observed to
be a flat line** — not because the norm was constant, but because the fit was
never re-run across $n$.

The cause was a misclassification. The ridge / procrustes fits genuinely do not
involve the probe, so hoisting them out of the loop is correct; whereas
`probe_aligned` / `e2_minimised` / `e2_map` / `e2_r0` all refit inside the loop
with `w_n` ([:669](transfer.py#L669), [:679](transfer.py#L679),
[:690](transfer.py#L690), [:701](transfer.py#L701)). R\* got grouped with the
former because its data term contains only $\mathrm{ent}_t$ and no source probe
— but $w_s$ still enters the penalty via
$\alpha_{\text{eff}} = \lambda/\|w_s\|^2$.

**After the fix** ([transfer.py:713-714](transfer.py#L713-L714)) it matches its
siblings, and the hoisted pre-fit has been removed:

```python
if "e2_rstar" in curves:
    Me2rs_n, be2rs_n = fit_e2_rstar_map(Zt[pool], w_n, ent_t[pool], lam=_lam_e2)
    a_te2rs_n, c_te2rs_n = transfer_probe(src_native, Me2rs_n, be2rs_n)
```

The map-budget axis ([transfer.py:614](transfer.py#L614)) is unchanged and still
passes `w_s`: there $n$ controls the alignment/supervision sample size, and
holding `w_s` fixed is precisely what keeps $\alpha_{\text{eff}}$
$n$-independent.

### How to read the curve after the fix

By (10), `w_n` affects the solution only through
$\alpha_{\text{eff}} = \lambda/\|w_n\|^2$; its **direction does not enter**. So
the fixed probe-budget curve is no longer exactly flat, but 100% of its
variation comes from "$\|w_n\|$ grows with $n$ $\Rightarrow$ weaker
regularisation $\Rightarrow$ $a$ moves toward the OLS solution" — equivalent to
traversing a segment of a $\lambda$ sweep, and **unrelated to source probe
quality improving with label count**.

This is intrinsic to the R\* objective (it regresses $\mathrm{ent}_t$ and
carries no source-probe information), not an implementation issue. To remove the
regularisation-strength confound, hold $\alpha_{\text{eff}}$ fixed rather than
$\lambda$ (i.e. pass `lam = _lam_e2 * ||w_n||^2 / ||w_s||^2`); the curve then
becomes exactly constant, recovering its intended meaning as a horizontal
"upper bound given target labels" reference line.

> Note: the fix changes the values of `curveB_e2_rstar_src_probe_budget`; prior
> results need to be re-run.

---

## 5. The `transfer2.py` implementation

The same method is ported into [transfer2.py](transfer2.py) under the `e2_rstar`
interface (`--aligners e2_rstar`, tags like `e2_rstar_l1.0`). Three differences:

1. **Only the rank-1 factors are stored.** Phase 2 caches $(a, \beta, w)$ rather
   than the $d_t \times d_s$ matrix $M$ (~150 MB per $n$). By (7) the two are
   exactly equivalent; the map is applied with `_e2_rstar_transfer`
   ([transfer2.py:192](transfer2.py#L192)), which uses
   $\kappa = w_s^{\top}a_s/\|w_s\|^2$ to get $a_t = \kappa a$,
   $c_t = \kappa\beta + c_s$ without ever building $M$. The cache carries a
   `rank1: True` flag that Phase 3 branches on.
2. **The probe grid refits per $n$.** The alignment data is fixed at the largest
   $n$, but (10) is re-solved for each probe $w_n$ and stored as
   `probe_a_{n}` / `probe_beta_{n}` / `probe_w_{n}`
   ([transfer2.py:648-667](transfer2.py#L648-L667)) — the §4 fix.
3. **Supervision comes from the alignment dataset.** $\mathrm{ent}_t$ is the
   target entropy of the align split ([transfer2.py:596](transfer2.py#L596)),
   consistent with the semantics of the alignment-budget axis.
