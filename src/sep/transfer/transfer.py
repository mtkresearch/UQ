"""Step 2: fit a cross-scale map and transfer the SE probe analytically.

Setup (fail-fast, single best layer per the research proposal's Phase 1):
  * Source model M_s (Qwen3-1.7B) and target M_t (Qwen3-8B) were run on the SAME
    validation examples (aligned ids), so hidden states pair up row-for-row.
  * A source SE probe is trained ONCE on source hidden states + source SE labels.
  * A linear map f(z_t) = z_t @ M + b (target space -> source space) is fit on an
    ALIGNMENT set of paired, *unlabeled* target examples.
  * The probe transfers in closed form: applying score = z_s @ a_s + c_s to the
    mapped features gives a_t = M @ a_s, c_t = b @ a_s + c_s -- no target labels.

We standardize (z-score) both hidden spaces before fitting anything. Step 1 (CKA)
showed raw-space ridge R^2 goes negative exactly in the SE layers because Qwen's
massive-activation dims (values ~1e3) dominate the least-squares objective;
standardizing removes that pathology.

Outputs a sample-efficiency curve: AUROC vs number of target examples used, for
  Curve A -- native target probe trained on N *labeled* target examples,
  Curve B -- source probe transferred through a map fit on N *unlabeled* pairs.
The transfer hypothesis holds if Curve B reaches useful AUROC with far fewer
labeled target examples than Curve A needs.
"""
import argparse
from sep.uncertainty.utils.config import apply_yaml_config
import json
import os
import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import kendalltau, spearmanr

from sep.transfer.cka import load_hidden


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def load_entropy(gen_path):
    """cluster_assignment_entropy lives in the sibling uncertainty_measures.pkl."""
    unc_path = os.path.join(os.path.dirname(gen_path), "uncertainty_measures.pkl")
    with open(unc_path, "rb") as f:
        measures = pickle.load(f)
    return np.asarray(
        measures["uncertainty_measures"]["cluster_assignment_entropy"],
        dtype=np.float64,
    )


def best_split(ents):
    """Threshold minimizing within-group SSE (paper Section 4)."""
    splits = np.linspace(1e-10, ents.max(), 100)
    best, best_mse = splits[0], np.inf
    for s in splits:
        low, high = ents < s, ents >= s
        lm = ents[low].mean() if low.any() else 0.0
        hm = ents[high].mean() if high.any() else 0.0
        mse = np.sum((ents[low] - lm) ** 2) + np.sum((ents[high] - hm) ** 2)
        if mse < best_mse:
            best_mse, best = mse, s
    return best


def binarize(ents, thr):
    return (ents >= thr).astype(np.int64)


def align_ids(ids_s, ids_t):
    """Index array `order` such that [ids_t[i] for i in order] == ids_s.

    Multi-GPU sharding can scramble the ROW ORDER of a run without changing the
    example-id SET (shards are merged in completion order). Reordering the target
    to the source order restores the row-for-row pairing the transfer relies on.
    Hard-fail if the sets genuinely differ: then the rows are not the same
    examples (e.g. runs from different RNG states / folders) and no reordering
    can make the transfer valid.
    """
    if ids_s == ids_t:
        return np.arange(len(ids_s))
    set_s, set_t = set(ids_s), set(ids_t)
    if set_s != set_t:
        raise ValueError(
            f"example-id SETS differ: {len(set_s & set_t)} shared, "
            f"{len(set_s - set_t)} only in source, {len(set_t - set_s)} only in "
            f"target -- runs are not on the same examples, cannot transfer.")
    pos = {k: i for i, k in enumerate(ids_t)}
    return np.array([pos[k] for k in ids_s], dtype=int)


# --------------------------------------------------------------------------- #
# Per-layer SE layer selection
# --------------------------------------------------------------------------- #
def best_se_layer(H, y, scaler_frac=0.7, seed=0):
    """Single layer whose standardized features best predict binarized SE."""
    rng = np.random.default_rng(seed)
    n = H.shape[1]
    idx = rng.permutation(n)
    tr, te = idx[: int(scaler_frac * n)], idx[int(scaler_frac * n):]
    best_layer, best_auc = 0, -np.inf
    for L in range(H.shape[0]):
        X = H[L].astype(np.float64)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xz = (X - mu) / sd
        clf = LogisticRegression(max_iter=1000).fit(Xz[tr], y[tr])
        try:
            au = roc_auc_score(y[te], clf.predict_proba(Xz[te])[:, 1])
        except ValueError:
            au = 0.5
        if au > best_auc:
            best_auc, best_layer = au, L
    return best_layer, best_auc


# --------------------------------------------------------------------------- #
# Maps: target space -> source space, f(z_t) = z_t @ M + b
# --------------------------------------------------------------------------- #
def fit_ridge_map(Zt, Zs, alpha=1e3):
    """Multi-output ridge; Zt (n, d_t) -> Zs (n, d_s). Returns (M, b)."""
    mu_t, mu_s = Zt.mean(0), Zs.mean(0)
    A = Zt - mu_t
    B = Zs - mu_s
    d = A.shape[1]
    M = np.linalg.solve(A.T @ A + alpha * np.eye(d), A.T @ B)  # (d_t, d_s)
    b = mu_s - mu_t @ M
    return M, b


def fit_procrustes_map(Zt, Zs):
    """Orthogonal Procrustes: rotation-only map, extremely sample-efficient.

    Solves min ||Zt_c @ M - Zs_c|| with M semi-orthogonal via SVD of Zt_c^T Zs_c.
    """
    mu_t, mu_s = Zt.mean(0), Zs.mean(0)
    A = Zt - mu_t
    B = Zs - mu_s
    U, _, Vt = np.linalg.svd(A.T @ B, full_matrices=False)  # (d_t,k),(k,d_s)
    M = U @ Vt  # (d_t, d_s), columns orthonormal
    b = mu_s - mu_t @ M
    return M, b


def fit_probe_aligned_map(Zt, Zs, w_s, alpha=1e3):
    """Rank-1 map that directly minimises the logit gap w_s^T(R h_t - h_s).

    Instead of matching all d_s dimensions of h_s, fits only the scalar
    w_s^T h_s from h_t via ridge regression, then lifts to a rank-1 map M
    such that M @ w_s = u (the fitted scalar-regression weights).

    This directly minimises E2 = f_s(R h_t) - f_s(h_s) by minimising the
    logit gap w_s^T(R h_t - h_s) that upper-bounds |E2| via the 1/4-Lipschitz
    property of sigmoid.
    """
    mu_t = Zt.mean(0)
    A = Zt - mu_t                             # (n, d_t) centred target feats
    target = Zs @ w_s                         # (n,) scalar: w_s^T h_s per sample
    mu_target = target.mean()
    t = target - mu_target

    d = A.shape[1]
    u = np.linalg.solve(A.T @ A + alpha * np.eye(d), A.T @ t)  # (d_t,)

    ws_sq = np.dot(w_s, w_s)
    M = np.outer(u, w_s) / ws_sq             # (d_t, d_s): M @ w_s = u exactly
    b_scalar = mu_target - mu_t @ u
    b = b_scalar * w_s / ws_sq               # (d_s,): b @ w_s = b_scalar
    return M, b


def fit_e2_r0_map(Zt, Zs, w_s, lam=1e3):
    """R_0: probe-aware alignment, min_M ||w_s^T(Mh_t-h_s)||^2 + lam*||M||^2_F.

    Keeps w_s fixed and penalises the Frobenius norm of M directly, unlike
    fit_probe_aligned_map which penalises a_t = M w_s.  KKT shows the optimal
    M is rank-1; p = M w_s solves the ridge system with alpha_eff = lam/||w_s||^2
    (n-independent, so the same lam gives consistent regularisation across the
    whole n-grid — unlike the 1/n-data-term variant where alpha_eff scales with n).
    """
    ws_sq = np.dot(w_s, w_s)
    mu_t = Zt.mean(0)
    A = Zt - mu_t
    target = Zs @ w_s
    mu_target = target.mean()
    t = target - mu_target

    d = A.shape[1]
    p = np.linalg.solve(A.T @ A + (lam / ws_sq) * np.eye(d), A.T @ t)  # (d_t,)

    M = np.outer(p, w_s) / ws_sq             # (d_t, d_s): M @ w_s = p exactly
    b_scalar = mu_target - mu_t @ p
    b = b_scalar * w_s / ws_sq
    return M, b


def fit_e2_rstar_map(Zt, w_s, u_t, lam=1e3):
    """R*: supervised probe-aware map, min_M ||w_s^T M h_t + b - u_t||^2 + lam*||M||^2_F.

    Uses labeled target entropy values u_t directly as regression targets.
    Drops the 1/n_t factor from the data term so alpha_eff = lam/||w_s||^2
    is n-independent (same reasoning as fit_e2_r0_map).
    """
    ws_sq = np.dot(w_s, w_s)
    mu_t = Zt.mean(0)
    A = Zt - mu_t
    mu_u = u_t.mean()
    t = u_t - mu_u

    d = A.shape[1]
    a = np.linalg.solve(A.T @ A + (lam / ws_sq) * np.eye(d), A.T @ t)  # (d_t,)

    M = np.outer(a, w_s) / ws_sq             # (d_t, d_s): M @ w_s = a
    b_scalar = mu_u - mu_t @ a
    b = b_scalar * w_s / ws_sq
    return M, b


def fit_e2_weights(Zt, Zs, w_s, c_s, init_alpha=1e3, max_iter=500):
    """Directly minimise E2 = f_s(R h_t) - f_s(h_s) via gradient descent.

    E2 depends on R only through a_t = R @ w_s, so we optimise (a_t, c_t):

        min_{a_t, c_t}  (1/n) sum_x ( sigma(Zt @ a_t + c_t)
                                     - sigma(Zs @ w_s + c_s) )^2

    Initialised from the ridge solution. Returns (a_t, c_t) directly.
    """
    from scipy.optimize import minimize

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    p_s = _sigmoid(Zs @ w_s + c_s)          # (n,) source probe probs, fixed targets

    # Initialise from ridge on the logit (same as probe_aligned / ridge)
    mu_t = Zt.mean(0)
    A = Zt - mu_t
    logit_s = Zs @ w_s + c_s
    t = logit_s - logit_s.mean()
    d = A.shape[1]
    a0 = np.linalg.solve(A.T @ A + init_alpha * np.eye(d), A.T @ t)
    c0 = logit_s.mean() - mu_t @ a0
    x0 = np.append(a0, c0)

    n = len(Zt)

    def loss_and_grad(x):
        a, c = x[:-1], x[-1]
        p_t = _sigmoid(Zt @ a + c)                        # (n,)
        resid = p_t - p_s                                  # (n,)
        loss = np.dot(resid, resid) / n
        # chain rule: d(sigma(z))/dz = sigma(z) * (1 - sigma(z))
        common = (2.0 / n) * resid * p_t * (1.0 - p_t)   # (n,)
        grad_a = Zt.T @ common                             # (d_t,)
        grad_c = common.sum()
        return loss, np.append(grad_a, grad_c)

    res = minimize(loss_and_grad, x0, method="L-BFGS-B", jac=True,
                   options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-7})
    return res.x[:-1], float(res.x[-1])


def fit_e2_map(Zt, Zs, w_s, c_s, lam=1e3, max_iter=500):
    """Minimise E2 by optimising the full map M ∈ R^{d_t x d_s} and bias b ∈ R^{d_s}.

    Solves:
        min_{M, b}  (1/n) sum_i ( sigma( (z_t_i @ M + b) @ w_s + c_s )
                                 - sigma( z_s_i @ w_s + c_s ) )^2
                  + lam * ||M||^2_F

    Unlike fit_e2_weights (which optimises a_t = M w_s ∈ R^{d_t} directly), this
    optimises all d_t*d_s entries of M.  The ridge penalty on M is genuinely
    different: the minimum-norm M satisfying M w_s = a_t is the rank-1 matrix
    outer(a_t, w_s) / ||w_s||^2, whose Frobenius norm equals ||a_t|| / ||w_s||,
    so the effective regularisation strength on a_t scales as 1/||w_s||^2.

    Returns (M, b): the fitted map (same interface as fit_ridge_map etc.).
    """
    from scipy.optimize import minimize

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    n, d_t = Zt.shape
    d_s = Zs.shape[1]

    p_s = _sigmoid(Zs @ w_s + c_s)    # (n,) fixed target probabilities

    # Warm-start: rank-1 M0 = outer(a0, w_s) / ||w_s||^2 is the minimum-norm M
    # satisfying M0 w_s = a0, where a0 is the ridge-on-logit solution.
    mu_t = Zt.mean(0)
    A = Zt - mu_t
    logit_s = Zs @ w_s + c_s
    t = logit_s - logit_s.mean()
    a0 = np.linalg.solve(A.T @ A + lam * np.eye(d_t), A.T @ t)
    c0_scalar = logit_s.mean() - mu_t @ a0
    ws_sq = np.dot(w_s, w_s)
    M0 = np.outer(a0, w_s) / ws_sq    # (d_t, d_s)
    b0 = c0_scalar * w_s / ws_sq      # (d_s,)
    x0 = np.append(M0.ravel(), b0)    # (d_t*d_s + d_s,)

    def loss_and_grad(x):
        M = x[:d_t * d_s].reshape(d_t, d_s)
        b = x[d_t * d_s:]                           # (d_s,)
        logit_t = (Zt @ M + b) @ w_s + c_s         # (n,)
        p_t = _sigmoid(logit_t)                     # (n,)
        resid = p_t - p_s                           # (n,)
        loss = np.dot(resid, resid) / n + lam * np.dot(M.ravel(), M.ravel())
        # chain rule: d(sigma)/dz = sigma*(1-sigma)
        common = (2.0 / n) * resid * p_t * (1.0 - p_t)          # (n,)
        # grad_M[j,k] = (Z_t^T @ common)[j] * w_s[k] + 2*lam*M[j,k]
        grad_M = np.outer(Zt.T @ common, w_s) + 2.0 * lam * M   # (d_t, d_s)
        grad_b = common.sum() * w_s                               # (d_s,)
        return loss, np.append(grad_M.ravel(), grad_b)

    res = minimize(loss_and_grad, x0, method="L-BFGS-B", jac=True,
                   options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-7})
    M_fit = res.x[:d_t * d_s].reshape(d_t, d_s)
    b_fit = res.x[d_t * d_s:]
    return M_fit, b_fit


def transfer_probe(clf, M, b):
    """Push a fitted LogisticRegression through f(z_t)=z_t@M+b analytically."""
    a_s = clf.coef_.ravel()          # (d_s,)
    c_s = float(clf.intercept_[0])
    a_t = M @ a_s                    # (d_t,)
    c_t = float(b @ a_s) + c_s
    return a_t, c_t


def probe_scores(Z, a, c):
    return Z @ a + c


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def zscore(train, full):
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return (full - mu) / sd


ALL_CURVES  = {"target_probe", "source_probe", "ridge", "procrustes", "probe_aligned", "e2_minimised", "e2_map", "e2_r0", "e2_rstar"}
ALL_METRICS = {"auroc", "spearman", "kendall", "error_rate"}


def run(source_gen, target_gen, token, out_dir, n_eval, n_grid, seed, alpha=1e3,
        lam_e2_map=None, curves=None, metrics=None, out_suffix="", save_venn=False):
    curves  = set(curves)  if curves  is not None else set(ALL_CURVES)
    metrics = set(metrics) if metrics is not None else set(ALL_METRICS)
    unknown = (curves - ALL_CURVES) | (metrics - ALL_METRICS)
    if unknown:
        raise ValueError(f"Unknown curves/metrics: {unknown}. "
                         f"Valid curves: {ALL_CURVES}, metrics: {ALL_METRICS}")
    _probe_budget_curves = {"source_probe", "ridge", "procrustes", "probe_aligned", "e2_minimised", "e2_map", "e2_r0", "e2_rstar"}
    _run_probe_budget = bool(curves & _probe_budget_curves)

    rng = np.random.default_rng(seed)

    Hs, ids_s = load_hidden(source_gen, token)   # (Ls, N, ds)
    Ht, ids_t = load_hidden(target_gen, token)   # (Lt, N, dt)
    # Reorder target rows to the source id order (no-op when already aligned;
    # fixes shard-scrambled order; hard-fails if the id SETS genuinely differ).
    order = align_ids(ids_s, ids_t)
    Ht, ids_t = Ht[:, order], [ids_t[i] for i in order]
    N = Hs.shape[1]

    ent_s = load_entropy(source_gen)
    ent_t = load_entropy(target_gen)[order]
    ys = binarize(ent_s, best_split(ent_s))
    yt = binarize(ent_t, best_split(ent_t))
    print(f"N={N} src pos-rate={ys.mean():.3f} tgt pos-rate={yt.mean():.3f}")

    # Fixed eval set (last n_eval examples); pool is everything before it.
    perm = rng.permutation(N)
    eval_idx, pool = perm[:n_eval], perm[n_eval:]
    print(f"eval={len(eval_idx)} pool={len(pool)}")

    # Best SE layer on each model (chosen on the pool only).
    Ls, aucs = best_se_layer(Hs[:, pool], ys[pool], seed=seed)
    Lt, auct = best_se_layer(Ht[:, pool], yt[pool], seed=seed)
    print(f"best src layer {Ls} (SE AUROC {aucs:.3f}); "
          f"best tgt layer {Lt} (SE AUROC {auct:.3f})")

    Xs = Hs[Ls].astype(np.float64)   # (N, ds)
    Xt = Ht[Lt].astype(np.float64)   # (N, dt)

    # Standardize using pool statistics (unlabeled -> cheap, stable).
    mu_s, sd_s = Xs[pool].mean(0), Xs[pool].std(0) + 1e-6
    mu_t, sd_t = Xt[pool].mean(0), Xt[pool].std(0) + 1e-6
    Zs = (Xs - mu_s) / sd_s
    Zt = (Xt - mu_t) / sd_t

    # Source probe: trained ONCE on the full pool with source labels.
    src_probe = LogisticRegression(max_iter=1000).fit(Zs[pool], ys[pool])
    src_pool_auc = roc_auc_score(ys[pool], src_probe.predict_proba(Zs[pool])[:, 1])
    print(f"source probe fit on {len(pool)} examples (train AUROC {src_pool_auc:.3f})")

    Zt_eval, yt_eval = Zt[eval_idx], yt[eval_idx]
    Zs_eval, ys_eval = Zs[eval_idx], ys[eval_idx]

    grid = [n for n in n_grid if n <= len(pool)]
    w_s = src_probe.coef_.ravel()

    # Fixed full-pool maps for the probe-budget plot (x = labeled source examples).
    c_s = float(src_probe.intercept_[0])
    _lam_e2 = lam_e2_map if lam_e2_map is not None else alpha
    if _run_probe_budget and "ridge" in curves:
        M_full, b_full = fit_ridge_map(Zt[pool], Zs[pool], alpha=alpha)
    if _run_probe_budget and "procrustes" in curves:
        Mp_full, bp_full = fit_procrustes_map(Zt[pool], Zs[pool])
    if _run_probe_budget and "probe_aligned" in curves:
        Mpa_full, bpa_full = fit_probe_aligned_map(Zt[pool], Zs[pool], w_s, alpha=alpha)
    if _run_probe_budget and "e2_minimised" in curves:
        a_e2_full, c_e2_full = fit_e2_weights(Zt[pool], Zs[pool], w_s, c_s, init_alpha=alpha)
    if _run_probe_budget and "e2_map" in curves:
        Me2m_full, be2m_full = fit_e2_map(Zt[pool], Zs[pool], w_s, c_s, lam=_lam_e2)
    if _run_probe_budget and "e2_r0" in curves:
        Me2r0_full, be2r0_full = fit_e2_r0_map(Zt[pool], Zs[pool], w_s, lam=_lam_e2)
    # NB: no e2_rstar entry here -- like the other probe-aware maps it is refit
    # inside the loop with the n-sample probe w_n (w_s enters its solution via
    # alpha_eff = lam/||w_s||^2, so hoisting it out would freeze the curve).

    ent_t_eval = ent_t[eval_idx]
    ent_s_eval = ent_s[eval_idx]

    def _rank(true_ent, scores):
        """Returns (spearman_rho, kendall_tau) between continuous entropy and scores."""
        rho, _ = spearmanr(true_ent, scores)
        tau, _ = kendalltau(true_ent, scores)
        return float(rho), float(tau)

    # ---- Venn error-overlap bookkeeping (aggregate counts only) ------------ #
    # Per eval example, three binary events decompose the transferred probe's
    # error on the target:
    #   A = source probe disagrees with the source label   (source probe error)
    #   B = transfer flips the prediction vs the source probe
    #   C = source and target SE labels disagree           (label mismatch)
    #   D = transferred probe disagrees with the target label
    # Over GF(2), D = A xor B xor C exactly, i.e. D = A_only+B_only+C_only+ABC,
    # which is asserted below as a consistency check on the bookkeeping.
    venn = {"n_grid": grid, "n_eval": int(len(eval_idx)),
            "alpha": float(alpha), "lam_e2_map": float(_lam_e2),
            "label_mismatch": int((ys_eval != yt_eval).sum()),
            "total": int(len(ys_eval)),
            "map_budget": {}, "probe_budget": {},
            "tgt_probe_error": {}} if save_venn else None
    C_mask = ys_eval != yt_eval
    src_pred_full = src_probe.predict(Zs_eval) if save_venn else None

    def _venn(aligner, axis, n, src_pred, scores):
        """Record the 7 region counts for one (aligner, budget axis, n)."""
        if venn is None:
            return
        pred = (scores > 0).astype(int)
        A = src_pred != ys_eval
        B = pred != src_pred
        C = C_mask
        entry = {
            "A_only":  int(( A & ~B & ~C).sum()),
            "B_only":  int((~A &  B & ~C).sum()),
            "C_only":  int((~A & ~B &  C).sum()),
            "AB_only": int(( A &  B & ~C).sum()),
            "AC_only": int(( A & ~B &  C).sum()),
            "BC_only": int((~A &  B &  C).sum()),
            "ABC":     int(( A &  B &  C).sum()),
            "D":       int((pred != yt_eval).sum()),
            "total":   int(len(ys_eval)),
        }
        d_id = entry["A_only"] + entry["B_only"] + entry["C_only"] + entry["ABC"]
        assert d_id == entry["D"], (
            f"{aligner}/{axis}/n={n}: D identity failed: {d_id} != {entry['D']}"
        )
        venn[axis].setdefault(aligner, {})[str(n)] = entry

    # Map from curve name -> (map-budget key, probe-budget key)
    _curve_keys = {
        "target_probe":  ("curveA_native",         None),
        "source_probe":  ("curveC_source_native",   None),
        "ridge":         ("curveB_ridge",            "curveB_ridge_src_probe_budget"),
        "procrustes":    ("curveB_procrustes",       "curveB_procrustes_src_probe_budget"),
        "probe_aligned": ("curveB_probe_aligned",    "curveB_probe_aligned_src_probe_budget"),
        "e2_minimised":  ("curveB_e2_minimised",     "curveB_e2_src_probe_budget"),
        "e2_map":        ("curveB_e2_map",           "curveB_e2_map_src_probe_budget"),
        "e2_r0":         ("curveB_e2_r0",            "curveB_e2_r0_src_probe_budget"),
        "e2_rstar":      ("curveB_e2_rstar",          "curveB_e2_rstar_src_probe_budget"),
    }

    results = {"src_layer": int(Ls), "tgt_layer": int(Lt),
               "src_layer_auc": float(aucs), "tgt_layer_auc": float(auct),
               "alpha": float(alpha), "lam_e2_map": float(_lam_e2),
               "curves": sorted(curves), "metrics": sorted(metrics),
               "n_grid": grid}
    for c in curves:
        mb_key, pb_key = _curve_keys[c]
        results[mb_key] = []
        if "spearman"   in metrics: results[mb_key + "_spearman"]   = []
        if "kendall"    in metrics: results[mb_key + "_kendall"]    = []
        if "error_rate" in metrics: results[mb_key + "_error_rate"] = []
        if pb_key and _run_probe_budget:
            results[pb_key] = []
            if "spearman"   in metrics: results[pb_key + "_spearman"]   = []
            if "kendall"    in metrics: results[pb_key + "_kendall"]    = []
            if "error_rate" in metrics: results[pb_key + "_error_rate"] = []

    def _append(key, val):
        if key in results:
            results[key].append(val)

    for n in grid:
        sub = pool[:n]
        print_parts = [f"n={n:4d}"]

        # target_probe (Curve A): native target probe on n LABELED target examples.
        if "target_probe" in curves:
            if len(np.unique(yt[sub])) < 2:
                _append("curveA_native", None)
                _append("curveA_native_spearman", None)
                _append("curveA_native_kendall", None)
                _append("curveA_native_error_rate", None)
            else:
                native = LogisticRegression(max_iter=1000).fit(Zt[sub], yt[sub])
                proba_a = native.predict_proba(Zt_eval)[:, 1]
                au_a = roc_auc_score(yt_eval, proba_a)
                _append("curveA_native", float(au_a))
                rho_a, tau_a = _rank(ent_t_eval, proba_a)
                _append("curveA_native_spearman", rho_a)
                _append("curveA_native_kendall", tau_a)
                _append("curveA_native_error_rate", float(np.mean(native.predict(Zt_eval) != yt_eval)))
                if venn is not None:
                    venn["tgt_probe_error"][str(n)] = int((native.predict(Zt_eval) != yt_eval).sum())
                print_parts.append(f"A(native)={au_a:.3f}")

        # Curve B (map budget): map fit on n UNLABELED pairs, fixed source probe transferred.
        if "ridge" in curves:
            M, b = fit_ridge_map(Zt[sub], Zs[sub], alpha=alpha)
            a_t, c_t = transfer_probe(src_probe, M, b)
            sc_r = probe_scores(Zt_eval, a_t, c_t)
            au_r = roc_auc_score(yt_eval, sc_r)
            _append("curveB_ridge", float(au_r))
            rho_r, tau_r = _rank(ent_t_eval, sc_r)
            _append("curveB_ridge_spearman", rho_r)
            _append("curveB_ridge_kendall", tau_r)
            _append("curveB_ridge_error_rate", float(np.mean((sc_r > 0).astype(int) != yt_eval)))
            _venn("ridge", "map_budget", n, src_pred_full, sc_r)
            print_parts.append(f"B(ridge)={au_r:.3f}")

        if "procrustes" in curves:
            Mp, bp = fit_procrustes_map(Zt[sub], Zs[sub])
            a_tp, c_tp = transfer_probe(src_probe, Mp, bp)
            sc_p = probe_scores(Zt_eval, a_tp, c_tp)
            au_p = roc_auc_score(yt_eval, sc_p)
            _append("curveB_procrustes", float(au_p))
            rho_p, tau_p = _rank(ent_t_eval, sc_p)
            _append("curveB_procrustes_spearman", rho_p)
            _append("curveB_procrustes_kendall", tau_p)
            _append("curveB_procrustes_error_rate", float(np.mean((sc_p > 0).astype(int) != yt_eval)))
            _venn("procrustes", "map_budget", n, src_pred_full, sc_p)
            print_parts.append(f"B(procrustes)={au_p:.3f}")

        if "probe_aligned" in curves:
            Mpa, bpa = fit_probe_aligned_map(Zt[sub], Zs[sub], w_s, alpha=alpha)
            a_tpa, c_tpa = transfer_probe(src_probe, Mpa, bpa)
            sc_pa = probe_scores(Zt_eval, a_tpa, c_tpa)
            au_pa = roc_auc_score(yt_eval, sc_pa)
            _append("curveB_probe_aligned", float(au_pa))
            rho_pa, tau_pa = _rank(ent_t_eval, sc_pa)
            _append("curveB_probe_aligned_spearman", rho_pa)
            _append("curveB_probe_aligned_kendall", tau_pa)
            _append("curveB_probe_aligned_error_rate", float(np.mean((sc_pa > 0).astype(int) != yt_eval)))
            _venn("probe_aligned", "map_budget", n, src_pred_full, sc_pa)
            print_parts.append(f"B(probe-aligned)={au_pa:.3f}")

        if "e2_minimised" in curves:
            a_te2, c_te2 = fit_e2_weights(Zt[sub], Zs[sub], w_s, c_s, init_alpha=alpha)
            sc_e2 = probe_scores(Zt_eval, a_te2, c_te2)
            au_e2 = roc_auc_score(yt_eval, sc_e2)
            _append("curveB_e2_minimised", float(au_e2))
            rho_e2, tau_e2 = _rank(ent_t_eval, sc_e2)
            _append("curveB_e2_minimised_spearman", rho_e2)
            _append("curveB_e2_minimised_kendall", tau_e2)
            _append("curveB_e2_minimised_error_rate", float(np.mean((sc_e2 > 0).astype(int) != yt_eval)))
            _venn("e2_minimised", "map_budget", n, src_pred_full, sc_e2)
            print_parts.append(f"B(e2-min)={au_e2:.3f}")

        if "e2_map" in curves:
            Me2m, be2m = fit_e2_map(Zt[sub], Zs[sub], w_s, c_s, lam=_lam_e2)
            a_te2m, c_te2m = transfer_probe(src_probe, Me2m, be2m)
            sc_e2m = probe_scores(Zt_eval, a_te2m, c_te2m)
            au_e2m = roc_auc_score(yt_eval, sc_e2m)
            _append("curveB_e2_map", float(au_e2m))
            rho_e2m, tau_e2m = _rank(ent_t_eval, sc_e2m)
            _append("curveB_e2_map_spearman", rho_e2m)
            _append("curveB_e2_map_kendall", tau_e2m)
            _append("curveB_e2_map_error_rate", float(np.mean((sc_e2m > 0).astype(int) != yt_eval)))
            _venn("e2_map", "map_budget", n, src_pred_full, sc_e2m)
            print_parts.append(f"B(e2-map)={au_e2m:.3f}")

        if "e2_r0" in curves:
            Me2r0, be2r0 = fit_e2_r0_map(Zt[sub], Zs[sub], w_s, lam=_lam_e2)
            a_te2r0, c_te2r0 = transfer_probe(src_probe, Me2r0, be2r0)
            sc_e2r0 = probe_scores(Zt_eval, a_te2r0, c_te2r0)
            au_e2r0 = roc_auc_score(yt_eval, sc_e2r0)
            _append("curveB_e2_r0", float(au_e2r0))
            rho_e2r0, tau_e2r0 = _rank(ent_t_eval, sc_e2r0)
            _append("curveB_e2_r0_spearman", rho_e2r0)
            _append("curveB_e2_r0_kendall", tau_e2r0)
            _append("curveB_e2_r0_error_rate", float(np.mean((sc_e2r0 > 0).astype(int) != yt_eval)))
            _venn("e2_r0", "map_budget", n, src_pred_full, sc_e2r0)
            print_parts.append(f"B(e2-r0)={au_e2r0:.3f}")

        if "e2_rstar" in curves:
            Me2rs, be2rs = fit_e2_rstar_map(Zt[sub], w_s, ent_t[sub], lam=_lam_e2)
            a_te2rs, c_te2rs = transfer_probe(src_probe, Me2rs, be2rs)
            sc_e2rs = probe_scores(Zt_eval, a_te2rs, c_te2rs)
            au_e2rs = roc_auc_score(yt_eval, sc_e2rs)
            _append("curveB_e2_rstar", float(au_e2rs))
            rho_e2rs, tau_e2rs = _rank(ent_t_eval, sc_e2rs)
            _append("curveB_e2_rstar_spearman", rho_e2rs)
            _append("curveB_e2_rstar_kendall", tau_e2rs)
            _append("curveB_e2_rstar_error_rate", float(np.mean((sc_e2rs > 0).astype(int) != yt_eval)))
            _venn("e2_rstar", "map_budget", n, src_pred_full, sc_e2rs)
            print_parts.append(f"B(e2-rstar)={au_e2rs:.3f}")

        # source_probe (Curve C) + probe-budget variants.
        if _run_probe_budget:
            if len(np.unique(ys[sub])) < 2:
                for k in list(results):
                    if k.startswith("curveC") or k.endswith("_src_probe_budget") \
                            or "_src_probe_budget_" in k:
                        results[k].append(None)
            else:
                src_native = LogisticRegression(max_iter=1000).fit(Zs[sub], ys[sub])
                w_n = src_native.coef_.ravel()
                c_n = float(src_native.intercept_[0])
                src_pred_n = src_native.predict(Zs_eval) if venn is not None else None

                if "source_probe" in curves:
                    sc_c = src_native.predict_proba(Zs_eval)[:, 1]
                    au_c = roc_auc_score(ys_eval, sc_c)
                    _append("curveC_source_native", float(au_c))
                    rho_c, tau_c = _rank(ent_s_eval, sc_c)
                    _append("curveC_source_native_spearman", rho_c)
                    _append("curveC_source_native_kendall", tau_c)
                    _append("curveC_source_native_error_rate", float(np.mean(src_native.predict(Zs_eval) != ys_eval)))
                    print_parts.append(f"C(src-native)={au_c:.3f}")

                if "ridge" in curves:
                    a_t_n, c_t_n = transfer_probe(src_native, M_full, b_full)
                    sc_rn = probe_scores(Zt_eval, a_t_n, c_t_n)
                    _append("curveB_ridge_src_probe_budget", float(roc_auc_score(yt_eval, sc_rn)))
                    rho_rn, tau_rn = _rank(ent_t_eval, sc_rn)
                    _append("curveB_ridge_src_probe_budget_spearman", rho_rn)
                    _append("curveB_ridge_src_probe_budget_kendall", tau_rn)
                    _append("curveB_ridge_src_probe_budget_error_rate", float(np.mean((sc_rn > 0).astype(int) != yt_eval)))
                    _venn("ridge", "probe_budget", n, src_pred_n, sc_rn)

                if "procrustes" in curves:
                    a_tp_n, c_tp_n = transfer_probe(src_native, Mp_full, bp_full)
                    sc_pn = probe_scores(Zt_eval, a_tp_n, c_tp_n)
                    _append("curveB_procrustes_src_probe_budget", float(roc_auc_score(yt_eval, sc_pn)))
                    rho_pn, tau_pn = _rank(ent_t_eval, sc_pn)
                    _append("curveB_procrustes_src_probe_budget_spearman", rho_pn)
                    _append("curveB_procrustes_src_probe_budget_kendall", tau_pn)
                    _append("curveB_procrustes_src_probe_budget_error_rate", float(np.mean((sc_pn > 0).astype(int) != yt_eval)))
                    _venn("procrustes", "probe_budget", n, src_pred_n, sc_pn)

                if "probe_aligned" in curves:
                    Mpa_n, bpa_n = fit_probe_aligned_map(Zt[pool], Zs[pool], w_n, alpha=alpha)
                    a_tpa_n, c_tpa_n = transfer_probe(src_native, Mpa_n, bpa_n)
                    sc_pan = probe_scores(Zt_eval, a_tpa_n, c_tpa_n)
                    _append("curveB_probe_aligned_src_probe_budget", float(roc_auc_score(yt_eval, sc_pan)))
                    rho_pan, tau_pan = _rank(ent_t_eval, sc_pan)
                    _append("curveB_probe_aligned_src_probe_budget_spearman", rho_pan)
                    _append("curveB_probe_aligned_src_probe_budget_kendall", tau_pan)
                    _append("curveB_probe_aligned_src_probe_budget_error_rate", float(np.mean((sc_pan > 0).astype(int) != yt_eval)))
                    _venn("probe_aligned", "probe_budget", n, src_pred_n, sc_pan)

                if "e2_minimised" in curves:
                    a_te2_n, c_te2_n = fit_e2_weights(Zt[pool], Zs[pool], w_n, c_n, init_alpha=alpha)
                    sc_e2n = probe_scores(Zt_eval, a_te2_n, c_te2_n)
                    _append("curveB_e2_src_probe_budget", float(roc_auc_score(yt_eval, sc_e2n)))
                    rho_e2n, tau_e2n = _rank(ent_t_eval, sc_e2n)
                    _append("curveB_e2_src_probe_budget_spearman", rho_e2n)
                    _append("curveB_e2_src_probe_budget_kendall", tau_e2n)
                    _append("curveB_e2_src_probe_budget_error_rate", float(np.mean((sc_e2n > 0).astype(int) != yt_eval)))
                    _venn("e2_minimised", "probe_budget", n, src_pred_n, sc_e2n)

                if "e2_map" in curves:
                    Me2m_n, be2m_n = fit_e2_map(Zt[pool], Zs[pool], w_n, c_n, lam=_lam_e2)
                    a_te2m_n, c_te2m_n = transfer_probe(src_native, Me2m_n, be2m_n)
                    sc_e2mn = probe_scores(Zt_eval, a_te2m_n, c_te2m_n)
                    _append("curveB_e2_map_src_probe_budget", float(roc_auc_score(yt_eval, sc_e2mn)))
                    rho_e2mn, tau_e2mn = _rank(ent_t_eval, sc_e2mn)
                    _append("curveB_e2_map_src_probe_budget_spearman", rho_e2mn)
                    _append("curveB_e2_map_src_probe_budget_kendall", tau_e2mn)
                    _append("curveB_e2_map_src_probe_budget_error_rate", float(np.mean((sc_e2mn > 0).astype(int) != yt_eval)))
                    _venn("e2_map", "probe_budget", n, src_pred_n, sc_e2mn)

                if "e2_r0" in curves:
                    Me2r0_n, be2r0_n = fit_e2_r0_map(Zt[pool], Zs[pool], w_n, lam=_lam_e2)
                    a_te2r0_n, c_te2r0_n = transfer_probe(src_native, Me2r0_n, be2r0_n)
                    sc_e2r0n = probe_scores(Zt_eval, a_te2r0_n, c_te2r0_n)
                    _append("curveB_e2_r0_src_probe_budget", float(roc_auc_score(yt_eval, sc_e2r0n)))
                    rho_e2r0n, tau_e2r0n = _rank(ent_t_eval, sc_e2r0n)
                    _append("curveB_e2_r0_src_probe_budget_spearman", rho_e2r0n)
                    _append("curveB_e2_r0_src_probe_budget_kendall", tau_e2r0n)
                    _append("curveB_e2_r0_src_probe_budget_error_rate", float(np.mean((sc_e2r0n > 0).astype(int) != yt_eval)))
                    _venn("e2_r0", "probe_budget", n, src_pred_n, sc_e2r0n)

                if "e2_rstar" in curves:
                    Me2rs_n, be2rs_n = fit_e2_rstar_map(Zt[pool], w_n, ent_t[pool], lam=_lam_e2)
                    a_te2rs_n, c_te2rs_n = transfer_probe(src_native, Me2rs_n, be2rs_n)
                    sc_e2rsn = probe_scores(Zt_eval, a_te2rs_n, c_te2rs_n)
                    _append("curveB_e2_rstar_src_probe_budget", float(roc_auc_score(yt_eval, sc_e2rsn)))
                    rho_e2rsn, tau_e2rsn = _rank(ent_t_eval, sc_e2rsn)
                    _append("curveB_e2_rstar_src_probe_budget_spearman", rho_e2rsn)
                    _append("curveB_e2_rstar_src_probe_budget_kendall", tau_e2rsn)
                    _append("curveB_e2_rstar_src_probe_budget_error_rate", float(np.mean((sc_e2rsn > 0).astype(int) != yt_eval)))
                    _venn("e2_rstar", "probe_budget", n, src_pred_n, sc_e2rsn)

        print("  " + "  ".join(print_parts))

    os.makedirs(out_dir, exist_ok=True)
    # Suffix keeps the α=1e3 baseline artifacts intact when sweeping α, and keeps
    # per-seed runs from clobbering each other (seed 0 keeps the clean α suffix).
    suffix = "" if alpha == 1e3 else f"_a{alpha:.0e}"
    if seed != 0:
        suffix += f"_s{seed}"
    suffix += out_suffix
    with open(os.path.join(out_dir, f"transfer_{token}{suffix}.json"), "w") as f:
        json.dump(results, f, indent=2)
    if venn is not None:
        # Separate file: keeps the transfer_*.json artifact byte-identical to runs
        # without --save-venn, so existing plotters/diffs are unaffected.
        venn_path = os.path.join(out_dir, f"venn_{token}{suffix}.json")
        with open(venn_path, "w") as f:
            json.dump(venn, f, indent=2)
        print(f"saved venn counts -> {venn_path}")
    if "auroc" in metrics:
        _plot(results, token, out_dir, suffix=suffix)
        _plot_probe_budget(results, token, out_dir, suffix=suffix)
    for metric in ("spearman", "kendall"):
        if metric in metrics:
            _plot_ranking(results, token, out_dir, metric=metric, suffix=suffix)
            _plot_probe_budget_ranking(results, token, out_dir, metric=metric, suffix=suffix)
    if "error_rate" in metrics:
        _plot_error_rate(results, token, out_dir, suffix=suffix)
        _plot_probe_budget_error_rate(results, token, out_dir, suffix=suffix)
    print(f"saved transfer results -> {out_dir}")
    return results


def _plot(res, token, out_dir, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["n_grid"]
    fig, ax = plt.subplots(figsize=(7, 5))
    if "curveA_native" in res:
        a = [v if v is not None else np.nan for v in res["curveA_native"]]
        ax.plot(g, a, "o-", label="Curve A: native target probe (labeled)")
    if "curveB_ridge" in res:
        ax.plot(g, res["curveB_ridge"], "s--", label="Curve B: transfer (ridge, unlabeled)")
    if "curveB_procrustes" in res:
        ax.plot(g, res["curveB_procrustes"], "^--",
                label="Curve B: transfer (Procrustes, unlabeled)")
    if "curveB_probe_aligned" in res:
        ax.plot(g, res["curveB_probe_aligned"], "D--",
                label="Curve B: transfer (probe-aligned, unlabeled)")
    if "curveB_e2_minimised" in res:
        ax.plot(g, res["curveB_e2_minimised"], "P--",
                label="Curve B: transfer (E2-minimised, unlabeled)")
    if "curveB_e2_map" in res:
        ax.plot(g, res["curveB_e2_map"], "X--",
                label="Curve B: transfer (E2-map, unlabeled)")
    if "curveB_e2_r0" in res:
        ax.plot(g, res["curveB_e2_r0"], "h--",
                label="Curve B: transfer (E2-R0, unlabeled)")
    if "curveB_e2_rstar" in res:
        ax.plot(g, res["curveB_e2_rstar"], "*--",
                label="Curve B: transfer (E2-R*, labeled)")
    if res.get("curveC_source_native"):
        c = [v if v is not None else np.nan for v in res["curveC_source_native"]]
        ax.plot(g, c, "v:", color="green", label="Curve C: native source probe (labeled)")
    ax.set_xscale("log")
    ax.set_xlabel("target examples used (labeled for A, unlabeled for B)")
    ax.set_ylabel("eval AUROC (target SE)")
    alpha_txt = f", α={res.get('alpha', 1e3):.0e}" if res.get("alpha", 1e3) != 1e3 else ""
    ax.set_title(f"Cross-scale SE probe transfer ({token.upper()}{alpha_txt})")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"transfer_{token}{suffix}.pdf")
    plt.savefig(path, format="pdf", dpi=200)
    plt.close()
    print(f"  saved curve -> {path}")


def _plot_probe_budget(res, token, out_dir, suffix=""):
    """Second plot: x = labeled source examples used to train the probe.

    Curve A: n labeled target examples -> native target probe (eval on target).
    Curve B: n labeled source examples -> source probe transferred through the
             fixed full-pool map (eval on target).
    Curve C: n labeled source examples -> native source probe (eval on source).
    All three share the same x-axis meaning: probe training budget.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["n_grid"]
    fig, ax = plt.subplots(figsize=(7, 5))
    if "curveA_native" in res:
        a = [v if v is not None else np.nan for v in res["curveA_native"]]
        ax.plot(g, a, "o-", color="#333333",
                label="Curve A: native target probe (n labeled target examples)")
    if "curveB_ridge_src_probe_budget" in res:
        b = [v if v is not None else np.nan for v in res["curveB_ridge_src_probe_budget"]]
        ax.plot(g, b, "s--", color="#4C72B0",
                label="Curve B: src probe + ridge map -> target")
    if "curveB_procrustes_src_probe_budget" in res:
        b = [v if v is not None else np.nan for v in res["curveB_procrustes_src_probe_budget"]]
        ax.plot(g, b, "^--", color="#DD8452",
                label="Curve B: src probe + Procrustes map -> target")
    if "curveB_probe_aligned_src_probe_budget" in res:
        b = [v if v is not None else np.nan for v in res["curveB_probe_aligned_src_probe_budget"]]
        ax.plot(g, b, "D--", color="#8172B2",
                label="Curve B: src probe + probe-aligned map -> target")
    if "curveB_e2_src_probe_budget" in res:
        b = [v if v is not None else np.nan for v in res["curveB_e2_src_probe_budget"]]
        ax.plot(g, b, "P--", color="#C44E52",
                label="Curve B: src probe + E2-minimised map -> target")
    if "curveB_e2_map_src_probe_budget" in res:
        b = [v if v is not None else np.nan for v in res["curveB_e2_map_src_probe_budget"]]
        ax.plot(g, b, "X--", color="#2CA02C",
                label="Curve B: src probe + E2-map -> target")
    if "curveB_e2_r0_src_probe_budget" in res:
        b = [v if v is not None else np.nan for v in res["curveB_e2_r0_src_probe_budget"]]
        ax.plot(g, b, "h--", color="#9467BD",
                label="Curve B: src probe + E2-R0 -> target")
    if "curveB_e2_rstar_src_probe_budget" in res:
        b = [v if v is not None else np.nan for v in res["curveB_e2_rstar_src_probe_budget"]]
        ax.plot(g, b, "*--", color="#d62728",
                label="Curve B: src probe + E2-R* -> target")
    if res.get("curveC_source_native"):
        c = [v if v is not None else np.nan for v in res["curveC_source_native"]]
        ax.plot(g, c, "v:", color="green",
                label="Curve C: native source probe (n labeled source examples)")
    ax.set_xscale("log")
    ax.set_xlabel("labeled examples used to train the probe")
    ax.set_ylabel("eval AUROC")
    alpha_txt = f", α={res.get('alpha', 1e3):.0e}" if res.get("alpha", 1e3) != 1e3 else ""
    ax.set_title(f"SE probe transfer — probe training budget ({token.upper()}{alpha_txt})")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"transfer_{token}{suffix}_probe_budget.pdf")
    plt.savefig(path, format="pdf", dpi=200)
    plt.close()
    print(f"  saved probe-budget curve -> {path}")


def _plot_ranking(res, token, out_dir, metric="spearman", suffix=""):
    """Map-budget plot with Spearman ρ or Kendall τ on the y-axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["n_grid"]
    label = "Spearman ρ" if metric == "spearman" else "Kendall τ"
    fig, ax = plt.subplots(figsize=(7, 5))
    key_a = f"curveA_native_{metric}"
    if key_a in res:
        a = [v if v is not None else np.nan for v in res[key_a]]
        ax.plot(g, a, "o-", label="Curve A: native target probe (labeled)")
    for curve, marker, lbl in [
        (f"curveB_ridge_{metric}", "s--", "Curve B: transfer (ridge, unlabeled)"),
        (f"curveB_procrustes_{metric}", "^--", "Curve B: transfer (Procrustes, unlabeled)"),
        (f"curveB_probe_aligned_{metric}", "D--", "Curve B: transfer (probe-aligned, unlabeled)"),
        (f"curveB_e2_minimised_{metric}", "P--", "Curve B: transfer (E2-minimised, unlabeled)"),
        (f"curveB_e2_map_{metric}", "X--", "Curve B: transfer (E2-map, unlabeled)"),
        (f"curveB_e2_r0_{metric}", "h--", "Curve B: transfer (E2-R0, unlabeled)"),
        (f"curveB_e2_rstar_{metric}", "*--", "Curve B: transfer (E2-R*, labeled)"),
    ]:
        if curve in res:
            ax.plot(g, res[curve], marker, label=lbl)
    key_c = f"curveC_source_native_{metric}"
    if key_c in res:
        c = [v if v is not None else np.nan for v in res[key_c]]
        ax.plot(g, c, "v:", color="green", label="Curve C: native source probe (labeled)")
    ax.set_xscale("log")
    ax.set_xlabel("target examples used (labeled for A, unlabeled for B)")
    ax.set_ylabel(f"eval {label} (vs true entropy)")
    alpha_txt = f", α={res.get('alpha', 1e3):.0e}" if res.get("alpha", 1e3) != 1e3 else ""
    ax.set_title(f"Cross-scale SE probe transfer — {label} ({token.upper()}{alpha_txt})")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"transfer_{token}{suffix}_{metric}.pdf")
    plt.savefig(path, format="pdf", dpi=200)
    plt.close()
    print(f"  saved {label} curve -> {path}")


def _plot_probe_budget_ranking(res, token, out_dir, metric="spearman", suffix=""):
    """Probe-budget plot with Spearman ρ or Kendall τ on the y-axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["n_grid"]
    label = "Spearman ρ" if metric == "spearman" else "Kendall τ"
    fig, ax = plt.subplots(figsize=(7, 5))
    key_a = f"curveA_native_{metric}"
    if key_a in res:
        a = [v if v is not None else np.nan for v in res[key_a]]
        ax.plot(g, a, "o-", color="#333333",
                label="Curve A: native target probe (n labeled target examples)")
    for curve, marker, color, lbl in [
        (f"curveB_ridge_src_probe_budget_{metric}", "s--", "#4C72B0",
         "Curve B: src probe + ridge map -> target"),
        (f"curveB_procrustes_src_probe_budget_{metric}", "^--", "#DD8452",
         "Curve B: src probe + Procrustes map -> target"),
        (f"curveB_probe_aligned_src_probe_budget_{metric}", "D--", "#8172B2",
         "Curve B: src probe + probe-aligned map -> target"),
        (f"curveB_e2_src_probe_budget_{metric}", "P--", "#C44E52",
         "Curve B: src probe + E2-minimised map -> target"),
        (f"curveB_e2_map_src_probe_budget_{metric}", "X--", "#2CA02C",
         "Curve B: src probe + E2-map -> target"),
        (f"curveB_e2_r0_src_probe_budget_{metric}", "h--", "#9467BD",
         "Curve B: src probe + E2-R0 -> target"),
        (f"curveB_e2_rstar_src_probe_budget_{metric}", "*--", "#d62728",
         "Curve B: src probe + E2-R* -> target"),
    ]:
        if curve in res:
            b = [v if v is not None else np.nan for v in res[curve]]
            ax.plot(g, b, marker, color=color, label=lbl)
    key_c = f"curveC_source_native_{metric}"
    if key_c in res:
        c = [v if v is not None else np.nan for v in res[key_c]]
        ax.plot(g, c, "v:", color="green",
                label="Curve C: native source probe (n labeled source examples)")
    ax.set_xscale("log")
    ax.set_xlabel("labeled examples used to train the probe")
    ax.set_ylabel(f"eval {label} (vs true entropy)")
    alpha_txt = f", α={res.get('alpha', 1e3):.0e}" if res.get("alpha", 1e3) != 1e3 else ""
    ax.set_title(f"SE probe transfer — probe budget — {label} ({token.upper()}{alpha_txt})")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"transfer_{token}{suffix}_probe_budget_{metric}.pdf")
    plt.savefig(path, format="pdf", dpi=200)
    plt.close()
    print(f"  saved {label} probe-budget curve -> {path}")


def _plot_error_rate(res, token, out_dir, suffix=""):
    """Map-budget plot with Pr(ẑ ≠ z) on the y-axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["n_grid"]
    fig, ax = plt.subplots(figsize=(7, 5))
    if "curveA_native_error_rate" in res:
        a = [v if v is not None else np.nan for v in res["curveA_native_error_rate"]]
        ax.plot(g, a, "o-", label="Curve A: native target probe (labeled)")
    for curve, marker, lbl in [
        ("curveB_ridge_error_rate",         "s--", "Curve B: transfer (ridge, unlabeled)"),
        ("curveB_procrustes_error_rate",     "^--", "Curve B: transfer (Procrustes, unlabeled)"),
        ("curveB_probe_aligned_error_rate",  "D--", "Curve B: transfer (probe-aligned, unlabeled)"),
        ("curveB_e2_minimised_error_rate",   "P--", "Curve B: transfer (E2-minimised, unlabeled)"),
        ("curveB_e2_map_error_rate",         "X--", "Curve B: transfer (E2-map, unlabeled)"),
        ("curveB_e2_r0_error_rate",          "h--", "Curve B: transfer (E2-R0, unlabeled)"),
        ("curveB_e2_rstar_error_rate",        "*--", "Curve B: transfer (E2-R*, labeled)"),
    ]:
        if curve in res:
            ax.plot(g, res[curve], marker, label=lbl)
    if "curveC_source_native_error_rate" in res:
        c = [v if v is not None else np.nan for v in res["curveC_source_native_error_rate"]]
        ax.plot(g, c, "v:", color="green", label="Curve C: native source probe (labeled)")
    ax.set_xscale("log")
    ax.set_xlabel("target examples used (labeled for A, unlabeled for B)")
    ax.set_ylabel("error rate  Pr(ẑ ≠ z)")
    alpha_txt = f", α={res.get('alpha', 1e3):.0e}" if res.get("alpha", 1e3) != 1e3 else ""
    ax.set_title(f"Cross-scale SE probe transfer — error rate ({token.upper()}{alpha_txt})")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"transfer_{token}{suffix}_error_rate.pdf")
    plt.savefig(path, format="pdf", dpi=200)
    plt.close()
    print(f"  saved error-rate curve -> {path}")


def _plot_probe_budget_error_rate(res, token, out_dir, suffix=""):
    """Probe-budget plot with Pr(ẑ ≠ z) on the y-axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["n_grid"]
    fig, ax = plt.subplots(figsize=(7, 5))
    if "curveA_native_error_rate" in res:
        a = [v if v is not None else np.nan for v in res["curveA_native_error_rate"]]
        ax.plot(g, a, "o-", color="#333333",
                label="Curve A: native target probe (n labeled target examples)")
    for curve, marker, color, lbl in [
        ("curveB_ridge_src_probe_budget_error_rate",        "s--", "#4C72B0",
         "Curve B: src probe + ridge map -> target"),
        ("curveB_procrustes_src_probe_budget_error_rate",   "^--", "#DD8452",
         "Curve B: src probe + Procrustes map -> target"),
        ("curveB_probe_aligned_src_probe_budget_error_rate","D--", "#8172B2",
         "Curve B: src probe + probe-aligned map -> target"),
        ("curveB_e2_src_probe_budget_error_rate",           "P--", "#C44E52",
         "Curve B: src probe + E2-minimised map -> target"),
        ("curveB_e2_map_src_probe_budget_error_rate",       "X--", "#2CA02C",
         "Curve B: src probe + E2-map -> target"),
        ("curveB_e2_r0_src_probe_budget_error_rate",        "h--", "#9467BD",
         "Curve B: src probe + E2-R0 -> target"),
        ("curveB_e2_rstar_src_probe_budget_error_rate",     "*--", "#d62728",
         "Curve B: src probe + E2-R* -> target"),
    ]:
        if curve in res:
            b = [v if v is not None else np.nan for v in res[curve]]
            ax.plot(g, b, marker, color=color, label=lbl)
    if "curveC_source_native_error_rate" in res:
        c = [v if v is not None else np.nan for v in res["curveC_source_native_error_rate"]]
        ax.plot(g, c, "v:", color="green",
                label="Curve C: native source probe (n labeled source examples)")
    ax.set_xscale("log")
    ax.set_xlabel("labeled examples used to train the probe")
    ax.set_ylabel("error rate  Pr(ẑ ≠ z)")
    alpha_txt = f", α={res.get('alpha', 1e3):.0e}" if res.get("alpha", 1e3) != 1e3 else ""
    ax.set_title(f"SE probe transfer — probe budget — error rate ({token.upper()}{alpha_txt})")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"transfer_{token}{suffix}_probe_budget_error_rate.pdf")
    plt.savefig(path, format="pdf", dpi=200)
    plt.close()
    print(f"  saved error-rate probe-budget curve -> {path}")


def main():
    p = argparse.ArgumentParser(description="Cross-scale SE probe transfer.")
    p.add_argument("--source-gen", required=True)
    p.add_argument("--target-gen", required=True)
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--n-grid", type=int, nargs="+",
                   default=[50, 100, 200, 400, 800, 1500])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1e3,
                   help="ridge-map regularization (conditioning study: 1e4 is optimal)")
    p.add_argument("--lam-e2-map", type=float, default=None,
                   help="ridge penalty lambda for fit_e2_map (defaults to --alpha if unset)")
    p.add_argument("--curves", nargs="+", default=None,
                   metavar="CURVE",
                   help=f"curves to compute and plot (default: all). "
                        f"choices: {sorted(ALL_CURVES)}")
    p.add_argument("--metrics", nargs="+", default=None,
                   metavar="METRIC",
                   help=f"metrics to plot (default: all). "
                        f"choices: {sorted(ALL_METRICS)}")
    p.add_argument("--out-suffix", default="",
                   help="extra string appended to output filenames (e.g. _v2)")
    p.add_argument("--save-venn", action="store_true",
                   help="also write venn_<token><suffix>.json with the A/B/C/D "
                        "error-overlap counts per aligner, budget axis and n "
                        "(plot with sep.transfer.plot_venn)")
    args = apply_yaml_config(p)
    run(args.source_gen, args.target_gen, args.token, args.out_dir,
        args.n_eval, args.n_grid, args.seed, alpha=args.alpha,
        lam_e2_map=args.lam_e2_map, curves=args.curves, metrics=args.metrics,
        out_suffix=args.out_suffix, save_venn=args.save_venn)


if __name__ == "__main__":
    main()
