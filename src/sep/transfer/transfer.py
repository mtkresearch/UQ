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


def run(source_gen, target_gen, token, out_dir, n_eval, n_grid, seed, alpha=1e3, out_suffix=""):
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
    M_full, b_full = fit_ridge_map(Zt[pool], Zs[pool], alpha=alpha)
    Mp_full, bp_full = fit_procrustes_map(Zt[pool], Zs[pool])
    Mpa_full, bpa_full = fit_probe_aligned_map(Zt[pool], Zs[pool], w_s, alpha=alpha)
    c_s = float(src_probe.intercept_[0])
    a_e2_full, c_e2_full = fit_e2_weights(Zt[pool], Zs[pool], w_s, c_s, init_alpha=alpha)

    ent_t_eval = ent_t[eval_idx]
    ent_s_eval = ent_s[eval_idx]

    def _rank(true_ent, scores):
        """Returns (spearman_rho, kendall_tau) between continuous entropy and scores."""
        rho, _ = spearmanr(true_ent, scores)
        tau, _ = kendalltau(true_ent, scores)
        return float(rho), float(tau)

    results = {"src_layer": int(Ls), "tgt_layer": int(Lt),
               "src_layer_auc": float(aucs), "tgt_layer_auc": float(auct),
               "alpha": float(alpha),
               "n_grid": grid, "curveA_native": [], "curveB_ridge": [],
               "curveB_procrustes": [], "curveB_probe_aligned": [],
               "curveB_e2_minimised": [],
               "curveC_source_native": [],
               "curveB_ridge_src_probe_budget": [],
               "curveB_procrustes_src_probe_budget": [],
               "curveB_probe_aligned_src_probe_budget": [],
               "curveB_e2_src_probe_budget": [],
               "curveA_native_spearman": [], "curveB_ridge_spearman": [],
               "curveB_procrustes_spearman": [], "curveB_probe_aligned_spearman": [],
               "curveB_e2_minimised_spearman": [], "curveC_source_native_spearman": [],
               "curveB_ridge_src_probe_budget_spearman": [],
               "curveB_procrustes_src_probe_budget_spearman": [],
               "curveB_probe_aligned_src_probe_budget_spearman": [],
               "curveB_e2_src_probe_budget_spearman": [],
               "curveA_native_kendall": [], "curveB_ridge_kendall": [],
               "curveB_procrustes_kendall": [], "curveB_probe_aligned_kendall": [],
               "curveB_e2_minimised_kendall": [], "curveC_source_native_kendall": [],
               "curveB_ridge_src_probe_budget_kendall": [],
               "curveB_procrustes_src_probe_budget_kendall": [],
               "curveB_probe_aligned_src_probe_budget_kendall": [],
               "curveB_e2_src_probe_budget_kendall": []}

    for n in grid:
        sub = pool[:n]

        # Curve A: native target probe on n LABELED target examples.
        if len(np.unique(yt[sub])) < 2:
            results["curveA_native"].append(None)
            results["curveA_native_spearman"].append(None)
            results["curveA_native_kendall"].append(None)
        else:
            native = LogisticRegression(max_iter=1000).fit(Zt[sub], yt[sub])
            au = roc_auc_score(yt_eval, native.predict_proba(Zt_eval)[:, 1])
            results["curveA_native"].append(float(au))
            rho_a, tau_a = _rank(ent_t_eval, native.predict_proba(Zt_eval)[:, 1])
            results["curveA_native_spearman"].append(rho_a)
            results["curveA_native_kendall"].append(tau_a)

        # Curve B (map budget): map fit on n UNLABELED pairs, fixed source probe transferred.
        M, b = fit_ridge_map(Zt[sub], Zs[sub], alpha=alpha)
        a_t, c_t = transfer_probe(src_probe, M, b)
        sc_r = probe_scores(Zt_eval, a_t, c_t)
        au_r = roc_auc_score(yt_eval, sc_r)
        results["curveB_ridge"].append(float(au_r))
        rho_r, tau_r = _rank(ent_t_eval, sc_r)
        results["curveB_ridge_spearman"].append(rho_r)
        results["curveB_ridge_kendall"].append(tau_r)

        Mp, bp = fit_procrustes_map(Zt[sub], Zs[sub])
        a_tp, c_tp = transfer_probe(src_probe, Mp, bp)
        sc_p = probe_scores(Zt_eval, a_tp, c_tp)
        au_p = roc_auc_score(yt_eval, sc_p)
        results["curveB_procrustes"].append(float(au_p))
        rho_p, tau_p = _rank(ent_t_eval, sc_p)
        results["curveB_procrustes_spearman"].append(rho_p)
        results["curveB_procrustes_kendall"].append(tau_p)

        Mpa, bpa = fit_probe_aligned_map(Zt[sub], Zs[sub], w_s, alpha=alpha)
        a_tpa, c_tpa = transfer_probe(src_probe, Mpa, bpa)
        sc_pa = probe_scores(Zt_eval, a_tpa, c_tpa)
        au_pa = roc_auc_score(yt_eval, sc_pa)
        results["curveB_probe_aligned"].append(float(au_pa))
        rho_pa, tau_pa = _rank(ent_t_eval, sc_pa)
        results["curveB_probe_aligned_spearman"].append(rho_pa)
        results["curveB_probe_aligned_kendall"].append(tau_pa)

        a_te2, c_te2 = fit_e2_weights(Zt[sub], Zs[sub], w_s, c_s, init_alpha=alpha)
        sc_e2 = probe_scores(Zt_eval, a_te2, c_te2)
        au_e2 = roc_auc_score(yt_eval, sc_e2)
        results["curveB_e2_minimised"].append(float(au_e2))
        rho_e2, tau_e2 = _rank(ent_t_eval, sc_e2)
        results["curveB_e2_minimised_spearman"].append(rho_e2)
        results["curveB_e2_minimised_kendall"].append(tau_e2)

        # Curve C + probe-budget variants: source probe trained on n LABELED source
        # examples, evaluated on source (Curve C) and on target via fixed maps (Curve B*).
        if len(np.unique(ys[sub])) < 2:
            results["curveC_source_native"].append(None)
            results["curveC_source_native_spearman"].append(None)
            results["curveC_source_native_kendall"].append(None)
            results["curveB_ridge_src_probe_budget"].append(None)
            results["curveB_ridge_src_probe_budget_spearman"].append(None)
            results["curveB_ridge_src_probe_budget_kendall"].append(None)
            results["curveB_procrustes_src_probe_budget"].append(None)
            results["curveB_procrustes_src_probe_budget_spearman"].append(None)
            results["curveB_procrustes_src_probe_budget_kendall"].append(None)
            results["curveB_probe_aligned_src_probe_budget"].append(None)
            results["curveB_probe_aligned_src_probe_budget_spearman"].append(None)
            results["curveB_probe_aligned_src_probe_budget_kendall"].append(None)
            results["curveB_e2_src_probe_budget"].append(None)
            results["curveB_e2_src_probe_budget_spearman"].append(None)
            results["curveB_e2_src_probe_budget_kendall"].append(None)
        else:
            src_native = LogisticRegression(max_iter=1000).fit(Zs[sub], ys[sub])
            sc_c = src_native.predict_proba(Zs_eval)[:, 1]
            au_c = roc_auc_score(ys_eval, sc_c)
            results["curveC_source_native"].append(float(au_c))
            rho_c, tau_c = _rank(ent_s_eval, sc_c)
            results["curveC_source_native_spearman"].append(rho_c)
            results["curveC_source_native_kendall"].append(tau_c)

            a_t_n, c_t_n = transfer_probe(src_native, M_full, b_full)
            sc_rn = probe_scores(Zt_eval, a_t_n, c_t_n)
            results["curveB_ridge_src_probe_budget"].append(
                float(roc_auc_score(yt_eval, sc_rn)))
            rho_rn, tau_rn = _rank(ent_t_eval, sc_rn)
            results["curveB_ridge_src_probe_budget_spearman"].append(rho_rn)
            results["curveB_ridge_src_probe_budget_kendall"].append(tau_rn)

            a_tp_n, c_tp_n = transfer_probe(src_native, Mp_full, bp_full)
            sc_pn = probe_scores(Zt_eval, a_tp_n, c_tp_n)
            results["curveB_procrustes_src_probe_budget"].append(
                float(roc_auc_score(yt_eval, sc_pn)))
            rho_pn, tau_pn = _rank(ent_t_eval, sc_pn)
            results["curveB_procrustes_src_probe_budget_spearman"].append(rho_pn)
            results["curveB_procrustes_src_probe_budget_kendall"].append(tau_pn)

            w_n = src_native.coef_.ravel()
            c_n = float(src_native.intercept_[0])
            # Refit probe-aligned for w_n so the map is oriented toward the new probe.
            # This is equivalent to ridge analytically but kept for completeness.
            Mpa_n, bpa_n = fit_probe_aligned_map(Zt[pool], Zs[pool], w_n, alpha=alpha)
            a_tpa_n, c_tpa_n = transfer_probe(src_native, Mpa_n, bpa_n)
            sc_pan = probe_scores(Zt_eval, a_tpa_n, c_tpa_n)
            results["curveB_probe_aligned_src_probe_budget"].append(
                float(roc_auc_score(yt_eval, sc_pan)))
            rho_pan, tau_pan = _rank(ent_t_eval, sc_pan)
            results["curveB_probe_aligned_src_probe_budget_spearman"].append(rho_pan)
            results["curveB_probe_aligned_src_probe_budget_kendall"].append(tau_pan)

            a_te2_n, c_te2_n = fit_e2_weights(Zt[pool], Zs[pool], w_n, c_n, init_alpha=alpha)
            sc_e2n = probe_scores(Zt_eval, a_te2_n, c_te2_n)
            results["curveB_e2_src_probe_budget"].append(
                float(roc_auc_score(yt_eval, sc_e2n)))
            rho_e2n, tau_e2n = _rank(ent_t_eval, sc_e2n)
            results["curveB_e2_src_probe_budget_spearman"].append(rho_e2n)
            results["curveB_e2_src_probe_budget_kendall"].append(tau_e2n)

        print(f"  n={n:4d}  A(native)={results['curveA_native'][-1]}  "
              f"B(ridge)={au_r:.3f}  B(procrustes)={au_p:.3f}  "
              f"B(probe-aligned)={au_pa:.3f}  B(e2-min)={au_e2:.3f}  "
              f"C(src-native)={results['curveC_source_native'][-1]}  "
              f"B(src-probe-budget)={results['curveB_ridge_src_probe_budget'][-1]}")

    os.makedirs(out_dir, exist_ok=True)
    # Suffix keeps the α=1e3 baseline artifacts intact when sweeping α, and keeps
    # per-seed runs from clobbering each other (seed 0 keeps the clean α suffix).
    suffix = "" if alpha == 1e3 else f"_a{alpha:.0e}"
    if seed != 0:
        suffix += f"_s{seed}"
    suffix += out_suffix
    with open(os.path.join(out_dir, f"transfer_{token}{suffix}.json"), "w") as f:
        json.dump(results, f, indent=2)
    _plot(results, token, out_dir, suffix=suffix)
    _plot_probe_budget(results, token, out_dir, suffix=suffix)
    for metric in ("spearman", "kendall"):
        _plot_ranking(results, token, out_dir, metric=metric, suffix=suffix)
        _plot_probe_budget_ranking(results, token, out_dir, metric=metric, suffix=suffix)
    print(f"saved transfer results -> {out_dir}")
    return results


def _plot(res, token, out_dir, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["n_grid"]
    fig, ax = plt.subplots(figsize=(7, 5))
    a = [v if v is not None else np.nan for v in res["curveA_native"]]
    ax.plot(g, a, "o-", label="Curve A: native target probe (labeled)")
    ax.plot(g, res["curveB_ridge"], "s--", label="Curve B: transfer (ridge, unlabeled)")
    ax.plot(g, res["curveB_procrustes"], "^--",
            label="Curve B: transfer (Procrustes, unlabeled)")
    if "curveB_probe_aligned" in res:
        ax.plot(g, res["curveB_probe_aligned"], "D--",
                label="Curve B: transfer (probe-aligned, unlabeled)")
    if "curveB_e2_minimised" in res:
        ax.plot(g, res["curveB_e2_minimised"], "P--",
                label="Curve B: transfer (E2-minimised, unlabeled)")
    if "curveC_source_native" in res:
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
    if "curveC_source_native" in res:
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
    p.add_argument("--out-suffix", default="",
                   help="extra string appended to output filenames (e.g. _v2)")
    args = apply_yaml_config(p)
    run(args.source_gen, args.target_gen, args.token, args.out_dir,
        args.n_eval, args.n_grid, args.seed, alpha=args.alpha,
        out_suffix=args.out_suffix)


if __name__ == "__main__":
    main()
