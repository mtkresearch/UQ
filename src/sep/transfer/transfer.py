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


def run(source_gen, target_gen, token, out_dir, n_eval, n_grid, seed, alpha=1e3):
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

    # Ceiling: source probe transferred with a map fit on the FULL pool.
    grid = [n for n in n_grid if n <= len(pool)]
    results = {"src_layer": int(Ls), "tgt_layer": int(Lt),
               "src_layer_auc": float(aucs), "tgt_layer_auc": float(auct),
               "alpha": float(alpha),
               "n_grid": grid, "curveA_native": [], "curveB_ridge": [],
               "curveB_procrustes": []}

    for n in grid:
        sub = pool[:n]

        # Curve A: native target probe on n LABELED target examples.
        if len(np.unique(yt[sub])) < 2:
            results["curveA_native"].append(None)
        else:
            native = LogisticRegression(max_iter=1000).fit(Zt[sub], yt[sub])
            au = roc_auc_score(yt_eval, native.predict_proba(Zt_eval)[:, 1])
            results["curveA_native"].append(float(au))

        # Curve B: map fit on n UNLABELED pairs, source probe transferred.
        M, b = fit_ridge_map(Zt[sub], Zs[sub], alpha=alpha)
        a_t, c_t = transfer_probe(src_probe, M, b)
        au_r = roc_auc_score(yt_eval, probe_scores(Zt_eval, a_t, c_t))
        results["curveB_ridge"].append(float(au_r))

        Mp, bp = fit_procrustes_map(Zt[sub], Zs[sub])
        a_tp, c_tp = transfer_probe(src_probe, Mp, bp)
        au_p = roc_auc_score(yt_eval, probe_scores(Zt_eval, a_tp, c_tp))
        results["curveB_procrustes"].append(float(au_p))

        print(f"  n={n:4d}  A(native)={results['curveA_native'][-1]}  "
              f"B(ridge)={au_r:.3f}  B(procrustes)={au_p:.3f}")

    os.makedirs(out_dir, exist_ok=True)
    # Suffix keeps the α=1e3 baseline artifacts intact when sweeping α, and keeps
    # per-seed runs from clobbering each other (seed 0 keeps the clean α suffix).
    suffix = "" if alpha == 1e3 else f"_a{alpha:.0e}"
    if seed != 0:
        suffix += f"_s{seed}"
    with open(os.path.join(out_dir, f"transfer_{token}{suffix}.json"), "w") as f:
        json.dump(results, f, indent=2)
    _plot(results, token, out_dir, suffix=suffix)
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
    args = apply_yaml_config(p)
    run(args.source_gen, args.target_gen, args.token, args.out_dir,
        args.n_eval, args.n_grid, args.seed, alpha=args.alpha)


if __name__ == "__main__":
    main()
