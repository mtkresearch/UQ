"""A2 -- SE-vs-truth probe orthogonality (checklist item A2 / task 24).

Confronts *Geometries of Truth* (2506.08572), which reports that **truth/correctness**
probe directions are near-orthogonal across tasks (cosine ~0, they don't transfer).
Our cross-scale + cross-dataset transfer works, so the reframe is: we probe
*uncertainty*, not *truth*, and uncertainty geometry is more shared. This script
tests that directly on the SAME 8B model:

  Prediction: SE probe directions are LESS orthogonal (higher pairwise cosine)
  across trivia_qa/squad/nq than correctness probe directions.

Method (fair cross-dataset direction comparison):
  * Pick a single fixed layer L* -- the one maximizing mean held-out AUROC across
    the three datasets (chosen separately for SE and for correctness). Comparing
    directions only makes sense in one coordinate system / one layer.
  * Fit a SHARED StandardScaler on the pooled activations of all datasets at L*,
    so every probe's coefficients live in the same standardized basis and cosine
    is meaningful.
  * Fit a standardized logistic probe per dataset; take the coefficient vector as
    the concept direction.
  * Report the 3x3 pairwise cosine matrix for SE and for correctness, plus a
    top-k |coef| feature-support Jaccard (the "sparse-support overlap" proxy).

Cross-reference: is our transfer strongest on the TriviaQA-NQ pair (their aligned
0.73 pair)? We print the cross_dataset transfer AUROCs alongside the cosines.
"""
import argparse
from sep.uncertainty.utils.config import apply_yaml_config
import json
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sep.transfer.cka import load_hidden
from sep.transfer.transfer import best_split, binarize, load_entropy


def load_labels(gen_path):
    with open(gen_path, "rb") as f:
        gens = pickle.load(f)
    ids = list(gens.keys())
    ents = load_entropy(gen_path)
    assert len(ents) == len(ids)
    y_se = binarize(ents, best_split(ents))
    acc = np.asarray(
        [float(gens[k]["most_likely_answer"]["accuracy"]) for k in ids],
        dtype=np.float64,
    )
    y_err = (acc < 0.5).astype(np.int64)  # 1 = incorrect
    return ids, y_se, y_err


def layer_auroc(Xz, y, tr, te):
    clf = LogisticRegression(max_iter=1000).fit(Xz[tr], y[tr])
    try:
        return roc_auc_score(y[te], clf.predict_proba(Xz[te])[:, 1])
    except ValueError:
        return 0.5


def pick_layer(Hs, ys, seed=0, scaler_frac=0.7):
    """Layer maximizing mean held-out AUROC across datasets (per-dataset z-score)."""
    n_layers = Hs[0].shape[0]
    best_layer, best_mean = 0, -np.inf
    per_layer = []
    for L in range(n_layers):
        aucs = []
        for H, y in zip(Hs, ys):
            n = H.shape[1]
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n)
            tr, te = idx[: int(scaler_frac * n)], idx[int(scaler_frac * n):]
            X = H[L].astype(np.float64)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xz = (X - mu) / sd
            aucs.append(layer_auroc(Xz, y, tr, te))
        m = float(np.mean(aucs))
        per_layer.append((L, m, [round(a, 3) for a in aucs]))
        if m > best_mean:
            best_mean, best_layer = m, L
    return best_layer, best_mean, per_layer


def shared_directions(Hs, ys, L, seed=0, scaler_frac=0.7):
    """Fit probes at layer L in a SHARED standardized basis.

    Returns (unit_dirs, funcmat) where funcmat[i][j] is the AUROC of applying
    dataset-i's probe (direction + intercept) to dataset-j's HELD-OUT data --
    the *functional* analog of cosine: does i's geometry retain usable
    discriminative power on j? This is the metric that matters (a low-dim shared
    component can transfer even when full-vector cosine ~ 0).
    """
    pooled = np.concatenate([H[L].astype(np.float64) for H in Hs], axis=0)
    mu, sd = pooled.mean(0), pooled.std(0) + 1e-6
    dirs, clfs, splits = [], [], []
    for H, y in zip(Hs, ys):
        n = H.shape[1]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        tr, te = idx[: int(scaler_frac * n)], idx[int(scaler_frac * n):]
        Xz = (H[L].astype(np.float64) - mu) / sd
        clf = LogisticRegression(max_iter=1000).fit(Xz[tr], y[tr])
        a = clf.coef_.ravel()
        dirs.append(a / (np.linalg.norm(a) + 1e-12))
        clfs.append(clf)
        splits.append((Xz, te))
    n_ds = len(Hs)
    func = np.zeros((n_ds, n_ds))
    for i in range(n_ds):
        for j in range(n_ds):
            Xz_j, te_j = splits[j]
            score = clfs[i].decision_function(Xz_j[te_j])
            yj = ys[j][te_j]
            try:
                func[i, j] = roc_auc_score(yj, score)
            except ValueError:
                func[i, j] = 0.5
    return np.stack(dirs), func


def cosine_matrix(dirs):
    return dirs @ dirs.T


def topk_jaccard(dirs, k_frac=0.05):
    """Pairwise Jaccard of the top-k |coef| feature supports."""
    d = dirs.shape[1]
    k = max(1, int(k_frac * d))
    supports = [set(np.argsort(-np.abs(v))[:k].tolist()) for v in dirs]
    n = len(supports)
    J = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(supports[i] & supports[j])
            union = len(supports[i] | supports[j])
            J[i, j] = J[j, i] = inter / union if union else 0.0
    return J, k


def offdiag_mean(M):
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(M[mask].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--token", default="slt", choices=["slt", "tbg"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_frac", type=float, default=0.05)
    ap.add_argument("--cross_dataset_json", default=None,
                    help="optional cross_dataset_slt.json for transfer cross-ref")
    ap.add_argument("--out", required=True)
    args = apply_yaml_config(ap)
    assert len(args.gens) == len(args.names)

    Hs, y_se, y_err = [], [], []
    for name, gp in zip(args.names, args.gens):
        ids, yse, yerr = load_labels(gp)
        H, ids_h = load_hidden(gp, args.token)
        assert ids_h == ids, f"{name}: hidden ids not aligned"
        Hs.append(H); y_se.append(yse); y_err.append(yerr)
        print(f"[{name}] n={H.shape[1]} layers={H.shape[0]} "
              f"SE base={yse.mean():.3f} err={yerr.mean():.3f}")

    L_se, mean_se, _ = pick_layer(Hs, y_se, seed=args.seed)
    L_err, mean_err, _ = pick_layer(Hs, y_err, seed=args.seed)
    print(f"\nchosen SE layer L{L_se} (mean AUROC {mean_se:.3f}); "
          f"correctness layer L{L_err} (mean AUROC {mean_err:.3f})")

    se_dirs, se_func = shared_directions(Hs, y_se, L_se, seed=args.seed)
    err_dirs, err_func = shared_directions(Hs, y_err, L_err, seed=args.seed)

    C_se = cosine_matrix(se_dirs)
    C_err = cosine_matrix(err_dirs)
    J_se, k = topk_jaccard(se_dirs, args.k_frac)
    J_err, _ = topk_jaccard(err_dirs, args.k_frac)

    names = args.names

    def mat_to_dict(M):
        return {a: {b: float(M[i, j]) for j, b in enumerate(names)}
                for i, a in enumerate(names)}

    out = {
        "token": args.token,
        "names": names,
        "layer_se": int(L_se), "layer_err": int(L_err),
        "mean_auroc_se": mean_se, "mean_auroc_err": mean_err,
        "topk": {"k_frac": args.k_frac, "k": int(k)},
        "cosine_se": mat_to_dict(C_se),
        "cosine_err": mat_to_dict(C_err),
        "jaccard_se": mat_to_dict(J_se),
        "jaccard_err": mat_to_dict(J_err),
        "func_se": mat_to_dict(se_func),
        "func_err": mat_to_dict(err_func),
        "offdiag_mean_cosine_se": offdiag_mean(C_se),
        "offdiag_mean_cosine_err": offdiag_mean(C_err),
        "offdiag_mean_jaccard_se": offdiag_mean(J_se),
        "offdiag_mean_jaccard_err": offdiag_mean(J_err),
        "offdiag_mean_func_se": offdiag_mean(se_func),
        "offdiag_mean_func_err": offdiag_mean(err_func),
        "diag_mean_func_se": float(np.diag(se_func).mean()),
        "diag_mean_func_err": float(np.diag(err_func).mean()),
    }

    print("\n=== SE direction cosine matrix ===")
    print("        " + "  ".join(f"{n:>10}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>8}  " + "  ".join(f"{C_se[i,j]:10.3f}" for j in range(len(names))))
    print(f"off-diag mean cosine (SE):  {out['offdiag_mean_cosine_se']:.3f}")

    print("\n=== correctness direction cosine matrix ===")
    print("        " + "  ".join(f"{n:>10}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>8}  " + "  ".join(f"{C_err[i,j]:10.3f}" for j in range(len(names))))
    print(f"off-diag mean cosine (correctness):  {out['offdiag_mean_cosine_err']:.3f}")

    print(f"\ntop-{int(k)}-feature Jaccard off-diag: "
          f"SE={out['offdiag_mean_jaccard_se']:.3f}  "
          f"correctness={out['offdiag_mean_jaccard_err']:.3f}")

    print("\n=== FUNCTIONAL overlap: apply row-dataset probe to col-dataset held-out ===")
    print("SE (diag = own-data AUROC, off-diag = cross-apply):")
    print("        " + "  ".join(f"{n:>10}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>8}  " + "  ".join(f"{se_func[i,j]:10.3f}" for j in range(len(names))))
    print(f"SE  functional: diag={out['diag_mean_func_se']:.3f}  "
          f"off-diag={out['offdiag_mean_func_se']:.3f}")
    print("correctness:")
    for i, n in enumerate(names):
        print(f"{n:>8}  " + "  ".join(f"{err_func[i,j]:10.3f}" for j in range(len(names))))
    print(f"ERR functional: diag={out['diag_mean_func_err']:.3f}  "
          f"off-diag={out['offdiag_mean_func_err']:.3f}")

    cos_verdict = ("SE LESS orthogonal (cosine)"
                   if out["offdiag_mean_cosine_se"] > out["offdiag_mean_cosine_err"]
                   else "cosine near-identical/worse for SE")
    func_verdict = ("SE geometry MORE functionally shared than correctness"
                    if out["offdiag_mean_func_se"] > out["offdiag_mean_func_err"]
                    else "SE geometry NOT more functionally shared than correctness")
    out["verdict_cosine"] = cos_verdict
    out["verdict_functional"] = func_verdict
    print(f"\nVERDICT (cosine):     {cos_verdict}")
    print(f"VERDICT (functional): {func_verdict}")

    if args.cross_dataset_json and os.path.exists(args.cross_dataset_json):
        with open(args.cross_dataset_json) as f:
            cd = json.load(f)
        out["transfer_cross_ref"] = cd.get("matrix")
        print("\n=== cross-ref: is transfer strongest on TriviaQA-NQ? ===")
        mat = cd["matrix"]
        pairs = []
        for a in mat:
            for b in mat[a]:
                if a != b:
                    pairs.append((f"{a}->{b}", mat[a][b]))
        pairs.sort(key=lambda x: -x[1])
        for name, v in pairs:
            print(f"  {name:>22}: {v:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
