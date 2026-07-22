"""Q7 -- label robustness: do conclusions survive a different uncertainty target?

Everything so far binarizes `cluster_assignment_entropy` (CAE) -- the discrete
entropy of the semantic-cluster assignment (only ~56 unique values on 2000
examples). Q7 asks whether H1 (transfer) survives when the *target label* is a
different, richer uncertainty estimator. The natural alternate is the continuous
`semantic_entropy` (SE): the log-likelihood-weighted entropy over semantic
clusters (~1235 unique values), i.e. the quantity SEPs were originally designed
to predict.

We rerun the tuned transfer (alpha=1e4) with the label = binarized chosen measure
(best_split threshold, same as CAE) on BOTH source and target, over 5 seeds, and
compare best-N transfer AUROC + native reference to the CAE baseline. If the
transfer story is a property of the label choice it will break; if it's a property
of the shared uncertainty geometry it will survive.
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
from sep.transfer.transfer import (best_se_layer, best_split, binarize,
                                   fit_ridge_map, probe_scores, transfer_probe)


def load_measure(gen_path, key):
    unc = os.path.join(os.path.dirname(gen_path), "uncertainty_measures.pkl")
    with open(unc, "rb") as f:
        m = pickle.load(f)
    return np.asarray(m["uncertainty_measures"][key], dtype=np.float64)


def run_seed(source_gen, target_gen, token, key, seed, n_eval, alpha, n_grid):
    rng = np.random.default_rng(seed)
    Hs, ids_s = load_hidden(source_gen, token)
    Ht, ids_t = load_hidden(target_gen, token)
    assert ids_s == ids_t
    N = Hs.shape[1]

    ent_s, ent_t = load_measure(source_gen, key), load_measure(target_gen, key)
    ys = binarize(ent_s, best_split(ent_s))
    yt = binarize(ent_t, best_split(ent_t))

    perm = rng.permutation(N)
    eval_idx, pool = perm[:n_eval], perm[n_eval:]

    Ls, _ = best_se_layer(Hs[:, pool], ys[pool], seed=seed)
    Lt, _ = best_se_layer(Ht[:, pool], yt[pool], seed=seed)
    Xs, Xt = Hs[Ls].astype(np.float64), Ht[Lt].astype(np.float64)
    mu_s, sd_s = Xs[pool].mean(0), Xs[pool].std(0) + 1e-6
    mu_t, sd_t = Xt[pool].mean(0), Xt[pool].std(0) + 1e-6
    Zs, Zt = (Xs - mu_s) / sd_s, (Xt - mu_t) / sd_t
    Zt_eval, yt_eval = Zt[eval_idx], yt[eval_idx]

    src = LogisticRegression(max_iter=1000).fit(Zs[pool], ys[pool])

    grid = [n for n in n_grid if n <= len(pool)]
    ridge, native = [], []
    for n in grid:
        sub = pool[:n]
        M, b = fit_ridge_map(Zt[sub], Zs[sub], alpha=alpha)
        a_t, c_t = transfer_probe(src, M, b)
        ridge.append(float(roc_auc_score(yt_eval, probe_scores(Zt_eval, a_t, c_t))))
        if len(np.unique(yt[sub])) < 2:
            native.append(None)
        else:
            nat = LogisticRegression(max_iter=1000).fit(Zt[sub], yt[sub])
            native.append(float(roc_auc_score(yt_eval, nat.predict_proba(Zt_eval)[:, 1])))
    return {"grid": grid, "ridge": ridge, "native": native,
            "src_layer": int(Ls), "tgt_layer": int(Lt),
            "pos_rate_src": float(ys.mean()), "pos_rate_tgt": float(yt.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--token", default="slt", choices=["slt", "tbg"])
    ap.add_argument("--key", default="semantic_entropy",
                    help="uncertainty_measures key to use as the label")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1e4)
    ap.add_argument("--n-grid", type=int, nargs="+",
                    default=[50, 100, 200, 400, 800, 1500])
    ap.add_argument("--out", required=True)
    args = apply_yaml_config(ap)

    all_res = {}
    for nm, sg, tg in zip(args.names, args.sources, args.targets):
        print(f"\n===== {nm}  (label={args.key}) =====")
        seed_res = [run_seed(sg, tg, args.token, args.key, s, args.n_eval,
                             args.alpha, args.n_grid) for s in args.seeds]
        grid = seed_res[0]["grid"]
        R = np.array([r["ridge"] for r in seed_res])
        A = np.array([[x if x is not None else np.nan for x in r["native"]]
                      for r in seed_res])
        bi = int(np.argmax(R.mean(0)))
        agg = {
            "key": args.key, "grid": grid,
            "ridge_mean": [float(x) for x in R.mean(0)],
            "ridge_std": [float(x) for x in R.std(0)],
            "native_mean": [None if np.isnan(x) else float(x) for x in np.nanmean(A, 0)],
            "best_n": grid[bi],
            "best_ridge_mean": float(R.mean(0)[bi]), "best_ridge_std": float(R.std(0)[bi]),
            "best_native_mean": float(np.nanmean(A, 0)[bi]),
            "at50_ridge_mean": float(R.mean(0)[0]), "at50_ridge_std": float(R.std(0)[0]),
            "src_layer": seed_res[0]["src_layer"], "tgt_layer": seed_res[0]["tgt_layer"],
            "pos_rate_tgt": seed_res[0]["pos_rate_tgt"],
        }
        all_res[nm] = agg
        print(f"  tgt pos-rate {agg['pos_rate_tgt']:.3f}  src{agg['src_layer']}->tgt{agg['tgt_layer']}")
        print(f"  ridge mean: {[f'{x:.3f}' for x in agg['ridge_mean']]}")
        print(f"  @50 {agg['at50_ridge_mean']:.3f}±{agg['at50_ridge_std']:.3f}  "
              f"best-N(n={agg['best_n']}) {agg['best_ridge_mean']:.3f}±{agg['best_ridge_std']:.3f}  "
              f"native {agg['best_native_mean']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
