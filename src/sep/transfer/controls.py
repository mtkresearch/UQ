"""Items 1-4 from the research-proposal gap analysis, in one pass.

For each dataset (source 1.7B -> target 8B, aligned example ids) we sweep the
alignment-set size and, over multiple seeds, report mean +/- std AUROC for:

  maps (all transfer the SAME fixed source probe, target->source):
    * ridge        -- the workhorse affine map (item baseline)
    * procrustes   -- rotation-only
    * random       -- Gaussian map, matched Frobenius scale   (item 1, lower bound)
    * identity     -- truncate/zero-pad target to source dim  (item 1; dims differ
                       2048 vs 4096 so this is the naive no-alignment baseline)
  reference:
    * native       -- target probe trained on N *labeled* target examples (Curve A)

  metrics (both reported):
    * SE AUROC          -- predict target binarized semantic entropy   (direct target)
    * correctness AUROC -- predict target error (1 - accuracy)         (item 3, Q6)

Target-layer selection (item 2): three variants, so the reader sees how much rides
on supervised layer picking --
    * supervised   -- best target layer by target-SE AUROC (spends target labels)
    * reldepth     -- target layer at the source's best fractional depth (label-free)
    * cka          -- target layer with max CKA to the fixed source layer (label-free)

Also reports ridge map conditioning kappa(W) (proposal 4.5).
"""
import argparse
from sep.uncertainty.utils.config import apply_yaml_config
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sep.transfer.cka import linear_cka, load_hidden
from sep.transfer.transfer import (
    best_se_layer, best_split, binarize, fit_procrustes_map, fit_ridge_map,
    load_entropy, probe_scores, transfer_probe,
)


def load_accuracy(gen_path):
    """Target error rate (1 - accuracy) as the correctness label."""
    import pickle
    with open(gen_path, "rb") as f:
        gens = pickle.load(f)
    acc = np.array([g["most_likely_answer"]["accuracy"] for g in gens.values()],
                   dtype=np.int64)
    return 1 - acc  # 1 == wrong == positive class for hallucination detection


def fit_random_map(Zt, Zs, seed):
    """Gaussian map with columns scaled to match a ridge map's Frobenius norm.

    Lower bound: if this already transfers well, the 'shared subspace' reading
    collapses -- the alignment set is just leaking target structure (Q1).
    """
    rng = np.random.default_rng(seed)
    d_t, d_s = Zt.shape[1], Zs.shape[1]
    M = rng.standard_normal((d_t, d_s)) / np.sqrt(d_t)
    b = Zs.mean(0) - Zt.mean(0) @ M
    return M, b


def fit_identity_map(Zt, Zs):
    """Naive no-alignment baseline across mismatched dims (2048 vs 4096).

    Take the first d_s target coords (both standardized), no rotation. Tests
    whether *any* alignment is needed at all.
    """
    d_s = Zs.shape[1]
    d_t = Zt.shape[1]
    M = np.zeros((d_t, d_s))
    k = min(d_s, d_t)
    M[:k, :k] = np.eye(k)
    b = Zs.mean(0) - Zt.mean(0) @ M
    return M, b


def pick_target_layer(Ht, yt, pool, Hs, Ls, variant, seed):
    """Item 2: supervised vs two label-free target-layer choices."""
    if variant == "supervised":
        Lt, _ = best_se_layer(Ht[:, pool], yt[pool], seed=seed)
        return Lt
    if variant == "reldepth":
        # Match the source's best fractional depth; no target labels.
        frac = Ls / (Hs.shape[0] - 1)
        return int(round(frac * (Ht.shape[0] - 1)))
    if variant == "cka":
        # Max CKA between the fixed source layer and each target layer.
        Xs = Hs[Ls]
        ckas = [linear_cka(Xs[pool], Ht[j][pool]) for j in range(Ht.shape[0])]
        return int(np.argmax(ckas))
    raise ValueError(variant)


def zstats(X, idx):
    return X[idx].mean(0), X[idx].std(0) + 1e-6


def one_seed(Hs, Ht, ys, yt, ye, n_grid, n_eval, layer_variant, seed):
    rng = np.random.default_rng(seed)
    N = Hs.shape[1]
    perm = rng.permutation(N)
    eval_idx, pool = perm[:n_eval], perm[n_eval:]

    Ls, _ = best_se_layer(Hs[:, pool], ys[pool], seed=seed)
    Lt = pick_target_layer(Ht, yt, pool, Hs, Ls, layer_variant, seed)

    Xs, Xt = Hs[Ls].astype(np.float64), Ht[Lt].astype(np.float64)
    mu_s, sd_s = zstats(Xs, pool)
    mu_t, sd_t = zstats(Xt, pool)
    Zs, Zt = (Xs - mu_s) / sd_s, (Xt - mu_t) / sd_t

    src_probe = LogisticRegression(max_iter=1000).fit(Zs[pool], ys[pool])
    Zt_eval = Zt[eval_idx]
    yt_e, ye_e = yt[eval_idx], ye[eval_idx]

    maps = {"ridge": fit_ridge_map, "procrustes": fit_procrustes_map}
    out = {"src_layer": int(Ls), "tgt_layer": int(Lt), "kappa": {}}
    for metric in ("se", "correctness"):
        out[metric] = {m: [] for m in
                       ["native", "ridge", "procrustes", "random", "identity"]}

    for n in n_grid:
        sub = pool[:n]

        # Native (Curve A): separate probe per metric label.
        for metric, lbl in (("se", yt), ("correctness", ye)):
            if len(np.unique(lbl[sub])) < 2:
                out[metric]["native"].append(None)
            else:
                clf = LogisticRegression(max_iter=1000).fit(Zt[sub], lbl[sub])
                tgt = yt_e if metric == "se" else ye_e
                out[metric]["native"].append(
                    float(roc_auc_score(tgt, clf.predict_proba(Zt_eval)[:, 1])))

        # Transfer maps: probe is fixed (source SE); score both target metrics.
        fitted = {}
        M, b = fit_ridge_map(Zt[sub], Zs[sub])
        fitted["ridge"] = (M, b)
        out["kappa"].setdefault("ridge", []).append(float(np.linalg.cond(M)))
        Mp, bp = fit_procrustes_map(Zt[sub], Zs[sub])
        fitted["procrustes"] = (Mp, bp)
        Mr, br = fit_random_map(Zt[sub], Zs[sub], seed)
        fitted["random"] = (Mr, br)
        Mi, bi = fit_identity_map(Zt[sub], Zs[sub])
        fitted["identity"] = (Mi, bi)

        for name, (Mm, bb) in fitted.items():
            a_t, c_t = transfer_probe(src_probe, Mm, bb)
            scores = probe_scores(Zt_eval, a_t, c_t)
            out["se"][name].append(float(roc_auc_score(yt_e, scores)))
            out["correctness"][name].append(float(roc_auc_score(ye_e, scores)))

    out["kappa"] = {k: float(np.mean(v)) for k, v in out["kappa"].items()}
    return out


def aggregate(runs):
    """Mean +/- std across seeds for each (metric, map, grid-point)."""
    agg = {}
    for metric in ("se", "correctness"):
        agg[metric] = {}
        for m in ["native", "ridge", "procrustes", "random", "identity"]:
            stacked = np.array(
                [[v if v is not None else np.nan for v in r[metric][m]]
                 for r in runs], dtype=float)
            agg[metric][m] = {"mean": np.nanmean(stacked, 0).tolist(),
                              "std": np.nanstd(stacked, 0).tolist()}
    agg["kappa_ridge"] = float(np.mean([r["kappa"]["ridge"] for r in runs]))
    agg["tgt_layers"] = [r["tgt_layer"] for r in runs]
    agg["src_layers"] = [r["src_layer"] for r in runs]
    return agg


def run(source_gen, target_gen, token, out_dir, n_grid, n_eval,
        layer_variant, seeds):
    Hs, ids_s = load_hidden(source_gen, token)
    Ht, ids_t = load_hidden(target_gen, token)
    assert ids_s == ids_t
    ys = binarize(load_entropy(source_gen), best_split(load_entropy(source_gen)))
    yt = binarize(load_entropy(target_gen), best_split(load_entropy(target_gen)))
    ye = load_accuracy(target_gen)
    grid = [n for n in n_grid if n <= Hs.shape[1] - n_eval]
    print(f"layer_variant={layer_variant} seeds={seeds} grid={grid}")
    print(f"target: SE pos-rate={yt.mean():.3f} error-rate={ye.mean():.3f}")

    runs = [one_seed(Hs, Ht, ys, yt, ye, grid, n_eval, layer_variant, s)
            for s in seeds]
    agg = aggregate(runs)
    agg["n_grid"] = grid
    agg["layer_variant"] = layer_variant
    agg["seeds"] = list(seeds)

    def fmt(metric, m, i):
        return f"{agg[metric][m]['mean'][i]:.3f}±{agg[metric][m]['std'][i]:.3f}"

    print(f"\nridge kappa(W) ~ {agg['kappa_ridge']:.1f}; "
          f"src layers {agg['src_layers']} tgt layers {agg['tgt_layers']}")
    for metric in ("se", "correctness"):
        print(f"\n== {metric.upper()} AUROC (mean±std over {len(seeds)} seeds) ==")
        print(f"{'n':>5} {'native':>13} {'ridge':>13} {'procrustes':>13} "
              f"{'random':>13} {'identity':>13}")
        for i, n in enumerate(grid):
            print(f"{n:>5} {fmt(metric,'native',i):>13} {fmt(metric,'ridge',i):>13} "
                  f"{fmt(metric,'procrustes',i):>13} {fmt(metric,'random',i):>13} "
                  f"{fmt(metric,'identity',i):>13}")

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{token}_{layer_variant}"
    with open(os.path.join(out_dir, f"controls_{tag}.json"), "w") as f:
        json.dump(agg, f, indent=2)
    _plot(agg, grid, tag, out_dir)
    print(f"\nsaved -> {out_dir}")
    return agg


def _plot(agg, grid, tag, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    styles = {"native": "o-", "ridge": "s--", "procrustes": "^--",
              "random": "x:", "identity": "d:"}
    for ax, metric in zip(axes, ("se", "correctness")):
        for m, st in styles.items():
            mean = np.array(agg[metric][m]["mean"])
            std = np.array(agg[metric][m]["std"])
            ax.plot(grid, mean, st, label=m)
            ax.fill_between(grid, mean - std, mean + std, alpha=0.15)
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("target examples used")
        ax.set_ylabel(f"{metric} AUROC")
        ax.set_title(f"{metric.upper()} ({tag})")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"controls_{tag}.pdf")
    plt.savefig(path, format="pdf", dpi=200)
    plt.close()
    print(f"  saved curve -> {path}")


def main():
    p = argparse.ArgumentParser(description="Transfer controls + metrics (items 1-4).")
    p.add_argument("--source-gen", required=True)
    p.add_argument("--target-gen", required=True)
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-grid", type=int, nargs="+",
                   default=[50, 100, 200, 400, 800, 1500])
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--layer-variant", default="supervised",
                   choices=["supervised", "reldepth", "cka"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = apply_yaml_config(p)
    run(args.source_gen, args.target_gen, args.token, args.out_dir,
        args.n_grid, args.n_eval, args.layer_variant, args.seeds)


if __name__ == "__main__":
    main()
