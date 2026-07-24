"""Per-feature correlation between hidden-state features and semantic uncertainty.

The single-feature view of the probe. For each hidden dimension j of the SE layer,
we ask: does that one feature, on its own, relate to the model's binarized
semantic-uncertainty label y? Three complementary per-feature measures:

  1. point-biserial correlation  corr(x_j, y) in [-1, 1] (signed): Pearson between
     the continuous feature and the 0/1 label. Positive => feature tends to be
     larger when the model is uncertain.
  2. univariate AUROC  auroc(x_j -> y) in [0, 1]: how well that single feature
     ranks certain vs uncertain (0.5 = useless). |auroc-0.5| is unsigned strength.
  3. |correlation| sorted: concentration read-off -- do a few features carry the
     signal or is it spread across many?

Run per model (source and target) so the shapes can be compared. NOTE: features
are NOT comparable dimension-by-dimension ACROSS models (dim j of Llama != dim j
of Mistral), so the cross-model comparison is on the DISTRIBUTION/concentration
(e.g. how many features exceed a correlation threshold), not per-dimension.

Features are the SE layer's hidden states, z-scored on the pool -- the exact
features the probe/transfer use. --eval-on {pool,all} chooses the sample set
(default pool: the unlabeled fitting set, consistent with transfer.py).
"""
import argparse
import json
import os

import numpy as np

from sep.transfer.cka import load_hidden
from sep.transfer.transfer import (
    load_entropy, best_split, binarize, best_se_layer,
)
from sklearn.metrics import roc_auc_score


def point_biserial(Z, y):
    """Pearson corr of each column of Z (n, d) with binary y (n,). Vectorized."""
    y = y.astype(np.float64)
    yc = y - y.mean()
    Zc = Z - Z.mean(0)
    num = Zc.T @ yc                                   # (d,)
    den = np.sqrt((Zc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12
    return num / den                                  # (d,) in [-1, 1]


def univariate_auroc(Z, y):
    """AUROC of each single feature as a predictor of y. Returns (d,)."""
    aucs = np.empty(Z.shape[1])
    for j in range(Z.shape[1]):
        try:
            aucs[j] = roc_auc_score(y, Z[:, j])
        except ValueError:
            aucs[j] = 0.5
    return aucs


def _summary(corr, auroc, k=20):
    a = np.abs(corr)
    order = np.argsort(a)[::-1]
    top = [{"feature": int(j), "corr": float(corr[j]), "auroc": float(auroc[j])}
           for j in order[:k]]
    return {
        "n_features": int(len(corr)),
        "max_abs_corr": float(a.max()),
        "mean_abs_corr": float(a.mean()),
        "median_abs_corr": float(np.median(a)),
        # concentration: how many features clear correlation thresholds.
        "n_abs_corr_ge_0.1": int((a >= 0.1).sum()),
        "n_abs_corr_ge_0.2": int((a >= 0.2).sum()),
        "n_abs_corr_ge_0.3": int((a >= 0.3).sum()),
        # fraction of total corr^2 held by the top-k features (like energy).
        "top20_corr2_fraction": float(np.sort(corr ** 2)[::-1][:20].sum()
                                      / ((corr ** 2).sum() + 1e-12)),
        "max_abs_auroc_dev": float(np.abs(auroc - 0.5).max()),
        "top_features": top,
    }


def analyze_one(gen_path, token, n_eval, seed, eval_on, model_name):
    rng = np.random.default_rng(seed)
    H, ids = load_hidden(gen_path, token)             # (L, N, d)
    N = H.shape[1]
    ent = load_entropy(gen_path)
    y = binarize(ent, best_split(ent))

    perm = rng.permutation(N)
    pool = perm[n_eval:]
    idx = pool if eval_on == "pool" else np.arange(N)

    # SE layer + pool z-scoring, exactly as transfer.py.
    L, auc = best_se_layer(H[:, pool], y[pool], seed=seed)
    X = H[L].astype(np.float64)
    mu, sd = X[pool].mean(0), X[pool].std(0) + 1e-6
    Z = (X - mu) / sd

    corr = point_biserial(Z[idx], y[idx])
    auroc = univariate_auroc(Z[idx], y[idx])
    summ = _summary(corr, auroc)
    summ.update({"model": model_name, "se_layer": int(L),
                 "se_layer_probe_auroc": float(auc),
                 "n_evaluated": int(len(idx)), "pos_rate": float(y[idx].mean())})
    return corr, auroc, summ


def run(source_gen, target_gen, token, out_dir, n_eval, seed, eval_on,
        source_name, target_name):
    corr_s, auc_s, sum_s = analyze_one(
        source_gen, token, n_eval, seed, eval_on, source_name)
    corr_t, auc_t, sum_t = analyze_one(
        target_gen, token, n_eval, seed, eval_on, target_name)

    results = {"token": token, "eval_on": eval_on,
               "source": sum_s, "target": sum_t}
    os.makedirs(out_dir, exist_ok=True)
    base = f"feature_uncertainty_corr_{token}_{eval_on}"
    with open(os.path.join(out_dir, base + ".json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(out_dir, base + ".npz"),
             source_corr=corr_s, source_auroc=auc_s,
             target_corr=corr_t, target_auroc=auc_t)
    _plot(corr_s, corr_t, sum_s, sum_t, token, eval_on, out_dir, base)
    _print(results)
    print(f"\nsaved -> {out_dir}/{base}.json")
    return results


def _plot(corr_s, corr_t, sum_s, sum_t, token, eval_on, out_dir, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) histogram of per-feature signed correlation, both models.
    ax = axes[0, 0]
    ax.hist(corr_s, bins=80, alpha=0.6, color="#4C72B0", label=sum_s["model"])
    ax.hist(corr_t, bins=80, alpha=0.6, color="#DD8452", label=sum_t["model"])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("per-feature correlation with uncertainty  corr(x_j, y)")
    ax.set_ylabel("number of features")
    ax.set_title("Feature–uncertainty correlation distribution")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6)

    # (b) sorted |correlation| (concentration curve), both models.
    ax = axes[0, 1]
    for c, s, col in ((corr_s, sum_s, "#4C72B0"), (corr_t, sum_t, "#DD8452")):
        a = np.sort(np.abs(c))[::-1]
        ax.plot(np.arange(1, len(a) + 1), a, lw=1.3, color=col, label=s["model"])
    ax.set_xscale("log")
    ax.set_xlabel("feature rank (by |corr|)")
    ax.set_ylabel("|correlation with uncertainty|")
    ax.set_title("Concentration: sorted |feature–uncertainty corr|")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)

    # (c) how many features clear each |corr| threshold.
    ax = axes[1, 0]
    ths = np.linspace(0, max(np.abs(corr_s).max(), np.abs(corr_t).max()), 50)
    for c, s, col in ((corr_s, sum_s, "#4C72B0"), (corr_t, sum_t, "#DD8452")):
        counts = [(np.abs(c) >= t).sum() for t in ths]
        ax.plot(ths, counts, lw=1.3, color=col, label=s["model"])
    ax.set_yscale("log")
    ax.set_xlabel("|correlation| threshold")
    ax.set_ylabel("# features above threshold")
    ax.set_title("Feature count vs correlation threshold")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)

    # (d) top-feature strengths, side by side.
    ax = axes[1, 1]
    ts = sum_s["top_features"][:15]
    tt = sum_t["top_features"][:15]
    x = np.arange(15)
    ax.bar(x - 0.2, [abs(f["corr"]) for f in ts], 0.4,
           color="#4C72B0", label=sum_s["model"])
    ax.bar(x + 0.2, [abs(f["corr"]) for f in tt], 0.4,
           color="#DD8452", label=sum_t["model"])
    ax.set_xlabel("top-15 features (each model, by |corr|)")
    ax.set_ylabel("|correlation with uncertainty|")
    ax.set_title("Strongest single features")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6)

    fig.suptitle(f"Feature–uncertainty correlation ({token.upper()}, "
                 f"eval_on={eval_on})", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, base + ".png")
    plt.savefig(p, dpi=150)
    plt.savefig(p.replace(".png", ".pdf"))
    plt.close()
    print(f"saved figure -> {p}")


def _print(r):
    print("=" * 72)
    print(f"token={r['token']}  eval_on={r['eval_on']}")
    for side in ("source", "target"):
        s = r[side]
        print(f"\n--- {side}: {s['model']}  (SE layer {s['se_layer']}, "
              f"probe AUROC {s['se_layer_probe_auroc']:.3f}, "
              f"n={s['n_evaluated']}, pos-rate {s['pos_rate']:.3f}) ---")
        print(f"  max|corr|={s['max_abs_corr']:.3f}  mean|corr|={s['mean_abs_corr']:.3f}  "
              f"median|corr|={s['median_abs_corr']:.3f}")
        print(f"  #|corr|>=0.1: {s['n_abs_corr_ge_0.1']}   "
              f">=0.2: {s['n_abs_corr_ge_0.2']}   >=0.3: {s['n_abs_corr_ge_0.3']}   "
              f"(of {s['n_features']})")
        print(f"  top-20 corr^2 fraction = {s['top20_corr2_fraction']:.3f}  "
              f"(1 => all signal in 20 features)")
        top5 = s["top_features"][:5]
        print("  top features: " + ", ".join(
            f"#{f['feature']}(corr={f['corr']:+.3f},auc={f['auroc']:.3f})"
            for f in top5))


def main():
    p = argparse.ArgumentParser(
        description="Per-feature correlation between features and uncertainty.")
    p.add_argument("--source-gen", required=True)
    p.add_argument("--target-gen", required=True)
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--source-name", default="source")
    p.add_argument("--target-name", default="target")
    p.add_argument("--eval-on", default="pool", choices=["pool", "all"])
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run(a.source_gen, a.target_gen, a.token, a.out_dir, a.n_eval, a.seed,
        a.eval_on, a.source_name, a.target_name)


if __name__ == "__main__":
    main()
