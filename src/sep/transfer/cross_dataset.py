"""Cross-distribution test: does the cross-scale map itself generalize?

Step 2 fit and evaluated the map within a single dataset (trivia_qa alignment ->
trivia_qa eval). That tests unseen prompts but not an unseen *distribution*. Here
we freeze the ENTIRE source-side pipeline on a fit dataset A and apply it, unchanged,
to a different eval dataset B:

  * best SE layer pair chosen on A,
  * standardization statistics (mu, sd) from A's pool,
  * source probe trained on A (source labels),
  * ridge map W fit on A's unlabeled alignment pairs.

Then B's target hidden states are standardized with A's stats and scored by the
transferred probe. Nothing about B is used to build the pipeline, so this is a
true cross-distribution transfer of both the probe and the map.

Reported as a matrix of eval AUROC (rows = fit dataset A, cols = eval dataset B):
  * diagonal  -> within-distribution transfer (A==B), the step-2 number,
  * off-diag  -> cross-distribution transfer,
  * native    -> per-eval-dataset probe trained on B's own labels (upper reference).
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sep.transfer.cka import load_hidden
from sep.transfer.transfer import (
    best_se_layer, best_split, binarize, fit_ridge_map, load_entropy,
    probe_scores, transfer_probe,
)


def load_dataset(source_gen, target_gen, token):
    Hs, ids_s = load_hidden(source_gen, token)
    Ht, ids_t = load_hidden(target_gen, token)
    assert ids_s == ids_t, "example ids not aligned across models"
    ys = binarize(load_entropy(source_gen), best_split(load_entropy(source_gen)))
    yt = binarize(load_entropy(target_gen), best_split(load_entropy(target_gen)))
    return Hs, Ht, ys, yt


def build_pipeline(D, n_align, seed):
    """Freeze the source-side pipeline on fit dataset A.

    Returns everything needed to score any other dataset's target hidden states.
    """
    Hs, Ht, ys, yt = D
    rng = np.random.default_rng(seed)
    N = Hs.shape[1]
    pool = rng.permutation(N)  # A contributes only training data here

    Ls, _ = best_se_layer(Hs[:, pool], ys[pool], seed=seed)
    Lt, _ = best_se_layer(Ht[:, pool], yt[pool], seed=seed)

    Xs, Xt = Hs[Ls].astype(np.float64), Ht[Lt].astype(np.float64)
    mu_s, sd_s = Xs[pool].mean(0), Xs[pool].std(0) + 1e-6
    mu_t, sd_t = Xt[pool].mean(0), Xt[pool].std(0) + 1e-6
    Zs, Zt = (Xs - mu_s) / sd_s, (Xt - mu_t) / sd_t

    src_probe = LogisticRegression(max_iter=1000).fit(Zs[pool], ys[pool])

    align = pool[:n_align]
    M, b = fit_ridge_map(Zt[align], Zs[align])
    a_t, c_t = transfer_probe(src_probe, M, b)

    return {"Lt": int(Lt), "Ls": int(Ls), "mu_t": mu_t, "sd_t": sd_t,
            "a_t": a_t, "c_t": c_t}


def eval_on(pipe, D_eval):
    """Score eval dataset B with A's frozen (layer, standardization, probe)."""
    _, Ht, _, yt = D_eval
    Xt = Ht[pipe["Lt"]].astype(np.float64)
    Zt = (Xt - pipe["mu_t"]) / pipe["sd_t"]
    return roc_auc_score(yt, probe_scores(Zt, pipe["a_t"], pipe["c_t"]))


def native_auc(D, seed):
    """Reference: target probe trained on B's own labels at B's best layer."""
    _, Ht, _, yt = D
    rng = np.random.default_rng(seed)
    N = Ht.shape[1]
    idx = rng.permutation(N)
    tr, te = idx[: int(0.75 * N)], idx[int(0.75 * N):]
    Lt, _ = best_se_layer(Ht[:, tr], yt[tr], seed=seed)
    X = Ht[Lt].astype(np.float64)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X - mu) / sd
    clf = LogisticRegression(max_iter=1000).fit(Z[tr], yt[tr])
    return roc_auc_score(yt[te], clf.predict_proba(Z[te])[:, 1])


def run(gens, names, token, out_dir, n_align, seed):
    Ds = {name: load_dataset(s, t, token) for name, (s, t) in zip(names, gens)}
    for name in names:
        print(f"loaded {name}: src{Ds[name][0].shape} tgt{Ds[name][1].shape}")

    pipes = {a: build_pipeline(Ds[a], n_align, seed) for a in names}
    for a in names:
        print(f"pipeline[{a}]: src layer {pipes[a]['Ls']} -> tgt layer {pipes[a]['Lt']}")

    matrix = {a: {} for a in names}
    for a in names:
        for b in names:
            matrix[a][b] = float(eval_on(pipes[a], Ds[b]))
    native = {b: float(native_auc(Ds[b], seed)) for b in names}

    print(f"\ntransfer AUROC (rows=fit A, cols=eval B), n_align={n_align}:")
    header = "        " + "".join(f"{b:>11}" for b in names)
    print(header)
    for a in names:
        row = "".join(f"{matrix[a][b]:>11.3f}" for b in names)
        print(f"{a:>8}{row}")
    print("  native " + "".join(f"{native[b]:>11.3f}" for b in names))

    diag = np.mean([matrix[a][a] for a in names])
    off = np.mean([matrix[a][b] for a in names for b in names if a != b])
    print(f"\nmean within-distribution (diag): {diag:.3f}")
    print(f"mean cross-distribution (off-diag): {off:.3f}")
    print(f"mean native reference: {np.mean(list(native.values())):.3f}")

    os.makedirs(out_dir, exist_ok=True)
    out = {"names": names, "n_align": n_align, "matrix": matrix,
           "native": native, "mean_within": diag, "mean_cross": off}
    with open(os.path.join(out_dir, f"cross_dataset_{token}.json"), "w") as f:
        json.dump(out, f, indent=2)
    _plot(matrix, native, names, token, out_dir)
    print(f"saved -> {out_dir}")


def _plot(matrix, native, names, token, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.array([[matrix[a][b] for b in names] for a in names])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="viridis", vmin=0.5, vmax=0.85)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("eval dataset B")
    ax.set_ylabel("fit dataset A")
    ax.set_title(f"Cross-distribution transfer AUROC ({token.upper()})")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.72 else "black")
    fig.colorbar(im, ax=ax, label="eval AUROC")
    plt.tight_layout()
    path = os.path.join(out_dir, f"cross_dataset_{token}.pdf")
    plt.savefig(path, format="pdf", dpi=200)
    plt.close()
    print(f"  saved heatmap -> {path}")


def main():
    p = argparse.ArgumentParser(description="Cross-distribution SE probe transfer.")
    p.add_argument("--source-gens", nargs=3, required=True,
                   help="1.7B validation_generations.pkl for trivia_qa squad nq")
    p.add_argument("--target-gens", nargs=3, required=True,
                   help="8B validation_generations.pkl for trivia_qa squad nq")
    p.add_argument("--names", nargs=3, default=["trivia_qa", "squad", "nq"])
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-align", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    gens = list(zip(args.source_gens, args.target_gens))
    run(gens, args.names, args.token, args.out_dir, args.n_align, args.seed)


if __name__ == "__main__":
    main()
