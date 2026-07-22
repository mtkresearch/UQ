"""Step 1 of the cross-scale transfer study: layer-correspondence analysis.

Computes a linear-CKA matrix between every source-model layer and every
target-model layer on paired hidden states (the same validation example ids run
through both models, guaranteed aligned because both used the same seed/sample).
Also reports, for the argmax target of each source layer, the ridge linear R^2 --
a direct read on how well an affine map could carry that layer across scale.

Outputs a heatmap PDF and a JSON of the best target layer per source layer, which
drives the layer pairing used when fitting the transfer map in step 2.
"""
import argparse
from sep.uncertainty.utils.config import apply_yaml_config
import json
import os
import pickle

import numpy as np
import torch


def load_hidden(path, token):
    """Return (n_layers, n_samples, hidden) float32 for one merged run.

    token is 'slt' (emb_tok_before_eos) or 'tbg' (emb_last_tok_before_gen).
    Keys are returned too so the caller can assert cross-model alignment.
    """
    key = "emb_tok_before_eos" if token == "slt" else "emb_last_tok_before_gen"
    with open(path, "rb") as f:
        gens = pickle.load(f)
    ids = list(gens.keys())
    h = torch.stack(
        [gens[k]["most_likely_answer"][key] for k in ids]
    ).squeeze(-2).transpose(0, 1).to(torch.float32)  # (layers, samples, hidden)
    return h.numpy(), ids


def linear_cka(X, Y):
    """Linear CKA between two (n_samples, dim) matrices.

    CKA is invariant to isotropic scaling and orthogonal transforms, so it
    measures shared representational structure independent of dimensionality --
    exactly what we need to compare a 2048-d and a 4096-d layer.
    """
    # float64 throughout: Qwen has massive-activation dims (values ~1e3), so the
    # squared Frobenius norms reach ~1e26 and their product overflows float32.
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    # Use the Gram/HSIC form; ||Y^T X||_F^2 avoids building n x n matrices.
    xty = X.T @ Y
    hsic = np.sum(xty ** 2)
    norm_x = np.sum((X.T @ X) ** 2)
    norm_y = np.sum((Y.T @ Y) ** 2)
    denom = np.sqrt(norm_x * norm_y)
    # Degenerate layer (e.g. the embedding position is constant across samples)
    # has zero centered norm -> CKA undefined; treat as no shared structure.
    if denom == 0.0:
        return 0.0
    return hsic / denom


def ridge_r2(Xs, Xt, alpha=1e2, n_train=1000, seed=0):
    """Held-out R^2 of a ridge map Xs -> Xt (source layer to target layer)."""
    Xs = Xs.astype(np.float64)
    Xt = Xt.astype(np.float64)
    rng = np.random.default_rng(seed)
    n = Xs.shape[0]
    idx = rng.permutation(n)
    tr, te = idx[:n_train], idx[n_train:]
    Xs_tr, Xt_tr = Xs[tr], Xt[tr]
    mu_s, mu_t = Xs_tr.mean(0), Xt_tr.mean(0)
    A = Xs_tr - mu_s
    B = Xt_tr - mu_t
    d = A.shape[1]
    W = np.linalg.solve(A.T @ A + alpha * np.eye(d), A.T @ B)  # (d_s, d_t)
    pred = (Xs[te] - mu_s) @ W + mu_t
    resid = np.sum((Xt[te] - pred) ** 2)
    total = np.sum((Xt[te] - Xt_tr.mean(0)) ** 2)
    return 1.0 - resid / total


def main():
    p = argparse.ArgumentParser(description="Cross-scale layer CKA matrix.")
    p.add_argument("--source-gen", required=True, help="1.7B validation_generations.pkl")
    p.add_argument("--target-gen", required=True, help="8B validation_generations.pkl")
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--r2-alpha", type=float, default=1e2)
    args = apply_yaml_config(p)

    Hs, ids_s = load_hidden(args.source_gen, args.token)
    Ht, ids_t = load_hidden(args.target_gen, args.token)
    assert ids_s == ids_t, "example ids not aligned across models"
    ns, nt = Hs.shape[0], Ht.shape[0]
    print(f"source layers={ns} target layers={nt} samples={Hs.shape[1]}")

    cka = np.zeros((ns, nt))
    for i in range(ns):
        for j in range(nt):
            cka[i, j] = linear_cka(Hs[i], Ht[j])
        print(f"  src layer {i:2d}: best target {cka[i].argmax():2d} "
              f"CKA={cka[i].max():.3f}")

    best_target = cka.argmax(1)
    pairing = []
    for i in range(ns):
        j = int(best_target[i])
        r2 = ridge_r2(Hs[i], Ht[j], alpha=args.r2_alpha)
        pairing.append({"src": i, "tgt": j,
                        "cka": float(cka[i, j]), "ridge_r2": float(r2)})

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, f"cka_matrix_{args.token}.npy"), cka)
    with open(os.path.join(args.out_dir, f"pairing_{args.token}.json"), "w") as f:
        json.dump(pairing, f, indent=2)

    _plot(cka, args.token, args.out_dir)
    print(f"saved cka matrix + pairing -> {args.out_dir}")

    # Print the pairing for the layers that matter for SE (later source layers).
    print("src -> tgt (CKA, ridge R^2):")
    for row in pairing:
        print(f"  {row['src']:2d} -> {row['tgt']:2d}  "
              f"CKA={row['cka']:.3f}  R2={row['ridge_r2']:.3f}")


def _plot(cka, token, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cka, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Target (Qwen3-8B) layer")
    ax.set_ylabel("Source (Qwen3-1.7B) layer")
    ax.set_title(f"Linear CKA ({token.upper()})")
    fig.colorbar(im, ax=ax, label="CKA")
    plt.tight_layout()
    path = os.path.join(out_dir, f"cka_matrix_{token}.pdf")
    plt.savefig(path, format="pdf", dpi=200)
    plt.close()
    print(f"  saved heatmap -> {path}")


if __name__ == "__main__":
    main()
