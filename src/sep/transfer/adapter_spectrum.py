"""Spectral analysis of the linear transfer adapters (any set of pairs).

Each transfer "adapter" is the ridge map M (target space -> source space) from
fit_ridge_map, fit exactly as transfer.py fits it: on the SE layer's pool-z-scored
features, over the full pool (the "ceiling" adapter). M has shape (d_t, d_s); its
SVD  M = U S V^T  decomposes the map into:
  * S : singular values (the SPECTRUM) -- how much the map stretches each of its
        principal directions.
  * V : right singular vectors, directions in SOURCE space (the inputs the map
        cares about); rows of Vt.
  * U : left singular vectors, directions in TARGET space (where they are sent).

We report three things:

1. SPECTRUM. The sorted singular values (normalized by S[0]) + cumulative energy.
   Fast decay => the map lives in a few directions (low-dim shared structure);
   flat => it needs the whole space.

2. EFFECTIVE RANK. One number summarizing "how many directions the map really
   uses". Reported three ways: energy thresholds (#components for 90/95/99% of
   sum S^2), participation ratio (sum S^2)^2 / sum S^4, and spectral entropy
   exp(-sum p log p) with p = S^2 / sum S^2. Robust to the full-rank numerical
   floor (algebraic rank is meaningless here).

3. CROSS-PAIR SUBSPACE ANGLES. Whether different adapters use the SAME directions.
   For the top-k right singular vectors V_k (source space), the principal angles
   between two pairs' V_k are acos of the singular values of V_k(A)^T V_k(B); we
   summarize with mean cos^2 (1=identical subspace, 0=orthogonal). ONLY defined
   between pairs sharing a space: V-subspaces compared across pairs with the same
   source model, U-subspaces across pairs with the same target model. Entries that
   are not comparable are left as null.

CLI takes parallel lists (one entry per pair), like cross_dataset.py:
  --names A B  --source-gens sa sb  --target-gens ta tb
  [--source-keys ...] [--target-keys ...]   (default keys = source/target name in
  the pair name; keys identify the shared space for the angle comparison)
"""
import argparse
import itertools
import json
import os

import numpy as np

from sep.transfer.cka import load_hidden
from sep.transfer.transfer import (
    load_entropy, best_split, binarize, best_se_layer, fit_ridge_map,
)


# --------------------------------------------------------------------------- #
# Fit one adapter exactly as transfer.py does (ceiling map on the full pool).
# --------------------------------------------------------------------------- #
def fit_adapter(source_gen, target_gen, token, n_eval, seed, alpha):
    rng = np.random.default_rng(seed)
    Hs, ids_s = load_hidden(source_gen, token)
    Ht, ids_t = load_hidden(target_gen, token)
    assert ids_s == ids_t, "example ids not aligned across models"
    N = Hs.shape[1]

    ys = binarize(*(lambda e: (e, best_split(e)))(load_entropy(source_gen)))
    yt = binarize(*(lambda e: (e, best_split(e)))(load_entropy(target_gen)))

    perm = rng.permutation(N)
    pool = perm[n_eval:]

    Ls, aucs = best_se_layer(Hs[:, pool], ys[pool], seed=seed)
    Lt, auct = best_se_layer(Ht[:, pool], yt[pool], seed=seed)
    Xs = Hs[Ls].astype(np.float64)
    Xt = Ht[Lt].astype(np.float64)
    mu_s, sd_s = Xs[pool].mean(0), Xs[pool].std(0) + 1e-6
    mu_t, sd_t = Xt[pool].mean(0), Xt[pool].std(0) + 1e-6
    Zs = (Xs - mu_s) / sd_s
    Zt = (Xt - mu_t) / sd_t

    M, b = fit_ridge_map(Zt[pool], Zs[pool], alpha=alpha)  # (d_t, d_s)
    meta = {
        "d_source": int(Xs.shape[1]), "d_target": int(Xt.shape[1]),
        "src_layer": int(Ls), "tgt_layer": int(Lt),
        "src_layer_se_auroc": float(aucs), "tgt_layer_se_auroc": float(auct),
        "n_pool": int(len(pool)),
    }
    return M, meta


# --------------------------------------------------------------------------- #
# 1+2: spectrum and effective rank.
# --------------------------------------------------------------------------- #
def spectrum_and_rank(M):
    """SVD spectrum + effective-rank summaries. Returns (s, U, Vt, stats)."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)  # U (dt,k), s (k,), Vt (k,ds)
    s = s.astype(np.float64)
    energy = s ** 2
    total = energy.sum()
    cum = np.cumsum(energy) / total                    # cumulative energy fraction

    def rank_at(frac):
        return int(np.searchsorted(cum, frac) + 1)     # #components to reach frac

    p = energy / total
    p = p[p > 0]
    spectral_entropy = float(-np.sum(p * np.log(p)))
    stats = {
        "n_singular_values": int(len(s)),
        "sigma_max": float(s[0]), "sigma_min": float(s[-1]),
        "condition_number": float(s[0] / s[-1]) if s[-1] > 0 else float("inf"),
        "rank_90pct_energy": rank_at(0.90),
        "rank_95pct_energy": rank_at(0.95),
        "rank_99pct_energy": rank_at(0.99),
        "participation_ratio": float(total ** 2 / np.sum(energy ** 2)),
        "spectral_entropy_effrank": float(np.exp(spectral_entropy)),
        "stable_rank": float(total / s[0] ** 2),        # ||M||_F^2 / ||M||_2^2
    }
    return s, U, Vt, stats


# --------------------------------------------------------------------------- #
# 3: cross-pair subspace angles (only between pairs sharing a space).
# --------------------------------------------------------------------------- #
def subspace_similarity(Bk_a, Bk_b):
    """mean cos^2 of principal angles between two orthonormal bases (d, k).

    Columns of each are orthonormal (SVD singular vectors), so the singular
    values of Bk_a^T Bk_b are exactly the cosines of the principal angles.
    Returns (mean_cos2, list_of_angles_degrees).
    """
    k = min(Bk_a.shape[1], Bk_b.shape[1])
    cross = Bk_a[:, :k].T @ Bk_b[:, :k]        # (k, k)
    cos_angles = np.linalg.svd(cross, compute_uv=False)
    cos_angles = np.clip(cos_angles, 0.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_angles))
    return float(np.mean(cos_angles ** 2)), angles_deg.tolist()


def _pairwise_matrix(pairs, get_basis, get_key, topk):
    """Similarity matrix over pairs; null where the shared space differs."""
    n = len(pairs)
    mat = [[None] * n for _ in range(n)]
    angles = {}
    for i, j in itertools.product(range(n), range(n)):
        if get_key(pairs[i]) != get_key(pairs[j]):
            continue                            # not comparable (different space)
        sim, ang = subspace_similarity(get_basis(pairs[i], topk),
                                       get_basis(pairs[j], topk))
        mat[i][j] = sim
        if i < j:
            angles[f"{pairs[i]['name']}__vs__{pairs[j]['name']}"] = {
                "mean_cos2": sim, "principal_angles_deg": ang,
            }
    return mat, angles


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(names, source_gens, target_gens, source_keys, target_keys,
        token, out_dir, n_eval, seed, alpha, topk):
    pairs = []
    for name, sg, tg, sk, tk in zip(
            names, source_gens, target_gens, source_keys, target_keys):
        M, meta = fit_adapter(sg, tg, token, n_eval, seed, alpha)
        s, U, Vt, stats = spectrum_and_rank(M)
        pairs.append({
            "name": name, "source_key": sk, "target_key": tk,
            "meta": meta, "spectrum_stats": stats,
            "singular_values": s, "U": U, "Vt": Vt,
        })
        print(f"[{name}] d_t={meta['d_target']} d_s={meta['d_source']} "
              f"k={stats['n_singular_values']} | "
              f"rank@95%={stats['rank_95pct_energy']} "
              f"PR={stats['participation_ratio']:.1f} "
              f"H-effrank={stats['spectral_entropy_effrank']:.1f} "
              f"cond={stats['condition_number']:.1f}")

    # V lives in source space (rows of Vt); compare across shared SOURCE key.
    # U lives in target space (cols of U); compare across shared TARGET key.
    def v_basis(pr, k):
        return pr["Vt"][:k].T                 # (d_s, k)

    def u_basis(pr, k):
        return pr["U"][:, :k]                 # (d_t, k)

    src_sim, src_ang = _pairwise_matrix(pairs, v_basis,
                                        lambda p: p["source_key"], topk)
    tgt_sim, tgt_ang = _pairwise_matrix(pairs, u_basis,
                                        lambda p: p["target_key"], topk)

    os.makedirs(out_dir, exist_ok=True)
    asuf = "" if alpha == 1e3 else f"_a{alpha:.0e}"
    base = f"adapter_spectrum_{token}{asuf}"

    results = {
        "token": token, "alpha": float(alpha), "topk": int(topk),
        "pairs": [{
            "name": p["name"], "source_key": p["source_key"],
            "target_key": p["target_key"], "meta": p["meta"],
            "spectrum_stats": p["spectrum_stats"],
        } for p in pairs],
        "source_subspace_similarity": {
            "note": "mean cos^2 of top-k right-singular-vector (source-space) "
                    "subspaces; null where source_key differs",
            "pair_names": [p["name"] for p in pairs],
            "matrix": src_sim, "angles": src_ang,
        },
        "target_subspace_similarity": {
            "note": "mean cos^2 of top-k left-singular-vector (target-space) "
                    "subspaces; null where target_key differs",
            "pair_names": [p["name"] for p in pairs],
            "matrix": tgt_sim, "angles": tgt_ang,
        },
    }
    with open(os.path.join(out_dir, base + ".json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(out_dir, base + "_svd.npz"),
             **{f"{p['name']}__sigma": p["singular_values"] for p in pairs})

    _plot(pairs, src_sim, tgt_sim, results, token, out_dir, base)
    _print(results)
    print(f"\nsaved -> {out_dir}/{base}.json")
    return results


def _plot(pairs, src_sim, tgt_sim, r, token, out_dir, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(pairs)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) normalized spectrum, log-y.
    ax = axes[0, 0]
    for p in pairs:
        s = p["singular_values"]
        ax.plot(np.arange(1, len(s) + 1), s / s[0], lw=1.2, label=p["name"])
    ax.set_yscale("log")
    ax.set_xlabel("singular-value index")
    ax.set_ylabel(r"$\sigma_i/\sigma_1$ (normalized)")
    ax.set_title("Adapter spectrum")
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)
    ax.legend(fontsize=8)

    # (b) cumulative energy.
    ax = axes[0, 1]
    for p in pairs:
        e = p["singular_values"] ** 2
        cum = np.cumsum(e) / e.sum()
        ax.plot(np.arange(1, len(cum) + 1), cum, lw=1.2, label=p["name"])
    for frac in (0.9, 0.95, 0.99):
        ax.axhline(frac, color="k", lw=0.5, ls=":")
    ax.set_xlabel("number of singular values")
    ax.set_ylabel("cumulative energy fraction")
    ax.set_title("Cumulative energy (effective rank read-off)")
    ax.set_xscale("log")
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)
    ax.legend(fontsize=8)

    # (c) effective-rank bar chart.
    ax = axes[1, 0]
    labels = [p["name"] for p in pairs]
    x = np.arange(n)
    w = 0.25
    r95 = [p["spectrum_stats"]["rank_95pct_energy"] for p in pairs]
    pr = [p["spectrum_stats"]["participation_ratio"] for p in pairs]
    he = [p["spectrum_stats"]["spectral_entropy_effrank"] for p in pairs]
    ax.bar(x - w, r95, w, label="rank @95% energy", color="#4C72B0")
    ax.bar(x, pr, w, label="participation ratio", color="#8172B3")
    ax.bar(x + w, he, w, label="spectral-entropy effrank", color="#CCB974")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel("effective rank (dimensions)")
    ax.set_title("Effective rank per adapter")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6)

    # (d) source-subspace similarity heatmap (top-k right singular vectors).
    ax = axes[1, 1]
    mat = np.array([[np.nan if v is None else v for v in row] for row in src_sim])
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=20)
    ax.set_yticks(x); ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="w")
    ax.set_title(f"Source-subspace similarity (top-{r['topk']}, mean cos²)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Transfer-adapter spectra ({token.upper()}, α={r['alpha']:.0e})",
                 fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, base + ".png")
    plt.savefig(p, dpi=150)
    plt.savefig(p.replace(".png", ".pdf"))
    plt.close()
    print(f"saved spectrum figure -> {p}")


def _print(r):
    print("=" * 72)
    print(f"token={r['token']}  alpha={r['alpha']:.0e}  topk={r['topk']}")
    print(f"{'pair':<28}{'d_t':>6}{'d_s':>6}{'r@95':>7}{'PR':>8}{'Heff':>8}{'cond':>9}")
    for p in r["pairs"]:
        s = p["spectrum_stats"]; m = p["meta"]
        print(f"  {p['name']:<26}{m['d_target']:>6}{m['d_source']:>6}"
              f"{s['rank_95pct_energy']:>7}{s['participation_ratio']:>8.1f}"
              f"{s['spectral_entropy_effrank']:>8.1f}{s['condition_number']:>9.1f}")
    for tag, key in (("SOURCE", "source_subspace_similarity"),
                     ("TARGET", "target_subspace_similarity")):
        ang = r[key]["angles"]
        if ang:
            print(f"\n--- {tag}-space subspace overlap (top-{r['topk']}) ---")
            for name, d in ang.items():
                print(f"  {name}: mean cos² = {d['mean_cos2']:.3f}")
        else:
            print(f"\n--- {tag}-space overlap: needs >=2 pairs sharing a "
                  f"{tag.lower()} key (none here) ---")


def _rep(single, n):
    """Broadcast a single-element list to length n (for shared defaults)."""
    return single * n if len(single) == 1 else single


def main():
    p = argparse.ArgumentParser(
        description="Spectral analysis of transfer adapters (SVD).")
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--source-gens", nargs="+", required=True)
    p.add_argument("--target-gens", nargs="+", required=True)
    p.add_argument("--source-keys", nargs="+", default=None,
                   help="space id per pair; pairs with equal key share source "
                        "space (default: the --names entries, i.e. none shared)")
    p.add_argument("--target-keys", nargs="+", default=None)
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1e3)
    p.add_argument("--topk", type=int, default=20,
                   help="subspace dimension for the principal-angle comparison")
    a = p.parse_args()
    n = len(a.names)
    assert len(a.source_gens) == n == len(a.target_gens), \
        "names / source-gens / target-gens must be parallel lists"
    source_keys = _rep(a.source_keys, n) if a.source_keys else list(a.names)
    target_keys = _rep(a.target_keys, n) if a.target_keys else list(a.names)
    run(a.names, a.source_gens, a.target_gens, source_keys, target_keys,
        a.token, a.out_dir, a.n_eval, a.seed, a.alpha, a.topk)


if __name__ == "__main__":
    main()
