"""Confound B -- piecewise-linear map fragmentation check for MoE targets.

Concern: `transfer.py`'s cross-scale probe transfer fits a SINGLE global
affine map f(z_t) = z_t @ M + b from target hidden states to source hidden
states. That assumption held for dense Qwen3-1.7B -> Qwen3-8B because every
example goes through the same dense computation graph, so the map only has to
absorb a smooth change of basis. A MoE target routes different examples
through different expert subsets -- effectively a different (near-)linear
region per routing pattern -- so a single global affine map may fit worse
than the SAME map fit separately per routing "regime". If per-bucket R^2 is
much higher than pooled R^2, MoE hidden states are more piecewise-linear than
dense ones and the global-map assumption in `transfer.py` is confounded by
routing.

Method:
  * Bucket examples by a discrete "expert signature": k-means over each
    example's mean top-1-expert histogram (aggregated over generated tokens,
    same features as `route_control.py`'s hard router feature, but clustered
    rather than classified).
  * At the paired (source dense layer, target MoE layer) discovered by the
    existing CKA/ridge machinery (`cka.ridge_r2`), compare:
      - pooled R^2: one ridge map fit on ALL training examples
      - per-bucket R^2: separate ridge maps, one per expert-signature bucket,
        evaluated on that bucket's held-out examples, then averaged
        (weighted by bucket size)
  * Report the CKA of the same layer pair as a control -- if raw
    representational similarity is already low, fragmentation isn't the
    story, misalignment is.

Verdict: fragmentation is flagged if per-bucket R^2 exceeds pooled R^2 by
more than `--gap-threshold` (default 0.05) on average, i.e. examples with
different dominant experts are affinely-related to the dense source in
detectably different ways.
"""
import argparse
from sep.uncertainty.utils.config import apply_yaml_config
import json
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans

from sep.transfer.cka import linear_cka, load_hidden, ridge_r2


def load_moe_hidden_and_router(gen_path, token):
    """(n_layers, n_samples, hidden) hidden states + list of router_logits, aligned ids."""
    key = "emb_tok_before_eos" if token == "slt" else "emb_last_tok_before_gen"
    with open(gen_path, "rb") as f:
        gens = pickle.load(f)
    ids = list(gens.keys())
    import torch
    h = torch.stack(
        [gens[k]["most_likely_answer"][key] for k in ids]
    ).squeeze(-2).transpose(0, 1).to(torch.float32).numpy()
    routers = [gens[k]["router_logits"] for k in ids]
    return h, routers, ids


def expert_signature(router_logits, n_layers_moe):
    """Normalized top-1-expert histogram per layer, flattened -- one vector per example."""
    sigs = []
    for r in router_logits:
        L, _, E = r.shape
        assert L == n_layers_moe
        top1 = np.argmax(r, axis=-1)
        hist = np.stack([np.bincount(top1[l], minlength=E) for l in range(L)]).astype(np.float64)
        hist /= hist.sum(-1, keepdims=True)
        sigs.append(hist.ravel())
    return np.stack(sigs)


def bucket_examples(signatures, n_buckets, seed):
    km = KMeans(n_clusters=n_buckets, random_state=seed, n_init=10).fit(signatures)
    return km.labels_


def pooled_vs_bucketed_r2(Xs, Xt, buckets, alpha, n_train_frac, seed):
    """Xs (source layer, n, ds), Xt (target layer, n, dt). Returns (pooled_r2, bucket_r2s)."""
    n = Xs.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n_train_frac * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    # ridge_r2 does its own internal train/test split on the array passed in, so
    # feed it the full pool restricted to the SAME train/test partition used per-bucket.
    combined_idx = np.concatenate([train_idx, test_idx])
    n_tr_combined = len(train_idx)
    pooled_r2 = ridge_r2(Xs[combined_idx], Xt[combined_idx], alpha=alpha,
                         n_train=n_tr_combined, seed=seed)

    bucket_r2s = {}
    bucket_sizes = {}
    for b in np.unique(buckets):
        b_train = train_idx[buckets[train_idx] == b]
        b_test = test_idx[buckets[test_idx] == b]
        if len(b_train) < 20 or len(b_test) < 5:
            continue
        b_idx = np.concatenate([b_train, b_test])
        r2 = ridge_r2(Xs[b_idx], Xt[b_idx], alpha=alpha, n_train=len(b_train), seed=seed)
        bucket_r2s[int(b)] = float(r2)
        bucket_sizes[int(b)] = int(len(b_test))

    return float(pooled_r2), bucket_r2s, bucket_sizes


def run(dense_gen, moe_gen, token, n_buckets, alpha, n_train_frac, seed, gap_threshold):
    Hs, ids_s = load_hidden(dense_gen, token)
    Ht, routers, ids_t = load_moe_hidden_and_router(moe_gen, token)

    common = [i for i in ids_t if i in set(ids_s)]
    assert len(common) > 0, "no overlapping example ids between dense source and MoE target"
    idx_s = [ids_s.index(i) for i in common]
    idx_t = [ids_t.index(i) for i in common]
    Hs = Hs[:, idx_s]
    Ht = Ht[:, idx_t]
    routers = [routers[i] for i in idx_t]
    n = len(common)
    print(f"aligned n={n} (source ids={len(ids_s)}, target ids={len(ids_t)})")

    n_layers_moe = routers[0].shape[0]
    sigs = expert_signature(routers, n_layers_moe)
    buckets = bucket_examples(sigs, n_buckets, seed)
    print(f"expert-signature buckets: {np.bincount(buckets)}")

    # CKA to find the best-aligned (source layer, target layer) pair, same
    # convention as cka.py: argmax target per source layer, then take the
    # overall best pair among the later (SE-relevant) half of source layers.
    ns, nt = Hs.shape[0], Ht.shape[0]
    cka = np.zeros((ns, nt))
    for i in range(ns):
        for j in range(nt):
            cka[i, j] = linear_cka(Hs[i], Ht[j])
    src_layer, tgt_layer = np.unravel_index(np.argmax(cka), cka.shape)
    print(f"best CKA pair: src={src_layer} tgt={tgt_layer} CKA={cka[src_layer, tgt_layer]:.3f}")

    Xs, Xt = Hs[src_layer].astype(np.float64), Ht[tgt_layer].astype(np.float64)
    pooled_r2, bucket_r2s, bucket_sizes = pooled_vs_bucketed_r2(
        Xs, Xt, buckets, alpha, n_train_frac, seed)

    sizes = np.array([bucket_sizes[b] for b in bucket_r2s])
    r2s = np.array([bucket_r2s[b] for b in bucket_r2s])
    mean_bucket_r2 = float(np.average(r2s, weights=sizes)) if len(r2s) else float("nan")
    gap = mean_bucket_r2 - pooled_r2

    if pooled_r2 < 0:
        verdict = ("INCONCLUSIVE -- pooled R^2 < 0 (ridge under-regularized at this alpha/n): "
                   "any gap here reflects noise in a poorly-fit map, not routing fragmentation. "
                   "Re-run with higher --alpha until pooled_r2 > 0.")
    else:
        verdict = ("FRAGMENTATION -- per-bucket R^2 exceeds pooled by "
                   f">= {gap_threshold}: routing splits the map into distinct affine regimes"
                   if gap >= gap_threshold else
                   "NO FRAGMENTATION -- a single global affine map fits about as well as "
                   "per-routing-bucket maps")

    out = {
        "dense_gen": dense_gen, "moe_gen": moe_gen, "token": token, "n": n,
        "n_buckets_requested": n_buckets, "bucket_counts": np.bincount(buckets).tolist(),
        "src_layer": int(src_layer), "tgt_layer": int(tgt_layer),
        "cka": float(cka[src_layer, tgt_layer]),
        "alpha": alpha, "seed": seed,
        "pooled_r2": pooled_r2,
        "bucket_r2": {str(k): v for k, v in bucket_r2s.items()},
        "bucket_sizes": {str(k): v for k, v in bucket_sizes.items()},
        "mean_bucket_r2": mean_bucket_r2,
        "gap": gap,
        "gap_threshold": gap_threshold,
        "verdict": verdict,
    }

    print(f"\npooled R^2={pooled_r2:.3f}  size-weighted mean per-bucket R^2={mean_bucket_r2:.3f}  "
          f"gap={gap:.3f}")
    print(f"VERDICT: {verdict}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Confound B: MoE piecewise-linear map fragmentation.")
    ap.add_argument("--dense-gen", required=True, help="dense source validation_generations.pkl")
    ap.add_argument("--moe-gen", required=True, help="OLMoE validation_generations.pkl")
    ap.add_argument("--token", default="slt", choices=["slt", "tbg"])
    ap.add_argument("--n-buckets", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1e7,
                     help="Ridge regularization. NOTE: at this n (~300) and dim (2048x2048), "
                          "the R^2 landscape is extremely alpha-sensitive -- alpha<=1e4 gives "
                          "NEGATIVE R^2 for both pooled and per-bucket fits, which can produce a "
                          "spurious 'fragmentation' verdict from noise, not signal. Sweep alpha "
                          "and confirm pooled R^2 > 0 before trusting the gap.")
    ap.add_argument("--n-train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap-threshold", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = apply_yaml_config(ap)

    out = run(args.dense_gen, args.moe_gen, args.token, args.n_buckets,
              args.alpha, args.n_train_frac, args.seed, args.gap_threshold)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
