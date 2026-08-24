"""transfer2.py — Efficient SE probe transfer with caching.

Three phases:
  1. probe_cache   : for each (model, dataset), find best layer + fit probe on
                     all n_grid sizes. Save everything to disk.
  2. align_cache   : for each (src_model, tgt_model, eval_ds, align_ds), load
                     best-layer hidden states (from cache) and fit alignment
                     matrices at all n_grid sizes. Save M, b to disk.
  3. evaluate      : combine probe cache + alignment cache to compute AUROC
                     curves (alignment_grid + probe_grid) and save results/plots.

Nothing in this file imports or modifies any existing transfer.py code.

Usage:
    python -m sep.transfer.transfer2 probe_cache  --out-dir <cache_dir> ...
    python -m sep.transfer.transfer2 align_cache  --out-dir <cache_dir> ...
    python -m sep.transfer.transfer2 evaluate     --out-dir <cache_dir> ...
    python -m sep.transfer.transfer2 summary      --out-dir <cache_dir> ...

Or run all phases via the shell wrapper slurm/run_transfer_v2.sh.
"""
import argparse
import glob
import json
import math
import os
import pickle
import re
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ============================================================================
# Helpers
# ============================================================================

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _repo_root():
    """Repo root, i.e. 4 levels up from src/sep/transfer/<this file>."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))


def _dataset_tag(path):
    for part in path.replace("\\", "/").split("/"):
        m = re.match(r'^([a-zA-Z_]+?)_all_', part)
        if m:
            return m.group(1)
    return "unknown"


def _model_tag(path):
    parts = path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if re.match(r'^[a-zA-Z_]+?_all_', part) and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


_TOKEN_KEY = {
    "slt": "emb_tok_before_eos",
    "tbg": "emb_last_tok_before_gen",
}

def _load_hidden(gen_path, token="slt"):
    """Load hidden states from validation_generations.pkl.

    Returns (H, ids) where H has shape (n_layers, n_samples, hidden_dim).
    """
    emb_key = _TOKEN_KEY.get(token, f"emb_{token}")
    with open(gen_path, "rb") as f:
        data = pickle.load(f)
    ids = list(data.keys())
    tensors = []
    for k in ids:
        t = data[k]["most_likely_answer"][emb_key]  # (n_layers, 1, d)
        tensors.append(t[:, 0, :].float().numpy())  # (n_layers, d)
    H = np.stack(tensors, axis=1).astype(np.float32)  # (n_layers, N, d)
    return H, ids


def _align_ids(ids_s, ids_t):
    """Return index array such that H_tgt[:, order] rows match H_src rows by ID.

    Multi-GPU sharding can merge shards in completion order, scrambling the row
    order without changing the example-id set.  Reordering fixes this silently.
    Hard-fails if the id sets genuinely differ (different runs / examples).
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


def _load_entropy(gen_path):
    unc_path = os.path.join(os.path.dirname(gen_path), "uncertainty_measures.pkl")
    with open(unc_path, "rb") as f:
        measures = pickle.load(f)
    return np.asarray(
        measures["uncertainty_measures"]["cluster_assignment_entropy"],
        dtype=np.float64,
    )


def _best_split(ents):
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


def _binarize(ents, thr):
    return (ents >= thr).astype(np.int64)


def _zscore(train, full):
    mu = train.mean(0)
    sd = train.std(0) + 1e-6
    return (full - mu) / sd, mu, sd


def _fit_ridge(Zt, Zs, alpha=1e3):
    mu_t, mu_s = Zt.mean(0), Zs.mean(0)
    A, B = Zt - mu_t, Zs - mu_s
    d = A.shape[1]
    M = np.linalg.solve(A.T @ A + alpha * np.eye(d), A.T @ B)
    b = mu_s - mu_t @ M
    return M, b


def _fit_procrustes(Zt, Zs):
    mu_t, mu_s = Zt.mean(0), Zs.mean(0)
    A, B = Zt - mu_t, Zs - mu_s
    U, _, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    M = U @ Vt
    b = mu_s - mu_t @ M
    return M, b


def _fit_e2_rstar(Zt, w_s, u_t, lam=1e3):
    """E2-R*: supervised probe-aware map, min_M ||w_s^T M h_t + b - u_t||^2 + lam*||M||^2_F.

    Regresses the TARGET entropy u_t directly, so this is the labelled-target
    oracle.  Substituting a = M w_s and beta = w_s^T b turns it into a plain
    ridge on (Zt, u_t) with alpha_eff = lam/||w_s||^2 (n-independent: the data
    term is not divided by n).  The optimal M is the minimum-Frobenius-norm
    solution of M w_s = a, i.e. the rank-1 outer(a, w_s)/||w_s||^2, so we return
    the rank-1 factors instead of the dense d_t x d_s matrix -- see
    README_E2R.md.  Reconstruct with _e2_rstar_dense or apply directly with
    _e2_rstar_transfer.

    Returns (a, beta): a in R^{d_t}, beta scalar.
    """
    ws_sq = float(np.dot(w_s, w_s))
    mu_t = Zt.mean(0)
    A = Zt - mu_t
    mu_u = float(u_t.mean())
    t = u_t - mu_u

    d = A.shape[1]
    a = np.linalg.solve(A.T @ A + (lam / ws_sq) * np.eye(d), A.T @ t)
    beta = mu_u - float(mu_t @ a)
    return a, beta


def _e2_rstar_dense(a, beta, w_s):
    """Materialise the rank-1 (M, b) pair.  Only needed for interop/debugging."""
    ws_sq = float(np.dot(w_s, w_s))
    return np.outer(a, w_s) / ws_sq, beta * np.asarray(w_s) / ws_sq


def _e2_rstar_transfer(a, beta, w_s, a_s, c_s):
    """Push a probe (a_s, c_s) through the rank-1 E2-R* map without building M.

    M = outer(a, w_s)/||w_s||^2 and b = beta*w_s/||w_s||^2, so with
    kappa = (w_s . a_s)/||w_s||^2 we get M @ a_s = kappa*a and b @ a_s = kappa*beta.
    Exactly equivalent to _transfer_probe on the dense matrices, at O(d) cost.
    """
    kappa = float(np.dot(w_s, a_s)) / float(np.dot(w_s, w_s))
    return kappa * np.asarray(a), kappa * beta + c_s


def _transfer_probe(clf, M, b):
    a_s = clf.coef_.ravel()
    c_s = float(clf.intercept_[0])
    return M @ a_s, float(b @ a_s) + c_s


def _probe_scores(Z, a, c):
    return Z @ a + c


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _fmt_pow10(s):
    """1000.0 / '1000.0' -> '1e3';  2500 -> '2500'. Swept values are usually 10^k.

    Used both to build the e2_rstar tag and to read a value back out of a label.
    """
    try:
        v = float(s)
    except (TypeError, ValueError):
        return s
    if v > 0:
        e = math.log10(v)
        if abs(e - round(e)) < 1e-9:
            return f"1e{int(round(e))}"
    return f"{v:g}"


def _aligner_tag(aligner, hyperparams=None):
    """Return a short tag encoding aligner + hyperparams.

    hyperparams is a dict of {param_name: value} specific to the aligner.
    Each aligner defines its own relevant params:
      ridge      -> {"alpha": 1e3}      -> "ridge_a1e3"
      procrustes -> {}                  -> "procrustes"
      e2_rstar   -> {"lambda_val": 1e3} -> "e2_rstar_l1e3"
      e2_rstar   -> {"lambda_val": 0.5} -> "e2_rstar_l0.5"
    """
    if not hyperparams:
        return aligner
    parts = [aligner]
    if aligner == "ridge" and "alpha" in hyperparams:
        exp = int(round(np.log10(hyperparams["alpha"])))
        parts.append(f"a1e{exp}")
    elif aligner == "e2_rstar" and "lambda_val" in hyperparams:
        # Powers of ten collapse to 1e3 rather than argparse's float repr 1000.0,
        # matching the ridge tag. Non-powers keep a plain %g form (0.5, 2500).
        parts.append(f"l{_fmt_pow10(hyperparams['lambda_val'])}")
    else:
        for k, v in hyperparams.items():
            parts.append(f"{k}{v}")
    return "_".join(parts)


def _build_hyperparams(args):
    """Build per-aligner hyperparams dict from CLI args."""
    return {
        "ridge":      {"alpha": args.alpha},
        "procrustes": {},
        "e2_rstar":   {"lambda_val": args.lambda_val},
    }


def _load_timing(out_dir):
    path = os.path.join(out_dir, "timing.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"probe_cache": {}, "align_cache": {}}


def _save_timing(out_dir, timing):
    path = os.path.join(out_dir, "timing.json")
    with open(path, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"  -> timing saved {path}")


# ============================================================================
# Phase 1: probe_cache
# ============================================================================
# Cache structure (one file per model+dataset):
#   <cache_dir>/probes/<dataset>/<model>.pkl
#   Contains:
#     best_layer    : int
#     best_auc      : float
#     src_eval_auc  : float  (probe evaluated on its own eval set)
#     mu, sd        : (d,) arrays — z-score stats from pool
#     probe_<n>     : dict with coef_ (d,) and intercept_ float — for each n in n_grid
#     eval_idx, pool: index arrays
#     y             : binary SE labels (N,)
#     layer_aucs    : list of (layer, auc) for all layers

def phase_probe_cache(args):
    """For each (gen_path, token), run layer search and fit probes at all n_grid sizes."""
    n_grid = args.n_grid
    n_eval = args.n_eval
    seed = args.seed
    token = args.token
    rng = np.random.default_rng(seed)

    model_path_map = _load_model_paths(args.model_paths)
    entries = []
    for ds in args.datasets:
        if ds not in model_path_map:
            raise ValueError(f"No entries for dataset '{ds}' in {args.model_paths}")
        for model, path in model_path_map[ds].items():
            entries.append((ds, model, path))

    timing = _load_timing(args.out_dir)

    for ds, model, gen_path in entries:
        out_path = os.path.join(args.out_dir, "probes", ds, f"{model}.pkl")
        if os.path.exists(out_path) and not args.force:
            print(f"[probe_cache] skip {ds}/{model} (exists)")
            continue

        print(f"\n[probe_cache] {ds} / {model}")
        print(f"  hidden states: {gen_path}")

        H, ids = _load_hidden(gen_path, token)
        n_layers, N, d = H.shape
        ent = _load_entropy(gen_path)
        y = _binarize(ent, _best_split(ent))

        perm = rng.permutation(N)
        eval_idx = perm[:n_eval]
        pool = perm[n_eval:]

        # Layer search on full pool
        print(f"  layer search ({n_layers} layers, {len(pool)} pool samples)")
        best_layer, best_auc = 0, -np.inf
        layer_aucs = []
        inner_rng = np.random.default_rng(seed)
        idx = inner_rng.permutation(len(pool))
        tr_idx = pool[idx[:int(0.7 * len(pool))]]
        te_idx = pool[idx[int(0.7 * len(pool)):]]
        t0_layer = time.time()
        for L in tqdm(range(n_layers), desc="  layers", ncols=80):
            X = H[L].astype(np.float64)
            mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-6
            Xz = (X - mu) / sd
            clf = LogisticRegression(max_iter=1000).fit(Xz[tr_idx], y[tr_idx])
            try:
                au = roc_auc_score(y[te_idx], clf.predict_proba(Xz[te_idx])[:, 1])
            except ValueError:
                au = 0.5
            layer_aucs.append(float(au))
            if au > best_auc:
                best_auc, best_layer = au, L
        layer_search_s = round(time.time() - t0_layer, 2)
        print(f"  -> best layer {best_layer}  AUROC {best_auc:.3f}  ({layer_search_s}s)")

        # Extract best-layer features and z-score
        X_best = H[best_layer].astype(np.float64)
        Xz_best, mu_best, sd_best = _zscore(X_best[pool], X_best)

        # Fit probes at each n_grid size.  Time each fit separately: the reported
        # probe_fit_s must be the cost of training ONE probe at the largest n, not
        # the sum over the whole grid (which is a sweep artefact, ~3x larger).
        probes = {}
        per_n_s = {}
        t0_grid = time.time()
        for n in n_grid:
            sub = pool[:n]
            if len(np.unique(y[sub])) < 2:
                probes[n] = None
                continue
            t0_n = time.time()
            clf = LogisticRegression(max_iter=1000).fit(Xz_best[sub], y[sub])
            per_n_s[n] = round(time.time() - t0_n, 3)
            probes[n] = {
                "coef": clf.coef_.ravel().tolist(),
                "intercept": float(clf.intercept_[0]),
            }

        # Source eval AUROC using full-pool probe
        clf_full = LogisticRegression(max_iter=1000).fit(Xz_best[pool], y[pool])
        grid_s = round(time.time() - t0_grid, 2)
        src_eval_auc = roc_auc_score(y[eval_idx],
                                     clf_full.predict_proba(Xz_best[eval_idx])[:, 1])

        n_reported = max(per_n_s) if per_n_s else None
        timing["probe_cache"].setdefault(model, {})[ds] = {
            "layer_search_s": layer_search_s,
            # single probe fit at the largest n -- this is what the paper reports
            "probe_fit_s": per_n_s.get(n_reported),
            "probe_fit_n": n_reported,
            "probe_fit_per_n_s": per_n_s,
            # whole-sweep cost, kept for reference (was what probe_fit_s used to hold)
            "probe_fit_grid_total_s": grid_s,
        }
        _save_timing(args.out_dir, timing)

        cache = {
            "gen_path": gen_path,
            "model": model,
            "dataset": ds,
            "token": token,
            "best_layer": int(best_layer),
            "best_auc": float(best_auc),
            "src_eval_auc": float(src_eval_auc),
            "mu": mu_best.tolist(),
            "sd": sd_best.tolist(),
            "probe_full": {
                "coef": clf_full.coef_.ravel().tolist(),
                "intercept": float(clf_full.intercept_[0]),
            },
            "probes_by_n": probes,
            "layer_aucs": layer_aucs,
            "eval_idx": eval_idx.tolist(),
            "pool": pool.tolist(),
            "y": y.tolist(),
            "N": int(N),
            "n_grid": n_grid,
            "n_eval": n_eval,
            "seed": seed,
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"  -> saved {out_path}")


# ============================================================================
# Pair resolution helper
# ============================================================================

def _load_model_paths(model_paths_file):
    """Parse model_paths.txt -> {dataset: {model: path}}."""
    registry = {}  # {dataset: {model: path}}
    for line in open(model_paths_file):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            print(f"WARNING: bad model_paths line (expected 3 fields): {line}")
            continue
        ds, model, path = parts
        registry.setdefault(ds, {})[model] = path
    return registry


def _resolve_pairs(args):
    """Return a list of (src_path, tgt_path, eval_ds, align_ds) tuples."""
    # 1. Parse model_paths.txt -> {(dataset, model): path}
    nested = _load_model_paths(args.model_paths)
    registry = {(ds, model): path
                for ds, models in nested.items()
                for model, path in models.items()}

    # 2. Parse pair_list.txt as 2-field model pairs
    model_pairs = []
    for line in open(args.pair_list):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            print(f"WARNING: bad pair line (expected 2 fields): {line}")
            continue
        model_pairs.append((parts[0], parts[1]))

    # 3. Parse cross-align mappings: ["eval_ds:align_ds", ...]
    cross_map = []  # list of (eval_ds, align_ds)
    for mapping in (args.cross_align_dataset or []):
        if ":" not in mapping:
            print(f"WARNING: bad --cross-align-dataset entry (expected eval_ds:align_ds): {mapping}")
            continue
        ed, ad = mapping.split(":", 1)
        cross_map.append((ed, ad))

    # 4. Expand pairs
    pairs = []
    for src_model, tgt_model in model_pairs:
        for dataset in args.datasets:
            src_path = registry.get((dataset, src_model))
            tgt_path = registry.get((dataset, tgt_model))
            if src_path is None:
                raise ValueError(f"no path registered for ({dataset}, {src_model}) in model_paths.txt")
            if tgt_path is None:
                raise ValueError(f"no path registered for ({dataset}, {tgt_model}) in model_paths.txt")
                continue
            # same-align pair
            pairs.append((src_model, src_path, tgt_model, tgt_path, dataset, dataset))
            # cross-align pairs
            for eval_ds, align_ds in cross_map:
                if eval_ds != dataset:
                    continue
                align_src = registry.get((align_ds, src_model))
                align_tgt = registry.get((align_ds, tgt_model))
                if align_src is None:
                    raise ValueError(f"no path registered for ({align_ds}, {src_model}) in model_paths.txt")
                if align_tgt is None:
                    raise ValueError(f"no path registered for ({align_ds}, {tgt_model}) in model_paths.txt")
                pairs.append((src_model, src_path, tgt_model, tgt_path, dataset, align_ds))
    return pairs


# ============================================================================
# Phase 2: align_cache
# ============================================================================
# Cache structure:
#   <cache_dir>/alignments/<eval_ds>/<src_model>_to_<tgt_model>_align_<align_ds>.pkl
#   Contains for each n in n_grid:
#     ridge_M_<n>, ridge_b_<n>
#     procrustes_M_<n>, procrustes_b_<n>

def phase_align_cache(args):
    n_grid = args.n_grid
    token = args.token

    timing = _load_timing(args.out_dir)

    pairs = _resolve_pairs(args)
    n_pairs = len(pairs)

    for i, (src_model, src_gen, tgt_model, tgt_gen, eval_ds, align_ds) in enumerate(pairs, 1):
        prog = f"{i}/{n_pairs}"
        pair_name = f"{src_model}_to_{tgt_model}"
        # one sub-directory per pair, one file per aligner+hyperparam combo
        pair_align_dir = os.path.join(args.out_dir, "alignments", eval_ds, pair_name)
        hyperparams = _build_hyperparams(args)
        aligner_tags = {a: _aligner_tag(a, hyperparams.get(a, {}))
                        for a in args.aligners}
        aligner_out_paths = {
            a: os.path.join(pair_align_dir, f"align_{align_ds}_{aligner_tags[a]}.pkl")
            for a in args.aligners
        }
        pending = [a for a in args.aligners
                   if not os.path.exists(aligner_out_paths[a]) or args.force]
        if not pending:
            print(f"[align_cache {prog}] skip {pair_name} eval={eval_ds} align={align_ds} (all exist)")
            continue

        # Load probe caches to get best layers and z-score stats
        src_probe_path = os.path.join(args.out_dir, "probes", eval_ds, f"{src_model}.pkl")
        tgt_probe_path = os.path.join(args.out_dir, "probes", eval_ds, f"{tgt_model}.pkl")
        if not os.path.exists(src_probe_path):
            print(f"  ERROR: missing probe cache {src_probe_path}")
            continue
        if not os.path.exists(tgt_probe_path):
            print(f"  ERROR: missing probe cache {tgt_probe_path}")
            continue
        with open(src_probe_path, "rb") as f:
            src_cache = pickle.load(f)
        with open(tgt_probe_path, "rb") as f:
            tgt_cache = pickle.load(f)

        Ls = src_cache["best_layer"]
        Lt = tgt_cache["best_layer"]

        print(f"\n[align_cache {prog}] {pair_name}  eval={eval_ds}  align={align_ds}")
        print(f"  src best layer: {Ls}  tgt best layer: {Lt}")

        # Load alignment hidden states
        # For same-dataset alignment, align_gen = eval_gen (src/tgt gen paths).
        # We need the gen paths for the alignment dataset.
        # These are passed via --align-gen-map file or we resolve from cache.
        # We resolve: look up the probe cache for align_ds
        src_align_probe_path = os.path.join(
            args.out_dir, "probes", align_ds, f"{src_model}.pkl")
        tgt_align_probe_path = os.path.join(
            args.out_dir, "probes", align_ds, f"{tgt_model}.pkl")
        if not os.path.exists(src_align_probe_path):
            print(f"  ERROR: missing align probe cache {src_align_probe_path}")
            continue
        if not os.path.exists(tgt_align_probe_path):
            print(f"  ERROR: missing align probe cache {tgt_align_probe_path}")
            continue
        with open(src_align_probe_path, "rb") as f:
            src_align_cache = pickle.load(f)
        with open(tgt_align_probe_path, "rb") as f:
            tgt_align_cache = pickle.load(f)

        # Load raw hidden states for alignment dataset at the best layers
        print(f"  loading align hidden states ({align_ds})")
        H_src_align, ids_src_align = _load_hidden(src_align_cache["gen_path"], token)
        H_tgt_align, ids_tgt_align = _load_hidden(tgt_align_cache["gen_path"], token)
        order_align = _align_ids(ids_src_align, ids_tgt_align)
        H_tgt_align = H_tgt_align[:, order_align]

        Xs_align = H_src_align[Ls].astype(np.float64)
        Xt_align = H_tgt_align[Lt].astype(np.float64)

        # Z-score using pool stats from probe cache (same 1500-example stats used
        # in Phase 3 evaluation) so M is fit and applied in the same coordinate system.
        N_align_rows = Xs_align.shape[0]
        mu_sa = np.array(src_align_cache["mu"])
        sd_sa = np.array(src_align_cache["sd"])
        mu_ta = np.array(tgt_align_cache["mu"])
        sd_ta = np.array(tgt_align_cache["sd"])
        Zs_align = (Xs_align - mu_sa) / sd_sa
        Zt_align = (Xt_align - mu_ta) / sd_ta
        # Rows the alignment may be fit on.  MUST exclude the Phase-3 eval rows: the
        # eval set is the probe cache's random eval_idx, and `pool` is its complement,
        # so pool is the only leak-free choice.  Taking np.arange(N_align) instead (the
        # first n rows in file order) overlaps eval_idx in n*n_eval/N rows -- 375 of 500
        # at n=1500 for NQ -- which for e2_rstar is a hard label leak, since it regresses
        # ent_t and the eval label is a binarization of that same ent_t (AUROC 0.95 vs
        # 0.74 leak-free).  This matches transfer.py:512's `sub = pool[:n]`.
        #
        # Only same-align can overlap: for align_ds != eval_ds the alignment rows are a
        # different dataset's questions, and eval_ds's pool indices are meaningless in
        # that row space, so the full range is both correct and safe there.
        if align_ds == eval_ds:
            align_all = np.array(src_cache["pool"])
        else:
            align_all = np.arange(N_align_rows)
        # N_align is the number of USABLE alignment rows (Phase 3 caps its grid with it),
        # which is now pool size, not the dataset row count.
        N_align = len(align_all)

        valid_grid = [n for n in n_grid if n <= N_align]

        # E2-R* is supervised by the TARGET entropy on the alignment dataset, and
        # is probe-aware (w_s enters via alpha_eff = lam/||w_s||^2), so it needs
        # two extra inputs the unsupervised aligners do not.
        if "e2_rstar" in pending:
            ent_t_align = _load_entropy(tgt_align_cache["gen_path"])[order_align]
            w_s_full = np.array(src_cache["probe_full"]["coef"])
            # probe-grid probes, keyed by probe training size (None where a
            # subset had a single class)
            src_probes_by_n = src_cache["probes_by_n"]

        print(f"  fitting alignment matrices for n in {valid_grid}  aligners={pending}")
        os.makedirs(pair_align_dir, exist_ok=True)
        pair_timing = timing["align_cache"].setdefault(pair_name, {})
        ds_timing = pair_timing.setdefault(f"{eval_ds}_probe", {}).setdefault(f"{align_ds}_align", {})

        for aligner in pending:
            tag = aligner_tags[aligner]
            alignment = {
                "src_model": src_model, "tgt_model": tgt_model,
                "eval_ds": eval_ds, "align_ds": align_ds,
                "aligner": aligner, "aligner_tag": tag,
                "src_best_layer": Ls, "tgt_best_layer": Lt,
                "N_align": N_align, "n_grid": valid_grid,
            }
            # Time each n separately: the reported cost must be ONE alignment fit at
            # the largest n, not the sum over the grid.  Ridge is dominated by the
            # d^3 solve and is nearly n-independent, so the grid total is ~5x the
            # single fit -- reporting the total would badly overstate transfer cost.
            per_n_s = {}
            t0_grid = time.time()
            for n in tqdm(valid_grid, desc=f"  {tag} n", ncols=80):
                sub = align_all[:n]
                t0_n = time.time()
                if aligner == "ridge":
                    M, b = _fit_ridge(Zt_align[sub], Zs_align[sub], alpha=args.alpha)
                elif aligner == "procrustes":
                    M, b = _fit_procrustes(Zt_align[sub], Zs_align[sub])
                elif aligner == "e2_rstar":
                    # rank-1: store the (a, beta) factors, not the d_t x d_s matrix
                    a_e2, beta_e2 = _fit_e2_rstar(
                        Zt_align[sub], w_s_full, ent_t_align[sub], lam=args.lambda_val)
                    per_n_s[n] = round(time.time() - t0_n, 3)
                    alignment[f"a_{n}"] = a_e2.tolist()
                    alignment[f"beta_{n}"] = beta_e2
                    alignment[f"w_{n}"] = w_s_full.tolist()
                    continue
                else:
                    raise ValueError(f"Unknown aligner: {aligner}")
                per_n_s[n] = round(time.time() - t0_n, 3)
                alignment[f"M_{n}"] = M.tolist()
                alignment[f"b_{n}"] = b.tolist()

            # E2-R* probe grid: alignment data fixed at the largest n, but the map
            # refit with EACH probe w_n.  Skipping this refit would freeze a, beta
            # across n and (because M is rank-1) leave AUROC exactly flat -- the
            # bug fixed in transfer.py:713.  See README_E2R.md section 4.
            if aligner == "e2_rstar":
                n_align_full = max(valid_grid)
                sub_full = align_all[:n_align_full]
                alignment["n_align_full"] = n_align_full
                alignment["rank1"] = True
                probe_per_n_s = {}
                for n in tqdm(valid_grid, desc=f"  {tag} probe-n", ncols=80):
                    entry = src_probes_by_n.get(n)
                    if entry is None:
                        continue
                    w_n = np.array(entry["coef"])
                    t0_n = time.time()
                    a_pn, beta_pn = _fit_e2_rstar(
                        Zt_align[sub_full], w_n, ent_t_align[sub_full],
                        lam=args.lambda_val)
                    probe_per_n_s[n] = round(time.time() - t0_n, 3)
                    alignment[f"probe_a_{n}"] = a_pn.tolist()
                    alignment[f"probe_beta_{n}"] = beta_pn
                    alignment[f"probe_w_{n}"] = w_n.tolist()
                ds_timing[f"{tag}_probe_per_n_s"] = probe_per_n_s

            grid_s = round(time.time() - t0_grid, 2)
            n_reported = max(per_n_s) if per_n_s else None
            fit_s = per_n_s.get(n_reported)

            out_path = aligner_out_paths[aligner]
            with open(out_path, "wb") as f:
                pickle.dump(alignment, f)
            print(f"  -> saved {out_path}  (fit n={n_reported}: {fit_s}s, "
                  f"whole grid: {grid_s}s)")

            ds_timing[tag] = fit_s
            ds_timing[f"{tag}_n"] = n_reported
            ds_timing[f"{tag}_per_n_s"] = per_n_s
            ds_timing[f"{tag}_grid_total_s"] = grid_s
        _save_timing(args.out_dir, timing)


# ============================================================================
# Phase 3: evaluate
# ============================================================================
# For each (pair, eval_ds, align_ds), load probe cache + alignment cache,
# compute alignment_grid and probe_grid AUROC curves, save json + pdf.

def phase_evaluate(args):
    n_grid = args.n_grid
    results_dir = os.path.join(args.out_dir, "results")
    timing = _load_timing(args.out_dir)
    timing.setdefault("evaluate", {})

    pairs = _resolve_pairs(args)
    n_pairs = len(pairs)

    for i, (src_model, src_gen, tgt_model, tgt_gen, eval_ds, align_ds) in enumerate(pairs, 1):
        prog = f"{i}/{n_pairs}"
        pair_name = f"{src_model}_to_{tgt_model}"
        tag = f"eval_{eval_ds}_align_{align_ds}"

        print(f"\n[evaluate {prog}] {pair_name}  {tag}")

        # Load probe caches
        src_probe_path = os.path.join(args.out_dir, "probes", eval_ds, f"{src_model}.pkl")
        tgt_probe_path = os.path.join(args.out_dir, "probes", eval_ds, f"{tgt_model}.pkl")
        for p in [src_probe_path, tgt_probe_path]:
            if not os.path.exists(p):
                print(f"  ERROR: missing {p}")
                continue
        with open(src_probe_path, "rb") as f:
            sc = pickle.load(f)
        with open(tgt_probe_path, "rb") as f:
            tc = pickle.load(f)

        # Load alignment caches — one file per aligner+hyperparam combo
        pair_align_dir = os.path.join(args.out_dir, "alignments", eval_ds, pair_name)
        hyperparams = _build_hyperparams(args)
        available_aligners = []
        align_caches = {}
        for a in ["ridge", "procrustes", "e2_rstar"]:
            tag = _aligner_tag(a, hyperparams.get(a, {}))
            p = os.path.join(pair_align_dir, f"align_{align_ds}_{tag}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    align_caches[a] = pickle.load(f)
                available_aligners.append(a)
        if not available_aligners:
            print(f"  ERROR: no alignment cache found in {pair_align_dir}")
            continue
        # use first available cache for metadata (layers, N_align, n_grid)
        ac = align_caches[available_aligners[0]]

        # Reconstruct eval features from cached z-score stats
        # Load raw hidden states for eval dataset
        H_src, ids_src = _load_hidden(sc["gen_path"], args.token)
        H_tgt, ids_tgt = _load_hidden(tc["gen_path"], args.token)
        order = _align_ids(ids_src, ids_tgt)
        H_tgt = H_tgt[:, order]

        Ls, Lt = sc["best_layer"], tc["best_layer"]
        eval_idx = np.array(sc["eval_idx"])
        pool = np.array(sc["pool"])
        ys = np.array(sc["y"])
        yt = np.array(tc["y"])[order]

        mu_s = np.array(sc["mu"])
        sd_s = np.array(sc["sd"])
        mu_t = np.array(tc["mu"])
        sd_t = np.array(tc["sd"])

        Xs = H_src[Ls].astype(np.float64)
        Xt = H_tgt[Lt].astype(np.float64)
        Zs = (Xs - mu_s) / sd_s
        Zt = (Xt - mu_t) / sd_t

        Zt_eval = Zt[eval_idx]
        yt_eval = yt[eval_idx]
        src_eval_auc = sc["src_eval_auc"]

        valid_grid = [n for n in n_grid if n <= len(pool) and n <= ac["N_align"]]

        # ---- probe grid ----
        # x-axis = probe training size n; alignment = fixed at max n_grid (1500)
        n_align_full = min(max(valid_grid), ac["N_align"])

        # pre-load the fixed alignment matrices for each aligner (each has its own cache).
        # e2_rstar is stored as rank-1 factors refit per probe size, so it is handled
        # per-n inside the loop instead.
        align_matrices = {
            a: (np.array(align_caches[a][f"M_{n_align_full}"]),
                np.array(align_caches[a][f"b_{n_align_full}"]))
            for a in available_aligners
            if not align_caches[a].get("rank1")
        }

        # load source eval hidden states for src_probe_pred and src AUROC curve
        Zs_eval = (Xs - mu_s) / sd_s
        Zs_eval = Zs_eval[np.array(sc["eval_idx"])]
        ys_eval = ys[eval_idx]

        # ---- check if aligner-independent native files already exist ----
        pair_out_dir = os.path.join(results_dir, eval_ds, pair_name)
        os.makedirs(pair_out_dir, exist_ok=True)
        native_curves_path = os.path.join(pair_out_dir, f"native_curves_{eval_ds}.json")
        native_preds_path = os.path.join(pair_out_dir, f"native_preds_{eval_ds}.json")
        need_native = not os.path.exists(native_curves_path)

        native_curves = {
            "src_layer": Ls, "tgt_layer": Lt,
            "src_layer_auc": sc["best_auc"], "tgt_layer_auc": tc["best_auc"],
            "src_eval_auc": src_eval_auc,
            "eval_ds": eval_ds,
            "n_grid": valid_grid,
            "curveA_native": [],
            "curveA_src": [],
        } if need_native else None

        native_preds = {
            "src_label": ys_eval.tolist(),
            "tgt_label": yt_eval.tolist(),
            "by_n": {},
        } if need_native else None

        pg = {"eval_ds": eval_ds, "align_ds": align_ds,
              "n_align_fixed": n_align_full,
              "aligners": available_aligners,
              "n_grid": valid_grid}
        for a in available_aligners:
            pg[f"curveB_{a}"] = []

        preds = {"by_n": {}}

        native_inference_by_n = {}
        aligned_inference_by_n = {}

        for n in valid_grid:
            sub = pool[:n]
            by_n_entry = {}

            # Curve A native: target probe of size n on target eval set
            if need_native:
                if len(np.unique(yt[sub])) < 2:
                    native_curves["curveA_native"].append(None)
                    tgt_probe_pred = None
                else:
                    clf = LogisticRegression(max_iter=1000).fit(Zt[sub], yt[sub])
                    # time only the predict (inference), not the fit
                    _t0 = time.time()
                    au = roc_auc_score(yt_eval, clf.predict_proba(Zt_eval)[:, 1])
                    native_inference_by_n[n] = round(time.time() - _t0, 4)
                    native_curves["curveA_native"].append(float(au))
                    tgt_probe_pred = clf.predict(Zt_eval).tolist()
                native_preds["by_n"][str(n)] = {"tgt_probe_pred": tgt_probe_pred}

            # Curve B: source probe of size n transferred through fixed alignment
            if sc["probes_by_n"].get(n) is None:
                if need_native:
                    native_curves["curveA_src"].append(None)
                    native_preds["by_n"][str(n)]["src_probe_pred"] = None
                for a in available_aligners:
                    pg[f"curveB_{a}"].append(None)
                    by_n_entry[f"{a}_pred"] = None
                preds["by_n"][str(n)] = by_n_entry
                continue

            a_s_n = np.array(sc["probes_by_n"][n]["coef"])
            c_s_n = sc["probes_by_n"][n]["intercept"]

            # Curve A src: source probe of size n on source eval set
            if need_native:
                src_scores = _probe_scores(Zs_eval, a_s_n, c_s_n)
                au_src = roc_auc_score(ys_eval, src_scores)
                native_curves["curveA_src"].append(float(au_src))
                native_preds["by_n"][str(n)]["src_probe_pred"] = (src_scores >= 0).astype(int).tolist()

            _t0 = time.time()
            for a in available_aligners:
                if align_caches[a].get("rank1"):
                    # e2_rstar: rank-1 factors refit with this n's probe.  Apply the
                    # map without materialising M (exactly equivalent, O(d) not O(d^2)).
                    cache_a = align_caches[a]
                    if f"probe_a_{n}" not in cache_a:
                        pg[f"curveB_{a}"].append(None)
                        by_n_entry[f"{a}_pred"] = None
                        continue
                    a_t, c_t = _e2_rstar_transfer(
                        np.array(cache_a[f"probe_a_{n}"]),
                        cache_a[f"probe_beta_{n}"],
                        np.array(cache_a[f"probe_w_{n}"]),
                        a_s_n, c_s_n)
                else:
                    M_full, b_full = align_matrices[a]
                    a_t = M_full @ a_s_n
                    c_t = float(b_full @ a_s_n) + c_s_n
                scores = _probe_scores(Zt_eval, a_t, c_t)
                au = roc_auc_score(yt_eval, scores)
                pg[f"curveB_{a}"].append(float(au))
                by_n_entry[f"{a}_pred"] = (scores >= 0).astype(int).tolist()
            aligned_inference_by_n[n] = round(time.time() - _t0, 4)

            preds["by_n"][str(n)] = by_n_entry

        # save native files if newly computed
        if need_native:
            _save_json(native_curves_path, native_curves)
            _save_json(native_preds_path, native_preds)
            pair_timing = timing["evaluate"].setdefault(pair_name, {})
            pair_timing[eval_ds] = {
                "native_inference_s":  {str(n): v for n, v in native_inference_by_n.items()},
                "aligned_inference_s": {str(n): v for n, v in aligned_inference_by_n.items()},
            }
            _save_timing(args.out_dir, timing)

        aligner_suffix = "_".join(
            _aligner_tag(a, hyperparams.get(a, {}))
            for a in available_aligners
        )
        out_pg = os.path.join(pair_out_dir, f"probe_grid_align_{align_ds}_{aligner_suffix}.json")
        _save_json(out_pg, pg)

        out_preds = os.path.join(pair_out_dir, f"predictions_align_{align_ds}_{aligner_suffix}.json")
        _save_json(out_preds, preds)
        print(f"  -> saved probe_grid + predictions for {pair_name} ({tag}) [{aligner_suffix}]")


# ============================================================================
# Phase 4: summary plots
# ============================================================================

C_NATIVE      = "#1a1a1a"
C_RIDGE_SAME  = "#4472c4"
C_RIDGE_CROSS = "#70ad47"
C_SRC_BASE    = "#888888"
SURFACE       = "white"
INK_PRI       = "#1a1a1a"
INK_SEC       = "#404040"
GRIDLINE      = "#d0d0d0"

# extra colors for additional aligners
C_ALIGNER_SAME  = ["#4472c4", "#ed7d31", "#a020f0", "#00b0f0"]
C_ALIGNER_CROSS = ["#70ad47", "#c00000", "#ff8c00", "#00b0b0"]


# Ordered 5-model groups for the structured layout:
#   rows 0-3: cross-family (4 sources × 4 targets)
#   row 4:    llama-family (llama-3.2-1b → llama-3.1-8b only)
_CROSS_SOURCES = ["gemma-4-12b", "llama-3.1-8b", "mistral-nemo", "phi-4", "qwen3-8b"]
_CROSS_TARGETS = ["gemma-4-12b", "llama-3.1-8b", "mistral-nemo", "phi-4", "qwen3-8b"]
_LLAMA_FAMILY  = [("llama-3.2-1b", "llama-3.1-8b")]


def _structured_grid(results_dir, eval_ds):
    """Return a 5×4 grid of pair_keys (or None for empty cells).

    Layout:
      rows 0-4 : one source per row (gemma, llama-3.1, mistral, phi-4, qwen3)
      cols 0-3 : targets (excluding same-model diagonal)
      row 4    : llama family transfer in col 0; cols 1-3 are None
    """
    ds_dir = os.path.join(results_dir, eval_ds)
    present = set(
        p for p in os.listdir(ds_dir)
        if os.path.isdir(os.path.join(ds_dir, p))
    ) if os.path.isdir(ds_dir) else set()

    grid = []
    for src in _CROSS_SOURCES:
        targets = [t for t in _CROSS_TARGETS if t != src]   # 4 targets
        row = []
        for tgt in targets:
            key = f"{src}_to_{tgt}"
            row.append(key if key in present else None)
        grid.append(row)

    # llama-family row
    fam_row = []
    for src, tgt in _LLAMA_FAMILY:
        key = f"{src}_to_{tgt}"
        fam_row.append(key if key in present else None)
    while len(fam_row) < 4:
        fam_row.append(None)
    grid.append(fam_row)

    return grid   # list of 5 rows, each 4 elements


def _discover_pairs(results_dir, eval_ds):
    """Return sorted list of pair directory names found under results/<eval_ds>/."""
    ds_dir = os.path.join(results_dir, eval_ds)
    if not os.path.isdir(ds_dir):
        return []
    return sorted(p for p in os.listdir(ds_dir)
                  if os.path.isdir(os.path.join(ds_dir, p)))


def _pair_title(pair_key):
    """Convert 'llama-3.1-8b_to_qwen3-8b' to a readable title and kind."""
    SCALE_PAIRS = {"llama-3.2-1b_to_llama-3.1-8b"}
    src, _, tgt = pair_key.partition("_to_")
    title = f"{src} → {tgt}"
    kind = "cross-scale" if pair_key in SCALE_PAIRS else "cross-family"
    return title, kind


def make_run_variant_selector(grid_type, hyperparams):
    """Selector for phase_summary: pick the grid file matching THIS run's hyperparams.

    A probe_grid filename is  <grid_type>_align_<ds>_<tag1>[_<tag2>...].json  where each
    tag encodes one aligner + its hyperparams (e.g. "ridge_a1e4", "procrustes", "e2_rstar_l1e3").
    A file belongs to the current run iff every hyperparam-bearing tag in it matches the
    tag implied by the current args (so a ridge_a1e1 file is rejected when alpha=1e4).

    Returns a callable (pair_dir, ds_tag) -> (label, pg_dict) or None.
    """
    expected_ridge = _aligner_tag("ridge", hyperparams.get("ridge", {}))
    expected_e2    = _aligner_tag("e2_rstar", hyperparams.get("e2_rstar", {}))

    def _select(pair_dir, ds_tag):
        if not os.path.isdir(pair_dir):
            return None
        prefix = f"{grid_type}_align_{ds_tag}_"
        for fname in sorted(os.listdir(pair_dir)):
            if not (fname.startswith(prefix) and fname.endswith(".json")):
                continue
            suffix = fname[len(prefix):-len(".json")]
            # reject files whose ridge/e2_rstar hyperparams differ from this run's
            m = re.search(r'ridge_a1e-?\d+', suffix)
            if m and m.group(0) != expected_ridge:
                continue
            # [0-9.eE+-], not [\d.]: the tag is 1e3-style, not 1000.0-style
            m = re.search(r'e2_rstar_l[0-9.eE+-]+', suffix)
            if m and m.group(0) != expected_e2:
                continue
            return suffix, _load_json(os.path.join(pair_dir, fname))
        return None

    return _select


# Symbols in the order they are annotated on panel titles.
_HPARAM_SYMBOLS = ["α", "λ"]


def _hparams_from_label(label):
    """Pull the swept hyperparams out of an aligner label, as {symbol: value_str}.

    Handles both label flavours in use:
      * Phase 4 passes the grid-file suffix, e.g. 'ridge_a1e4_e2_rstar_l1000.0'
      * sweep_summary passes a prose label, e.g. 'ridge alpha=1e4 (best)' or
        'E2-R* lambda=1e5 (best)'

    Returns {} when the label carries no recognised hyperparam (or there is no label
    at all), so callers can skip the annotation instead of printing a bogus value.
    """
    out = {}
    if not label:
        return out
    # The lookbehind keeps the alpha pattern from firing on the trailing 'a' of
    # 'lambda=1e5', which would report that lambda as an alpha.
    m = re.search(r'(?<![A-Za-z])a(?:lpha)?[=_]?\s*(1e-?\d+)', label)
    if m:
        out["α"] = m.group(1)
    m = re.search(r'(?:e2_rstar_l|lambda[=_]?\s*|λ[=_]?\s*)'
                  r'([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)', label)
    if m:
        out["λ"] = _fmt_pow10(m.group(1))
    return out


def _make_summary_fig(eval_ds, grid_type, results_dir, select_variant, hparam_note,
                      cross_datasets=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    _SHORT = {
        "gemma-4-12b":  "Gemma-4-12B",
        "llama-3.1-8b": "Llama-3.1-8B",
        "mistral-nemo": "Mistral-Nemo",
        "phi-4":        "Phi-4",
        "qwen3-8b":     "Qwen3-8B",
        "llama-3.2-1b": "Llama-3.2-1B",
    }

    # build structured 5×4 grid
    grid = _structured_grid(results_dir, eval_ds)
    nrows, ncols = len(grid), len(grid[0])

    row_labels = [_SHORT.get(s, s) for s in _CROSS_SOURCES] + [
        _SHORT.get(src, src) for src, _ in _LLAMA_FAMILY
    ]

    same_ds      = eval_ds
    if cross_datasets is None:
        cross_datasets = ["squad"] if eval_ds == "nq" else ["nq"]
    ds_label     = eval_ds.upper()
    cross_label  = " / ".join(d.upper() for d in cross_datasets)
    grid_label  = "Probe grid"
    xlabel      = "source probe training examples"

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5),
                             facecolor=SURFACE, constrained_layout=True)

    fig.suptitle(
        f"SE probe transfer — {grid_label}  |  eval: {ds_label}  |  {hparam_note}  |  "
        f"same-align: {ds_label} (solid)  cross-align: {cross_label} (dashed)",
        fontsize=11, fontweight="normal", color=INK_PRI,
    )

    for row_idx, row in enumerate(grid):
        for col_idx, pair_key in enumerate(row):
            ax = axes[row_idx][col_idx]

            # empty cell
            if pair_key is None:
                ax.axis("off")
                continue

            pair_dir = os.path.join(results_dir, eval_ds, pair_key)
            _, kind = _pair_title(pair_key)
            tgt = pair_key.partition("_to_")[2]
            tgt_short = _SHORT.get(tgt, tgt)

            ax.set_facecolor(SURFACE)
            for spine in ax.spines.values():
                spine.set_color("#b0b0b0")
                spine.set_linewidth(0.8)
            ax.grid(which="major", color=GRIDLINE, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)

            # load aligner-independent native curves
            native_curves_path = os.path.join(pair_dir, f"native_curves_{eval_ds}.json")
            if not os.path.exists(native_curves_path):
                ax.set_title(f"→ {tgt_short}\n(missing)", fontsize=8, color="red")
                ax.axis("off")
                continue

            native_data = _load_json(native_curves_path)
            g       = native_data["n_grid"]
            native  = [v if v is not None else float("nan") for v in native_data["curveA_native"]]
            ceiling = native_data.get("tgt_layer_auc", float("nan"))

            ax.plot(g, native, color=C_NATIVE, linewidth=1.5, marker="o",
                    markersize=4, markeredgewidth=0, label="A: native target", zorder=3)

            src_curve = native_data.get("curveA_src")
            if src_curve:
                src_vals = [v if v is not None else float("nan") for v in src_curve]
                ax.plot(g, src_vals, color=C_SRC_BASE, linewidth=1.5, marker="D",
                        markersize=4, markeredgewidth=0, linestyle="-",
                        label="A: source probe on src", zorder=3)

            # one grid file for same-align; one per cross-align dataset
            same_pick   = select_variant(pair_dir, same_ds)
            cross_picks = [(ds, select_variant(pair_dir, ds)) for ds in cross_datasets]

            if same_pick is not None:
                same_label, same = same_pick
                for i, a in enumerate(same.get("aligners", ["ridge"])):
                    curve = same.get(f"curveB_{a}")
                    if curve:
                        c = C_ALIGNER_SAME[i % len(C_ALIGNER_SAME)]
                        ax.plot(g, curve, color=c, linewidth=1.5, marker="s",
                                markersize=4, markeredgewidth=0, linestyle="-",
                                label=f"B: {same_label} ({ds_label})", zorder=3)

            color_idx = 0
            for ds, cross_pick in cross_picks:
                if cross_pick is not None:
                    cross_label_txt, cross = cross_pick
                    xg = cross["n_grid"]
                    for i, a in enumerate(cross.get("aligners", ["ridge"])):
                        curve = cross.get(f"curveB_{a}")
                        if curve:
                            c = C_ALIGNER_CROSS[color_idx % len(C_ALIGNER_CROSS)]
                            ax.plot(xg, curve, color=c, linewidth=1.5, marker="s",
                                    markersize=4, markeredgewidth=0, linestyle="--",
                                    label=f"B: {cross_label_txt} ({ds.upper()})", zorder=3)
                            color_idx += 1

            ax.set_xscale("log")
            ax.set_xlabel(xlabel, fontsize=7, color=INK_SEC)
            ax.set_ylabel("eval AUROC", fontsize=7, color=INK_SEC)
            ax.tick_params(colors=INK_SEC, labelsize=7, length=3)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_tick_params(which="minor", bottom=False)
            ax.set_xticks(g)
            ax.set_xticklabels([str(v) for v in g], fontsize=6.5, rotation=30)
            ax.set_ylim(0.45, 0.95)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            # Which hyperparams each of the two B-curves came from (α for ridge, λ for
            # E2-R*). With a fixed-hyperparam selector this is the same everywhere; with
            # sweep_summary's best-value selector it varies per panel, so it has to be
            # shown per panel. Only symbols actually present in a label are annotated.
            same_hp = _hparams_from_label(same_pick[0] if same_pick else None)
            first_cross = next((p for _, p in cross_picks if p is not None), None)
            cross_hp = _hparams_from_label(first_cross[0] if first_cross else None)
            notes = []
            for sym in _HPARAM_SYMBOLS:
                if sym in same_hp or sym in cross_hp:
                    parts = [f"{ds_label}={same_hp.get(sym, '-')}"]
                    if cross_label:
                        parts.append(f"{cross_label}={cross_hp.get(sym, '-')}")
                    notes.append(f"{sym}: " + ", ".join(parts))
            hp_note = ("  " + "  ".join(notes)) if notes else ""

            ax.set_title(f"→ {tgt_short}\n[{kind}]  ceil={ceiling:.3f}{hp_note}",
                         fontsize=7.5, color=INK_PRI, pad=3)
            ax.legend(fontsize=5.5, frameon=True, framealpha=0.9,
                      edgecolor="#cccccc", loc="upper left")

        # row label on the left side of each row
        axes[row_idx][0].set_ylabel(
            f"src: {row_labels[row_idx]}\neval AUROC",
            fontsize=7, color=INK_SEC,
        )

    return fig


def phase_summary(args):
    results_dir = os.path.join(args.out_dir, "results")
    plots_dir = os.path.join(args.out_dir, "summary_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Figures reflect exactly ONE experiment configuration: the aligner hyperparams
    # passed to this invocation. Cross-hyperparameter (sweep) comparison lives in
    # sep.transfer.sweep_summary, not here.
    hyperparams = _build_hyperparams(args)
    run_tag = _aligner_tag("ridge", hyperparams["ridge"])
    selector = make_run_variant_selector("probe_grid", hyperparams)
    hparam_note = f"aligner hyperparams: {run_tag}"

    import matplotlib.pyplot as plt
    cross_map = {}  # eval_ds -> [align_ds, ...]
    for mapping in (args.cross_align_dataset or []):
        ed, ad = mapping.split(":")
        cross_map.setdefault(ed, []).append(ad)

    for eval_ds in args.datasets:
        cross_ds_list = cross_map.get(eval_ds) or None
        fig = _make_summary_fig(eval_ds, "probe_grid", results_dir,
                                selector, hparam_note, cross_datasets=cross_ds_list)
        stem = f"summary_{eval_ds}_probe_grid_{run_tag}"
        for ext in ("pdf", "png"):
            path = os.path.join(plots_dir, f"{stem}.{ext}")
            dpi = 200 if ext == "pdf" else 150
            fig.savefig(path, format=ext, dpi=dpi,
                        bbox_inches="tight", facecolor=SURFACE)
            print(f"Saved: {path}")
        plt.close(fig)

    # Venn stats + figures for THIS run's hyperparams only (same tag as the figures
    # above). Both steps are also runnable standalone:
    #   python -m sep.transfer.compute_venn --out-dir ... --aligner-suffix ridge_a1e4
    #   python -m sep.transfer.plot_venn_grid    --out-dir ... --alpha 1e4
    if getattr(args, "skip_venn", False):
        print("  [venn] skipped: --skip-venn")
    else:
        from sep.transfer.compute_venn import compute_all as compute_venn_all
        venn_files = compute_venn_all(args.out_dir, eval_datasets=args.datasets,
                                      aligner_suffix=run_tag, verbose=False)
        if not venn_files:
            print(f"  [venn] skipped: no predictions_align_*_{run_tag}.json found "
                  f"(run evaluate for alpha={args.alpha:g} first)")
        else:
            print(f"  [venn] computed {len(venn_files)} venn stat files ({run_tag})")
            try:
                from sep.transfer.plot_venn_grid import build_all as build_venn_figs
                all_align_ds = list(args.datasets)
                for ad in [ad for ads in cross_map.values() for ad in ads]:
                    if ad not in all_align_ds:
                        all_align_ds.append(ad)
                written = build_venn_figs(args.out_dir, run_tag, n_values=args.n_grid,
                                          datasets=tuple(args.datasets),
                                          align_datasets=tuple(all_align_ds),
                                          verbose=False)
                print(f"  [venn] saved {len(written)} files to "
                      f"{os.path.join(args.out_dir, 'summary_plots', 'venn_' + run_tag)}")
            except ImportError as e:
                print(f"  [venn] stats written, figures skipped ({e}); "
                      f"pip install matplotlib-venn")

    # Auto-generate timing table only when the T1/T2 inputs are actually present.
    # An existing-but-empty data_generation_timing/ would otherwise yield an empty
    # timing_collect_data.json and timing tables with blank T1/T2 columns.
    data_gen_timing_dir = os.path.join(args.out_dir, "data_generation_timing")
    pair_list = os.path.join(_repo_root(), "slurm", "inputs", "pair_list.txt")
    timing_inputs = glob.glob(os.path.join(data_gen_timing_dir, "*.json"))
    if timing_inputs and os.path.exists(pair_list):
        from sep.transfer.make_timing_table import build_tables
        timing_json = os.path.join(args.out_dir, "timing.json")
        out_json = os.path.join(args.out_dir, "timing_collect_data.json")
        build_tables(data_gen_timing_dir, timing_json, pair_list, out_json, args.out_dir,
                     datasets=tuple(args.datasets))
        print(f"Timing tables saved to {args.out_dir}")
    else:
        if not os.path.isdir(data_gen_timing_dir):
            print(f"  [timing table] skipped: {data_gen_timing_dir} not found")
        elif not timing_inputs:
            print(f"  [timing table] skipped: no *.json in {data_gen_timing_dir} "
                  f"(run slurm/run_collect_timing.sh first)")
        if not os.path.exists(pair_list):
            print(f"  [timing table] skipped: pair_list not found at {pair_list}")


# ============================================================================
# CLI
# ============================================================================

def _common_args(p):
    p.add_argument("--out-dir", required=True,
                   help="root cache/results directory")
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--n-grid", type=int, nargs="+",
                   default=[50, 100, 200, 400, 800, 1500])
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1e3)
    p.add_argument("--lambda-val", type=float, default=1.0,
                   help="lambda hyperparameter for the e2_rstar aligner")
    p.add_argument("--force", action="store_true",
                   help="recompute even if cache exists")
    _root = _repo_root()
    p.add_argument("--model-paths",
                   default=os.path.join(_root, "slurm", "inputs", "model_paths.txt"),
                   help="text file: dataset  model  path")
    p.add_argument("--datasets", nargs="+", required=True,
                   help="dataset names to expand, e.g. --datasets squad nq")
    p.add_argument("--cross-align-dataset", nargs="*", default=[],
                   help="eval_ds:align_ds mappings, e.g. squad:nq nq:squad")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="phase", required=True)

    p1 = sub.add_parser("probe_cache", help="Phase 1: fit probes for all models")
    _common_args(p1)

    p2 = sub.add_parser("align_cache", help="Phase 2: fit alignment matrices")
    _common_args(p2)
    p2.add_argument("--pair-list",
                    default=os.path.join(_repo_root(), "slurm", "inputs", "pair_list.txt"),
                    help="text file: src_model tgt_model  (or legacy 4-field format)")
    p2.add_argument("--aligners", nargs="+", default=["ridge"],
                    choices=["ridge", "procrustes", "e2_rstar"],
                    help="alignment methods to fit (default: ridge only)")

    p3 = sub.add_parser("evaluate", help="Phase 3: compute AUROC curves")
    _common_args(p3)
    p3.add_argument("--pair-list",
                    default=os.path.join(_repo_root(), "slurm", "inputs", "pair_list.txt"))

    p4 = sub.add_parser("summary", help="Phase 4: plot summary figures for THIS run's hyperparams")
    _common_args(p4)
    p4.add_argument("--skip-venn", action="store_true",
                    help="skip the venn stats + figures step")

    args = p.parse_args()

    if args.phase == "probe_cache":
        phase_probe_cache(args)
    elif args.phase == "align_cache":
        phase_align_cache(args)
    elif args.phase == "evaluate":
        phase_evaluate(args)
    elif args.phase == "summary":
        phase_summary(args)


if __name__ == "__main__":
    main()
