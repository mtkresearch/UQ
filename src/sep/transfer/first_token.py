"""A1 -- first-token-confidence baseline (checklist item A1 / task 23).

Decision rule (from CHECKLIST.md): a transferred/native SE probe must beat a
*free* confidence scalar read straight off generation, else the probe machinery
is unjustified.

The paper-exact phi_first (Fadeeva-style top-K=100 renormalized-logit entropy at
the first content token) needs the full top-K logit vector. Our cached
generations store only ``token_log_likelihoods`` -- the log-prob of each
*greedy-chosen* answer token. Because greedy decoding takes the argmax, that
first value is exactly log p(top-1), so we can still derive genuinely free
confidence scalars from cache:

  * u_first  = 1 - exp(logprob[first content token])   (1 - top-1 prob)
  * u_mean   = -mean(token_log_likelihoods)             (length-norm surprisal)
  * u_min    = -min(token_log_likelihoods)              (least-confident token)

Each is oriented so *higher = more uncertain*. We report, per dataset:
  - AUROC of each free scalar for (a) SE detection and (b) error detection,
  - the native SE / correctness *probe* AUROC at the best SE layer,
so the head-to-head "does the probe beat a free scalar" is explicit.

The strict paper phi_first (top-K entropy) is left as a regen-required follow-up;
u_first is its cache-available, strictly-cheaper cousin (top-1 vs top-100).
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


def _first_content_idx(lls):
    """First greedy token that isn't deterministic (logprob ~ 0 => prob ~ 1).

    Templates/formatting tokens are emitted with prob 1 (logprob 0). The first
    content-bearing token is the first with meaningfully negative log-prob.
    """
    for i, v in enumerate(lls):
        if v < -1e-6:
            return i
    return 0


def free_scalars(gens, ids):
    """Return dict of oriented (higher=more uncertain) free confidence scalars."""
    u_first, u_mean, u_min = [], [], []
    for k in ids:
        lls = np.asarray(gens[k]["most_likely_answer"]["token_log_likelihoods"],
                         dtype=np.float64)
        if lls.size == 0:
            u_first.append(0.0); u_mean.append(0.0); u_min.append(0.0)
            continue
        fi = _first_content_idx(lls)
        u_first.append(1.0 - float(np.exp(lls[fi])))
        u_mean.append(float(-lls.mean()))
        u_min.append(float(-lls.min()))
    return {
        "u_first": np.asarray(u_first),
        "u_mean": np.asarray(u_mean),
        "u_min": np.asarray(u_min),
    }


def accuracies(gens, ids):
    return np.asarray(
        [float(gens[k]["most_likely_answer"]["accuracy"]) for k in ids],
        dtype=np.float64,
    )


def probe_auroc(H, y, scaler_frac=0.7, seed=0):
    """Best-layer standardized logistic-probe AUROC for target y (held-out)."""
    rng = np.random.default_rng(seed)
    n = H.shape[1]
    idx = rng.permutation(n)
    tr, te = idx[: int(scaler_frac * n)], idx[int(scaler_frac * n):]
    best_auc, best_layer = -np.inf, 0
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
    return best_layer, best_auc, (tr, te)


def scalar_auroc(u, y, split=None):
    """AUROC of scalar u predicting binary y. If split given, eval on test only."""
    if split is not None:
        _, te = split
        u, y = u[te], y[te]
    if len(np.unique(y)) < 2:
        return 0.5
    return roc_auc_score(y, u)


def run(gen_path, token, seed):
    gens_key_path = gen_path
    with open(gen_path, "rb") as f:
        gens = pickle.load(f)
    ids = list(gens.keys())

    ents = load_entropy(gen_path)
    assert len(ents) == len(ids), f"ent/id mismatch {len(ents)} vs {len(ids)}"
    thr = best_split(ents)
    y_se = binarize(ents, thr)

    acc = accuracies(gens, ids)
    y_err = (acc < 0.5).astype(np.int64)  # 1 = incorrect answer

    H, ids_h = load_hidden(gen_path, token)
    assert ids_h == ids, "hidden-state ids not aligned with gen order"

    scal = free_scalars(gens, ids)

    # Native probes at their own best layer (held-out AUROC).
    se_layer, se_probe_auc, se_split = probe_auroc(H, y_se, seed=seed)
    err_layer, err_probe_auc, err_split = probe_auroc(H, y_err, seed=seed)

    out = {
        "gen_path": gen_path,
        "token": token,
        "n": len(ids),
        "se_threshold": float(thr),
        "se_base_rate": float(y_se.mean()),
        "error_rate": float(y_err.mean()),
        "probe": {
            "se_detection": {"layer": int(se_layer), "auroc": float(se_probe_auc)},
            "error_detection": {"layer": int(err_layer), "auroc": float(err_probe_auc)},
        },
        "free_scalars": {},
    }
    # Evaluate free scalars on the SAME held-out split as the probe for fairness.
    for name, u in scal.items():
        out["free_scalars"][name] = {
            "se_detection_auroc": float(scalar_auroc(u, y_se, se_split)),
            "error_detection_auroc": float(scalar_auroc(u, y_err, err_split)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", nargs="+", required=True,
                    help="validation_generations.pkl paths (one per dataset)")
    ap.add_argument("--names", nargs="+", required=True,
                    help="dataset label per --gens entry")
    ap.add_argument("--token", default="slt", choices=["slt", "tbg"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = apply_yaml_config(ap)
    assert len(args.gens) == len(args.names)

    results = {}
    for name, gp in zip(args.names, args.gens):
        print(f"[{name}] {gp}")
        results[name] = run(gp, args.token, args.seed)
        r = results[name]
        print(f"  SE   probe(L{r['probe']['se_detection']['layer']})="
              f"{r['probe']['se_detection']['auroc']:.3f}  "
              f"free: " + "  ".join(
                  f"{k}={v['se_detection_auroc']:.3f}"
                  for k, v in r["free_scalars"].items()))
        print(f"  ERR  probe(L{r['probe']['error_detection']['layer']})="
              f"{r['probe']['error_detection']['auroc']:.3f}  "
              f"free: " + "  ".join(
                  f"{k}={v['error_detection_auroc']:.3f}"
                  for k, v in r["free_scalars"].items()))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
