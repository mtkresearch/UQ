"""Confound A -- router-as-topic-shortcut control for MoE models.

Concern: OLMoE's router selects experts per token, and expert selection may
correlate with topic/domain, which in turn correlates with semantic entropy
(harder/more ambiguous questions cluster in certain topics). If so, a
classifier reading ONLY router state (which experts fired, how concentrated
the routing distribution was) could mimic a hidden-state SE/correctness probe
without reading any model-internal uncertainty signal -- a shortcut, not
evidence that "SE lives in the MoE's representations" the way it does for
dense Qwen-Qwen transfer.

Router-only features per example, aggregated over the generated-token span
(prefill/topic tokens excluded -- see `generate_olmoe.py`):
  * mean softmax routing distribution per layer (n_layers x num_experts)
  * mean per-token routing entropy per layer (n_layers)
Both computed from `router_logits` (n_layers, n_gen_tokens, num_experts).

Pass criterion: router-only AUROC must NOT approach the hidden-state probe's
AUROC (same eval protocol: single best layer via `best_se_layer`, on the same
train/test split). A small gap (router-only ~ hidden probe) would mean the
"probe" is largely reading a topic/routing shortcut, not internal uncertainty
representations -- exactly the PARALLAX-style lexical-leakage worry, applied
to routing state instead of prompt text.
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
from sep.transfer.transfer import best_se_layer, best_split, binarize, load_entropy


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
    y_err = (acc < 0.5).astype(np.int64)
    return gens, ids, y_se, y_err


def router_features(gens, ids):
    """(n_examples, n_layers*(num_experts+1)) -- mean routing dist + entropy per layer."""
    feats = []
    for k in ids:
        logits = gens[k]["router_logits"].astype(np.float64)  # (L, T, E)
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        mean_dist = probs.mean(1)  # (L, E)
        ent = -(probs * np.log(probs + 1e-12)).sum(-1).mean(1)  # (L,)
        feats.append(np.concatenate([mean_dist.ravel(), ent]))
    return np.stack(feats)


def router_top1_hist_features(gens, ids):
    """Alternative hard-routing feature: normalized top-1 expert histogram per layer."""
    feats = []
    for k in ids:
        logits = gens[k]["router_logits"]  # (L, T, E)
        L, _, E = logits.shape
        top1 = np.argmax(logits, axis=-1)  # (L, T)
        hist = np.stack([np.bincount(top1[l], minlength=E) for l in range(L)]).astype(np.float64)
        hist /= hist.sum(-1, keepdims=True)
        feats.append(hist.ravel())
    return np.stack(feats)


def classifier_auroc(X, y, seed=0, frac=0.7):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    tr, te = idx[: int(frac * n)], idx[int(frac * n):]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xz = (X - mu) / sd
    clf = LogisticRegression(max_iter=2000).fit(Xz[tr], y[tr])
    try:
        return float(roc_auc_score(y[te], clf.predict_proba(Xz[te])[:, 1]))
    except ValueError:
        return 0.5


def hidden_probe_auroc(gen_path, token, y, seed):
    H, ids_h = load_hidden(gen_path, token)
    _, auc = best_se_layer(H, y, seed=seed)
    return float(auc)


def run(gen_path, token, seed, gap_threshold):
    gens, ids, y_se, y_err = load_labels(gen_path)
    print(f"n={len(ids)}  SE base-rate={y_se.mean():.3f}  error-rate={y_err.mean():.3f}")

    X_soft = router_features(gens, ids)
    X_hard = router_top1_hist_features(gens, ids)
    print(f"router soft-feature dim={X_soft.shape[1]}  hard-feature dim={X_hard.shape[1]}")

    out = {"gen_path": gen_path, "token": token, "n": len(ids), "seed": seed,
           "se_base_rate": float(y_se.mean()), "error_rate": float(y_err.mean()),
           "gap_threshold": gap_threshold}

    for metric_name, y in (("se", y_se), ("correctness", y_err)):
        router_soft = classifier_auroc(X_soft, y, seed=seed)
        router_hard = classifier_auroc(X_hard, y, seed=seed)
        hidden = hidden_probe_auroc(gen_path, token, y, seed=seed)
        gap = hidden - max(router_soft, router_hard)
        out[metric_name] = {
            "router_soft_auroc": router_soft,
            "router_hard_auroc": router_hard,
            "hidden_probe_auroc": hidden,
            "gap": gap,
            "verdict": ("PASS -- router-only does not approach hidden probe"
                        if gap >= gap_threshold else
                        "FAIL -- router-only AUROC approaches/exceeds hidden probe "
                        "(possible routing-as-topic-shortcut)"),
        }
        print(f"\n[{metric_name.upper()}] router(soft)={router_soft:.3f} "
              f"router(hard)={router_hard:.3f} hidden_probe={hidden:.3f} "
              f"gap={gap:.3f}\n  -> {out[metric_name]['verdict']}")

    return out


def main():
    ap = argparse.ArgumentParser(description="Confound A: router-as-topic-shortcut control.")
    ap.add_argument("--gen", required=True, help="OLMoE validation_generations.pkl")
    ap.add_argument("--token", default="slt", choices=["slt", "tbg"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap-threshold", type=float, default=0.05,
                     help="Minimum (hidden_probe - router_only) AUROC gap required to pass.")
    ap.add_argument("--out", required=True)
    args = apply_yaml_config(ap)

    out = run(args.gen, args.token, args.seed, args.gap_threshold)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
