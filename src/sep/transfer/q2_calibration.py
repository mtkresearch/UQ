"""Q2 -- ranking vs calibration of the transferred probe (checklist Q2).

We have shown the transferred SE probe preserves *ranking* (AUROC ~ native, even
beats it). Q2 asks the recipe-defining question: does it also preserve
*calibration*? A cross-scale affine map carries the probe direction, but the
target-space logit scale and the decision threshold (bias) need not land where a
native target probe would. If ranking transfers but calibration does not, the
"zero target labels" claim needs an asterisk: you still need a small labeled
budget to set the threshold / temperature.

We measure, per dataset, mean +- std over seeds:
  * Ranking:    AUROC (should be preserved).
  * Calibration as-is: ECE (15-bin), Brier, NLL of sigmoid(transferred logit),
    vs the native target probe (which is calibrated by construction).
  * Threshold cost: accuracy at the probe's own threshold (logit=0) vs accuracy
    at the ORACLE threshold picked on eval labels -- the gap is what miscalibration
    of the *bias* costs a zero-label user.
  * Label budget: fit a 1-D Platt recalibration p=sigmoid(A*logit+B) on k labeled
    target examples; report ECE and threshold-accuracy vs k. The smallest k that
    closes the calibration gap is the answer to "how many labels for threshold-setting".

Uses the tuned alpha=1e4 map and the same prep() pipeline as the conditioning study.
"""
import argparse
from sep.uncertainty.utils.config import apply_yaml_config
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sep.transfer.conditioning import prep, ridge_map_parts, transfer_with_M


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def ece(p, y, n_bins=15):
    """Expected calibration error, equal-width bins on [0,1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        e += (m.sum() / n) * abs(acc - conf)
    return float(e)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def nll(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def acc_at_threshold(logit, y, thr):
    return float(((logit >= thr).astype(int) == y).mean())


def oracle_threshold(logit, y):
    """Threshold on the logit maximizing accuracy (eval-set oracle, upper bound)."""
    order = np.argsort(logit)
    ls = logit[order]
    best_thr, best_acc = 0.0, -1.0
    cands = np.concatenate([[ls[0] - 1.0], (ls[:-1] + ls[1:]) / 2, [ls[-1] + 1.0]])
    for t in cands:
        a = acc_at_threshold(logit, y, t)
        if a > best_acc:
            best_acc, best_thr = a, float(t)
    return best_thr, best_acc


def platt_fit(logit_cal, y_cal):
    """1-D logistic recalibration: returns (A, B) for p=sigmoid(A*logit+B)."""
    if len(np.unique(y_cal)) < 2:
        return 1.0, 0.0
    lr = LogisticRegression(max_iter=1000).fit(logit_cal.reshape(-1, 1), y_cal)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def run_seed(source_gen, target_gen, token, seed, n_eval, alpha, n_align,
             k_grid):
    d = prep(source_gen, target_gen, token, seed, n_eval, alpha)
    Zt, yt, pool, eval_idx = d["Zt"], d["yt"], d["pool"], d["eval_idx"]
    Zs, a_s, c_s = d["Zs"], d["a_s"], d["c_s"]
    Zt_eval, yt_eval = Zt[eval_idx], yt[eval_idx]

    align = pool[:min(n_align, len(pool))]
    M, mu_t, mu_s = ridge_map_parts(Zt[align], Zs[align], alpha=alpha)
    a_t, c_t = transfer_with_M(a_s, c_s, M, mu_t, mu_s)

    logit_tr = Zt_eval @ a_t + c_t
    p_tr = sigmoid(logit_tr)

    # Native target probe (calibrated by construction) on the full pool.
    native = LogisticRegression(max_iter=1000).fit(Zt[pool], yt[pool])
    p_nat = native.predict_proba(Zt_eval)[:, 1]
    logit_nat = native.decision_function(Zt_eval)

    orc_thr, orc_acc = oracle_threshold(logit_tr, yt_eval)

    res = {
        "auroc_transfer": float(roc_auc_score(yt_eval, logit_tr)),
        "auroc_native": float(roc_auc_score(yt_eval, p_nat)),
        "ece_transfer": ece(p_tr, yt_eval), "ece_native": ece(p_nat, yt_eval),
        "brier_transfer": brier(p_tr, yt_eval), "brier_native": brier(p_nat, yt_eval),
        "nll_transfer": nll(p_tr, yt_eval), "nll_native": nll(p_nat, yt_eval),
        "acc_thr0_transfer": acc_at_threshold(logit_tr, yt_eval, 0.0),
        "acc_oracle_transfer": orc_acc,
        "acc_thr0_native": acc_at_threshold(logit_nat, yt_eval, 0.0),
        # label-budget recalibration curve
        "recal": {},
    }

    # Platt recalibration with k labeled target examples drawn from the pool
    # (disjoint from eval). Averaged over a few draws per k to stabilize small k.
    rng = np.random.default_rng(10_000 + seed)
    for k in k_grid:
        eces, accs, aurocs = [], [], []
        n_draws = 5 if k <= 50 else 2
        for _ in range(n_draws):
            cal = rng.choice(pool, size=k, replace=False)
            lc = Zt[cal] @ a_t + c_t
            A, B = platt_fit(lc, yt[cal])
            p_re = sigmoid(A * logit_tr + B)
            eces.append(ece(p_re, yt_eval))
            # recalibrated threshold at p=0.5  <=>  A*logit+B=0
            accs.append(acc_at_threshold(logit_tr, yt_eval, -B / A if A != 0 else 0.0))
            aurocs.append(roc_auc_score(yt_eval, p_re))  # invariant, sanity
        res["recal"][str(k)] = {
            "ece": float(np.mean(eces)), "acc": float(np.mean(accs)),
            "auroc": float(np.mean(aurocs)),
        }
    return res


def aggregate(seed_results):
    keys_scalar = [k for k in seed_results[0] if k != "recal"]
    out = {}
    for k in keys_scalar:
        vals = np.array([r[k] for r in seed_results])
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    ks = list(seed_results[0]["recal"].keys())
    out["recal"] = {}
    for kk in ks:
        for metric in ["ece", "acc", "auroc"]:
            vals = np.array([r["recal"][kk][metric] for r in seed_results])
            out["recal"].setdefault(kk, {})[metric] = {
                "mean": float(vals.mean()), "std": float(vals.std())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--token", default="slt", choices=["slt", "tbg"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1e4)
    ap.add_argument("--n-align", type=int, default=1500)
    ap.add_argument("--k-grid", type=int, nargs="+",
                    default=[0, 10, 20, 50, 100, 200, 500])
    ap.add_argument("--out", required=True)
    args = apply_yaml_config(ap)
    assert len(args.sources) == len(args.targets) == len(args.names)

    all_res = {}
    for nm, sg, tg in zip(args.names, args.sources, args.targets):
        print(f"\n===== {nm} =====")
        # k=0 means "no recalibration"; keep it out of the Platt loop.
        k_grid = [k for k in args.k_grid if k > 0]
        seed_res = [run_seed(sg, tg, args.token, s, args.n_eval, args.alpha,
                             args.n_align, k_grid) for s in args.seeds]
        agg = aggregate(seed_res)
        all_res[nm] = agg
        print(f"  AUROC  transfer {agg['auroc_transfer']['mean']:.3f}"
              f"±{agg['auroc_transfer']['std']:.3f}  "
              f"native {agg['auroc_native']['mean']:.3f}")
        print(f"  ECE    transfer {agg['ece_transfer']['mean']:.3f}"
              f"±{agg['ece_transfer']['std']:.3f}  "
              f"native {agg['ece_native']['mean']:.3f}")
        print(f"  Brier  transfer {agg['brier_transfer']['mean']:.3f}  "
              f"native {agg['brier_native']['mean']:.3f}")
        print(f"  NLL    transfer {agg['nll_transfer']['mean']:.3f}  "
              f"native {agg['nll_native']['mean']:.3f}")
        print(f"  acc@thr0 transfer {agg['acc_thr0_transfer']['mean']:.3f}  "
              f"acc@oracle {agg['acc_oracle_transfer']['mean']:.3f}  "
              f"(threshold gap {agg['acc_oracle_transfer']['mean']-agg['acc_thr0_transfer']['mean']:.3f})")
        print("  recalibration (k labels -> ECE / acc):")
        for kk in k_grid:
            r = agg["recal"][str(kk)]
            print(f"    k={kk:4d}: ECE {r['ece']['mean']:.3f}  acc {r['acc']['mean']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
