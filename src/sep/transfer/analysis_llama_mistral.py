"""Two boss-requested diagnostics for the Llama-2-7b -> Mistral-7B transfer.

Both run cache-only on aligned paired runs (same seed/samples, ids asserted equal).

TASK 1 -- Model disagreement, split by certain/uncertain.
  Each model's cluster_assignment_entropy is binarized (best_split, as in
  transfer.py) into certain(0)/uncertain(1). We cross-tabulate the two label
  vectors into a 2x2 contingency table and report the disagreement rate
  P[y_s != y_t]. This is NOT the positive-rate mismatch |mean(y_s)-mean(y_t)|:
  the mismatch lets the two off-diagonal cells cancel; the disagreement rate
  (mean|y_s-y_t|) sums them. We report both so the distinction is explicit.

TASK 2 -- w_s^T (R h_t - h_s), the alignment-induced prediction error in logit
  space (term 2 of the decomposition, written out for the linear source probe).
  * w_s : source (Llama) probe weights, trained on standardized Llama features
          + Llama SE labels (exactly transfer.py's source probe).
  * R   : target->source ridge map (Mistral -> Llama), fit on standardized
          paired features (fit_ridge_map). "R h_t" here = Zt @ M + b.
  * h_s : standardized Llama features.
  Per example this is a scalar = how much the map's reconstruction error, seen
  through the probe's weight direction, perturbs the probe logit. Reported as
  mean (signed bias), RMS and mean-abs (typical magnitude), std, and percentiles.

  Sign convention follows the decomposition f_s(R h_t) - f_s(h_s), i.e.
  (R h_t - h_s); this equals term 2 exactly. Everything is computed on the
  z-scored spaces the probe and map are actually fit in, so the logits are the
  probe's real decision scores.
"""
import argparse
import json
import os

import numpy as np

from sep.transfer.cka import load_hidden
from sep.transfer.transfer import (
    load_entropy, best_split, binarize, best_se_layer,
    fit_ridge_map, zscore,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def disagreement_table(ys, yt):
    """2x2 contingency of binarized SE labels + disagreement vs pos-rate mismatch."""
    n = len(ys)
    # cells: [source_label][target_label]
    both_certain = int(np.sum((ys == 0) & (yt == 0)))
    src_cert_tgt_unc = int(np.sum((ys == 0) & (yt == 1)))
    src_unc_tgt_cert = int(np.sum((ys == 1) & (yt == 0)))
    both_uncertain = int(np.sum((ys == 1) & (yt == 1)))
    disagree = src_cert_tgt_unc + src_unc_tgt_cert
    return {
        "n": n,
        "table": {
            "source_certain__target_certain": both_certain,
            "source_certain__target_uncertain": src_cert_tgt_unc,
            "source_uncertain__target_certain": src_unc_tgt_cert,
            "source_uncertain__target_uncertain": both_uncertain,
        },
        "source_pos_rate": float(ys.mean()),
        "target_pos_rate": float(yt.mean()),
        "positive_rate_mismatch": float(abs(ys.mean() - yt.mean())),
        "disagreement_rate": disagree / n,
        "disagreement_count": disagree,
        "agreement_rate": (both_certain + both_uncertain) / n,
        # split of the disagreement into the two directions
        "src_certain_tgt_uncertain_rate": src_cert_tgt_unc / n,
        "src_uncertain_tgt_certain_rate": src_unc_tgt_cert / n,
    }


def alignment_logit_error(Zs, Zt, ys, pool, w_s, c_s, alpha):
    """w_s^T (R h_t - h_s) per example, R = ridge map target->source on `pool`.

    Zs, Zt already z-scored. Map fit on the unlabeled pool pairs; the per-example
    logit error is evaluated on ALL examples. Returns per-example array + stats.
    """
    M, b = fit_ridge_map(Zt[pool], Zs[pool], alpha=alpha)
    RHt = Zt @ M + b                       # mapped target -> source space
    resid = RHt - Zs                       # (n, d_s) reconstruction residual
    err = resid @ w_s                      # (n,) logit-space alignment error
    a = np.abs(err)
    return err, {
        "alpha": float(alpha),
        "mean": float(err.mean()),
        "rms": float(np.sqrt((err ** 2).mean())),
        "mean_abs": float(a.mean()),
        "std": float(err.std()),
        "p05": float(np.percentile(err, 5)),
        "p50": float(np.percentile(err, 50)),
        "p95": float(np.percentile(err, 95)),
        "max_abs": float(a.max()),
    }


def run(source_gen, target_gen, token, out_dir, n_eval, seed, alpha):
    rng = np.random.default_rng(seed)
    Hs, ids_s = load_hidden(source_gen, token)
    Ht, ids_t = load_hidden(target_gen, token)
    assert ids_s == ids_t, "example ids not aligned across models"
    N = Hs.shape[1]

    ent_s, ent_t = load_entropy(source_gen), load_entropy(target_gen)
    ys = binarize(ent_s, best_split(ent_s))
    yt = binarize(ent_t, best_split(ent_t))

    # ---- Task 1: disagreement (uses ALL N examples) ----
    task1 = disagreement_table(ys, yt)

    # ---- Task 2 setup: mirror transfer.py's splits/standardization ----
    perm = rng.permutation(N)
    eval_idx, pool = perm[:n_eval], perm[n_eval:]
    Ls, aucs = best_se_layer(Hs[:, pool], ys[pool], seed=seed)
    Lt, auct = best_se_layer(Ht[:, pool], yt[pool], seed=seed)

    Xs = Hs[Ls].astype(np.float64)
    Xt = Ht[Lt].astype(np.float64)
    mu_s, sd_s = Xs[pool].mean(0), Xs[pool].std(0) + 1e-6
    mu_t, sd_t = Xt[pool].mean(0), Xt[pool].std(0) + 1e-6
    Zs = (Xs - mu_s) / sd_s
    Zt = (Xt - mu_t) / sd_t

    src_probe = LogisticRegression(max_iter=1000).fit(Zs[pool], ys[pool])
    w_s = src_probe.coef_.ravel()
    c_s = float(src_probe.intercept_[0])
    src_pool_auc = roc_auc_score(ys[pool], src_probe.predict_proba(Zs[pool])[:, 1])

    err, task2 = alignment_logit_error(Zs, Zt, ys, pool, w_s, c_s, alpha)

    results = {
        "dataset": os.path.basename(os.path.dirname(os.path.dirname(source_gen))),
        "source_model": "Llama-2-7b", "target_model": "Mistral-7B-v0.1",
        "token": token, "n": int(N),
        "src_layer": int(Ls), "tgt_layer": int(Lt),
        "src_layer_se_auroc": float(aucs), "tgt_layer_se_auroc": float(auct),
        "source_probe_pool_auroc": float(src_pool_auc),
        "task1_disagreement": task1,
        "task2_alignment_logit_error": task2,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"analysis_llama_mistral_{token}.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.save(os.path.join(out_dir, f"analysis_llama_mistral_{token}_logit_error.npy"), err)
    _plot_hist(err, task2, token, out_dir)

    _print(results)
    print(f"\nsaved -> {out_dir}/analysis_llama_mistral_{token}.json")
    return results


def _plot_hist(err, t2, token, out_dir):
    """Histogram of the per-query alignment-induced logit error w_s^T(R h_t - h_s)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(err, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", lw=0.8, ls="-")
    ax.axvline(t2["mean"], color="#C44E52", lw=1.5, ls="--",
               label=f"mean={t2['mean']:+.2f}")
    for q, name in ((t2["p05"], "p05"), (t2["p95"], "p95")):
        ax.axvline(q, color="#55A868", lw=1.0, ls=":")
    ax.axvspan(t2["p05"], t2["p95"], color="#55A868", alpha=0.08,
               label="p05–p95 (90% of queries)")
    ax.set_xlabel(r"per-query logit error  $w_s^\top(R\,h_t - h_s)$")
    ax.set_ylabel("number of queries")
    ax.set_title(f"Alignment-induced logit error  (Llama→Mistral, {token.upper()})\n"
                 f"RMS={t2['rms']:.2f}, mean|·|={t2['mean_abs']:.2f}, "
                 f"max|·|={t2['max_abs']:.1f}")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6)
    plt.tight_layout()
    p = os.path.join(out_dir, f"analysis_llama_mistral_{token}_logit_error_hist.png")
    plt.savefig(p, dpi=150)
    plt.savefig(p.replace(".png", ".pdf"))
    plt.close()
    print(f"saved histogram -> {p}")


def _print(r):
    t1 = r["task1_disagreement"]; tbl = t1["table"]
    print("=" * 66)
    print(f"{r['source_model']} (source) -> {r['target_model']} (target)  "
          f"| {r['dataset']} | token={r['token']} | N={r['n']}")
    print(f"src SE layer {r['src_layer']} (AUROC {r['src_layer_se_auroc']:.3f}), "
          f"tgt SE layer {r['tgt_layer']} (AUROC {r['tgt_layer_se_auroc']:.3f}), "
          f"source probe pool AUROC {r['source_probe_pool_auroc']:.3f}")
    print("\n--- TASK 1: model disagreement (binarized SE) ---")
    print(f"                       | Mistral certain | Mistral uncertain")
    print(f"  Llama certain        | {tbl['source_certain__target_certain']:>15d} "
          f"| {tbl['source_certain__target_uncertain']:>17d}")
    print(f"  Llama uncertain      | {tbl['source_uncertain__target_certain']:>15d} "
          f"| {tbl['source_uncertain__target_uncertain']:>17d}")
    print(f"  source pos-rate (Llama uncertain)   = {t1['source_pos_rate']:.3f}")
    print(f"  target pos-rate (Mistral uncertain) = {t1['target_pos_rate']:.3f}")
    print(f"  positive-rate mismatch (they cancel) = {t1['positive_rate_mismatch']:.3f}")
    print(f"  DISAGREEMENT RATE (no cancel)        = {t1['disagreement_rate']:.3f} "
          f"({t1['disagreement_count']}/{t1['n']})")
    print(f"    Llama-certain / Mistral-uncertain  = {t1['src_certain_tgt_uncertain_rate']:.3f}")
    print(f"    Llama-uncertain / Mistral-certain  = {t1['src_uncertain_tgt_certain_rate']:.3f}")
    t2 = r["task2_alignment_logit_error"]
    print("\n--- TASK 2: w_s^T (R h_t - h_s)  alignment-induced logit error ---")
    print(f"  (α={t2['alpha']:.0e})  mean={t2['mean']:+.4f}  rms={t2['rms']:.4f}  "
          f"mean|.|={t2['mean_abs']:.4f}  std={t2['std']:.4f}")
    print(f"  pctiles: p05={t2['p05']:+.3f}  p50={t2['p50']:+.3f}  p95={t2['p95']:+.3f}  "
          f"max|.|={t2['max_abs']:.3f}")


def main():
    p = argparse.ArgumentParser(description="Llama->Mistral tasks 1&2 (cache-only).")
    p.add_argument("--source-gen", required=True, help="Llama validation_generations.pkl")
    p.add_argument("--target-gen", required=True, help="Mistral validation_generations.pkl")
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1e3)
    a = p.parse_args()
    run(a.source_gen, a.target_gen, a.token, a.out_dir, a.n_eval, a.seed, a.alpha)


if __name__ == "__main__":
    main()
