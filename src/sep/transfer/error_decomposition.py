"""Variance/covariance report for the transfer-error decomposition (any pair).

The transferred-probe error splits, per example x, EXACTLY into three additive
terms in probability space (sigmoid outputs / {0,1} labels):

    p_{s->t}(x) - y_t(x)
      = [ p_s(h_s) - y_s ]              (E1) source probe error
      + [ f_s(R h_t) - f_s(h_s) ]       (E2) alignment-induced prediction error
      + [ y_s - y_t ]                   (E3) model disagreement

where p_s(h)=sigma(w_s^T h + b_s) is the source logistic probe, R is the fitted
target->source ridge map (R h_t := Zt @ M + b), and y_s,y_t are the binarized
semantic-uncertainty labels of the two models.

Each term is a per-sample vector; we report its histogram + mean/var/std. Because
the three terms sum to the total IDENTICALLY, the total error variance decomposes
exactly:

    Var(total) = Var(E1)+Var(E2)+Var(E3)
               + 2Cov(E1,E2)+2Cov(E1,E3)+2Cov(E2,E3).

Covariance is taken ACROSS samples but BETWEEN the three term-vectors: positive
Cov(Ei,Ej) => on the same samples both errors are large (they reinforce);
negative => one offsets the other (they cancel). We report the 3x3 covariance and
correlation matrices and the full variance budget (each own-variance and each
2*covariance as a share of Var(total)). Also carried: the Lipschitz logit gap
|w_s^T(R h_t - h_s)|/4, the guaranteed upper bound on |E2|.

The map R, the standardization, and the source probe are ALWAYS fit on the pool
(never the eval set). --eval-on chooses WHERE the three terms are evaluated:
  * eval : held-out eval set only (unbiased; E1/E2 not seen during fitting)
  * all  : every aligned example (matches the original histogram; E1/E2 biased low)
"""
import argparse
import json
import os

import numpy as np

from sep.transfer.cka import load_hidden
from sep.transfer.transfer import (
    load_entropy, best_split, binarize, best_se_layer, fit_ridge_map,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def decompose(Zs, Zt, ys, yt, pool, idx, w_s, c_s, alpha):
    """Return the three error terms (on `idx`) + the total, all in prob space.

    Map/standardization/probe are fit on `pool`; terms evaluated on `idx`.
    """
    M, b = fit_ridge_map(Zt[pool], Zs[pool], alpha=alpha)   # target -> source
    RHt = Zt @ M + b                                        # mapped target feats

    g_hs = Zs @ w_s + c_s          # source-probe logit on native source feats
    g_RHt = RHt @ w_s + c_s        # source-probe logit on mapped target feats
    p_s = _sigmoid(g_hs)           # p_s(h_s)
    p_st = _sigmoid(g_RHt)         # p_{s->t}

    E1 = p_s[idx] - ys[idx]                    # source probe error
    E2 = p_st[idx] - p_s[idx]                  # alignment-induced prediction error
    E3 = (ys[idx] - yt[idx]).astype(np.float64)  # model disagreement
    total = p_st[idx] - yt[idx]                # = E1+E2+E3 (identically)
    logit_gap = (RHt - Zs)[idx] @ w_s          # w_s^T(R h_t - h_s); |E2| <= |gap|/4
    return E1, E2, E3, total, logit_gap


def _term_stats(x):
    return {"mean": float(x.mean()), "var": float(x.var(ddof=1)),
            "std": float(x.std(ddof=1)),
            "mean_abs": float(np.abs(x).mean()),
            "rms": float(np.sqrt((x ** 2).mean())),
            "min": float(x.min()), "max": float(x.max())}


def variance_report(E1, E2, E3, total):
    """3x3 cov/corr matrices + the exact variance budget of the total error."""
    names = ["E1_source_probe", "E2_alignment", "E3_disagreement"]
    stack = np.vstack([E1, E2, E3])            # (3, n)
    C = np.cov(stack, ddof=1)                  # 3x3 covariance
    corr = np.corrcoef(stack)                  # 3x3 correlation
    var_total = float(total.var(ddof=1))       # == C.sum() up to fp

    def frac(v):
        return float(v / var_total) if var_total else float("nan")

    budget = {
        "var_E1": float(C[0, 0]), "var_E2": float(C[1, 1]), "var_E3": float(C[2, 2]),
        "2cov_E1_E2": float(2 * C[0, 1]),
        "2cov_E1_E3": float(2 * C[0, 2]),
        "2cov_E2_E3": float(2 * C[1, 2]),
        "sum_check": float(C.sum()), "var_total": var_total,
    }
    budget_pct = {k: frac(v) for k, v in budget.items()
                  if k not in ("sum_check", "var_total")}
    return {
        "term_names": names,
        "covariance_matrix": C.tolist(),
        "correlation_matrix": corr.tolist(),
        "variance_budget": budget,
        "variance_budget_pct_of_total": budget_pct,
    }


def run(source_gen, target_gen, token, out_dir, n_eval, seed, alpha,
        source_name, target_name, eval_on):
    rng = np.random.default_rng(seed)
    Hs, ids_s = load_hidden(source_gen, token)
    Ht, ids_t = load_hidden(target_gen, token)
    assert ids_s == ids_t, "example ids not aligned across models"
    N = Hs.shape[1]

    ent_s, ent_t = load_entropy(source_gen), load_entropy(target_gen)
    ys = binarize(ent_s, best_split(ent_s))
    yt = binarize(ent_t, best_split(ent_t))

    perm = rng.permutation(N)
    eval_idx, pool = perm[:n_eval], perm[n_eval:]
    # Where the three terms are evaluated. Fitting is always on the pool.
    idx = eval_idx if eval_on == "eval" else np.arange(N)

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

    E1, E2, E3, total, gap = decompose(
        Zs, Zt, ys, yt, pool, idx, w_s, c_s, alpha)

    # correctness: the three terms must sum to the total identically.
    identity_resid = float(np.max(np.abs(total - (E1 + E2 + E3))))

    # disagreement rate = P[y_s != y_t] = mean(|E3|), split by direction.
    disagree = {
        "disagreement_rate": float(np.mean(np.abs(E3))),
        "disagreement_count": int(np.sum(E3 != 0)),
        "src_certain_tgt_uncertain_rate": float(np.mean(E3 == -1)),  # y_s=0,y_t=1
        "src_uncertain_tgt_certain_rate": float(np.mean(E3 == 1)),   # y_s=1,y_t=0
        "agreement_rate": float(np.mean(E3 == 0)),
    }

    vr = variance_report(E1, E2, E3, total)
    results = {
        "source_model": source_name, "target_model": target_name,
        "dataset": os.path.basename(os.path.dirname(os.path.dirname(source_gen))),
        "token": token, "alpha": float(alpha), "n_total": int(N),
        "eval_on": eval_on, "n_terms_evaluated": int(len(idx)),
        "n_eval": int(len(eval_idx)), "n_pool": int(len(pool)),
        "src_layer": int(Ls), "tgt_layer": int(Lt),
        "src_layer_se_auroc": float(aucs), "tgt_layer_se_auroc": float(auct),
        "source_probe_pool_auroc": float(src_pool_auc),
        "identity_max_abs_residual": identity_resid,
        "disagreement": disagree,
        "terms": {
            "E1_source_probe": _term_stats(E1),
            "E2_alignment": _term_stats(E2),
            "E3_disagreement": _term_stats(E3),
            "total_transfer_error": _term_stats(total),
            "E2_logit_gap_bound": _term_stats(np.abs(gap) / 4.0),
        },
        "variance_covariance": vr,
    }

    os.makedirs(out_dir, exist_ok=True)
    asuf = "" if alpha == 1e3 else f"_a{alpha:.0e}"
    base = f"error_decomposition_{token}_{eval_on}{asuf}"
    with open(os.path.join(out_dir, base + ".json"), "w") as f:
        json.dump(results, f, indent=2)
    np.save(os.path.join(out_dir, base + "_terms.npy"),
            np.vstack([E1, E2, E3, total]))
    _plot_hists(E1, E2, E3, total, gap, results, token, out_dir, base)
    _plot_variance(results, token, out_dir, base)

    _print(results)
    print(f"\nsaved -> {out_dir}/{base}.json")
    return results


def _plot_hists(E1, E2, E3, total, gap, r, token, out_dir, base):
    """Histogram per term + identity verification (LHS vs E1+E2+E3) + logit gap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sum_terms = E1 + E2 + E3
    resid = total - sum_terms
    resid_max = float(np.max(np.abs(resid)))

    dis = r["disagreement"]
    e3_title = (f"E3: model disagreement  $y_s - y_t$\n"
                f"disagreement rate = {dis['disagreement_rate']:.1%} "
                f"({dis['disagreement_count']}/{r['n_terms_evaluated']})")

    fig, axes = plt.subplots(2, 4, figsize=(22, 8))
    ax_list = axes.ravel()

    # Row 1: three error terms + logit gap
    term_panels = [
        ("E1_source_probe",  E1,  "#4C72B0", r"$E_1$: source probe error  $\hat{p}_s - y_s$"),
        ("E2_alignment",     E2,  "#8172B3", r"$E_2$: alignment error  $f_s(Rh_t) - f_s(h_s)$"),
        ("E3_disagreement",  E3,  "#CCB974", e3_title),
    ]
    for ax, (key, x, color, title) in zip(ax_list[:3], term_panels):
        s = r["terms"][key]
        bins = np.array([-1.5, -0.5, 0.5, 1.5]) if key == "E3_disagreement" else 80
        ax.hist(x, bins=bins, color=color, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(s["mean"], color="#C44E52", lw=1.5, ls="--",
                   label=f"mean={s['mean']:+.3f}")
        ax.axvspan(s["mean"] - s["std"], s["mean"] + s["std"],
                   color="k", alpha=0.06, label=f"±std ({s['std']:.3f})")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("queries")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6)

    # Panel [0,3]: logit gap
    ax = ax_list[3]
    ax.hist(gap, bins=80, color="#55A868", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(float(gap.mean()), color="#C44E52", lw=1.5, ls="--",
               label=f"mean={gap.mean():+.2f}")
    ax.set_title(r"logit gap  $w_s^\top(R h_t - h_s)$" "\n"
                 f"RMS={np.sqrt((gap**2).mean()):.2f}, "
                 f"mean|·|={np.abs(gap).mean():.2f}, max|·|={np.abs(gap).max():.1f}",
                 fontsize=10)
    ax.set_ylabel("queries")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6)

    # Row 2: identity verification — LHS, E1+E2+E3, residual; spare cell off
    bins_id = np.linspace(-1.2, 1.2, 81)
    s_total = r["terms"]["total_transfer_error"]

    # Panel [1,0]: LHS computed directly
    ax = ax_list[4]
    ax.hist(total, bins=bins_id, color="#C44E52", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(s_total["mean"], color="k", lw=1.5, ls="--",
               label=f"mean={s_total['mean']:+.4f}")
    ax.set_title(r"LHS (direct):  $\hat{p}_{s\to t}(x) - y_t(x)$", fontsize=10)
    ax.set_ylabel("queries"); ax.set_xlabel("error value")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6)

    # Panel [1,1]: RHS = E1+E2+E3
    ax = ax_list[5]
    ax.hist(sum_terms, bins=bins_id, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(float(sum_terms.mean()), color="k", lw=1.5, ls="--",
               label=f"mean={sum_terms.mean():+.4f}")
    ax.set_title(r"RHS (sum):  $E_1 + E_2 + E_3$", fontsize=10)
    ax.set_ylabel("queries"); ax.set_xlabel("error value")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6)

    # Panel [1,2]: residual LHS - RHS
    ax = ax_list[6]
    ax.hist(resid, bins=80, color="#55A868", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"Residual: LHS $-$ RHS\nmax|·| = {resid_max:.2e}  (identity check)",
                 fontsize=10)
    ax.set_ylabel("queries"); ax.set_xlabel("residual value")
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6)

    ax_list[7].axis("off")

    fig.suptitle(f"{r['source_model']}→{r['target_model']} error terms "
                 f"({token.upper()}, α={r['alpha']:.0e}, eval_on={r['eval_on']}, "
                 f"n={r['n_terms_evaluated']})", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out_dir, base + "_hists.png")
    plt.savefig(p, dpi=150)
    plt.savefig(p.replace(".png", ".pdf"))
    plt.close()
    print(f"saved term histograms -> {p}")


def _plot_variance(r, token, out_dir, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = r["variance_covariance"]["variance_budget"]
    corr = np.array(r["variance_covariance"]["correlation_matrix"])
    cov = np.array(r["variance_covariance"]["covariance_matrix"])
    labels = ["Var(E1)\nprobe", "Var(E2)\nalign", "Var(E3)\ndisagree",
              "2Cov\n(E1,E2)", "2Cov\n(E1,E3)", "2Cov\n(E2,E3)"]
    vals = [b["var_E1"], b["var_E2"], b["var_E3"],
            b["2cov_E1_E2"], b["2cov_E1_E3"], b["2cov_E2_E3"]]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = ["#4C72B0", "#8172B3", "#CCB974", "#C44E52", "#C44E52", "#C44E52"]
    ax1.bar(range(len(vals)), vals, color=colors, edgecolor="white")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.axhline(b["var_total"], color="#55A868", lw=1.4, ls="--",
                label=f"Var(total)={b['var_total']:.4f}")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("contribution to Var(total transfer error)")
    ax1.set_title("Variance budget  (own var + 2·cov)")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6)

    tn = ["E1 probe", "E2 align", "E3 disagree"]
    # Covariance heatmap (diagonal = variances). Symmetric scale about 0.
    cmax = float(np.abs(cov).max()) or 1.0
    im2 = ax2.imshow(cov, vmin=-cmax, vmax=cmax, cmap="RdBu_r")
    ax2.set_xticks(range(3)); ax2.set_xticklabels(tn, fontsize=8, rotation=20)
    ax2.set_yticks(range(3)); ax2.set_yticklabels(tn, fontsize=8)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{cov[i, j]:+.4f}", ha="center", va="center",
                     fontsize=9, color="k")
    ax2.set_title("Covariance of error terms\n(diagonal = variances)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, label="cov")

    im = ax3.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax3.set_xticks(range(3)); ax3.set_xticklabels(tn, fontsize=8, rotation=20)
    ax3.set_yticks(range(3)); ax3.set_yticklabels(tn, fontsize=8)
    for i in range(3):
        for j in range(3):
            ax3.text(j, i, f"{corr[i, j]:+.2f}", ha="center", va="center",
                     fontsize=9, color="k")
    ax3.set_title("Correlation of error terms")
    fig.colorbar(im, ax=ax3, fraction=0.046, label="corr")
    fig.suptitle(f"{r['source_model']}→{r['target_model']} variance decomposition "
                 f"({token.upper()}, α={r['alpha']:.0e}, eval_on={r['eval_on']})",
                 fontsize=11)
    plt.tight_layout()
    p = os.path.join(out_dir, base + "_variance.png")
    plt.savefig(p, dpi=150)
    plt.savefig(p.replace(".png", ".pdf"))
    plt.close()
    print(f"saved variance plot -> {p}")


def _print(r):
    t = r["terms"]; vc = r["variance_covariance"]
    b = vc["variance_budget"]; pct = vc["variance_budget_pct_of_total"]
    C = np.array(vc["covariance_matrix"]); corr = np.array(vc["correlation_matrix"])
    print("=" * 72)
    print(f"{r['source_model']} (source) -> {r['target_model']} (target) | "
          f"{r['dataset']} | token={r['token']} | a={r['alpha']:.0e} | "
          f"eval_on={r['eval_on']} (n={r['n_terms_evaluated']})")
    print(f"src layer {r['src_layer']} (SE AUROC {r['src_layer_se_auroc']:.3f}), "
          f"tgt layer {r['tgt_layer']} (SE AUROC {r['tgt_layer_se_auroc']:.3f}), "
          f"probe pool AUROC {r['source_probe_pool_auroc']:.3f}")
    print(f"identity check  max|total-(E1+E2+E3)| = {r['identity_max_abs_residual']:.2e}")
    d = r["disagreement"]
    print(f"disagreement rate P[y_s!=y_t] = {d['disagreement_rate']:.4f} "
          f"({d['disagreement_rate']:.1%}, {d['disagreement_count']}/{r['n_terms_evaluated']})"
          f"  [src-cert/tgt-unc {d['src_certain_tgt_uncertain_rate']:.1%}, "
          f"src-unc/tgt-cert {d['src_uncertain_tgt_certain_rate']:.1%}]")
    print("\n--- per-term stats (probability space) ---")
    print(f"  {'term':<26}{'mean':>10}{'std':>10}{'var':>12}{'mean|.|':>10}")
    for k in ["E1_source_probe", "E2_alignment", "E3_disagreement",
              "total_transfer_error", "E2_logit_gap_bound"]:
        s = t[k]
        print(f"  {k:<26}{s['mean']:>+10.4f}{s['std']:>10.4f}"
              f"{s['var']:>12.5f}{s['mean_abs']:>10.4f}")
    print("\n--- 3x3 covariance matrix (E1,E2,E3) ---")
    for i, nm in enumerate(["E1", "E2", "E3"]):
        print(f"  {nm}: " + "  ".join(f"{C[i, j]:>+11.6f}" for j in range(3)))
    print("--- 3x3 correlation matrix ---")
    for i, nm in enumerate(["E1", "E2", "E3"]):
        print(f"  {nm}: " + "  ".join(f"{corr[i, j]:>+7.3f}" for j in range(3)))
    print("\n--- variance budget: Var(total) = sum of these (share of total) ---")
    print(f"  Var(E1)  source probe error   {b['var_E1']:>+11.6f}  ({pct['var_E1']:>+6.1%})")
    print(f"  Var(E2)  alignment error      {b['var_E2']:>+11.6f}  ({pct['var_E2']:>+6.1%})")
    print(f"  Var(E3)  model disagreement   {b['var_E3']:>+11.6f}  ({pct['var_E3']:>+6.1%})")
    print(f"  2Cov(E1,E2)                   {b['2cov_E1_E2']:>+11.6f}  ({pct['2cov_E1_E2']:>+6.1%})")
    print(f"  2Cov(E1,E3)                   {b['2cov_E1_E3']:>+11.6f}  ({pct['2cov_E1_E3']:>+6.1%})")
    print(f"  2Cov(E2,E3)                   {b['2cov_E2_E3']:>+11.6f}  ({pct['2cov_E2_E3']:>+6.1%})")
    print(f"  {'-' * 54}")
    print(f"  Var(total transfer error)     {b['var_total']:>+11.6f}   "
          f"(sum={b['sum_check']:>+11.6f})")


def main():
    p = argparse.ArgumentParser(
        description="Variance/covariance report for the transfer-error decomposition.")
    p.add_argument("--source-gen", required=True,
                   help="source (probe-trained) validation_generations.pkl")
    p.add_argument("--target-gen", required=True,
                   help="target (probe-transferred) validation_generations.pkl")
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--source-name", default="source")
    p.add_argument("--target-name", default="target")
    p.add_argument("--eval-on", default="eval", choices=["eval", "all"],
                   help="evaluate terms on held-out eval set or all examples "
                        "(fitting is always on the pool)")
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1e3)
    a = p.parse_args()
    run(a.source_gen, a.target_gen, a.token, a.out_dir, a.n_eval, a.seed,
        a.alpha, a.source_name, a.target_name, a.eval_on)


if __name__ == "__main__":
    main()
