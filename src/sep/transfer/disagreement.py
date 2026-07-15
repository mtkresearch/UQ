"""Cross-scale SE disagreement detector (idea #16 core; cache-only).

Claim (DESIGN_disagreement.md): single-model semantic entropy has a documented
blind spot -- self-consistent errors, where the model resamples the same wrong
answer, SE is low, the detector passes it (Tan et al. 2025). We test whether that
blind spot is partly *scale-specific*: an error the 1.7B is confidently wrong about
(low SE) is often one the 8B is unsure about (high SE). If so, cross-scale SE
disagreement recovers the low-SE-wrong tail neither model's own SE catches.

Target model = the model whose errors we detect. We run BOTH directions:
  * PRIMARY   target = 1.7B (cheaper/deployed). Interesting cell = 1.7B-low/8B-high.
  * SECONDARY target = 8B.  Interesting cell = 8B-low/1.7B-high.

Everything here runs on the SAME aligned examples for both models (ids asserted
equal). Sampled SE (`semantic_entropy`) and gold correctness (`validation_is_false`)
come from the sibling uncertainty_measures.pkl. Answer-disagreement (greedy answer
mismatch) and per-model logprob come from `most_likely_answer` in the generations.

Core deliverable (an afternoon):
  1. Marginals  -- AUROC of each model's SE vs its own correctness (sanity vs P1).
  2. Joint grid -- target error rate per SE-decile x SE-decile cell + extreme 2x2.
  3. Decomposition on the target-SE-low subset -- blindspot precision/recall/flag/FP.
  4. Detector suite -- AUROC/AURC of target-SE, other-SE, mean, max, signed
     (SE_other - SE_target), |delta|, and a CV'd logistic fusion, vs target-wrong.
  5. Decision bars:
       HEADLINE 1  signed (SE_other - SE_target) beats max(SE) AND other-SE-alone
                   (paired bootstrap CI on the AUROC gap excludes 0).
       HEADLINE 2  SE-disagreement beats greedy answer-mismatch.
       SEMANTIC GATE  within the target-SE-low subset, other-SE beats other-logprob
                   at predicting target-wrong (necessary: other-SE within-low > 0.5).
       CHEAP FIREWALL  incremental AUROC of SE terms over cheap features
                   (logprob_s, logprob_t, |len diff|, answer-mismatch).

Pooled-across-datasets paired bootstrap is the primary evidence for the headline
bars because the anti-correlated low-target/high-other cell is small per dataset
(SE_s and SE_t are positively correlated); the extreme 2x2 is illustrative only.
"""
import argparse
import json
import os
import pickle
import re
import string

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# --------------------------------------------------------------------------- #
# Loading (cache-only, aligned ids)
# --------------------------------------------------------------------------- #
def _sibling(gen_path, name):
    return os.path.join(os.path.dirname(gen_path), name)


def load_measures(gen_path):
    """SE (sampled) + gold correctness from the sibling uncertainty_measures.pkl.

    Returns (se, wrong) as float64 / int arrays. `wrong` = validation_is_false
    (1 = model was wrong), the gold label all detectors are scored against.
    """
    with open(_sibling(gen_path, "uncertainty_measures.pkl"), "rb") as f:
        m = pickle.load(f)
    se = np.asarray(m["uncertainty_measures"]["semantic_entropy"], dtype=np.float64)
    wrong = np.asarray(m["validation_is_false"], dtype=np.int64)
    return se, wrong


_PUNCT = str.maketrans("", "", string.punctuation)


def _norm_answer(s):
    """Normalize a short answer for greedy-mismatch comparison (SQuAD-style)."""
    s = s.lower().strip()
    s = s.translate(_PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_greedy(gen_path):
    """Per-example greedy answer + mean-token logprob from most_likely_answer.

    Returns (ids, answers[list[str]], mean_logprob[np.float64], length[int]).
    mean_logprob = mean of token_log_likelihoods (higher = more confident); it is
    the confidence confound the semantic claim must beat within the low-SE subset.
    """
    with open(gen_path, "rb") as f:
        g = pickle.load(f)
    ids = list(g.keys())
    ans, lp, ln = [], [], []
    for k in ids:
        mla = g[k]["most_likely_answer"]
        ans.append(_norm_answer(mla["response"]))
        tll = mla["token_log_likelihoods"]
        lp.append(float(np.mean(tll)) if len(tll) else 0.0)
        ln.append(len(tll))
    return ids, ans, np.asarray(lp, dtype=np.float64), np.asarray(ln, dtype=np.float64)


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def safe_auroc(y, score):
    """AUROC with y=target-wrong; higher score should mean more likely wrong."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def aurc(y, score):
    """Area under the risk-coverage curve (lower = better detector).

    Sort by ascending score (most confident first = lowest predicted risk);
    risk at coverage c = error rate among the c fraction we keep.
    """
    order = np.argsort(score, kind="mergesort")
    y_sorted = y[order]
    csum = np.cumsum(y_sorted)
    n = len(y)
    cov = np.arange(1, n + 1)
    risk = csum / cov
    return float(np.mean(risk))


def cv_fusion_scores(X, y, seed=0, n_splits=5):
    """Out-of-fold logistic-fusion scores (no in-sample overfit).

    Standardizes within each training fold; class-balanced. Returns per-example
    OOF P(wrong). If a fold is single-class, falls back to the base rate.
    """
    scores = np.zeros(len(y), dtype=np.float64)
    if len(np.unique(y)) < 2:
        return np.full(len(y), float(y.mean()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit((X[tr] - mu) / sd, y[tr])
        scores[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    return scores


def paired_bootstrap_auroc_gap(y, score_a, score_b, n_boot=2000, seed=0):
    """Bootstrap CI of AUROC(score_a) - AUROC(score_b) over shared resamples.

    Positive gap means score_a is the better target-wrong detector. CI excluding
    0 is the bar for the headline bars.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            gaps[b] = np.nan
            continue
        gaps[b] = (roc_auc_score(yb, score_a[idx])
                   - roc_auc_score(yb, score_b[idx]))
    gaps = gaps[~np.isnan(gaps)]
    return {
        "point": float(safe_auroc(y, score_a) - safe_auroc(y, score_b)),
        "ci_lo": float(np.percentile(gaps, 2.5)),
        "ci_hi": float(np.percentile(gaps, 97.5)),
        "p_gt_0": float(np.mean(gaps > 0)),
    }


# --------------------------------------------------------------------------- #
# Analysis for one direction (one target model)
# --------------------------------------------------------------------------- #
def joint_grid(se_target, se_other, wrong_target, n_bins=10):
    """Target error rate per (target-SE decile) x (other-SE decile) cell.

    Deciles are label-free quantile bins. Returns the error-rate grid and count
    grid; NaN where a cell is empty.
    """
    qt = np.quantile(se_target, np.linspace(0, 1, n_bins + 1))
    qo = np.quantile(se_other, np.linspace(0, 1, n_bins + 1))
    bt = np.clip(np.digitize(se_target, qt[1:-1]), 0, n_bins - 1)
    bo = np.clip(np.digitize(se_other, qo[1:-1]), 0, n_bins - 1)
    err = np.full((n_bins, n_bins), np.nan)
    cnt = np.zeros((n_bins, n_bins), dtype=int)
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (bt == i) & (bo == j)
            cnt[i, j] = int(mask.sum())
            if mask.any():
                err[i, j] = float(wrong_target[mask].mean())
    return err, cnt


def extreme_2x2(se_target, se_other, wrong_target, lo=0.25, hi=0.75):
    """Illustrative 2x2 on the extreme quantile corners (middle band dropped)."""
    t_lo = se_target <= np.quantile(se_target, lo)
    t_hi = se_target >= np.quantile(se_target, hi)
    o_lo = se_other <= np.quantile(se_other, lo)
    o_hi = se_other >= np.quantile(se_other, hi)
    out = {}
    for tn, tm in [("target_lo", t_lo), ("target_hi", t_hi)]:
        for on, om in [("other_lo", o_lo), ("other_hi", o_hi)]:
            mask = tm & om
            out[f"{tn}__{on}"] = {
                "n": int(mask.sum()),
                "target_err": float(wrong_target[mask].mean()) if mask.any() else float("nan"),
            }
    return out


def blindspot_decomposition(se_target, se_other, wrong_target, lo=0.25, hi=0.75):
    """Metrics conditioned on the target-SE-low subset (DESIGN_disagreement.md §4.3).

    blindspot precision = P(target wrong | target-SE low, other-SE high)
    blindspot recall    = P(other-SE high | target-SE low, target wrong)
    flag rate           = P(target-SE low, other-SE high)
    FP cost             = P(target correct | target-SE low, other-SE high)
    """
    t_lo = se_target <= np.quantile(se_target, lo)
    o_hi = se_other >= np.quantile(se_other, hi)
    n = len(se_target)
    low = t_lo
    low_and_flag = t_lo & o_hi
    low_and_wrong = t_lo & (wrong_target == 1)
    n_flag = int(low_and_flag.sum())
    prec = float(wrong_target[low_and_flag].mean()) if n_flag else float("nan")
    recall = (float((low_and_flag & (wrong_target == 1)).sum() / low_and_wrong.sum())
              if low_and_wrong.sum() else float("nan"))
    return {
        "n_target_low": int(low.sum()),
        "n_flagged": n_flag,
        "flag_rate": n_flag / n,
        "blindspot_precision": prec,
        "blindspot_recall": recall,
        "fp_cost": (1.0 - prec) if n_flag else float("nan"),
        "target_err_in_low": (float(wrong_target[low].mean()) if low.any() else float("nan")),
        "target_err_overall": float(wrong_target.mean()),
    }


def within_low_semantic_gate(se_target, se_other, other_logprob, wrong_target, lo=0.25):
    """Semantic-claim gate: within target-SE-low, does other-SE beat other-logprob?

    other-SE within-low AUROC > 0.5 is necessary; beating other-logprob (a
    difficulty proxy) is what earns the *semantic* claim. Higher logprob = more
    confident = less likely wrong, so we score with -logprob.
    """
    t_lo = se_target <= np.quantile(se_target, lo)
    y = wrong_target[t_lo]
    res = {"n": int(t_lo.sum()), "target_err": float(y.mean()) if len(y) else float("nan")}
    if len(np.unique(y)) < 2:
        res.update(other_se_auroc=float("nan"), other_logprob_auroc=float("nan"),
                   se_beats_logprob=None)
        return res
    au_se = safe_auroc(y, se_other[t_lo])
    au_lp = safe_auroc(y, -other_logprob[t_lo])
    res.update(other_se_auroc=au_se, other_logprob_auroc=au_lp,
               se_minus_logprob=au_se - au_lp,
               other_se_gt_half=bool(au_se > 0.5),
               se_beats_logprob=bool(au_se > au_lp))
    return res


def cheap_firewall(y, cheap, full, seed=0):
    """Incremental OOF AUROC of SE terms over cheap features.

    cheap / full are (n, d) feature blocks; full = cheap + SE terms. Returns OOF
    AUROC for each and the increment.
    """
    au_cheap = safe_auroc(y, cv_fusion_scores(cheap, y, seed=seed))
    au_full = safe_auroc(y, cv_fusion_scores(full, y, seed=seed))
    return {"auroc_cheap": au_cheap, "auroc_full": au_full,
            "increment": au_full - au_cheap}


def run_direction(se_target, se_other, wrong_target, lp_target, lp_other,
                  len_target, len_other, answer_mismatch, seed=0, n_boot=2000):
    """All core analyses for one target model. `signed` = SE_other - SE_target."""
    zt, zo = zscore(se_target), zscore(se_other)
    signed = zo - zt                       # other more uncertain than target
    absd = np.abs(zo - zt)
    mean_se = (zt + zo) / 2.0
    max_se = np.maximum(zt, zo)
    fusion = cv_fusion_scores(np.column_stack([zt, zo]), wrong_target, seed=seed)

    detectors = {
        "target_se": zt, "other_se": zo, "mean_se": mean_se, "max_se": max_se,
        "signed_other_minus_target": signed, "abs_delta": absd,
        "fusion_logistic": fusion,
        "answer_mismatch": answer_mismatch.astype(np.float64),
    }
    auroc_suite = {k: safe_auroc(wrong_target, v) for k, v in detectors.items()}
    aurc_suite = {k: aurc(wrong_target, v) for k, v in detectors.items()}

    # HEADLINE 1: signed disagreement must beat max(SE) AND other-SE-alone.
    h1 = {
        "vs_max_se": paired_bootstrap_auroc_gap(
            wrong_target, signed, max_se, n_boot=n_boot, seed=seed),
        "vs_other_se": paired_bootstrap_auroc_gap(
            wrong_target, signed, zo, n_boot=n_boot, seed=seed),
    }
    h1["passes"] = bool(h1["vs_max_se"]["ci_lo"] > 0 and h1["vs_other_se"]["ci_lo"] > 0)

    # HEADLINE 2: SE-disagreement must beat greedy answer-mismatch.
    h2 = paired_bootstrap_auroc_gap(
        wrong_target, signed, answer_mismatch.astype(np.float64),
        n_boot=n_boot, seed=seed)
    h2["passes"] = bool(h2["ci_lo"] > 0)

    # SEMANTIC GATE within the target-SE-low subset.
    gate = within_low_semantic_gate(se_target, se_other, lp_other, wrong_target)

    # CHEAP FIREWALL.
    cheap = np.column_stack([lp_target, lp_other,
                             np.abs(len_target - len_other),
                             answer_mismatch.astype(np.float64)])
    full = np.column_stack([cheap, zt, zo, signed])
    firewall = cheap_firewall(wrong_target, cheap, full, seed=seed)

    err_grid, cnt_grid = joint_grid(se_target, se_other, wrong_target)
    return {
        "n": int(len(wrong_target)),
        "target_err_rate": float(wrong_target.mean()),
        "se_corr_spearman": float(_spearman(se_target, se_other)),
        "auroc": auroc_suite,
        "aurc": aurc_suite,
        "headline1_signed_vs_better_model": h1,
        "headline2_se_vs_answer_disagreement": h2,
        "semantic_gate_within_low": gate,
        "cheap_firewall": firewall,
        "blindspot": blindspot_decomposition(se_target, se_other, wrong_target),
        "extreme_2x2": extreme_2x2(se_target, se_other, wrong_target),
        "_grids": (err_grid, cnt_grid),
    }


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float(np.mean(ra * rb))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_dataset(gen_1p7b, gen_8b):
    """Aligned per-example arrays for one dataset. Asserts id alignment."""
    ids_1, ans_1, lp_1, len_1 = load_greedy(gen_1p7b)
    ids_8, ans_8, lp_8, len_8 = load_greedy(gen_8b)
    assert ids_1 == ids_8, "example ids not aligned across models"
    se_1, wrong_1 = load_measures(gen_1p7b)
    se_8, wrong_8 = load_measures(gen_8b)
    answer_mismatch = np.array([a != b for a, b in zip(ans_1, ans_8)], dtype=np.int64)
    return dict(se_1p7b=se_1, se_8b=se_8, wrong_1p7b=wrong_1, wrong_8b=wrong_8,
                lp_1p7b=lp_1, lp_8b=lp_8, len_1p7b=len_1, len_8b=len_8,
                answer_mismatch=answer_mismatch, n=len(ids_1),
                answer_mismatch_rate=float(answer_mismatch.mean()))


def run_dataset(d, seed=0, n_boot=2000):
    """Both directions for one dataset (primary=1.7B target, secondary=8B)."""
    primary = run_direction(
        se_target=d["se_1p7b"], se_other=d["se_8b"], wrong_target=d["wrong_1p7b"],
        lp_target=d["lp_1p7b"], lp_other=d["lp_8b"],
        len_target=d["len_1p7b"], len_other=d["len_8b"],
        answer_mismatch=d["answer_mismatch"], seed=seed, n_boot=n_boot)
    secondary = run_direction(
        se_target=d["se_8b"], se_other=d["se_1p7b"], wrong_target=d["wrong_8b"],
        lp_target=d["lp_8b"], lp_other=d["lp_1p7b"],
        len_target=d["len_8b"], len_other=d["len_1p7b"],
        answer_mismatch=d["answer_mismatch"], seed=seed, n_boot=n_boot)
    # Marginals (sanity vs P1): SE vs own correctness.
    marginals = {
        "auroc_1p7b_se_vs_1p7b_wrong": safe_auroc(d["wrong_1p7b"], d["se_1p7b"]),
        "auroc_8b_se_vs_8b_wrong": safe_auroc(d["wrong_8b"], d["se_8b"]),
        "answer_mismatch_rate": d["answer_mismatch_rate"],
    }
    return {"marginals": marginals, "primary_target_1p7b": primary,
            "secondary_target_8b": secondary}


def strip_grids(res):
    """JSON-serializable copy: pull the numpy grids out of each direction."""
    grids = {}
    clean = json.loads(json.dumps({
        k: {kk: (vv if kk != "_grids" else None) for kk, vv in v.items()}
           if isinstance(v, dict) and "_grids" in v else v
        for k, v in res.items()
    }, default=lambda o: None))
    for key in ("primary_target_1p7b", "secondary_target_8b"):
        if key in res and "_grids" in res[key]:
            err, cnt = res[key]["_grids"]
            grids[key] = {"err": err.tolist(), "cnt": cnt.tolist()}
    return clean, grids


def pooled_bootstrap(datasets, direction_key, seed=0, n_boot=2000):
    """Pooled-across-datasets paired bootstrap for the headline bars (§6 primary).

    Concatenate all datasets' per-example target-wrong labels and detector scores,
    then paired-bootstrap the AUROC gaps. This is the primary power source because
    the anti-correlated cell is small per dataset.
    """
    ys, signeds, maxes, others, mism = [], [], [], [], []
    for d in datasets:
        if direction_key == "primary_target_1p7b":
            se_t, se_o, y = d["se_1p7b"], d["se_8b"], d["wrong_1p7b"]
        else:
            se_t, se_o, y = d["se_8b"], d["se_1p7b"], d["wrong_8b"]
        zt, zo = zscore(se_t), zscore(se_o)
        ys.append(y); signeds.append(zo - zt)
        maxes.append(np.maximum(zt, zo)); others.append(zo)
        mism.append(d["answer_mismatch"].astype(np.float64))
    y = np.concatenate(ys)
    signed = np.concatenate(signeds); mx = np.concatenate(maxes)
    other = np.concatenate(others); am = np.concatenate(mism)
    return {
        "n_pooled": int(len(y)),
        "signed_vs_max": paired_bootstrap_auroc_gap(y, signed, mx, n_boot, seed),
        "signed_vs_other_se": paired_bootstrap_auroc_gap(y, signed, other, n_boot, seed),
        "signed_vs_answer_mismatch": paired_bootstrap_auroc_gap(y, signed, am, n_boot, seed),
    }


def _plot_grids(all_grids, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(all_grids.keys())
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8), squeeze=False)
    for c, nm in enumerate(names):
        for r, dirk in enumerate(("primary_target_1p7b", "secondary_target_8b")):
            ax = axes[r][c]
            err = np.array(all_grids[nm][dirk]["err"], dtype=float)
            im = ax.imshow(err, origin="lower", aspect="auto", cmap="magma",
                           vmin=0, vmax=1)
            tgt = "1.7B" if r == 0 else "8B"
            oth = "8B" if r == 0 else "1.7B"
            ax.set_title(f"{nm}: {tgt} error rate")
            ax.set_xlabel(f"{oth} SE decile")
            ax.set_ylabel(f"{tgt} SE decile")
            fig.colorbar(im, ax=ax, fraction=0.046, label="P(target wrong)")
    plt.tight_layout()
    p = os.path.join(out_dir, "disagreement_grids.pdf")
    plt.savefig(p, format="pdf", dpi=200)
    plt.savefig(p.replace(".pdf", ".png"), format="png", dpi=150)
    plt.close()
    print(f"saved joint grids -> {p}")


def main():
    ap = argparse.ArgumentParser(description="Cross-scale SE disagreement (cache-only).")
    ap.add_argument("--gen-1p7b", nargs="+", required=True,
                    help="1.7B validation_generations.pkl, one per dataset")
    ap.add_argument("--gen-8b", nargs="+", required=True,
                    help="8B validation_generations.pkl, aligned order")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    assert len(args.gen_1p7b) == len(args.gen_8b) == len(args.names)

    datasets, per_ds, all_grids = [], {}, {}
    for nm, g1, g8 in zip(args.names, args.gen_1p7b, args.gen_8b):
        print(f"\n===== {nm} =====")
        d = load_dataset(g1, g8)
        datasets.append(d)
        res = run_dataset(d, seed=args.seed, n_boot=args.n_boot)
        clean, grids = strip_grids(res)
        per_ds[nm] = clean
        all_grids[nm] = grids

        m = res["marginals"]
        p, s = res["primary_target_1p7b"], res["secondary_target_8b"]
        print(f"  n={d['n']}  SE-corr(spearman)={p['se_corr_spearman']:.3f}  "
              f"answer-mismatch={m['answer_mismatch_rate']:.3f}")
        print(f"  marginals  1.7B SE->wrong {m['auroc_1p7b_se_vs_1p7b_wrong']:.3f}  "
              f"8B SE->wrong {m['auroc_8b_se_vs_8b_wrong']:.3f}")
        for tag, r in (("PRIMARY(1.7B)", p), ("SECONDARY(8B)", s)):
            au = r["auroc"]
            print(f"  {tag}  AUROC target-SE={au['target_se']:.3f} "
                  f"signed={au['signed_other_minus_target']:.3f} "
                  f"max={au['max_se']:.3f} fusion={au['fusion_logistic']:.3f} "
                  f"answer-mism={au['answer_mismatch']:.3f}")
            h1 = r["headline1_signed_vs_better_model"]
            print(f"      H1 signed>max ci=[{h1['vs_max_se']['ci_lo']:.3f},"
                  f"{h1['vs_max_se']['ci_hi']:.3f}]  signed>other-SE ci="
                  f"[{h1['vs_other_se']['ci_lo']:.3f},{h1['vs_other_se']['ci_hi']:.3f}]"
                  f"  pass={h1['passes']}")
            h2 = r["headline2_se_vs_answer_disagreement"]
            print(f"      H2 signed>answer-mism ci=[{h2['ci_lo']:.3f},"
                  f"{h2['ci_hi']:.3f}] pass={h2['passes']}")
            g = r["semantic_gate_within_low"]
            print(f"      GATE within-low(n={g['n']}) other-SE={g.get('other_se_auroc')}"
                  f" vs logprob={g.get('other_logprob_auroc')} "
                  f"se>0.5={g.get('other_se_gt_half')} se>lp={g.get('se_beats_logprob')}")
            fw = r["cheap_firewall"]
            print(f"      FIREWALL cheap={fw['auroc_cheap']:.3f} full={fw['auroc_full']:.3f}"
                  f" +{fw['increment']:.3f}")
            bs = r["blindspot"]
            print(f"      BLINDSPOT flag={bs['flag_rate']:.3f} prec={bs['blindspot_precision']}"
                  f" recall={bs['blindspot_recall']} err_in_low={bs['target_err_in_low']:.3f}"
                  f" (overall {bs['target_err_overall']:.3f})")

    print("\n===== POOLED (primary evidence) =====")
    pooled = {
        "primary_target_1p7b": pooled_bootstrap(datasets, "primary_target_1p7b",
                                                args.seed, args.n_boot),
        "secondary_target_8b": pooled_bootstrap(datasets, "secondary_target_8b",
                                                args.seed, args.n_boot),
    }
    for k, pv in pooled.items():
        print(f"  {k} (n={pv['n_pooled']}):")
        for bar in ("signed_vs_max", "signed_vs_other_se", "signed_vs_answer_mismatch"):
            b = pv[bar]
            print(f"    {bar}: gap={b['point']:.4f} "
                  f"ci=[{b['ci_lo']:.4f},{b['ci_hi']:.4f}] P(>0)={b['p_gt_0']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_dataset": per_ds, "pooled": pooled,
                   "config": {"seed": args.seed, "n_boot": args.n_boot,
                              "names": args.names}}, f, indent=2)
    with open(args.out.replace(".json", "_grids.json"), "w") as f:
        json.dump(all_grids, f, indent=2)
    _plot_grids(all_grids, os.path.dirname(args.out))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
