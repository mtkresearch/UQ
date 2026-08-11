"""Cross-alignment transfer: map learned on SQuAD, everything else on TriviaQA.

Identical to transfer.py in every way EXCEPT the inter-model alignment map is
fit on hidden states from a *different* dataset (SQuAD) rather than TriviaQA.

  TriviaQA  ->  source SE probe, native target probe (Curve A), all evaluation
  SQuAD     ->  inter-model map only (Curve B, unlabeled pairs)

Usage mirrors transfer.py; two extra required args supply the SQuAD gen paths:
  --source-gen-align  .../squad/.../llama-2-7b/.../validation_generations.pkl
  --target-gen-align  .../squad/.../mistral-7b/.../validation_generations.pkl

Normalisation
-------------
Source: z-scored with TriviaQA source-pool stats for both datasets (probe and
  map output live in that space).
Target: z-scored with SQuAD target stats for ALL map-related work — fitting the
  map and scoring TriviaQA target examples at eval time.  This keeps the map's
  input distribution identical at fit and apply time, so any AUROC drop vs. the
  same-dataset baseline is attributable to content distribution shift alone.
Curve A (native target probe) uses TriviaQA target stats independently.

Output files are named transfer_cross_align_<token>... to avoid clobbering
the original transfer_<token>... artefacts.
"""
import argparse
from sep.uncertainty.utils.config import apply_yaml_config
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import kendalltau, spearmanr

from sep.transfer.cka import load_hidden
from sep.transfer.transfer import (
    align_ids,
    best_se_layer,
    best_split,
    binarize,
    fit_e2_map,
    fit_e2_weights,
    fit_probe_aligned_map,
    fit_procrustes_map,
    fit_ridge_map,
    load_entropy,
    probe_scores,
    transfer_probe,
)


ALL_CURVES  = {"target_probe", "source_probe", "ridge", "procrustes",
               "probe_aligned", "e2_minimised", "e2_map"}
ALL_METRICS = {"auroc", "spearman", "kendall"}


def run(source_gen, target_gen,                  # TriviaQA gen paths
        source_gen_align, target_gen_align,      # SQuAD gen paths (map only)
        token, out_dir, n_eval, n_grid, seed, alpha=1e3,
        lam_e2_map=None, curves=None, metrics=None, out_suffix=""):
    curves  = set(curves)  if curves  is not None else set(ALL_CURVES)
    metrics = set(metrics) if metrics is not None else set(ALL_METRICS)
    unknown = (curves - ALL_CURVES) | (metrics - ALL_METRICS)
    if unknown:
        raise ValueError(f"Unknown curves/metrics: {unknown}. "
                         f"Valid curves: {ALL_CURVES}, metrics: {ALL_METRICS}")
    _probe_budget_curves = {"source_probe", "ridge", "procrustes",
                            "probe_aligned", "e2_minimised", "e2_map"}
    _run_probe_budget = bool(curves & _probe_budget_curves)

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1. TriviaQA hidden states
    # ------------------------------------------------------------------ #
    Hs, ids_s = load_hidden(source_gen, token)   # (Ls, N, ds)
    Ht, ids_t = load_hidden(target_gen, token)   # (Lt, N, dt)
    order = align_ids(ids_s, ids_t)
    Ht, ids_t = Ht[:, order], [ids_t[i] for i in order]
    N = Hs.shape[1]

    ent_s = load_entropy(source_gen)
    ent_t = load_entropy(target_gen)[order]
    ys = binarize(ent_s, best_split(ent_s))
    yt = binarize(ent_t, best_split(ent_t))
    print(f"[TriviaQA] N={N}  src pos-rate={ys.mean():.3f}  tgt pos-rate={yt.mean():.3f}")

    # ------------------------------------------------------------------ #
    # 2. SQuAD hidden states (alignment only — no labels used)
    # ------------------------------------------------------------------ #
    Hs_al, ids_s_al = load_hidden(source_gen_align, token)
    Ht_al, ids_t_al = load_hidden(target_gen_align, token)
    order_al = align_ids(ids_s_al, ids_t_al)
    Ht_al = Ht_al[:, order_al]
    ids_t_al = [ids_t_al[i] for i in order_al]
    N_al = Hs_al.shape[1]
    print(f"[SQuAD]    N={N_al}  (alignment pairs, no labels used)")

    # ------------------------------------------------------------------ #
    # 3. Train/eval split on TriviaQA
    # ------------------------------------------------------------------ #
    perm = rng.permutation(N)
    eval_idx, pool = perm[:n_eval], perm[n_eval:]
    print(f"eval={len(eval_idx)}  pool={len(pool)}")

    # ------------------------------------------------------------------ #
    # 4. Best SE layer (chosen on TriviaQA pool only)
    # ------------------------------------------------------------------ #
    Ls, aucs = best_se_layer(Hs[:, pool], ys[pool], seed=seed)
    Lt, auct = best_se_layer(Ht[:, pool], yt[pool], seed=seed)
    print(f"best src layer {Ls} (SE AUROC {aucs:.3f});  "
          f"best tgt layer {Lt} (SE AUROC {auct:.3f})")

    # ------------------------------------------------------------------ #
    # 5. Extract layer features
    # ------------------------------------------------------------------ #
    Xs    = Hs[Ls].astype(np.float64)      # TriviaQA source  (N, ds)
    Xt    = Ht[Lt].astype(np.float64)      # TriviaQA target  (N, dt)
    Xs_al = Hs_al[Ls].astype(np.float64)   # SQuAD source     (N_al, ds)
    Xt_al = Ht_al[Lt].astype(np.float64)   # SQuAD target     (N_al, dt)

    # ------------------------------------------------------------------ #
    # 6. Normalisation
    #
    #  Source stats  : TriviaQA pool  (probe lives here; SQuAD source is
    #                  normalised the same way so map output is in probe space)
    #  Target stats  : SQuAD full set (map fit in this space; TriviaQA target
    #                  normalised with SQuAD stats at eval so map input
    #                  distribution matches between fit and apply time)
    #  Target native : TriviaQA pool  (Curve A only — fully independent)
    # ------------------------------------------------------------------ #
    mu_s,    sd_s    = Xs[pool].mean(0), Xs[pool].std(0) + 1e-6
    mu_t_al, sd_t_al = Xt_al.mean(0),   Xt_al.std(0)    + 1e-6   # SQuAD target
    mu_t_tr, sd_t_tr = Xt[pool].mean(0), Xt[pool].std(0) + 1e-6  # TriviaQA target

    Zs       = (Xs    - mu_s)    / sd_s        # TriviaQA source  (probe space)
    Zs_al    = (Xs_al - mu_s)    / sd_s        # SQuAD source     (same space)
    Zt_al    = (Xt_al - mu_t_al) / sd_t_al    # SQuAD target     (map fitting)
    Zt       = (Xt    - mu_t_al) / sd_t_al    # TriviaQA target  (map eval, SQuAD stats)
    Zt_native = (Xt   - mu_t_tr) / sd_t_tr    # TriviaQA target  (Curve A native probe)

    # ------------------------------------------------------------------ #
    # 7. Source probe — trained on full TriviaQA pool (fixed)
    # ------------------------------------------------------------------ #
    src_probe = LogisticRegression(max_iter=1000).fit(Zs[pool], ys[pool])
    src_pool_auc = roc_auc_score(ys[pool],
                                 src_probe.predict_proba(Zs[pool])[:, 1])
    print(f"source probe fit on {len(pool)} TriviaQA examples "
          f"(train AUROC {src_pool_auc:.3f})")

    w_s = src_probe.coef_.ravel()
    c_s = float(src_probe.intercept_[0])
    _lam_e2 = lam_e2_map if lam_e2_map is not None else alpha

    # Eval slices
    Zt_eval        = Zt[eval_idx]           # for map-transferred probe (SQuAD stats)
    Zt_native_eval = Zt_native[eval_idx]    # for native target probe   (TriviaQA stats)
    yt_eval        = yt[eval_idx]
    Zs_eval        = Zs[eval_idx]
    ys_eval        = ys[eval_idx]
    ent_t_eval     = ent_t[eval_idx]
    ent_s_eval     = ent_s[eval_idx]

    # n_grid is capped by the SQuAD alignment set for Curve B;
    # Curve A can use up to len(pool) TriviaQA examples
    grid = [n for n in n_grid if n <= N_al]
    if len(grid) < len(n_grid):
        dropped = [n for n in n_grid if n > N_al]
        print(f"  note: n_grid values {dropped} exceed SQuAD N={N_al}, skipped")

    # SQuAD alignment permutation (for n_grid subsets)
    perm_al = rng.permutation(N_al)

    # ------------------------------------------------------------------ #
    # 8. Pre-fit full-SQuAD maps for probe-budget variants
    # ------------------------------------------------------------------ #
    if _run_probe_budget and "ridge" in curves:
        M_full, b_full = fit_ridge_map(Zt_al, Zs_al, alpha=alpha)
    if _run_probe_budget and "procrustes" in curves:
        Mp_full, bp_full = fit_procrustes_map(Zt_al, Zs_al)
    if _run_probe_budget and "probe_aligned" in curves:
        Mpa_full, bpa_full = fit_probe_aligned_map(Zt_al, Zs_al, w_s, alpha=alpha)
    if _run_probe_budget and "e2_minimised" in curves:
        a_e2_full, c_e2_full = fit_e2_weights(Zt_al, Zs_al, w_s, c_s,
                                               init_alpha=alpha)
    if _run_probe_budget and "e2_map" in curves:
        Me2m_full, be2m_full = fit_e2_map(Zt_al, Zs_al, w_s, c_s, lam=_lam_e2)

    # ------------------------------------------------------------------ #
    # 9. Result containers  (same key schema as transfer.py)
    # ------------------------------------------------------------------ #
    _curve_keys = {
        "target_probe":  ("curveA_native",         None),
        "source_probe":  ("curveC_source_native",   None),
        "ridge":         ("curveB_ridge",            "curveB_ridge_src_probe_budget"),
        "procrustes":    ("curveB_procrustes",       "curveB_procrustes_src_probe_budget"),
        "probe_aligned": ("curveB_probe_aligned",    "curveB_probe_aligned_src_probe_budget"),
        "e2_minimised":  ("curveB_e2_minimised",     "curveB_e2_src_probe_budget"),
        "e2_map":        ("curveB_e2_map",           "curveB_e2_map_src_probe_budget"),
    }

    results = {"src_layer": int(Ls), "tgt_layer": int(Lt),
               "src_layer_auc": float(aucs), "tgt_layer_auc": float(auct),
               "alpha": float(alpha), "lam_e2_map": float(_lam_e2),
               "curves": sorted(curves), "metrics": sorted(metrics),
               "n_grid": grid}
    for c in curves:
        mb_key, pb_key = _curve_keys[c]
        results[mb_key] = []
        if "spearman" in metrics: results[mb_key + "_spearman"] = []
        if "kendall"  in metrics: results[mb_key + "_kendall"]  = []
        if pb_key and _run_probe_budget:
            results[pb_key] = []
            if "spearman" in metrics: results[pb_key + "_spearman"] = []
            if "kendall"  in metrics: results[pb_key + "_kendall"]  = []

    def _append(key, val):
        if key in results:
            results[key].append(val)

    def _rank(true_ent, scores):
        rho, _ = spearmanr(true_ent, scores)
        tau, _ = kendalltau(true_ent, scores)
        return float(rho), float(tau)

    # ------------------------------------------------------------------ #
    # 10. Main grid loop
    #     Curve A  x=n : n labeled TriviaQA target examples (capped at pool)
    #     Curve B  x=n : n unlabeled SQuAD pairs            (capped at N_al)
    # ------------------------------------------------------------------ #
    for n in grid:
        sub_tr = pool[:min(n, len(pool))]   # TriviaQA subset for Curve A
        sub_al = perm_al[:n]                # SQuAD subset for Curve B map
        print_parts = [f"n={n:4d}"]

        # Curve A: native target probe on n labeled TriviaQA target examples
        if "target_probe" in curves:
            if len(np.unique(yt[sub_tr])) < 2:
                _append("curveA_native", None)
                _append("curveA_native_spearman", None)
                _append("curveA_native_kendall", None)
            else:
                native = LogisticRegression(max_iter=1000).fit(
                    Zt_native[sub_tr], yt[sub_tr])
                proba_a = native.predict_proba(Zt_native_eval)[:, 1]
                au_a = roc_auc_score(yt_eval, proba_a)
                _append("curveA_native", float(au_a))
                rho_a, tau_a = _rank(ent_t_eval, proba_a)
                _append("curveA_native_spearman", rho_a)
                _append("curveA_native_kendall", tau_a)
                print_parts.append(f"A(native)={au_a:.3f}")

        # Curve B: map fit on n unlabeled SQuAD pairs, fixed source probe transferred
        if "ridge" in curves:
            M, b = fit_ridge_map(Zt_al[sub_al], Zs_al[sub_al], alpha=alpha)
            a_t, c_t = transfer_probe(src_probe, M, b)
            sc_r = probe_scores(Zt_eval, a_t, c_t)
            au_r = roc_auc_score(yt_eval, sc_r)
            _append("curveB_ridge", float(au_r))
            rho_r, tau_r = _rank(ent_t_eval, sc_r)
            _append("curveB_ridge_spearman", rho_r)
            _append("curveB_ridge_kendall", tau_r)
            print_parts.append(f"B(ridge)={au_r:.3f}")

        if "procrustes" in curves:
            Mp, bp = fit_procrustes_map(Zt_al[sub_al], Zs_al[sub_al])
            a_tp, c_tp = transfer_probe(src_probe, Mp, bp)
            sc_p = probe_scores(Zt_eval, a_tp, c_tp)
            au_p = roc_auc_score(yt_eval, sc_p)
            _append("curveB_procrustes", float(au_p))
            rho_p, tau_p = _rank(ent_t_eval, sc_p)
            _append("curveB_procrustes_spearman", rho_p)
            _append("curveB_procrustes_kendall", tau_p)
            print_parts.append(f"B(procrustes)={au_p:.3f}")

        if "probe_aligned" in curves:
            Mpa, bpa = fit_probe_aligned_map(
                Zt_al[sub_al], Zs_al[sub_al], w_s, alpha=alpha)
            a_tpa, c_tpa = transfer_probe(src_probe, Mpa, bpa)
            sc_pa = probe_scores(Zt_eval, a_tpa, c_tpa)
            au_pa = roc_auc_score(yt_eval, sc_pa)
            _append("curveB_probe_aligned", float(au_pa))
            rho_pa, tau_pa = _rank(ent_t_eval, sc_pa)
            _append("curveB_probe_aligned_spearman", rho_pa)
            _append("curveB_probe_aligned_kendall", tau_pa)
            print_parts.append(f"B(probe-aligned)={au_pa:.3f}")

        if "e2_minimised" in curves:
            a_te2, c_te2 = fit_e2_weights(
                Zt_al[sub_al], Zs_al[sub_al], w_s, c_s, init_alpha=alpha)
            sc_e2 = probe_scores(Zt_eval, a_te2, c_te2)
            au_e2 = roc_auc_score(yt_eval, sc_e2)
            _append("curveB_e2_minimised", float(au_e2))
            rho_e2, tau_e2 = _rank(ent_t_eval, sc_e2)
            _append("curveB_e2_minimised_spearman", rho_e2)
            _append("curveB_e2_minimised_kendall", tau_e2)
            print_parts.append(f"B(e2-min)={au_e2:.3f}")

        if "e2_map" in curves:
            Me2m, be2m = fit_e2_map(
                Zt_al[sub_al], Zs_al[sub_al], w_s, c_s, lam=_lam_e2)
            a_te2m, c_te2m = transfer_probe(src_probe, Me2m, be2m)
            sc_e2m = probe_scores(Zt_eval, a_te2m, c_te2m)
            au_e2m = roc_auc_score(yt_eval, sc_e2m)
            _append("curveB_e2_map", float(au_e2m))
            rho_e2m, tau_e2m = _rank(ent_t_eval, sc_e2m)
            _append("curveB_e2_map_spearman", rho_e2m)
            _append("curveB_e2_map_kendall", tau_e2m)
            print_parts.append(f"B(e2-map)={au_e2m:.3f}")

        # Probe-budget variants: full-SQuAD map fixed,
        # probe trained on n labeled TriviaQA source examples.
        if _run_probe_budget:
            if len(np.unique(ys[sub_tr])) < 2:
                for k in list(results):
                    if k.startswith("curveC") or k.endswith("_src_probe_budget") \
                            or "_src_probe_budget_" in k:
                        results[k].append(None)
            else:
                src_native = LogisticRegression(max_iter=1000).fit(
                    Zs[sub_tr], ys[sub_tr])
                w_n = src_native.coef_.ravel()
                c_n = float(src_native.intercept_[0])

                if "source_probe" in curves:
                    sc_c = src_native.predict_proba(Zs_eval)[:, 1]
                    au_c = roc_auc_score(ys_eval, sc_c)
                    _append("curveC_source_native", float(au_c))
                    rho_c, tau_c = _rank(ent_s_eval, sc_c)
                    _append("curveC_source_native_spearman", rho_c)
                    _append("curveC_source_native_kendall", tau_c)
                    print_parts.append(f"C(src-native)={au_c:.3f}")

                if "ridge" in curves:
                    a_t_n, c_t_n = transfer_probe(src_native, M_full, b_full)
                    sc_rn = probe_scores(Zt_eval, a_t_n, c_t_n)
                    _append("curveB_ridge_src_probe_budget",
                            float(roc_auc_score(yt_eval, sc_rn)))
                    rho_rn, tau_rn = _rank(ent_t_eval, sc_rn)
                    _append("curveB_ridge_src_probe_budget_spearman", rho_rn)
                    _append("curveB_ridge_src_probe_budget_kendall", tau_rn)

                if "procrustes" in curves:
                    a_tp_n, c_tp_n = transfer_probe(src_native, Mp_full, bp_full)
                    sc_pn = probe_scores(Zt_eval, a_tp_n, c_tp_n)
                    _append("curveB_procrustes_src_probe_budget",
                            float(roc_auc_score(yt_eval, sc_pn)))
                    rho_pn, tau_pn = _rank(ent_t_eval, sc_pn)
                    _append("curveB_procrustes_src_probe_budget_spearman", rho_pn)
                    _append("curveB_procrustes_src_probe_budget_kendall", tau_pn)

                if "probe_aligned" in curves:
                    Mpa_n, bpa_n = fit_probe_aligned_map(
                        Zt_al, Zs_al, w_n, alpha=alpha)
                    a_tpa_n, c_tpa_n = transfer_probe(src_native, Mpa_n, bpa_n)
                    sc_pan = probe_scores(Zt_eval, a_tpa_n, c_tpa_n)
                    _append("curveB_probe_aligned_src_probe_budget",
                            float(roc_auc_score(yt_eval, sc_pan)))
                    rho_pan, tau_pan = _rank(ent_t_eval, sc_pan)
                    _append("curveB_probe_aligned_src_probe_budget_spearman", rho_pan)
                    _append("curveB_probe_aligned_src_probe_budget_kendall", tau_pan)

                if "e2_minimised" in curves:
                    a_te2_n, c_te2_n = fit_e2_weights(
                        Zt_al, Zs_al, w_n, c_n, init_alpha=alpha)
                    sc_e2n = probe_scores(Zt_eval, a_te2_n, c_te2_n)
                    _append("curveB_e2_src_probe_budget",
                            float(roc_auc_score(yt_eval, sc_e2n)))
                    rho_e2n, tau_e2n = _rank(ent_t_eval, sc_e2n)
                    _append("curveB_e2_src_probe_budget_spearman", rho_e2n)
                    _append("curveB_e2_src_probe_budget_kendall", tau_e2n)

                if "e2_map" in curves:
                    Me2m_n, be2m_n = fit_e2_map(
                        Zt_al, Zs_al, w_n, c_n, lam=_lam_e2)
                    a_te2m_n, c_te2m_n = transfer_probe(src_native, Me2m_n, be2m_n)
                    sc_e2mn = probe_scores(Zt_eval, a_te2m_n, c_te2m_n)
                    _append("curveB_e2_map_src_probe_budget",
                            float(roc_auc_score(yt_eval, sc_e2mn)))
                    rho_e2mn, tau_e2mn = _rank(ent_t_eval, sc_e2mn)
                    _append("curveB_e2_map_src_probe_budget_spearman", rho_e2mn)
                    _append("curveB_e2_map_src_probe_budget_kendall", tau_e2mn)

        print("  " + "  ".join(print_parts))

    # ------------------------------------------------------------------ #
    # 11. Save + plot
    # ------------------------------------------------------------------ #
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if alpha == 1e3 else f"_a{alpha:.0e}"
    if seed != 0:
        suffix += f"_s{seed}"
    suffix += out_suffix
    out_path = os.path.join(out_dir, f"transfer_cross_align_{token}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if "auroc" in metrics:
        _plot(results, token, out_dir, suffix=suffix)
        _plot_probe_budget(results, token, out_dir, suffix=suffix)
    for metric in ("spearman", "kendall"):
        if metric in metrics:
            _plot_ranking(results, token, out_dir, metric=metric, suffix=suffix)
            _plot_probe_budget_ranking(results, token, out_dir, metric=metric, suffix=suffix)
    print(f"saved cross-align transfer results -> {out_dir}")
    return results


# --------------------------------------------------------------------------- #
# Plots  (same style as transfer.py; labels note the dataset split)
# --------------------------------------------------------------------------- #
def _plot(res, token, out_dir, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["n_grid"]
    fig, ax = plt.subplots(figsize=(7, 5))
    if "curveA_native" in res:
        a = [v if v is not None else np.nan for v in res["curveA_native"]]
        ax.plot(g, a, "o-", label="Curve A: native target probe (labeled TriviaQA)")
    if "curveB_ridge" in res:
        ax.plot(g, res["curveB_ridge"], "s--",
                label="Curve B: transfer (ridge, unlabeled SQuAD)")
    if "curveB_procrustes" in res:
        ax.plot(g, res["curveB_procrustes"], "^--",
                label="Curve B: transfer (Procrustes, unlabeled SQuAD)")
    if "curveB_probe_aligned" in res:
        ax.plot(g, res["curveB_probe_aligned"], "D--",
                label="Curve B: transfer (probe-aligned, unlabeled SQuAD)")
    if "curveB_e2_minimised" in res:
        ax.plot(g, res["curveB_e2_minimised"], "P--",
                label="Curve B: transfer (E2-minimised, unlabeled SQuAD)")
    if "curveB_e2_map" in res:
        ax.plot(g, res["curveB_e2_map"], "X--",
                label="Curve B: transfer (E2-map, unlabeled SQuAD)")
    if res.get("curveC_source_native"):
        c = [v if v is not None else np.nan for v in res["curveC_source_native"]]
        ax.plot(g, c, "v:", color="green",
                label="Curve C: native source probe (labeled TriviaQA)")
    ax.set_xscale("log")
    ax.set_xlabel("N  [Curve A: labeled TriviaQA target  |  Curve B: unlabeled SQuAD pairs]")
    ax.set_ylabel("eval AUROC (TriviaQA target SE)")
    alpha_txt = f", α={res.get('alpha', 1e3):.0e}" if res.get("alpha", 1e3) != 1e3 else ""
    ax.set_title(f"SE probe transfer — map on SQuAD, eval on TriviaQA "
                 f"({token.upper()}{alpha_txt})")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"transfer_cross_align_{token}{suffix}.pdf")
    plt.savefig(path, format="pdf", dpi=200)
    plt.close()
    print(f"  saved curve -> {path}")


def _plot_probe_budget(res, token, out_dir, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["n_grid"]
    fig, ax = plt.subplots(figsize=(7, 5))
    if "curveA_native" in res:
        a = [v if v is not None else np.nan for v in res["curveA_native"]]
        ax.plot(g, a, "o-", color="#333333",
                label="Curve A: native target probe (n labeled TriviaQA target)")
    if "curveB_ridge_src_probe_budget" in res:
        b = [v if v is not None else np.nan
             for v in res["curveB_ridge_src_probe_budget"]]
        ax.plot(g, b, "s--", color="#4C72B0",
                label="Curve B: src probe + SQuAD ridge map -> target")
    if "curveB_procrustes_src_probe_budget" in res:
        b = [v if v is not None else np.nan
             for v in res["curveB_procrustes_src_probe_budget"]]
        ax.plot(g, b, "^--", color="#DD8452",
                label="Curve B: src probe + SQuAD Procrustes map -> target")
    if "curveB_probe_aligned_src_probe_budget" in res:
        b = [v if v is not None else np.nan
             for v in res["curveB_probe_aligned_src_probe_budget"]]
        ax.plot(g, b, "D--", color="#8172B2",
                label="Curve B: src probe + SQuAD probe-aligned map -> target")
    if "curveB_e2_src_probe_budget" in res:
        b = [v if v is not None else np.nan
             for v in res["curveB_e2_src_probe_budget"]]
        ax.plot(g, b, "P--", color="#C44E52",
                label="Curve B: src probe + SQuAD E2-minimised map -> target")
    if "curveB_e2_map_src_probe_budget" in res:
        b = [v if v is not None else np.nan
             for v in res["curveB_e2_map_src_probe_budget"]]
        ax.plot(g, b, "X--", color="#2CA02C",
                label="Curve B: src probe + SQuAD E2-map -> target")
    if res.get("curveC_source_native"):
        c = [v if v is not None else np.nan for v in res["curveC_source_native"]]
        ax.plot(g, c, "v:", color="green",
                label="Curve C: native source probe (n labeled TriviaQA source)")
    ax.set_xscale("log")
    ax.set_xlabel("labeled TriviaQA examples used to train the probe")
    ax.set_ylabel("eval AUROC (TriviaQA target SE)")
    alpha_txt = f", α={res.get('alpha', 1e3):.0e}" if res.get("alpha", 1e3) != 1e3 else ""
    ax.set_title(f"SE probe transfer — probe budget, map on SQuAD "
                 f"({token.upper()}{alpha_txt})")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"transfer_cross_align_{token}{suffix}_probe_budget.pdf")
    plt.savefig(path, format="pdf", dpi=200)
    plt.close()
    print(f"  saved probe-budget curve -> {path}")


def _plot_ranking(res, token, out_dir, metric="spearman", suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["n_grid"]
    label = "Spearman ρ" if metric == "spearman" else "Kendall τ"
    fig, ax = plt.subplots(figsize=(7, 5))
    key_a = f"curveA_native_{metric}"
    if key_a in res:
        a = [v if v is not None else np.nan for v in res[key_a]]
        ax.plot(g, a, "o-", label="Curve A: native target probe (labeled TriviaQA)")
    for curve, marker, lbl in [
        (f"curveB_ridge_{metric}",         "s--",
         "Curve B: transfer (ridge, unlabeled SQuAD)"),
        (f"curveB_procrustes_{metric}",    "^--",
         "Curve B: transfer (Procrustes, unlabeled SQuAD)"),
        (f"curveB_probe_aligned_{metric}", "D--",
         "Curve B: transfer (probe-aligned, unlabeled SQuAD)"),
        (f"curveB_e2_minimised_{metric}",  "P--",
         "Curve B: transfer (E2-minimised, unlabeled SQuAD)"),
        (f"curveB_e2_map_{metric}",        "X--",
         "Curve B: transfer (E2-map, unlabeled SQuAD)"),
    ]:
        if curve in res:
            ax.plot(g, res[curve], marker, label=lbl)
    key_c = f"curveC_source_native_{metric}"
    if key_c in res:
        c = [v if v is not None else np.nan for v in res[key_c]]
        ax.plot(g, c, "v:", color="green",
                label="Curve C: native source probe (labeled TriviaQA)")
    ax.set_xscale("log")
    ax.set_xlabel("N  [Curve A: labeled TriviaQA target  |  Curve B: unlabeled SQuAD pairs]")
    ax.set_ylabel(f"eval {label} (vs true entropy)")
    alpha_txt = f", α={res.get('alpha', 1e3):.0e}" if res.get("alpha", 1e3) != 1e3 else ""
    ax.set_title(f"SE probe transfer — map on SQuAD — {label} ({token.upper()}{alpha_txt})")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"transfer_cross_align_{token}{suffix}_{metric}.pdf")
    plt.savefig(path, format="pdf", dpi=200)
    plt.close()
    print(f"  saved {label} curve -> {path}")


def _plot_probe_budget_ranking(res, token, out_dir, metric="spearman", suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["n_grid"]
    label = "Spearman ρ" if metric == "spearman" else "Kendall τ"
    fig, ax = plt.subplots(figsize=(7, 5))
    key_a = f"curveA_native_{metric}"
    if key_a in res:
        a = [v if v is not None else np.nan for v in res[key_a]]
        ax.plot(g, a, "o-", color="#333333",
                label="Curve A: native target probe (n labeled TriviaQA target)")
    for curve, marker, color, lbl in [
        (f"curveB_ridge_src_probe_budget_{metric}",         "s--", "#4C72B0",
         "Curve B: src probe + SQuAD ridge map -> target"),
        (f"curveB_procrustes_src_probe_budget_{metric}",    "^--", "#DD8452",
         "Curve B: src probe + SQuAD Procrustes map -> target"),
        (f"curveB_probe_aligned_src_probe_budget_{metric}", "D--", "#8172B2",
         "Curve B: src probe + SQuAD probe-aligned map -> target"),
        (f"curveB_e2_src_probe_budget_{metric}",            "P--", "#C44E52",
         "Curve B: src probe + SQuAD E2-minimised map -> target"),
        (f"curveB_e2_map_src_probe_budget_{metric}",        "X--", "#2CA02C",
         "Curve B: src probe + SQuAD E2-map -> target"),
    ]:
        if curve in res:
            b = [v if v is not None else np.nan for v in res[curve]]
            ax.plot(g, b, marker, color=color, label=lbl)
    key_c = f"curveC_source_native_{metric}"
    if key_c in res:
        c = [v if v is not None else np.nan for v in res[key_c]]
        ax.plot(g, c, "v:", color="green",
                label="Curve C: native source probe (n labeled TriviaQA source)")
    ax.set_xscale("log")
    ax.set_xlabel("labeled TriviaQA examples used to train the probe")
    ax.set_ylabel(f"eval {label} (vs true entropy)")
    alpha_txt = f", α={res.get('alpha', 1e3):.0e}" if res.get("alpha", 1e3) != 1e3 else ""
    ax.set_title(f"SE probe transfer — probe budget, map on SQuAD — "
                 f"{label} ({token.upper()}{alpha_txt})")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(
        out_dir, f"transfer_cross_align_{token}{suffix}_probe_budget_{metric}.pdf")
    plt.savefig(path, format="pdf", dpi=200)
    plt.close()
    print(f"  saved {label} probe-budget curve -> {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="SE probe transfer: map on SQuAD, probe+eval on TriviaQA "
                    "(Llama-2-7B source / Mistral-7B target).")
    # TriviaQA (train / eval)
    p.add_argument("--source-gen", required=True,
                   help="TriviaQA validation_generations.pkl for source model (Llama-2-7B)")
    p.add_argument("--target-gen", required=True,
                   help="TriviaQA validation_generations.pkl for target model (Mistral-7B)")
    # SQuAD (alignment map only)
    p.add_argument("--source-gen-align", required=True,
                   help="SQuAD validation_generations.pkl for source model (Llama-2-7B)")
    p.add_argument("--target-gen-align", required=True,
                   help="SQuAD validation_generations.pkl for target model (Mistral-7B)")

    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--n-grid", type=int, nargs="+",
                   default=[50, 100, 200, 400, 800, 1500])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1e3,
                   help="ridge-map regularization")
    p.add_argument("--lam-e2-map", type=float, default=None)
    p.add_argument("--curves", nargs="+", default=None, metavar="CURVE",
                   help=f"curves to compute (default: all). choices: {sorted(ALL_CURVES)}")
    p.add_argument("--metrics", nargs="+", default=None, metavar="METRIC",
                   help=f"metrics to plot (default: all). choices: {sorted(ALL_METRICS)}")
    p.add_argument("--out-suffix", default="",
                   help="extra string appended to output filenames")
    args = apply_yaml_config(p)
    run(args.source_gen, args.target_gen,
        args.source_gen_align, args.target_gen_align,
        args.token, args.out_dir,
        args.n_eval, args.n_grid, args.seed,
        alpha=args.alpha, lam_e2_map=args.lam_e2_map,
        curves=args.curves, metrics=args.metrics,
        out_suffix=args.out_suffix)


if __name__ == "__main__":
    main()
