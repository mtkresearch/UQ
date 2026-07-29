"""Aggregate comparison plots for all pairs, two x-axis variants.

Produces two multi-panel PNG figures (one panel per pair):

  1. ALL_pairs_mapbudget_<token>_a<alpha>.png
     x = number of unlabeled alignment pairs used to fit the map
     Curves: A (native target), B ridge, B Procrustes, B probe-aligned,
             B E2-minimised, C (native source)

  2. ALL_pairs_probebudget_<token>_a<alpha>.png
     x = number of labeled source examples used to train the source probe
     Curves: A (native target), B ridge, B Procrustes, B probe-aligned (refit),
             B E2-minimised, C (native source)

Note: probe-aligned is algebraically identical to ridge under isotropic
regularization, so curves will overlap. Both are shown for verification.

Also writes ALL_pairs_summary_<token>_a<alpha>.json with key numbers per pair.

CLI usage:
  python -m sep.transfer.plot_transfer_comparison_probe_aligned \
    --pairs-dir sep_scratch/transfer/all_maps \
    --pair-names llama2_to_mistral llama32-1b_to_llama31-8b \
                 llama31-8b_to_qwen3-8b llama31-8b_to_phi4 \
                 llama31-8b_to_gemma llama31-8b_to_nemo \
    --pair-labels "Llama2-7B->Mistral-7B" "Llama3.2-1B->Llama3.1-8B" \
                  "Llama3.1-8B->Qwen3-8B" "Llama3.1-8B->Phi-4" \
                  "Llama3.1-8B->Gemma-12B" "Llama3.1-8B->Mistral-Nemo" \
    --pair-types cross-family cross-scale cross-family cross-family cross-family cross-family \
    --token slt --alpha 1e4 \
    --out-dir sep_scratch/transfer/all_maps
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colours and markers shared across both figures
STYLE = {
    "A":   dict(marker="o", ls="-",  color="#333333"),
    "B_r": dict(marker="s", ls="--", color="#4C72B0"),
    "B_p": dict(marker="^", ls="--", color="#DD8452"),
    "B_pa": dict(marker="D", ls="--", color="#9467BD"),
    "B_e": dict(marker="P", ls="--", color="#C44E52"),
    "C":   dict(marker="v", ls=":",  color="#55A868"),
}


def load_results(pairs_dir, pair_names, token, alpha):
    suffix = "" if alpha == 1e3 else f"_a{alpha:.0e}"
    data = {}
    for name in pair_names:
        path = os.path.join(pairs_dir, name, f"transfer_{token}{suffix}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"result not found: {path}")
        with open(path) as f:
            data[name] = json.load(f)
    return data


def _nan(v):
    return [x if x is not None else np.nan for x in v]


def build_summary(data, pair_names, pair_labels, pair_types):
    rows = []
    for name, label, ptype in zip(pair_names, pair_labels, pair_types):
        r = data[name]
        g = r["n_grid"]
        full_n = max(g)
        i = g.index(full_n)
        a   = r["curveA_native"][i]
        br  = r["curveB_ridge"][i]
        bp  = r["curveB_procrustes"][i]
        be2 = r["curveB_e2_minimised"][i] if "curveB_e2_minimised" in r else None
        thr = 0.95 * a if a is not None else None
        nb  = next((g[j] for j, v in enumerate(r["curveB_ridge"])
                    if v >= thr), None) if thr else None
        na  = next((g[j] for j, v in enumerate(r["curveA_native"])
                    if v is not None and v >= thr), None) if thr else None
        rows.append({
            "pair": label, "dir_name": name, "type": ptype,
            "src_layer": r["src_layer"], "tgt_layer": r["tgt_layer"],
            "tgt_se_auroc": r["tgt_layer_auc"],
            f"A_{full_n}": a,
            f"B_ridge_{full_n}": br,
            f"B_proc_{full_n}": bp,
            f"B_e2_minimised_{full_n}": be2,
            "B_ridge_minus_A": (br  - a) if (br  is not None and a is not None) else None,
            "B_e2_minus_A":    (be2 - a) if (be2 is not None and a is not None) else None,
            "nA_95pct_ceiling": na,
            "nB_ridge_95pct_ceiling": nb,
        })
    return rows


def _draw_panel(ax, g, r, label, ptype, x_is_probe_budget):
    A = _nan(r["curveA_native"])
    ax.plot(g, A, label="A: native target (labeled)", **STYLE["A"])
    if x_is_probe_budget:
        if "curveB_ridge_src_probe_budget" in r:
            ax.plot(g, _nan(r["curveB_ridge_src_probe_budget"]),
                    label="B: ridge (fixed map)", **STYLE["B_r"])
        if "curveB_procrustes_src_probe_budget" in r:
            ax.plot(g, _nan(r["curveB_procrustes_src_probe_budget"]),
                    label="B: Procrustes (fixed map)", **STYLE["B_p"])
        if "curveB_probe_aligned_src_probe_budget" in r:
            ax.plot(g, _nan(r["curveB_probe_aligned_src_probe_budget"]),
                    label="B: probe-aligned (refit)", **STYLE["B_pa"])
        if "curveB_e2_src_probe_budget" in r:
            ax.plot(g, _nan(r["curveB_e2_src_probe_budget"]),
                    label="B: E2-minimised (refit)", **STYLE["B_e"])
        if "curveC_source_native" in r:
            ax.plot(g, _nan(r["curveC_source_native"]),
                    label="C: native source (labeled)", **STYLE["C"])
    else:
        ax.plot(g, r["curveB_ridge"],
                label="B: ridge", **STYLE["B_r"])
        ax.plot(g, r["curveB_procrustes"],
                label="B: Procrustes", **STYLE["B_p"])
        if "curveB_probe_aligned" in r:
            ax.plot(g, r["curveB_probe_aligned"],
                    label="B: probe-aligned", **STYLE["B_pa"])
        if "curveB_e2_minimised" in r:
            ax.plot(g, r["curveB_e2_minimised"],
                    label="B: E2-minimised", **STYLE["B_e"])
        if "curveC_source_native" in r:
            ax.plot(g, _nan(r["curveC_source_native"]),
                    label="C: native source (labeled)", **STYLE["C"])

    ax.set_xscale("log")
    ax.set_ylim(0.55, 0.95)
    ax.set_title(f"{label}\n[{ptype}]  tgt ceiling={r['tgt_layer_auc']:.3f}",
                 fontsize=10)
    ax.set_xlabel("labeled source probe examples" if x_is_probe_budget
                  else "unlabeled alignment pairs")
    ax.set_ylabel("eval AUROC")
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)
    ax.legend(fontsize=7)


def plot_all_pairs(data, pair_names, pair_labels, pair_types,
                   token, alpha, out_dir, x_is_probe_budget):
    n = len(pair_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    ax_list = np.array(axes).ravel()

    for ax, name, label, ptype in zip(ax_list, pair_names, pair_labels, pair_types):
        _draw_panel(ax, data[name]["n_grid"], data[name], label, ptype, x_is_probe_budget)

    for ax in ax_list[n:]:
        ax.axis("off")

    alpha_txt = f", α={alpha:.0e}" if alpha != 1e3 else ""
    xkind = "probe training budget" if x_is_probe_budget else "map alignment budget"
    fig.suptitle(
        f"SEP cross-model transfer ({token.upper()}{alpha_txt}) — {xkind}",
        fontsize=13)
    plt.tight_layout()

    suffix = "" if alpha == 1e3 else f"_a{alpha:.0e}"
    tag = "probebudget" if x_is_probe_budget else "mapbudget"
    p = os.path.join(out_dir, f"ALL_pairs_{tag}_{token}{suffix}.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"saved -> {p}")
    return p


def _print_summary(rows):
    print(f"\n{'pair':<30}{'type':<14}{'tgtAUC':>8}{'A@full':>8}"
          f"{'B-ridge':>8}{'B-proc':>8}{'B-e2':>8}{'dRidge':>7}{'dE2':>7}")
    for r in rows:
        keys = list(r.keys())
        a_key  = next(k for k in keys if k.startswith("A_"))
        br_key = next(k for k in keys if k.startswith("B_ridge_"))
        bp_key = next(k for k in keys if k.startswith("B_proc_"))
        be_key = next(k for k in keys if k.startswith("B_e2_minimised_"))
        a, br, bp, be = r[a_key], r[br_key], r[bp_key], r[be_key]
        print(f"{r['pair']:<30}{r['type']:<14}{r['tgt_se_auroc']:>8.3f}"
              f"{(a or 0):>8.3f}{(br or 0):>8.3f}{(bp or 0):>8.3f}{(be or 0):>8.3f}"
              f"{(r['B_ridge_minus_A'] or 0):>+7.3f}{(r['B_e2_minus_A'] or 0):>+7.3f}")


def run(pairs_dir, pair_names, pair_labels, pair_types, token, alpha, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data = load_results(pairs_dir, pair_names, token, alpha)
    summary = build_summary(data, pair_names, pair_labels, pair_types)

    suffix = "" if alpha == 1e3 else f"_a{alpha:.0e}"
    summary_path = os.path.join(out_dir, f"ALL_pairs_summary_{token}{suffix}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary -> {summary_path}")

    _print_summary(summary)
    plot_all_pairs(data, pair_names, pair_labels, pair_types,
                   token, alpha, out_dir, x_is_probe_budget=False)
    plot_all_pairs(data, pair_names, pair_labels, pair_types,
                   token, alpha, out_dir, x_is_probe_budget=True)


def main():
    p = argparse.ArgumentParser(
        description="Aggregate transfer figures (map budget + probe budget).")
    p.add_argument("--pairs-dir", required=True)
    p.add_argument("--pair-names", nargs="+", required=True)
    p.add_argument("--pair-labels", nargs="+", default=None)
    p.add_argument("--pair-types",  nargs="+", default=None)
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--alpha", type=float, default=1e3)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    n = len(a.pair_names)
    labels = a.pair_labels if a.pair_labels else a.pair_names
    types  = a.pair_types  if a.pair_types  else ["unknown"] * n
    assert len(labels) == n, "--pair-labels must match --pair-names length"
    assert len(types)  == n, "--pair-types must match --pair-names length"
    out = a.out_dir if a.out_dir else a.pairs_dir
    run(a.pairs_dir, a.pair_names, labels, types, a.token, a.alpha, out)


if __name__ == "__main__":
    main()
