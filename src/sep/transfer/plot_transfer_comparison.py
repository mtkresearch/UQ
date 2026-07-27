"""Aggregate comparison plots across multiple transfer pairs.

Reads the per-pair transfer_slt_a{alpha}.json files produced by transfer.py
and generates two figures:

  1. ALL_pairs_transfer_<token>_a<alpha>.png  --  2xN grid of per-pair
     Curve A (native) / Curve B ridge / Curve B Procrustes vs pool size.

  2. ALL_pairs_bar_<token>_a<alpha>.png  --  side-by-side bars comparing
     native probe (Curve A at full pool) vs transferred probe (Curve B ridge
     at full pool), one bar group per pair.

Also writes ALL_pairs_summary_<token>_a<alpha>.json with key numbers per pair.

CLI usage (minimal):
  python -m sep.transfer.plot_transfer_comparison \
    --pairs-dir sep_scratch/transfer \
    --pair-names llama2_to_mistral llama32-1b_to_llama31-8b ... \
    --pair-labels "Llama2-7B->Mistral-7B" "Llama3.2-1B->Llama3.1-8B" ... \
    --pair-types cross-family cross-scale ... \
    --token slt --alpha 1e4 --out-dir sep_scratch/transfer

If --pair-labels or --pair-types are omitted they default to the --pair-names.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def build_summary(data, pair_names, pair_labels, pair_types):
    rows = []
    for name, label, ptype in zip(pair_names, pair_labels, pair_types):
        r = data[name]
        g = r["n_grid"]
        full_n = max(g)
        i = g.index(full_n)
        a  = r["curveA_native"][i]
        br = r["curveB_ridge"][i]
        bp = r["curveB_procrustes"][i]
        thr = 0.95 * a if a is not None else None
        nb = next((g[j] for j, v in enumerate(r["curveB_ridge"])
                   if v >= thr), None) if thr else None
        na = next((g[j] for j, v in enumerate(r["curveA_native"])
                   if v is not None and v >= thr), None) if thr else None
        rows.append({
            "pair": label, "dir_name": name, "type": ptype,
            "src_layer": r["src_layer"], "tgt_layer": r["tgt_layer"],
            "tgt_se_auroc": r["tgt_layer_auc"],
            f"A_{full_n}": a,
            f"B_ridge_{full_n}": br,
            f"B_proc_{full_n}": bp,
            "B_minus_A": (br - a) if (br is not None and a is not None) else None,
            "nA_95pct_ceiling": na,
            "nB_95pct_ceiling": nb,
        })
    return rows


def plot_curves(data, pair_names, pair_labels, pair_types, token, alpha, out_dir):
    n = len(pair_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    ax_list = np.array(axes).ravel()

    for ax, name, label, ptype in zip(ax_list, pair_names, pair_labels, pair_types):
        r = data[name]
        g = r["n_grid"]
        A = [v if v is not None else np.nan for v in r["curveA_native"]]
        ax.plot(g, A, "o-", color="#333333", label="A: native (labeled)")
        ax.plot(g, r["curveB_ridge"], "s--", color="#4C72B0",
                label="B: transfer ridge")
        ax.plot(g, r["curveB_procrustes"], "^--", color="#DD8452",
                label="B: transfer Procrustes")
        ax.set_xscale("log")
        ax.set_ylim(0.55, 0.92)
        ax.set_title(f"{label}\n[{ptype}]  tgt-SE ceiling={r['tgt_layer_auc']:.3f}",
                     fontsize=10)
        ax.set_xlabel("target examples used")
        ax.set_ylabel("eval AUROC")
        ax.grid(True, ls="--", lw=0.4, alpha=0.6)
        ax.legend(fontsize=7)

    for ax in ax_list[n:]:
        ax.axis("off")

    alpha_txt = f", α={alpha:.0e}" if alpha != 1e3 else ""
    fig.suptitle(
        f"SEP cross-model transfer ({token.upper()}{alpha_txt}): "
        f"native vs transferred probe",
        fontsize=13)
    plt.tight_layout()
    suffix = "" if alpha == 1e3 else f"_a{alpha:.0e}"
    p = os.path.join(out_dir, f"ALL_pairs_transfer_{token}{suffix}.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"saved curves figure -> {p}")
    return p


def plot_bars(data, pair_names, pair_labels, pair_types, token, alpha, out_dir):
    n = len(pair_names)
    full_n = max(data[pair_names[0]]["n_grid"])
    A  = [data[k]["curveA_native"][data[k]["n_grid"].index(full_n)] for k in pair_names]
    Br = [data[k]["curveB_ridge"][data[k]["n_grid"].index(full_n)] for k in pair_names]

    fig, ax = plt.subplots(figsize=(max(9, 2 * n), 5))
    x = np.arange(n)
    w = 0.4
    ax.bar(x - w / 2, A,  w, label=f"A: native target probe ({full_n} labeled)",
           color="#333333")
    ax.bar(x + w / 2, Br, w, label="B: transferred probe (ridge, 0 labels)",
           color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("eval AUROC")
    ax.set_ylim(0.70, 0.92)
    ax.legend()
    ax.set_title(
        f"Transferred (0 target labels) vs native ({full_n} target labels) — "
        f"full-pool AUROC ({token.upper()})")
    for i, (a, b) in enumerate(zip(A, Br)):
        if a is not None:
            ax.text(i - w / 2, a + 0.003, f"{a:.3f}", ha="center", fontsize=7)
        if b is not None:
            ax.text(i + w / 2, b + 0.003, f"{b:.3f}", ha="center", fontsize=7)
    plt.tight_layout()
    suffix = "" if alpha == 1e3 else f"_a{alpha:.0e}"
    p = os.path.join(out_dir, f"ALL_pairs_bar_{token}{suffix}.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"saved bar figure -> {p}")
    return p


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
    plot_curves(data, pair_names, pair_labels, pair_types, token, alpha, out_dir)
    plot_bars(data, pair_names, pair_labels, pair_types, token, alpha, out_dir)


def _print_summary(rows):
    print(f"\n{'pair':<30}{'type':<14}{'tgtAUC':>8}{'A@full':>8}"
          f"{'B-ridge':>8}{'B-proc':>8}{'B-A':>7}{'nA95':>6}{'nB95':>6}")
    for r in rows:
        key = list(r.keys())
        a_key  = next(k for k in key if k.startswith("A_"))
        br_key = next(k for k in key if k.startswith("B_ridge_"))
        bp_key = next(k for k in key if k.startswith("B_proc_"))
        a, br, bp = r[a_key], r[br_key], r[bp_key]
        d = r["B_minus_A"]
        print(f"{r['pair']:<30}{r['type']:<14}{r['tgt_se_auroc']:>8.3f}"
              f"{(a or 0):>8.3f}{(br or 0):>8.3f}{(bp or 0):>8.3f}"
              f"{(d or 0):>+7.3f}"
              f"{(r['nA_95pct_ceiling'] or -1):>6}"
              f"{(r['nB_95pct_ceiling'] or -1):>6}")


def main():
    p = argparse.ArgumentParser(
        description="Aggregate transfer-curve comparison figures.")
    p.add_argument("--pairs-dir", required=True,
                   help="directory containing one sub-dir per pair")
    p.add_argument("--pair-names", nargs="+", required=True,
                   help="sub-directory names (one per pair)")
    p.add_argument("--pair-labels", nargs="+", default=None,
                   help="display labels (default: same as --pair-names)")
    p.add_argument("--pair-types", nargs="+", default=None,
                   help="type tags e.g. cross-family / cross-scale "
                        "(default: 'unknown' for all)")
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--alpha", type=float, default=1e3)
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: --pairs-dir)")
    a = p.parse_args()
    n = len(a.pair_names)
    labels = a.pair_labels if a.pair_labels else a.pair_names
    types  = a.pair_types  if a.pair_types  else ["unknown"] * n
    assert len(labels) == n, "--pair-labels must be same length as --pair-names"
    assert len(types)  == n, "--pair-types must be same length as --pair-names"
    out = a.out_dir if a.out_dir else a.pairs_dir
    run(a.pairs_dir, a.pair_names, labels, types, a.token, a.alpha, out)


if __name__ == "__main__":
    main()
