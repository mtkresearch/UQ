"""Plot the selected best alpha/lambda per pair: 5 curves per panel.

  - black  : target probe (curve A), from baselines
  - grey   : source probe (curve C), from baselines
  - blue   : ridge, fixed alpha per pair
  - green  : E2-map, fixed lambda per pair
  - orange : E2-R0, best lambda auto-picked (max AUROC at largest n_grid point)
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAIRS_DIR = "sep_scratch/transfer/hyperparam_sweep_v3"

PAIR_NAMES = [
    "llama2_to_mistral",
    "llama32-1b_to_llama31-8b",
    "llama31-8b_to_qwen3-8b",
    "llama31-8b_to_phi4",
    "llama31-8b_to_gemma",
    "llama31-8b_to_nemo",
]

PAIR_LABELS = [
    "Llama2-7B→Mistral-7B",
    "Llama3.2-1B→Llama3.1-8B",
    "Llama3.1-8B→Qwen3-8B",
    "Llama3.1-8B→Phi-4",
    "Llama3.1-8B→Gemma-12B",
    "Llama3.1-8B→Mistral-Nemo",
]

RIDGE_ALPHA = 1e4

E2MAP_LAMS   = [0.1, 1.0, 10.0, 100.0, 1000.0]
E2R0_LAMS    = [10, 100, 1000, 10000, 100000]
E2RSTAR_LAMS = [10, 100, 1000, 10000, 100000]

TOKEN = "slt"
OUT_DIR = PAIRS_DIR


def _lam_str(lam):
    """Format float lambda to match bash-generated filenames (e.g. 1 not 1.0, 1e-5 not 1e-05)."""
    import re
    return re.sub(r'e([+-])0+(\d)', r'e\1\2', f"{lam:g}")


def _nan(v):
    return [x if x is not None else np.nan for x in v]


def _metric_suffix(metric):
    return "" if metric == "auroc" else f"_{metric}"


def _best_e2map_lam(pair_dir, token, lams, metric="auroc", axis="probe_budget"):
    ms = _metric_suffix(metric)
    bs = "_src_probe_budget" if axis == "probe_budget" else ""
    key = f"curveB_e2_map{bs}{ms}"
    best_is_max = (metric == "auroc")
    best_lam = lams[0]
    best_val = -np.inf if best_is_max else np.inf
    for lam in lams:
        path = os.path.join(pair_dir, f"transfer_{token}_lam{lam}_e2m.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        vals = d.get(key, [])
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        if best_is_max and vals[-1] > best_val:
            best_val, best_lam = vals[-1], lam
        elif not best_is_max and vals[-1] < best_val:
            best_val, best_lam = vals[-1], lam
    return best_lam


def _best_e2r0_lam(pair_dir, token, lams, metric="auroc", axis="probe_budget"):
    """Return the lambda with best metric value at the largest n_grid point.

    For auroc: highest value. For error_rate: lowest value.
    """
    ms = _metric_suffix(metric)
    bs = "_src_probe_budget" if axis == "probe_budget" else ""
    key = f"curveB_e2_r0{bs}{ms}"
    best_is_max = (metric == "auroc")
    best_lam = lams[0]
    best_val = -np.inf if best_is_max else np.inf
    for lam in lams:
        path = os.path.join(pair_dir, f"transfer_{token}_lam{_lam_str(lam)}_e2r0.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        vals = d.get(key, [])
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        if best_is_max and vals[-1] > best_val:
            best_val, best_lam = vals[-1], lam
        elif not best_is_max and vals[-1] < best_val:
            best_val, best_lam = vals[-1], lam
    return best_lam


def _best_e2rstar_lam(pair_dir, token, lams, metric="auroc", axis="probe_budget"):
    """Return the lambda with best metric value at the largest n_grid point for R*."""
    ms = _metric_suffix(metric)
    bs = "_src_probe_budget" if axis == "probe_budget" else ""
    key = f"curveB_e2_rstar{bs}{ms}"
    best_is_max = (metric == "auroc")
    best_lam = lams[0]
    best_val = -np.inf if best_is_max else np.inf
    for lam in lams:
        path = os.path.join(pair_dir, f"transfer_{token}_lam{_lam_str(lam)}_e2rstar.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        vals = d.get(key, [])
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        if best_is_max and vals[-1] > best_val:
            best_val, best_lam = vals[-1], lam
        elif not best_is_max and vals[-1] < best_val:
            best_val, best_lam = vals[-1], lam
    return best_lam


def load(path):
    with open(path) as f:
        return json.load(f)


def _draw_panel(ax, name, label, metric="auroc", axis="probe_budget"):
    d = os.path.join(PAIRS_DIR, name)
    ms = _metric_suffix(metric)
    bs = "_src_probe_budget" if axis == "probe_budget" else ""

    # baselines
    b = load(os.path.join(d, f"transfer_{TOKEN}_baselines.json"))
    g = b["n_grid"]
    key_a = f"curveA_native{ms}"
    key_c = f"curveC_source_native{ms}"
    if b.get(key_a):
        ax.plot(g, _nan(b[key_a]),
                color="black", ls="-", marker="o", ms=5, lw=1.8,
                label="A: native target probe")
    if b.get(key_c):
        ax.plot(g, _nan(b[key_c]),
                color="grey", ls=":", marker="v", ms=5, lw=1.5,
                label="C: native source probe")

    # ridge α=1e4
    rdg = load(os.path.join(d, f"transfer_{TOKEN}_a1e+04_rdg.json"))
    key_r = f"curveB_ridge{bs}{ms}"
    if key_r in rdg:
        ax.plot(g, _nan(rdg[key_r]),
                color="#1f77b4", ls="--", marker="s", ms=5, lw=1.8,
                label="ridge α=1e+04")

    # E2-map, best lambda auto-picked by chosen metric + axis
    best_lam_e2m = _best_e2map_lam(d, TOKEN, E2MAP_LAMS, metric=metric, axis=axis)
    e2m_path = os.path.join(d, f"transfer_{TOKEN}_lam{best_lam_e2m}_e2m.json")
    if os.path.exists(e2m_path):
        e2 = load(e2m_path)
        key_e = f"curveB_e2_map{bs}{ms}"
        if key_e in e2:
            ax.plot(g, _nan(e2[key_e]),
                    color="#2ca02c", ls="-.", marker="X", ms=5, lw=1.8,
                    label=f"E2-map λ={best_lam_e2m:.0e}")

    # E2-R0, best lambda auto-picked by chosen metric + axis
    best_lam = _best_e2r0_lam(d, TOKEN, E2R0_LAMS, metric=metric, axis=axis)
    e2r0 = load(os.path.join(d, f"transfer_{TOKEN}_lam{_lam_str(best_lam)}_e2r0.json"))
    key_r0 = f"curveB_e2_r0{bs}{ms}"
    if key_r0 in e2r0:
        ax.plot(g, _nan(e2r0[key_r0]),
                color="#ff7f0e", ls=":", marker="h", ms=5, lw=1.8,
                label=f"E2-R0 λ={best_lam:.0e}")

    # E2-R*, best lambda auto-picked by chosen metric + axis
    best_lam_rs = _best_e2rstar_lam(d, TOKEN, E2RSTAR_LAMS, metric=metric, axis=axis)
    path_rs = os.path.join(d, f"transfer_{TOKEN}_lam{_lam_str(best_lam_rs)}_e2rstar.json")
    if os.path.exists(path_rs):
        e2rs = load(path_rs)
        key_rs = f"curveB_e2_rstar{bs}{ms}"
        if key_rs in e2rs:
            ax.plot(g, _nan(e2rs[key_rs]),
                    color="#d62728", ls="-", marker="*", ms=5, lw=1.8,
                    label=f"E2-R* λ={best_lam_rs:.0e}")

    ylabel = "eval AUROC" if metric == "auroc" else "error rate  Pr(ẑ ≠ z)"
    xlabel = "unlabeled map-fitting pairs" if axis == "map_budget" else "labeled source probe examples"
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(label, fontsize=11)
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)
    ax.legend(fontsize=8)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Plot best-hyperparameter curves per pair.")
    p.add_argument("--metric", default="auroc", choices=["auroc", "error_rate"],
                   help="metric used to select best lambda and for y-axis")
    p.add_argument("--axis", default="probe_budget", choices=["probe_budget", "map_budget"],
                   help="which budget axis to plot")
    p.add_argument("--pairs-dir", default=None,
                   help="directory containing per-pair subdirs (default: PAIRS_DIR constant)")
    p.add_argument("--e2map-lams", nargs="+", type=float, default=None)
    p.add_argument("--e2r0-lams",  nargs="+", type=float, default=None)
    p.add_argument("--e2rstar-lams", nargs="+", type=float, default=None)
    a = p.parse_args()
    global PAIRS_DIR, OUT_DIR, E2MAP_LAMS, E2R0_LAMS, E2RSTAR_LAMS
    metric = a.metric
    axis = a.axis
    if a.pairs_dir:
        PAIRS_DIR = a.pairs_dir
        OUT_DIR = a.pairs_dir
    if a.e2map_lams:
        E2MAP_LAMS = a.e2map_lams
    if a.e2r0_lams:
        E2R0_LAMS = a.e2r0_lams
    if a.e2rstar_lams:
        E2RSTAR_LAMS = a.e2rstar_lams

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    ax_list = np.array(axes).ravel()

    for ax, name, label in zip(ax_list, PAIR_NAMES, PAIR_LABELS):
        _draw_panel(ax, name, label, metric=metric, axis=axis)

    metric_label = "AUROC" if metric == "auroc" else "error rate  Pr(ẑ ≠ z)"
    axis_label = "map fitting budget" if axis == "map_budget" else "probe training budget"
    fig.suptitle(
        f"Ridge (α=1e+04) vs E2-map vs E2-R0 vs E2-R* (best λ per pair) — {axis_label} (SLT) — {metric_label}",
        fontsize=13)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    ms = _metric_suffix(metric)
    axis_tag = "_mapbudget" if axis == "map_budget" else ""
    out = os.path.join(OUT_DIR, f"ALL_pairs_best_hyperparams_{TOKEN}{axis_tag}{ms}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
