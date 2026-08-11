"""Combine per-pair transfer results into multi-panel figures.

Produces 6 combined PDFs (3 metrics × 2 budget types):
  - combined_map_budget_auroc.pdf       -- AUROC vs map alignment budget
  - combined_map_budget_spearman.pdf    -- Spearman ρ vs map alignment budget
  - combined_map_budget_kendall.pdf     -- Kendall τ vs map alignment budget
  - combined_probe_budget_auroc.pdf     -- AUROC vs probe training budget
  - combined_probe_budget_spearman.pdf  -- Spearman ρ vs probe training budget
  - combined_probe_budget_kendall.pdf   -- Kendall τ vs probe training budget

Usage:
  python -m sep.transfer.plot_panel \
      --in-dir sep_scratch/transfer/all_maps_ranking \
      --out-dir sep_scratch/transfer/all_maps_ranking \
      --token slt --alpha 1e4
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAIRS = [
    ("llama2_to_mistral",         "Llama-2-7B → Mistral-7B"),
    ("llama32-1b_to_llama31-8b",  "Llama-3.2-1B → Llama-3.1-8B"),
    ("llama31-8b_to_qwen3-8b",    "Llama-3.1-8B → Qwen3-8B"),
    ("llama31-8b_to_phi4",        "Llama-3.1-8B → Phi-4"),
    ("llama31-8b_to_gemma",       "Llama-3.1-8B → Gemma-4-12B"),
    ("llama31-8b_to_nemo",        "Llama-3.1-8B → Mistral-Nemo"),
]

# Curve specs: (json_key_suffix, marker+linestyle, color, short label)
MAP_BUDGET_CURVES = [
    ("",              "o-",  "#333333", "Curve A: native target (labeled)"),
    ("_ridge",        "s--", "#4C72B0", "Curve B: ridge (unlabeled)"),
    ("_procrustes",   "^--", "#DD8452", "Curve B: Procrustes (unlabeled)"),
    ("_probe_aligned","D--", "#8172B2", "Curve B: probe-aligned (unlabeled)"),
    ("_e2_minimised", "P--", "#C44E52", "Curve B: E2-minimised (unlabeled)"),
    ("_source_native","v:",  "#2ca02c", "Curve C: native source (labeled)"),
]

PROBE_BUDGET_CURVES = [
    ("",                        "o-",  "#333333", "Curve A: native target (labeled target examples)"),
    ("_src_probe_budget",       "s--", "#4C72B0", "Curve B: ridge map → target"),
    ("_procrustes_src_probe_budget", "^--", "#DD8452", "Curve B: Procrustes map → target"),
    ("_probe_aligned_src_probe_budget", "D--", "#8172B2", "Curve B: probe-aligned map → target"),
    ("_e2_src_probe_budget",    "P--", "#C44E52", "Curve B: E2-minimised map → target"),
    ("_source_native",          "v:",  "#2ca02c", "Curve C: native source (labeled source examples)"),
]

METRIC_META = {
    "auroc":    {"ylabel": "AUROC (target SE)",       "suffix": ""},
    "spearman": {"ylabel": "Spearman ρ (vs entropy)", "suffix": "_spearman"},
    "kendall":  {"ylabel": "Kendall τ (vs entropy)",  "suffix": "_kendall"},
}


def _load(in_dir, pair_dir, token, alpha_str):
    path = os.path.join(in_dir, pair_dir, f"transfer_{token}_a{alpha_str}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def _get_curve(res, base_key, metric_suffix):
    """Return curve values array or None if not in res."""
    key = base_key + metric_suffix
    if key not in res:
        return None
    return [v if v is not None else np.nan for v in res[key]]


def _plot_panel(in_dir, out_dir, token, alpha_str, metric, budget):
    """Draw one 2×3 panel figure."""
    meta = METRIC_META[metric]
    metric_suffix = meta["suffix"]

    if budget == "map":
        x_label = "target examples used (labeled for A, unlabeled for B)"
        curves_spec = MAP_BUDGET_CURVES
        # key prefixes in json for map budget
        # curveA_native{metric}, curveB_ridge{metric}, ...
        def curve_key(suffix, metric_suffix):
            if suffix == "":
                return f"curveA_native{metric_suffix}"
            if suffix == "_source_native":
                return f"curveC_source_native{metric_suffix}"
            return f"curveB{suffix}{metric_suffix}"
    else:
        x_label = "labeled examples used to train the probe"
        curves_spec = PROBE_BUDGET_CURVES
        def curve_key(suffix, metric_suffix):
            if suffix == "":
                return f"curveA_native{metric_suffix}"
            if suffix == "_source_native":
                return f"curveC_source_native{metric_suffix}"
            # probe-budget curve keys differ from map-budget
            raw = suffix.lstrip("_")
            if raw == "src_probe_budget":
                return f"curveB_ridge_src_probe_budget{metric_suffix}"
            if raw == "procrustes_src_probe_budget":
                return f"curveB_procrustes_src_probe_budget{metric_suffix}"
            if raw == "probe_aligned_src_probe_budget":
                return f"curveB_probe_aligned_src_probe_budget{metric_suffix}"
            if raw == "e2_src_probe_budget":
                return f"curveB_e2_src_probe_budget{metric_suffix}"
            return None

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             sharex=True, sharey=False)
    axes_flat = axes.flatten()

    legend_handles, legend_labels = [], []

    for idx, (pair_dir, pair_label) in enumerate(PAIRS):
        ax = axes_flat[idx]
        try:
            res = _load(in_dir, pair_dir, token, alpha_str)
        except FileNotFoundError as e:
            ax.text(0.5, 0.5, f"Missing:\n{e}", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=8)
            ax.set_title(pair_label, fontsize=10)
            continue

        g = res["n_grid"]
        for c_suffix, style, color, label in curves_spec:
            key = curve_key(c_suffix, metric_suffix)
            if key is None or key not in res:
                continue
            vals = [v if v is not None else np.nan for v in res[key]]
            line, = ax.plot(g, vals, style, color=color, label=label,
                            linewidth=1.5, markersize=5)
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        ax.set_xscale("log")
        ax.set_title(pair_label, fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
        ax.tick_params(labelsize=8)

        row = idx // ncols
        col = idx % ncols
        if row == nrows - 1:
            ax.set_xlabel(x_label, fontsize=8)
        if col == 0:
            ax.set_ylabel(meta["ylabel"], fontsize=9)

    metric_title = {"auroc": "AUROC", "spearman": "Spearman ρ", "kendall": "Kendall τ"}[metric]
    budget_title = "map alignment budget" if budget == "map" else "probe training budget"
    fig.suptitle(
        f"Cross-model SE probe transfer — {metric_title} vs {budget_title} "
        f"({token.upper()}, α={alpha_str})",
        fontsize=13, y=1.01,
    )

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center",
                   ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04),
                   frameon=True, framealpha=0.9)

    plt.tight_layout()
    fname = f"combined_{budget}_budget_{metric}.pdf"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True,
                   help="Directory containing the per-pair subdirs (e.g. sep_scratch/transfer/all_maps_ranking)")
    p.add_argument("--out-dir", default=None,
                   help="Where to save combined PDFs (defaults to --in-dir)")
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--alpha", default="1e+04",
                   help="Alpha string as it appears in filenames, e.g. '1e+04'")
    args = p.parse_args()

    out_dir = args.out_dir or args.in_dir
    os.makedirs(out_dir, exist_ok=True)

    for metric in ("auroc", "spearman", "kendall"):
        for budget in ("map", "probe"):
            print(f"Plotting {metric} / {budget} budget ...")
            _plot_panel(args.in_dir, out_dir, args.token, args.alpha, metric, budget)

    print("Done.")


if __name__ == "__main__":
    main()
