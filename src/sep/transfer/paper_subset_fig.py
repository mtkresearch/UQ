"""paper_subset_fig.py — Generate a 2×3 subset figure for the main paper.

Plots 6 representative source→target pairs from the full 21-pair grid,
covering good transfer, a harder case, and cross-scale transfer.

Follows the same style as plot_final_figures.make_paper_fig: no suptitle,
no diagnostic annotations (ceil, alpha), publication-ready legend labels.

Usage:
    python -m sep.transfer.paper_subset_fig \
        --out-dir /path/to/sep_scratch/transfer_v2_trivia_qa \
        --dataset trivia_qa \
        --alpha 1e4

The figure is saved to <out-dir>/summary_plots/ as both PDF and PNG.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sep.transfer.transfer2 import (
    _load_json,
    _aligner_tag,
    make_run_variant_selector,
    C_NATIVE, C_SRC_BASE, C_ALIGNER_SAME, C_ALIGNER_CROSS,
    SURFACE, INK_PRI, INK_SEC, GRIDLINE,
)

# ---------------------------------------------------------------------------
# Subset definition
# ---------------------------------------------------------------------------

# 2×3 grid of pair keys (row, col).
# Row 0: good-transfer cases  |  Row 1: harder + cross-scale
SUBSET_GRID = [
    [
        "llama-3.1-8b_to_gemma-4-12b",   # near-parity
        "gemma-4-12b_to_phi-4",           # transfer beats native
        "mistral-nemo_to_qwen3-8b",       # best overall AUROC
    ],
    [
        "phi-4_to_llama-3.1-8b",          # mild gap
        "qwen3-8b_to_mistral-nemo",       # harder case
        "llama-3.2-1b_to_llama-3.1-8b",  # cross-scale
    ],
]

_SHORT = {
    "gemma-4-12b":  "Gemma-4-12B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "mistral-nemo": "Mistral-Nemo",
    "phi-4":        "Phi-4",
    "qwen3-8b":     "Qwen3-8B",
    "llama-3.2-1b": "Llama-3.2-1B",
}


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def make_subset_fig(eval_ds, results_dir, select_variant, cross_datasets=None):
    if cross_datasets is None:
        cross_datasets = ["nq", "squad"]

    nrows = len(SUBSET_GRID)
    ncols = max(len(r) for r in SUBSET_GRID)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5),
                             facecolor=SURFACE, constrained_layout=True)

    for row_idx, row in enumerate(SUBSET_GRID):
        for col_idx, pair_key in enumerate(row):
            ax = axes[row_idx][col_idx]

            if pair_key is None:
                ax.axis("off")
                continue

            pair_dir  = os.path.join(results_dir, eval_ds, pair_key)
            src, _, tgt = pair_key.partition("_to_")
            src_short = _SHORT.get(src, src)
            tgt_short = _SHORT.get(tgt, tgt)

            ax.set_facecolor(SURFACE)
            for spine in ax.spines.values():
                spine.set_color("#b0b0b0")
                spine.set_linewidth(0.8)
            ax.grid(which="major", color=GRIDLINE, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)

            native_path = os.path.join(pair_dir, f"native_curves_{eval_ds}.json")
            if not os.path.exists(native_path):
                ax.set_title(f"target: {tgt_short}\n(missing)", fontsize=8, color="red")
                ax.axis("off")
                continue

            nd     = _load_json(native_path)
            g      = nd["n_grid"]
            native = [v if v is not None else float("nan") for v in nd["curveA_native"]]

            ax.plot(g, native, color=C_NATIVE, linewidth=1.5, marker="o",
                    markersize=4, markeredgewidth=0, label="native target", zorder=3)

            src_curve = nd.get("curveA_src")
            if src_curve:
                src_vals = [v if v is not None else float("nan") for v in src_curve]
                ax.plot(g, src_vals, color=C_SRC_BASE, linewidth=1.5, marker="D",
                        markersize=4, markeredgewidth=0, linestyle="-",
                        label="native source", zorder=3)

            same_pick = select_variant(pair_dir, eval_ds)
            if same_pick is not None:
                _, same = same_pick
                for i, a in enumerate(same.get("aligners", ["ridge"])):
                    curve = same.get(f"curveB_{a}")
                    if curve:
                        ax.plot(g, curve,
                                color=C_ALIGNER_SAME[i % len(C_ALIGNER_SAME)],
                                linewidth=1.5, marker="s", markersize=4,
                                markeredgewidth=0, linestyle="-",
                                label=f"transferred alignment ({eval_ds.upper()})",
                                zorder=3)

            color_idx = 0
            for ds in cross_datasets:
                cross_pick = select_variant(pair_dir, ds)
                if cross_pick is None:
                    continue
                _, cross = cross_pick
                xg = cross["n_grid"]
                for a in cross.get("aligners", ["ridge"]):
                    curve = cross.get(f"curveB_{a}")
                    if curve:
                        c = C_ALIGNER_CROSS[color_idx % len(C_ALIGNER_CROSS)]
                        ax.plot(xg, curve, color=c, linewidth=1.5, marker="s",
                                markersize=4, markeredgewidth=0, linestyle="--",
                                label=f"transferred alignment ({ds.upper()})", zorder=3)
                        color_idx += 1

            ax.set_xscale("log")
            ax.set_xlabel("No. of source probe training samples",
                          fontsize=7, color=INK_SEC)
            ax.set_ylabel("AUROC", fontsize=7, color=INK_SEC)
            ax.tick_params(colors=INK_SEC, labelsize=7, length=3)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_tick_params(which="minor", bottom=False)
            ax.set_xticks(g)
            ax.set_xticklabels([str(v) for v in g], fontsize=6.5, rotation=30)
            ax.set_ylim(0.45, 0.95)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.set_title(f"source: {src_short} → target: {tgt_short}",
                         fontsize=8.5, color=INK_PRI, pad=3)
            ax.legend(fontsize=5.5, frameon=True, framealpha=0.9,
                      edgecolor="#cccccc", loc="upper left")

        axes[row_idx][0].set_ylabel("AUROC", fontsize=7, color=INK_SEC)

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True,
                        help="Experiment output directory (contains results/ subfolder)")
    parser.add_argument("--dataset", default="trivia_qa",
                        help="Eval dataset tag, e.g. trivia_qa, nq, squad")
    parser.add_argument("--alpha", type=float, default=1e4,
                        help="Ridge regularisation alpha used when building the results")
    parser.add_argument("--cross-align", nargs="*", default=None,
                        help="Cross-alignment datasets to overlay (default: nq squad)")
    args = parser.parse_args()

    results_dir = os.path.join(args.out_dir, "results")
    plots_dir   = os.path.join(args.out_dir, "summary_plots")
    os.makedirs(plots_dir, exist_ok=True)

    hyperparams    = {"ridge": {"alpha": args.alpha}}
    run_tag        = _aligner_tag("ridge", hyperparams["ridge"])
    select_variant = make_run_variant_selector("probe_grid", hyperparams)

    cross_datasets = args.cross_align if args.cross_align else ["nq", "squad"]
    cross_datasets = [d for d in cross_datasets if d != args.dataset]

    fig = make_subset_fig(
        eval_ds        = args.dataset,
        results_dir    = results_dir,
        select_variant = select_variant,
        cross_datasets = cross_datasets,
    )

    stem = f"subset_{args.dataset}_probe_grid_{run_tag}"
    for ext in ("pdf", "png"):
        path = os.path.join(plots_dir, f"{stem}.{ext}")
        dpi  = 200 if ext == "pdf" else 150
        fig.savefig(path, format=ext, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
