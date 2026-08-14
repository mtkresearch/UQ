"""
Plot Venn diagrams and summary tables for SE probe transfer experiments.

Outputs (under <out_dir>/summary_plots/venn_<aligner_suffix>/):
  - 24 PNG figures: venn_<eval_ds>_align_<align_ds>_n<n>.png
  - 2 CSV tables:  table_<eval_ds>_<aligner_suffix>.csv
"""

import os
import json
import argparse
import numpy as np
import csv


# ── layout constants (mirrors transfer2.py) ──────────────────────────────────
_CROSS_SOURCES = ["gemma-4-12b", "llama-3.1-8b", "mistral-nemo", "phi-4", "qwen3-8b"]
_CROSS_TARGETS = ["gemma-4-12b", "llama-3.1-8b", "mistral-nemo", "phi-4", "qwen3-8b"]
_LLAMA_FAMILY  = [("llama-3.2-1b", "llama-3.1-8b")]

_SHORT = {
    "gemma-4-12b":  "Gemma-4-12B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "mistral-nemo": "Mistral-Nemo",
    "phi-4":        "Phi-4",
    "qwen3-8b":     "Qwen3-8B",
    "llama-3.2-1b": "Llama-3.2-1B",
}

SURFACE  = "white"
INK_PRI  = "#1a1a1a"
INK_SEC  = "#404040"

# Colors per circle (A, B, C) — used by venn3 for the base fill
# Individual region overrides are applied after drawing
C_CIRCLE = {"A": "#d0d0d0", "B": "#c00000", "C": "#d0d0d0"}

# Region patch id → fill color (venn3 patch ids: '100','010','001','110','101','011','111')
# '010' = B only, '110' = AB only, '011' = BC only, '111' = ABC
# all others = light grey / uncolored
C_PATCH = {
    "100": "#d8d8d8",   # A only
    "010": "#c00000",   # B only        — pure transfer harm
    "001": "#d8d8d8",   # C only
    "110": "#1a3a8f",   # A∩B only      — transfer fixes source error
    "101": "#d8d8d8",   # A∩C only
    "011": "#aac4e8",   # B∩C only      — transfer adapts to target mismatch
    "111": "#f4a0a0",   # A∩B∩C         — harm with source/target mismatch
}


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _structured_grid(results_dir, eval_ds):
    """Return 5×4 grid of pair keys (or None). Mirrors transfer2.py."""
    ds_dir = os.path.join(results_dir, eval_ds)
    present = set(
        p for p in os.listdir(ds_dir)
        if os.path.isdir(os.path.join(ds_dir, p))
    ) if os.path.isdir(ds_dir) else set()

    grid = []
    for src in _CROSS_SOURCES:
        targets = [t for t in _CROSS_TARGETS if t != src]
        row = [key if (key := f"{src}_to_{tgt}") in present else None
               for tgt in targets]
        grid.append(row)

    fam_row = []
    for src, tgt in _LLAMA_FAMILY:
        key = f"{src}_to_{tgt}"
        fam_row.append(key if key in present else None)
    while len(fam_row) < 4:
        fam_row.append(None)
    grid.append(fam_row)
    return grid


# region key ↔ venn3 patch id
_REGION_ID = {
    "A_only":  "100",
    "B_only":  "010",
    "AB_only": "110",
    "C_only":  "001",
    "AC_only": "101",
    "BC_only": "011",
    "ABC":     "111",
}
# venn3 subset order: (Abc, aBc, ABc, abC, AbC, aBC, ABC)
_VENN3_ORDER = ["A_only", "B_only", "AB_only", "C_only", "AC_only", "BC_only", "ABC"]


def _draw_venn(ax, entry, n, total, title):
    """Draw an area-proportional three-circle Venn (radii/centres solved from the
    actual subset sizes) with colored regions and count labels."""
    from matplotlib_venn import venn3, venn3_circles
    from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    counts = {k: int(entry.get(k, 0) or 0) for k in _VENN3_ORDER}
    subsets = tuple(counts[k] for k in _VENN3_ORDER)

    if sum(subsets) == 0:
        ax.text(0.5, 0.5, "all regions empty", transform=ax.transAxes,
                fontsize=6, ha="center", va="center", color=INK_SEC)
        ax.set_title(title, fontsize=6.5, color=INK_PRI, pad=2)
        return

    # matplotlib_venn solves circle radii from set sizes and pairwise distances
    # from the pairwise intersection areas → subset areas track the real data.
    layout = DefaultLayoutAlgorithm(normalize_to=1.0)
    v = venn3(subsets=subsets, set_labels=("A", "B", "C"), ax=ax,
              layout_algorithm=layout)
    venn3_circles(subsets=subsets, ax=ax, layout_algorithm=layout,
                  linewidth=1.0, color="#555555")

    for key in _VENN3_ORDER:
        pid   = _REGION_ID[key]
        patch = v.get_patch_by_id(pid)
        if patch is not None:
            patch.set_facecolor(C_PATCH.get(pid, "#d8d8d8"))
            patch.set_edgecolor("none")
            patch.set_alpha(1.0)

        label = v.get_label_by_id(pid)
        if label is None:          # region has zero area → no label drawn
            continue
        val = counts[key]
        pct = 100 * val / total if total > 0 else 0
        label.set_text(f"{val}\n({pct:.0f}%)")
        label.set_fontsize(5.2)
        label.set_color("white" if pid in ("010", "110") else INK_PRI)

    for sl in v.set_labels:
        if sl is not None:
            sl.set_fontsize(8)
            sl.set_fontweight("bold")
            sl.set_color(INK_PRI)

    # D annotation
    d_val = entry.get("D", 0)
    d_pct = 100 * d_val / total if total > 0 else 0
    ax.annotate(f"D={d_val} ({d_pct:.0f}%)",
                xy=(1, 0), xycoords="axes fraction",
                fontsize=5.5, ha="right", va="bottom", color="#800000")

    ax.set_title(title, fontsize=6.5, color=INK_PRI, pad=2)


def plot_venn_fig(eval_ds, align_ds, n, results_dir, aligner_suffix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = _structured_grid(results_dir, eval_ds)
    nrows, ncols = len(grid), len(grid[0])
    row_labels = [_SHORT.get(s, s) for s in _CROSS_SOURCES] + [
        _SHORT.get(src, src) for src, _ in _LLAMA_FAMILY
    ]

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 3.2),
                             facecolor=SURFACE, constrained_layout=True)

    fig.suptitle(
        f"Venn — eval: {eval_ds.upper()}  align: {align_ds.upper()}  "
        f"n={n}  [{aligner_suffix}]",
        fontsize=10, color=INK_PRI,
    )

    for row_idx, row in enumerate(grid):
        for col_idx, pair_key in enumerate(row):
            ax = axes[row_idx][col_idx]

            if pair_key is None:
                ax.axis("off")
                continue

            tgt       = pair_key.partition("_to_")[2]
            tgt_short = _SHORT.get(tgt, tgt)

            venn_path = os.path.join(
                results_dir, eval_ds, pair_key,
                f"venn_align_{align_ds}_{aligner_suffix}.json"
            )

            if not os.path.exists(venn_path):
                ax.axis("off")
                ax.set_title(f"→ {tgt_short}\n(missing)", fontsize=6.5, color="red")
                continue

            venn_data = _load_json(venn_path)
            entry     = venn_data["by_n"].get(str(n))

            if entry is None:
                ax.axis("off")
                ax.set_title(f"→ {tgt_short}\n(no data)", fontsize=6.5, color="gray")
                continue

            src       = pair_key.partition("_to_")[0]
            src_short = _SHORT.get(src, src)
            title     = f"{src_short} → {tgt_short}"
            _draw_venn(ax, entry, n, entry["total"], title)

        # row ylabel
        axes[row_idx][0].set_ylabel(
            f"src: {row_labels[row_idx]}", fontsize=6.5, color=INK_SEC,
        )

    return fig


def make_table(eval_ds, results_dir, aligner_suffix, out_path, datasets=("nq", "squad")):
    """Write a CSV with one row per (pair, align_ds, n)."""
    fields = ["pair", "eval_ds", "align_ds", "n",
              "A_only", "B_only", "C_only", "AB_only", "AC_only", "BC_only",
              "ABC", "D", "tgt_probe_error", "total"]

    rows = []
    ds_dir = os.path.join(results_dir, eval_ds)
    if not os.path.isdir(ds_dir):
        return

    pairs = sorted(p for p in os.listdir(ds_dir) if os.path.isdir(os.path.join(ds_dir, p)))

    for pair in pairs:
        pair_dir = os.path.join(ds_dir, pair)
        for align_ds in datasets:
            venn_path = os.path.join(pair_dir, f"venn_align_{align_ds}_{aligner_suffix}.json")
            if not os.path.exists(venn_path):
                continue
            venn_data = _load_json(venn_path)
            for n in venn_data["n_grid"]:
                entry = venn_data["by_n"].get(str(n))
                if entry is None:
                    continue
                rows.append({
                    "pair":     pair,
                    "eval_ds":  eval_ds,
                    "align_ds": align_ds,
                    "n":        n,
                    **{k: entry.get(k, "") for k in fields[4:]},
                })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved table: {out_path}")


def build_all(out_dir, aligner_suffix, n_values=(50, 100, 200, 400, 800, 1500),
              datasets=("nq", "squad"), verbose=True):
    """Write the 24 venn figures + 2 tables for ONE aligner+hyperparam combo.

    Outputs go to <out_dir>/summary_plots/venn_<aligner_suffix>/.
    Returns list of written paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = os.path.join(out_dir, "results")
    plots_dir = os.path.join(out_dir, "summary_plots", f"venn_{aligner_suffix}")
    os.makedirs(plots_dir, exist_ok=True)
    written = []

    # ── figures: eval_ds × align_ds × len(n_values) ──────────────────────────
    for eval_ds in datasets:
        for align_ds in datasets:
            for n in n_values:
                fig = plot_venn_fig(eval_ds, align_ds, n, results_dir, aligner_suffix)
                path = os.path.join(plots_dir,
                                    f"venn_{eval_ds}_align_{align_ds}_n{n}.png")
                fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
                plt.close(fig)
                written.append(path)
                if verbose:
                    print(f"Saved: {path}")

    # ── tables: one per eval_ds ───────────────────────────────────────────────
    for eval_ds in datasets:
        csv_path = os.path.join(plots_dir, f"table_{eval_ds}_{aligner_suffix}.csv")
        make_table(eval_ds, results_dir, aligner_suffix, csv_path, datasets=datasets)
        written.append(csv_path)

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",  required=True)
    parser.add_argument("--alpha",    type=float, default=1e3)
    parser.add_argument("--aligners", nargs="+", default=["ridge"])
    args = parser.parse_args()

    # derive aligner suffix (same logic as transfer2.py _aligner_tag)
    exp = int(round(np.log10(args.alpha)))
    build_all(args.out_dir, f"ridge_a1e{exp}")


if __name__ == "__main__":
    main()
