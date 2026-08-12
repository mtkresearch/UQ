"""Area-proportional Venn diagrams of the transferred probe's error decomposition.

Reads the per-pair venn_<token><suffix>.json files written by transfer.py
--save-venn and draws, for each (budget axis, aligner), a grid of Venn diagrams
with one row per pair and one column per probe/map budget n.

Each eval example contributes three binary events:
  A = source probe disagrees with the source label   (source probe error)
  B = transfer flips the prediction vs the source probe
  C = source and target SE labels disagree           (label mismatch)
and the transferred probe's error on the target is exactly
  D = A xor B xor C = A_only + B_only + C_only + ABC,
so the four "odd-parity" regions are the error and the three "even-parity"
regions (AB, BC, AC) are cases where two effects cancel.

Circle radii and centre distances are solved from the actual counts
(matplotlib_venn), so region areas track the data rather than a fixed layout.
Note that no three-circle layout can match all seven regions at once: the three
set sizes and three pairwise intersections are matched exactly and the triple
region follows from that geometry, so very small ABC slivers are not to scale.
The printed counts are always exact.

CLI usage (minimal):
  python -m sep.transfer.plot_venn \
    --pairs-dir sep_scratch/transfer \
    --pair-names llama32-1b_to_llama31-8b llama2_to_mistral \
    --pair-labels "Llama3.2-1B->Llama3.1-8B" "Llama2-7B->Mistral-7B" \
    --token slt --alpha 1e4

If --pair-labels is omitted it defaults to --pair-names.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SURFACE = "white"
INK_PRI = "#1a1a1a"
INK_SEC = "#404040"

# venn3 patch id -> fill colour.  Odd-parity (= error) regions are saturated,
# even-parity (= cancellation) regions are pale, uninvolved regions are grey.
C_PATCH = {
    "100": "#d8d8d8",   # A only   — source probe error survives transfer
    "010": "#c00000",   # B only   — pure transfer harm
    "001": "#d8d8d8",   # C only   — pure label mismatch
    "110": "#1a3a8f",   # A∩B      — transfer fixes a source error
    "101": "#d8d8d8",   # A∩C      — source error cancelled by label mismatch
    "011": "#aac4e8",   # B∩C      — transfer adapts to label mismatch
    "111": "#f4a0a0",   # A∩B∩C    — harm plus mismatch
}

# region key <-> venn3 patch id
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


def result_suffix(alpha, seed=0, out_suffix=""):
    """Mirror the suffix transfer.py builds for its output filenames."""
    suffix = "" if alpha == 1e3 else f"_a{alpha:.0e}"
    if seed != 0:
        suffix += f"_s{seed}"
    return suffix + out_suffix


def load_venn(pairs_dir, pair_names, token, alpha, seed=0, out_suffix=""):
    suffix = result_suffix(alpha, seed, out_suffix)
    data = {}
    for name in pair_names:
        path = os.path.join(pairs_dir, name, f"venn_{token}{suffix}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"venn file not found: {path}\n"
                f"(re-run transfer.py for this pair with --save-venn)")
        with open(path) as f:
            data[name] = json.load(f)
    return data


def draw_venn(ax, entry, title):
    """Area-proportional three-circle Venn with count/percentage labels."""
    from matplotlib_venn import venn3, venn3_circles
    from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    counts = {k: int(entry.get(k, 0) or 0) for k in _VENN3_ORDER}
    subsets = tuple(counts[k] for k in _VENN3_ORDER)
    total = int(entry.get("total") or sum(subsets))

    if sum(subsets) == 0:
        ax.text(0.5, 0.5, "no errors,\nno mismatch", transform=ax.transAxes,
                fontsize=6, ha="center", va="center", color=INK_SEC)
        ax.set_title(title, fontsize=7, color=INK_PRI, pad=2)
        return

    layout = DefaultLayoutAlgorithm(normalize_to=1.0)
    v = venn3(subsets=subsets, set_labels=("A", "B", "C"), ax=ax,
              layout_algorithm=layout)
    venn3_circles(subsets=subsets, ax=ax, layout_algorithm=layout,
                  linewidth=1.0, color="#555555")

    for key in _VENN3_ORDER:
        pid   = _REGION_ID[key]
        patch = v.get_patch_by_id(pid)
        if patch is not None:
            patch.set_facecolor(C_PATCH[pid])
            patch.set_edgecolor("none")
            patch.set_alpha(1.0)

        label = v.get_label_by_id(pid)
        if label is None:          # zero-area region: venn3 draws no label
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

    # D = transferred-probe error on the target (the odd-parity union).
    d_val = int(entry.get("D", 0) or 0)
    d_pct = 100 * d_val / total if total > 0 else 0
    ax.annotate(f"D={d_val} ({d_pct:.0f}%)",
                xy=(1, 0), xycoords="axes fraction",
                fontsize=6, ha="right", va="bottom", color="#800000")

    ax.set_title(title, fontsize=7, color=INK_PRI, pad=2)


def plot_grid(data, pair_names, pair_labels, axis, aligner, n_values):
    """One figure: rows = pairs, cols = n."""
    nrows, ncols = len(pair_names), len(n_values)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.6, nrows * 2.6),
                             facecolor=SURFACE, squeeze=False,
                             constrained_layout=True)

    drew_any = False
    for r, (name, label) in enumerate(zip(pair_names, pair_labels)):
        by_n = data[name].get(axis, {}).get(aligner, {})
        for c, n in enumerate(n_values):
            ax = axes[r][c]
            entry = by_n.get(str(n))
            if entry is None:
                ax.axis("off")
                ax.set_title(f"n={n}\n(no data)", fontsize=7, color="gray")
                continue
            draw_venn(ax, entry, f"n={n}")
            drew_any = True
        axes[r][0].text(-0.08, 0.5, label, transform=axes[r][0].transAxes,
                        fontsize=8, color=INK_SEC, rotation=90,
                        ha="right", va="center")

    if not drew_any:
        plt.close(fig)
        return None

    # Aligner / budget axis / alpha / token stay in the filename rather than the
    # title, so the figure drops straight into a paper or slide.
    fig.suptitle("Transferred probe error decomposition",
                 fontsize=10, color=INK_PRI)
    return fig


def main():
    p = argparse.ArgumentParser(
        description="Area-proportional Venn plots of transfer error decomposition.")
    p.add_argument("--pairs-dir", required=True,
                   help="directory containing <pair-name>/venn_<token><suffix>.json")
    p.add_argument("--pair-names", nargs="+", required=True)
    p.add_argument("--pair-labels", nargs="+", default=None,
                   help="display labels (defaults to --pair-names)")
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--alpha", type=float, default=1e3,
                   help="must match the --alpha of the transfer.py run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--in-suffix", default="",
                   help="the --out-suffix used by the transfer.py run")
    p.add_argument("--axes", nargs="+", default=["map_budget", "probe_budget"],
                   choices=["map_budget", "probe_budget"])
    p.add_argument("--aligners", nargs="+", default=None,
                   help="aligners to plot (default: every one present in the files)")
    p.add_argument("--n-grid", type=int, nargs="+", default=None,
                   help="budget values to show (default: every one present)")
    p.add_argument("--out-dir", default=None,
                   help="where to write figures (default: --pairs-dir)")
    args = p.parse_args()

    pair_labels = args.pair_labels or args.pair_names
    if len(pair_labels) != len(args.pair_names):
        raise ValueError("--pair-labels must have the same length as --pair-names")

    data = load_venn(args.pairs_dir, args.pair_names, args.token,
                     args.alpha, args.seed, args.in_suffix)
    out_dir = args.out_dir or args.pairs_dir
    os.makedirs(out_dir, exist_ok=True)
    suffix = result_suffix(args.alpha, args.seed, args.in_suffix)

    for axis in args.axes:
        # aligners / n values present across the requested pairs
        aligners = args.aligners
        if aligners is None:
            found = {}
            for name in args.pair_names:
                for a in data[name].get(axis, {}):
                    found[a] = None
            aligners = list(found)
        if not aligners:
            print(f"no aligners found for axis {axis}, skipping")
            continue

        for aligner in aligners:
            n_values = args.n_grid
            if n_values is None:
                found_n = set()
                for name in args.pair_names:
                    found_n |= set(
                        int(k) for k in data[name].get(axis, {}).get(aligner, {}))
                n_values = sorted(found_n)
            if not n_values:
                print(f"no data for {axis}/{aligner}, skipping")
                continue

            fig = plot_grid(data, args.pair_names, pair_labels, axis, aligner,
                            n_values)
            if fig is None:
                print(f"no data for {axis}/{aligner}, skipping")
                continue
            path = os.path.join(
                out_dir, f"venn_{axis}_{aligner}_{args.token}{suffix}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
            plt.close(fig)
            print(f"saved {path}")


if __name__ == "__main__":
    main()
