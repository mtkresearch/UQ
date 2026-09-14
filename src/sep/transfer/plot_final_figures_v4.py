"""All paper figures and tables for the V4 SE probe transfer experiments, in one run.

V4 ran all four transfer modes end to end, so the outputs are organised BY MODE:

  <out-dir>/<mode>/            one folder per mode in --modes, holding only that
                               mode's results -- 3 probe-grid figures (21 pairs),
                               3 subset figures (6 pairs) and that mode's own Venn
                               figures + CSV + LaTeX tables
  <out-dir>/overlay_<m1>_<m2>/ one folder per --overlay / --overlay-set combo, with
                               those modes' curves drawn on top of each other in the
                               same 21-pair and 6-pair figures (1 + 4k curves per
                               panel; no Venn -- the Venn stats live per mode)

Each figure, at ridge alpha=1e4, carries native source, native target, same-dataset
alignment and the two cross-dataset alignments per panel, per mode drawn.

The four runs live in separate out-dirs, so each group assembles its OWN merged
results tree of symlinks (nothing is copied or modified -- the source dirs are
read-only inputs) and hands that tree to the figure builder.

TRANSFER MODES (--modes, default: all four)
-------------------------------------------
Every mode uses the SAME source probe (the source model's best layer) and the same
alignment fit, probe-size grid and 500-row eval.  They differ only in WHICH TARGET
LAYER the alignment maps into, and therefore in how many target SE labels that
choice costs:

  best   best-to-best        target's best layer, picked with all 1500 target labels
  btl    best-to-last        target's LAST layer            (0 labels, fixed layer)
  bbs    best-to-best-sub    target's best layer via 5-fold CV on 150 labels
  b2a    best-to-align       layer minimising the held-out alignment residual
                             (0 labels: no target SE label is read at all)

b2b / b2l are accepted as aliases of best / btl everywhere a mode is named.

--modes takes any subset and gives each of them its own folder.  --overlay takes a
subset to draw together in one extra folder: each mode there contributes 4 curves
per panel (its native target + its same-align + its two cross-align alignments) and
the grey native-source curve is mode-independent and drawn once, so k overlaid modes
give 1 + 4k curves per panel (`--overlay b2b b2l` -> 9).  A mode whose run has not
produced results yet is reported and skipped, not fatal.

  COLOUR    = which dataset the alignment was fitted on
              (blue = same as eval, green/red = the two cross datasets;
               black = native target, grey = native source)
  LINESTYLE = the mode:  solid = best, dash-dot = btl, dotted = bbs,
              long-dash = b2a  (the native-target curves use a second, dashier
              variant of the same idea so they stay distinguishable in black)

In a single-mode folder the cross curves stay dashed, exactly as in the published
figures; in an overlay folder they take their mode's linestyle, because there
linestyle has to mean "mode" and nothing else.

EXACTLY WHICH FILES ARE READ
----------------------------
Source roots, all under /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v4:
  transfer_v4_b2b/results   best   (files have no mode infix)
  transfer_v4_btl/results   btl
  transfer_v4_bbs/results   bbs
  transfer_v4_b2a/results   b2a    -- NOT PRODUCED YET: the run has only written
                                      probes/ and tgt_layers/, so this mode is
                                      skipped with a warning until it finishes

Unlike v2 -- where best-to-best had to be stitched together out of a pooled root, a
legacy root and a separate TriviaQA root -- each v4 run holds all 9
(eval_ds, align_ds) cells of its mode under one tree, so native, same-align and both
cross-align curves of a mode always come from the same place.

<PAIR> ranges over the 21 dirs found under <root>/<eval_ds>/, e.g.
"gemma-4-12b_to_qwen3-8b".  Per panel line, at alpha=1e4, with I = "" for best and
"<mode>_" otherwise:

  grey  source probe on src    <root>/<eval_ds>/<PAIR>/native_curves_I<eval_ds>.json  (curveA_src)
  black native target          <root>/<eval_ds>/<PAIR>/native_curves_I<eval_ds>.json  (curveA_native)
  blue  align <eval_ds> (same) <root>/<eval_ds>/<PAIR>/probe_grid_Ialign_<eval_ds>_ridge_a1e4.json
  green/red the two crosses    <root>/<eval_ds>/<PAIR>/probe_grid_Ialign_<ads>_ridge_a1e4.json

Beside each probe_grid file sits the venn_align_<align_ds>_ridge_a1e4.json used for
that mode's Venn outputs, so figures and tables never disagree about their source.
The "_<mode>_" infix is what lets several modes be symlinked into one merged pair dir
without collisions.  A panel simply omits the curves of a mode whose files are
absent, rather than blanking, so a partially-finished run can still be plotted.

VENN OUTPUTS
------------
Venn stats (venn_align_<ds>_<tag>.json) carry NO mode infix -- compute_venn.py drops
it when naming its output (it writes venn_align_<ds>_<tag>.json out of
predictions_<mode>_align_<ds>_<tag>.json) -- so two modes' Venn files are
indistinguishable by name and only their directory tells them apart.  That is
exactly why the Venn output is built per mode here, inside <out-dir>/<mode>/, and
never for an overlay folder.

WHERE THEY GO
-------------
Per group (a mode folder or an overlay folder), the files above are symlinked to
  <group>/_merged_ridge_a1e4/results/<eval_ds>/<PAIR>/<same basename>
and the outputs are written to
  <group>/final_<eval_ds>_probe_grid_ridge_a1e4_<tag>.{pdf,png}  21-pair figures
  <group>/final_<eval_ds>_subset6_ridge_a1e4_<tag>.{pdf,png}     6-pair subset figures
  <group>/venn_ridge_a1e4/venn_<eval_ds>_align_<ds>_n<n>.png     Venn diagrams
  <group>/venn_ridge_a1e4/table_<eval_ds>_ridge_a1e4.csv         raw Venn counts
  <group>/venn_ridge_a1e4/table_<eval_ds>_align_<ds>_n1500.tex   LaTeX tables
where <tag> is the group's modes joined by "_", e.g. "best", "btl", "best_btl".

The merged tree under _merged_* is disposable: it is rebuilt from scratch on every run
(and the Venn files are copied out of it, not left behind as symlinks).

Run with --manifest to dump every single dst <- src line to
  <out-dir>/manifest_ridge_a1e4.txt
(and equivalently: `ls -l` / `readlink` inside the _merged_* tree shows the same thing).

Usage:
    # all available modes, one folder each (figures + Venn), into transfer_v4_final/
    python -m sep.transfer.plot_final_figures_v4

    # per-mode folders plus one b2b+b2l overlay folder
    python -m sep.transfer.plot_final_figures_v4 --overlay b2b b2l

    # only overlays, two combos, no Venn
    python -m sep.transfer.plot_final_figures_v4 --skip-per-mode \
        --overlay-set b2b+b2l b2b+bbs

    python -m sep.transfer.plot_final_figures_v4 --skip-venn --datasets nq  # quick iter
"""
import argparse
import os
import shutil

from sep.transfer.transfer2 import (
    C_ALIGNER_CROSS,
    C_ALIGNER_SAME,
    C_NATIVE,
    C_SRC_BASE,
    GRIDLINE,
    INK_PRI,
    INK_SEC,
    SURFACE,
    _CROSS_SOURCES,
    _LLAMA_FAMILY,
    _aligner_tag,
    _load_json,
    _structured_grid,
    make_run_variant_selector,
)

_SHORT = {
    "gemma-4-12b":  "Gemma-4-12B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "mistral-nemo": "Mistral-Nemo",
    "phi-4":        "Phi-4",
    "qwen3-8b":     "Qwen3-8B",
    "llama-3.2-1b": "Llama-3.2-1B",
}

V4 = "/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v4"

# One self-contained run per transfer mode.  Unlike v2 -- where best-to-best was
# stitched together out of three roots (pooled / legacy / trivia_qa) -- each v4 run
# holds all 9 (eval_ds, align_ds) cells of its mode under a single results tree, so
# native, same-align and both cross-align curves always come from the same place.
B2B = f"{V4}/transfer_v4_b2b/results"
BTL = f"{V4}/transfer_v4_btl/results"
BBS = f"{V4}/transfer_v4_bbs/results"
B2A = f"{V4}/transfer_v4_b2a/results"     # not produced yet (probes/tgt_layers only)

# eval_ds -> (dir holding native curves + same-align, [(cross_align_ds, dir), ...])
SOURCES = {
    "nq":        (B2B, [("squad", B2B), ("trivia_qa", B2B)]),
    "squad":     (B2B, [("nq", B2B), ("trivia_qa", B2B)]),
    "trivia_qa": (B2B, [("nq", B2B), ("squad", B2B)]),
}

# ---- transfer modes -----------------------------------------------------------
# All modes share the source probe, the alignment fit and the eval; they differ
# only in which TARGET layer the alignment maps into (and thus in how many target
# SE labels picking it costs).  Colour is reserved for the alignment dataset, so a
# mode is identified by LINESTYLE + marker alone:
#     ls      alignment curves of this mode
#     nat_ls  its native-target curve (dashier variant, so two black curves of
#             neighbouring modes stay distinguishable)
#     label   what the legend appends, e.g. "Transfer (NQ, 150-label)"
# dir=None marks best-to-best, whose files have no mode infix and whose root is
# reached through SOURCES (all three eval datasets point at B2B in v4).
MODES = {
    "best": dict(dir=None, ls="-",                       marker="s",
                 nat_marker="o", label=None,
                 desc="best-to-best (1500 target labels pick the layer)"),
    "btl":  dict(dir=BTL,  ls=(0, (5, 2)),               marker="v",
                 nat_marker="^", label="final",
                 desc="best-to-last (target's final layer, 0 labels)"),
    "bbs":  dict(dir=BBS,  ls=(0, (4, 1.5, 1, 1.5)),     marker="P",
                 nat_marker="X", label="150-label",
                 desc="best-to-best-sub (150 target labels pick the layer)"),
    "b2a":  dict(dir=B2A,  ls=(0, (1.4, 1.4)),           marker="*",
                 nat_marker="d", label="label-free",
                 desc="best-to-align (alignment residual picks the layer, 0 labels)"),
}

# Style used whenever a figure draws ONE mode only (the per-mode folders).  There is
# no second mode to tell apart there, so linestyle carries no information and every
# mode is drawn in the canonical published look -- solid native + solid same-align +
# dashed cross-align -- which makes the four per-mode figures directly comparable
# instead of each having its own dash pattern.
SOLO = dict(ls="-", marker="s", nat_marker="o", cross_ls="--")
MODE_ORDER = list(MODES)          # drawing + naming order, independent of CLI order

# The run scripts and the paper call best-to-best "b2b" and best-to-last "b2l";
# accept both spellings on the command line and normalise to the keys above.
MODE_ALIAS = {m: m for m in MODE_ORDER}
MODE_ALIAS.update({"b2b": "best", "b2l": "btl", "b2b_sub": "bbs"})

OUT_DIR = "/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v4_final"


def mode_infix(mode):
    """Filename infix for a mode: "" for best-to-best, else "<mode>_"."""
    return "" if mode == "best" else f"{mode}_"


# ---- shared publication style -------------------------------------------------
# One place for every size, so the full 6x4 grid and the 2x3 subset figure stay
# visually identical.  Sizes are tuned for a figure printed at ~half page width
# per panel: readable without zooming in the PDF.
LW           = 2.2         # curve line width
MS           = 7           # marker size
FS_TITLE     = 13          # panel title ("target: X" / "source -> target")
FS_LABEL     = 12          # axis labels
FS_TICK      = 11          # y ticks + x tick labels
FS_LEGEND    = 9.5         # per-panel legend

_DS_NAME = {"nq": "NQ", "squad": "SQuAD", "trivia_qa": "TriviaQA"}

# Fixed fallback y windows, used only with --shared-ylim: one per eval dataset,
# from the observed best-to-best/btl min-max plus headroom for the legend.  The
# default is to autoscale each FIGURE to its own curves instead (see make_paper_fig).
YLIM = {
    "nq":        (0.45, 0.92),
    "squad":     (0.50, 0.95),
    "trivia_qa": (0.50, 1.00),
}
YLIM_DEFAULT = (0.45, 0.95)


def _legend_frac(n_rows):
    """Fraction of a panel's height to keep clear for an n_rows-tall legend box.

    Calibrated on the two cases that were tuned by eye: 3 rows (one mode, single
    column) -> 0.22, 5 rows (two modes, two columns of 5+4) -> 0.33.  n_rows=0 means
    the legend sits outside the panels, so no headroom is reserved.
    """
    return min(0.55, 0.055 * n_rows + 0.055) if n_rows else 0.03


# Short legend labels.  "Native"/"Transfer" instead of the old "native"/
# "transferred alignment", plus the mode tag ("final", "150-label", "label-free").
# The tag is dropped when a figure draws ONE mode: it would then be the same on every
# entry of every panel -- 105 repetitions of a constant on a 21-pair figure, which is
# what used to force the legend font down and made the four per-mode figures differ in
# legend width.  Which mode a per-mode figure shows is in its folder and file name.
def _lbl_transfer(ds, mode, tag=True):
    name = _DS_NAME.get(ds, ds.upper())
    t = MODES[mode]["label"] if tag else None
    return f"Transfer ({name}, {t})" if t else f"Transfer ({name})"


def _lbl_native(mode, tag=True):
    t = MODES[mode]["label"] if tag else None
    return f"Native target ({t} layer)" if t else "Native target"


def _link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def build_merged_tree(work_dir, eval_ds, native_dir, cross_specs, ridge_tag,
                      modes=("best",), mode_dirs=None, venn_mode="best"):
    """Symlink the files needed for `eval_ds` into work_dir/results/<eval_ds>/<pair>/.

    For each mode in `modes`, four files per pair are linked: its native curves and
    its three alignment grids (same-align + both cross-align).  The "_<mode>_"
    infix keeps them distinct inside the one pair dir; best-to-best has no infix
    and is the only mode whose cross grids come from a different root than its
    same-align one.

    The Venn stats are linked from `venn_mode` only -- compute_venn.py strips the
    mode from those filenames, so two modes' Venn files would collide.

    Returns (pairs, missing, links) where links is the [(dst, src), ...] audit trail.
    """
    mode_dirs = mode_dirs or {}
    out_ds = os.path.join(work_dir, "results", eval_ds)
    src_ds = os.path.join(native_dir, eval_ds)
    pairs = sorted(p for p in os.listdir(src_ds)
                   if os.path.isdir(os.path.join(src_ds, p)))
    missing, links = [], []

    def mode_root(mode, align_ds):
        """Root holding <mode>'s files for this (eval_ds, align_ds) cell."""
        if mode != "best":
            return mode_dirs.get(mode) or MODES[mode]["dir"]
        if align_ds == eval_ds:
            return native_dir
        return dict(cross_specs)[align_ds]

    align_ds_list = [eval_ds] + [ds for ds, _ in cross_specs]

    for pair in pairs:
        pair_out = os.path.join(out_ds, pair)
        os.makedirs(pair_out, exist_ok=True)

        wanted = []
        for mode in modes:
            infix = mode_infix(mode)
            wanted.append((mode_root(mode, eval_ds),
                           f"native_curves_{infix}{eval_ds}.json"))
            for ads in align_ds_list:
                wanted.append((mode_root(mode, ads),
                               f"probe_grid_{infix}align_{ads}_{ridge_tag}.json"))
        for ads in align_ds_list:
            wanted.append((mode_root(venn_mode, ads),
                           f"venn_align_{ads}_{ridge_tag}.json"))

        for base, fname in wanted:
            path = os.path.join(base, eval_ds, pair, fname)
            dst = os.path.join(pair_out, fname)
            if os.path.exists(path):
                _link(path, dst)
                links.append((dst, path))
            else:
                missing.append(f"{eval_ds}/{pair}/{fname}")
    return pairs, missing, links


def make_paper_fig(eval_ds, results_dir, selectors, cross_datasets,
                   modes=("best",), pair_grid=None, shared_ylim=False,
                   only_same_align=False):
    """Paper version of transfer2._make_summary_fig.

    Same data and layout; publication labelling only -- no suptitle, no per-panel
    ceil/hyperparam annotation, panel title is just "target: <model>", and short
    legend labels ("Native target" / "Transfer (NQ, 150-label)").

    selectors maps each mode in `modes` to its probe-grid variant selector (the
    files are probe_grid[_<mode>]_align_<ds>_<tag>.json).  Every mode draws its own
    native-target curve plus its same-align and cross-align curves, so colour means
    "alignment dataset" and linestyle means "mode"; the grey native-source curve is
    mode-independent and drawn once.  A mode whose files are missing from a panel is
    silently skipped, so partially-finished runs still plot.

    pair_grid overrides the full 6x4 source-major grid with an explicit list of rows
    of pair keys (used for the 2x3 paper subset).  Since such a grid has no single
    source per row, panels are then titled "source -> target" and the shared
    "source: <model>" row label is dropped; everything else -- data, colours,
    linestyles, sizes -- is unchanged, so the subset figure is a literal excerpt.

    Y RANGE: one range for the whole figure, derived from the curves of THIS
    figure's panels only, plus headroom sized to the tallest legend box.  So all
    panels stay directly comparable, while the 2x3 subset figure gets a tighter
    window than the 6x4 grid whenever its 6 pairs do not span the full spread.
    shared_ylim instead pins the figure to the fixed YLIM window of its eval
    dataset, which also makes the subset and full figures share one range.

    only_same_align drops the two cross-dataset alignment curves, leaving 1 + 2k
    curves per panel instead of 1 + 4k: the readable version of a 3- or 4-mode
    overlay, where the point is comparing the modes, not the alignment datasets.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    modes = [m for m in MODE_ORDER if m in modes]
    by_source = pair_grid is None
    grid = _structured_grid(results_dir, eval_ds) if by_source else pair_grid
    nrows, ncols = len(grid), max(len(r) for r in grid)
    row_labels = [_SHORT.get(s, s) for s in _CROSS_SOURCES] + [
        _SHORT.get(src, src) for src, _ in _LLAMA_FAMILY
    ]

    # Panels are a touch larger than the original, to give the bigger fonts and the
    # multi-mode legend room without overlapping the curves.
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.8, nrows * 4.2),
                             facecolor=SURFACE, squeeze=False,
                             constrained_layout=True)

    drawn = []          # axes that got curves; y range + legends are set in one go
    legend_hl = ([], [])  # handles/labels of the panel with the most entries

    for row_idx, row in enumerate(grid):
        for col_idx, pair_key in enumerate(row):
            ax = axes[row_idx][col_idx]
            if pair_key is None:
                ax.axis("off")
                continue

            pair_dir = os.path.join(results_dir, eval_ds, pair_key)
            src, _, tgt = pair_key.partition("_to_")
            tgt_short = _SHORT.get(tgt, tgt)
            panel_title = (f"target: {tgt_short}" if by_source else
                           f"source: {_SHORT.get(src, src)} → target: {tgt_short}")

            ax.set_facecolor(SURFACE)
            for spine in ax.spines.values():
                spine.set_color("#b0b0b0")
                spine.set_linewidth(0.8)
            ax.grid(which="major", color=GRIDLINE, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)

            # Native curves, one file per mode.  The first one that exists also
            # supplies the x grid and the mode-independent native-source curve.
            native = {}
            for mode in modes:
                path = os.path.join(
                    pair_dir, f"native_curves_{mode_infix(mode)}{eval_ds}.json")
                if os.path.exists(path):
                    native[mode] = _load_json(path)
            if not native:
                ax.set_title(f"{panel_title}\n(missing)", fontsize=FS_TITLE,
                             color="red")
                ax.axis("off")
                continue

            first = native[next(iter(native))]
            g = first["n_grid"]

            src_curve = first.get("curveA_src")
            if src_curve:
                # Source probe on the source model: identical in every mode (they
                # only differ in the TARGET layer), so this is drawn once.
                vals = [v if v is not None else float("nan") for v in src_curve]
                ax.plot(g, vals, color=C_SRC_BASE, linewidth=LW, marker="D",
                        markersize=MS, markeredgewidth=0, linestyle="-",
                        label="Native source", zorder=3)

            # With more than one mode overlaid, linestyle has to mean ONE thing --
            # the mode -- so the cross curves are drawn in their mode's linestyle
            # like the same-align ones (colour already says which dataset).  A
            # single-mode figure has nothing to disambiguate and uses SOLO instead.
            single_mode = len(modes) == 1

            for mode in modes:
                spec = SOLO if single_mode else MODES[mode]
                z = 3 if mode == "best" else 4

                if mode in native:
                    nd = native[mode]
                    # Its own n_grid: a partially-finished run can be shorter.
                    gm = nd["n_grid"]
                    vals = [v if v is not None else float("nan")
                            for v in nd["curveA_native"]]
                    ax.plot(gm, vals, color=C_NATIVE, linewidth=LW,
                            marker=spec["nat_marker"], markersize=MS,
                            markeredgewidth=0, linestyle=spec["ls"],
                            label=_lbl_native(mode, tag=not single_mode),
                            zorder=z)

                select_variant = selectors[mode]

                same_pick = select_variant(pair_dir, eval_ds)
                if same_pick is not None:
                    _, same = same_pick
                    xg = same["n_grid"]
                    for i, a in enumerate(same.get("aligners", ["ridge"])):
                        curve = same.get(f"curveB_{a}")
                        if curve:
                            ax.plot(xg, curve,
                                    color=C_ALIGNER_SAME[i % len(C_ALIGNER_SAME)],
                                    linewidth=LW, marker=spec["marker"],
                                    markersize=MS, markeredgewidth=0,
                                    linestyle=spec["ls"],
                                    label=_lbl_transfer(eval_ds, mode,
                                                        tag=not single_mode),
                                    zorder=z)

                cross_ls = SOLO["cross_ls"] if single_mode else spec["ls"]
                for ds_idx, ds in enumerate([] if only_same_align else cross_datasets):
                    cross_pick = select_variant(pair_dir, ds)
                    if cross_pick is None:
                        continue
                    _, cross = cross_pick
                    xg = cross["n_grid"]
                    for a in cross.get("aligners", ["ridge"]):
                        curve = cross.get(f"curveB_{a}")
                        if curve:
                            # Indexed by POSITION in cross_datasets, not by a
                            # running counter: one mode's file being absent must not
                            # shift another's colours, so "green = NQ" holds
                            # for every linestyle.
                            c = C_ALIGNER_CROSS[ds_idx % len(C_ALIGNER_CROSS)]
                            ax.plot(xg, curve, color=c, linewidth=LW,
                                    marker=spec["marker"], markersize=MS,
                                    markeredgewidth=0, linestyle=cross_ls,
                                    label=_lbl_transfer(ds, mode,
                                                        tag=not single_mode),
                                    zorder=z)

            ax.set_xscale("log")
            ax.set_xlabel("No. of source probe training samples",
                          fontsize=FS_LABEL, color=INK_SEC)
            ax.set_ylabel("AUROC", fontsize=FS_LABEL, color=INK_SEC)
            ax.tick_params(colors=INK_SEC, labelsize=FS_TICK, length=4)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_tick_params(which="minor", bottom=False)
            ax.set_xticks(g)
            ax.set_xticklabels([str(v) for v in g], fontsize=FS_TICK, rotation=30)
            drawn.append(ax)
            ax.set_title(panel_title, fontsize=FS_TITLE, color=INK_PRI, pad=4)

            # Legends are drawn in a second pass: whether they fit inside the
            # panels at all depends on the largest entry count on the figure.
            hl = ax.get_legend_handles_labels()
            if len(hl[0]) > len(legend_hl[0]):
                legend_hl = hl

        if by_source:
            axes[row_idx][0].set_ylabel(f"source: {row_labels[row_idx]}\nAUROC",
                                        fontsize=FS_LABEL, color=INK_SEC)

    # ---- legends ---------------------------------------------------------------
    # A single mode's 5 entries fit inside every panel, which is how the published
    # figures are labelled.  Any overlay (>=9 entries, and the labels carry a mode tag
    # on top) needs a box wider than the panel: it would either squeeze the axes to
    # nothing or, on the fixed canvas, be clipped at the figure edge.  So overlays get
    # ONE legend below all panels instead, and the panels keep their full height.
    n_entries = len(legend_hl[0])
    inside = n_entries <= 6
    if inside:
        ncol = 1 if n_entries <= 4 else 2
        legend_rows = -(-n_entries // ncol)

        def put_legend(ax, fontsize):
            handles, labels = ax.get_legend_handles_labels()
            leg = ax.legend(handles, labels, fontsize=fontsize, frameon=True,
                            framealpha=0.9, edgecolor="#cccccc", loc="upper left",
                            ncol=ncol, handlelength=1.6, handletextpad=0.5,
                            columnspacing=1.0, labelspacing=0.35, borderpad=0.4)
            # constrained_layout counts an in-axes legend as part of the axes'
            # decorations, so a mode with longer labels ("Transfer (NQ, 150-label)")
            # got a NARROWER data box than "best" -- the panels of the four per-mode
            # figures then did not line up.  Taking the legend out of the layout
            # fixes the box geometry to the text-independent one.
            leg.set_in_layout(False)
            return leg

        # ... but out of the layout, an over-wide box now spills into the next panel
        # instead of shrinking its own, so shrink the FONT until it fits the box.
        # Measured on the widest-labelled panel; the same size is then used for all,
        # so every panel of a figure is labelled identically.
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        widest = max(drawn, key=lambda ax: put_legend(ax, FS_LEGEND)
                     .get_window_extent(rend).width)
        fs = FS_LEGEND
        while fs > 6.0:
            leg = put_legend(widest, fs)
            if (leg.get_window_extent(rend).width
                    <= 0.98 * widest.get_window_extent(rend).width):
                break
            fs -= 0.5
        for ax in drawn:
            put_legend(ax, fs)
    else:
        legend_rows = 0   # nothing overlaps the curves, so no headroom needed
        # ABOVE the panels, not below: on the 6x4 grid a bottom legend sits ~3800px
        # past 24 panels and is missed entirely.  Columns are chosen to keep it at
        # most 4 rows tall, so 9 entries (2 modes) give 3 rows and 17 (4 modes) give 4.
        ncol = min(5, max(4, -(-n_entries // 4)))
        fig.legend(*legend_hl, fontsize=FS_LEGEND + 1.5, frameon=True,
                   framealpha=0.9, edgecolor="#cccccc",
                   loc="outside upper center", ncol=ncol,
                   handlelength=2.0, handletextpad=0.6, columnspacing=1.4,
                   labelspacing=0.4, borderpad=0.5)

    # ---- one y range for the whole figure --------------------------------------
    # Read back off the curves that were actually drawn, so the window fits THIS
    # figure: the 6-pair subset gets a tighter one than the 21-pair grid when its
    # pairs sit in a narrower band, while panels within either figure stay
    # comparable.  The top _legend_frac(rows) of the range is kept clear so the
    # legend box lands above the curves rather than on them.
    if shared_ylim:
        lo_hi = YLIM.get(eval_ds, YLIM_DEFAULT)
        for ax in drawn:
            ax.set_ylim(*lo_hi)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    else:
        ys = [v for ax in drawn for ln in ax.lines for v in ln.get_ydata()
              if v == v]   # drop the NaNs standing in for missing points
        if ys:
            lo, hi = min(ys), max(ys)
            span = max(hi - lo, 0.06)
            lo -= 0.04 * span
            top = lo + (hi + 0.04 * span - lo) / (1 - _legend_frac(legend_rows))
            for ax in drawn:
                ax.set_ylim(lo, top)
        for ax in drawn:
            ax.yaxis.set_major_locator(
                ticker.MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]))

    return fig


def build_venn_outputs(work_dir, out_dir, eval_ds, cross_specs, ridge_tag, venn_n,
                       pub_suffix=""):
    """Venn figures + CSV (plot_venn_grid) and LaTeX tables (gen_venn_latex_tables).

    Reads the venn_align_*.json already symlinked into the merged tree, so the Venn
    outputs come from exactly the same run as the curves of --venn-mode. Everything
    lands in <out-dir>/venn_<tag><pub_suffix>/ -- pub_suffix is "_<mode>" for a
    non-"best" venn mode, since those Venn stats carry no mode infix of their own
    and would otherwise silently overwrite the published best-to-best outputs.
    """
    from sep.transfer.plot_venn_grid import build_all as build_venn
    from sep.transfer.gen_venn_latex_tables import load_csv, make_table

    align_ds_list = [eval_ds] + [ds for ds, _ in cross_specs]
    build_venn(work_dir, ridge_tag, datasets=(eval_ds,),
               align_datasets=tuple(align_ds_list), verbose=False)

    # build_all writes under <work_dir>/summary_plots/venn_<tag>/; publish that next
    # to the figures as real files, so the merged symlink tree stays disposable.
    src_dir = os.path.join(work_dir, "summary_plots", f"venn_{ridge_tag}")
    pub_dir = os.path.join(out_dir, f"venn_{ridge_tag}{pub_suffix}")
    os.makedirs(pub_dir, exist_ok=True)
    for fname in sorted(os.listdir(src_dir)):
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(pub_dir, fname))

    csv_path = os.path.join(pub_dir, f"table_{eval_ds}_{ridge_tag}.csv")
    rows = load_csv(csv_path)
    for align_ds in align_ds_list:
        tex = make_table(rows, eval_ds, align_ds, venn_n)
        tex_path = os.path.join(pub_dir,
                                f"table_{eval_ds}_align_{align_ds}_n{venn_n}.tex")
        with open(tex_path, "w") as fh:
            fh.write(tex + "\n")
    print(f"  Venn: {len(align_ds_list)} LaTeX tables (n={venn_n}) + "
          f"figures + CSV in {pub_dir}")


def _group_tag(modes):
    """Directory + figure-name tag for a group of modes, e.g. "best" / "best_btl"."""
    return "_".join(m for m in MODE_ORDER if m in modes)


def render_group(out_dir, modes, args, ridge_tag, selectors, mode_dirs, do_venn):
    """Draw one output folder: the 21-pair and 6-pair figures for `modes` overlaid.

    A group is self-contained: its own merged symlink tree, its own figures, and --
    for a single-mode group -- its own Venn outputs, taken from that same mode.  So
    <out-dir>/best/ holds only best-to-best results, <out-dir>/best_btl/ holds the
    overlay of the two, and neither can overwrite the other.

    Returns the [(dst, src), ...] link audit trail.
    """
    import matplotlib.pyplot as plt

    tag = _group_tag(modes)
    work_dir = os.path.join(out_dir, f"_merged_{ridge_tag}")
    # rebuild from scratch: stale symlinks from an earlier alpha would silently survive
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Venn stats carry no mode infix, so one mode has to supply them; inside a
    # single-mode folder that is unambiguously the folder's own mode.
    venn_mode = modes[0] if len(modes) == 1 else (args.venn_mode or "best")

    all_links = []
    for eval_ds in args.datasets:
        native_dir, cross_specs = SOURCES[eval_ds]
        pairs, missing, links = build_merged_tree(work_dir, eval_ds, native_dir,
                                                 cross_specs, ridge_tag,
                                                 modes=modes, mode_dirs=mode_dirs,
                                                 venn_mode=venn_mode)
        all_links.extend(links)
        print(f"[{tag}/{eval_ds}] {len(pairs)} pairs  modes: {' '.join(modes)}  "
              f"root: {native_dir}  venn: {venn_mode if do_venn else 'skipped'}")
        for m in missing:
            print(f"  [warn] missing {m}")

        for stem, pair_grid in _figure_specs(args, eval_ds, ridge_tag, f"_{tag}"):
            fig = make_paper_fig(
                eval_ds, os.path.join(work_dir, "results"), selectors,
                cross_datasets=[ds for ds, _ in cross_specs],
                modes=modes, pair_grid=pair_grid,
                shared_ylim=args.shared_ylim,
                only_same_align=args.only_same_align,
            )
            for ext in args.formats:
                path = os.path.join(out_dir, f"{stem}.{ext}")
                # NO bbox_inches="tight": it crops to the ink, and an in-panel legend
                # box may stick out past the axes, so a mode with longer legend text
                # ("Transfer (NQ, 150-label)") produced a wider canvas than "best" and
                # its panels looked smaller at equal display width.  constrained_layout
                # already handles the padding, so every figure is exactly
                # ncols*4.8 x nrows*4.2 in and all modes' panels line up.
                fig.savefig(path, format=ext, dpi=200 if ext == "pdf" else 150,
                            facecolor=SURFACE)
                print(f"  Saved: {path}")
            plt.close(fig)

        if do_venn:
            build_venn_outputs(work_dir, out_dir, eval_ds, cross_specs,
                               ridge_tag, args.venn_n)
    return all_links


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help=f"root for all output folders (default {OUT_DIR})")
    ap.add_argument("--alpha", type=float, default=1e4, help="ridge alpha (default 1e4)")
    ap.add_argument("--datasets", nargs="+", default=["nq", "squad", "trivia_qa"])
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--modes", nargs="+", default=list(MODE_ORDER),
                    metavar="MODE",
                    help="transfer modes to give their own <out-dir>/<mode>/ folder "
                         "(figures + Venn): "
                         + "; ".join(f"{m} = {MODES[m]['desc']}" for m in MODE_ORDER)
                         + ".  b2b/b2l are accepted as aliases of best/btl.  "
                           "Default: all four.")
    ap.add_argument("--overlay", nargs="+", default=None, metavar="MODE",
                    help="modes to overlay in ONE extra folder "
                         "<out-dir>/overlay_<m1>_<m2>.../: e.g. `--overlay b2b b2l` "
                         "draws 1+4+4 = 9 curves per panel.  Default: no overlay "
                         "folder.  Repeatable via --overlay-set for several combos.")
    ap.add_argument("--overlay-set", nargs="+", default=[], metavar="M1+M2",
                    help="several overlay folders at once, each a +-joined combo, "
                         "e.g. --overlay-set b2b+b2l b2b+bbs+b2a")
    ap.add_argument("--skip-per-mode", action="store_true",
                    help="only the overlay folder(s); no per-mode folders")
    ap.add_argument("--only-same-align", action="store_true",
                    help="drop the two cross-dataset alignment curves, leaving "
                         "1 + 2k curves per panel instead of 1 + 4k -- the readable "
                         "version of a 3- or 4-mode overlay.  Figures get a "
                         "_samealign stem, so they do not overwrite the full ones.")
    ap.add_argument("--venn-n", type=int, default=1500,
                    help="n at which the LaTeX Venn tables are cut (default 1500)")
    ap.add_argument("--venn-mode", choices=MODE_ORDER, default=None,
                    help="unused for per-mode folders (each builds its own mode's "
                         "Venn); only relevant if you ever ask an overlay folder for "
                         "Venn output, where exactly one mode can supply it because "
                         "compute_venn.py strips the mode from those filenames.")
    ap.add_argument("--skip-venn", action="store_true",
                    help="only the probe-grid figures; no Venn figures/CSV/LaTeX")
    ap.add_argument("--mode-dir", nargs="+", default=[], metavar="MODE=DIR",
                    help="override a mode's results root, e.g. bbs=/path/results")
    ap.add_argument("--shared-ylim", action="store_true",
                    help="use one fixed y range per eval dataset for all panels "
                         "instead of autoscaling each figure to its own curves")
    ap.add_argument("--skip-subset", action="store_true",
                    help="do not draw the extra 2x3 subset figure of the 6 "
                         "representative pairs from paper_subset_fig.SUBSET_GRID")
    ap.add_argument("--manifest", action="store_true",
                    help="also write <out-dir>/manifest_<tag>.txt listing every "
                         "merged-tree path and the source file it points at")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def resolve(names, what):
        out = []
        for n in names:
            m = MODE_ALIAS.get(n)
            if m is None:
                ap.error(f"{what}: unknown mode {n!r}; "
                         f"expected one of {sorted(MODE_ALIAS)}")
            if m not in out:
                out.append(m)
        return [m for m in MODE_ORDER if m in out]

    mode_dirs = {}
    for spec in args.mode_dir:
        mode, _, path = spec.partition("=")
        mode = MODE_ALIAS.get(mode, mode)
        if mode not in MODES or not path:
            ap.error(f"--mode-dir expects MODE=DIR with MODE in {MODE_ORDER}: {spec}")
        mode_dirs[mode] = path

    def root_of(mode):
        return mode_dirs.get(mode) or (MODES[mode]["dir"] if mode != "best" else B2B)

    per_mode = [] if args.skip_per_mode else resolve(args.modes, "--modes")

    overlays = []
    if args.overlay:
        overlays.append(resolve(args.overlay, "--overlay"))
    for combo in args.overlay_set:
        overlays.append(resolve([c for c in combo.split("+") if c], "--overlay-set"))

    # A mode whose run has not finished (b2a at the time of writing) has no results
    # tree at all; drop it with a warning rather than aborting the whole run, so the
    # other three modes' folders still get built.
    def available(modes, what):
        keep = [m for m in modes if os.path.isdir(root_of(m))]
        for m in modes:
            if m not in keep:
                print(f"[skip] {what}: mode {m} has no results yet ({root_of(m)})")
        return keep

    per_mode = available(per_mode, "--modes")
    overlays = [ms for ms in (available(ms, "--overlay") for ms in overlays) if ms]
    groups = [(os.path.join(args.out_dir, m), [m], not args.skip_venn)
              for m in per_mode]
    groups += [(os.path.join(args.out_dir, "overlay_" + _group_tag(ms)), ms, False)
               for ms in overlays]
    if not groups:
        ap.error("nothing to draw: no mode with results was selected")

    ridge_tag = _aligner_tag("ridge", {"alpha": args.alpha})
    # One selector per mode: its grids are probe_grid[_<mode>]_align_<ds>_<tag>.
    selectors = {
        m: make_run_variant_selector(f"probe_grid_{m}" if m != "best" else "probe_grid",
                                    {"ridge": {"alpha": args.alpha}})
        for m in MODE_ORDER
    }

    os.makedirs(args.out_dir, exist_ok=True)
    all_links = []
    for group_dir, group_modes, do_venn in groups:
        print(f"=== {group_dir}  ({' + '.join(group_modes)})")
        all_links += render_group(group_dir, group_modes, args, ridge_tag,
                                 selectors, mode_dirs, do_venn)

    if args.manifest:
        path = os.path.join(args.out_dir, f"manifest_{ridge_tag}.txt")
        with open(path, "w") as fh:
            fh.write(f"# every file read for these figures, as  <merged tree path>  <-  "
                     f"<source file>\n# {len(all_links)} files, ridge tag {ridge_tag}, "
                     f"groups {'; '.join(_group_tag(g[1]) for g in groups)}\n")
            for dst, src in all_links:
                fh.write(f"{dst}  <-  {src}\n")
        print(f"Manifest ({len(all_links)} files): {path}")


def _figure_specs(args, eval_ds, ridge_tag, stem_suffix):
    """(output stem, pair_grid) for each figure to draw for this eval dataset.

    The 6-pair subset figure is built from the same merged tree and the same call
    as the full grid, so it can never disagree with the matching panel of it.
    """
    # --only-same-align gets its own stem, so the reduced figure never overwrites the
    # full one of the same group.
    if args.only_same_align:
        stem_suffix += "_samealign"
    specs = [(f"final_{eval_ds}_probe_grid_{ridge_tag}{stem_suffix}", None)]
    if not args.skip_subset:
        from sep.transfer.paper_subset_fig import SUBSET_GRID
        specs.append((f"final_{eval_ds}_subset6_{ridge_tag}{stem_suffix}",
                      SUBSET_GRID))
    return specs


if __name__ == "__main__":
    main()
