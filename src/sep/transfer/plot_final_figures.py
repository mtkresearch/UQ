"""All paper figures and tables for the SE probe transfer experiments, in one run.

Produces, at ridge alpha=1e4:
  * 3 probe-grid figures (one per eval dataset), 6x4 = 21 model pairs each, 5 lines per
    panel: native source, native target, same-dataset alignment, two cross-dataset
    alignments.
  * The Venn error decomposition: 54 diagrams (3 eval x 3 align x 6 n), 3 CSVs, and
    9 LaTeX tables at n=1500 via plot_venn_grid + gen_venn_latex_tables.

The runs live in three separate out-dirs, so this assembles ONE merged results tree of
symlinks (nothing is copied or modified -- the source dirs are read-only inputs) and
hands that tree to transfer2._make_summary_fig.

EXACTLY WHICH FILES ARE READ
----------------------------
Source roots:
  P = /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled/results     (post-debug)
  L = /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2/results            (older run)
  T = /build_bak/UQ/UQ-transfer/sep_scratch/transfer_v2_trivia_qa/results

<PAIR> ranges over the 21 dirs found under <native root>/<eval_ds>/, e.g.
"gemma-4-12b_to_qwen3-8b". Per panel line, at alpha=1e4:

  Figure NQ  (native root P; pair list = ls P/nq/)
    grey  source probe on src   P/nq/<PAIR>/native_curves_nq.json              (curveA_src)
    black native target         P/nq/<PAIR>/native_curves_nq.json              (curveA_native)
    blue  align NQ        (same) P/nq/<PAIR>/probe_grid_align_nq_ridge_a1e4.json
    green align SQuAD    (cross) L/nq/<PAIR>/probe_grid_align_squad_ridge_a1e4.json
    red   align TriviaQA (cross) L/nq/<PAIR>/probe_grid_align_trivia_qa_ridge_a1e4.json
  plus, for the Venn outputs, the venn_align_<align_ds>_ridge_a1e4.json sitting beside
  each probe_grid file above -- same root per align_ds (P for the same-ds one, L/T for
  the cross ones), so figures and tables never disagree about their source.

  Figure SQuAD  (native root P; pair list = ls P/squad/)
    grey  source probe on src   P/squad/<PAIR>/native_curves_squad.json        (curveA_src)
    black native target         P/squad/<PAIR>/native_curves_squad.json        (curveA_native)
    blue  align SQuAD    (same) P/squad/<PAIR>/probe_grid_align_squad_ridge_a1e4.json
    green align NQ       (cross) L/squad/<PAIR>/probe_grid_align_nq_ridge_a1e4.json
    red   align TriviaQA (cross) L/squad/<PAIR>/probe_grid_align_trivia_qa_ridge_a1e4.json

  Figure TriviaQA  (native root T; pair list = ls T/trivia_qa/)
    grey  source probe on src   T/trivia_qa/<PAIR>/native_curves_trivia_qa.json  (curveA_src)
    black native target         T/trivia_qa/<PAIR>/native_curves_trivia_qa.json  (curveA_native)
    blue  align TriviaQA (same) T/trivia_qa/<PAIR>/probe_grid_align_trivia_qa_ridge_a1e4.json
    green align NQ       (cross) T/trivia_qa/<PAIR>/probe_grid_align_nq_ridge_a1e4.json
    red   align SQuAD    (cross) T/trivia_qa/<PAIR>/probe_grid_align_squad_ridge_a1e4.json

Caveat: for NQ/SQuAD the same-align curve is leak-free (P, alignment fit on `pool`)
while the two cross-align curves come from the older L run (alignment fit on the eval
rows). Solid vs dashed in those two figures are therefore not strictly comparable.

WHERE THEY GO
-------------
Each file above is symlinked to
  <out-dir>/_merged_ridge_a1e4/results/<eval_ds>/<PAIR>/<same basename>
and the outputs are written to
  <out-dir>/final_<eval_ds>_probe_grid_ridge_a1e4.{pdf,png}       probe-grid figures
  <out-dir>/venn_ridge_a1e4/venn_<eval_ds>_align_<ds>_n<n>.png    Venn diagrams
  <out-dir>/venn_ridge_a1e4/table_<eval_ds>_ridge_a1e4.csv        raw Venn counts
  <out-dir>/venn_ridge_a1e4/table_<eval_ds>_align_<ds>_n1500.tex  LaTeX tables

The merged tree under _merged_* is disposable: it is rebuilt from scratch on every run
(and the Venn files are copied out of it, not left behind as symlinks).

Run with --manifest to dump every single dst <- src line to
  <out-dir>/manifest_ridge_a1e4.txt
(and equivalently: `ls -l` / `readlink` inside the _merged_* tree shows the same thing).

Usage:
    python -m sep.transfer.plot_final_figures                 # everything, ./final_figures
    python -m sep.transfer.plot_final_figures --out-dir /tmp/figs --manifest
    python -m sep.transfer.plot_final_figures --skip-venn --datasets nq   # quick iteration
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

POOLED = "/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2_pooled/results"
LEGACY = "/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v2/results"
TRIVIA = "/build_bak/UQ/UQ-transfer/sep_scratch/transfer_v2_trivia_qa/results"

# eval_ds -> (dir holding native curves + same-align, [(cross_align_ds, dir), ...])
SOURCES = {
    "nq":        (POOLED, [("squad", LEGACY), ("trivia_qa", LEGACY)]),
    "squad":     (POOLED, [("nq", LEGACY), ("trivia_qa", LEGACY)]),
    "trivia_qa": (TRIVIA, [("nq", TRIVIA), ("squad", TRIVIA)]),
}


def _link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def build_merged_tree(work_dir, eval_ds, native_dir, cross_specs, ridge_tag):
    """Symlink the files needed for `eval_ds` into work_dir/results/<eval_ds>/<pair>/.

    Returns (pairs, missing, links) where links is the [(dst, src), ...] audit trail.
    """
    out_ds = os.path.join(work_dir, "results", eval_ds)
    src_ds = os.path.join(native_dir, eval_ds)
    pairs = sorted(p for p in os.listdir(src_ds)
                   if os.path.isdir(os.path.join(src_ds, p)))
    missing, links = [], []
    for pair in pairs:
        pair_out = os.path.join(out_ds, pair)
        os.makedirs(pair_out, exist_ok=True)

        wanted = [(native_dir, f"native_curves_{eval_ds}.json"),
                  (native_dir, f"probe_grid_align_{eval_ds}_{ridge_tag}.json"),
                  (native_dir, f"venn_align_{eval_ds}_{ridge_tag}.json")]
        for ads, d in cross_specs:
            wanted.append((d, f"probe_grid_align_{ads}_{ridge_tag}.json"))
            wanted.append((d, f"venn_align_{ads}_{ridge_tag}.json"))

        for base, fname in wanted:
            path = os.path.join(base, eval_ds, pair, fname)
            dst = os.path.join(pair_out, fname)
            if os.path.exists(path):
                _link(path, dst)
                links.append((dst, path))
            else:
                missing.append(f"{eval_ds}/{pair}/{fname}")
    return pairs, missing, links


def make_paper_fig(eval_ds, results_dir, select_variant, cross_datasets):
    """Paper version of transfer2._make_summary_fig.

    Same data and layout; publication labelling only -- no suptitle, no per-panel
    ceil/hyperparam annotation, panel title is just "target: <model>", and the legend
    reads "native target / native source / transferred alignment (<DATASET>)".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    grid = _structured_grid(results_dir, eval_ds)
    nrows, ncols = len(grid), len(grid[0])
    row_labels = [_SHORT.get(s, s) for s in _CROSS_SOURCES] + [
        _SHORT.get(src, src) for src, _ in _LLAMA_FAMILY
    ]

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5),
                             facecolor=SURFACE, constrained_layout=True)

    for row_idx, row in enumerate(grid):
        for col_idx, pair_key in enumerate(row):
            ax = axes[row_idx][col_idx]
            if pair_key is None:
                ax.axis("off")
                continue

            pair_dir = os.path.join(results_dir, eval_ds, pair_key)
            tgt = pair_key.partition("_to_")[2]
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

            native_data = _load_json(native_path)
            g = native_data["n_grid"]
            native = [v if v is not None else float("nan")
                      for v in native_data["curveA_native"]]
            ax.plot(g, native, color=C_NATIVE, linewidth=1.5, marker="o",
                    markersize=4, markeredgewidth=0, label="native target", zorder=3)

            src_curve = native_data.get("curveA_src")
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
                        ax.plot(g, curve, color=C_ALIGNER_SAME[i % len(C_ALIGNER_SAME)],
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
            ax.set_title(f"target: {tgt_short}", fontsize=8.5, color=INK_PRI, pad=3)
            ax.legend(fontsize=5.5, frameon=True, framealpha=0.9,
                      edgecolor="#cccccc", loc="upper left")

        axes[row_idx][0].set_ylabel(f"source: {row_labels[row_idx]}\nAUROC",
                                    fontsize=7, color=INK_SEC)

    return fig


def build_venn_outputs(work_dir, out_dir, eval_ds, cross_specs, ridge_tag, venn_n):
    """Venn figures + CSV (plot_venn_grid) and LaTeX tables (gen_venn_latex_tables).

    Reads the venn_align_*.json already symlinked into the merged tree, so the Venn
    outputs come from exactly the same files as the probe-grid figures. Everything
    lands in <out-dir>/venn_<tag>/.
    """
    from sep.transfer.plot_venn_grid import build_all as build_venn
    from sep.transfer.gen_venn_latex_tables import load_csv, make_table

    align_ds_list = [eval_ds] + [ds for ds, _ in cross_specs]
    build_venn(work_dir, ridge_tag, datasets=(eval_ds,),
               align_datasets=tuple(align_ds_list), verbose=False)

    # build_all writes under <work_dir>/summary_plots/venn_<tag>/; publish that next
    # to the figures as real files, so the merged symlink tree stays disposable.
    src_dir = os.path.join(work_dir, "summary_plots", f"venn_{ridge_tag}")
    pub_dir = os.path.join(out_dir, f"venn_{ridge_tag}")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="final_figures",
                    help="where the figures (and the merged symlink tree) are written")
    ap.add_argument("--alpha", type=float, default=1e4, help="ridge alpha (default 1e4)")
    ap.add_argument("--datasets", nargs="+", default=["nq", "squad", "trivia_qa"])
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--venn-n", type=int, default=1500,
                    help="n at which the LaTeX Venn tables are cut (default 1500)")
    ap.add_argument("--skip-venn", action="store_true",
                    help="only the probe-grid figures; no Venn figures/CSV/LaTeX")
    ap.add_argument("--manifest", action="store_true",
                    help="also write <out-dir>/manifest_<tag>.txt listing every "
                         "merged-tree path and the source file it points at")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ridge_tag = _aligner_tag("ridge", {"alpha": args.alpha})
    selector = make_run_variant_selector("probe_grid", {"ridge": {"alpha": args.alpha}})

    work_dir = os.path.join(args.out_dir, f"_merged_{ridge_tag}")
    # rebuild from scratch: stale symlinks from an earlier alpha would silently survive
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    all_links = []
    for eval_ds in args.datasets:
        native_dir, cross_specs = SOURCES[eval_ds]
        pairs, missing, links = build_merged_tree(work_dir, eval_ds, native_dir,
                                                 cross_specs, ridge_tag)
        all_links.extend(links)
        print(f"[{eval_ds}] {len(pairs)} pairs  "
              f"same-align+native: {native_dir}  "
              f"cross: {', '.join(f'{d}<-{p}' for d, p in cross_specs)}")
        for m in missing:
            print(f"  [warn] missing {m}")

        fig = make_paper_fig(
            eval_ds, os.path.join(work_dir, "results"), selector,
            cross_datasets=[ds for ds, _ in cross_specs],
        )
        for ext in args.formats:
            path = os.path.join(args.out_dir, f"final_{eval_ds}_probe_grid_{ridge_tag}.{ext}")
            fig.savefig(path, format=ext, dpi=200 if ext == "pdf" else 150,
                        bbox_inches="tight", facecolor=SURFACE)
            print(f"  Saved: {path}")
        plt.close(fig)

        if not args.skip_venn:
            build_venn_outputs(work_dir, args.out_dir, eval_ds, cross_specs,
                               ridge_tag, args.venn_n)

    if args.manifest:
        path = os.path.join(args.out_dir, f"manifest_{ridge_tag}.txt")
        with open(path, "w") as fh:
            fh.write(f"# every file read for these figures, as  <merged tree path>  <-  "
                     f"<source file>\n# {len(all_links)} files, ridge tag {ridge_tag}\n")
            for dst, src in all_links:
                fh.write(f"{dst}  <-  {src}\n")
        print(f"Manifest ({len(all_links)} files): {path}")


if __name__ == "__main__":
    main()
