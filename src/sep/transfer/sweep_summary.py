"""Cross-hyperparameter (sweep) summary for SE probe transfer.

This module is deliberately SEPARATE from transfer2.py's Phase 4:
  * Phase 4 (transfer2.py summary) = one complete experiment, figures conditioned
    on the single aligner/hyperparam configuration that invocation was given.
  * This module = compares ACROSS the hyperparam variants already on disk and
    reports which one wins. It reads only Phase 3 outputs, computes nothing new,
    and is meant to run ONCE after a sweep has finished all its values.

Two sweep axes are supported, selected with --hparam:
  alpha   ridge alignment, files probe_grid_align_<ds>_ridge_a1e<exp>.json,
          curve curveB_ridge                                  (the original mode)
  lambda  E2-R* alignment, files probe_grid_align_<ds>_[...]e2_rstar_l<val>.json,
          curve curveB_e2_rstar

Which (eval_ds, align_ds) combos are reported follows --datasets: the cross product
of the datasets given. With `--datasets nq` that is just (nq, nq), i.e. the probe and
the alignment both trained on NQ and no cross-align curves.

Outputs (under <out_dir>/sweep_summary/, <hp> = alpha|lambda):
  hparam_<hp>_table_rank.{csv,md}   best value per pair: avg-rank across n, and at n=1500
  hparam_<hp>_table_per_n.{csv,md}  best value at EVERY n, unaggregated
  summary_{eval_ds}_probe_grid_<hp>_best.{pdf,png}
                                    main grid figure where same-align and cross-align
                                    each use their own best value (by avg rank)
  <hp>_curves_{eval_ds}_align_{align_ds}.{pdf,png}
                                    every value as its own curve, one figure per
                                    alignment dataset (same-align vs cross-align), so
                                    the ordering itself is visible rather than
                                    collapsed to a single winner

Usage:
    # ridge alpha sweep, squad + nq (original behaviour, unchanged defaults)
    python -m sep.transfer.sweep_summary --out-dir <cache_dir>

    # E2-R* lambda sweep, NQ only
    python -m sep.transfer.sweep_summary --out-dir <cache_dir> \
        --hparam lambda --datasets nq

    # optional: [--pair-list slurm/inputs/pair_list.txt] [--n-target 1500]
"""
import argparse
import csv
import os
import re

from sep.transfer.transfer2 import (
    _fmt_pow10,
    _load_json,
    _make_summary_fig,
    _model_tag,
    _pair_title,
    _repo_root,
    _structured_grid,
    _CROSS_SOURCES,
    _LLAMA_FAMILY,
    C_NATIVE,
    C_SRC_BASE,
    GRIDLINE,
    INK_PRI,
    INK_SEC,
    SURFACE,
)


# ============================================================================
# Sweep-axis specs
# ============================================================================
# Each spec says how to (a) recognise a Phase-3 grid file for this axis and pull the
# hyperparam value out of its name, (b) which curve in that file is the swept aligner,
# and (c) how to name things in tables/figures.
#
# The trailing "(?:_.+)?" in both regexes matters: Phase 3 names a grid file after
# EVERY aligner cache it found, so a run that has both a ridge and an e2_rstar cache
# on disk emits "probe_grid_align_nq_ridge_a1e3_e2_rstar_l1e3.json". Without the
# optional extra-tag group such files would be invisible to both axes.

class _Spec:
    def __init__(self, key, file_re_tmpl, parse, curve, aligner_label, sweep_label):
        self.key = key                      # "alpha" | "lambda"
        self.file_re_tmpl = file_re_tmpl    # format-string with {align_ds}
        self.parse = parse                  # regex group 1 -> float value
        self.curve = curve                  # curve key inside the grid json
        self.aligner_label = aligner_label  # e.g. "ridge alpha"
        self.sweep_label = sweep_label      # figure suptitle fragment

    def file_re(self, align_ds):
        return re.compile(self.file_re_tmpl.format(align_ds=re.escape(align_ds)))


SPECS = {
    "alpha": _Spec(
        key="alpha",
        file_re_tmpl=r'^probe_grid_align_{align_ds}_ridge_a1e(-?\d+)(?:_.+)?\.json$',
        parse=lambda s: 10.0 ** int(s),
        curve="curveB_ridge",
        aligner_label="ridge alpha",
        sweep_label="ridge alpha sweep",
    ),
    "lambda": _Spec(
        key="lambda",
        file_re_tmpl=r'^probe_grid_align_{align_ds}_(?:.+_)?e2_rstar_l([0-9.eE+-]+)\.json$',
        parse=float,
        curve="curveB_e2_rstar",
        aligner_label="E2-R* lambda",
        sweep_label="E2-R* lambda sweep",
    ),
}


def _combos(datasets):
    return [(e, a) for e in datasets for a in datasets]


# ============================================================================
# Reading per-value results
# ============================================================================

def collect_variants(pair_dir, align_ds, spec):
    """Return {hp_value: probe_grid dict} for every swept file for this align_ds."""
    if not os.path.isdir(pair_dir):
        return {}
    pattern = spec.file_re(align_ds)
    variants = {}
    for fname in os.listdir(pair_dir):
        m = pattern.match(fname)
        if m:
            variants[spec.parse(m.group(1))] = _load_json(os.path.join(pair_dir, fname))
    return variants


def _auc_by_n(variants, spec):
    """Reshape {hp: pg} into {n: {hp: auc}}, dropping None entries."""
    by_n = {}
    for hp, pg in variants.items():
        curve = pg.get(spec.curve)
        n_grid = pg.get("n_grid")
        if not curve or not n_grid:
            continue
        for n, v in zip(n_grid, curve):
            if v is not None:
                by_n.setdefault(n, {})[hp] = v
    return by_n


def rank_and_n_target(variants, spec, n_target=1500):
    """Compute the rank-aggregated best value and the best value at n=n_target.

    Ranking: at each n, values are ranked by AUROC (rank 1 = best). The winner is the
    value with the lowest average rank across all n, ties broken by more per-n wins
    then by smaller value. This is scale-free, so it is not distorted by the fact
    that different n have different AUROC ranges/variance.

    Returns {"best_rank": (hp, avg_rank, wins, n_compared) | None,
             "best_n":    (hp, auc) | None}  or None if nothing usable.
    """
    by_n = _auc_by_n(variants, spec)

    best_n = None
    if n_target in by_n and by_n[n_target]:
        best_n = max(by_n[n_target].items(), key=lambda kv: kv[1])

    if not by_n:
        return None if best_n is None else {"best_rank": None, "best_n": best_n}

    hps = sorted(variants.keys())
    rank_sum = {h: 0.0 for h in hps}
    rank_cnt = {h: 0 for h in hps}
    wins = {h: 0 for h in hps}
    for auc_by_hp in by_n.values():
        ranked = sorted(auc_by_hp.items(), key=lambda kv: kv[1], reverse=True)
        for i, (hp, _) in enumerate(ranked):
            rank_sum[hp] += i + 1
            rank_cnt[hp] += 1
        wins[ranked[0][0]] += 1

    avg_rank = {h: rank_sum[h] / rank_cnt[h] for h in hps if rank_cnt[h] > 0}
    if not avg_rank:
        return None if best_n is None else {"best_rank": None, "best_n": best_n}

    best_hp = min(avg_rank, key=lambda h: (avg_rank[h], -wins[h], h))
    return {
        "best_rank": (best_hp, avg_rank[best_hp], wins[best_hp], len(by_n)),
        "best_n": best_n,
    }


def discover_pair_names(pair_list_path):
    pairs = set()
    for line in open(pair_list_path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # 2-field format: "<src_model> <tgt_model>" (matches transfer2._resolve_pairs).
        # Legacy 4-field format held generation paths, so model names needed extracting.
        if len(parts) == 2:
            src, tgt = parts
        elif len(parts) == 4:
            src, tgt = _model_tag(parts[0]), _model_tag(parts[1])
        else:
            print(f"WARNING: bad pair line (expected 2 fields): {line}")
            continue
        pairs.add(f"{src}_to_{tgt}")
    return sorted(pairs)


# ============================================================================
# Tables
# ============================================================================

def _fmt_n_cell(entry):
    if entry is None:
        return "-"
    hp, auc = entry
    return f"{_fmt_pow10(hp)} ({auc:.3f})"


def _fmt_rank_cell(entry):
    if entry is None:
        return "-"
    hp, avg_rank, wins, n_total = entry
    return f"{_fmt_pow10(hp)} (rank={avg_rank:.2f}, {wins}/{n_total} wins)"


def _write_table(header, rows_of_vals, out_dir, stem):
    lines_md = ["| " + " | ".join(header) + " |",
                "|" + "|".join(["---"] * len(header)) + "|"]
    for vals in rows_of_vals:
        lines_md.append("| " + " | ".join(vals) + " |")
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    md_path = os.path.join(out_dir, f"{stem}.md")
    # csv.writer, not ",".join: rank cells embed a comma ("rank=1.00, 6/6 wins")
    # and would otherwise split into two columns, shifting every later column.
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_of_vals)
    with open(md_path, "w") as f:
        f.write("\n".join(lines_md) + "\n")
    return csv_path, md_path


def build_rank_table(results_dir, pair_list_path, spec, combos, n_target=1500):
    """rows: {"pair": name, "cells": {(eval_ds, align_ds, "rank"|"n"): entry_or_None}}"""
    rows = []
    for pair_name in discover_pair_names(pair_list_path):
        cells = {}
        for eval_ds, align_ds in combos:
            pair_dir = os.path.join(results_dir, eval_ds, pair_name)
            best = rank_and_n_target(collect_variants(pair_dir, align_ds, spec),
                                     spec, n_target=n_target)
            cells[(eval_ds, align_ds, "rank")] = best["best_rank"] if best else None
            cells[(eval_ds, align_ds, "n")] = best["best_n"] if best else None
        rows.append({"pair": pair_name, "cells": cells})
    return rows


def save_rank_table(rows, out_dir, spec, combos, n_target=1500):
    header = ["pair"]
    for eval_ds, align_ds in combos:
        header.append(f"eval={eval_ds}/align={align_ds} best_{spec.key}(avg_rank)")
        header.append(f"eval={eval_ds}/align={align_ds} best_{spec.key}(n={n_target})")
    vals_rows = []
    for row in rows:
        vals = [row["pair"]]
        for eval_ds, align_ds in combos:
            vals.append(_fmt_rank_cell(row["cells"][(eval_ds, align_ds, "rank")]))
            vals.append(_fmt_n_cell(row["cells"][(eval_ds, align_ds, "n")]))
        vals_rows.append(vals)
    return _write_table(header, vals_rows, out_dir, f"hparam_{spec.key}_table_rank")


def build_per_n_table(results_dir, pair_list_path, spec, combos):
    """Best value at every n, no aggregation.

    Returns (sorted n union, rows) with rows: {"pair", "cells": {(eval,align,n): entry}}
    """
    rows = []
    n_union = set()
    for pair_name in discover_pair_names(pair_list_path):
        cells = {}
        for eval_ds, align_ds in combos:
            pair_dir = os.path.join(results_dir, eval_ds, pair_name)
            variants = collect_variants(pair_dir, align_ds, spec)
            for n, auc_by_hp in _auc_by_n(variants, spec).items():
                n_union.add(n)
                cells[(eval_ds, align_ds, n)] = max(auc_by_hp.items(),
                                                    key=lambda kv: kv[1])
        rows.append({"pair": pair_name, "cells": cells})
    return sorted(n_union), rows


def save_per_n_table(n_grid_union, rows, out_dir, spec, combos):
    header = ["pair"]
    for eval_ds, align_ds in combos:
        for n in n_grid_union:
            header.append(f"eval={eval_ds}/align={align_ds} n={n}")
    vals_rows = []
    for row in rows:
        vals = [row["pair"]]
        for eval_ds, align_ds in combos:
            for n in n_grid_union:
                vals.append(_fmt_n_cell(row["cells"].get((eval_ds, align_ds, n))))
        vals_rows.append(vals)
    return _write_table(header, vals_rows, out_dir, f"hparam_{spec.key}_table_per_n")


# ============================================================================
# Best-value figure
# ============================================================================

def make_best_selector(spec, n_target=1500):
    """Selector for _make_summary_fig: per (pair, align_ds), use that combo's best value."""
    def _select(pair_dir, ds_tag):
        variants = collect_variants(pair_dir, ds_tag, spec)
        if not variants:
            return None
        best = rank_and_n_target(variants, spec, n_target=n_target)
        if not best or not best.get("best_rank"):
            return None
        hp = best["best_rank"][0]
        # Keep the "<name>=<value>" spelling: _make_summary_fig's _hparams_from_label
        # parses it back out to annotate each panel with the value that panel's curve
        # actually used, which differs per (pair, align_ds) under this selector.
        return f"{spec.aligner_label}={_fmt_pow10(hp)} (best)", variants[hp]
    return _select


def save_best_figs(results_dir, out_dir, spec, datasets, n_target=1500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selector = make_best_selector(spec, n_target=n_target)
    saved = []
    for eval_ds in datasets:
        cross = [d for d in datasets if d != eval_ds]
        fig = _make_summary_fig(
            eval_ds, "probe_grid", results_dir, selector,
            f"{spec.aligner_label}: best per pair "
            f"(avg rank across n; per-panel value in titles)",
            cross_datasets=cross,
        )
        stem = f"summary_{eval_ds}_probe_grid_{spec.key}_best"
        for ext in ("pdf", "png"):
            path = os.path.join(out_dir, f"{stem}.{ext}")
            fig.savefig(path, format=ext, dpi=200 if ext == "pdf" else 150,
                        bbox_inches="tight", facecolor=SURFACE)
            saved.append(path)
        plt.close(fig)
    return saved


# ============================================================================
# All-value figures (one per alignment dataset)
# ============================================================================

# Sequential ramp: weak regularization -> light, strong -> dark. A sequential scale is
# the right encoding here because the hyperparam is ordinal, so the reader should be
# able to see monotone trends without consulting the legend.
_HP_RAMP = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#04306b"]


def _hp_color(hp, all_hps):
    i = sorted(all_hps).index(hp)
    span = max(len(all_hps) - 1, 1)
    return _HP_RAMP[round(i / span * (len(_HP_RAMP) - 1))]


def make_hp_curves_fig(eval_ds, align_ds, results_dir, spec, n_target=1500):
    """Grid figure: for each pair, one curve per swept value for a single align_ds.

    Complements the best-value figure: that one answers "which value wins", this one
    answers "how much does the hyperparam matter, and is the response monotone".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    _SHORT = {
        "gemma-4-12b":  "Gemma-4-12B",
        "llama-3.1-8b": "Llama-3.1-8B",
        "mistral-nemo": "Mistral-Nemo",
        "phi-4":        "Phi-4",
        "qwen3-8b":     "Qwen3-8B",
        "llama-3.2-1b": "Llama-3.2-1B",
    }

    grid = _structured_grid(results_dir, eval_ds)
    nrows, ncols = len(grid), len(grid[0])
    row_labels = [_SHORT.get(s, s) for s in _CROSS_SOURCES] + [
        _SHORT.get(src, src) for src, _ in _LLAMA_FAMILY
    ]

    # Global value set so colors and legend entries mean the same thing in every panel.
    all_hps = set()
    for row in grid:
        for pair_key in row:
            if pair_key:
                all_hps |= set(collect_variants(
                    os.path.join(results_dir, eval_ds, pair_key), align_ds, spec))
    all_hps = sorted(all_hps)

    kind_lbl = "same-align" if align_ds == eval_ds else "cross-align"
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5),
                             facecolor=SURFACE, constrained_layout=True)
    fig.suptitle(
        f"SE probe transfer — {spec.sweep_label}  |  eval: {eval_ds.upper()}  |  "
        f"align: {align_ds.upper()} ({kind_lbl})  |  "
        f"one curve per value (light = weak reg. → dark = strong)  |  "
        f"best by avg rank marked ★",
        fontsize=11, color=INK_PRI,
    )

    for row_idx, row in enumerate(grid):
        for col_idx, pair_key in enumerate(row):
            ax = axes[row_idx][col_idx]
            if pair_key is None:
                ax.axis("off")
                continue

            pair_dir = os.path.join(results_dir, eval_ds, pair_key)
            _, kind = _pair_title(pair_key)
            tgt_short = _SHORT.get(pair_key.partition("_to_")[2],
                                   pair_key.partition("_to_")[2])

            ax.set_facecolor(SURFACE)
            for spine in ax.spines.values():
                spine.set_color("#b0b0b0")
                spine.set_linewidth(0.8)
            ax.grid(which="major", color=GRIDLINE, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)

            native_path = os.path.join(pair_dir, f"native_curves_{eval_ds}.json")
            if not os.path.exists(native_path):
                ax.set_title(f"→ {tgt_short}\n(missing)", fontsize=8, color="red")
                ax.axis("off")
                continue

            native_data = _load_json(native_path)
            g = native_data["n_grid"]
            ceiling = native_data.get("tgt_layer_auc", float("nan"))
            ax.plot(g, [v if v is not None else float("nan")
                        for v in native_data["curveA_native"]],
                    color=C_NATIVE, linewidth=1.5, marker="o", markersize=4,
                    markeredgewidth=0, label="A: native target", zorder=4)
            if native_data.get("curveA_src"):
                ax.plot(g, [v if v is not None else float("nan")
                            for v in native_data["curveA_src"]],
                        color=C_SRC_BASE, linewidth=1.5, marker="D", markersize=4,
                        markeredgewidth=0, label="A: source probe on src", zorder=4)

            variants = collect_variants(pair_dir, align_ds, spec)
            best = rank_and_n_target(variants, spec, n_target=n_target) if variants else None
            best_hp = best["best_rank"][0] if best and best.get("best_rank") else None

            for hp in sorted(variants):
                pg = variants[hp]
                curve = pg.get(spec.curve)
                if not curve:
                    continue
                is_best = hp == best_hp
                ax.plot(pg["n_grid"], curve, color=_hp_color(hp, all_hps),
                        linewidth=2.0 if is_best else 1.2,
                        marker="*" if is_best else "s",
                        markersize=9 if is_best else 3.5, markeredgewidth=0,
                        label=f"B: {spec.key}={_fmt_pow10(hp)}" + (" ★" if is_best else ""),
                        zorder=3)

            ax.set_xscale("log")
            ax.set_xlabel("source probe training examples", fontsize=7, color=INK_SEC)
            ax.set_ylabel("eval AUROC", fontsize=7, color=INK_SEC)
            ax.tick_params(colors=INK_SEC, labelsize=7, length=3)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_tick_params(which="minor", bottom=False)
            ax.set_xticks(g)
            ax.set_xticklabels([str(v) for v in g], fontsize=6.5, rotation=30)
            ax.set_ylim(0.45, 0.95)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.set_title(f"→ {tgt_short}\n[{kind}]  ceil={ceiling:.3f}",
                         fontsize=7.5, color=INK_PRI, pad=3)
            ax.legend(fontsize=5.5, frameon=True, framealpha=0.9,
                      edgecolor="#cccccc", loc="upper left")

        axes[row_idx][0].set_ylabel(f"src: {row_labels[row_idx]}\neval AUROC",
                                    fontsize=7, color=INK_SEC)
    return fig


def save_hp_curves_figs(results_dir, out_dir, spec, datasets, n_target=1500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = []
    for eval_ds in datasets:
        for align_ds in datasets:
            fig = make_hp_curves_fig(eval_ds, align_ds, results_dir, spec,
                                     n_target=n_target)
            stem = f"{spec.key}_curves_{eval_ds}_align_{align_ds}"
            for ext in ("pdf", "png"):
                path = os.path.join(out_dir, f"{stem}.{ext}")
                fig.savefig(path, format=ext, dpi=200 if ext == "pdf" else 150,
                            bbox_inches="tight", facecolor=SURFACE)
                saved.append(path)
            plt.close(fig)
    return saved


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Cross-hyperparam sweep summary (run once after a sweep finishes)")
    p.add_argument("--out-dir", required=True, help="root cache/results directory")
    p.add_argument("--hparam", default="alpha", choices=sorted(SPECS),
                   help="which sweep axis to summarise: alpha (ridge) or lambda (E2-R*)")
    p.add_argument("--datasets", nargs="+", default=["nq", "squad"],
                   help="datasets to report; combos are the cross product "
                        "(use a single dataset for a same-align-only sweep)")
    p.add_argument("--pair-list", default=None,
                   help="defaults to <repo>/slurm/inputs/pair_list.txt")
    p.add_argument("--n-target", type=int, default=1500,
                   help="n at which the 'best value at n' column is computed")
    args = p.parse_args()

    spec = SPECS[args.hparam]
    datasets = list(args.datasets)
    combos = _combos(datasets)

    results_dir = os.path.join(args.out_dir, "results")
    out_dir = os.path.join(args.out_dir, "sweep_summary")
    os.makedirs(out_dir, exist_ok=True)

    pair_list = args.pair_list or os.path.join(
        _repo_root(), "slurm", "inputs", "pair_list.txt")
    if not os.path.exists(pair_list):
        raise SystemExit(f"pair_list not found: {pair_list}")

    rank_rows = build_rank_table(results_dir, pair_list, spec, combos,
                                 n_target=args.n_target)
    if not any(v is not None for r in rank_rows for v in r["cells"].values()):
        raise SystemExit(
            f"No per-{spec.key} probe_grid files found under "
            f"{results_dir}. Run phases 2-3 for at least one {spec.key} first.")

    for path in save_rank_table(rank_rows, out_dir, spec, combos,
                                n_target=args.n_target):
        print(f"Saved: {path}")

    n_union, per_n_rows = build_per_n_table(results_dir, pair_list, spec, combos)
    for path in save_per_n_table(n_union, per_n_rows, out_dir, spec, combos):
        print(f"Saved: {path}")

    for path in save_best_figs(results_dir, out_dir, spec, datasets,
                               n_target=args.n_target):
        print(f"Saved: {path}")

    for path in save_hp_curves_figs(results_dir, out_dir, spec, datasets,
                                    n_target=args.n_target):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
