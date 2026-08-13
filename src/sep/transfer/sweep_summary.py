"""Cross-hyperparameter (sweep) summary for SE probe transfer.

This module is deliberately SEPARATE from transfer2.py's Phase 4:
  * Phase 4 (transfer2.py summary) = one complete experiment, figures conditioned
    on the single aligner/hyperparam configuration that invocation was given.
  * This module = compares ACROSS the ridge-alpha variants already on disk and
    reports which alpha wins. It reads only Phase 3 outputs, computes nothing new,
    and is meant to run ONCE after a sweep has finished all its alphas.

Outputs (under <out_dir>/sweep_summary/):
  hparam_alpha_table_rank.{csv,md}   best alpha per pair: avg-rank across n, and at n=1500
  hparam_alpha_table_per_n.{csv,md}  best alpha at EVERY n, unaggregated
  summary_{eval_ds}_probe_grid_alpha_best.{pdf,png}
                                     main grid figure where same-align and cross-align
                                     each use their own best alpha (by avg rank)
  alpha_curves_{eval_ds}_align_{align_ds}.{pdf,png}
                                     every alpha as its own curve, one figure per
                                     alignment dataset (same-align vs cross-align), so
                                     the alpha ordering itself is visible rather than
                                     collapsed to a single winner

Usage:
    python -m sep.transfer.sweep_summary --out-dir <cache_dir> \
        [--pair-list slurm/inputs/pair_list.txt] [--n-target 1500]
"""
import argparse
import os
import re

from sep.transfer.transfer2 import (
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

_ALPHA_FILE_RE_TMPL = r'^probe_grid_align_{align_ds}_ridge_a1e(-?\d+)\.json$'

_COMBOS = [(e, a) for e in ("nq", "squad") for a in ("nq", "squad")]


# ============================================================================
# Reading per-alpha results
# ============================================================================

def collect_alpha_variants(pair_dir, align_ds):
    """Return {alpha_exp: probe_grid dict} for every ridge-alpha file for this align_ds."""
    if not os.path.isdir(pair_dir):
        return {}
    pattern = re.compile(_ALPHA_FILE_RE_TMPL.format(align_ds=re.escape(align_ds)))
    variants = {}
    for fname in os.listdir(pair_dir):
        m = pattern.match(fname)
        if m:
            variants[int(m.group(1))] = _load_json(os.path.join(pair_dir, fname))
    return variants


def _auc_by_n(variants):
    """Reshape {alpha_exp: pg} into {n: {alpha_exp: auc}}, dropping None entries."""
    by_n = {}
    for exp, pg in variants.items():
        curve = pg.get("curveB_ridge")
        n_grid = pg.get("n_grid")
        if not curve or not n_grid:
            continue
        for n, v in zip(n_grid, curve):
            if v is not None:
                by_n.setdefault(n, {})[exp] = v
    return by_n


def rank_and_n_target(variants, n_target=1500):
    """Compute the rank-aggregated best alpha and the best alpha at n=n_target.

    Ranking: at each n, alphas are ranked by AUROC (rank 1 = best). The winner is the
    alpha with the lowest average rank across all n, ties broken by more per-n wins
    then by smaller exponent. This is scale-free, so it is not distorted by the fact
    that different n have different AUROC ranges/variance.

    Returns {"best_rank": (exp, avg_rank, wins, n_compared) | None,
             "best_n":    (exp, auc) | None}  or None if nothing usable.
    """
    by_n = _auc_by_n(variants)

    best_n = None
    if n_target in by_n and by_n[n_target]:
        best_n = max(by_n[n_target].items(), key=lambda kv: kv[1])

    if not by_n:
        return None if best_n is None else {"best_rank": None, "best_n": best_n}

    exps = sorted(variants.keys())
    rank_sum = {e: 0.0 for e in exps}
    rank_cnt = {e: 0 for e in exps}
    wins = {e: 0 for e in exps}
    for auc_by_exp in by_n.values():
        ranked = sorted(auc_by_exp.items(), key=lambda kv: kv[1], reverse=True)
        for i, (exp, _) in enumerate(ranked):
            rank_sum[exp] += i + 1
            rank_cnt[exp] += 1
        wins[ranked[0][0]] += 1

    avg_rank = {e: rank_sum[e] / rank_cnt[e] for e in exps if rank_cnt[e] > 0}
    if not avg_rank:
        return None if best_n is None else {"best_rank": None, "best_n": best_n}

    best_exp = min(avg_rank, key=lambda e: (avg_rank[e], -wins[e], e))
    return {
        "best_rank": (best_exp, avg_rank[best_exp], wins[best_exp], len(by_n)),
        "best_n": best_n,
    }


def discover_pair_names(pair_list_path):
    pairs = set()
    for line in open(pair_list_path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        src_gen, tgt_gen, _, _ = parts
        pairs.add(f"{_model_tag(src_gen)}_to_{_model_tag(tgt_gen)}")
    return sorted(pairs)


# ============================================================================
# Tables
# ============================================================================

def _fmt_n_cell(entry):
    if entry is None:
        return "-"
    exp, auc = entry
    return f"1e{exp} ({auc:.3f})"


def _fmt_rank_cell(entry):
    if entry is None:
        return "-"
    exp, avg_rank, wins, n_total = entry
    return f"1e{exp} (rank={avg_rank:.2f}, {wins}/{n_total} wins)"


def _write_table(header, rows_of_vals, out_dir, stem):
    lines_csv = [",".join(header)]
    lines_md = ["| " + " | ".join(header) + " |",
                "|" + "|".join(["---"] * len(header)) + "|"]
    for vals in rows_of_vals:
        lines_csv.append(",".join(vals))
        lines_md.append("| " + " | ".join(vals) + " |")
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    md_path = os.path.join(out_dir, f"{stem}.md")
    with open(csv_path, "w") as f:
        f.write("\n".join(lines_csv) + "\n")
    with open(md_path, "w") as f:
        f.write("\n".join(lines_md) + "\n")
    return csv_path, md_path


def build_rank_table(results_dir, pair_list_path, n_target=1500):
    """rows: {"pair": name, "cells": {(eval_ds, align_ds, "rank"|"n"): entry_or_None}}"""
    rows = []
    for pair_name in discover_pair_names(pair_list_path):
        cells = {}
        for eval_ds, align_ds in _COMBOS:
            pair_dir = os.path.join(results_dir, eval_ds, pair_name)
            best = rank_and_n_target(collect_alpha_variants(pair_dir, align_ds),
                                     n_target=n_target)
            cells[(eval_ds, align_ds, "rank")] = best["best_rank"] if best else None
            cells[(eval_ds, align_ds, "n")] = best["best_n"] if best else None
        rows.append({"pair": pair_name, "cells": cells})
    return rows


def save_rank_table(rows, out_dir, n_target=1500):
    header = ["pair"]
    for eval_ds, align_ds in _COMBOS:
        header.append(f"eval={eval_ds}/align={align_ds} best_alpha(avg_rank)")
        header.append(f"eval={eval_ds}/align={align_ds} best_alpha(n={n_target})")
    vals_rows = []
    for row in rows:
        vals = [row["pair"]]
        for eval_ds, align_ds in _COMBOS:
            vals.append(_fmt_rank_cell(row["cells"][(eval_ds, align_ds, "rank")]))
            vals.append(_fmt_n_cell(row["cells"][(eval_ds, align_ds, "n")]))
        vals_rows.append(vals)
    return _write_table(header, vals_rows, out_dir, "hparam_alpha_table_rank")


def build_per_n_table(results_dir, pair_list_path):
    """Best alpha at every n, no aggregation.

    Returns (sorted n union, rows) with rows: {"pair", "cells": {(eval,align,n): entry}}
    """
    rows = []
    n_union = set()
    for pair_name in discover_pair_names(pair_list_path):
        cells = {}
        for eval_ds, align_ds in _COMBOS:
            pair_dir = os.path.join(results_dir, eval_ds, pair_name)
            for n, auc_by_exp in _auc_by_n(collect_alpha_variants(pair_dir, align_ds)).items():
                n_union.add(n)
                cells[(eval_ds, align_ds, n)] = max(auc_by_exp.items(),
                                                    key=lambda kv: kv[1])
        rows.append({"pair": pair_name, "cells": cells})
    return sorted(n_union), rows


def save_per_n_table(n_grid_union, rows, out_dir):
    header = ["pair"]
    for eval_ds, align_ds in _COMBOS:
        for n in n_grid_union:
            header.append(f"eval={eval_ds}/align={align_ds} n={n}")
    vals_rows = []
    for row in rows:
        vals = [row["pair"]]
        for eval_ds, align_ds in _COMBOS:
            for n in n_grid_union:
                vals.append(_fmt_n_cell(row["cells"].get((eval_ds, align_ds, n))))
        vals_rows.append(vals)
    return _write_table(header, vals_rows, out_dir, "hparam_alpha_table_per_n")


# ============================================================================
# Best-alpha figure
# ============================================================================

def make_best_alpha_selector(n_target=1500):
    """Selector for _make_summary_fig: per (pair, align_ds), use that combo's best alpha."""
    def _select(pair_dir, ds_tag):
        variants = collect_alpha_variants(pair_dir, ds_tag)
        if not variants:
            return None
        best = rank_and_n_target(variants, n_target=n_target)
        if not best or not best.get("best_rank"):
            return None
        exp = best["best_rank"][0]
        # Keep "alpha=1e<exp>" in the label: _make_summary_fig parses it back out to
        # annotate each panel with the alpha that panel's curve actually used, which
        # differs per (pair, align_ds) under this selector.
        return f"ridge alpha=1e{exp} (best)", variants[exp]
    return _select


def save_best_alpha_figs(results_dir, out_dir, n_target=1500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selector = make_best_alpha_selector(n_target=n_target)
    saved = []
    for eval_ds in ("nq", "squad"):
        fig = _make_summary_fig(
            eval_ds, "probe_grid", results_dir, selector,
            "ridge alpha: best per pair (avg rank across n; per-panel alpha in titles)",
        )
        stem = f"summary_{eval_ds}_probe_grid_alpha_best"
        for ext in ("pdf", "png"):
            path = os.path.join(out_dir, f"{stem}.{ext}")
            fig.savefig(path, format=ext, dpi=200 if ext == "pdf" else 150,
                        bbox_inches="tight", facecolor=SURFACE)
            saved.append(path)
        plt.close(fig)
    return saved


# ============================================================================
# All-alpha figures (one per alignment dataset)
# ============================================================================

# Sequential ramp: low alpha (weak regularization) -> light, high alpha -> dark.
# A sequential scale is the right encoding here because alpha is ordinal, so the
# reader should be able to see monotone trends without consulting the legend.
_ALPHA_RAMP = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#04306b"]


def _alpha_color(exp, all_exps):
    i = sorted(all_exps).index(exp)
    span = max(len(all_exps) - 1, 1)
    return _ALPHA_RAMP[round(i / span * (len(_ALPHA_RAMP) - 1))]


def make_alpha_curves_fig(eval_ds, align_ds, results_dir, n_target=1500):
    """Grid figure: for each pair, one curve per ridge alpha for a single align_ds.

    Complements the best-alpha figure: that one answers "which alpha wins", this one
    answers "how much does alpha matter, and is the response monotone".
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

    # Global alpha set so colors and legend entries mean the same thing in every panel.
    all_exps = set()
    for row in grid:
        for pair_key in row:
            if pair_key:
                all_exps |= set(collect_alpha_variants(
                    os.path.join(results_dir, eval_ds, pair_key), align_ds))
    all_exps = sorted(all_exps)

    kind_lbl = "same-align" if align_ds == eval_ds else "cross-align"
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5),
                             facecolor=SURFACE, constrained_layout=True)
    fig.suptitle(
        f"SE probe transfer — ridge alpha sweep  |  eval: {eval_ds.upper()}  |  "
        f"align: {align_ds.upper()} ({kind_lbl})  |  "
        f"one curve per alpha (light = weak reg. → dark = strong)  |  "
        f"best alpha by avg rank marked ★",
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

            variants = collect_alpha_variants(pair_dir, align_ds)
            best = rank_and_n_target(variants, n_target=n_target) if variants else None
            best_exp = best["best_rank"][0] if best and best.get("best_rank") else None

            for exp in sorted(variants):
                pg = variants[exp]
                curve = pg.get("curveB_ridge")
                if not curve:
                    continue
                is_best = exp == best_exp
                ax.plot(pg["n_grid"], curve, color=_alpha_color(exp, all_exps),
                        linewidth=2.0 if is_best else 1.2,
                        marker="*" if is_best else "s",
                        markersize=9 if is_best else 3.5, markeredgewidth=0,
                        label=f"B: alpha=1e{exp}" + (" ★" if is_best else ""),
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


def save_alpha_curves_figs(results_dir, out_dir, n_target=1500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = []
    for eval_ds in ("nq", "squad"):
        for align_ds in ("nq", "squad"):
            fig = make_alpha_curves_fig(eval_ds, align_ds, results_dir,
                                        n_target=n_target)
            stem = f"alpha_curves_{eval_ds}_align_{align_ds}"
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
        description="Cross-alpha sweep summary (run once after a sweep finishes)")
    p.add_argument("--out-dir", required=True, help="root cache/results directory")
    p.add_argument("--pair-list", default=None,
                   help="defaults to <repo>/slurm/inputs/pair_list.txt")
    p.add_argument("--n-target", type=int, default=1500,
                   help="n at which the 'best alpha at n' column is computed")
    args = p.parse_args()

    results_dir = os.path.join(args.out_dir, "results")
    out_dir = os.path.join(args.out_dir, "sweep_summary")
    os.makedirs(out_dir, exist_ok=True)

    pair_list = args.pair_list or os.path.join(
        _repo_root(), "slurm", "inputs", "pair_list.txt")
    if not os.path.exists(pair_list):
        raise SystemExit(f"pair_list not found: {pair_list}")

    rank_rows = build_rank_table(results_dir, pair_list, n_target=args.n_target)
    if not any(v is not None for r in rank_rows for v in r["cells"].values()):
        raise SystemExit(
            "No per-alpha probe_grid files found under "
            f"{results_dir}. Run phases 2-3 for at least one alpha first.")

    for path in save_rank_table(rank_rows, out_dir, n_target=args.n_target):
        print(f"Saved: {path}")

    n_union, per_n_rows = build_per_n_table(results_dir, pair_list)
    for path in save_per_n_table(n_union, per_n_rows, out_dir):
        print(f"Saved: {path}")

    for path in save_best_alpha_figs(results_dir, out_dir, n_target=args.n_target):
        print(f"Saved: {path}")

    for path in save_alpha_curves_figs(results_dir, out_dir, n_target=args.n_target):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
