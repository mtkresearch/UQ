"""sub_layer_report.py — inspect the best-to-best-sub layer choices after Phase 1.

Reads the probe caches written/updated by

    python -m sep.transfer.transfer2 probe_cache --transfer-mode best-to-best-sub ...

and tabulates, per (dataset, model), the layer the cheap 150-label search picked
against the layer the full 1500-label search picked.  Cache-only: no hidden
states are loaded, so this takes seconds and is safe to run mid-pipeline.

The decision-relevant column is `regret` -- how much AUROC the sub-budget layer
gives up relative to the full-budget layer, both scored on the SAME full-budget
curve (`layer_aucs`, the 450-row selection split).  `last_reg` is the same
quantity for the target's last layer, i.e. what best-to-last already costs.  If
regret is consistently well below last_reg, a 150-label layer search is worth
keeping; if it is comparable, best-to-best-sub buys nothing over best-to-last.

`oof` is the sub search's own out-of-fold score.  It is the MAX over layers of a
noisy estimate, so it is biased upward and is NOT a performance estimate -- it
only ranks layers.  Expect it to exceed the honest AUROC; a large gap is normal,
not a bug.

Usage:
    python -m sep.transfer.sub_layer_report --out-dir <cache_dir> [--datasets nq ...]
"""
import argparse
import glob
import os
import pickle

import numpy as np


def collect(out_dir, datasets=None):
    """Return one row dict per probe cache that carries a sub_best_layer."""
    rows, missing = [], []
    for path in sorted(glob.glob(os.path.join(out_dir, "probes", "*", "*.pkl"))):
        ds = os.path.basename(os.path.dirname(path))
        model = os.path.basename(path)[:-4]
        if datasets and ds not in datasets:
            continue
        with open(path, "rb") as f:
            c = pickle.load(f)
        if c.get("sub_best_layer") is None:
            missing.append(f"{ds}/{model}")
            continue

        fa = c["layer_aucs"]          # full-budget curve, one AUROC per layer
        Lf, Ls = c["best_layer"], c["sub_best_layer"]
        n_layers = len(fa)
        y = np.array(c["y"])
        sub = np.array(c["pool"])[:c["sub_n"]]
        pos = float(y[sub].mean())

        rows.append({
            "dataset": ds, "model": model, "n_layers": n_layers,
            "L_full": Lf, "L_sub": Ls, "L_last": n_layers - 1,
            "dL": Ls - Lf,
            "auc_full": fa[Lf], "auc_sub": fa[Ls], "auc_last": fa[-1],
            "regret": fa[Lf] - fa[Ls],
            "last_reg": fa[Lf] - fa[-1],
            "oof": c["sub_best_auc"],
            "sub_n": c["sub_n"], "folds": c["sub_folds"], "pos_rate": pos,
        })
    return rows, missing


_COLS = [("dataset", "%-10s"), ("model", "%-14s"), ("n_layers", "%8s"),
         ("L_full", "%6s"), ("L_sub", "%5s"), ("L_last", "%6s"), ("dL", "%4s"),
         ("auc_full", "%8s"), ("auc_sub", "%7s"), ("auc_last", "%8s"),
         ("regret", "%7s"), ("last_reg", "%8s"), ("oof", "%6s"),
         ("pos_rate", "%8s")]


def render(rows):
    """Fixed-width table plus the aggregate that decides whether to continue."""
    if not rows:
        return "no probe cache carries a sub_best_layer yet\n"
    out = [" ".join(f % h for h, f in _COLS)]
    for r in rows:
        cells = []
        for k, f in _COLS:
            v = r[k]
            cells.append(f % (f"{v:.3f}" if isinstance(v, float) else v))
        out.append(" ".join(cells))

    reg = np.array([r["regret"] for r in rows])
    lst = np.array([r["last_reg"] for r in rows])
    dl = np.abs([r["dL"] for r in rows])
    out += [
        "",
        f"pairs                      : {len(rows)}",
        f"regret   (sub vs full)     : mean {reg.mean():+.4f}  median "
        f"{np.median(reg):+.4f}  max {reg.max():+.4f}",
        f"last_reg (last vs full)    : mean {lst.mean():+.4f}  median "
        f"{np.median(lst):+.4f}  max {lst.max():+.4f}",
        f"sub beats last layer       : {int((reg < lst).sum())}/{len(rows)}",
        f"regret <= 0.01             : {int((reg <= 0.01).sum())}/{len(rows)}",
        f"regret >  0.03             : {int((reg > 0.03).sum())}/{len(rows)}",
        f"|L_sub - L_full|           : mean {dl.mean():.1f}  max {dl.max()}",
        "",
        "regret/last_reg are scored on the full-budget layer_aucs curve (450 rows),",
        "not the 500-row Phase-3 eval set -- treat as a fast proxy, not the result.",
        "oof is a max over layers and is biased upward; do not read it as accuracy.",
    ]
    return "\n".join(out) + "\n"


def write_csv(rows, path):
    keys = [k for k, _ in _COLS] + ["sub_n", "folds"]
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(
                f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])
                for k in keys) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--csv", default=None,
                   help="output path (default <out-dir>/sub_layer_report.csv)")
    args = p.parse_args()

    rows, missing = collect(args.out_dir, args.datasets)
    print(render(rows))
    if missing:
        print(f"no sub_best_layer in {len(missing)} cache(s): "
              f"{', '.join(missing)}")
    if rows:
        csv = args.csv or os.path.join(args.out_dir, "sub_layer_report.csv")
        write_csv(rows, csv)
        print(f"-> {csv}")


if __name__ == "__main__":
    main()
