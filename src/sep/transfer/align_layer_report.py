"""align_layer_report.py — inspect the best-to-align layer choices after Phase 1.5.

Reads the json files written by

    python -m sep.transfer.transfer2 tgt_layer_cache --transfer-mode best-to-align ...

and tabulates, per (eval_ds, align_ds, pair), the target layer the LABEL-FREE
alignment-residual search picked against the layer the full 1500-label search
picked.  Cache-only (jsons + probe pkls, no hidden states), so it takes seconds
and is safe to run mid-pipeline -- the same role sub_layer_report.py plays for
best-to-best-sub.

`regret` is how much AUROC the label-free layer gives up relative to the
full-budget layer, both scored on the SAME full-budget curve (`layer_aucs`, the
450-row selection split).  `last_reg` is the same quantity for the last layer,
i.e. what best-to-last already costs for free.  best-to-align is only worth its
selection time if regret sits clearly below last_reg.

`resid` is the chosen layer's held-out relative reconstruction error and `rho` is
the Spearman correlation, across scored layers, between -resid and layer_aucs --
i.e. how well the label-free signal ranks layers on this pair.  rho is a
diagnostic computed with labels after the fact; nothing in the pipeline uses it.

Usage:
    python -m sep.transfer.align_layer_report --out-dir <cache_dir> [--datasets nq ...]
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np


def _spearman(a, b):
    """Rank correlation without a scipy dependency (ties broken arbitrarily)."""
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float)))
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float)))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def collect(out_dir, datasets=None):
    """Return one row dict per tgt_layers/<eval_ds>/<pair>/<align_ds>_b2a.json."""
    rows, missing = [], []
    pattern = os.path.join(out_dir, "tgt_layers", "*", "*", "*.json")
    for path in sorted(glob.glob(pattern)):
        sel = json.load(open(path))
        eval_ds = sel["eval_ds"]
        if datasets and eval_ds not in datasets:
            continue
        tgt_probe = os.path.join(out_dir, "probes", eval_ds,
                                 f"{sel['tgt_model']}.pkl")
        if not os.path.exists(tgt_probe):
            missing.append(os.path.relpath(path, out_dir))
            continue
        with open(tgt_probe, "rb") as f:
            tc = pickle.load(f)

        fa = tc["layer_aucs"]          # full-budget curve, one AUROC per layer
        Lf, Lb = tc["best_layer"], sel["tgt_best_layer"]
        resid = sel["layer_rel_resid"]
        scored = [L for L, r in enumerate(resid) if r == r]

        rows.append({
            "eval_ds": eval_ds, "align_ds": sel["align_ds"],
            "pair": f"{sel['src_model']}->{sel['tgt_model']}",
            "L_src": sel["src_layer"],
            "L_full": Lf, "L_b2a": Lb, "L_last": len(fa) - 1,
            "dL": Lb - Lf,
            "auc_full": fa[Lf], "auc_b2a": fa[Lb], "auc_last": fa[-1],
            "regret": fa[Lf] - fa[Lb],
            "last_reg": fa[Lf] - fa[-1],
            "resid": resid[Lb],
            "rho": _spearman([-resid[L] for L in scored], [fa[L] for L in scored]),
            "n_scored": len(scored),
            "sel_s": sel["select_s"],
        })
    return rows, missing


_COLS = [("eval_ds", "%-10s"), ("align_ds", "%-10s"), ("pair", "%-30s"),
         ("L_src", "%5s"), ("L_full", "%6s"), ("L_b2a", "%5s"), ("L_last", "%6s"),
         ("dL", "%4s"), ("auc_full", "%8s"), ("auc_b2a", "%7s"),
         ("auc_last", "%8s"), ("regret", "%7s"), ("last_reg", "%8s"),
         ("resid", "%6s"), ("rho", "%6s"), ("sel_s", "%7s")]


def render(rows):
    """Fixed-width table plus the aggregate that decides whether to continue."""
    if not rows:
        return "no tgt_layers/ selection json found yet\n"
    out = [" ".join(f % h for h, f in _COLS)]
    for r in rows:
        out.append(" ".join(
            f % (f"{r[k]:.3f}" if isinstance(r[k], float) else r[k])
            for k, f in _COLS))

    reg = np.array([r["regret"] for r in rows])
    lst = np.array([r["last_reg"] for r in rows])
    dl = np.abs([r["dL"] for r in rows])
    rho = np.array([r["rho"] for r in rows])
    rho = rho[~np.isnan(rho)]
    out += [
        "",
        f"selections                 : {len(rows)}",
        f"regret   (b2a vs full)     : mean {reg.mean():+.4f}  median "
        f"{np.median(reg):+.4f}  max {reg.max():+.4f}",
        f"last_reg (last vs full)    : mean {lst.mean():+.4f}  median "
        f"{np.median(lst):+.4f}  max {lst.max():+.4f}",
        f"b2a beats last layer       : {int((reg < lst).sum())}/{len(rows)}",
        f"regret <= 0.01             : {int((reg <= 0.01).sum())}/{len(rows)}",
        f"regret >  0.03             : {int((reg > 0.03).sum())}/{len(rows)}",
        f"|L_b2a - L_full|           : mean {dl.mean():.1f}  max {dl.max()}",
        f"rho(-resid, auc)           : mean {rho.mean():+.3f}  min {rho.min():+.3f}"
        if len(rho) else "rho(-resid, auc)           : n/a",
        f"selection time             : total {sum(r['sel_s'] for r in rows) / 3600:.2f} h",
        "",
        "regret/last_reg are scored on the full-budget layer_aucs curve (450 rows),",
        "not the 500-row Phase-3 eval set -- treat as a fast proxy, not the result.",
        "The b2a choice itself used ZERO target labels; auc/rho here are post-hoc.",
    ]
    return "\n".join(out) + "\n"


def write_csv(rows, path):
    keys = [k for k, _ in _COLS] + ["n_scored"]
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
                   help="output path (default <out-dir>/align_layer_report.csv)")
    args = p.parse_args()

    rows, missing = collect(args.out_dir, args.datasets)
    print(render(rows))
    if missing:
        print(f"no eval-dataset probe cache for {len(missing)} selection(s): "
              f"{', '.join(missing)}")
    if rows:
        csv = args.csv or os.path.join(args.out_dir, "align_layer_report.csv")
        write_csv(rows, csv)
        print(f"-> {csv}")


if __name__ == "__main__":
    main()
