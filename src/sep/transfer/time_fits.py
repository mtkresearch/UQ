"""Re-time the two CPU-bound fitting stages under one controlled environment.

Why this exists
---------------
The timings recorded inside `transfer2.py` runs are not comparable across datasets:

  * SQuAD/NQ ran under the repo conda env (numpy 1.26 + MKL); TriviaQA ran under
    /build_bak/mtk53686/semantic-entropy-probes/.venv (numpy 2.5 + OpenBLAS).  On the
    same host the same ridge grid costs 26.4 s vs 43.6 s -- a 1.65x env-only gap.
  * Neither run pinned BLAS thread counts, so the shared 64-core host's load added
    another few-fold swing.
  * The recorded numbers summed the whole n-grid (6 ridge fits, 7 probe fits) rather
    than a single fit at n=1500.  transfer2.py now records the single fit, but the
    already-collected caches predate that fix.

This script reads the existing probe/alignment caches, so nothing is refit for real:
it only re-measures `probe training` (one LogisticRegression at n) and `alignment
fitting` (one ridge solve at n), for every dataset, in this process, back to back.

Two phases, because the inputs live in 1.1 GB pickles:

  extract  once per (dataset, model): read validation_generations.pkl, pull the
           best layer recorded in the probe cache, z-score with the cached mu/sd,
           and write a small .npz (2000 x d float64, ~65 MB).
  time     from those .npz only: 1 probe fit per (dataset, target) and 1 ridge fit
           per (dataset, pair).  Pure CPU, no big IO.

The z-scoring reproduces transfer2.py exactly: the probe phase uses
`_zscore(X[pool], X)` and the align phase re-uses the same cached mu/sd, so a single
cached Z serves both.  Row order and target row reordering follow `_align_ids`, and
the ridge subset is `arange(N)[:n]` -- raw id order, as in phase_align_cache.

Output
------
`--out-json` uses the same schema as transfer_v2/timing.json:

    {"probe_cache": {model: {dataset: {"probe_fit_s": ...}}},
     "align_cache": {"<src>_to_<tgt>": {"<ds>_probe": {"<ds>_align": {"ridge_a1e4": ...}}}},
     "_meta": {env, thread pinning, load average, ...}}

so it drops straight into `make_timing_table --transfer-v2-timing`.  Only the one
alpha needed for the table is measured (default 1e4), not the whole sweep, and no
`evaluate` block is produced (eval-time probe inference is excluded from the table).

Usage
-----
    python -m sep.transfer.time_fits extract \
        --cache-root squad=/proj/.../transfer_v2 \
        --cache-root nq=/proj/.../transfer_v2 \
        --cache-root trivia_qa=/build_bak/.../transfer_v2_trivia_qa \
        --work-dir /proj/.../transfer_v2/fit_timing

    python -m sep.transfer.time_fits time \
        --work-dir /proj/.../transfer_v2/fit_timing \
        --pair-list slurm/inputs/pair_list.txt \
        --datasets squad nq trivia_qa --n 1500 --alpha 1e4 --repeats 3 \
        --out-json /proj/.../transfer_v2/timing_fits.json
"""
import argparse
import json
import os
import pickle
import platform
import statistics
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from sep.transfer.transfer2 import (
    _align_ids,
    _aligner_tag,
    _best_split,
    _binarize,
    _fit_ridge,
    _load_entropy,
    _load_hidden,
)

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


# ---------------------------------------------------------------------------
# phase: extract
# ---------------------------------------------------------------------------

def _npz_path(work_dir, ds, model):
    return os.path.join(work_dir, "z", f"{ds}__{model}.npz")


def phase_extract(args):
    cache_roots = _parse_cache_roots(args.cache_root)
    os.makedirs(os.path.join(args.work_dir, "z"), exist_ok=True)

    for ds, root in cache_roots.items():
        probe_dir = os.path.join(root, "probes", ds)
        if not os.path.isdir(probe_dir):
            raise SystemExit(f"no probe cache dir: {probe_dir}")
        models = sorted(f[:-4] for f in os.listdir(probe_dir) if f.endswith(".pkl"))
        print(f"\n=== {ds}: {len(models)} models from {probe_dir}")
        for model in models:
            out = _npz_path(args.work_dir, ds, model)
            if os.path.exists(out) and not args.force:
                print(f"  skip {ds}/{model} (exists)")
                continue
            with open(os.path.join(probe_dir, f"{model}.pkl"), "rb") as f:
                cache = pickle.load(f)
            gen_path = cache["gen_path"]
            print(f"  {ds}/{model}: layer {cache['best_layer']}  <- {gen_path}")
            t0 = time.time()
            H, ids = _load_hidden(gen_path, args.token)
            X = H[cache["best_layer"]].astype(np.float64)
            del H
            mu = np.asarray(cache["mu"], dtype=np.float64)
            sd = np.asarray(cache["sd"], dtype=np.float64)
            Z = (X - mu) / sd
            ent = _load_entropy(gen_path)
            y = _binarize(ent, _best_split(ent))
            np.savez(
                out,
                Z=Z,
                y=y,
                pool=np.asarray(cache["pool"], dtype=int),
                ids=np.asarray(ids, dtype=object),
                best_layer=cache["best_layer"],
            )
            print(f"    -> {out}  Z{Z.shape}  ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# phase: time
# ---------------------------------------------------------------------------

def _load_z(work_dir, ds, model, cache):
    key = (ds, model)
    if key not in cache:
        p = _npz_path(work_dir, ds, model)
        if not os.path.exists(p):
            return None
        with np.load(p, allow_pickle=True) as d:
            cache[key] = {
                "Z": d["Z"], "y": d["y"], "pool": d["pool"],
                "ids": list(d["ids"]),
            }
    return cache[key]


def _bench(fn, repeats):
    """Run fn `repeats` times; return (min, median, all) in seconds.

    min is the headline number: it is the least contaminated by CPU contention from
    other users of the shared host.
    """
    fn()  # warm up BLAS threads / allocator, not counted
    ts = []
    for _ in range(repeats):
        t0 = time.time()
        fn()
        ts.append(time.time() - t0)
    return min(ts), statistics.median(ts), ts


def _parse_pairs(pair_list_path):
    pairs = []
    with open(pair_list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def phase_time(args):
    zc = {}
    pairs = _parse_pairs(args.pair_list)
    tag = _aligner_tag("ridge", {"alpha": args.alpha})  # e.g. "ridge_a1e4"

    # Same schema as transfer_v2/timing.json, so this file can be passed straight to
    # make_timing_table as --transfer-v2-timing.  Only the alpha actually needed for
    # the table is measured (one aligner tag), not the whole sweep.
    result = {
        "probe_cache": {},
        "align_cache": {},
        "_meta": {
            "produced_by": "sep.transfer.time_fits",
            "n": args.n,
            "alpha": args.alpha,
            "aligner_tag": tag,
            "repeats": args.repeats,
            "reported_statistic": "min over repeats",
            "host": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "blas": _blas_name(),
            "threads": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
            "loadavg_at_start": os.getloadavg(),
            "note": ("probe_fit_s: ONE LogisticRegression(max_iter=1000) on n pool "
                     f"samples.  {tag}: ONE _fit_ridge solve on the first n rows in "
                     "raw id order.  Single fits, not n-grid sums.  layer_search_s is "
                     "not re-measured here (it is not in the paper table)."),
        },
    }

    for ds in args.datasets:
        # ---- probe training: one fit per model ----
        print(f"\n{'=' * 70}\n{ds}: probe training (one fit at n={args.n})\n{'=' * 70}")
        models = sorted({m for pair in pairs for m in pair})
        for model in models:
            z = _load_z(args.work_dir, ds, model, zc)
            if z is None:
                print(f"  skip {model} (no extracted Z)")
                continue
            pool = z["pool"][:args.n]
            X, y = z["Z"][pool], z["y"][pool]
            if len(np.unique(y)) < 2:
                print(f"  skip {model} (single class)")
                continue
            lo, med, all_ts = _bench(
                lambda: LogisticRegression(max_iter=1000).fit(X, y), args.repeats)
            result["probe_cache"].setdefault(model, {})[ds] = {
                "probe_fit_s": round(lo, 3),
                "probe_fit_n": int(len(pool)),
                "probe_fit_d": int(X.shape[1]),
                "probe_fit_median_s": round(med, 3),
                "probe_fit_all_s": [round(t, 3) for t in all_ts],
            }
            print(f"  {model:14s} d={X.shape[1]:5d}  min {lo:7.3f}s  median {med:7.3f}s")

        # ---- alignment fitting: one ridge per pair ----
        print(f"\n{'=' * 70}\n{ds}: alignment fitting "
              f"(one ridge solve at n={args.n}, alpha={args.alpha})\n{'=' * 70}")
        for src, tgt in pairs:
            zs = _load_z(args.work_dir, ds, src, zc)
            zt = _load_z(args.work_dir, ds, tgt, zc)
            if zs is None or zt is None:
                print(f"  skip {src}->{tgt} (no extracted Z)")
                continue
            order = _align_ids(zs["ids"], zt["ids"])
            Zs = zs["Z"]
            Zt = zt["Z"][order]
            n = min(args.n, Zs.shape[0])
            sub = np.arange(Zs.shape[0])[:n]
            A, B = Zt[sub], Zs[sub]
            lo, med, all_ts = _bench(
                lambda: _fit_ridge(A, B, alpha=args.alpha), args.repeats)
            pair_entry = result["align_cache"].setdefault(f"{src}_to_{tgt}", {})
            # eval_ds == align_ds: no cross-alignment is needed for the timing table
            pair_entry.setdefault(f"{ds}_probe", {})[f"{ds}_align"] = {
                tag: round(lo, 3),
                f"{tag}_n": int(n),
                f"{tag}_d_src": int(Zs.shape[1]),
                f"{tag}_d_tgt": int(Zt.shape[1]),
                f"{tag}_median_s": round(med, 3),
                f"{tag}_all_s": [round(t, 3) for t in all_ts],
            }
            print(f"  {src:14s} -> {tgt:14s} d {Zs.shape[1]:5d}->{Zt.shape[1]:5d}  "
                  f"min {lo:7.3f}s  median {med:7.3f}s")

    result["_meta"]["loadavg_at_end"] = os.getloadavg()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out_json}")

    _print_summary(result, args.datasets, tag)


def _print_summary(result, datasets, tag):
    """Per-target summary: probe fit, and ridge averaged over that target's sources."""
    print(f"\n{'=' * 78}")
    print("  Per-target summary (seconds, single fit) -- what the paper table needs")
    print(f"{'=' * 78}")
    for ds in datasets:
        by_tgt = {}
        for pair, probe_keys in result["align_cache"].items():
            entry = probe_keys.get(f"{ds}_probe", {}).get(f"{ds}_align")
            if not entry:
                continue
            by_tgt.setdefault(pair.split("_to_")[1], []).append(entry[tag])
        if not by_tgt:
            continue
        print(f"\n  {ds}")
        print(f"    {'target':16s} {'probe fit':>10s} {'ridge mean':>11s} "
              f"{'ridge sd':>9s} {'n_src':>6s}")
        for tgt in sorted(by_tgt, key=lambda t: -len(by_tgt[t])):
            vals = by_tgt[tgt]
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            pf = result["probe_cache"].get(tgt, {}).get(ds, {}).get("probe_fit_s")
            pf_s = f"{pf:10.2f}" if pf is not None else f"{'N/A':>10s}"
            print(f"    {tgt:16s} {pf_s} {statistics.mean(vals):11.2f} "
                  f"{sd:9.2f} {len(vals):6d}")


def _blas_name():
    try:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            np.show_config()
        for line in buf.getvalue().splitlines():
            if "name:" in line:
                return line.split("name:")[1].strip()
    except Exception:
        pass
    return "unknown"


def _parse_cache_roots(items):
    roots = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--cache-root expects <dataset>=<out_dir>, got {it!r}")
        ds, root = it.split("=", 1)
        roots[ds] = root
    if not roots:
        raise SystemExit("--cache-root is required")
    return roots


def main():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="phase", required=True)

    pe = sub.add_parser("extract", help="cache best-layer z-scored features per model")
    pe.add_argument("--cache-root", action="append", default=[],
                    metavar="DATASET=OUT_DIR",
                    help="transfer2 --out-dir holding probes/<dataset>/*.pkl; repeatable")
    pe.add_argument("--work-dir", required=True)
    pe.add_argument("--token", default="slt")
    pe.add_argument("--force", action="store_true")

    pt = sub.add_parser("time", help="time one probe fit and one ridge fit each")
    pt.add_argument("--work-dir", required=True)
    pt.add_argument("--pair-list", required=True)
    pt.add_argument("--datasets", nargs="+", required=True)
    pt.add_argument("--n", type=int, default=1500)
    pt.add_argument("--alpha", type=float, default=1e4)
    pt.add_argument("--repeats", type=int, default=3)
    pt.add_argument("--out-json", required=True)

    args = p.parse_args()
    if args.phase == "extract":
        phase_extract(args)
    else:
        phase_time(args)


if __name__ == "__main__":
    main()
