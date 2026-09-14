"""Add per-layer pool statistics (mu_all/sd_all) to pre-existing probe caches.

Probe caches written before convention D stored only the scalar `mu`/`sd` pair
for `best_layer`.  transfer2.py now needs the statistics at OTHER layers too --
the source at the eval dataset's best layer, the target at whatever
--transfer-mode picked, every candidate layer during a best-to-align search --
and `_layer_stats()` refuses to guess, so those caches raise a KeyError.

Everything a probe cache decides is left untouched: this reads the hidden states
and recomputes only the pool means and standard deviations, then asserts that
`mu_all[best_layer]` reproduces the stored `mu` bitwise.  That assertion is the
whole point -- if it holds, the cache's split, layer search and fitted probes are
provably the same objects they were before, so a migrated cache is
interchangeable with a freshly fitted one.  Re-running phase probe_cache instead
would refit ~33-49 logistic regressions per (model, dataset) to arrive at the
same answer.

Idempotent: caches that already carry mu_all are skipped.

Usage:
    python -m sep.transfer.migrate_probe_stats --probes-dir <run>/probes
    python -m sep.transfer.migrate_probe_stats --probes-dir <run>/probes --dry-run
"""
import argparse
import os
import pickle
import sys

import numpy as np

from sep.transfer.transfer2 import _load_hidden


def migrate_one(path, dry_run=False):
    """Returns one of 'skip', 'ok', or an error string."""
    with open(path, "rb") as f:
        cache = pickle.load(f)

    if "mu_all" in cache:
        return "skip"

    gen_path = cache["gen_path"]
    if not os.path.exists(gen_path):
        return f"ERROR: hidden states gone: {gen_path}"

    H, _ids = _load_hidden(gen_path, cache["token"])
    n_layers = H.shape[0]
    pool = np.array(cache["pool"])

    mu_all = np.empty((n_layers, H.shape[2]), dtype=np.float64)
    sd_all = np.empty((n_layers, H.shape[2]), dtype=np.float64)
    for L in range(n_layers):
        X_L = H[L].astype(np.float64)
        mu_all[L] = X_L[pool].mean(0)
        sd_all[L] = X_L[pool].std(0) + 1e-6

    # The bitwise check.  Same rows, same dtype, same +1e-6 as phase 1 used, so
    # exact equality is the right assertion -- np.allclose would hide a pool
    # mismatch, which is precisely the failure that would silently invalidate
    # every result seeded from this cache.
    L0 = cache["best_layer"]
    if not (np.array_equal(mu_all[L0], np.asarray(cache["mu"], dtype=np.float64))
            and np.array_equal(sd_all[L0], np.asarray(cache["sd"], dtype=np.float64))):
        return (f"ERROR: mu_all[{L0}] != stored mu -- the cache's pool or token no "
                f"longer reproduces its own statistics; refit it instead")

    if dry_run:
        return "ok"

    cache["mu_all"] = mu_all
    cache["sd_all"] = sd_all
    # Write-then-rename: a crash mid-write would otherwise leave a truncated pkl
    # where a perfectly good cache used to be.
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    os.replace(tmp, path)
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probes-dir", required=True,
                    help="<run>/probes -- scanned for <dataset>/<model>.pkl")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify the bitwise check without writing")
    args = ap.parse_args()

    # -L: a symlinked probes/ is a legitimate layout, but writing through one
    # mutates the tree it points at, so refuse it.
    if os.path.islink(args.probes_dir.rstrip("/")) and not args.dry_run:
        sys.exit(f"{args.probes_dir} is a symlink; migrating through it would "
                 f"modify another run's caches.  Materialise it first.")

    paths = []
    for root, _dirs, files in os.walk(args.probes_dir, followlinks=True):
        paths += [os.path.join(root, f) for f in sorted(files) if f.endswith(".pkl")]
    if not paths:
        sys.exit(f"no *.pkl under {args.probes_dir}")

    n_ok = n_skip = n_err = 0
    for p in sorted(paths):
        rel = os.path.relpath(p, args.probes_dir)
        res = migrate_one(p, dry_run=args.dry_run)
        if res == "skip":
            n_skip += 1
            print(f"  skip    {rel} (already has mu_all)")
        elif res == "ok":
            n_ok += 1
            print(f"  {'check ' if args.dry_run else 'wrote '} {rel}")
        else:
            n_err += 1
            print(f"  {rel}: {res}")

    print(f"\n{len(paths)} caches: {n_ok} migrated, {n_skip} already done, "
          f"{n_err} failed")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
