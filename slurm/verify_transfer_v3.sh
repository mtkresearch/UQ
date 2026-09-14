#!/bin/bash
# Delete every artefact in the transfer_v3 tree that a killed copy may have left
# half-written, so the next seed run re-copies it.
#
# WHY THIS IS NEEDED
# ------------------
# seed_transfer_v3.sh's cp_one() never overwrites an existing destination -- that
# is what makes it resumable.  The flip side: a file that is present but WRONG is
# preserved forever, silently.  A pre-atomic-cp interruption already produced one
# such file (a 522 MB stub of a 906 MB alignment matrix), which is invisible until
# something tries to unpickle it.
#
# Two things get removed:
#   *.part            an interrupted atomic copy.  Harmless (it will never be
#                     renamed into place) but it wastes space, so clear it.
#   size != source    a file copied before cp_one became atomic, truncated at
#                     whatever byte the kill landed on.
#
# probes/ is deliberately EXEMPT from the size check: migrate_probe_stats adds
# mu_all/sd_all to those pkls in place, so a migrated cache is legitimately
# larger than its v2 source.  Their integrity is checked differently -- the
# migration asserts mu_all[best_layer] == mu bitwise before writing, and refuses
# to touch a cache it cannot verify.
#
# Usage:
#   bash slurm/verify_transfer_v3.sh            # report and delete
#   DRY_RUN=1 bash slurm/verify_transfer_v3.sh  # report only

set -euo pipefail

V2=${V2:-/proj/MR_dataset/mtk53728/UQ/sep_scratch}
OUT_ROOT=${OUT_ROOT:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v3}
DRY_RUN=${DRY_RUN:-0}

V2="$V2" OUT_ROOT="$OUT_ROOT" DRY_RUN="$DRY_RUN" python3 - <<'PY'
import os, sys

V2       = os.environ["V2"]
OUT_ROOT = os.environ["OUT_ROOT"]
DRY      = os.environ["DRY_RUN"] != "0"

# destination subtree -> the v2 tree it was seeded from
SRC_OF = {
    "_probes_shared":   "transfer_v2_btl",
    "transfer_v3_btl":  "transfer_v2_btl",
    "transfer_v3_bbs":  "transfer_v2_bbs",
    "transfer_v3_b2a":  "transfer_v2_b2a",
}

checked = parts = trunc = nosrc = 0
doomed = []

for dst, src in SRC_OF.items():
    D = os.path.join(OUT_ROOT, dst)
    if not os.path.isdir(D):
        print(f"--- {dst}: not present yet, skipping")
        continue
    for root, _, files in os.walk(D):
        for f in files:
            p = os.path.join(root, f)
            if f.endswith(".part"):
                parts += 1
                doomed.append(("part", p, os.path.getsize(p), -1))
                continue
            rel = os.path.relpath(p, D)
            # see header: migration rewrites these, so size divergence is correct
            if rel.split(os.sep)[0] == "probes":
                continue
            checked += 1
            s = os.path.join(V2, src, rel)
            if not os.path.exists(s):
                # produced by this run rather than copied (or the v2 layout moved);
                # nothing to compare against, so leave it alone
                nosrc += 1
                continue
            a, b = os.path.getsize(p), os.path.getsize(s)
            if a != b:
                trunc += 1
                doomed.append(("trunc", p, a, b))

print(f"\nsize-checked {checked} copied files "
      f"({nosrc} had no v2 counterpart, left alone)")
print(f"found {parts} interrupted *.part, {trunc} truncated")

for kind, p, a, b in doomed:
    what = f"{a} bytes" if kind == "part" else f"{a} != {b} bytes"
    if DRY:
        print(f"  would delete [{kind}] {p}  ({what})")
    else:
        os.remove(p)
        print(f"  deleted [{kind}] {p}  ({what})")

if not doomed:
    print("  nothing to clean -- tree is consistent with its v2 sources")
PY
