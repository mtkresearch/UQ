"""
Compute per-n Venn diagram statistics for SE probe transfer experiments.

For each sample and each n, defines:
  A = src_probe_pred != src_label        (source probe error)
  B = ridge_pred     != src_probe_pred   (transfer changes prediction)
  C = src_label      != tgt_label        (source/target label mismatch)
  D = ridge_pred     != tgt_label        (transfer prediction error on target)
     = A XOR B XOR C  (equivalent to A_only | B_only | C_only | ABC)

Writes results/<eval_ds>/<pair>/venn_align_<align_ds>_<aligner_suffix>.json
"""

import os
import json
import argparse
import numpy as np


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def compute_venn(native_preds, align_preds, n_grid, aligner="ridge"):
    """
    native_preds : dict from native_preds_<eval_ds>.json
    align_preds  : dict from predictions_align_<align_ds>_<suffix>.json
    Returns dict with keys 'n_grid' and 'by_n'.
    """
    src_label = np.array(native_preds["src_label"])
    tgt_label = np.array(native_preds["tgt_label"])

    by_n = {}
    valid_ns = [str(n) for n in n_grid
                if str(n) in native_preds["by_n"] and str(n) in align_preds["by_n"]]

    for ns in valid_ns:
        nb = native_preds["by_n"][ns]
        ab = align_preds["by_n"][ns]

        src_probe_pred = nb.get("src_probe_pred")
        tgt_probe_pred = nb.get("tgt_probe_pred")
        ridge_pred     = ab.get(f"{aligner}_pred")

        if src_probe_pred is None or ridge_pred is None:
            by_n[ns] = None
            continue

        src_probe = np.array(src_probe_pred)
        tgt_probe = np.array(tgt_probe_pred) if tgt_probe_pred is not None else None
        ridge      = np.array(ridge_pred)

        A = (src_probe != src_label)
        B = (ridge     != src_probe)
        C = (src_label != tgt_label)
        D = (ridge     != tgt_label)   # definition: ridge_pred != tgt_label

        entry = {
            "A_only":  int(( A & ~B & ~C).sum()),
            "B_only":  int((~A &  B & ~C).sum()),
            "C_only":  int((~A & ~B &  C).sum()),
            "AB_only": int(( A &  B & ~C).sum()),
            "AC_only": int(( A & ~B &  C).sum()),
            "BC_only": int((~A &  B &  C).sum()),
            "ABC":     int(( A &  B &  C).sum()),
            "D":       int(D.sum()),
            "total":   len(src_label),
        }

        # verify D == A_only | B_only | C_only | ABC  (theoretical identity)
        d_from_venn = entry["A_only"] + entry["B_only"] + entry["C_only"] + entry["ABC"]
        assert d_from_venn == entry["D"], (
            f"n={ns}: D identity failed: A_only+B_only+C_only+ABC={d_from_venn} != D={entry['D']}"
        )

        # also record target native probe accuracy if available
        if tgt_probe is not None:
            entry["tgt_probe_error"] = int((tgt_probe != tgt_label).sum())

        by_n[ns] = entry

    return {"n_grid": n_grid, "by_n": by_n}


def process_pair(pair_dir, eval_ds, results, aligner_suffix=None, verbose=True):
    """Compute venn stats for one pair.

    aligner_suffix: if given (e.g. "ridge_a1e4"), only the prediction file for that
    aligner+hyperparam combo is processed; otherwise every combo on disk is.
    """
    native_preds_path = os.path.join(pair_dir, f"native_preds_{eval_ds}.json")
    if not os.path.exists(native_preds_path):
        if verbose:
            print(f"  skip (no native_preds): {pair_dir}")
        return

    native_preds = _load_json(native_preds_path)

    # find all predictions_align_*.json files
    align_files = sorted(
        f for f in os.listdir(pair_dir)
        if f.startswith("predictions_align_") and f.endswith(".json")
    )
    if aligner_suffix is not None:
        align_files = [f for f in align_files
                       if f.endswith(f"_{aligner_suffix}.json")]

    for fname in align_files:
        # filename: predictions_align_<align_ds>_<aligner_suffix>.json
        stem = fname[len("predictions_align_"):-len(".json")]
        # Use the known aligner_suffix parameter to correctly split off align_ds,
        # handling dataset names that contain underscores (e.g. "trivia_qa").
        if aligner_suffix is not None and stem.endswith(f"_{aligner_suffix}"):
            file_align_ds  = stem[:-len(f"_{aligner_suffix}")]
            file_suffix    = aligner_suffix
        else:
            parts          = stem.split("_", 1)
            file_align_ds  = parts[0]
            file_suffix    = parts[1] if len(parts) > 1 else "unknown"

        aligner = file_suffix.split("_")[0]

        align_preds = _load_json(os.path.join(pair_dir, fname))
        n_grid_list = [int(k) for k in sorted(align_preds["by_n"].keys(), key=int)]

        venn = compute_venn(native_preds, align_preds, n_grid_list, aligner=aligner)

        out_path = os.path.join(pair_dir, f"venn_align_{file_align_ds}_{file_suffix}.json")
        _save_json(out_path, venn)
        results.append(out_path)
        if verbose:
            print(f"  saved: {out_path}")


def compute_all(out_dir, eval_datasets=("nq", "squad"), aligner_suffix=None, verbose=True):
    """Compute venn stats for every pair. Returns list of written paths.

    aligner_suffix restricts the work to one aligner+hyperparam combo (see process_pair).
    """
    results_dir = os.path.join(out_dir, "results")
    saved = []
    for eval_ds in eval_datasets:
        ds_dir = os.path.join(results_dir, eval_ds)
        if not os.path.isdir(ds_dir):
            if verbose:
                print(f"skipping {eval_ds}: directory not found")
            continue
        pairs = sorted(p for p in os.listdir(ds_dir) if os.path.isdir(os.path.join(ds_dir, p)))
        if verbose:
            print(f"\n=== eval_ds={eval_ds}  ({len(pairs)} pairs) ===")
        for pair in pairs:
            pair_dir = os.path.join(ds_dir, pair)
            if verbose:
                print(f"  {pair}")
            process_pair(pair_dir, eval_ds, saved,
                         aligner_suffix=aligner_suffix, verbose=verbose)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Compute Venn stats for transfer experiments")
    parser.add_argument("--out-dir", required=True, help="root results directory (transfer_v2)")
    parser.add_argument("--eval-ds", nargs="+", default=["nq", "squad"])
    parser.add_argument("--aligner-suffix", default=None,
                        help="e.g. ridge_a1e4; default processes every combo on disk")
    args = parser.parse_args()

    saved = compute_all(args.out_dir, args.eval_ds, args.aligner_suffix)
    print(f"\nDone. {len(saved)} venn files written.")


if __name__ == "__main__":
    main()
