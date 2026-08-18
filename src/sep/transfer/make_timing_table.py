"""Build timing comparison tables: build a probe on the target model directly
("target-trained") vs. transfer a source probe through a learned alignment.

Sources
-------
T1  answer sampling, 11 generations/question   collect_timing run → wandb timing.json
T2  semantic clustering (NLI entailment)       same
T4  target probe fit                           transfer_v2/timing.json  probe_cache
T5  ridge alignment (alpha=1e4)                transfer_v2/timing.json  align_cache

Derived
  T3  hidden-state extraction = T1 / 11 — the single low-temperature forward pass whose
      hidden states the transfer path needs (the other 10 generations are the samples).

Normalisation target: 1500 questions
  T1  collected on 100 questions → scale ×15
  T2  collected on 100 questions → scale ×15
  T4  transfer_v2 probe_fit_s already at 1500 → no scaling
  T5  transfer_v2 ridge_a1e4 already at n=1500 → no scaling

Eval-time probe inference is deliberately excluded from both paths: applying a linear
probe to already-extracted hidden states costs <0.05 s per 500-question eval set, and
costs the same on either path.  (transfer_v2/timing.json still records it under
evaluate → native_inference_s / aligned_inference_s.)

Table columns (per dataset), each total being the sum of the columns to its left:
  Source → Target | T1 | T2 | T4 | Target-trained total | T3 | T5 | Transfer total | Speedup
"""
import argparse
import csv
import glob
import json
import os
from datetime import datetime

DEFAULT_DATASETS = ("squad", "nq")


def secs(t1, t2):
    fmt = "%Y-%m-%d %H:%M:%S"
    return (datetime.strptime(t2, fmt) - datetime.strptime(t1, fmt)).total_seconds()


def find_wandb_timing(run_base):
    """Find the timing.json saved by generate_answers in a wandb run dir."""
    matches = glob.glob(os.path.join(run_base, "**/wandb/**/files/timing.json"), recursive=True)
    return matches[0] if matches else None


def collect_t1_t2(data_gen_timing_dir, datasets=DEFAULT_DATASETS):
    """
    Returns dict: {(dataset, model): {"T1_s": float, "T2_s": float}}
    Reads flat files named {dataset}_{model}.json from data_generation_timing/.
    """
    result = {}
    skipped = []
    # Longest prefix first so e.g. "trivia_qa" wins over a hypothetical "trivia".
    for tf in glob.glob(os.path.join(data_gen_timing_dir, "*.json")):
        fname = os.path.splitext(os.path.basename(tf))[0]  # e.g. "squad_llama-3.1-8b"
        for ds in sorted(datasets, key=len, reverse=True):
            if fname.startswith(ds + "_"):
                model = fname[len(ds) + 1:]
                dataset = ds
                break
        else:
            skipped.append(os.path.basename(tf))
            continue
        with open(tf) as f:
            t = json.load(f)
        T1 = secs(t["t1_inference_start"], t["t1_inference_end"]) if (
            "t1_inference_start" in t and "t1_inference_end" in t) else None
        T2 = secs(t["t2a_clustering_se_start"], t["t2a_clustering_se_end"]) if (
            "t2a_clustering_se_start" in t and "t2a_clustering_se_end" in t) else None
        result[(dataset, model)] = {"T1_s": T1, "T2_s": T2}
    if skipped:
        print(f"WARNING: {len(skipped)} timing file(s) in {data_gen_timing_dir} match none "
              f"of --datasets {list(datasets)}, so their T1/T2 are missing: {sorted(skipped)}")
    return result


def load_transfer_v2_timing(path):
    with open(path) as f:
        return json.load(f)


def _model_tag(gen_path):
    """Extract model tag from a validation_generations.pkl path."""
    parts = gen_path.split(os.sep)
    for i, p in enumerate(parts):
        if p.endswith("_all_" + parts[i].split("_all_")[-1] if "_all_" in p else ""):
            pass
    # path pattern: .../sep_scratch/{ds}_all_{ts}/{model}/...
    for i, p in enumerate(parts):
        if "_all_" in p and i + 1 < len(parts):
            return parts[i + 1]
    return None


def parse_pairs(pair_list_path, datasets=DEFAULT_DATASETS):
    """Parse the pair list.

    Two accepted line formats:
      "<src_model>  <tgt_model>"                       → expanded over `datasets`
      "<src_path>  <tgt_path>  <eval_ds>  <align_ds>"  → cross-alignment lines skipped
    """
    pairs = []
    with open(pair_list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                src, tgt = parts
                # model names given directly; if they look like paths, extract the tag
                src_model = _model_tag(src) if os.sep in src else src
                tgt_model = _model_tag(tgt) if os.sep in tgt else tgt
                if src_model and tgt_model:
                    for ds in datasets:
                        pairs.append((src_model, tgt_model, ds))
            elif len(parts) == 4:
                src_path, tgt_path, eval_ds, align_ds = parts
                if eval_ds != align_ds:
                    continue  # skip cross-alignment
                src_model = _model_tag(src_path)
                tgt_model = _model_tag(tgt_path)
                if src_model and tgt_model:
                    pairs.append((src_model, tgt_model, eval_ds))
    return pairs


GEN_SAMPLES = 100   # collect_timing ran on 100 samples
TARGET_N    = 1500  # normalise everything to 1500 samples
T1T2_SCALE  = TARGET_N / GEN_SAMPLES  # 15.0


def build_row(src, tgt, ds, t1t2, v2):
    pair_name = f"{src}_to_{tgt}"

    # T1, T2 — scale from the 100 questions of the collection run up to 1500
    t1t2_tgt = t1t2.get((ds, tgt), {})
    T1_tgt = t1t2_tgt.get("T1_s") * T1T2_SCALE if t1t2_tgt.get("T1_s") is not None else None
    T2_tgt = t1t2_tgt.get("T2_s") * T1T2_SCALE if t1t2_tgt.get("T2_s") is not None else None

    # T4 — target probe_fit_s, already at 1500 samples
    T4 = v2.get("probe_cache", {}).get(tgt, {}).get(ds, {}).get("probe_fit_s")

    # T5 — ridge alignment alpha=1e4, already at n=1500
    align_entry = v2.get("align_cache", {}).get(pair_name, {})
    T5 = align_entry.get(f"{ds}_probe", {}).get(f"{ds}_align", {}).get("ridge_a1e4")
    if T5 is None:
        for probe_key in align_entry.values():
            for align_key in probe_key.values():
                v = align_key.get("ridge_a1e4")
                if v is not None:
                    T5 = v
                    break
            if T5 is not None:
                break

    # Eval-time probe inference (formerly T6 native / T7 transferred) is deliberately
    # excluded: both paths apply a single linear probe to already-extracted target hidden
    # states, costing <0.05 s per 500-sample eval set, identically for the two paths.

    def _add(*vals):
        if any(v is None for v in vals):
            return None
        return sum(vals)

    # T1 covers all 11 generations per question: 1 low-temperature pass (the one that
    # yields the hidden states) + 10 sampled answers.  The target-trained path needs all
    # 11; the transfer path needs only the single hidden-state pass.
    T1_tgt_train = T1_tgt
    T3_hidden    = T1_tgt / 11 if T1_tgt is not None else None
    train_total    = _add(T1_tgt_train, T2_tgt, T4)
    transfer_total = _add(T3_hidden, T5)
    speedup = (train_total / transfer_total
               if train_total is not None and transfer_total is not None and transfer_total > 0
               else None)

    def fmt(v):
        return f"{v:.2f}" if v is not None else "N/A"

    return {
        "pair":            f"{src} → {tgt}",
        "T1_tgt_s":        fmt(T1_tgt_train),
        "T2_tgt_s":        fmt(T2_tgt),
        "T3_hidden_s":     fmt(T3_hidden),
        "T4_probe_s":      fmt(T4),
        "train_total_s":   fmt(train_total),
        "T5_ridge_s":      fmt(T5),
        "transfer_total_s":fmt(transfer_total),
        "speedup_x":       fmt(speedup),
    }


HEADERS = [
    "pair",
    # Build a probe on the target model directly
    "T1_tgt_s", "T2_tgt_s", "T4_probe_s", "train_total_s",
    # Transfer a source probe to the target model
    "T3_hidden_s", "T5_ridge_s", "transfer_total_s",
    # Speedup
    "speedup_x",
]

HEADER_DISPLAY = [
    "Source → Target",
    # Build a probe on the target model directly
    "Answer sampling (s)", "Semantic clustering (s)", "Probe training (s)",
    "Target-trained total (s)",
    # Transfer a source probe to the target model
    "Hidden-state extraction (s)", "Alignment fitting (s)", "Transfer total (s)",
    # Speedup
    "Speedup (×)",
]

TABLE_NOTE = (
    "Answer sampling: all 11 generations per question on the target model "
    "(1 low-temperature pass yielding the hidden states + 10 sampled answers).  "
    "Hidden-state extraction: the single low-temperature pass the transfer path needs, "
    "estimated as 1/11 of the above.  "
    "Semantic clustering: NLI entailment clustering for the semantic-entropy labels.  "
    "Alignment fitting: ridge regression (alpha=1e4) mapping source → target hidden states.  "
    "Eval-time probe inference is excluded: applying a linear probe to already-extracted "
    "hidden states costs <0.05 s per 500-question eval set, identically for both paths."
)


def _ascii(s):
    """CSV-safe rendering: keep the pretty glyphs for the console only."""
    return s.replace("→", "->").replace("×", "x")


def build_tables(collect_timing_base, transfer_v2_timing, pair_list, out_json, out_csv_dir,
                 datasets=DEFAULT_DATASETS):
    t1t2  = collect_t1_t2(collect_timing_base, datasets=datasets)
    v2    = load_transfer_v2_timing(transfer_v2_timing)
    pairs = parse_pairs(pair_list, datasets=datasets)

    collect_data = {
        f"{ds}/{model}": vals
        for (ds, model), vals in sorted(t1t2.items())
    }
    out_json_dir = os.path.dirname(out_json)
    if out_json_dir:
        os.makedirs(out_json_dir, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(collect_data, f, indent=2)
    print(f"T1/T2 timing saved to {out_json}")

    for ds in datasets:
        ds_pairs = [(s, t) for s, t, d in pairs if d == ds]
        rows = [build_row(s, t, ds, t1t2, v2) for s, t in ds_pairs]

        csv_path = os.path.join(out_csv_dir, f"timing_table_{ds}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([_ascii(h) for h in HEADER_DISPLAY])
            for row in rows:
                writer.writerow([_ascii(row[k]) for k in HEADERS])
        print(f"Table saved: {csv_path}")

        print(f"\n{'='*100}")
        print(f"  Timing table — {ds.upper()}  (all times in seconds, normalised to 1500 samples)")
        print(f"{'='*100}")
        col_w = [max(len(HEADER_DISPLAY[i]), max((len(row[k]) for row in rows), default=0))
                 for i, k in enumerate(HEADERS)]
        header_line = "  " + "  ".join(h.ljust(col_w[i]) for i, h in enumerate(HEADER_DISPLAY))
        print(header_line)
        print("  " + "-" * (sum(col_w) + 2 * len(col_w)))
        for row in rows:
            print("  " + "  ".join(row[k].ljust(col_w[i]) for i, k in enumerate(HEADERS)))
        print(f"\n  Note: {TABLE_NOTE}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collect-timing-base", required=True)
    p.add_argument("--transfer-v2-timing",  required=True)
    p.add_argument("--pair-list",           required=True)
    p.add_argument("--out-json",            required=True)
    p.add_argument("--out-csv-dir",         required=True)
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                   help="datasets to tabulate; must match the <dataset>_<model>.json "
                        "filenames in --collect-timing-base (e.g. trivia_qa)")
    args = p.parse_args()
    build_tables(args.collect_timing_base, args.transfer_v2_timing,
                 args.pair_list, args.out_json, args.out_csv_dir,
                 datasets=tuple(args.datasets))


if __name__ == "__main__":
    main()
