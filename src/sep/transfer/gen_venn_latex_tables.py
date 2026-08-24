"""
Generate LaTeX Venn decomposition tables from the combined CSV.

Produces one .tex file per (eval_ds, align_ds) combination, equivalent to the
hand-written table for eval_ds=trivia_qa, align_ds=trivia_qa at n=1500.

Usage:
    python -m sep.transfer.gen_venn_latex_tables \
        --csv sep_scratch/transfer_v2_trivia_qa/summary_plots/venn_ridge_a1e4/table_trivia_qa_ridge_a1e4.csv \
        --out-dir sep_scratch/transfer_v2_trivia_qa/summary_plots/venn_ridge_a1e4/ \
        --n 1500
"""

import argparse
import csv
import os

_SHORT = {
    "gemma-4-12b":  "Gemma-4-12B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "llama-3.2-1b": "Llama-3.2-1B",
    "mistral-nemo": "Mistral-Nemo",
    "phi-4":        "Phi-4",
    "qwen3-8b":     "Qwen3-8B",
}

# Ordered source groups; each group is a list of (src, tgt) pairs.
_GROUPS = [
    [("gemma-4-12b",  "llama-3.1-8b"),
     ("gemma-4-12b",  "mistral-nemo"),
     ("gemma-4-12b",  "phi-4"),
     ("gemma-4-12b",  "qwen3-8b")],
    [("llama-3.1-8b", "gemma-4-12b"),
     ("llama-3.1-8b", "mistral-nemo"),
     ("llama-3.1-8b", "phi-4"),
     ("llama-3.1-8b", "qwen3-8b")],
    [("llama-3.2-1b", "llama-3.1-8b")],
    [("mistral-nemo", "gemma-4-12b"),
     ("mistral-nemo", "llama-3.1-8b"),
     ("mistral-nemo", "phi-4"),
     ("mistral-nemo", "qwen3-8b")],
    [("phi-4",        "gemma-4-12b"),
     ("phi-4",        "llama-3.1-8b"),
     ("phi-4",        "mistral-nemo"),
     ("phi-4",        "qwen3-8b")],
    [("qwen3-8b",     "gemma-4-12b"),
     ("qwen3-8b",     "llama-3.1-8b"),
     ("qwen3-8b",     "mistral-nemo"),
     ("qwen3-8b",     "phi-4")],
]

_ALIGN_DS_LABEL = {
    "trivia_qa": "TriviaQA",
    "nq":        "NQ",
    "squad":     "SQuAD",
}


def load_csv(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["eval_ds"], r["align_ds"], r["pair"], int(r["n"]))
            rows[key] = r
    return rows


def fmt(val, total):
    return f"{100 * int(val) / int(total):.1f}"


def make_table(rows, eval_ds, align_ds, n):
    align_label = _ALIGN_DS_LABEL.get(align_ds, align_ds.upper())
    eval_label  = _ALIGN_DS_LABEL.get(eval_ds,  eval_ds.upper())

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\caption{")
    lines.append(
        f"    Full Venn error decomposition at $n{{=}}{n}$ on {eval_label}"
        f" (\\% of 500 test examples), alignment dataset: {align_label}."
    )
    lines.append(r"}")
    lines.append(r"\label{tab:venn_full_" + f"{eval_ds}_{align_ds}" + r"}")
    lines.append(r"\begin{tabular}{lrrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Source $\to$ Target"
        r"  & $\mathcal{E}$\textsubscript{only}"
        r"  & $\mathcal{F}$\textsubscript{only}"
        r"  & $\mathcal{D}$\textsubscript{only}"
        r"  & $\mathcal{EF}$\textsubscript{only}"
        r"  & $\mathcal{ED}$\textsubscript{only}"
        r"  & $\mathcal{FD}$\textsubscript{only}"
        r"  & $\mathcal{EFD}$"
        r"  & \shortstack{Transfer\\error}"
        r"  & $\Delta_{\mathrm{transfer}}$ \\"
    )

    _COLS = ["A_only", "B_only", "C_only", "AB_only", "AC_only", "BC_only", "ABC", "D"]
    all_pcts = [[] for _ in _COLS]

    for g_idx, group in enumerate(_GROUPS):
        lines.append(r"\midrule")
        for src, tgt in group:
            pair_key = f"{src}_to_{tgt}"
            key = (eval_ds, align_ds, pair_key, n)
            if key not in rows:
                continue
            r = rows[key]
            total = int(r["total"])
            src_s = _SHORT.get(src, src)
            tgt_s = _SHORT.get(tgt, tgt)
            pair_label = f"{src_s} $\\to$ {tgt_s}"
            cols = []
            for i, col in enumerate(_COLS):
                pct = 100 * int(r[col]) / total
                all_pcts[i].append(pct)
                cols.append(f"{pct:.1f}")
            # Delta = EF_only + FD_only - F_only  (AB_only + BC_only - B_only)
            delta = (100 * int(r["AB_only"]) / total
                     + 100 * int(r["BC_only"]) / total
                     - 100 * int(r["B_only"])  / total)
            cols.append(f"{delta:+.1f}")
            row_str = " & ".join([f"{pair_label:<40}"] + [f"{c:>5}" for c in cols])
            lines.append(row_str + r" \\")

    # Mean row
    means = [sum(v) / len(v) if v else 0.0 for v in all_pcts]
    # mean Delta = mean(EF_only) + mean(FD_only) - mean(F_only)
    # indices: AB_only=3, BC_only=5, B_only=1
    mean_delta = means[3] + means[5] - means[1]
    mean_cols = [f"{m:.1f}" for m in means] + [f"{mean_delta:+.1f}"]
    lines.append(r"\midrule")
    mean_str = " & ".join([r"\textbf{Mean}                               "]
                          + [f"\\textbf{{{c:>5}}}" for c in mean_cols])
    lines.append(mean_str + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--eval-ds", nargs="+", default=["trivia_qa"])
    p.add_argument("--align-ds", nargs="+", default=["trivia_qa", "nq", "squad"])
    args = p.parse_args()

    rows = load_csv(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)

    for eval_ds in args.eval_ds:
        for align_ds in args.align_ds:
            tex = make_table(rows, eval_ds, align_ds, args.n)
            fname = f"table_{eval_ds}_align_{align_ds}_n{args.n}.tex"
            out_path = os.path.join(args.out_dir, fname)
            with open(out_path, "w") as f:
                f.write(tex + "\n")
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
