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


def fmt_delta(v):
    """Signed percentage-point delta, but an exact zero carries no sign."""
    return "0.0" if f"{v:.1f}" in ("0.0", "-0.0") else f"{v:+.1f}"


def make_table(rows, eval_ds, align_ds, n):
    align_label = _ALIGN_DS_LABEL.get(align_ds, align_ds.upper())
    eval_label  = _ALIGN_DS_LABEL.get(eval_ds,  eval_ds.upper())

    # The caption states the test-set size and the pair count, so read both off the
    # data instead of hard-coding 500 / 21.
    present = [(src, tgt) for g in _GROUPS for src, tgt in g
               if (eval_ds, align_ds, f"{src}_to_{tgt}", n) in rows]
    n_test = (int(rows[(eval_ds, align_ds, f"{present[0][0]}_to_{present[0][1]}", n)]
                  ["total"]) if present else 500)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\caption{")
    lines.append(
        f"Full error decomposition at $n{{=}}{n}$ on {eval_label};"
        f" alignment dataset: {align_label}."
    )
    lines.append(
        f"Entries are percentages of the ${n_test}$ held-out test examples."
        " Transfer error is the classification error of the transferred probe"
        " with respect"
    )
    lines.append("to the target SE labels.")
    lines.append(r"$\Delta_{\mathrm{excl}}=EF_{\mathrm{only}}+FD_{\mathrm{only}}"
                 r"-F_{\mathrm{only}}$")
    lines.append("is the corrective margin excluding the triple-interaction region,"
                 " while")
    lines.append(r"$\Delta_{\mathrm{acc}}=\Delta_{\mathrm{excl}}-EFD$")
    lines.append("is the exact change in target classification accuracy.")
    lines.append(f"The mean row reports the macro-average across the "
                 f"${len(present)}$ source--target pairs.")
    lines.append(r"}")
    lines.append(r"\label{tab:venn_full_" + f"{eval_ds}_{align_ds}" + r"}")
    lines.append(r"\begin{tabular}{lrrrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Source $\to$ Target"
        r" & $E_{\mathrm{only}}$"
        r" & $F_{\mathrm{only}}$"
        r" & $D_{\mathrm{only}}$"
        r" & $EF_{\mathrm{only}}$"
        r" & $ED_{\mathrm{only}}$"
        r" & $FD_{\mathrm{only}}$"
        r" & $EFD$"
        r" & \shortstack{Transfer\\error}"
        r" & $\Delta_{\mathrm{excl}}$"
        r" & $\Delta_{\mathrm{acc}}$ \\"
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
            pct = {}
            for i, col in enumerate(_COLS):
                pct[col] = 100 * int(r[col]) / total
                all_pcts[i].append(pct[col])
                cols.append(f"{pct[col]:.1f}")
            # Corrective margin without the triple region, then the exact accuracy
            # change: Delta_acc = Delta_excl - EFD.
            d_excl = pct["AB_only"] + pct["BC_only"] - pct["B_only"]
            cols.append(fmt_delta(d_excl))
            cols.append(fmt_delta(d_excl - pct["ABC"]))
            row_str = " & ".join([f"{pair_label:<45}"] + [f"{c:>5}" for c in cols])
            lines.append(row_str + r" \\")

    # Mean row: macro-average over pairs, and the two deltas recomputed from the
    # averaged components (so they stay consistent with the columns above them).
    means = [sum(v) / len(v) if v else 0.0 for v in all_pcts]
    # indices: B_only=1, AB_only=3, BC_only=5, ABC=6
    mean_excl = means[3] + means[5] - means[1]
    mean_cols = ([f"{m:.1f}" for m in means]
                 + [fmt_delta(mean_excl), fmt_delta(mean_excl - means[6])])
    lines.append(r"\midrule")
    mean_str = " & ".join([r"\textbf{Mean}"]
                          + [f"\\textbf{{{c}}}" for c in mean_cols])
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
