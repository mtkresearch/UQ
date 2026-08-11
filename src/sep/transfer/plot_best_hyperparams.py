"""Plot the selected best alpha/lambda per pair: 4 curves per panel.

  - black  : target probe (curve A), from baselines
  - grey   : source probe (curve C), from baselines
  - blue   : ridge, fixed alpha per pair
  - green  : E2-map, fixed lambda per pair
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAIRS_DIR = "sep_scratch/transfer/hyperparam_sweep_v2"

PAIR_NAMES = [
    "llama2_to_mistral",
    "llama32-1b_to_llama31-8b",
    "llama31-8b_to_qwen3-8b",
    "llama31-8b_to_phi4",
    "llama31-8b_to_gemma",
    "llama31-8b_to_nemo",
]

PAIR_LABELS = [
    "Llama2-7B→Mistral-7B",
    "Llama3.2-1B→Llama3.1-8B",
    "Llama3.1-8B→Qwen3-8B",
    "Llama3.1-8B→Phi-4",
    "Llama3.1-8B→Gemma-12B",
    "Llama3.1-8B→Mistral-Nemo",
]

# alpha=1e4 for all pairs (suffix: _a1e+04)
RIDGE_ALPHA = 1e4

# per-pair lambda
E2MAP_LAM = {
    "llama2_to_mistral":        10.0,
    "llama32-1b_to_llama31-8b": 10.0,
    "llama31-8b_to_qwen3-8b":   100.0,
    "llama31-8b_to_phi4":       100.0,
    "llama31-8b_to_gemma":      10.0,
    "llama31-8b_to_nemo":       10.0,
}

TOKEN = "slt"
OUT_DIR = PAIRS_DIR


def _nan(v):
    return [x if x is not None else np.nan for x in v]


def load(path):
    with open(path) as f:
        return json.load(f)


def _draw_panel(ax, name, label):
    d = os.path.join(PAIRS_DIR, name)

    # baselines
    b = load(os.path.join(d, f"transfer_{TOKEN}_baselines.json"))
    g = b["n_grid"]
    if b.get("curveA_native"):
        ax.plot(g, _nan(b["curveA_native"]),
                color="black", ls="-", marker="o", ms=5, lw=1.8,
                label="A: native target probe")
    if b.get("curveC_source_native"):
        ax.plot(g, _nan(b["curveC_source_native"]),
                color="grey", ls=":", marker="v", ms=5, lw=1.5,
                label="C: native source probe")

    # ridge α=1e4
    rdg = load(os.path.join(d, f"transfer_{TOKEN}_a1e+04_rdg.json"))
    key_r = "curveB_ridge_src_probe_budget"
    if key_r in rdg:
        ax.plot(g, _nan(rdg[key_r]),
                color="#1f77b4", ls="--", marker="s", ms=5, lw=1.8,
                label=f"ridge α=1e+04")

    # E2-map, pair-specific lambda
    lam = E2MAP_LAM[name]
    e2 = load(os.path.join(d, f"transfer_{TOKEN}_lam{lam}_e2m.json"))
    key_e = "curveB_e2_map_src_probe_budget"
    if key_e in e2:
        ax.plot(g, _nan(e2[key_e]),
                color="#2ca02c", ls="-.", marker="X", ms=5, lw=1.8,
                label=f"E2-map λ={lam:.0e}")

    ax.set_xscale("log")
    ax.set_xlabel("labeled source probe examples")
    ax.set_ylabel("eval AUROC")
    ax.set_title(label, fontsize=11)
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)
    ax.legend(fontsize=8)


def main():
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    ax_list = np.array(axes).ravel()

    for ax, name, label in zip(ax_list, PAIR_NAMES, PAIR_LABELS):
        _draw_panel(ax, name, label)

    fig.suptitle(
        "Ridge (α=1e+04) vs E2-map (best λ per pair) — probe training budget (SLT)",
        fontsize=13)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"ALL_pairs_best_hyperparams_{TOKEN}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
