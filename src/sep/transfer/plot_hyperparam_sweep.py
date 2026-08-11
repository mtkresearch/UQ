"""Plot probe-budget AUROC for ridge (alpha sweep) and E2-map (lambda sweep).

Loads:
  - baselines (curve A, curve C) from transfer_slt_a1e+04_baselines.json  (once per pair)
  - ridge sweep from transfer_slt_a<alpha>_rdg.json        (one per alpha value)
  - E2-map sweep from transfer_slt_a1e+04_lam<lam>_e2m.json (one per lambda value)

Produces one PNG with 6 panels (one per model pair). Each panel:
  - 5 blue curves  : ridge, alpha in {1e1, 1e2, 1e3, 1e4, 1e5}, light->dark
  - 5 green curves : E2-map, lambda in {1e-2, 1e-1, 1e0, 1e1, 1e2}, light->dark
  - 1 black curve  : curve A (native target probe)
  - 1 grey curve   : curve C (native source probe)

Usage:
  python -m sep.transfer.plot_hyperparam_sweep \
    --pairs-dir sep_scratch/transfer/hyperparam_sweep \
    --pair-names llama2_to_mistral llama32-1b_to_llama31-8b \
                 llama31-8b_to_qwen3-8b llama31-8b_to_phi4 \
                 llama31-8b_to_gemma llama31-8b_to_nemo \
    --pair-labels "Llama2-7B->Mistral-7B" "Llama3.2-1B->Llama3.1-8B" \
                  "Llama3.1-8B->Qwen3-8B" "Llama3.1-8B->Phi-4" \
                  "Llama3.1-8B->Gemma-12B" "Llama3.1-8B->Mistral-Nemo" \
    --ridge-alphas 1e1 1e2 1e3 1e4 1e5 \
    --e2map-lams   1e-2 1e-1 1e0 1e1 1e2 \
    --token slt \
    --out-dir sep_scratch/transfer/hyperparam_sweep
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _blues(n):
    return plt.cm.Blues(np.linspace(0.35, 0.9, n))


def _greens(n):
    return plt.cm.Greens(np.linspace(0.35, 0.9, n))


def _nan(v):
    return [x if x is not None else np.nan for x in v]


def _alpha_suffix(alpha):
    """Matches transfer.py: no suffix for 1e3 (the default), _a<fmt> otherwise."""
    return "" if alpha == 1e3 else f"_a{alpha:.0e}"


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"not found: {path}")
    with open(path) as f:
        return json.load(f)


def _draw_panel(ax, pair_dir, pair_label, token, ridge_alphas, e2map_lams):

    blues  = _blues(len(ridge_alphas))
    greens = _greens(len(e2map_lams))

    # --- Baselines: curve A and curve C (loaded once) ---
    baseline_path = os.path.join(pair_dir, f"transfer_{token}_baselines.json")
    try:
        b = load_json(baseline_path)
        g = b["n_grid"]
        if "curveA_native" in b:
            ax.plot(g, _nan(b["curveA_native"]),
                    color="black", ls="-", marker="o", ms=4, lw=1.5,
                    label="A: native target (labeled)")
        if "curveC_source_native" in b:
            ax.plot(g, _nan(b["curveC_source_native"]),
                    color="grey", ls=":", marker="v", ms=4, lw=1.5,
                    label="C: native source (labeled)")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        g = None

    # --- Ridge alpha sweep (probe-budget) ---
    for idx, alpha in enumerate(ridge_alphas):
        fname = f"transfer_{token}{_alpha_suffix(alpha)}_rdg.json"
        path  = os.path.join(pair_dir, fname)
        try:
            r = load_json(path)
            if g is None:
                g = r["n_grid"]
            key = "curveB_ridge_src_probe_budget"
            if key in r:
                ax.plot(g, _nan(r[key]),
                        color=blues[idx], ls="--", marker="s", ms=4, lw=1.2,
                        label=f"ridge α={alpha:.0e}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    # --- E2-map lambda sweep (probe-budget) ---
    for idx, lam in enumerate(e2map_lams):
        fname = f"transfer_{token}_lam{lam}_e2m.json"
        path  = os.path.join(pair_dir, fname)
        try:
            r = load_json(path)
            if g is None:
                g = r["n_grid"]
            key = "curveB_e2_map_src_probe_budget"
            if key in r:
                ax.plot(g, _nan(r[key]),
                        color=greens[idx], ls="-.", marker="X", ms=4, lw=1.2,
                        label=f"E2-map λ={lam:.0e}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    ax.set_xscale("log")
    ax.set_xlabel("labeled source probe examples")
    ax.set_ylabel("eval AUROC")
    ax.set_title(pair_label, fontsize=10)
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)
    ax.legend(fontsize=6, ncol=2)


def plot_sweep(pairs_dir, pair_names, pair_labels, token,
               ridge_alphas, e2map_lams, out_dir):
    n = len(pair_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    ax_list = np.array(axes).ravel()

    for ax, name, label in zip(ax_list, pair_names, pair_labels):
        pair_dir = os.path.join(pairs_dir, name)
        _draw_panel(ax, pair_dir, label, token, ridge_alphas, e2map_lams)

    for ax in ax_list[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Ridge (α sweep) vs E2-map (λ sweep) — probe training budget ({token.upper()})",
        fontsize=13)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ALL_pairs_hyperparam_sweep_{token}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved -> {path}")


def main():
    p = argparse.ArgumentParser(
        description="Plot ridge alpha sweep vs E2-map lambda sweep.")
    p.add_argument("--pairs-dir", required=True)
    p.add_argument("--pair-names",  nargs="+", required=True)
    p.add_argument("--pair-labels", nargs="+", default=None)
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--ridge-alphas", nargs="+", type=float,
                   default=[1e1, 1e2, 1e3, 1e4, 1e5])
    p.add_argument("--e2map-lams",   nargs="+", type=float,
                   default=[1e-2, 1e-1, 1e0, 1e1, 1e2])
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    labels = a.pair_labels if a.pair_labels else a.pair_names
    assert len(labels) == len(a.pair_names), "--pair-labels must match --pair-names"
    out = a.out_dir if a.out_dir else a.pairs_dir
    plot_sweep(a.pairs_dir, a.pair_names, labels, a.token,
               a.ridge_alphas, a.e2map_lams, out)


if __name__ == "__main__":
    main()
