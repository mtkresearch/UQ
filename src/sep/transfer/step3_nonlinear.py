"""Step 3: does a nonlinear cross-scale map beat the linear one?

Step 2 showed a standardized ridge map already matches the native target probe,
so this is a refinement question, not a rescue. We fit a shallow MLP map
f: target-space -> source-space on the same unlabeled pairs, then apply the fixed
source probe to the mapped features (a nonlinear map can't transfer the probe in
closed form, so we compose src_probe(f(z_t)) instead -- still zero target labels).

Curves, all vs number of target examples:
  A  native target probe (labeled)      -- baseline
  B  ridge map + source probe (unlabeled)
  C  shallow MLP map + source probe (unlabeled)
"""
import argparse
from sep.uncertainty.utils.config import apply_yaml_config
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sep.transfer.cka import load_hidden
from sep.transfer.transfer import (
    best_se_layer, best_split, binarize, fit_ridge_map, load_entropy,
    probe_scores, transfer_probe,
)


class MLPMap(nn.Module):
    """target-dim -> hidden -> source-dim, one nonlinearity."""

    def __init__(self, d_in, d_out, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def fit_mlp_map(Zt, Zs, hidden=512, epochs=300, lr=1e-3, wd=1e-4, seed=0):
    """Fit f: Zt -> Zs by MSE. Small data, so full-batch Adam + weight decay."""
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.tensor(Zt, dtype=torch.float32, device=dev)
    ys = torch.tensor(Zs, dtype=torch.float32, device=dev)
    model = MLPMap(Zt.shape[1], Zs.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt), ys)
        loss.backward()
        opt.step()
    model.eval()
    return model, dev


def mlp_scores(model, dev, src_probe, Zt_eval):
    with torch.no_grad():
        mapped = model(torch.tensor(Zt_eval, dtype=torch.float32, device=dev))
        mapped = mapped.cpu().numpy()
    return src_probe.predict_proba(mapped)[:, 1]


def run(source_gen, target_gen, token, out_dir, n_eval, n_grid, hidden, seed):
    rng = np.random.default_rng(seed)
    Hs, ids_s = load_hidden(source_gen, token)
    Ht, ids_t = load_hidden(target_gen, token)
    assert ids_s == ids_t
    N = Hs.shape[1]

    ent_s, ent_t = load_entropy(source_gen), load_entropy(target_gen)
    ys = binarize(ent_s, best_split(ent_s))
    yt = binarize(ent_t, best_split(ent_t))

    perm = rng.permutation(N)
    eval_idx, pool = perm[:n_eval], perm[n_eval:]

    Ls, _ = best_se_layer(Hs[:, pool], ys[pool], seed=seed)
    Lt, _ = best_se_layer(Ht[:, pool], yt[pool], seed=seed)
    print(f"best src layer {Ls}, tgt layer {Lt}")

    Xs, Xt = Hs[Ls].astype(np.float64), Ht[Lt].astype(np.float64)
    mu_s, sd_s = Xs[pool].mean(0), Xs[pool].std(0) + 1e-6
    mu_t, sd_t = Xt[pool].mean(0), Xt[pool].std(0) + 1e-6
    Zs, Zt = (Xs - mu_s) / sd_s, (Xt - mu_t) / sd_t

    src_probe = LogisticRegression(max_iter=1000).fit(Zs[pool], ys[pool])
    Zt_eval, yt_eval = Zt[eval_idx], yt[eval_idx]

    grid = [n for n in n_grid if n <= len(pool)]
    res = {"src_layer": int(Ls), "tgt_layer": int(Lt), "n_grid": grid,
           "curveA_native": [], "curveB_ridge": [], "curveC_mlp": []}

    for n in grid:
        sub = pool[:n]

        if len(np.unique(yt[sub])) < 2:
            res["curveA_native"].append(None)
        else:
            native = LogisticRegression(max_iter=1000).fit(Zt[sub], yt[sub])
            res["curveA_native"].append(
                float(roc_auc_score(yt_eval, native.predict_proba(Zt_eval)[:, 1])))

        M, b = fit_ridge_map(Zt[sub], Zs[sub])
        a_t, c_t = transfer_probe(src_probe, M, b)
        au_r = roc_auc_score(yt_eval, probe_scores(Zt_eval, a_t, c_t))
        res["curveB_ridge"].append(float(au_r))

        model, dev = fit_mlp_map(Zt[sub], Zs[sub], hidden=hidden, seed=seed)
        au_m = roc_auc_score(yt_eval, mlp_scores(model, dev, src_probe, Zt_eval))
        res["curveC_mlp"].append(float(au_m))

        print(f"  n={n:4d}  A={res['curveA_native'][-1]}  "
              f"B(ridge)={au_r:.3f}  C(mlp)={au_m:.3f}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"step3_{token}.json"), "w") as f:
        json.dump(res, f, indent=2)
    _plot(res, token, out_dir)
    print(f"saved step3 results -> {out_dir}")
    return res


def _plot(res, token, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["n_grid"]
    a = [v if v is not None else np.nan for v in res["curveA_native"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(g, a, "o-", label="A: native target (labeled)")
    ax.plot(g, res["curveB_ridge"], "s--", label="B: ridge map (unlabeled)")
    ax.plot(g, res["curveC_mlp"], "^--", label="C: MLP map (unlabeled)")
    ax.set_xscale("log")
    ax.set_xlabel("target examples used")
    ax.set_ylabel("eval AUROC (target SE)")
    ax.set_title(f"Linear vs nonlinear cross-scale map ({token.upper()})")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"step3_{token}.pdf")
    plt.savefig(path, format="pdf", dpi=200)
    plt.close()
    print(f"  saved curve -> {path}")


def main():
    p = argparse.ArgumentParser(description="Step 3: nonlinear cross-scale map.")
    p.add_argument("--source-gen", required=True)
    p.add_argument("--target-gen", required=True)
    p.add_argument("--token", default="slt", choices=["slt", "tbg"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--n-grid", type=int, nargs="+",
                   default=[50, 100, 200, 400, 800, 1500])
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    args = apply_yaml_config(p)
    run(args.source_gen, args.target_gen, args.token, args.out_dir,
        args.n_eval, args.n_grid, args.hidden, args.seed)


if __name__ == "__main__":
    main()
