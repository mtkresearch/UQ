"""Map-conditioning robustness study (5 checks).

The cross-scale ridge map M (target-space -> source-space) has condition number
~1e17. Prediction works, so this is not fatal, but a reviewer will ask whether the
transfer is numerically unstable. The hypothesis: the *full* map is ill-conditioned,
but the single projected direction the source probe actually needs (a_t = M @ a_s)
lives in the well-conditioned top subspace, so the transfer is stable.

We test that with five checks on cached assets (no regeneration):

  1. Singular-value spectrum + effective rank (participation ratio, stable rank).
  2. Low-rank truncation: AUROC of the transferred probe vs rank-k of M. If it
     saturates at small k, the enormous kappa is a red herring.
  3. Seed stability of the SHIPPED direction: cosine of a_t across alignment
     resamples (direct measure; AUROC std is only indirect).
  4. Ridge-alpha sweep: does increasing alpha just shrink effective rank while
     AUROC stays flat? (robust useful subspace)
  5. Energy decomposition: where does ||a_t||^2 sit on the spectrum? With
     M = U S V^T, a_t = U (S * (V^T a_s)); mode energy = (sigma_i * <v_i, a_s>)^2.
     If energy concentrates on large-sigma modes, the direction we use *avoids*
     the small singular values responsible for the bad kappa.

kappa is a full-map (inverse-problem) metric; at inference we only apply a forward
projection, so 1e17 ~ 1/eps(float64) means the bottom of the spectrum is at the
numerical noise floor and carries no signal -- these checks quantify that.
"""
import argparse
from sep.uncertainty.utils.config import apply_yaml_config
import json
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sep.transfer.cka import load_hidden
from sep.transfer.transfer import best_se_layer, best_split, binarize, load_entropy


def ridge_map_parts(Zt, Zs, alpha=1e3):
    """Return (M, mu_t, mu_s) so a bias can be rebuilt for any M variant."""
    mu_t, mu_s = Zt.mean(0), Zs.mean(0)
    A = Zt - mu_t
    B = Zs - mu_s
    d = A.shape[1]
    M = np.linalg.solve(A.T @ A + alpha * np.eye(d), A.T @ B)  # (d_t, d_s)
    return M, mu_t, mu_s


def transfer_with_M(a_s, c_s, M, mu_t, mu_s):
    """Analytic transfer for an arbitrary M (bias rebuilt consistently)."""
    b = mu_s - mu_t @ M          # (d_s,)
    a_t = M @ a_s                # (d_t,)
    c_t = float(b @ a_s) + c_s
    return a_t, c_t


def auroc(Zt_eval, yt_eval, a_t, c_t):
    return float(roc_auc_score(yt_eval, Zt_eval @ a_t + c_t))


def participation_ratio(s):
    """Effective rank via PR = (sum s^2)^2 / sum s^4 (energy-based)."""
    s2 = s ** 2
    return float((s2.sum() ** 2) / (np.sum(s2 ** 2) + 1e-300))


def stable_rank(s):
    """||M||_F^2 / sigma_max^2."""
    return float((s ** 2).sum() / (s[0] ** 2 + 1e-300))


def shannon_effrank(s):
    """exp(H) of the normalized singular-value energy distribution."""
    p = (s ** 2)
    p = p / (p.sum() + 1e-300)
    h = -np.sum(p * np.log(p + 1e-300))
    return float(np.exp(h))


def prep(source_gen, target_gen, token, seed, n_eval, alpha):
    rng = np.random.default_rng(seed)
    Hs, ids_s = load_hidden(source_gen, token)
    Ht, ids_t = load_hidden(target_gen, token)
    assert ids_s == ids_t, "example ids not aligned across models"
    N = Hs.shape[1]

    ent_s, ent_t = load_entropy(source_gen), load_entropy(target_gen)
    ys = binarize(ent_s, best_split(ent_s))
    yt = binarize(ent_t, best_split(ent_t))

    perm = rng.permutation(N)
    eval_idx, pool = perm[:n_eval], perm[n_eval:]

    Ls, _ = best_se_layer(Hs[:, pool], ys[pool], seed=seed)
    Lt, _ = best_se_layer(Ht[:, pool], yt[pool], seed=seed)

    Xs, Xt = Hs[Ls].astype(np.float64), Ht[Lt].astype(np.float64)
    mu_s, sd_s = Xs[pool].mean(0), Xs[pool].std(0) + 1e-6
    mu_t, sd_t = Xt[pool].mean(0), Xt[pool].std(0) + 1e-6
    Zs = (Xs - mu_s) / sd_s
    Zt = (Xt - mu_t) / sd_t

    src = LogisticRegression(max_iter=1000).fit(Zs[pool], ys[pool])
    a_s = src.coef_.ravel()
    c_s = float(src.intercept_[0])

    return dict(Zs=Zs, Zt=Zt, ys=ys, yt=yt, pool=pool, eval_idx=eval_idx,
                a_s=a_s, c_s=c_s, Ls=int(Ls), Lt=int(Lt), N=N, alpha=alpha)


def run(source_gen, target_gen, token, seed, n_eval, alpha, n_align, n_seeds,
        n_align_seed):
    d = prep(source_gen, target_gen, token, seed, n_eval, alpha)
    Zt, yt, pool, eval_idx = d["Zt"], d["yt"], d["pool"], d["eval_idx"]
    a_s, c_s = d["a_s"], d["c_s"]
    Zt_eval, yt_eval = Zt[eval_idx], yt[eval_idx]
    Zs = d["Zs"]

    align = pool[:n_align]
    M, mu_t, mu_s = ridge_map_parts(Zt[align], Zs[align], alpha=alpha)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)  # M=(d_t,d_s); U(d_t,r) S(r) Vt(r,d_s)
    r = len(S)
    kappa = float(S[0] / (S[-1] + 1e-300))

    a_t_full, c_t_full = transfer_with_M(a_s, c_s, M, mu_t, mu_s)
    auc_full = auroc(Zt_eval, yt_eval, a_t_full, c_t_full)

    out = {
        "source_gen": source_gen, "target_gen": target_gen, "token": token,
        "src_layer": d["Ls"], "tgt_layer": d["Lt"], "n_align": n_align,
        "alpha": alpha, "auc_full": auc_full,
        "spectrum": {
            "rank": int(r), "kappa": kappa,
            "sigma_max": float(S[0]), "sigma_min": float(S[-1]),
            "participation_ratio": participation_ratio(S),
            "stable_rank": stable_rank(S),
            "shannon_effrank": shannon_effrank(S),
            "singular_values": [float(x) for x in S],
        },
    }

    # --- Check 2: low-rank truncation curve ---
    q = S * (Vt @ a_s)                    # mode coeffs for a_t = U @ q
    ks = sorted(set([1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200,
                     300, 500, r]))
    ks = [k for k in ks if 1 <= k <= r]
    trunc = []
    for k in ks:
        a_t_k = U[:, :k] @ q[:k]          # truncated forward projection
        b_k = mu_s - mu_t @ (U[:, :k] * S[:k]) @ Vt[:k]  # consistent bias
        c_t_k = float(b_k @ a_s) + c_s
        trunc.append({"k": int(k),
                      "auc": auroc(Zt_eval, yt_eval, a_t_k, c_t_k),
                      "energy_frac": float(np.sum(q[:k] ** 2) / (np.sum(q ** 2) + 1e-300))})
    out["truncation"] = trunc

    # --- Check 5: energy decomposition on the spectrum ---
    energy = q ** 2
    cum = np.cumsum(energy) / (energy.sum() + 1e-300)
    # index (sorted by sigma desc, which svd already gives) at energy thresholds
    def modes_for(frac):
        return int(np.searchsorted(cum, frac) + 1)
    out["energy"] = {
        "modes_50pct": modes_for(0.5), "modes_90pct": modes_for(0.9),
        "modes_99pct": modes_for(0.99),
        "cum_energy_at_pr": float(cum[min(int(round(out["spectrum"]["participation_ratio"])), r - 1)]),
        "mode_energy_head": [float(x) for x in energy[:20]],
    }

    # --- Check 3: seed stability of the shipped direction a_t ---
    # Resample DISTINCT alignment subsets (< pool size) so each seed sees a
    # genuinely different draw; n_align==len(pool) would return the same set.
    m = min(n_align_seed, len(pool))
    dirs = []
    aucs_seed = []
    for s_i in range(n_seeds):
        rng = np.random.default_rng(1000 + s_i)
        sub = rng.choice(pool, size=m, replace=False)
        Ms, mts, mss = ridge_map_parts(Zt[sub], Zs[sub], alpha=alpha)
        a_ti, c_ti = transfer_with_M(a_s, c_s, Ms, mts, mss)
        dirs.append(a_ti / (np.linalg.norm(a_ti) + 1e-12))
        aucs_seed.append(auroc(Zt_eval, yt_eval, a_ti, c_ti))
    D = np.stack(dirs)
    Cos = D @ D.T
    n = len(dirs)
    off = Cos[~np.eye(n, dtype=bool)]
    out["seed_stability"] = {
        "n_seeds": n_seeds, "n_align_per_seed": int(m),
        "direction_cosine_mean": float(off.mean()),
        "direction_cosine_min": float(off.min()),
        "auc_mean": float(np.mean(aucs_seed)),
        "auc_std": float(np.std(aucs_seed)),
    }

    # --- Check 4: alpha sweep ---
    sweep = []
    for a in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
        Ma, mta, msa = ridge_map_parts(Zt[align], Zs[align], alpha=a)
        Sa = np.linalg.svd(Ma, compute_uv=False)
        a_ta, c_ta = transfer_with_M(a_s, c_s, Ma, mta, msa)
        sweep.append({
            "alpha": a, "auc": auroc(Zt_eval, yt_eval, a_ta, c_ta),
            "participation_ratio": participation_ratio(Sa),
            "kappa": float(Sa[0] / (Sa[-1] + 1e-300)),
        })
    out["alpha_sweep"] = sweep

    return out, (S, np.array([t["k"] for t in trunc]),
                 np.array([t["auc"] for t in trunc]), auc_full, cum)


def _plots(all_res, plotdata, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(all_res.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # spectrum (log)
    ax = axes[0]
    for nm in names:
        S = plotdata[nm][0]
        ax.plot(np.arange(1, len(S) + 1), S / S[0], label=nm)
    ax.set_yscale("log"); ax.set_xlabel("singular value index")
    ax.set_ylabel(r"$\sigma_i/\sigma_1$"); ax.set_title("Normalized SV spectrum")
    ax.grid(True, ls="--", lw=0.5); ax.legend()

    # truncation curve
    ax = axes[1]
    for nm in names:
        _, ks, aucs, auc_full, _ = plotdata[nm]
        ax.plot(ks, aucs, "o-", label=f"{nm} (full={auc_full:.3f})")
        ax.axhline(auc_full, color="gray", ls=":", lw=0.5)
    ax.set_xscale("log"); ax.set_xlabel("rank k of truncated map")
    ax.set_ylabel("eval AUROC"); ax.set_title("Low-rank truncation")
    ax.grid(True, ls="--", lw=0.5); ax.legend()

    # cumulative energy
    ax = axes[2]
    for nm in names:
        cum = plotdata[nm][4]
        ax.plot(np.arange(1, len(cum) + 1), cum, label=nm)
    ax.set_xscale("log"); ax.set_xlabel("mode index (sigma desc)")
    ax.set_ylabel(r"cumulative $\|a_t\|^2$ fraction")
    ax.set_title("Energy of shipped direction on spectrum")
    ax.grid(True, ls="--", lw=0.5); ax.legend()

    plt.tight_layout()
    p = os.path.join(out_dir, "conditioning_slt.pdf")
    plt.savefig(p, format="pdf", dpi=200)
    plt.savefig(p.replace(".pdf", ".png"), format="png", dpi=150)
    plt.close()
    print(f"saved plots -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--token", default="slt", choices=["slt", "tbg"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1e3)
    ap.add_argument("--n-align", type=int, default=1500)
    ap.add_argument("--n-align-seed", type=int, default=1000,
                    help="alignment size per seed for stability resampling (< pool)")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = apply_yaml_config(ap)
    assert len(args.sources) == len(args.targets) == len(args.names)

    all_res, plotdata = {}, {}
    for nm, sg, tg in zip(args.names, args.sources, args.targets):
        print(f"\n===== {nm} =====")
        res, pd = run(sg, tg, args.token, args.seed, args.n_eval,
                      args.alpha, args.n_align, args.n_seeds, args.n_align_seed)
        all_res[nm] = res
        plotdata[nm] = pd
        sp = res["spectrum"]
        print(f"  layers src{res['src_layer']}->tgt{res['tgt_layer']}  "
              f"kappa={sp['kappa']:.2e}  PR={sp['participation_ratio']:.1f}  "
              f"stable_rank={sp['stable_rank']:.1f}  full AUROC={res['auc_full']:.3f}")
        e = res["energy"]
        print(f"  energy: 50%={e['modes_50pct']} modes, 90%={e['modes_90pct']}, "
              f"99%={e['modes_99pct']}")
        ss = res["seed_stability"]
        print(f"  seed: dir-cosine mean={ss['direction_cosine_mean']:.3f} "
              f"min={ss['direction_cosine_min']:.3f}  AUROC {ss['auc_mean']:.3f}"
              f"+-{ss['auc_std']:.3f}")
        # truncation headline: smallest k within 0.01 of full
        for t in res["truncation"]:
            if t["auc"] >= res["auc_full"] - 0.01:
                print(f"  truncation: rank {t['k']} reaches full-0.01 "
                      f"(AUROC {t['auc']:.3f}, {100*t['energy_frac']:.1f}% energy)")
                break
        print("  alpha sweep (alpha: auc / PR / kappa):")
        for s in res["alpha_sweep"]:
            print(f"    {s['alpha']:.0e}: {s['auc']:.3f} / {s['participation_ratio']:.1f} "
                  f"/ {s['kappa']:.1e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_res, f, indent=2)
    _plots(all_res, plotdata, os.path.dirname(args.out))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
