"""Train Semantic Entropy Probes (SEPs) from cached generation runs.

Runnable port of ``semantic_entropy_probes/train-latent-probe.ipynb``. Reads local
run directories (each containing ``validation_generations.pkl`` and
``uncertainty_measures.pkl``), trains linear probes on the cached hidden states to
predict binarized semantic entropy (SEP) and accuracy (Acc. Pr.), evaluates them
in-distribution and out-of-distribution, and saves probes + AUROC figures.

wandb is optional: pass ``--wandb-run-ids`` to download runs first, otherwise point
``--run-dirs`` at directories that already contain the two pickles.

Example
-------
    sep-train-probes \
        --model-name Qwen3-8B \
        --run-dirs runs/trivia_qa runs/squad runs/nq runs/bioasq \
        --ds-names trivia_qa squad nq bioasq \
        --out-dir semantic_entropy_probes
"""
import argparse
import json
import os
import pickle
import warnings
from collections import defaultdict

import numpy as np
import scipy.stats
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

RUN_FILES = {
    "UNC_MEA": "uncertainty_measures.pkl",
    "VAL_GEN": "validation_generations.pkl",
    "WAN_SUM": "wandb-summary.json",
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
class Dataset:
    """Container for one run's hidden states, entropy and accuracy labels."""

    def __init__(self, tbg, slt, entropy, accuracies, name, path):
        self.tbg_dataset = tbg  # (n_layers, n_samples, hidden)
        self.slt_dataset = slt
        self.entropy = entropy
        self.accuracies = accuracies
        self.name = name
        self.path = path
        self.other_ids = []
        self.other_names = []


def load_run(path, n_sample=2000):
    """Load one run directory into (tbg, slt, entropy, accuracies)."""
    with open(os.path.join(path, RUN_FILES["VAL_GEN"]), "rb") as f:
        generations = pickle.load(f)
    with open(os.path.join(path, RUN_FILES["UNC_MEA"]), "rb") as g:
        measures = pickle.load(g)

    entropy = torch.tensor(
        measures["uncertainty_measures"]["cluster_assignment_entropy"]
    ).to(torch.float32)
    accuracies = torch.tensor(
        [r["most_likely_answer"]["accuracy"] for r in generations.values()]
    )
    # emb_last_tok_before_gen -> TBG; emb_tok_before_eos -> SLT.
    tbg = torch.stack(
        [r["most_likely_answer"]["emb_last_tok_before_gen"] for r in generations.values()]
    ).squeeze(-2).transpose(0, 1).to(torch.float32)
    slt = torch.stack(
        [r["most_likely_answer"]["emb_tok_before_eos"] for r in generations.values()]
    ).squeeze(-2).transpose(0, 1).to(torch.float32)

    return (
        tbg[:, :n_sample, :],
        slt[:, :n_sample, :],
        entropy[:n_sample],
        accuracies[:n_sample],
    )


def maybe_download_wandb(run_ids, model_name, dest_root):
    """Download run pickles from wandb; returns list of local dirs."""
    import wandb  # imported lazily so probe training works without wandb

    dirs = []
    api = wandb.Api()
    for run_id in run_ids:
        out = os.path.join(dest_root, model_name, f"run-{run_id.split('/')[-1]}")
        os.makedirs(out, exist_ok=True)
        run = api.run(run_id)
        for fname in RUN_FILES.values():
            try:
                run.file(fname).download(out, exist_ok=True, replace=True)
            except Exception as exc:  # wandb-summary may be absent for some runs
                print(f"  warn: could not download {fname} for {run_id}: {exc}")
        dirs.append(out)
    return dirs


# --------------------------------------------------------------------------- #
# Metrics / bootstrap (from uncertainty/utils/eval_utils.py)
# --------------------------------------------------------------------------- #
def auroc(y_true, y_score):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    return metrics.auc(fpr, tpr)


def _bootstrap(function, rng, n_resamples=1000):
    def inner(data):
        bs = scipy.stats.bootstrap(
            (data,), function, n_resamples=n_resamples,
            confidence_level=0.9, random_state=rng,
        )
        return {
            "std_err": bs.standard_error,
            "low": bs.confidence_interval.low,
            "high": bs.confidence_interval.high,
        }
    return inner


def _compatible_bootstrap(func, rng):
    def helper(y_true_y_score):
        y_true = np.array([i["y_true"] for i in y_true_y_score])
        y_score = np.array([i["y_score"] for i in y_true_y_score])
        return func(y_true, y_score)

    def converted(y_true, y_score):
        wrapped = [{"y_true": i, "y_score": j} for i, j in zip(y_true, y_score)]
        return _bootstrap(helper, rng=rng)(wrapped)
    return converted


def bootstrap_func(y_true, y_score, func, rng):
    return {"mean": func(y_true, y_score),
            "bootstrap": _compatible_bootstrap(func, rng)(y_true, y_score)}


# unpack a list of bootstrap dicts into their means
def auc(aucs):
    return [a["mean"] for a in aucs]


def idf(x):
    return x


# --------------------------------------------------------------------------- #
# Splitting / training helpers
# --------------------------------------------------------------------------- #
def create_Xs_and_ys(datasets, scores, val_test_splits=(0.2, 0.1),
                     test_only=False, no_val=False):
    X = np.array(datasets)
    y = np.array(scores)

    if test_only:
        return (None, None, [X[i] for i in range(X.shape[0])],
                None, None, [y for _ in range(X.shape[0])])

    valid_size, test_size = val_test_splits
    X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = [], [], [], [], [], []
    for i in range(X.shape[0]):
        X_tv, X_test, y_tv, y_test = train_test_split(
            X[i], y, test_size=test_size, random_state=42)
        X_tests.append(X_test)
        y_tests.append(y_test)
        if no_val:
            X_trains.append(X_tv)
            y_trains.append(y_tv)
            continue
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=valid_size, random_state=42)
        X_trains.append(X_train)
        y_trains.append(y_train)
        X_vals.append(X_val)
        y_vals.append(y_val)
    return X_trains, X_vals, X_tests, y_trains, y_vals, y_tests


def concat_Xs_and_ys(layer_range, X_trains, X_vals, X_tests, y_trains, y_vals,
                     y_tests, no_val=False, test_only=False):
    if not no_val:
        X_val_cc = np.concatenate(np.array(X_vals)[layer_range], axis=1)
        y_val_cc = y_vals[layer_range[0]]
    else:
        X_val_cc = y_val_cc = None
    if not test_only:
        X_train_cc = np.concatenate(np.array(X_trains)[layer_range], axis=1)
        y_train_cc = y_trains[layer_range[0]]
    else:
        X_train_cc = y_train_cc = None
    X_test_cc = np.concatenate(np.array(X_tests)[layer_range], axis=1)
    y_test_cc = y_tests[layer_range[0]]
    return X_train_cc, X_val_cc, X_test_cc, y_train_cc, y_val_cc, y_test_cc


def evaluate_on_test(model, X_test, y_test, rng, silent=True, bootstrap=True):
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    test_loss = log_loss(y_test, probs)
    test_acc = float(np.mean((preds == y_test).astype(int)))
    if bootstrap:
        score = bootstrap_func(y_test, probs[:, 1], auroc, rng)
    else:
        score = {"mean": roc_auc_score(y_test, probs[:, 1])}
    if not silent:
        print(f"    test_acc={test_acc:.4f} auroc={score['mean']:.4f} loss={test_loss:.4f}")
    return test_loss, test_acc, score


def train_single_metric(D, token_type, metric, rng):
    """Per-layer probe for one (token, metric) on one dataset."""
    var = token_type[0] + metric[0]
    Xtr, Xva, Xte, ytr, yva, yte = create_Xs_and_ys(
        getattr(D, f"{token_type}_dataset"), getattr(D, metric))
    accs, aucs, models = [], [], []
    for i, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(
            zip(Xtr, Xva, Xte, ytr, yva, yte)):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        _, test_acc, test_auc = evaluate_on_test(model, X_test, y_test, rng)
        accs.append(test_acc)
        aucs.append(test_auc)
        models.append(model)
    setattr(D, f"{var}_accs", accs)
    setattr(D, f"{var}_aucs", aucs)
    setattr(D, f"{var}_models", models)


# --------------------------------------------------------------------------- #
# SE binarization
# --------------------------------------------------------------------------- #
def best_split(entropy):
    """Best threshold minimizing within-group SSE (paper Section 4)."""
    ents = entropy.numpy()
    splits = np.linspace(1e-10, ents.max(), 100)
    mses = []
    for split in splits:
        low, high = ents < split, ents >= split
        low_mean = np.mean(ents[low]) if low.any() else 0.0
        high_mean = np.mean(ents[high]) if high.any() else 0.0
        mses.append(np.sum((ents[low] - low_mean) ** 2)
                    + np.sum((ents[high] - high_mean) ** 2))
    return splits[int(np.argmin(mses))]


def binarize_entropy(entropy, thres):
    b = torch.full_like(entropy, -1, dtype=torch.float)
    b[entropy < thres] = 0
    b[entropy >= thres] = 1
    return b


# --------------------------------------------------------------------------- #
# Layer-range selection + concatenated training
# --------------------------------------------------------------------------- #
def decide_layer_range(Ds, metric, limit, min_layers=5):
    if "entropy" in metric:
        aucs = [np.array(auc(D.sab_aucs)) for D in Ds]
    else:
        aucs = [np.array(auc(D.sa_aucs)) for D in Ds]
    best_mean, best_range = -np.inf, [0, min_layers]

    def average(a, b):
        return np.mean([np.mean(ac[a:b]) for ac in aucs])

    for i in range(limit):
        for j in range(i + 1, limit):
            if j - i < min_layers:
                continue
            m = average(i, j)
            if m > best_mean:
                best_mean, best_range = m, [i, j]
    return best_mean, best_range


def train_concat(D, layer_range, label_attr, out_attr):
    """Train a probe on all data using concatenated layers (for OOD reuse)."""
    all_xy = create_Xs_and_ys(D.slt_dataset, getattr(D, label_attr), test_only=True)
    _, _, X_cc, _, _, y_cc = concat_Xs_and_ys(
        layer_range, *all_xy, no_val=True, test_only=True)
    model = LogisticRegression()
    model.fit(X_cc, y_cc)
    setattr(D, out_attr, model)


def id_train_test(D, sep_range, ap_range, rng):
    """In-distribution: train SEP + Acc probe, test both on accuracy."""
    # SEP trained on binarized SE, tested on error rate.
    all_xy = create_Xs_and_ys(D.slt_dataset, D.b_entropy, no_val=True)
    X_train_cc, _, _, y_train_cc, _, _ = concat_Xs_and_ys(
        sep_range, *all_xy, no_val=True)
    sep = LogisticRegression().fit(X_train_cc, y_train_cc)

    all_xy = create_Xs_and_ys(D.slt_dataset, D.accuracies)
    _, _, X_test_cc, _, _, y_test_cc = concat_Xs_and_ys(sep_range, *all_xy)
    _, acc, au = evaluate_on_test(sep, X_test_cc, 1 - y_test_cc, rng)
    D.isb_accs, D.isb_aucs = [acc], [au]

    # Acc probe trained + tested on accuracy.
    X_train_cc, _, X_test_cc, y_train_cc, _, y_test_cc = concat_Xs_and_ys(
        ap_range, *all_xy, no_val=True)
    ap = LogisticRegression().fit(X_train_cc, y_train_cc)
    _, acc, au = evaluate_on_test(ap, X_test_cc, y_test_cc, rng)
    D.isa_accs, D.isa_aucs = [acc], [au]


def test_one_on_n(D, Ds, sep_range, ap_range, rng):
    """OOD: probes trained on D, evaluated on every other dataset's accuracy."""
    ob_aucs, oa_aucs = {}, {}
    for id_ in D.other_ids:
        D_id = Ds[id_]
        y = 1 - D_id.accuracies  # error rate
        all_xy = create_Xs_and_ys(D_id.slt_dataset, y, test_only=True)

        _, _, X_cc, _, _, y_cc = concat_Xs_and_ys(
            ap_range, *all_xy, no_val=True, test_only=True)
        _, _, au = evaluate_on_test(D.s_amodel, X_cc, 1 - y_cc, rng)
        oa_aucs[D_id.name] = [au]

        _, _, X_cc, _, _, y_cc = concat_Xs_and_ys(
            sep_range, *all_xy, no_val=True, test_only=True)
        _, _, au = evaluate_on_test(D.s_bmodel, X_cc, y_cc, rng)
        ob_aucs[D_id.name] = [au]
    D.osa_aucs, D.osb_aucs = oa_aucs, ob_aucs


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_layerwise(Ds, out_dir):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(Ds), figsize=(5 * len(Ds), 4.5), squeeze=False)
    for i, D in enumerate(Ds):
        ax = axs[0][i]
        for series, lbl in [(auc(D.tb_aucs), "SEP TBG"), (auc(D.sb_aucs), "SEP SLT")]:
            ax.plot(np.arange(len(series)) + 1, series, marker="o", label=lbl)
        ax.set_title(f"AUROC on {D.name.upper()}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Test AUROC")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend()
    plt.tight_layout()
    _save(plt, out_dir, "se_layerwise_both_tok")


def plot_id_ood_bars(Ds, out_dir):
    import matplotlib.pyplot as plt
    for kind in ("id", "ood"):
        fig, axes = plt.subplots(1, len(Ds), figsize=(len(Ds) * 4, 5),
                                 sharey=True, squeeze=False)
        for i, D in enumerate(Ds):
            ax = axes[0][i]
            if kind == "id":
                vals = [D.isb_aucs[0]["mean"], D.isa_aucs[0]["mean"]]
            else:
                vals = [D.sep_ood_avg, D.ap_ood_avg]
            labels = ["SEP", "Acc Pr."]
            bars = ax.bar(labels, vals, width=0.5)
            ax.set_ylim(0.2, 1.0)
            ax.set_title(f"{D.name.upper()} {kind.upper()}")
            if i == 0:
                ax.set_ylabel("Test AUROC")
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{b.get_height():.2f}", ha="center", va="bottom")
        plt.tight_layout()
        _save(plt, out_dir, f"{kind}_performance_barplot")


def _save(plt, out_dir, name):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, f"{name}.pdf")
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    print(f"  saved figure -> {path}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(run_dirs, ds_names, model_name, out_dir, n_sample, num_layers, seed):
    rng = np.random.default_rng(seed)

    Ds = []
    for path, name in zip(run_dirs, ds_names):
        tbg, slt, entropy, acc = load_run(path, n_sample=n_sample)
        Ds.append(Dataset(tbg, slt, entropy, acc, name, path))
        print(f"loaded {name}: layers={tbg.shape[0]} samples={tbg.shape[1]} "
              f"hidden={tbg.shape[2]}")
    for i, D in enumerate(Ds):
        D.other_ids = [j for j in range(len(Ds)) if j != i]
        D.other_names = [ds_names[j] for j in D.other_ids]

    if num_layers is None:
        num_layers = Ds[0].tbg_dataset.shape[0]

    # Binarize SE with a single best universal split across datasets.
    split = best_split(torch.cat([D.entropy for D in Ds], dim=0))
    print(f"best universal split = {split:.4f}")
    for D in Ds:
        D.b_entropy = binarize_entropy(D.entropy, split)
        dummy = max(D.b_entropy.mean().item(), 1 - D.b_entropy.mean().item())
        print(f"  {D.name}: dummy acc {dummy:.4f}")

    # Per-layer probes (SE + Acc, both token positions).
    print("training per-layer probes ...")
    for D in Ds:
        for tok in ("tbg", "slt"):
            train_single_metric(D, tok, "b_entropy", rng)
            train_single_metric(D, tok, "accuracies", rng)

    # SEP-on-SLT tested on accuracy per layer (needed by decide_layer_range).
    for D in Ds:
        sab_aucs = []
        _, _, Xte, _, _, _ = create_Xs_and_ys(D.slt_dataset, 1 - D.accuracies)
        _, _, _, _, _, yte = create_Xs_and_ys(D.slt_dataset, 1 - D.accuracies)
        for i, (X_test, y_test) in enumerate(zip(Xte, yte)):
            _, _, au = evaluate_on_test(D.sb_models[i], X_test, y_test, rng)
            sab_aucs.append(au)
        D.sab_aucs = sab_aucs

    emean, (e1, e2) = decide_layer_range(Ds, "entropy", num_layers)
    amean, (a1, a2) = decide_layer_range(Ds, "acc", num_layers)
    sep_range, ap_range = list(range(e1, e2)), list(range(a1, a2))
    print(f"SEP layer range [{e1},{e2}) mean={emean:.4f}; "
          f"Acc layer range [{a1},{a2}) mean={amean:.4f}")

    for D in Ds:
        D.sep_layer_range = (e1, e2)
        D.ap_layer_range = (a1, a2)

    # ID evaluation + OOD models.
    print("in-distribution eval ...")
    for D in Ds:
        id_train_test(D, sep_range, ap_range, rng)
        train_concat(D, sep_range, "b_entropy", "s_bmodel")
        train_concat(D, ap_range, "accuracies", "s_amodel")

    print("out-of-distribution eval ...")
    for D in Ds:
        test_one_on_n(D, Ds, sep_range, ap_range, rng)

    b_perf, a_perf, win = defaultdict(list), defaultdict(list), []
    for D in Ds:
        for name in D.other_names:
            b_perf[name].append(auc(D.osb_aucs[name])[0])
            a_perf[name].append(auc(D.osa_aucs[name])[0])
            win.append(1 if auc(D.osb_aucs[name])[0] > auc(D.osa_aucs[name])[0] else 0)
    for D in Ds:
        D.sep_ood_avg = float(np.mean(b_perf[D.name]))
        D.ap_ood_avg = float(np.mean(a_perf[D.name]))

    print(f"SEP>Acc OOD win rate: {np.mean(win) * 100:.1f}%")
    for D in Ds:
        print(f"  {D.name}: ID SEP={D.isb_aucs[0]['mean']:.4f} "
              f"ID Acc={D.isa_aucs[0]['mean']:.4f} | "
              f"OOD SEP={D.sep_ood_avg:.4f} OOD Acc={D.ap_ood_avg:.4f}")

    # Figures + inference pkl.
    plot_layerwise(Ds, out_dir)
    plot_id_ood_bars(Ds, out_dir)

    save_attrs = ["s_amodel", "s_bmodel", "sep_layer_range", "ap_layer_range", "name"]
    to_save = tuple({k: getattr(D, k) for k in save_attrs if hasattr(D, k)} for D in Ds)
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    out_pkl = os.path.join(models_dir, f"{model_name}_inference.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(to_save, f)
    print(f"saved probes -> {out_pkl}")


def build_parser():
    p = argparse.ArgumentParser(description="Train Semantic Entropy Probes.")
    p.add_argument("--model-name", required=True,
                   help="Label for output files, e.g. Qwen3-8B.")
    p.add_argument("--run-dirs", nargs="*", default=[],
                   help="Local dirs each with validation_generations.pkl + "
                        "uncertainty_measures.pkl.")
    p.add_argument("--wandb-run-ids", nargs="*", default=[],
                   help="entity/project/run_id to download instead of --run-dirs.")
    p.add_argument("--ds-names", nargs="+", required=True,
                   help="Dataset name per run (same order as run-dirs/wandb ids).")
    p.add_argument("--out-dir", default="semantic_entropy_probes",
                   help="Where to write figures/ and models/.")
    p.add_argument("--wandb-dest", default="wandb_runs",
                   help="Download root when using --wandb-run-ids.")
    p.add_argument("--n-sample", type=int, default=2000)
    p.add_argument("--num-layers", type=int, default=None,
                   help="Override layer count; default inferred from data.")
    p.add_argument("--seed", type=int, default=42)
    return p


def cli():
    args = build_parser().parse_args()

    if args.wandb_run_ids:
        run_dirs = maybe_download_wandb(
            args.wandb_run_ids, args.model_name, args.wandb_dest)
    else:
        run_dirs = args.run_dirs
    if not run_dirs:
        raise SystemExit("Provide --run-dirs or --wandb-run-ids.")
    if len(run_dirs) != len(args.ds_names):
        raise SystemExit("--ds-names must match the number of runs.")

    run(run_dirs, args.ds_names, args.model_name, args.out_dir,
        args.n_sample, args.num_layers, args.seed)


if __name__ == "__main__":
    cli()
