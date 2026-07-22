"""A3 -- TXTEMB lexical artefact control (checklist item A3 / task 25).

Inoculates against PARALLAX (2605.17028): on teacher-forced corpora a pure
lexical model (TF-IDF cosine / bag-of-words) can hit ~0.98 AUROC because the
answer leaks into the prompt text -- so a "probe" may be reading surface lexical
cues, not model-internal uncertainty. Our datasets are live-generation closed-book,
which should be the *good* regime.

Test: train a TF-IDF + logistic classifier on the PROMPT TEXT (question + context)
to predict binarized SE and correctness. Require **near-chance** (AUROC ~0.5), which
proves the SE signal our hidden-state probe reads is NOT recoverable from prompt
surface lexicon -- no answer-in-prompt leakage.

We report the lexical AUROC next to the hidden-state probe AUROC (from A1/A2) so the
gap is explicit: probe >> lexical means the signal is model-internal.
"""
import argparse
from sep.uncertainty.utils.config import apply_yaml_config
import json
import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sep.transfer.transfer import best_split, binarize, load_entropy


def load_text_labels(gen_path):
    with open(gen_path, "rb") as f:
        gens = pickle.load(f)
    ids = list(gens.keys())
    q = [str(gens[k].get("question", "")) for k in ids]
    ctx = [str(gens[k].get("context", "")) for k in ids]
    ans = [str(gens[k]["most_likely_answer"]["response"]) for k in ids]
    ents = load_entropy(gen_path)
    assert len(ents) == len(ids)
    y_se = binarize(ents, best_split(ents))
    acc = np.asarray(
        [float(gens[k]["most_likely_answer"]["accuracy"]) for k in ids],
        dtype=np.float64,
    )
    y_err = (acc < 0.5).astype(np.int64)
    return ids, q, ctx, ans, y_se, y_err


def tfidf_auroc(texts, y, seed=0, frac=0.7, max_features=20000):
    rng = np.random.default_rng(seed)
    n = len(texts)
    idx = rng.permutation(n)
    tr, te = idx[: int(frac * n)], idx[int(frac * n):]
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                          sublinear_tf=True)
    Xtr = vec.fit_transform([texts[i] for i in tr])
    Xte = vec.transform([texts[i] for i in te])
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, y[tr])
    try:
        return float(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    except ValueError:
        return 0.5


def run(gen_path, seed):
    ids, q, ctx, ans, y_se, y_err = load_text_labels(gen_path)
    prompt = [f"{a} {b}" for a, b in zip(q, ctx)]      # what the model conditions on
    prompt_plus = [f"{a} {b} {c}" for a, b, c in zip(q, ctx, ans)]  # + its own answer

    out = {
        "gen_path": gen_path, "n": len(ids),
        "se_base_rate": float(y_se.mean()), "error_rate": float(y_err.mean()),
        "lexical": {
            "question_only": {
                "se": tfidf_auroc(q, y_se, seed), "err": tfidf_auroc(q, y_err, seed)},
            "prompt_q_plus_context": {
                "se": tfidf_auroc(prompt, y_se, seed), "err": tfidf_auroc(prompt, y_err, seed)},
            "prompt_plus_answer": {
                "se": tfidf_auroc(prompt_plus, y_se, seed), "err": tfidf_auroc(prompt_plus, y_err, seed)},
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = apply_yaml_config(ap)
    assert len(args.gens) == len(args.names)

    results = {}
    for name, gp in zip(args.names, args.gens):
        print(f"[{name}] {gp}")
        r = run(gp, args.seed)
        results[name] = r
        lx = r["lexical"]
        print(f"  SE   lexical: q_only={lx['question_only']['se']:.3f}  "
              f"q+ctx={lx['prompt_q_plus_context']['se']:.3f}  "
              f"+answer={lx['prompt_plus_answer']['se']:.3f}   "
              f"(base rate {r['se_base_rate']:.3f})")
        print(f"  ERR  lexical: q_only={lx['question_only']['err']:.3f}  "
              f"q+ctx={lx['prompt_q_plus_context']['err']:.3f}  "
              f"+answer={lx['prompt_plus_answer']['err']:.3f}   "
              f"(error rate {r['error_rate']:.3f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
