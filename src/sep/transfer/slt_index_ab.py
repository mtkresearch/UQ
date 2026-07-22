"""A/B test: does the SLT hidden-state index convention change probe AUROC?

Background. In huggingface_models.py the SLT embedding (`emb_tok_before_eos`) is
taken at `hidden[n_generated - 2]`. Teacher-forcing shows the hidden tuple is
indexed by generation step with `hidden[0]` = prefill (last INPUT token), so the
contextualized representation of content token j lives at `hidden[j+1]`. Under
that "read-after" convention the *name-faithful* second-last content token sits at
`hidden[n_generated - 1]`, one step later than the code. The last content token's
own state (`hidden[n_generated]`) is frequently unavailable (fused stop token /
max_new_tokens), so any faithful index must be clamped to `n_new - 1`.

This script regenerates greedy answers for cached 1.7B validation examples and, in
the SAME forward pass, extracts the SLT stack at both indices:

  current  : hidden[n_generated - 2]                      (shipped)
  faithful : hidden[min(n_generated - 1, n_new - 1)]       (+1 shift, guarded)

It then trains per-layer SE probes on cached SE labels (cluster_assignment_entropy)
for each convention and reports best-layer AUROC. If the two are within noise, the
shipped index is fine and comparability with published SEP numbers is preserved;
if faithful wins materially, the convention is worth switching (and regenerating).

Features are freshly generated so absolute AUROC may differ slightly from the
committed transfer runs; the *delta between conventions* is the quantity of
interest, and it is measured on identical forward passes.
"""
import argparse
from sep.uncertainty.utils.config import apply_yaml_config
import os
import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sep.uncertainty.models.huggingface_models import HuggingfaceModel, StoppingCriteriaSub
from transformers import StoppingCriteriaList


def load_labels_and_prompts(gen_path, ed_path):
    """Cached SE labels + per-example question/context + shared few-shot prompt."""
    with open(gen_path, "rb") as f:
        gens = pickle.load(f)
    unc_path = os.path.join(os.path.dirname(gen_path), "uncertainty_measures.pkl")
    with open(unc_path, "rb") as f:
        measures = pickle.load(f)
    ent = np.asarray(
        measures["uncertainty_measures"]["cluster_assignment_entropy"],
        dtype=np.float64)
    with open(ed_path, "rb") as f:
        ed = pickle.load(f)
    ids = list(gens.keys())
    assert len(ids) == len(ent), (len(ids), len(ent))
    return gens, ids, ent, ed["prompt"], ed["BRIEF"], ed["args"]


def best_split(ents):
    splits = np.linspace(1e-10, ents.max(), 100)
    best, best_mse = splits[0], np.inf
    for s in splits:
        low, high = ents < s, ents >= s
        lm = ents[low].mean() if low.any() else 0.0
        hm = ents[high].mean() if high.any() else 0.0
        mse = np.sum((ents[low] - lm) ** 2) + np.sum((ents[high] - hm) ** 2)
        if mse < best_mse:
            best_mse, best = mse, s
    return best


def make_local_prompt(prompt, brief, args, context, question):
    """Reconstruct the validation prompt exactly as generate_answers.py did."""
    brief_always = args.brief_always and args.enable_brief
    p = ""
    if brief_always:
        p += brief
    if args.use_context and (context is not None):
        p += f"Context: {context}\n"
    p += f"Question: {question}\n"
    p += "Answer:"
    return prompt + p


def n_generated_and_new(model, inputs, outputs, input_data):
    """Replicate huggingface_models.predict()'s n_generated / n_new computation."""
    seq = outputs.sequences[0]
    n_input = len(inputs["input_ids"][0])
    n_new = len(seq) - n_input
    full_answer = model.tokenizer.decode(seq, skip_special_tokens=True)
    if not full_answer.startswith(input_data):
        return None, n_new  # matches the ValueError guard; skip such examples
    off = len(input_data)
    answer = full_answer[off:]
    stop_at = len(answer)
    for stop in model.stop_sequences:
        if answer.endswith(stop):
            stop_at = len(answer) - len(stop)
            break
    token_stop_index = model.tokenizer(
        full_answer[:off + stop_at], return_tensors="pt")["input_ids"].shape[1]
    n_generated = token_stop_index - n_input
    if n_generated == 0:
        n_generated = 1
    return n_generated, n_new


def slt_stacks(hidden, n_generated, n_new):
    """Return four stacks (n_layers, hidden) or Nones:

      SLT  current  = hidden[n_generated - 2]                (shipped, line 378)
      SLT  faithful = hidden[min(n_generated - 1, n_new-1)]  (read-after, clamped)
      LAST current  = hidden[n_generated - 1]                (shipped, line 364)
      LAST faithful = hidden[min(n_generated,     n_new-1)]  (read-after, clamped)
    """
    def stack_at(i):
        step = hidden[i]  # tuple over layers, each (1, seq, d)
        return torch.stack([layer[:, -1, :] for layer in step]).squeeze(1)  # (L, d)

    L = len(hidden)
    idxs = {
        "slt_cur": n_generated - 2,
        "slt_faith": min(n_generated - 1, n_new - 1),
        "last_cur": n_generated - 1,
        "last_faith": min(n_generated, n_new - 1),
    }
    if not all(0 <= i < L for i in idxs.values()):
        return None
    return {k: stack_at(i).float().cpu().numpy() for k, i in idxs.items()}


def per_layer_auroc(H, y, seed=0, scaler_frac=0.7):
    """Best-layer AUROC of standardized per-layer logistic SE probe."""
    rng = np.random.default_rng(seed)
    n = H.shape[1]
    idx = rng.permutation(n)
    tr, te = idx[: int(scaler_frac * n)], idx[int(scaler_frac * n):]
    best_L, best_auc, per = -1, -np.inf, []
    for L in range(H.shape[0]):
        X = H[L].astype(np.float64)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = (X - mu) / sd
        clf = LogisticRegression(max_iter=1000).fit(Z[tr], y[tr])
        auc = roc_auc_score(y[te], clf.decision_function(Z[te]))
        per.append(auc)
        if auc > best_auc:
            best_auc, best_L = auc, L
    return best_L, best_auc, np.asarray(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="merged validation_generations.pkl")
    ap.add_argument("--ed", required=True, help="a shard's experiment_details.pkl")
    ap.add_argument("--model", default="Qwen3-1.7B")
    ap.add_argument("--n-examples", type=int, default=400)
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = apply_yaml_config(ap)

    gens, ids, ent, prompt, brief, gargs = load_labels_and_prompts(args.gen, args.ed)
    thr = best_split(ent)
    y_all = (ent >= thr).astype(np.int64)

    rng = np.random.default_rng(args.seed)
    sel = rng.permutation(len(ids))[: args.n_examples]

    model = HuggingfaceModel(
        args.model, stop_sequences="default", max_new_tokens=args.max_new_tokens)

    stacks = {"slt_cur": [], "slt_faith": [], "last_cur": [], "last_faith": []}
    ys, kept, skipped, n_split_cases = [], 0, 0, 0
    for count, si in enumerate(sel):
        ex_id = ids[si]
        ex = gens[ex_id]
        local_prompt = make_local_prompt(
            prompt, brief, gargs, ex.get("context"), ex["question"])
        inputs = model.tokenizer(local_prompt, return_tensors="pt").to("cuda")
        n_input = inputs["input_ids"].shape[1]
        sc = StoppingCriteriaList([StoppingCriteriaSub(
            stops=model.stop_sequences, initial_length=n_input,
            tokenizer=model.tokenizer)])
        with torch.no_grad():
            out = model.model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True, output_scores=True,
                output_hidden_states=True, temperature=args.temperature,
                do_sample=True, stopping_criteria=sc,
                pad_token_id=model.tokenizer.eos_token_id)
        n_gen, n_new = n_generated_and_new(model, inputs, out, local_prompt)
        if n_gen is None:
            skipped += 1
            continue
        st = slt_stacks(out.hidden_states, n_gen, n_new)
        if st is None:
            skipped += 1
            continue
        for k in stacks:
            stacks[k].append(st[k])
        ys.append(y_all[si])
        if n_new > n_gen:
            n_split_cases += 1
        kept += 1
        if (count + 1) % 50 == 0:
            print(f"  {count+1}/{len(sel)} processed (kept {kept}, skipped {skipped})")

    ys = np.asarray(ys)
    H = {k: np.stack(v, axis=1) for k, v in stacks.items()}  # each (n_layers, n, d)
    print(f"kept {kept} examples, skipped {skipped}, split-cases (n_new>n_gen)={n_split_cases}; "
          f"label balance = {ys.mean():.3f} positive; layers = {H['slt_cur'].shape[0]}")

    res = {"model": args.model, "n_examples_kept": int(kept), "skipped": int(skipped),
           "label_pos_frac": float(ys.mean()), "se_threshold": float(thr),
           "n_split_cases": int(n_split_cases)}
    print()
    for slot, cur_key, faith_key, cur_expr, faith_expr in [
        ("SLT ", "slt_cur", "slt_faith",
         "hidden[n_generated-2]", "hidden[min(n_generated-1, n_new-1)]"),
        ("LAST", "last_cur", "last_faith",
         "hidden[n_generated-1]", "hidden[min(n_generated,   n_new-1)]")]:
        Lc, auc_c, per_c = per_layer_auroc(H[cur_key], ys, seed=args.seed)
        Lf, auc_f, per_f = per_layer_auroc(H[faith_key], ys, seed=args.seed)
        res[slot.strip()] = {
            "current_expr": cur_expr, "faithful_expr": faith_expr,
            "current": {"best_layer": int(Lc), "best_auroc": float(auc_c)},
            "faithful": {"best_layer": int(Lf), "best_auroc": float(auc_f)},
            "delta_best": float(auc_f - auc_c),
            "per_layer_delta_mean": float(np.mean(per_f - per_c)),
            "per_layer_delta_max_abs": float(np.max(np.abs(per_f - per_c))),
        }
        print(f"=== {slot} slot ===")
        print(f"  current  {cur_expr:38s} best L{Lc:2d}  AUROC {auc_c:.4f}")
        print(f"  faithful {faith_expr:38s} best L{Lf:2d}  AUROC {auc_f:.4f}")
        print(f"  delta (faithful-current) best={auc_f-auc_c:+.4f}  "
              f"per-layer mean={np.mean(per_f-per_c):+.4f}  max|.|={np.max(np.abs(per_f-per_c)):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        import json
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
