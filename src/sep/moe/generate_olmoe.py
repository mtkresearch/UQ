"""Generate OLMoE-1B-7B-0924 answers on trivia_qa, capturing both the SLT/TBG
hidden states (same schema `cka.py`/`transfer.py` already consume) and, in a
second full-sequence forward pass, per-layer router logits.

Why a second forward pass: in transformers 4.57.6, `output_router_logits=True`
passed to `.generate()` is not propagated through the sampling loop -- the
returned `GenerateDecoderOnlyOutput` has no `router_logits` field. Verified
empirically. The workaround (also verified end-to-end on this model) is to
generate normally, then re-run the full generated sequence once through
`model(...)` with `output_router_logits=True` to recover a router-logit tensor
per layer per token.

Example ids are a fixed N-subset of the ids already present in the cached
dense Qwen3-1.7B trivia_qa run, so the Confound B (map fragmentation)
diagnostic gets an aligned dense<->MoE pair for free, exactly like the
Qwen3-1.7B->Qwen3-8B cross-scale study.

Also draws `num_generations` high-temperature samples per example (text only,
no hidden states) and runs them through the existing DeBERTa entailment
clustering (`get_semantic_ids` / `cluster_assignment_entropy`) so the output
carries a real semantic-entropy label -- not just accuracy -- matching every
other diagnostic's label convention in `transfer.py::load_entropy`.

Output: `<out>` is a dict keyed by example id, schema-compatible with
`validation_generations.pkl` (`most_likely_answer.emb_tok_before_eos`,
`.emb_last_tok_before_gen`, `.accuracy`, `.response`) plus a `router_logits`
key holding a (n_layers, n_generated_tokens, num_experts) float16 array for
the generated-token span only (prefill tokens excluded -- topic content, not
what the model is uncertain about when answering). A sibling
`uncertainty_measures.pkl` (same directory, same basename convention as the
dense pipeline) holds `cluster_assignment_entropy` in the same key order, so
`transfer.py::load_entropy` reads it unmodified.
"""
import argparse
from sep.uncertainty.utils.config import apply_yaml_config
import json
import logging
import os
import pickle
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

from sep.uncertainty.data.data_utils import load_ds
from sep.uncertainty.models.huggingface_models import StoppingCriteriaSub
from sep.uncertainty.uncertainty_measures.semantic_entropy import (
    EntailmentDeberta, cluster_assignment_entropy, get_semantic_ids)
from sep.uncertainty.utils import utils

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

MODEL_ID = 'allenai/OLMoE-1B-7B-0924'
STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:']


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    return tokenizer, model


def stopping_criteria_for(tokenizer, n_input):
    stops = STOP_SEQUENCES + [tokenizer.eos_token]
    return StoppingCriteriaList([StoppingCriteriaSub(
        stops=stops, tokenizer=tokenizer, initial_length=n_input)])


def build_prompt(train_dataset, prompt_indices, make_prompt, brief, question, context):
    prompt = brief
    for idx in prompt_indices:
        ex = train_dataset[idx]
        prompt += make_prompt(ex['context'], ex['question'], ex['answers']['text'][0], brief, False)
    prompt += make_prompt(context, question, None, brief, False)
    return prompt


@torch.no_grad()
def generate_one(tokenizer, model, prompt, max_new_tokens):
    """Greedy (temperature~0) generation with hidden states, for SLT/TBG + accuracy."""
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    n_input = inputs['input_ids'].shape[1]

    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        return_dict_in_generate=True, output_scores=True, output_hidden_states=True,
        stopping_criteria=stopping_criteria_for(tokenizer, n_input),
        pad_token_id=tokenizer.pad_token_id)

    full_answer = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
    answer = full_answer[len(prompt):] if full_answer.startswith(prompt) else full_answer
    stop_at = len(answer)
    sliced_answer = answer
    for stop in STOP_SEQUENCES:
        if answer.endswith(stop):
            stop_at = len(answer) - len(stop)
            sliced_answer = answer[:stop_at]
            break
    sliced_answer = sliced_answer.strip()

    token_stop_index = tokenizer(
        full_answer[:len(prompt) + stop_at], return_tensors='pt')['input_ids'].shape[1]
    n_generated = max(token_stop_index - n_input, 1)

    hidden = out.hidden_states  # tuple of len n_new_steps, each a tuple of per-layer (1, seq, h)
    last_idx = min(n_generated, len(hidden) - 1) if len(hidden) > 1 else 0
    sec_last_idx = max(last_idx - 1, 0)

    emb_tok_before_eos = torch.stack(
        [layer[:, -1, :] for layer in hidden[sec_last_idx]]).cpu()
    emb_last_tok_before_gen = torch.stack(
        [layer[:, -1, :] for layer in hidden[0]]).cpu()

    log_liks = None  # unused by the diagnostics; skip transition-score bookkeeping.

    return {
        'sequences': out.sequences[0].cpu(),
        'n_input': n_input,
        'response': sliced_answer,
        'emb_tok_before_eos': emb_tok_before_eos,
        'emb_last_tok_before_gen': emb_last_tok_before_gen,
        'log_liks': log_liks,
    }


@torch.no_grad()
def sample_high_temp(tokenizer, model, prompt, max_new_tokens, num_generations, temperature=1.0):
    """Text-only high-temperature samples for the semantic-entropy estimate."""
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    n_input = inputs['input_ids'].shape[1]
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
        num_return_sequences=num_generations, return_dict_in_generate=True,
        stopping_criteria=stopping_criteria_for(tokenizer, n_input),
        output_scores=False, output_hidden_states=False, pad_token_id=tokenizer.pad_token_id)

    answers = []
    for s in range(num_generations):
        full_answer = tokenizer.decode(out.sequences[s], skip_special_tokens=True)
        answer = full_answer[len(prompt):] if full_answer.startswith(prompt) else full_answer
        sliced_answer = answer
        for stop in STOP_SEQUENCES:
            if answer.endswith(stop):
                sliced_answer = answer[:len(answer) - len(stop)]
                break
        answers.append(sliced_answer.strip())
    return answers


@torch.no_grad()
def router_logits_for_generated_span(model, sequences, n_input):
    """Second forward pass: router logits for every layer, generated-token span only."""
    seq = sequences.unsqueeze(0).to('cuda')
    out = model(seq, output_router_logits=True, output_hidden_states=False, use_cache=False)
    # router_logits[l] shape (seq_len, num_experts); keep generated tokens only.
    router = torch.stack([rl[n_input - 1:] for rl in out.router_logits])  # (n_layers, n_gen, E)
    return router.to(torch.float16).cpu().numpy()


def compute_accuracy(response, example, metric):
    if example['answers']['text']:
        return metric(response, example, None)
    return 0.0


def main():
    p = argparse.ArgumentParser(description='OLMoE trivia_qa generation with router logits.')
    p.add_argument('--dense-gen', required=True,
                   help='validation_generations.pkl of the cached dense run, whose example '
                        'ids are subset-sampled for alignment.')
    p.add_argument('--n-samples', type=int, default=300)
    p.add_argument('--max-new-tokens', type=int, default=50)
    p.add_argument('--num-few-shot', type=int, default=5)
    p.add_argument('--num-generations', type=int, default=10,
                   help='High-temperature samples per example for the SE cluster estimate.')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--random-seed', type=int, default=20)
    p.add_argument('--out', required=True)
    args = apply_yaml_config(p)

    random.seed(args.random_seed)

    with open(args.dense_gen, 'rb') as f:
        dense_gens = pickle.load(f)
    dense_ids = list(dense_gens.keys())
    sample_ids = set(random.sample(dense_ids, min(args.n_samples, len(dense_ids))))
    logging.info('Selected %d ids (of %d cached dense ids) for OLMoE generation.',
                 len(sample_ids), len(dense_ids))

    train_dataset, validation_dataset = load_ds('trivia_qa', seed=args.random_seed)
    by_id = {ex['id']: ex for ex in validation_dataset}
    missing = sample_ids - set(by_id)
    if missing:
        raise ValueError(f'{len(missing)} sampled ids not found in trivia_qa validation split.')

    answerable_indices = [i for i, ex in enumerate(train_dataset) if ex['answers']['text']]
    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    brief = 'Answer the following question as briefly as possible.\n'

    def make_prompt(context, question, answer, brief_str, brief_always):
        del brief_always
        prompt = f'Question: {question}\n'
        if answer:
            prompt += f'Answer: {answer}\n\n'
        else:
            prompt += 'Answer:'
        return prompt

    metric = utils.get_metric('squad')
    tokenizer, model = load_model()
    entailment_model = EntailmentDeberta()

    generations = {}
    cluster_entropies = []
    for it, eid in enumerate(sample_ids):
        example = by_id[eid]
        prompt = build_prompt(
            train_dataset, prompt_indices, make_prompt, brief,
            example['question'], example['context'])

        gen = generate_one(tokenizer, model, prompt, args.max_new_tokens)
        router = router_logits_for_generated_span(model, gen['sequences'], gen['n_input'])
        acc = compute_accuracy(gen['response'], example, metric)

        high_temp_answers = sample_high_temp(
            tokenizer, model, prompt, args.max_new_tokens, args.num_generations, args.temperature)
        cond_answers = [f"{example['question']} {a}" for a in high_temp_answers]
        semantic_ids = get_semantic_ids(
            cond_answers, model=entailment_model, strict_entailment=True, example=example)
        cae = cluster_assignment_entropy(semantic_ids)
        cluster_entropies.append(cae)

        generations[eid] = {
            'question': example['question'],
            'context': example['context'],
            'reference': utils.get_reference(example),
            'most_likely_answer': {
                'response': gen['response'],
                'accuracy': acc,
                'emb_tok_before_eos': gen['emb_tok_before_eos'],
                'emb_last_tok_before_gen': gen['emb_last_tok_before_gen'],
            },
            'router_logits': router,
            'semantic_ids': semantic_ids,
            'cluster_assignment_entropy': cae,
        }

        if (it + 1) % 20 == 0:
            logging.info('Generated %d/%d. Last acc=%.2f CAE=%.3f response=%r',
                         it + 1, len(sample_ids), acc, cae, gen['response'][:60])
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(generations, f)

    unc_path = os.path.join(os.path.dirname(args.out), 'uncertainty_measures.pkl')
    with open(unc_path, 'wb') as f:
        pickle.dump({'uncertainty_measures':
                     {'cluster_assignment_entropy': cluster_entropies}}, f)

    accs = [g['most_likely_answer']['accuracy'] for g in generations.values()]
    logging.info('Wrote %d generations -> %s (mean accuracy %.3f, mean CAE %.3f)',
                 len(generations), args.out, float(np.mean(accs)), float(np.mean(cluster_entropies)))

    meta = {'model_id': MODEL_ID, 'n_samples': len(generations),
            'mean_accuracy': float(np.mean(accs)),
            'mean_cluster_assignment_entropy': float(np.mean(cluster_entropies)),
            'random_seed': args.random_seed, 'source_dense_gen': args.dense_gen}
    with open(args.out + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
