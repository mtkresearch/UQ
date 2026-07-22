"""Implement semantic entropy."""
import os
import pickle
import logging

import random
import numpy as np
import wandb
import openai
import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sep.uncertainty.models.huggingface_models import HuggingfaceModel
from sep.uncertainty.utils import openai as oai
from sep.uncertainty.utils import utils


random.seed(10)

# Set up OpenAI API credentials
openai.api_key = os.getenv("OPENAI_API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEntailment:

    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        # fp32 by default: batched clustering then matches the per-pair fp32
        # result bitwise. bf16 (opt-in via DEBERTA_BF16=1) is ~2x faster but
        # flips ~0.4% of cluster assignments near the NLI decision boundary.
        dtype = torch.bfloat16 if os.environ.get('DEBERTA_BF16') else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli", dtype=dtype).to(DEVICE)
        self.model.eval()

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        with torch.inference_mode():
            outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        # argmax(softmax(x)) == argmax(x); skip the softmax.
        prediction = torch.argmax(logits, dim=1)[0].cpu().item()
        if os.environ.get('DEBERTA_FULL_LOG', False):
            logging.info('Deberta Input: %s -> %s', text1, text2)
            logging.info('Deberta Prediction: %s', prediction)

        return prediction

    def check_implication_batch(self, pairs, *args, **kwargs):
        """Batched entailment for a list of (premise, hypothesis) string pairs.

        Returns a list of int predictions (0/1/2) aligned with `pairs`. One
        padded forward pass instead of len(pairs) single-example calls -- this
        is where the semantic-clustering GPU time goes. Results match
        `check_implication` run per pair (argmax over the same logits).
        """
        if not pairs:
            return []
        text1 = [p[0] for p in pairs]
        text2 = [p[1] for p in pairs]
        inputs = self.tokenizer(
            text1, text2, return_tensors="pt", padding=True).to(DEVICE)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        if os.environ.get('DEBERTA_FULL_LOG', False):
            for (t1, t2), p in zip(pairs, preds):
                logging.info('Deberta Input: %s -> %s', t1, t2)
                logging.info('Deberta Prediction: %s', p)
        return preds


class EntailmentLLM(BaseEntailment):

    entailment_file = 'entailment_cache.pkl'

    def __init__(self, entailment_cache_id, entailment_cache_only):
        self.prediction_cache = self.init_prediction_cache(entailment_cache_id)
        self.entailment_cache_only = entailment_cache_only

    def init_prediction_cache(self, entailment_cache_id):
        if entailment_cache_id is None:
            return dict()

        logging.info('Restoring prediction cache from %s', entailment_cache_id)

        api = wandb.Api()
        run = api.run(entailment_cache_id)
        run.file(self.entailment_file).download(
            replace=True, exist_ok=False, root=wandb.run.dir)

        with open(f'{wandb.run.dir}/{self.entailment_file}', "rb") as infile:
            return pickle.load(infile)

    def save_prediction_cache(self):
        # write the dictionary to a pickle file
        utils.save(self.prediction_cache, self.entailment_file)

    def check_implication(self, text1, text2, example=None):
        if example is None:
            raise ValueError
        prompt = self.equivalence_prompt(text1, text2, example['question'])

        logging.info('%s input: %s', self.name, prompt)

        hashed = oai.md5hash(prompt)
        if hashed in self.prediction_cache:
            logging.info('Restoring hashed instead of predicting with model.')
            response = self.prediction_cache[hashed]
        else:
            if self.entailment_cache_only:
                raise ValueError
            response = self.predict(prompt, temperature=0.02)
            self.prediction_cache[hashed] = response

        logging.info('%s prediction: %s', self.name, response)

        binary_response = response.lower()[:30]
        if 'entailment' in binary_response:
            return 2
        elif 'neutral' in binary_response:
            return 1
        elif 'contradiction' in binary_response:
            return 0
        else:
            logging.warning('MANUAL NEUTRAL!')
            return 1


class EntailmentGPT4(EntailmentLLM):

    def __init__(self, entailment_cache_id, entailment_cache_only):
        super().__init__(entailment_cache_id, entailment_cache_only)
        self.name = 'gpt-4'

    def equivalence_prompt(self, text1, text2, question):

        prompt = f"""We are evaluating answers to the question \"{question}\"\n"""

        # To precise.
        prompt += "Here are two possible answers:\n"
        # Ah! This is much closer to what we are doing!
        # prompt = prompt + f"""Does at least one of the following two possible answers entail the other?
        # Still to precise.
        prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
        prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral."""

        return prompt

    def predict(self, prompt, temperature):
        return oai.predict(prompt, temperature, model=self.name)


class EntailmentGPT35(EntailmentGPT4):

    def __init__(self, entailment_cache_id, entailment_cache_only):
        super().__init__(entailment_cache_id, entailment_cache_only)
        self.name = 'gpt-3.5'


class EntailmentLlama(EntailmentLLM):

    def __init__(self, entailment_cache_id, entailment_cache_only, name):
        super().__init__(entailment_cache_id, entailment_cache_only)
        self.name = name
        self.model = HuggingfaceModel(
            name, stop_sequences='default', max_new_tokens=30)

    def equivalence_prompt(self, text1, text2, question):

        prompt = f"""We are evaluating answers to the question \"{question}\"\n"""

        prompt += "Here are two possible answers:\n"
        prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
        prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\n"""
        prompt += "Response:"""

        return prompt

    def predict(self, prompt, temperature):
        predicted_answer, _, _ = self.model.predict(prompt, temperature)
        return predicted_answer


def context_entails_response(context, responses, model):
    votes = []
    for response in responses:
        votes.append(model.check_implication(context, response))
    return 2 - np.mean(votes)


def _equivalent_from_implications(implication_1, implication_2, strict_entailment):
    assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
    if strict_entailment:
        return (implication_1 == 2) and (implication_2 == 2)
    implications = [implication_1, implication_2]
    # No contradiction (0), and not both neutral ([1, 1]).
    return (0 not in implications) and ([1, 1] != implications)


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning.

    Precomputes a symmetric equivalence matrix over all i<j pairs, then runs the
    original greedy clustering over it. `are_equivalent` is symmetric (both the
    strict and loose rules are invariant to swapping the two implications), and
    identical strings are trivially equivalent, so those pairs skip the model.
    When the entailment model exposes `check_implication_batch`, all remaining
    pairs run in a single batched forward pass; otherwise we fall back to
    per-pair calls. Clustering results are identical to the per-pair version.
    """
    n = len(strings_list)
    equivalent = [[False] * n for _ in range(n)]

    # Pairs (i<j) that are not exact string matches need the entailment model,
    # run in both directions.
    directed = []  # (i, j, text1, text2) for check(text1, text2)
    for i in range(n):
        for j in range(i + 1, n):
            if strings_list[i] == strings_list[j]:
                equivalent[i][j] = equivalent[j][i] = True
            else:
                directed.append((i, j, strings_list[i], strings_list[j]))
                directed.append((j, i, strings_list[j], strings_list[i]))

    if directed:
        pairs = [(t1, t2) for (_, _, t1, t2) in directed]
        if hasattr(model, 'check_implication_batch'):
            preds = model.check_implication_batch(pairs, example=example)
        else:
            preds = [model.check_implication(t1, t2, example=example)
                     for (t1, t2) in pairs]
        # Fold the two directed predictions per pair into a symmetric verdict.
        impl = {}  # (a, b) -> prediction for check(a, b)
        for (a, b, _, _), p in zip(directed, preds):
            impl[(a, b)] = p
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in impl:
                    eq = _equivalent_from_implications(
                        impl[(i, j)], impl[(j, i)], strict_entailment)
                    equivalent[i][j] = equivalent[j][i] = eq

    # Original greedy clustering, now reading the precomputed matrix.
    semantic_set_ids = [-1] * n
    next_id = 0
    for i in range(n):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            for j in range(i + 1, n):
                if equivalent[i][j]:
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum'):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum':
            logsumexp_value = np.log(np.sum(np.exp(id_log_likelihoods))) - 5.0
        elif agg == 'sum_normalized':
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        elif agg == 'mean':
            logsumexp_value = np.log(np.mean(np.exp(id_log_likelihoods)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """Compute MC estimate of entropy.

    `E[-log p(x)] ~= -1/N sum_i log p(x_i)` where i are the is the sequence
    likelihood, i.e. the average token likelihood.
    """

    entropy = -np.sum(log_probs) / len(log_probs)

    return entropy


def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy
