"""Implement HuggingfaceModel models."""
import copy
import logging
import os
from collections import Counter

import accelerate
import torch
from accelerate import Accelerator

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download


from sep.uncertainty.models.base_model import BaseModel
from sep.uncertainty.models.base_model import STOP_SEQUENCES


def resolve_model_path(model_name, hub_org):
    """Prefer a locally downloaded snapshot over a Hub id.

    Looks under `$SEP_MODELS_DIR` (default `/proj/MR_dataset/models`) for a
    directory named exactly `model_name`; falls back to `<hub_org>/<model_name>`
    for a normal Hub download.
    """
    models_dir = os.getenv('SEP_MODELS_DIR', '/proj/MR_dataset/models')
    local = os.path.join(models_dir, model_name)
    return local if os.path.isdir(local) else f'{hub_org}/{model_name}'


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer!
            raise ValueError(
                'More than one split layer.\n'
                f'Currently at layer {layer}.\n'
                f'In map: {device_map_in}\n'
                f'Out map: {device_map}\n')

        logging.info(f'Split layer is {layer}.')

        # remove split for that layer
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f'pop {name}')
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModel(BaseModel):
    """HuggingfaceModel."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None,
                 multi_gpu=False):
        if max_new_tokens is None:
            raise
        self.max_new_tokens = max_new_tokens
        # Opt-in only (see the caveat in the comment below): models too big for one
        # card (phi-4, gemma-4-12b, mistral-nemo at bf16) need `auto` to shard.
        _device_map = 'auto' if multi_gpu else 'cuda'

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
        print(model_name)
        # Single-visible-GPU loading. `device_map='auto'` shards a model across
        # every visible GPU, and cross-device .generate() produces NaN logits
        # under transformers 5.x (verified on this box). The pipeline is meant
        # to run one model per GPU (see slurm/run_multigpu.sh, which pins each
        # shard via CUDA_VISIBLE_DEVICES), so pin to a single device here. The
        # genuinely-too-big-for-one-card path (70b) keeps its explicit sharding.
        self._is_multimodal_wrapper = False
        if 'llama' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
                eightbit = True
            else:
                kwargs = {}
                eightbit = False

            if 'Llama-2' in model_name or 'Llama-3' in model_name:
                base = 'meta-llama'
                model_name = model_name + '-hf' if 'Llama-2' in model_name else model_name
            else:
                base = 'huggyllama'

            model_id = resolve_model_path(model_name, base)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, token_type_ids=None)

            llama65b = '65b' in model_name.lower() and base == 'huggyllama'
            llama2or3_70b = '70b' in model_name.lower() and base == 'meta-llama'

            if llama2or3_70b or llama65b:
                path = snapshot_download(
                    repo_id=f'{base}/{model_name}',
                    allow_patterns=['*.json', '*.model', '*.safetensors'],
                    ignore_patterns=['pytorch_model.bin.index.json']
                )
                config = AutoConfig.from_pretrained(f"{base}/{model_name}")
                with accelerate.init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(config)
                self.model.tie_weights()
                if 'chat' in model_name:
                    max_mem = 17.5 * 4686198491
                else:
                    max_mem = 15 * 4686198491

                device_map = accelerate.infer_auto_device_map(
                    self.model.model,
                    max_memory={0: max_mem, 1: max_mem},
                    dtype='float16'
                )
                device_map = remove_split_layer(device_map)
                full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
                full_model_device_map["lm_head"] = 0

                self.model = accelerate.load_checkpoint_and_dispatch(
                    self.model, path, device_map=full_model_device_map,
                    dtype='float16', skip_keys='past_key_values')

            else:
                # 7b/8b/13b (and any quantized variant): fits on a single card.
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, device_map='cuda', dtype=torch.bfloat16, **kwargs,)

        elif 'mistral' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
            elif model_name.endswith('-4bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_4bit=True,)}
                model_name = model_name[:-len('-4bit')]
            else:
                kwargs = {}

            model_id = resolve_model_path(model_name, 'mistralai')
            config = AutoConfig.from_pretrained(model_id)
            if config.model_type == 'mistral3':
                # Mistral-Small-3.x-Instruct is a multimodal wrapper
                # (Mistral3ForConditionalGeneration): not in the causal-LM
                # auto-map, so load via image-text-to-text and generate through
                # the language submodule.
                # NOTE: transformers 5.14.1's AutoTokenizer conversion of this
                # checkpoint's tekken.json yields vocab_size 151000 vs the
                # model's 131072 embedding table -- IDs misalign and generation
                # is garbage. Use the official `mistral-common` tokenizer for
                # these checkpoints (not yet wired here); AutoTokenizer is
                # unusable for Mistral-Small-3.2-24B.
                self._is_multimodal_wrapper = True
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id, token_type_ids=None,
                    clean_up_tokenization_spaces=False)
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id, device_map='auto', dtype=torch.bfloat16, **kwargs,)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id, token_type_ids=None,
                    clean_up_tokenization_spaces=False)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, device_map=_device_map, dtype=torch.bfloat16, **kwargs,)

        elif 'falcon' in model_name:
            model_id = f'tiiuae/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, token_type_ids=None,
                clean_up_tokenization_spaces=False)

            kwargs = {'quantization_config': BitsAndBytesConfig(
                load_in_8bit=True,)}

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='cuda',
                **kwargs,
            )
        elif 'phi' in model_name.lower():
            model_id = resolve_model_path(model_name, 'microsoft')  # e.g. phi-4
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, token_type_ids=None,
                clean_up_tokenization_spaces=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map=_device_map,
                dtype=torch.bfloat16,
            )
        elif 'gemma' in model_name:
            model_id = resolve_model_path(model_name, 'google')  # e.g. gemma-4-12B
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, token_type_ids=None,
                clean_up_tokenization_spaces=False)
            # gemma-4 (Gemma4UnifiedForConditionalGeneration) is multimodal but
            # is registered in the causal-LM auto-map, so AutoModelForCausalLM
            # resolves it and .generate() returns top-level hidden_states.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map=_device_map,
                dtype=torch.bfloat16,
            )
        elif 'qwen' in model_name.lower():
            model_id = resolve_model_path(model_name, 'Qwen')
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, token_type_ids=None,
                clean_up_tokenization_spaces=False)
            qwen_kwargs = dict(
                device_map=_device_map,
                dtype=torch.bfloat16,
            )
            try:
                import flash_attn  # noqa: F401
                qwen_kwargs['attn_implementation'] = 'flash_attention_2'
                logging.info('Using flash_attention_2 for Qwen.')
            except ImportError:
                logging.info('flash_attn not available; falling back to sdpa.')
                qwen_kwargs['attn_implementation'] = 'sdpa'
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, **qwen_kwargs)
        elif 'olmoe' in model_name.lower():
            # Sparse MoE (e.g. OLMoE-1B-7B-0924): same loading pattern as Qwen,
            # `output_router_logits` is a `.forward()`-only kwarg (not propagated
            # through `.generate()` in this transformers version), so a caller
            # wanting router logits must do a second forward pass on the full
            # generated sequence -- see `sep.moe.generate_olmoe`.
            model_id = resolve_model_path(model_name, 'allenai')
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, token_type_ids=None,
                clean_up_tokenization_spaces=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map='cuda', dtype=torch.bfloat16)
        else:
            raise ValueError

        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        if 'Llama-2' in model_name:
            self.token_limit = 4096
        elif 'qwen' in model_name.lower():
            self.token_limit = 8192
        elif 'olmoe' in model_name.lower():
            self.token_limit = 4096
        elif any(k in model_name.lower() for k in ('llama-3', 'mistral', 'phi', 'gemma')):
            # Modern models with >=16k context; the few-shot prompt alone can
            # exceed the legacy 2048 guard, so raise it well clear of it.
            self.token_limit = 8192
        else:
            self.token_limit = 2048

    
    def predict(self, input_data, temperature, return_full=False, return_latent=False):

        if isinstance(input_data, tuple):
            logging.WARNING("INPUT IS A TUPLE.")
            input_data = input_data[0]

        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        mn_lower = self.model_name.lower()
        if 'llama' in mn_lower or 'falcon' in mn_lower or 'mistral' in mn_lower or 'qwen' in mn_lower:
            if 'token_type_ids' in inputs:  # HF models seems has changed.
                del inputs['token_type_ids']
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        logging.debug('temperature: %f', temperature)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
            )

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        if return_full:
            return full_answer

        # For some models, we need to remove the input_data from the answer.
        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
        else:
            # Newer tokenizers (e.g. Llama-3) may not re-decode the prompt
            # byte-for-byte identical to `input_data`, so `startswith` fails.
            # Recover the offset by decoding the input portion of the actual
            # output sequence the same way `full_answer` was decoded.
            n_input_token = len(inputs['input_ids'][0])
            decoded_input = self.tokenizer.decode(
                outputs.sequences[0][:n_input_token], skip_special_tokens=True)
            if full_answer.startswith(decoded_input):
                input_data_offset = len(decoded_input)
            else:
                raise ValueError(
                    'Could not strip prompt from generation.\n'
                    f'input_data: >{input_data}<\n'
                    f'decoded_input: >{decoded_input}<\n'
                    f'full_answer: >{full_answer}<')

        # Remove input from answer.
        answer = full_answer[input_data_offset:]

        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = 'Error: Stop words not removed successfully!'
                error_msg += f'Answer: >{answer}< '
                error_msg += f'Sliced Answer: >{sliced_answer}<'
                logging.error(error_msg)

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()
        token_stop_index = self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(inputs['input_ids'][0])
        n_generated = token_stop_index - n_input_token

        if n_generated == 0:
            logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
            n_generated = 1

        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        # hidden[k][:, -1, :] encodes generated token k-1 (hidden[0] is the
        # prefill, encoding the last input token), so the last CONTENT token
        # lives at hidden[n_generated]. When the stop word fused into a content
        # token or generation hit max_new_tokens, that index is out of range
        # (len(hidden) == n_new), so clamp to the last available step.
        if len(hidden) == 1:
            logging.warning(
                'Taking first and only generation for hidden! '
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer,
                )
            last_idx = 0
        else:
            last_idx = min(n_generated, len(hidden) - 1)
        last_input = hidden[last_idx]

        # Then access last layer for input
        last_layer = last_input[-1]
        # Then access last token in input.
        last_token_embedding = last_layer[:, -1, :].cpu()

        if return_latent:
            # Second-last content token: exactly one decode step before last_idx,
            # floored at the prefill so it never underflows.
            sec_last_idx = max(last_idx - 1, 0)
            sec_last_input = hidden[sec_last_idx]
            sec_last_token_embedding = torch.stack([layer[:, -1, :] for layer in sec_last_input]).cpu()
    
            # Get the last input token embeddings (before generated tokens)
            last_tok_bef_gen_input = hidden[0]
            last_tok_bef_gen_embedding = torch.stack([layer[:, -1, :] for layer in last_tok_bef_gen_input]).cpu()

        # Get log_likelihoods.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning('Taking first and only generation for log likelihood!')
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning('Generation interrupted by max_token limit.')

        if len(log_likelihoods) == 0:
            raise ValueError

        hidden_states = (last_token_embedding,)

        if return_latent:
            hidden_states += (sec_last_token_embedding, last_tok_bef_gen_embedding)
        else:
            hidden_states += (None, None)

        return_values = (sliced_answer, log_likelihoods, hidden_states)

        return return_values

    def _slice_answer(self, full_answer, input_data, decoded_input=None):
        """Strip prompt and stop sequences from a decoded generation.

        Returns (sliced_answer, n_generated) where n_generated is the number of
        generated tokens up to (and excluding) the stop sequence.

        `decoded_input`, if given, is the prompt re-decoded from the output
        sequence's own input tokens; used as a fallback when a newer tokenizer
        (e.g. Llama-3) does not reproduce `input_data` byte-for-byte and the
        exact-prefix check fails.
        """
        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
        elif decoded_input is not None and full_answer.startswith(decoded_input):
            input_data_offset = len(decoded_input)
        else:
            raise ValueError(
                'Could not strip prompt from generation.\n'
                f'input_data: >{input_data}<\n'
                f'decoded_input: >{decoded_input}<\n'
                f'full_answer: >{full_answer}<')

        answer = full_answer[input_data_offset:]

        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = 'Error: Stop words not removed successfully!'
                error_msg += f'Answer: >{answer}< '
                error_msg += f'Sliced Answer: >{sliced_answer}<'
                logging.error(error_msg)

        sliced_answer = sliced_answer.strip()
        token_stop_index = self.tokenizer(
            full_answer[:input_data_offset + stop_at],
            return_tensors="pt")['input_ids'].shape[1]
        return sliced_answer, token_stop_index

    def predict_batch(self, input_data, temperature, num_return_sequences):
        """Sample `num_return_sequences` generations in a single forward pass.

        Used for the high-temperature samples that only feed the semantic
        entropy estimate: returns text + token log-likelihoods per sample and
        deliberately skips hidden states (unused downstream) for speed/memory.
        """
        if isinstance(input_data, tuple):
            logging.WARNING("INPUT IS A TUPLE.")
            input_data = input_data[0]

        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        mn_lower = self.model_name.lower()
        if 'llama' in mn_lower or 'falcon' in mn_lower or 'mistral' in mn_lower or 'qwen' in mn_lower:
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=False,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
            )

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)

        n_input_token = len(inputs['input_ids'][0])
        results = []
        for s in range(num_return_sequences):
            if len(outputs.sequences[s]) > self.token_limit:
                raise ValueError(
                    'Generation exceeding token limit %d > %d' % (
                        len(outputs.sequences[s]), self.token_limit))

            full_answer = self.tokenizer.decode(
                outputs.sequences[s], skip_special_tokens=True)
            # Re-decode the prompt tokens the same way as `full_answer`, as a
            # fallback for tokenizers that don't reproduce `input_data` exactly.
            decoded_input = self.tokenizer.decode(
                outputs.sequences[s][:n_input_token], skip_special_tokens=True)
            sliced_answer, token_stop_index = self._slice_answer(
                full_answer, input_data, decoded_input=decoded_input)
            n_generated = token_stop_index - n_input_token
            if n_generated <= 0:
                logging.warning(
                    'Only stop_words were generated. For likelihoods, taking '
                    'stop word instead.')
                n_generated = 1

            log_likelihoods = [score.item() for score in transition_scores[s]][:n_generated]
            if len(log_likelihoods) == self.max_new_tokens:
                logging.warning('Generation interrupted by max_token limit.')
            if len(log_likelihoods) == 0:
                raise ValueError

            results.append((sliced_answer, log_likelihoods))

        return results

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input"""

        input_data += ' A'
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']

        target_ids_true = tokenized_prompt_true.clone()
        # Set all target_ids except the last one to -100.
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

        loss_true = model_output_true.loss

        return -loss_true.item()

    def get_perplexity(self, input_data):
        """Get the probability of the model anwering A (True) for the given input"""

        tokenized_data = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']

        with torch.no_grad():
            model_output_true = self.model(tokenized_data, labels=tokenized_data)

        perplexity = - model_output_true.loss.item()


        return perplexity
