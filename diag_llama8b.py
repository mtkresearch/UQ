"""Isolation test: find a config where Llama-3.1-8B produces coherent text.

Loads the model under several settings and does ONE greedy generation each,
plus a finite-logits check. Prints which configs are healthy. ~read-only wrt
the pipeline; does not import the pipeline code.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B"
PROMPT = ("Answer the following question as briefly as possible.\n"
          "Question: What is the capital of France?\nAnswer:")

CONFIGS = [
    {"label": "bf16 + sdpa (default attn), single GPU",
     "kw": {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}},
    {"label": "bf16 + eager, single GPU",
     "kw": {"torch_dtype": torch.bfloat16, "attn_implementation": "eager"}},
    {"label": "fp16 + sdpa, single GPU",
     "kw": {"torch_dtype": torch.float16, "attn_implementation": "sdpa"}},
]

tok = AutoTokenizer.from_pretrained(MODEL)


def check(cfg):
    print("\n=== " + cfg["label"] + " ===")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map={"": 0}, **cfg["kw"])
    model.eval()
    inputs = tok(PROMPT, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits
    finite = torch.isfinite(logits).all().item()
    print("all logits finite on forward pass:", finite)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    text = tok.decode(gen[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    print("greedy generation:", repr(text))
    del model
    torch.cuda.empty_cache()


for cfg in CONFIGS:
    try:
        check(cfg)
    except Exception as e:
        print("FAILED:", type(e).__name__, str(e)[:200])
