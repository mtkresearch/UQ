import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
PROMPT = "Q: What is the capital of France?\nA:"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        forward_outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

    # Direct forward pass: final layer, last prompt token.
    forward_tbg = forward_outputs.hidden_states[-1][:, -1, :].detach().cpu()

    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    hidden = gen_outputs.hidden_states

    # Generation hidden[0]: final layer, last sequence token.
    generate_hidden0 = hidden[0][-1][:, -1, :].detach().cpu()

    print("Prompt:")
    print(PROMPT)

    print("\nGenerated text:")
    print(tokenizer.decode(gen_outputs.sequences[0], skip_special_tokens=True))

    print("\nShapes:")
    print("forward_tbg     :", forward_tbg.shape)
    print("generate hidden0:", generate_hidden0.shape)

    print("\nComparison:")
    print("allclose :", torch.allclose(forward_tbg, generate_hidden0, atol=1e-4, rtol=1e-4))
    print("max diff :", (forward_tbg - generate_hidden0).abs().max().item())
    print("mean diff:", (forward_tbg - generate_hidden0).abs().mean().item())

    print("\nHidden steps:")
    for i, step in enumerate(hidden):
        print(f"hidden[{i}] has {len(step)} layers; final layer shape = {step[-1].shape}")


if __name__ == "__main__":
    main()