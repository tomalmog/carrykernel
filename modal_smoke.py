"""Modal smoke test: load Qwen3.5-4B, inspect structure, find the GDN state hook.

This is a *cheap* diagnostic run. It loads the model, prints the layer types and
cache class so we know exactly where the recurrent state lives, then runs a
2-token generation to confirm the decode path works. No benchmarking yet.
"""

import modal

app = modal.App("statequant-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch",
        "accelerate",
        "einops",
        "safetensors",
        "git+https://github.com/huggingface/transformers.git@main",
    )
    .pip_install("flash-linear-attention[cuda]")
)


@app.function(gpu="A10G", image=image, timeout=1800)
def smoke():
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen3.5-4B"

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    print("tokenizer ok, vocab", tok.vocab_size)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    except Exception as e:
        print("AutoModelForCausalLM failed:", repr(e)[:300])
        from transformers import AutoModelForMultimodalLM
        model = AutoModelForMultimodalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    print("model class:", type(model).__module__, type(model).__name__)

    # enumerate layers
    lm = model.model if hasattr(model, "model") else model
    layers = lm.layers if hasattr(lm, "layers") else getattr(lm, "model", None)
    print("num layers:", len(layers))
    from collections import Counter
    c = Counter(type(l).__name__ for l in layers)
    print("layer types:", dict(c))

    # print one GDN-ish layer's submodules
    for i, l in enumerate(layers):
        print(i, type(l).__name__, [n for n, _ in l.named_modules()][:6])
        if i >= 2:
            break

    # try a 2-token generation
    inp = tok("Hello world", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=2, do_sample=False)
    print("gen ok:", tok.decode(out[0]))

    # inspect cache class used during generation
    import inspect
    try:
        # peek at what generate passes as past_key_values
        print("=== generate signature ===")
        print(str(inspect.signature(model.forward))[:400])
    except Exception as e:
        print("sig err", e)

    return "smoke-done"


@app.local_entrypoint()
def main():
    print(smoke.remote())
