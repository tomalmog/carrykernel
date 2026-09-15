"""Fast diagnostic: 3 GSM8K problems, print raw output + timing, check thinking mode."""

import re, time
import torch


def run_diag():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    torch.set_grad_enabled(False)
    name = "Qwen/Qwen3.5-4B"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    device = model.device

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    for i in range(3):
        q, a = gsm["question"][i], gsm["answer"][i]
        msgs = [{"role": "user", "content": q + " Please reason step by step and put your final answer in \\boxed{}."}]
        # try both ways
        try:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            mode = "no-think"
        except Exception as e:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            mode = f"think(fallback:{type(e).__name__})"
        inp = tok(text, return_tensors="pt").input_ids.to(device)
        t0 = time.time()
        out = model.generate(inp, max_new_tokens=256, do_sample=False)
        dt = time.time() - t0
        gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
        print(f"--- problem {i} [{mode}] {dt:.1f}s  ntokens={out.shape[1]-inp.shape[1]} ---")
        print("GOLD:", a.strip()[-60:])
        print("GEN :", gen[-200:].replace("\n", " ")[:200])
        print()
    return "diag-done"


import modal

app = modal.App("carrykernel-diag")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch", "accelerate", "einops", "safetensors", "datasets",
                 "git+https://github.com/huggingface/transformers.git@main")
    .pip_install("flash-linear-attention[cuda]")
)


@app.function(gpu="H100", image=image, timeout=1800)
def run():
    return run_diag()


@app.local_entrypoint()
def main():
    print(run.remote())
