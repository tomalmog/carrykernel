"""Full benchmark: complete GSM8K (1319) + MMLU subset, state-quantization schemes.

Runs detached on Modal. Schemes: bf16, int8 (uniform), int8-V+EF, int6-V+EF.
This is the definitive accuracy table. Results are printed as they complete.
"""

import re
import torch
import torch.nn.functional as F


def _qmax(bits):
    return 2 ** (bits - 1) - 1


def quantize(x, bits, granularity):
    x32 = x.float()
    qmax = _qmax(bits)
    dim = (-1, -2) if granularity == "head" else (-1 if granularity == "V" else -2)
    scale = x32.abs().amax(dim=dim, keepdim=True).clamp(min=1e-12) / qmax
    q = torch.round(x32 / scale).clamp(-qmax, qmax)
    return (q * scale).to(x.dtype)


class StateQuant:
    def __init__(self, bits=None, granularity="head", use_ef=True):
        self.bits, self.granularity, self.use_ef = bits, granularity, use_ef
        self.residuals = {}

    def reset(self):
        self.residuals = {}

    def apply(self, state, layer_idx):
        if self.bits is None:
            return state
        if self.use_ef:
            e = self.residuals.get(layer_idx)
            if e is None or e.shape != state.shape or e.dtype != state.dtype:
                e = torch.zeros_like(state)
            target = state + e
            out = quantize(target, self.bits, self.granularity)
            self.residuals[layer_idx] = target - out
            return out
        return quantize(state, self.bits, self.granularity)


_sq = StateQuant()


def install_hook(bits=None, granularity="head", use_ef=True):
    from transformers.cache_utils import DynamicCache
    _sq.bits, _sq.granularity, _sq.use_ef = bits, granularity, use_ef
    _sq.reset()
    _orig = DynamicCache.update_recurrent_state

    def patched(self, state, *args, **kwargs):
        layer_idx = args[0] if args else None
        return _orig(self, _sq.apply(state, layer_idx), *args, **kwargs)

    DynamicCache.update_recurrent_state = patched


def _model_input(tok, text, device):
    return tok(text, return_tensors="pt").input_ids.to(device)


def extract_answer(text):
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        return boxes[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def gsm8k(model, tok, qs, answers, device, max_new=256):
    correct = total = 0
    for q, a in zip(qs, answers):
        msgs = [{"role": "user", "content": q + " Please reason step by step and put your final answer in \\boxed{}."}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inp = tok(text, return_tensors="pt").input_ids.to(device)
        out = model.generate(inp, max_new_tokens=max_new, do_sample=False)
        gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
        pred, gold = extract_answer(gen), extract_answer(a)
        total += 1
        if pred is not None and gold is not None and pred == gold:
            correct += 1
    return correct, total


def mmlu(model, tok, qs, opts, answers, device, max_new=16):
    letters = ["A", "B", "C", "D"]
    correct = total = 0
    for q, o, a in zip(qs, opts, answers):
        choices = "\n".join(f"{l}. {c}" for l, c in zip(letters, o))
        prompt = f"{q}\n{choices}\nAnswer with only the letter:\n"
        inp = tok(prompt, return_tensors="pt").input_ids.to(device)
        out = model.generate(inp, max_new_tokens=max_new, do_sample=False)
        gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
        m = re.search(r"[ABCD]", gen)
        total += 1
        if m and m.group(0) == a:
            correct += 1
    return correct, total


def run_benchmark():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    name = "Qwen/Qwen3.5-4B"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    device = model.device

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_q, gsm_a = gsm["question"], gsm["answer"]

    mmlu_ds = load_dataset("cais/mmlu", "all", split="test")
    # use a fixed subset (first 600) across 57 subjects for a stable, cheap signal
    mmlu_q = mmlu_ds["question"][:600]
    mmlu_opts = mmlu_ds["choices"][:600]
    mmlu_a = [int(x) for x in mmlu_ds["answer"][:600]]  # 0..3

    schemes = [
        ("bf16", None, "head", False),
        ("int8", 8, "head", False),
        ("int8-V+EF", 8, "V", True),
        ("int6-V+EF", 6, "V", True),
    ]
    results = {}
    for label, bits, gran, ef in schemes:
        install_hook(bits=bits, granularity=gran, use_ef=ef)
        gc, gt = gsm8k(model, tok, gsm_q, gsm_a, device)
        mc, mt = mmlu(model, tok, mmlu_q, mmlu_opts, mmlu_a, device)
        results[label] = {"gsm8k": gc / gt, "gsm8k_n": gt, "mmlu": mc / mt, "mmlu_n": mt}
        print(f"{label:<12} GSM8K={gc}/{gt} ({gc/gt:.3f})  MMLU={mc}/{mt} ({mc/mt:.3f})", flush=True)
    return results


import modal

app = modal.App("statequant-fullbench")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch", "accelerate", "einops", "safetensors", "datasets",
                 "git+https://github.com/huggingface/transformers.git@main")
    .pip_install("flash-linear-attention[cuda]")
)


@app.function(gpu="H100", image=image, timeout=28800)
def run():
    return run_benchmark()


@app.local_entrypoint()
def main():
    print(run.remote())
