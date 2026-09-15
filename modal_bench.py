"""Broad benchmark: WikiText-2 PPL + GSM8K accuracy for state-quantization schemes.

Runs on Modal. Schemes: bf16 (FP32 state), int8 uniform (DAMP's failure case),
int8-V+EF (our method), int6-V+EF (extension), int4-V+EF (bound).
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


def forced_ppl(model, tok, tokens, device, chunk=512):
    """Forced-decode NLL over a token list, in chunks through the recurrent path."""
    ids = tokens.to(device)
    cache = None
    total_ll, n = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(ids) - 1, chunk):
            seg = ids[start:min(start + chunk + 1, len(ids))]
            for i in range(len(seg) - 1):
                out = model(input_ids=seg[i:i + 1].unsqueeze(0), past_key_values=cache, use_cache=True)
                nll = F.cross_entropy(out.logits[0, -1].unsqueeze(0), seg[i + 1].unsqueeze(0), reduction="sum")
                total_ll += nll.item()
                n += 1
                cache = out.past_key_values
    return float(torch.exp(torch.tensor(total_ll / n))), n


def _strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def _normalize(s):
    s = s.strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return s


def extract_answer(text):
    text = _strip_thinking(text)
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        return boxes[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def extract_gold(text):
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1).strip()
    return extract_answer(text)


_DEBUG_SAMPLES = []


def gsm8k_acc(model, tok, questions, answers, device, max_new=512):
    correct = 0
    total = 0
    _DEBUG_SAMPLES.clear()
    with torch.no_grad():
        for q, a in zip(questions, answers):
            msgs = [{"role": "user", "content": q + " Please reason step by step and put your final answer in \\boxed{}."}]
            try:
                text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            except Exception:
                text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inp = tok(text, return_tensors="pt").input_ids.to(device)
            out = model.generate(inp, max_new_tokens=max_new, do_sample=False)
            gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
            pred = extract_answer(gen)
            gold = extract_gold(a)
            total += 1
            _DEBUG_SAMPLES.append((gold, pred))
            if pred is not None and gold is not None and _normalize(pred) == _normalize(gold):
                correct += 1
    return correct / total, correct, total


def run_benchmark():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    name = "Qwen/Qwen3.5-4B"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    device = model.device

    # WikiText-2 test tokens
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in wt["text"] if t.strip()]
    wt_tokens = tok(" ".join(texts[:3]), return_tensors="pt").input_ids[0][:2048]

    # GSM8K subset
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    N = 40
    gsm_q = gsm["question"][:N]
    gsm_a = gsm["answer"][:N]

    schemes = [
        ("bf16", None, "head", False),
        ("int8", 8, "head", False),
        ("int8-V+EF", 8, "V", True),
        ("int6-V+EF", 6, "V", True),
        ("int4-V+EF", 4, "V", True),
    ]
    results = {}
    for label, bits, gran, ef in schemes:
        install_hook(bits=bits, granularity=gran, use_ef=ef)
        ppl, n = forced_ppl(model, tok, wt_tokens, device)
        acc, c, t = gsm8k_acc(model, tok, gsm_q, gsm_a, device)
        results[label] = {"ppl": ppl, "gsm8k": acc, "n_tokens": n}
        print(f"{label:<12} WikiText2_PPL={ppl:.3f} (n={n})  GSM8K_acc={acc:.3f} ({c}/{t})")
        if label == "bf16":
            print("  sample (gold, pred):", _DEBUG_SAMPLES[:3])
    return results


import modal

app = modal.App("statequant-bench")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.7.1", "accelerate", "einops", "safetensors", "datasets",
                 "git+https://github.com/huggingface/transformers.git@main")
    .pip_install(
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/"
        "causal_conv1d-1.7.0+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
    )
    .pip_install("flash-linear-attention[cuda]")
)


@app.function(gpu="A10G", image=image, timeout=5400)
def run():
    return run_benchmark()


@app.local_entrypoint()
def main():
    print(run.remote())
