"""End-to-end quality of the int4 error-feedback residual (GSM8K + MMLU, 100 each).

The 100-problem benchmark in `modal_bench_short.py` carried the error-feedback
residual in fp32. That is not the configuration the memory claim rests on: an
fp32 residual costs 4 B/element and makes the whole scheme *worse* than the
baseline. The configuration that actually wins in a serving engine is an
**int4-packed residual** (1.5 B/element total -> 1.28x vs bf16, measured in
`vllm_integration/`), and that configuration had no downstream quality number --
only a kernel-level error figure and an older per-head PPL data point.

This run closes that gap. Four arms, so the int4 delta is measured against this
run's own baselines rather than across runs:

  bf16                 no quantization (reference)
  int8-V+EF fp32resid  the configuration the published 81% came from
  int8-V+EF int8resid  2 B/elem -- break-even vs bf16
  int8-V+EF int4resid  1.5 B/elem -- the configuration that wins

Residual scaling note: the residual is scaled **per-V-channel**, matching
`vllm_integration/quantized_packed_decode.py` (which reduces amax over K, one
scale per V-row) and the per-V state quantization used throughout. This differs
from `modal_residual.py`, whose residual sweep used a *per-head* residual scale
(`dim=(-1,-2)`) -- so the +1.6% int4 PPL figure in RESULTS.md is not directly
comparable to the numbers this produces.

Extraction is copied verbatim from `modal_bench_short.py` (thinking mode
disabled and `<think>` stripped, gold from the `#### N` marker, 512 new tokens).
Those are the three gotchas that silently tank GSM8K to ~1%; do not "simplify"
them.

Usage (long job -- run detached):
    nohup modal run --detach modal_bench_int4resid.py > /tmp/int4resid.log 2>&1 &
"""

import re

import torch


def _qmax(bits):
    return 2 ** (bits - 1) - 1


def quantize_state(x, bits):
    """Per-V-channel symmetric quantization of the [.., K, V] state."""
    x32 = x.float()
    qmax = _qmax(bits)
    scale = x32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
    q = torch.round(x32 / scale).clamp(-qmax, qmax)
    return q * scale


def quantize_residual(e, precision):
    """Quantize the carried residual, per-V-channel (matches the vLLM kernel)."""
    e32 = e.float()
    if precision == "fp32":
        return e32
    if precision in ("int8", "int4"):
        qmax = _qmax(8) if precision == "int8" else _qmax(4)
        scale = e32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
        q = torch.round(e32 / scale).clamp(-qmax, qmax)
        return q * scale
    raise ValueError(precision)


class StateQuant:
    def __init__(self, bits=None, use_ef=True, residual_precision="fp32"):
        self.bits = bits
        self.use_ef = use_ef
        self.residual_precision = residual_precision
        self.residuals = {}

    def reset(self):
        self.residuals = {}

    def apply(self, state, layer_idx):
        if self.bits is None:
            return state
        if not self.use_ef:
            return quantize_state(state, self.bits).to(state.dtype)
        e = self.residuals.get(layer_idx)
        if e is None or e.shape != state.shape:
            e = torch.zeros(state.shape, dtype=torch.float32, device=state.device)
        target = state.float() + e
        out = quantize_state(target, self.bits)
        self.residuals[layer_idx] = quantize_residual(target - out,
                                                      self.residual_precision)
        return out.to(state.dtype)


_sq = StateQuant()


def install_hook(bits=None, use_ef=True, residual_precision="fp32"):
    from transformers.cache_utils import DynamicCache
    _sq.bits, _sq.use_ef = bits, use_ef
    _sq.residual_precision = residual_precision
    _sq.reset()
    _orig = getattr(DynamicCache, "_carrykernel_orig_update",
                    DynamicCache.update_recurrent_state)
    DynamicCache._carrykernel_orig_update = _orig

    def patched(self, state, *args, **kwargs):
        layer_idx = args[0] if args else None
        return _orig(self, _sq.apply(state, layer_idx), *args, **kwargs)

    DynamicCache.update_recurrent_state = patched


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
    return m.group(1).strip() if m else extract_answer(text)


def _chat(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except Exception:
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False)


def gsm8k(model, tok, qs, answers, device, label, max_new=512):
    correct = total = 0
    for i, (q, a) in enumerate(zip(qs, answers)):
        msgs = [{"role": "user", "content": q + " Please reason step by step and "
                 "put your final answer in \\boxed{}."}]
        inp = tok(_chat(tok, msgs), return_tensors="pt").input_ids.to(device)
        out = model.generate(inp, max_new_tokens=max_new, do_sample=False)
        gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
        pred, gold = extract_answer(gen), extract_gold(a)
        total += 1
        if pred is not None and gold is not None and _normalize(pred) == _normalize(gold):
            correct += 1
        if (i + 1) % 25 == 0:
            print(f"  [{label}] gsm8k {i+1}/{len(qs)}  correct={correct}/{total}",
                  flush=True)
    return correct, total


def mmlu(model, tok, qs, opts, answers, device, label, max_new=16):
    letters = "ABCD"
    correct = total = 0
    for i, (q, o, a) in enumerate(zip(qs, opts, answers)):
        choices = "\n".join(f"{l}. {c}" for l, c in zip(letters, o))
        msgs = [{"role": "user",
                 "content": f"{q}\n{choices}\nAnswer with only the letter:"}]
        inp = tok(_chat(tok, msgs), return_tensors="pt").input_ids.to(device)
        out = model.generate(inp, max_new_tokens=max_new, do_sample=False)
        gen = _strip_thinking(tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True))
        m = re.search(r"[ABCD]", gen)
        total += 1
        if m and m.group(0) == letters[int(a)]:
            correct += 1
        if (i + 1) % 50 == 0:
            print(f"  [{label}] mmlu {i+1}/{len(qs)}  correct={correct}/{total}",
                  flush=True)
    return correct, total


def run_benchmark(n_gsm=100, n_mmlu=100):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    name = "Qwen/Qwen3.5-4B"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    device = model.device

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_q, gsm_a = gsm["question"][:n_gsm], gsm["answer"][:n_gsm]

    mmlu_ds = load_dataset("cais/mmlu", "all", split="test")
    mmlu_q = mmlu_ds["question"][:n_mmlu]
    mmlu_opts = mmlu_ds["choices"][:n_mmlu]
    mmlu_a = [int(x) for x in mmlu_ds["answer"][:n_mmlu]]

    # (label, bits, use_ef, residual_precision)
    schemes = [
        ("bf16",                None, False, None),
        ("int8-V+EF fp32resid", 8,    True,  "fp32"),
        ("int8-V+EF int8resid", 8,    True,  "int8"),
        ("int8-V+EF int4resid", 8,    True,  "int4"),
    ]

    results = {}
    for label, bits, ef, rp in schemes:
        install_hook(bits=bits, use_ef=ef, residual_precision=rp)
        gc_, gt = gsm8k(model, tok, gsm_q, gsm_a, device, label)
        mc, mt = mmlu(model, tok, mmlu_q, mmlu_opts, mmlu_a, device, label)
        results[label] = {"gsm8k": gc_ / gt, "gsm8k_n": gt,
                          "mmlu": mc / mt, "mmlu_n": mt}
        print(f"RESULT {label:<22} GSM8K={gc_}/{gt} ({gc_/gt:.3f})  "
              f"MMLU={mc}/{mt} ({mc/mt:.3f})", flush=True)
    return results


import modal

app = modal.App("carrykernel-int4resid")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch", "accelerate", "einops", "safetensors", "datasets",
                 "git+https://github.com/huggingface/transformers.git@main")
    .pip_install("flash-linear-attention[cuda]")
)


@app.function(gpu="H100", image=image, timeout=14400)
def run(n_gsm: int = 100, n_mmlu: int = 100):
    return run_benchmark(n_gsm=n_gsm, n_mmlu=n_mmlu)


@app.local_entrypoint()
def main(n_gsm: int = 100, n_mmlu: int = 100):
    print(run.remote(n_gsm=n_gsm, n_mmlu=n_mmlu))
