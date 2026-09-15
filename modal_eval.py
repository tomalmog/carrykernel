"""Modal GPU experiment: error-feedback recurrent-state quantization on Qwen3.5-4B.

Hooks transformers' DynamicCache.update_recurrent_state to quantize the GDN
recurrent state between decode steps, then measures end-to-end quality
(perplexity + token-match vs the bf16 baseline) under autoregressive generation.

Modes: bf16 (baseline), int8 (DAMP's "degrades complex reasoning" ref), int4,
int5, int6 — each with error feedback, plus int4-noef as the DAMP failure case.
"""

import torch
import torch.nn.functional as F

# ---------------- quantization (mirrors statequant.quant) ----------------

def _qmax(bits):
    return 2 ** (bits - 1) - 1


def quantize_per_head(x, bits):
    """x: [..., K, V]. One symmetric scale per head (last two dims)."""
    x32 = x.float()
    qmax = _qmax(bits)
    scale = x32.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-12) / qmax
    q = torch.round(x32 / scale).clamp(-qmax, qmax)
    return (q * scale).to(x.dtype)


def quantize(x, bits, granularity):
    """granularity: 'head' (last2 dims), 'K' (dim -2), 'V' (dim -1)."""
    x32 = x.float()
    qmax = _qmax(bits)
    if granularity == "head":
        dim = (-1, -2)
    elif granularity == "K":
        dim = -2
    elif granularity == "V":
        dim = -1
    else:
        raise ValueError(granularity)
    scale = x32.abs().amax(dim=dim, keepdim=True).clamp(min=1e-12) / qmax
    q = torch.round(x32 / scale).clamp(-qmax, qmax)
    return (q * scale).to(x.dtype)


# ---------------- hook ----------------

class StateQuant:
    def __init__(self, bits=None, granularity="head", use_ef=True, ef_decay=1.0):
        self.bits = bits
        self.granularity = granularity
        self.use_ef = use_ef
        self.ef_decay = ef_decay
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
            else:
                e = e * self.ef_decay
            target = state + e
            out = quantize(target, self.bits, self.granularity)
            self.residuals[layer_idx] = target - out
            return out
        return quantize(state, self.bits, self.granularity)


_statequant = StateQuant()


def install_hook(bits=None, granularity="head", use_ef=True, ef_decay=1.0):
    from transformers.cache_utils import DynamicCache
    _statequant.bits = bits
    _statequant.granularity = granularity
    _statequant.use_ef = use_ef
    _statequant.ef_decay = ef_decay
    _statequant.reset()

    _orig = DynamicCache.update_recurrent_state

    def patched(self, state, *args, **kwargs):
        layer_idx = args[0] if args else None
        state = _statequant.apply(state, layer_idx)
        return _orig(self, state, *args, **kwargs)

    DynamicCache.update_recurrent_state = patched
    return _orig


# ---------------- eval ----------------

PROMPTS = [
    "What is the capital of France?",
    "Solve: 12 * 8 + 34",
    "Write a Python function to compute the nth Fibonacci number.",
    "Explain what a p-value means in one paragraph.",
    "If a train travels 60 mph for 2 hours, how far does it go?",
    "What is the derivative of x^3?",
    "List the first five prime numbers.",
    "Explain the difference between a list and a tuple in Python.",
    "What is 2^10?",
    "Name three countries in South America.",
    "Convert the binary number 1011 to decimal.",
    "What is the chemical symbol for gold?",
    "Explain recursion with an example.",
    "If a^2 + b^2 = c^2, what theorem is this?",
    "Write a haiku about winter.",
    "What is the largest planet in our solar system?",
    "Explain Bayes' theorem in simple terms.",
    "What is the time complexity of binary search?",
    "Give three examples of renewable energy sources.",
    "What does the acronym 'API' stand for?",
]


REF_TEXT = (
    "Machine learning systems have transformed how software is built and deployed. "
    "At the heart of modern language models lies the attention mechanism, which weighs "
    "the relevance of every token in a sequence against every other token. This quadratic "
    "cost becomes prohibitive as sequences grow to hundreds of thousands of tokens, so "
    "researchers have sought subquadratic alternatives. Linear attention replaces the "
    "explicit pairwise comparisons with a fixed-size recurrent state that summarizes the "
    "entire context. Each new token updates this state through a gated write operation, "
    "and a query reads from the state to produce the next prediction. The advantage is "
    "clear: memory no longer grows with sequence length, and decoding becomes constant "
    "time per token. The difficulty is that the state must retain enough information to "
    "answer questions about tokens seen long ago. Gated delta networks address this by "
    "combining an exponential decay with a delta rule that subtracts the existing read "
    "before writing, which reduces interference between similar keys. The result is a "
    "family of architectures that can recall distant facts while staying efficient. "
    "Serving these models introduces a new bottleneck: the recurrent state is large and "
    "must be read and written on every decoding step. At high batch sizes the aggregate "
    "memory traffic rivals that of the model weights themselves. Quantizing the state to "
    "fewer bits would reduce this traffic, but the accumulated rounding error may corrupt "
    "the very information the state is meant to preserve. The question is whether this "
    "error compounds over long contexts or washes out through the natural decay of the "
    "recurrence, and whether error feedback can keep it bounded. Understanding this trade "
    "off is essential for building efficient inference engines for the next generation of "
    "models. Beyond language, the same principles apply to audio, vision, and time series, "
    "where sequences are the fundamental data structure. A rigorous treatment requires "
    "careful measurement rather than intuition, because small numerical effects can grow "
    "into large behavioral differences over thousands of steps. The goal is not merely to "
    "compress memory but to preserve the model's reasoning ability while doing so."
)


def forced_nll(model, tok, text, device, max_len=None):
    """Teacher-forced NLL over a fixed text, through the recurrent decode path.

    Feeds one token at a time with an evolving cache so seq_len==1 after the first
    step, exercising the recurrent (stateful) branch with the quantized state.
    """
    ids = tok(text, return_tensors="pt").input_ids[0].to(device)
    if max_len is not None:
        ids = ids[:max_len]
    cache = None
    total_ll = 0.0
    n = 0
    for i in range(len(ids) - 1):
        out = model(input_ids=ids[i:i + 1].unsqueeze(0), past_key_values=cache, use_cache=True)
        logits = out.logits[0, -1]
        nll = F.cross_entropy(logits.unsqueeze(0), ids[i + 1].unsqueeze(0), reduction="sum")
        total_ll += nll.item()
        n += 1
        cache = out.past_key_values
    return torch.exp(torch.tensor(total_ll / n)).item(), n


def run_mode(model, tok, text, device, bits, use_ef, granularity="head", ef_decay=1.0):
    install_hook(bits=bits, granularity=granularity, use_ef=use_ef, ef_decay=ef_decay)
    model.eval()
    with torch.no_grad():
        ppl, n = forced_nll(model, tok, text, device)
    return ppl, n


def eval_gpu():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    name = "Qwen/Qwen3.5-4B"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    device = model.device
    print("state dtype/shape probe: will inspect on first baseline run")

    # inspect the recurrent state dtype/shape once
    from transformers.cache_utils import DynamicCache
    _orig = DynamicCache.update_recurrent_state

    seen = {}

    def probe(self, state, *args, **kwargs):
        layer_idx = args[0] if args else None
        if layer_idx not in seen:
            seen[layer_idx] = (tuple(state.shape), state.dtype)
        return _orig(self, state, *args, **kwargs)

    DynamicCache.update_recurrent_state = probe
    inp = tok(REF_TEXT[:50], return_tensors="pt").to(device)
    model.generate(**inp, max_new_tokens=4, do_sample=False)
    print("recurrent state layers seen:", len(seen))
    for k in sorted(seen)[:3]:
        print("  layer", k, "shape", seen[k][0], "dtype", seen[k][1])
    DynamicCache.update_recurrent_state = _orig

    results = {}
    configs = [
        ("bf16", None, False, "head", 1.0),
        ("int8", 8, False, "head", 1.0),
        ("int8+EF", 8, True, "head", 1.0),
        ("int8-V+EF", 8, True, "V", 1.0),
        ("int8-K+EF", 8, True, "K", 1.0),
        ("int6+EF", 6, True, "head", 1.0),
        ("int6-V+EF", 6, True, "V", 1.0),
        ("int6-V+EFdecay", 6, True, "V", 0.5),
        ("int5-V+EF", 5, True, "V", 1.0),
        ("int4-V+EF", 4, True, "V", 1.0),
    ]
    for label, bits, ef, gran, dec in configs:
        ppl, n = run_mode(model, tok, REF_TEXT, device, bits, ef, gran, dec)
        results[label] = {"ppl": ppl, "tokens": n}
        print(f"{label:<16} PPL={ppl:.3f}  (n={n})")
    return results


import modal

app = modal.App("statequant-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch", "accelerate", "einops", "safetensors",
                 "git+https://github.com/huggingface/transformers.git@main")
    .pip_install("flash-linear-attention[cuda]")
)


@app.function(gpu="A10G", image=image, timeout=3600)
def run():
    return eval_gpu()


@app.local_entrypoint()
def main():
    print(run.remote())
