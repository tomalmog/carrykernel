"""Modal GPU experiment: what residual precision does error feedback actually need?

Quantizes the GDN recurrent state (INT8, per-V-channel) between decode steps with
error feedback, but the carried residual `e` itself is stored at a chosen precision
(fp32 / fp16 / fp8 / int8 / int4). Measures forced-decode PPL to find the residual
precision that preserves quality, which determines the real memory win.

Schemes (state always INT8 for the EF rows):
  - bf16:                 baseline, no quantization (4 B/elem)
  - int8-uniform (no EF): DAMP's uniform per-head baseline
  - int8-V+EF fp32/fp16/fp8/int8/int4: per-V state quant + error feedback with the
    residual stored at the given precision.
"""

import torch
import torch.nn.functional as F


# ---------------- quantization ----------------

def _qmax(bits):
    return 2 ** (bits - 1) - 1


def quantize_state(x, bits, granularity):
    """Symmetric quantization of the state, returns dequantized fp32."""
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
    return q * scale


def quantize_residual(e, precision):
    """Quantize the error-feedback residual to a chosen precision (fp32 out)."""
    e32 = e.float()
    if precision == "fp32":
        return e32
    if precision == "fp16":
        return e32.half().float()
    if precision == "fp8":
        return e32.to(torch.float8_e4m3fn).float()
    if precision in ("int8", "int4"):
        qmax = 127 if precision == "int8" else 7
        scale = e32.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-12) / qmax
        q = torch.round(e32 / scale).clamp(-qmax, qmax)
        return q * scale
    raise ValueError(precision)


# ---------------- hook ----------------

class StateQuant:
    def __init__(self, bits=None, granularity="head", use_ef=True,
                 residual_precision="fp32", ef_decay=1.0):
        self.bits = bits
        self.granularity = granularity
        self.use_ef = use_ef
        self.residual_precision = residual_precision
        self.ef_decay = ef_decay
        self.residuals = {}

    def reset(self):
        self.residuals = {}

    def apply(self, state, layer_idx):
        if self.bits is None:
            return state
        if self.use_ef:
            e = self.residuals.get(layer_idx)
            if e is None or e.shape != state.shape:
                e = torch.zeros(state.shape, dtype=torch.float32, device=state.device)
            else:
                e = e * self.ef_decay
            target = state.float() + e
            out = quantize_state(target, self.bits, self.granularity)
            resid = target - out
            self.residuals[layer_idx] = quantize_residual(resid, self.residual_precision)
            return out.to(state.dtype)
        return quantize_state(state, self.bits, self.granularity).to(state.dtype)


_statequant = StateQuant()


def install_hook(bits=None, granularity="head", use_ef=True,
                 residual_precision="fp32", ef_decay=1.0):
    from transformers.cache_utils import DynamicCache
    _statequant.bits = bits
    _statequant.granularity = granularity
    _statequant.use_ef = use_ef
    _statequant.residual_precision = residual_precision
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


def run_mode(model, tok, text, device, bits, use_ef, granularity="head",
             residual_precision="fp32", ef_decay=1.0):
    install_hook(bits=bits, granularity=granularity, use_ef=use_ef,
                 residual_precision=residual_precision, ef_decay=ef_decay)
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

    configs = [
        ("bf16",                 None, False, "head", None,  1.0),
        ("int8-uniform (no EF)", 8,    False, "head", None,  1.0),
        ("int8-V+EF fp32",       8,    True,  "V",    "fp32", 1.0),
        ("int8-V+EF fp16",       8,    True,  "V",    "fp16", 1.0),
        ("int8-V+EF fp8",        8,    True,  "V",    "fp8",  1.0),
        ("int8-V+EF int8",       8,    True,  "V",    "int8", 1.0),
        ("int8-V+EF int4",       8,    True,  "V",    "int4", 1.0),
    ]

    residual_bytes = {"fp32": 4, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5, None: 0}

    results = []
    print("=" * 78)
    print(f"{'scheme':<20} {'PPL':>8} {'state B':>8} {'resid B':>8} {'total B':>8} {'compress':>9}")
    print("=" * 78)
    for label, bits, ef, gran, rp, dec in configs:
        ppl, n = run_mode(model, tok, REF_TEXT, device, bits, ef, gran, rp, dec)
        state_b = 4 if bits is None else 1
        resid_b = 0 if (bits is None or not ef) else residual_bytes[rp]
        total_b = state_b + resid_b
        comp = 4.0 / total_b if total_b > 0 else float("inf")
        results.append((label, ppl, state_b, resid_b, total_b, comp, n))
        print(f"{label:<20} {ppl:>8.3f} {state_b:>8} {resid_b:>8.1f} {total_b:>8.1f} {comp:>8.2f}x   (n={n})")
    print("=" * 78)
    return [dict(label=r[0], ppl=r[1], state_b=r[2], resid_b=r[3],
                 total_b=r[4], comp=r[5], n=r[6]) for r in results]


import modal

app = modal.App("statequant-residual")

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
