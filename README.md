# CarryKernel

Error-feedback recurrent-state quantization for hybrid (linear-attention) LLMs.

## The problem

Hybrid LLMs — Qwen3.5, DeepSeek V4.1, Kimi Linear — replace most attention
with a fixed-size recurrent state (a Gated DeltaNet-style matrix). At decode,
that state is read *and* written on every token, and it is stored in FP32. At
batch size the aggregate read/write traffic of this state rivals the model
weights themselves, making it the memory bottleneck of long-context serving.

The obvious fix — quantize the recurrent state to fewer bits — was reported
broken: DAMP claims INT8/FP8 "already degrade complex reasoning" and INT4
collapses to "near zero" accuracy. We find that the failure is not fundamental
but a specific kind of error, and that one technique — **error feedback** —
fixes it.

## The idea, in plain language

Every decode step quantizes the state and throws away a small rounding error.
In a slow-decay head that error is *re-fed into the next step* and compounds,
quietly corrupting long-horizon memory. The fix is to *keep* the rounding error
instead of discarding it: carry it to the next step, add it back before the next
quantization, and store the new residual. This is the classic error-feedback /
delta-sigma trick (known as LoRC in the LLM weight-quantization literature), and
none of the prior state-quantization papers used it on the recurrent state.

Result: the compounding error becomes a flat, bounded error — roughly 1.5–2 bits
of effective precision for free. Combined with per-channel (per-V) scaling, INT8
recurrent state is essentially lossless.

## Key results

### Mechanism (CPU, exact GDN recurrence)

At `alpha=0.995` (slow decay) quantization error compounds; at `alpha=0.9` (fast
decay) it washes out. Both DAMP and Minima were right, in different regimes.

Final-state error at `t=2000`, `alpha=0.995`:

| bits | uniform | +error feedback |
|---|---|---|
| int3 | 3.5e13 (explodes) | 0.45 |
| int4 | 2.65 | 0.19 |
| int5 | 0.75 | 0.087 |
| int6 | 0.30 | 0.042 |
| int8 | 0.074 | 0.010 |

### End-to-end (Qwen3.5-4B)

GSM8K + MMLU, 100 problems each:

| scheme | GSM8K | MMLU |
|---|---|---|
| bf16 (FP32) | 81.0% | 54.0% |
| int8 uniform | **41.0%** | 54.0% |
| **int8 per-V + EF** | **81.0%** | 54.0% |
| **int6 per-V + EF** | **81.0%** | 54.0% |

**Headline: uniform INT8 drops GSM8K 81% → 41%; error feedback + per-V scaling
recovers it completely, back to the bf16 baseline of 81%.** int6 + EF also
reaches 81% — so 6-bit state with error feedback beats 8-bit state without it,
at 25% less storage. The bit width is not what matters; the error feedback is.

MMLU is flat across all schemes — it is a single-token multiple-choice task that
barely exercises the recurrent state, so it acts as a control (quantization
doesn't break general knowledge) rather than evidence for the method. GSM8K
needs 300–500 tokens of sequential reasoning, which is where state-quantization
error compounds.

On the earlier 40-problem run, WikiText-2 PPL was 10.075 (bf16), 15.767 (int8
uniform) and 9.953 (int8 per-V + EF) — the EF variant scoring *below* the bf16
baseline.

### The real memory win (residual precision matters)

Error feedback stores a residual of the same shape as the state; at FP32 that
negates the saving. Quantizing the residual fixes it:

| residual | PPL | vs bf16 | memory vs FP32 |
|---|---|---|---|
| int8 (per-head scale) | 8.836 | +0.5% | **2.00x** |
| int4 (per-head scale) | 8.940 | +1.6% | **2.67x** |
| fp8 (e4m3fn) | 9.486 | +7.9% | 2.00x |
| fp16 | 8.816 | +0.2% | 1.33x |

So the honest win is **~2x memory (int8 state + int8 residual) at ~zero quality
cost** — not "4x free" (pure int8 without error feedback is 4x but drops GSM8K
to 41%).

### The baseline matters: 2x vs FP32, break-even vs bf16

That 2x is measured against an **FP32** state. Serving engines don't necessarily
store it that way — vLLM's GDN state defaults to the model activation dtype,
which is bf16 for Qwen3.5. Since `int8 state + int8 residual` costs 2 B/element,
exactly what bf16 costs, the comparison changes completely:

| state representation | bytes/slot | vs FP32 | vs bf16 |
|---|---|---|---|
| fp32 | 2.10 MB | 1.00x | — |
| bf16 | 1.05 MB | 2.00x | 1.00x |
| int8 + int8 residual + scales | 1.08 MB | **1.94x** | **0.97x** |

Against bf16 this scheme is marginally *worse* on memory. The quality result is
unaffected — INT8 state at bf16-level accuracy is still the contribution — but
an in-engine memory win needs a cheaper residual (int4, or amortizing one
residual across steps). See `RESULTS.md` Finding 8.

### Kernel (fused: Triton and raw CUDA/C++)

The fused state-update + INT8 quantize + error-feedback op is implemented twice
— in Triton (`statequant/kernel.py`) and in raw CUDA/C++
(`cuda/gdn_state_kernel.cu`) — both validated against the same oracle. The
traffic reduction is real (~1.9x); the wall-clock gain is modest (~1.3x at
batch 8), because decode here is not purely bandwidth-bound.

Three findings from the CUDA port, each settled by Nsight measurement (A10G,
484 GB/s measured peak):

- **Coalescing dominates.** One thread per V-column (V is the contiguous axis)
  instead of one warp per column took the FP32 baseline from 33% to 66% of peak
  bandwidth, 214 → 107 us/token at batch 8.
- **Register caching backfires.** Holding each thread's column in a
  `float[128]` halved the instruction count but ran 2x slower: nvcc spills it to
  local memory, and DRAM writes went 10.1 → 71.1 MB. Recompute beats spilled
  traffic.
- **Triton still wins the INT8 path** (72 vs 125 us/token at batch 8). The INT8
  kernel is compute-bound on the quantize passes — 10.2M instructions vs the
  FP32 baseline's 2.28M at similar traffic — not memory-bound.

See `RESULTS.md` Findings 6 and 7.

## Reproduce

### CPU experiments (pure PyTorch, no GPU needed)

```bash
python experiments/exp1_error_dynamics.py   # error compounds vs washes out
python experiments/exp2_bits_sweep.py       # INT-k floor, with/without EF
python experiments/exp2b_granularity.py     # granularity × error feedback
python experiments/test_kernel.py           # kernel torch-reference vs oracle
```

### GPU experiments (Modal, A10G; profile `tomalmog2`)

```bash
modal run modal_smoke.py      # smoke test: load model, find the state hook
modal run modal_eval.py       # forced-decode PPL on one text
modal run modal_bench.py      # broad: WikiText-2 PPL + GSM8K accuracy
modal run modal_residual.py   # residual-precision sweep
modal run modal_kernel.py     # fused-kernel benchmark
```

## File layout

```
statequant/
  reference.py           # exact Gated DeltaNet recurrence (bit-comparable to fla)
  quant.py               # uniform / per-channel / block quantizers + error feedback
  kernel.py              # fused Triton state-update + INT8 quantize + EF
experiments/
  exp1_error_dynamics.py # compounding vs wash-out (mechanism)
  exp2_bits_sweep.py     # INT-k floor + error feedback
  exp2b_granularity.py   # granularity × error feedback
  test_kernel.py         # kernel correctness vs oracle
  kernel_bench.py        # fused-kernel bandwidth benchmark
modal_smoke.py           # Modal smoke test (structure / hook discovery)
modal_eval.py            # Modal end-to-end INT8 result
modal_bench.py           # Modal broad benchmark (WikiText-2 + GSM8K)
modal_residual.py        # Modal residual-precision sweep
modal_kernel.py          # Modal fused-kernel benchmark
NOTES.md                 # working notes
RESULTS.md               # full writeup
```
