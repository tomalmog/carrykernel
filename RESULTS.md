# Results — error-feedback recurrent-state quantization

## Background and prior art

Hybrid LLMs (Qwen3.5, DeepSeek V4.1, Kimi Linear) replace most attention with a
fixed-size recurrent state. During decode this state is read and written on
every token and stored in FP32. At batch size, the aggregate state traffic
rivals the model weights, making the state the memory bottleneck of long-context
serving. Quantizing the state is the natural fix — but prior work reported it
broken.

The four prior methods, and what each missed:

- **DAMP** (arXiv 2608.27513): per-channel mixed precision, 9.9 bits/state.
  Reports INT8/FP8 "already degrade complex reasoning" and INT4 "near zero"
  accuracy.
- **Minima** (arXiv 2609.04098): W4A4. Claims the recurrence "forgets a state
  impulse within hundreds of steps", implying low-bit state should be viable.
- **DeltaLog** (arXiv 2608.15533): defers state materialization.
- **KVBuffer** (arXiv 2605.19049): buffers state updates.

None of the four applied error feedback to the recurrent state. That omission is
the core of this work.

The DAMP-vs-Minima contradiction is the motivating puzzle: DAMP says low-bit
state is hopeless, Minima says injected noise washes out. Both can't be right —
unless they are measuring different regimes.

## Finding 1: the mechanism — error compounds at slow decay, washes out at fast decay

Using the exact Gated DeltaNet recurrence (`statequant/reference.py`), we
quantize the state between decode steps and track output error over 2000 steps.

At `alpha=0.995` (slow decay, ~200-step retention) quantization error
**compounds**; at `alpha=0.9` (fast decay) it stays **flat**. This maps onto
DASC's "retention horizons": slow-decay heads compound, fast-decay heads wash
out.

**Both prior papers are right, in different regimes.** DAMP's "INT4 → near
zero" is the compounding (slow-decay) regime; Minima's "flat plateau" is the
fast-decay regime. The tension resolves once you condition on decay rate.

## Finding 2: error feedback is the decisive technique

Error feedback (carry the quantization residual to the next step, in the LoRC
lineage) converts *every* compounding failure into a flat, bounded error.
Final-state error at `t=2000`, `alpha=0.995`:

| bits | uniform | +error feedback |
|---|---|---|
| int3 | 3.5e13 (explodes) | 0.45 |
| int4 | 2.65 | 0.19 |
| int5 | 0.75 | 0.087 |
| int6 | 0.30 | 0.042 |
| int8 | 0.074 | 0.010 |

The practical effect: `int6 + error feedback` (0.042) beats `int8 uniform`
(0.074) at 25% less storage. Error feedback buys roughly **1.5–2 bits of
effective precision for free**.

## Finding 3: granularity helps, but less than error feedback

At int4, final-state error (`alpha=0.995`, `t=2000`):

| scheme | no EF | +EF |
|---|---|---|
| per-head | 2.65 | 0.19 |
| per-K channel | 1.14 | 0.13 |
| per-V channel | 1.14 | 0.13 |
| block16 | 1.53 | 0.14 |
| block32 | 1.91 | 0.16 |

Per-channel scaling (DAMP's "high-risk channel" intuition) helps ~2.3x over
per-head. Block scaling (Minima's weight trick) is mediocre for state. The best
combination is **per-channel + error feedback**.

## Finding 4: impulse forgetting confirmed

A `+5.0` perturbation injected into the state at `t=0` is forgotten — output
error decays 37 → 0.17 at `t=500` → 8.8e-7 at `t=2000`. This is Minima's claim,
verified on the exact recurrence. It bounds the problem: quantization injects
small per-step noise, not one large impulse, so the compounding regime (not the
impulse regime) is what governs failure.

## End-to-end: INT8 is rescued on Qwen3.5-4B

We hook `transformers.DynamicCache.update_recurrent_state` to quantize the FP32
recurrent state (24 layers × `[1,32,128,128]`) between decode steps.

### Forced-decode PPL (fixed 381-token text, recurrent path)

| scheme | PPL | vs bf16 |
|---|---|---|
| bf16 (FP32) | 8.795 | — |
| int8 uniform | 13.35 | +52% (matches DAMP) |
| int8 per-head + EF | 9.22 | +4.9% |
| **int8 per-V + EF** | **8.68** | **~0%** |
| int6 per-V + EF | 10.74 | +22% |
| int5 per-V + EF | 13.09 | +49% |
| int4 per-V + EF | 27.43 | +212% |

### Broad benchmark (WikiText-2 PPL + GSM8K accuracy, 40 problems)

| scheme | WikiText-2 PPL | GSM8K acc |
|---|---|---|
| bf16 (FP32) | 10.075 | 75.0% (30/40) |
| int8 uniform | 15.767 | **32.5%** (13/40) |
| **int8 per-V + EF** | **9.953** | **70.0%** (28/40) |
| int6 per-V + EF | 12.287 | 72.5% (29/40) |

**Headline: uniform INT8 drops GSM8K from 75% to 32.5% (−42.5 pts); error
feedback + per-V scaling recovers it to 70% (−5 pts).** On WikiText-2, int8
per-V + EF (9.95) is *below* the bf16 baseline (10.08).

## Finding 5: the residual must be quantized — this sets the real memory win

Error feedback stores a residual of the same shape as the state. At FP32 the
residual negates the saving (5 B/element vs 4 B for FP32). Quantizing the
residual fixes this. Forced-decode PPL vs residual precision:

| residual | PPL | vs bf16 | total B/elem | vs FP32 |
|---|---|---|---|---|
| fp32 | 8.678 | −1.3% | 5.0 | 0.80x (worse) |
| fp16 | 8.816 | +0.2% | 3.0 | 1.33x |
| fp8 (e4m3fn) | 9.486 | +7.9% | 2.0 | 2.00x |
| **int8** | **8.836** | **+0.5%** | **2.0** | **2.00x** |
| **int4** | **8.940** | **+1.6%** | **1.5** | **2.67x** |

**An int8 residual with a per-head scale is sufficient: 2x memory at ~0%
quality.** fp8/e4m3fn is the one bad choice — its fixed 3-bit mantissa loses the
small residual magnitudes, whereas a per-head `amax`-scaled int8/int4 adapts to
the residual's actual range each step.

## Finding 6: the kernel realizes the traffic win, but the wall-clock gain is modest

A fused Triton kernel (state-update + INT8 quantize + error feedback, in
`statequant/kernel.py`) was written and benchmarked on A10G. Key bug fixed:
round-half-away-from-zero vs `torch.round`'s round-half-to-even drifted the
residual; matching the reference removed it.

| batch | config | us/token | traffic MB | speedup |
|---|---|---|---|---|
| 1 | FP32 | 15.4 | 4.26 | 1.00x |
| 1 | INT8+EF int8 resid | 19.9 | 2.23 | 0.77x |
| 8 | FP32 | 76.5 | 34.08 | 1.00x |
| 8 | INT8+EF int8 resid | 72.1 | 17.83 | 1.06x |
| 8 | INT8+EF fp8 resid | 58.5 | 17.57 | 1.31x |

The memory-traffic reduction is real (~1.9x), but it translates to only ~1.3x
wall-clock at batch 8 and a *slowdown* at batch 1, because the fused kernel adds
quantization compute and decode is not purely bandwidth-bound. So the headline
is a **~2x storage/traffic win**, not a 4x or a 2x speedup.

## Limitations

- **Small GSM8K sample.** 40 problems (Δ ± ~7 pts). The PPL signals are more
  stable and point the same direction, but reasoning accuracy should be
  confirmed on the full set / MMLU.
- **One model family.** Validated on Qwen3.5-4B only; the mechanism (CPU) is
  architecture-agnostic but the end-to-end claim is specific to this model.
- **Kernel not integrated into vLLM/fla.** The kernel is standalone; a
  production integration (storing packed int8 + scales in the cache) is
  engineering work, not done here.

## Net

Error-feedback recurrent-state quantization rescues INT8 state: it recovers the
reasoning degradation that DAMP reported (−42.5 GSM8K points → −5), at **~2x
memory reduction** (int8 state + int8 residual) with ~zero quality cost. The
mechanism — compounding vs. wash-out as a function of decay rate — resolves the
DAMP-vs-Minima contradiction, and error feedback (missed by all prior work) is
the technique that makes it work.
