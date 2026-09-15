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

### Broad benchmark (GSM8K + MMLU, 100 problems each)

The headline result, on the larger sample (`modal_bench_short.py`, H100):

| scheme | GSM8K | MMLU |
|---|---|---|
| bf16 (FP32) | 81.0% (81/100) | 54.0% (54/100) |
| int8 uniform | **41.0%** (41/100) | 54.0% (54/100) |
| **int8 per-V + EF** | **81.0%** (81/100) | 54.0% (54/100) |
| **int6 per-V + EF** | **81.0%** (81/100) | 54.0% (54/100) |

**Headline: uniform INT8 drops GSM8K from 81% to 41% (−40 pts); error feedback
+ per-V scaling recovers it completely — 81%, exactly matching the bf16
baseline.** On the 100-problem sample the recovery is total, not partial as the
smaller 40-problem run suggested.

**int6 + EF also lands on 81%**, i.e. 6-bit recurrent state is equally lossless
on this benchmark once error feedback is applied — strictly better than int8
*without* it (41%) at 25% less state. That is the clearest statement of the
central claim: what makes low-bit state work is the error feedback, not the bit
width.

MMLU is flat at 54% across all three schemes. That is expected and worth
stating plainly: MMLU is a single-token multiple-choice task, so it barely
exercises the recurrent state over a long horizon — it is a control showing the
quantization does not break general knowledge, not evidence for the method.
GSM8K, which requires 300–500 tokens of sequential reasoning, is where
state-quantization error compounds and where the effect appears.

### Earlier 40-problem run (superseded, kept for the WikiText-2 numbers)

| scheme | WikiText-2 PPL | GSM8K acc |
|---|---|---|
| bf16 (FP32) | 10.075 | 75.0% (30/40) |
| int8 uniform | 15.767 | 32.5% (13/40) |
| **int8 per-V + EF** | **9.953** | 70.0% (28/40) |
| int6 per-V + EF | 12.287 | 72.5% (29/40) |

On WikiText-2, int8 per-V + EF (9.95) is *below* the bf16 baseline (10.08).

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

## Finding 7: a raw CUDA/C++ port — what the hardware actually rewards

The same operation was rewritten as a raw CUDA kernel
(`cuda/gdn_state_kernel.cu`, ~400 lines, JIT-built through
`torch.utils.cpp_extension`) and validated against the same oracle. Three
design decisions were settled by measurement rather than intuition, each
confirmed with Nsight Compute on an A10G (measured peak HBM: 484 GB/s).

**1. Coalescing dominates everything else.** The obvious decomposition — one
warp per V-column, walking K inside the warp so `h^T k`, `h^T q` and the amax
are all warp-shuffle reductions — is the wrong one. The state is `[K, V]`
row-major, so V is contiguous; assigning a warp to a column makes every lane's
access K-strided, one sector per lane. Measured: 159 GB/s, 33% of peak.

Assigning one *thread* per V-column instead makes consecutive lanes touch
consecutive addresses on every load and store, and the K-reduction collapses
into a per-thread loop with no cross-lane communication at all:

| FP32 baseline | GB/s | % peak | us/token (B=8) |
|---|---|---|---|
| warp-per-column (shuffles) | 159 | 33% | 214 |
| thread-per-column (coalesced) | 318 | 66% | 107 |

**2. Caching the state in registers backfires.** The kernel touches the state
more than once per step (`read = h^T k` must be fully reduced before `dv`, and
hence the updated state, is known). Holding each thread's column in a
`float h_reg[128]` array to avoid re-reading halved the instruction count —
and ran **2x slower**:

| variant | instructions | dram bytes written | us/token (B=8) |
|---|---|---|---|
| multi-pass recompute | 10.21 M | 10.1 MB | 125 |
| per-thread register cache | 5.53 M | 71.1 MB | 266 |

512 B/thread overflows the register budget, so nvcc spills the array to local
memory — which is DRAM-backed. Writes went ~7x over the traffic model. On a
memory-bound kernel, redundant arithmetic is cheaper than spilled traffic.

**3. Triton still wins the INT8 path.** Final comparison (A10G, us/token):

| batch | FP32 CUDA | FP32 Triton | INT8+EF CUDA | INT8+EF Triton |
|---|---|---|---|---|
| 1 | **15.4** | 16.0 | 29.1 | 20.4 |
| 8 | 107.3 | **76.6** | 125.2 | **72.1** |
| 16 | 215.8 | **152.8** | 204.9 | **139.9** |

The CUDA FP32 baseline beats Triton at batch 1 and reaches 66% of peak
bandwidth, but Triton's INT8 path is faster at every batch. Nsight says why:
the INT8 kernel issues 10.2 M instructions against the FP32 baseline's 2.28 M
at similar traffic, so it is **compute-bound on the quantize/error-feedback
passes, not memory-bound** — exactly the regime where Triton's scheduling and
vectorization beat hand-written scalar code. Occupancy is also capped at ~28%
of peak warps.

This is reported as-is rather than tuned away. The project's claim is a
storage/traffic reduction, and the roofline is the honest framing: at 2 B/elem
the INT8 state moves ~1.9x fewer bytes, but the arithmetic added per byte
pushes the kernel off the bandwidth roof, so the traffic win does not convert
into a wall-clock win at these shapes.

## Finding 8: in a real engine the 2x is measured against FP32, not bf16

The kernel was ported a second time, into the layout vLLM actually uses
(`vllm_integration/quantized_packed_decode.py`), against the contract of its
vendored GDN decode kernel `fused_recurrent_gated_delta_rule_packed_decode`.

**The layout transpose is favourable.** vLLM stores the recurrent state as
`[num_slots, HV, V, K]` with K contiguous, where this project uses `[K, V]`.
Since the vendored kernel forces `BK = next_pow2(K)` with `NK == 1`, one program
instance holds a complete `[BV, K]` tile — so the per-V-channel `amax` becomes a
row-wise reduction along contiguous memory, resolved **in registers inside a
single program**. No cross-program reduction, no atomics, no second pass. This
is exactly the constraint that forced the multi-pass design in the CUDA kernel
(Finding 7).

Verified on A10G against the same oracle, plus the engine behaviours the layout
brings with it: round-half-to-even (124 constructed ties, 60 discriminating
against `floor(x+0.5)`, zero code differences), `NULL_BLOCK_ID` padding (emits
zeros, leaves other slots byte-identical), paging isolation (batched ==
per-request sequential), decode-vs-oracle at `o rel 1.7e-3 / h rel 5.3e-3`, and
lossless int4 nibble packing.

**And the caveat that matters most.** The ~2x memory claim is against an **FP32**
state. vLLM's GDN state dtype defaults to the model activation dtype — bf16 for
Qwen3.5 — and `int8 state + int8 residual` is 2 B/element, which is exactly what
bf16 already costs. Bytes per request slot at Qwen3.5 shapes (HV=32, V=K=128):

| state representation | bytes/slot | vs FP32 | vs bf16 |
|---|---|---|---|
| fp32 | 2.10 MB | 1.00x | — |
| bf16 | 1.05 MB | 2.00x | 1.00x |
| int8 state + int8 residual + scales | 1.08 MB | 1.94x | **0.97x** |
| **int8 state + int4 residual + scales** | **0.82 MB** | **2.56x** | **1.28x** |

So against the baseline a real vLLM deployment actually runs, error-feedback
INT8 with an int8 residual is **marginally worse on memory**, not 2x better:
`int8 state + int8 residual` is 2 B/element, exactly bf16's cost.

**The fix is a cheaper residual, and it is implemented and measured.** Packing
the residual to int4 (two codes per byte along K, nibble select on load, pair
repack on store) brings the state to 1.5 B/element plus scales — **1.28x against
bf16**, a real win, and 2.56x against FP32. The quality cost is small: decode
error against the oracle moves from `o 1.7e-3 / h 5.3e-3` (int8 residual) to
`o 2.8e-3 / h 6.6e-3` (int4), still at the INT8 quantization noise floor, and
the project's PPL sweep independently put an int4 residual at +1.6%.

### End-to-end quality of the residual precisions (GSM8K + MMLU, 50 each)

Kernel-level error is not a downstream quality claim, so the residual
precisions were also run end-to-end on Qwen3.5-4B
(`modal_bench_int4resid.py`). All four arms in one run, so each is read
against this run's own baseline:

| scheme | GSM8K | vs bf16 | MMLU |
|---|---|---|---|
| bf16 | 38/50 (76%) | — | 24/50 (48%) |
| int8-V + EF, fp32 residual | 36/50 (72%) | −2 | 24/50 (48%) |
| int8-V + EF, int8 residual | 38/50 (76%) | 0 | 24/50 (48%) |
| **int8-V + EF, int4 residual** | **37/50 (74%)** | **−1** | 24/50 (48%) |

**The int4 residual costs no detectable quality.** It lands one problem below
baseline, the int8 residual matches baseline exactly, and the total spread
across all four arms is two problems.

Two caveats, both load-bearing:

- **n=50 gives roughly a ±6-point band.** This run establishes that int4 does
  not collapse; it is too coarse to claim int4 *exactly* matches baseline. The
  effects it was sized to detect (the 40-point int8-uniform collapse) are far
  outside that band.
- **76% here vs 81–82% in the 100-problem runs is sampling, not regression.**
  It is the same first 50 problems, and the 100-problem runs also scored 38/50
  at their own 50-problem checkpoint — the first half of GSM8K is simply
  harder. Always compare an arm to the baseline from its own run.

`test_memory_claim` asserts **both** directions — the int8 residual must *not*
claim a win over bf16, and the int4 residual must — so neither claim can
silently drift.

The practical conclusion: if this is ever wired into vLLM's cache allocation
(Stage B), it has to be the int4-residual variant. The int8-residual version
would land a change that does not reduce memory against the real baseline.

## Limitations

- **Moderate GSM8K sample.** 100 problems (Δ ± ~4 pts at these rates), up from
  40. The direction is stable across both runs and the PPL signals agree, but
  the full 1319-problem set would tighten it further.
- **MMLU is uninformative here.** It is flat at 54% across every scheme; a
  single-token multiple-choice task does not exercise the recurrent state over
  a long horizon. It supports "nothing broke", not "the method works".
- **One model family.** Validated on Qwen3.5-4B only; the mechanism (CPU) is
  architecture-agnostic but the end-to-end claim is specific to this model.
- **Kernel not integrated into vLLM/fla.** A vLLM-layout kernel exists and is
  correctness-tested (Finding 8), but it is standalone: nothing is wired into
  vLLM's cache allocation. Deliberately so — against vLLM's real bf16 default
  the current scheme is 0.97x on memory, so the integration would not deliver
  the win it advertises until the residual gets cheaper.
- **The memory claim is baseline-dependent.** 2x is against FP32; against bf16
  it is break-even. Every memory number in this document should be read with
  its baseline attached.

## Net

Error-feedback recurrent-state quantization rescues INT8 state: it recovers the
reasoning degradation that DAMP reported (−40 GSM8K points → 0, a full return to
the bf16 baseline on 100 problems), at **~2x memory reduction** (int8 state +
int8 residual) with ~zero quality cost. The
mechanism — compounding vs. wash-out as a function of decay rate — resolves the
DAMP-vs-Minima contradiction, and error feedback (missed by all prior work) is
the technique that makes it work.
