# Session handoff — 2026-09-15

Everything from this session, including what is not yet in `CLAUDE.md` /
`RESULTS.md`. Read `CLAUDE.md` first for the project itself; this document
covers **today's work, today's mistakes, and what is in flight right now**.

---

## 1. State right now

**Repo:** `tomalmog/carrykernel`, branch `main`, HEAD `89270c1`, working tree
clean, everything pushed.

**In flight:** one detached Modal run, `ap-zannS2bQXczvMFDLCHdqEB`
(`carrykernel-int4resid`, H100, started 10:08 EDT). Monitor task `b2ciisj9p` is
armed on it and notifies on each `RESULT` line, on per-50 progress, and on any
termination (cancellation / kill / app stopped).

First arm has landed:

```
RESULT bf16   GSM8K=82/100 (0.820)   MMLU=54/100 (0.540)
```

Three arms remain: `int8-V+EF fp32resid`, `int8-V+EF int8resid`,
`int8-V+EF int4resid`. Roughly 10–15 min per arm.

**Note on 82 vs 81.** The earlier 100-problem run scored bf16 at 81/100; this
one scores 82/100 on the same 100 problems. Same model, same greedy decoding —
the difference is that this harness quantizes nothing on the bf16 arm but
reaches the model through a freshly installed hook, and small numerical /
kernel-path differences move one problem. Treat ±1–2 problems as run-to-run
noise at n=100, and read the int4 arm against **this run's own** bf16 (82), not
against the older 81.

---

## 2. What was accomplished this session

| Piece | Commit | Status |
|---|---|---|
| Raw CUDA/C++ fused kernel | `093a2ae` | Done, verified vs oracle + Triton, Nsight-profiled |
| 100-problem GSM8K/MMLU benchmark | `093a2ae`, `e1bee61` | Done, all 4 schemes |
| vLLM Stage A (quantized decode in vLLM's layout) | `ee9ca25` | Done, 6 correctness checks |
| bf16-baseline caveat documented | `ff0a253` | Done |
| int4-packed residual | `e1bee61`, `4da518f` | Done, 1.28x vs bf16 |
| Repro commands fixed | `a5f8d43` | Done |
| Handoff + resume updated | `6bf5fd2` | Done |
| int4 quality harness | `36f2008` | Done, **running** |
| Head-to-head vs vLLM's real kernel | `89270c1` | Done |

---

## 3. All numbers, sorted by whether they have an external baseline

This split matters more than any individual number. The user pushed hard on it
and was right to.

### 3a. Externally anchored — these are the real results

**Quality** (Qwen3.5-4B, H100, 100 problems each; baseline is the unquantized
model; runs through a `DynamicCache.update_recurrent_state` hook, **no kernel of
ours is involved**):

| scheme | GSM8K | MMLU |
|---|---|---|
| bf16 (unquantized) | 81% | 54% |
| int8 uniform | **41%** | 54% |
| int8 per-V + EF | **81%** | 54% |
| int6 per-V + EF | **81%** | 54% |

The headline. Uniform INT8 loses 40 points; error feedback recovers all of them.
int6+EF also reaching 81% is the sharpest statement of the claim: 6-bit state
*with* EF beats 8-bit *without* it, at 25% less storage — the bit width is not
the variable, the error feedback is.

MMLU is flat at 54% everywhere. Recorded as a **control**, not a second win:
single-token multiple choice barely exercises the recurrent state.

**Speed vs vLLM 0.29.0's production kernel** (A10G, us/token, same process, same
shapes, against `fused_recurrent_gated_delta_rule_packed_decode`):

| batch | vLLM bf16 (default) | ours int8resid | ours int4resid |
|---|---|---|---|
| 1 | 53.2 | 54.2 (0.98x) | 56.0 (0.95x) |
| 8 | 53.2 | 55.5 (0.96x) | 56.1 (0.95x) |
| 16 | 72.0 | 82.2 (0.88x) | **69.5 (1.03x)** |

State traffic at batch 8: ours int4 **13.11 MB** vs vLLM bf16 **16.78 MB**.

Parity within ~5%, while moving less traffic. At batch 16 the int4 variant is
ahead on both time and bytes. For a kernel doing strictly more work per step
(dequantize, requantize, two amax reductions, nibble packing), parity is a good
outcome — and it is the honest one.

**Mechanism** (pure PyTorch, exact GDN recurrence, no kernel): error compounds at
`alpha=0.995`, washes out at `alpha=0.9`. Final-state error at t=2000, int4:
2.65 uniform → 0.19 with EF. This resolves the DAMP-vs-Minima contradiction and
stands independent of every kernel in the repo.

**Roofline** (against silicon): 66% of measured peak HBM bandwidth (318 of
484 GB/s) on the FP32 CUDA path.

**Memory** (arithmetic from tensor sizes, not a benchmark), bytes/slot at
HV=32, V=K=128:

| representation | bytes/slot | vs fp32 | vs bf16 |
|---|---|---|---|
| fp32 | 2.10 MB | 1.00x | — |
| bf16 | 1.05 MB | 2.00x | 1.00x |
| int8 state + int8 resid | 1.08 MB | 1.94x | **0.97x** |
| int8 state + int4 resid | 0.82 MB | 2.56x | **1.28x** |

### 3b. Self-referential — discount these

- **CUDA 125 us vs Triton 72 us** (batch 8, INT8 path). Our code vs our code.
  Says which of our two implementations is better, nothing about the state of
  the art. Superseded by the head-to-head above.
- **33% → 66% bandwidth.** The endpoint is externally anchored; the *delta* is
  us fixing our own mistake.
- **The register-spill finding.** Real and instructive, but it is us discovering
  our own bad idea.

---

## 4. Mistakes made today — all of them optimistic

Worth reading before trusting any new number. Every error ran in the flattering
direction, which is why "is this physically possible?" became the check that
caught them.

1. **Vacuous rounding test.** The first round-half-to-even test constructed zero
   actual `.5` ties — it passed by testing nothing, while guarding the exact bug
   `CLAUDE.md` warns about. Fixed by building ties from integer codes and adding
   asserts that the test *must* construct ties and that those ties *must*
   discriminate against `floor(x+0.5)`.

2. **Register caching (CUDA).** `float h_reg[128]` per thread halved the
   instruction count (10.06M → 5.53M) and ran **2x slower**: nvcc spilled to
   local memory, which is DRAM-backed, so `dram__bytes_write` went 10.4 → 71.1 MB.
   Reverted. On a memory-bound kernel, recompute beats spilled traffic.

3. **Head-to-head harness: reported 2.4x faster than vLLM.** Caused by casting
   `out` to bf16 *inside* the timed lambda, allocating a fresh tensor per
   iteration — so the bf16 arm timed allocation, not the kernel. The tell: vLLM's
   bf16 came out *slower than its own fp32* despite moving half the bytes, which
   is impossible. Fixed by pre-casting outside the timed region; the benchmark
   now refuses to report a bf16 ratio when that inversion appears.

4. **Monitor filter too narrow.** The first int4 run was killed at ~12 min and
   the monitor stayed silent for 30 minutes because its filter matched `RESULT`
   and crash signatures but not *cancellation*. Silence looked identical to
   "still running." The current filter covers cancellation / kill / stop and also
   breaks if the app state goes `stopped`.

5. **`setsid` does not exist on macOS.** A relaunch silently did nothing. Caught
   by checking `modal app list` rather than assuming the launch worked.

---

## 5. Infrastructure notes (hard-won, not in CLAUDE.md)

- **Modal's pip-torch image has no `nvcc`.** Compiling a `.cu` needs an
  `nvidia/cuda:*-devel` base (`modal_cuda.py` uses `12.4.1-devel-ubuntu22.04`).
  A10G is compute capability **8.6**.
- **Nsight in a Modal container cannot lock GPU clocks.** Pass
  `--clock-control=none` or `ncu` fails outright.
- **Triton `@triton.jit` cannot read plain Python globals** — they must be
  `tl.constexpr` mirrors. Hit this twice (`SOFTPLUS_THRESHOLD`, then
  `RESID_INT4`).
- **Detached Modal runs get cancelled. Use `modal deploy` + `Function.spawn()`
  instead.** Two `modal run --detach` attempts were killed mid-run by a
  cancellation signal — run 1 at ~12 min, run 2 at ~4.7 h (mid-arm-2). Log
  signature both times:

  ```
  Received a cancellation signal while processing input (...)
  Input ... failed to respond to cancellation for too long: 30 seconds - killing task
  ```

  The first hypothesis (the local launcher parent was reaped) is **refuted**:
  after the second kill, `ps` showed the `modal run --detach` process still
  alive, and `modal app list` showed the app `ephemeral (detached)` with 0
  tasks — the container died, not the app. A fixed timeout is also ruled out by
  the 12-min vs 4.7-h spread. The only factor both shared was a live client
  attached to an **ephemeral** app.

  The fix that does not depend on getting the root cause right: deploy the app
  (persistent, server-side, owned by no client session) and spawn the call.
  `spawn_int4resid.py` does this. Poll with
  `modal app logs carrykernel-int4resid`.

- **Long runs are expensive to lose, so size them accordingly.** The bf16 arm is
  the *only* arm that does no quantization work (`bits=None` returns the state
  untouched); every EF arm runs a per-V quantize + residual quantize +
  subtraction per token per GDN layer (24 layers) on a `[1,32,128,128]` tensor
  in eager PyTorch. Measured: EF arms are ~3x slower than bf16. So 4 arms at 100
  problems is 4-6 h, not the ~1 h a naive estimate gives. The relaunch uses 50
  problems/arm (~2-3 h); at n=50 the noise band is about +/-6 points, which is
  ample for the effects in question (the int8-uniform collapse is 40 points) but
  too coarse to claim an exact match to baseline.
- **vLLM's kernel imports from**
  `vllm.third_party.flash_linear_attention.ops.fused_recurrent` (verified on
  vllm 0.29.0, not the `layers/fla/ops/` path some docs show).

---

## 6. Files added this session

```
cuda/
  gdn_state_kernel.cu              raw CUDA/C++ fused op + FP32 baseline
  binding.py                       JIT build + Python wrapper
  __init__.py
experiments/
  test_cuda_kernel.py              CUDA vs oracle / Triton, all residual formats
  cuda_bench.py                    CUDA vs Triton vs FP32, batch 1/8/16 + roofline
vllm_integration/
  quantized_packed_decode.py       vLLM-layout [V,K] kernel, int8 or int4 residual
  test_quantized_packed_decode.py  oracle + paging + NULL_BLOCK_ID + packing tests
  __init__.py
modal_cuda.py                      CUDA correctness + benchmark (needs devel image)
modal_nsight.py                    Nsight Compute profiling
modal_vllm_stage_a.py              vLLM-layout kernel correctness
modal_bench_int4resid.py           int4-residual quality benchmark  <- RUNNING
modal_vllm_headtohead.py           ours vs vLLM's production kernel
HANDOFF.md                         this file
```

---

## 7. What is left

1. **Finish the int4 quality run** (in flight). Then fold the four-arm table into
   `RESULTS.md` / `README.md` / `NOTES.md` and push. **This is the only open
   item from the current thread.**
   - If int4 holds at ~82%: the project has parity-speed, 22%-less-traffic, and
     no quality cost. That is the complete story.
   - If int4 costs real accuracy: say so plainly. The in-engine memory win
     collapses to int8-only (0.97x vs bf16, i.e. nothing), while the quality and
     mechanism results survive untouched.

2. **Optional, none blocking:**
   - Second model (Qwen3.5-9B) for generality — currently one model family.
   - Tune the CUDA INT8 path; it is compute-bound on the quantize passes
     (10.2M instructions vs 2.28M for FP32) and loses to Triton.
   - Stage B: wire into vLLM's cache allocation. Only worth doing with the int4
     residual. The hard part is that the EF residual is per-request persistent
     state that must survive paging, preemption, prefix-cache reuse and
     spec-decode rollback.

3. **User's scope constraint (2026-09-15), still in force:** no upstream PRs, no
   open-sourcing beyond this repo. Pushing to `tomalmog/carrykernel` is fine.
   Commenting on vLLM RFC #55196 — which is blocked on exactly the accuracy
   evidence this project has — would be the cheapest high-value upstream move
   **if** the user ever lifts that constraint. Do not do it unilaterally.

---

## 8. How to talk about this project

Lead with the quality result, not the memory number:

> Uniform INT8 recurrent state drops GSM8K 81% → 41%. Error feedback recovers it
> to 81%, matching bf16. INT6 + error feedback also reaches 81%.

**Always attach the baseline to a memory claim.** "2x memory reduction" is true
only against FP32; against bf16 — what vLLM actually defaults to — the
int8-residual scheme is 0.97x, and only the int4 residual wins at 1.28x. A bare
"2x" does not survive the question "compared to what?".

**Do not claim a speedup.** The claim is parity with vLLM's production kernel at
lower traffic. For kernel-role interviews, the two measurement-contradicts-
intuition findings (coalescing doubled bandwidth; register caching halved
instructions and ran 2x slower) and the discipline of catching three optimistic
self-measurements are the parts that read as engineering judgment.
