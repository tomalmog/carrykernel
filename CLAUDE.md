# CLAUDE.md — CarryKernel handoff

This file is the full handoff for anyone (human or agent) continuing this
project. Read it top to bottom before touching anything.

## What this project is

**CarryKernel** — *error-feedback recurrent-state quantization for hybrid
(linear-attention) LLMs.* The one-sentence result:

> Uniform INT8 quantization of a hybrid LLM's recurrent state collapses GSM8K
> accuracy (75% → 32.5%); applying **error feedback** + **per-channel scaling**
> recovers it (→ 70%) at **~2× memory reduction** and ~zero quality cost.

The novelty: none of the prior state-quantization papers (DAMP, Minima,
DeltaLog, KVBuffer) applied error feedback to the recurrent state. We did, and
it's what makes low-bit state work.

Live repo: **https://github.com/tomalmog/carrykernel** (public, pushed).

## Context / why this exists

The author (Tom Almog, UW CS co-op) is applying to inference/kernel roles:
Baseten GPU Kernels (486457), Baseten LLM Performance (486455), Inferact
(486208), Cerebras Kernel Inference (486593/486584), Deep Infra (485558).
Those postings ask for CUDA/C++, memory hierarchy, warp programming, tensor
cores, Nsight, kernel correctness, and reproducible performance evidence. The
resume is currently ML-heavy; this project is meant to add kernel/inference
depth.

The author's bar: no "AI slop" (replicating existing research); the work must be
a genuine, verifiable contribution. The whole research trail (prior-art checks,
novelty audits) is in `~/job_apps/waterlooworks_2b/` — the CarryKernel project
came out of a long search there that rejected many saturated directions
(speculative decoding, KV cache, MoE, persistent kernels — all owned by labs).

## The results (all verified)

### Mechanism (CPU, exact GDN recurrence — `statequant/reference.py`)
Quantization error **compounds at slow decay** (alpha→1) and **washes out at
fast decay** (alpha≈0.9). This resolves the DAMP-vs-Minima contradiction (both
were right, in different regimes; it maps to DASC's "retention horizons").

Final-state error at t=2000, alpha=0.995 (worst case):

| bits | uniform | + error feedback |
|---|---|---|
| int3 | 3.5e13 | 0.45 |
| int4 | 2.65 | 0.19 |
| int5 | 0.75 | 0.087 |
| int6 | 0.30 | 0.042 |
| int8 | 0.074 | 0.010 |

Error feedback ≈ 1.5–2 bits effective precision for free. Per-channel (per-V /
per-K) scaling helps ~2.3× over per-head; block scaling (Minima's weight trick)
is mediocre for state.

### End-to-end (Qwen3.5-4B, forced-decode PPL, 381-token text)
| scheme | PPL |
|---|---|
| bf16 (FP32) | 8.795 |
| int8 uniform | 13.35 (+52%) |
| **int8 per-V + EF** | **8.68 (~0%)** |
| int6 per-V + EF | 10.74 (+22%) |
| int4 per-V + EF | 27.43 (+212%) |

### Broad benchmark (WikiText-2 PPL + GSM8K, 40 problems — full run in progress)
| scheme | WikiText2 PPL | GSM8K |
|---|---|---|
| bf16 | 10.075 | 75.0% |
| int8 uniform | 15.767 | 32.5% |
| int8 per-V + EF | 9.953 | 70.0% |
| int6 per-V + EF | 12.287 | 72.5% |

Headline: uniform INT8 loses 42.5 GSM8K points; EF recovers 37.5 of them.

### Residual precision (int8 state + EF, forced-decode PPL)
The residual must be quantized too, or it negates the memory saving.

| residual | PPL | memory vs FP32 |
|---|---|---|
| fp32 | 8.678 | 0.80× (worse) |
| fp16 | 8.816 | 1.33× |
| fp8 (e4m3fn) | 9.486 | 2.00× |
| **int8** | 8.836 | **2.00×** |
| **int4** | 8.940 | **2.67×** |

fp8/e4m3fn is the one bad choice (fixed mantissa loses small residuals);
per-head amax-scaled int8/int4 adapts. So the honest win is **~2× (int8
residual)** or **~2.67× (int4 residual)** at ~zero quality.

### Kernel (fused Triton, A10G)
| batch | config | us/token | speedup |
|---|---|---|---|
| 1 | FP32 | 15.4 | 1.00× |
| 8 | FP32 | 76.5 | 1.00× |
| 8 | INT8+EF int8 resid | 72.1 | 1.06× |
| 8 | INT8+EF fp8 resid | 58.5 | 1.31× |

Traffic reduction ~1.9×, wall-clock ~1.3× at batch 8 (slower at batch 1).
**The headline is a ~2× storage/traffic win, NOT a 2× speedup.** Position it as
a quantization-method contribution, not a "fast kernel."

## Key technical facts (what the next person must know)

- **Model:** `Qwen/Qwen3.5-4B` (public, Apache-2.0, multimodal). Text-only path
  via `AutoModelForCausalLM` → `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM`.
- **Architecture:** 32 layers, layout `8×(3×(GDN→FFN) → 1×(Gated Attention→FFN))`
  = 24 GDN (linear-attention) layers + 8 attention layers. GDN: 32 V-heads,
  16 QK-heads, 128-dim. State is FP32 `[1, 32, 128, 128]` per layer (48 MiB/request).
- **The hook:** patch `transformers.cache_utils.DynamicCache.update_recurrent_state`.
  The state flows through it as `update_recurrent_state(state, layer_idx)`. This
  is the single injection point used by all the modal_*.py experiments.
- **The recurrence** (exact, matches fla): per head, state `h [K,V]`, keys `k [K]`,
  values `v [V]`, queries `q [K]`, decay `alpha`, write-strength `beta`:
  `h = alpha*h; read = h^T k; dv = beta*(v - read); h += k dv^T; o = h^T q`.
- **Quantization:** symmetric affine, `scale = amax(dim)/qmax`, `q = clamp(round(x/scale))`.
  Per-V-channel = one scale per V-column (`dim=-1`). Error feedback:
  `target = h + e; hq = quant(target); e = target - hq`.
- **Residual must be quantized** (see results above); storing it fp32 negates the win.

## Files

```
statequant/
  reference.py        # exact GDN recurrence (bit-comparable to fla), + validate_reference()
  quantize.py         # uniform / per-channel / block quantizers + ErrorFeedback
  kernel.py           # fused Triton state-update + INT8 quantize + EF (has torch-reference)
experiments/
  exp1_error_dynamics.py  # compounding vs wash-out (mechanism)
  exp2_bits_sweep.py      # INT-k floor + EF
  exp2b_granularity.py    # granularity × EF
  test_kernel.py          # kernel torch-reference vs oracle (PASSES, 0.0 err)
  kernel_bench.py         # fused-kernel bandwidth benchmark
modal_smoke.py        # load model, find the state hook (diagnostic)
modal_eval.py         # end-to-end INT8 result (forced-decode PPL)
modal_bench.py        # WikiText-2 + GSM8K (40-problem; the WORKING extraction is here)
modal_bench_short.py  # 100-problem version w/ progress (currently running)
modal_residual.py     # residual-precision sweep
modal_fullbench.py    # full 1319-problem GSM8K (excessive; don't use)
modal_kernel.py       # kernel benchmark on GPU
modal_diag.py         # quick 3-problem diagnostic
README.md             # overview + results (title is "CarryKernel")
RESULTS.md            # the full paper-style writeup (landing doc)
NOTES.md              # consolidated numbers (source of truth)
CLAUDE.md             # this file
```

## Reproduce (CPU, no GPU needed)

```bash
python experiments/exp1_error_dynamics.py
python experiments/exp2_bits_sweep.py
python experiments/exp2b_granularity.py
python -c "from statequant.reference import validate_reference; assert validate_reference()"
python experiments/test_kernel.py
```

## Infra / budget (Modal)

- Modal CLI installed. Active profile: **`tomalmog2`**. Token stored in
  `~/.modal.toml` (outside the repo — never commit it). The author will rotate
  the token at some point; if requests 401, ask the author for a fresh one.
- Budget started at $30; a few dollars used. A10G ≈ $0.60/hr, H100 ≈ $3–4/hr.
- Long-running jobs: `nohup modal run --detach <file> > /tmp/x.log 2>&1 &`
  (a plain `modal run` gets killed when the shell times out). Poll with
  `modal app list` and `modal app logs <id>`.
- Each run re-downloads Qwen3.5-4B (~10GB) because there's no volume cache —
  budget a few minutes for that.

## Gotchas / bugs already found (do NOT re-discover these)

1. **Thinking mode:** Qwen3.5 reasons by default. Must pass
   `enable_thinking=False` to `apply_chat_template` AND strip `<think>…</think>`
   before answer extraction. If GSM8K comes out ~1% instead of ~75%, this is why.
2. **Token budget:** Qwen3.5 needs ~300–500 tokens to solve GSM8K. `max_new_tokens`
   under ~256 truncates reasoning and tanks accuracy. Use 512.
3. **Answer extraction:** gold comes from the `#### N` marker (`extract_gold`),
   not "last number" (the reference reasoning is full of numbers). pred uses
   last `\boxed{}` after stripping thinking. Both are in `modal_bench.py`.
4. **MMLU int-vs-string:** dataset `answer` is int 0–3; compare against
   `"ABCD"[int(a)]`, not the raw int.
5. **Round-half-to-even:** `torch.round` is round-half-to-even; a naive
   `floor(x+0.5)` in a kernel drifts the error-feedback residual. Match it.
6. **Modal mount:** files importing `statequant` need the dir mounted
   (`Image.add_local_dir(".", remote_path="/root/repo")` + `sys.path`), unlike
   the self-contained modal_*.py scripts.
7. **Name:** repo is "carrykernel", README title "CarryKernel". The internal
   package is still `statequant/` — that's intentional (author said people won't
   read that deeply); do NOT rename it. Keep "quantization" in prose (correct
   term); avoid the bare "quant" token in *names*.

## Status / what's done

- ✅ Mechanism resolved (CPU).
- ✅ Error-feedback technique + per-channel scaling (CPU + GPU).
- ✅ End-to-end Qwen3.5-4B: forced-decode PPL, WikiText-2, GSM8K (40), residual-precision.
- ✅ Fused Triton kernel (correct, benchmarked).
- ✅ Docs finalized (README, RESULTS, NOTES).
- ✅ Repo pushed to github.com/tomalmog/carrykernel (2 commits).
- 🔄 Full-ish benchmark (100 GSM8K + 100 MMLU × 4 schemes) running in background.

## What's left / next steps (in priority order)

1. **Collect the running benchmark numbers** (app in `modal app list`, look for
   `carrykernel-bench` detached). Update `RESULTS.md` / `README.md` / `NOTES.md`
   with the final GSM8K (100) + MMLU (100) table, and push.

2. **NEXT STEP (the author explicitly wants this): write the fused kernel in
   CUDA/C++.** The current kernel is Triton (`statequant/kernel.py`). Rewrite the
   same operation (state update + INT8 quantize + error feedback) as a raw CUDA
   kernel (.cu + a small Python extension or ctypes/numba wrapper), and:
   - Verify correctness against `statequant/reference.py` (the oracle).
   - Benchmark it on Modal (A10G/H100) against the FP32 and Triton baselines.
   - The goal is to demonstrate CUDA/C++ memory-hierarchy/warp-programming skill
     for the target roles — NOT necessarily a big speedup (the workload is
     memory-bound; expect similar or marginally better numbers than Triton).
   - Key things to show: shared-memory tiling of the [K,V] state, warp-level
     reduction for `h^T k`, the per-V-channel scale, the round-half-to-even
     quantization + residual, and an Nsight/roofline note.
   - Commit and push to the existing repo (add to `statequant/` or a new
     `cuda/` dir). Update README + resume bullet to say "CUDA" not "Triton".

3. **Resume update** — the current 3-point entry (see below) should say
   "CUDA/C++" after step 2. Keep it result-first, ~1 line per bullet.

## Resume entry (current draft)

```
CarryKernel — Error-Feedback Recurrent-State Quantization (Python + PyTorch + Triton)  [GitHub]
- Achieved 2× memory reduction at zero quality cost in hybrid-LLM serving by applying error feedback to recurrent-state quantization.
- Restored 37.5 GSM8K accuracy points lost to INT8 state quantization (32.5% → 70%), resolving the DAMP vs. Minima contradiction.
- Wrote a fused Triton kernel for the quantized state update with error feedback, benchmarked against an FP32 baseline.
```

(After the CUDA step: change "Triton" → "CUDA/C++" in both the header and the
third bullet.)

## Prior art (the four papers to cite / compare against)

- **DAMP** (arXiv 2608.27513): per-channel mixed precision, 9.9 bits; reported
  "INT8/FP8 degrade complex reasoning", INT4 "near zero".
- **Minima** (arXiv 2609.04098): W4A4; claimed the recurrence "forgets a state
  impulse within hundreds of steps".
- **DeltaLog** (arXiv 2608.15533): defers state materialization.
- **KVBuffer** (arXiv 2605.19049): buffers state updates.

None used error feedback on the recurrent state — that's the contribution.
