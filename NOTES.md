# StateQuant — working notes (source of truth)

Error-feedback recurrent-state quantization for hybrid (linear-attention) LLMs.

## Final numbers (all verified)

### Mechanism (CPU, exact GDN recurrence)
Quantization error **compounds at slow decay** (alpha=0.995) and **washes out at
fast decay** (alpha=0.9). Resolves the DAMP-vs-Minima contradiction.

### Error feedback bits sweep (final-state error, alpha=0.995, t=2000)
| bits | uniform | +EF |
|---|---|---|
| int3 | 3.5e13 | 0.45 |
| int4 | 2.65 | 0.19 |
| int5 | 0.75 | 0.087 |
| int6 | 0.30 | 0.042 |
| int8 | 0.074 | 0.010 |

Error feedback = ~1.5–2 bits effective precision for free. Per-channel
(per-V/per-K) helps ~2.3x over per-head; block scaling is mediocre.

### End-to-end (Qwen3.5-4B, forced-decode PPL, 381 tokens)
| scheme | PPL |
|---|---|
| bf16 | 8.795 |
| int8 uniform | 13.35 (+52%) |
| **int8 per-V + EF** | **8.68 (~0%)** |
| int6 per-V + EF | 10.74 (+22%) |
| int4 per-V + EF | 27.43 (+212%) |

### Broad benchmark (GSM8K + MMLU, 100 problems each) — CURRENT
| scheme | GSM8K | MMLU |
|---|---|---|
| bf16 | 81/100 (81.0%) | 54/100 (54.0%) |
| int8 uniform | 41/100 (41.0%) | 54/100 (54.0%) |
| int8 per-V + EF | 81/100 (81.0%) | 54/100 (54.0%) |
| int6 per-V + EF | (run in progress) | (run in progress) |

Uniform INT8 loses 40 GSM8K points; EF recovers ALL of them (back to the bf16
baseline exactly). MMLU flat across schemes — single-token multiple choice
barely exercises the recurrent state, so it is a control, not evidence.

### Earlier 40-problem run (superseded; source of the WikiText-2 numbers)
| scheme | WikiText2 | GSM8K |
|---|---|---|
| bf16 | 10.075 | 75.0% |
| int8 uniform | 15.767 | 32.5% |
| int8 per-V + EF | 9.953 | 70.0% |
| int6 per-V + EF | 12.287 | 72.5% |

### Residual precision (forced-decode PPL, int8 state)
| residual | PPL | vs bf16 | memory vs FP32 |
|---|---|---|---|
| fp32 | 8.678 | -1.3% | 0.80x (worse) |
| fp16 | 8.816 | +0.2% | 1.33x |
| fp8 e4m3fn | 9.486 | +7.9% | 2.00x |
| int8 | 8.836 | +0.5% | 2.00x |
| int4 | 8.940 | +1.6% | 2.67x |

int8 residual is sufficient → **~2x memory at ~0% quality**.

### Kernel (fused Triton, A10G)
| batch | config | us/token | speedup |
|---|---|---|---|
| 1 | FP32 | 15.4 | 1.00x |
| 8 | FP32 | 76.5 | 1.00x |
| 8 | INT8+EF int8 resid | 72.1 | 1.06x |
| 8 | INT8+EF fp8 resid | 58.5 | 1.31x |

Traffic reduction ~1.9x, wall-clock ~1.3x at batch 8 (slowdown at batch 1).
Key bug fixed: round-half-to-even vs round-half-away drifted the residual.

### Kernel (raw CUDA/C++, A10G, measured peak 484 GB/s)
us/token, CUDA vs Triton:

| batch | FP32 CUDA | FP32 Triton | INT8+EF CUDA | INT8+EF Triton |
|---|---|---|---|---|
| 1 | **15.4** | 16.0 | 29.1 | 20.4 |
| 8 | 107.3 | **76.6** | 125.2 | **72.1** |
| 16 | 215.8 | **152.8** | 204.9 | **139.9** |

Three measured design findings (Nsight):
1. Coalescing dominates: thread-per-V-column (V contiguous) vs warp-per-column
   took FP32 from 159 -> 318 GB/s (33% -> 66% peak), 214 -> 107 us at B=8.
2. Register-caching the column (`float h_reg[128]`) halved instructions
   (10.21M -> 5.53M) but was 2x SLOWER: nvcc spilled to local memory,
   dram writes 10.1 -> 71.1 MB. Recompute beats spilled traffic.
3. Triton wins the INT8 path. It is compute-bound on the quantize passes
   (10.2M inst vs 2.28M for FP32 at similar traffic), not memory-bound.
   Occupancy caps at ~28% of peak warps.

CUDA wins the FP32 baseline and batch 1; Triton wins INT8 everywhere.
Reported as-is: the claim is traffic/storage, not speedup.

## Bottom line
Error-feedback recurrent-state quantization rescues INT8 state (GSM8K 41%→81%
on 100 problems, a full recovery to the bf16 baseline) at ~2x memory reduction.
Novel vs DAMP/DeltaLog/Minima (none used EF on the recurrent state). No
wall-clock speedup to speak of (~1.3x at batch 8, and the CUDA INT8 path is
compute-bound), so position as a quantization-method contribution, not a
"4x kernel".
