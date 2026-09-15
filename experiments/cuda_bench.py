"""CUDA vs Triton vs FP32 benchmark for the fused GDN-state kernel.

Times one decode step at realistic Qwen3.5-4B shapes (HV=32, K=128, V=128) at
batch 1 and 8, for the FP32 baseline and INT8+EF across residual precisions,
comparing the CUDA kernel (``cuda/gdn_state_kernel.cu``) against the Triton
kernel (``statequant/kernel.py``).

The traffic model is shared with ``experiments/kernel_bench.py`` so the byte
counts are directly comparable. This workload is memory-bound: the point is that
INT8+EF moves ~2x fewer bytes, not that the kernel is dramatically faster.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statequant import kernel as tk
from cuda import binding
from experiments.kernel_bench import traffic_fp32, traffic_quant, MB


def time_fn(fn, steps, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / steps  # ms per token


def peak_bandwidth_gbs():
    """Measured peak HBM bandwidth: a big saturating device-to-device copy.

    Used as the roofline ceiling, so "achieved % of peak" is against what this
    GPU actually delivers rather than the marketing number.
    """
    n = 1 << 26  # 64 Mi floats = 256 MB per buffer
    a = torch.empty(n, device="cuda", dtype=torch.float32)
    b = torch.empty_like(a)
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    iters = 50
    for _ in range(iters):
        b.copy_(a)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return (2 * a.numel() * 4) / (ms * 1e-3) / 1e9  # read + write


def run(gpu_name="?", steps=1000, warmup=200, BLOCK_V_TRITON=32, BLOCK_V_CUDA=128,
        batches=(1, 8, 16), seed=0):
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    print(f"GPU: {torch.cuda.get_device_name(0)}  ({gpu_name})  torch {torch.__version__}")
    print(f"steps={steps} warmup={warmup} "
          f"BLOCK_V triton={BLOCK_V_TRITON} cuda={BLOCK_V_CUDA}")

    binding.load_extension()
    peak = peak_bandwidth_gbs()
    props = torch.cuda.get_device_properties(0)
    print(f"measured peak HBM bandwidth: {peak:.1f} GB/s  "
          f"({props.multi_processor_count} SMs)\n")

    results = {"_peak_gbs": peak, "_sms": props.multi_processor_count}

    for B in batches:
        HV, K, V = 32, 128, 128
        f32 = dict(device=dev, dtype=torch.float32)
        q = torch.randn(B, HV, K, **f32)
        k = torch.randn(B, HV, K, **f32)
        k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v = torch.randn(B, HV, V, **f32)
        alpha = torch.full((B, HV), 0.995, **f32)
        beta = torch.rand(B, HV, **f32).sigmoid()
        h0 = torch.randn(B, HV, K, V, **f32) * 0.1

        nblocks = (V // BLOCK_V_CUDA) * HV * B
        print(f"=== batch={B}  HV={HV} K={K} V={V}  "
              f"(state {B*HV*K*V*4/MB:.2f} MB fp32, "
              f"{nblocks} CUDA blocks for {props.multi_processor_count} SMs) ===")

        # ---------------- FP32 baselines ----------------
        b_fp32 = traffic_fp32(B, HV, K, V)

        hc = [h0.contiguous(), torch.empty_like(h0)]

        def loop_fp32_cuda():
            h_out, _ = binding.gdn_fp32_step_cuda(hc[0], q, k, v, alpha, beta,
                                                  BLOCK_V=BLOCK_V_CUDA)
            hc[0] = h_out

        t_fp32_cuda = time_fn(loop_fp32_cuda, steps, warmup)
        bw = b_fp32["total"] / (t_fp32_cuda * 1e-3) / 1e9
        print(f"FP32   CUDA   : {t_fp32_cuda*1e3:8.1f} us/token  "
              f"traffic {b_fp32['total']/MB:6.2f} MB  -> {bw:6.1f} GB/s "
              f"({100*bw/peak:4.1f}% peak)")

        t_fp32_triton = None
        if tk.HAS_TRITON:
            ha, hb = h0.contiguous(), torch.empty_like(h0)
            o_buf = torch.empty(B, HV, V, **f32)
            bufs = [ha, hb]

            def loop_fp32_triton():
                tk._gdn_fp32_kernel[(B, HV, tk._cdiv(V, BLOCK_V_TRITON))](
                    bufs[0], q, k, v, alpha, beta, o_buf, bufs[1],
                    HV, K=K, V=V, BLOCK_V=BLOCK_V_TRITON)
                bufs[0], bufs[1] = bufs[1], bufs[0]

            t_fp32_triton = time_fn(loop_fp32_triton, steps, warmup)
            bwt = b_fp32["total"] / (t_fp32_triton * 1e-3) / 1e9
            print(f"FP32   Triton : {t_fp32_triton*1e3:8.1f} us/token  "
                  f"traffic {b_fp32['total']/MB:6.2f} MB  -> {bwt:6.1f} GB/s")

        results[f"B{B}_fp32"] = {
            "cuda_us": t_fp32_cuda * 1e3,
            "triton_us": t_fp32_triton * 1e3 if t_fp32_triton else None,
            "MB": b_fp32["total"] / MB,
        }

        # ---------------- INT8 + EF, per residual precision ----------------
        for res_fmt in (tk.RES_FP32, tk.RES_FP16, tk.RES_FP8, tk.RES_INT8):
            label = tk.residual_label(res_fmt)
            b_q = traffic_quant(B, HV, K, V, res_fmt)
            compress = b_fp32["total"] / b_q["total"]

            # --- CUDA ---
            h_i, sc, e0, es0 = tk.init_quant_state(h0, 8, res_fmt)
            st = {"h": h_i.contiguous(), "s": sc.contiguous(), "e": e0.contiguous(),
                  "es": es0.contiguous() if es0 is not None else None}

            def loop_cuda():
                o, h2, s2, e2, es2 = binding.gdn_quant_step_cuda(
                    st["h"], st["s"], st["e"], st["es"], q, k, v, alpha, beta,
                    8, res_fmt, BLOCK_V_CUDA)
                st["h"], st["s"], st["e"] = h2, s2, e2
                if es2 is not None:
                    st["es"] = es2

            t_cuda = time_fn(loop_cuda, steps, warmup)
            bw_c = b_q["total"] / (t_cuda * 1e-3) / 1e9
            spd_c = t_fp32_cuda / t_cuda
            print(f"INT8+EF[{label:<5}] CUDA   : {t_cuda*1e3:8.1f} us/token  "
                  f"total {b_q['total']/MB:6.2f} MB  -> {bw_c:6.1f} GB/s "
                  f"({100*bw_c/peak:4.1f}% peak)  "
                  f"vs FP32 {spd_c:5.2f}x  compress {compress:5.2f}x")

            # --- Triton (same shapes, double-buffered like kernel_bench) ---
            t_tri = None
            if tk.HAS_TRITON:
                h_i2, sc2, e02, es02 = tk.init_quant_state(h0, 8, res_fmt)
                s2 = {"h": h_i2.contiguous(), "h2": torch.empty_like(h_i2),
                      "s": sc2.contiguous(), "s2": torch.empty_like(sc2),
                      "e": e02.contiguous(), "e2": torch.empty_like(e02)}
                if res_fmt == tk.RES_INT8:
                    s2["es"], s2["es2"] = es02.contiguous(), torch.empty_like(es02)
                else:
                    s2["es"] = torch.zeros(B, HV, V, **f32)
                    s2["es2"] = torch.empty(B, HV, V, **f32)
                o_buf2 = torch.empty(B, HV, V, **f32)

                def loop_triton():
                    tk._gdn_quant_kernel[(B, HV, tk._cdiv(V, BLOCK_V_TRITON))](
                        s2["h"], s2["s"], s2["e"], s2["es"], q, k, v, alpha, beta,
                        o_buf2, s2["h2"], s2["s2"], s2["e2"], s2["es2"],
                        HV, K=K, V=V, QMAX=127.0,
                        BLOCK_V=BLOCK_V_TRITON, RES_FMT=res_fmt)
                    s2["h"], s2["h2"] = s2["h2"], s2["h"]
                    s2["s"], s2["s2"] = s2["s2"], s2["s"]
                    s2["e"], s2["e2"] = s2["e2"], s2["e"]
                    s2["es"], s2["es2"] = s2["es2"], s2["es"]

                t_tri = time_fn(loop_triton, steps, warmup)
                bw_t = b_q["total"] / (t_tri * 1e-3) / 1e9
                spd_t = (t_fp32_triton / t_tri) if t_fp32_triton else float("nan")
                print(f"INT8+EF[{label:<5}] Triton : {t_tri*1e3:8.1f} us/token  "
                      f"total {b_q['total']/MB:6.2f} MB  -> {bw_t:6.1f} GB/s   "
                      f"vs FP32 {spd_t:5.2f}x  compress {compress:5.2f}x")

            results[f"B{B}_{label}"] = {
                "cuda_us": t_cuda * 1e3,
                "triton_us": t_tri * 1e3 if t_tri else None,
                "MB": b_q["total"] / MB,
                "compression": compress,
                "cuda_speedup_vs_fp32": spd_c,
                "cuda_vs_triton": (t_tri / t_cuda) if t_tri else None,
                "cuda_gbs": bw_c,
                "pct_peak": 100 * bw_c / peak,
            }
        print()

    return results


def main():
    assert torch.cuda.is_available(), "cuda_bench requires a CUDA GPU"
    return run()


if __name__ == "__main__":
    main()
