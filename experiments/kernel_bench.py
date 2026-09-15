"""Fused GDN-state kernel benchmark: FP32 vs INT8+EF, memory-bandwidth bound.

Measures the fused Triton kernels (read/write the recurrent state once per decode
step) at realistic Qwen3.5-4B shapes (HV=32, K=128, V=128), batch 1 and 8.

Reports per-token kernel time, actual bytes moved (in MB), achieved HBM bandwidth,
and the FP32 -> INT8+EF speedup, sweeping the error-feedback residual precision
(fp32 / fp16 / fp8 / int8). Also validates the Triton kernels against the torch
reference before timing so the numbers are meaningful.

Traffic model (per decode step, per element of the [K, V] state):
    FP32 baseline:  read 4B + write 4B = 8 B/elem
    INT8 state:     read 1B + write 1B = 2 B/elem
    residual (P):   read P + write P = 2P B/elem   (P = 4/2/1/1 for fp32/fp16/fp8/int8)
    scales:         per-V fp32 scale (state, and residual for int8) = 8/K B/elem

The honest total is the sum.  fp32 residual is *worse* than the fp32 baseline
(10 vs 8 B/elem); fp16 is ~25% less; fp8/int8 is ~2x less.
"""

import torch

from statequant import kernel

if kernel.HAS_TRITON and torch.cuda.is_available():
    import triton  # noqa: F401

MB = 1e6  # report MB (decimal); the repo's "~12 MiB" figures use 2^20


def _other_bytes(B, HV, K, V):
    qk = 2 * B * HV * K * 4
    vv = B * HV * V * 4
    ab = 2 * B * HV * 4
    o = B * HV * V * 4
    return qk + vv + ab + o


def traffic_fp32(B, HV, K, V):
    st = B * HV * K * V * 4
    other = _other_bytes(B, HV, K, V)
    return {
        "state_read": st, "state_write": st, "other": other,
        "total": 2 * st + other,
    }


def traffic_quant(B, HV, K, V, res_fmt):
    st = B * HV * K * V * 1
    sc = B * HV * V * 4
    e = B * HV * K * V * kernel.residual_bytes(res_fmt)
    es = B * HV * V * 4 if res_fmt == kernel.RES_INT8 else 0
    other = _other_bytes(B, HV, K, V)
    return {
        "state_rw": 2 * st, "scale_rw": 2 * sc, "resid_rw": 2 * e,
        "resid_scale_rw": 2 * es, "other": other,
        "total": 2 * st + 2 * sc + 2 * e + 2 * es + other,
    }


def time_kernel(fn, steps):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / steps  # ms per token


def _validate(B, HV, K, V, h_int8, scale, e, e_scale, q, k, v, alpha, beta,
              res_fmt, o, h2, s2, e2, es2, BLOCK_V):
    """One-step Triton vs torch reference. Returns (err_h, err_o)."""
    o_ref, hr, sr, er, esr = kernel.gdn_quant_step_torch(
        h_int8, scale, e, e_scale, q, k, v, alpha, beta, 8, res_fmt)
    h_ref_deq = hr.float() * sr[..., None, :]
    kernel._gdn_quant_kernel[(B, HV, kernel._cdiv(V, BLOCK_V))](
        h_int8, scale, e, e_scale, q, k, v, alpha, beta,
        o, h2, s2, e2, es2, HV, K=K, V=V, QMAX=127.0,
        BLOCK_V=BLOCK_V, RES_FMT=res_fmt)
    torch.cuda.synchronize()
    h_deq = h2.float() * s2[..., None, :]
    err_h = (h_deq - h_ref_deq).abs().max().item()
    err_o = (o - o_ref).abs().max().item()
    return err_h, err_o


def run(gpu_name="?", BLOCK_V=32, steps=1000, warmup=200, seed=0):
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    print(f"GPU: {torch.cuda.get_device_name(0)}  ({gpu_name})")
    print(f"steps={steps} warmup={warmup} BLOCK_V={BLOCK_V}")

    results = {}
    for B in (1, 8):
        HV, K, V = 32, 128, 128
        fp32 = dict(device=dev, dtype=torch.float32)
        q = torch.randn(B, HV, K, **fp32)
        k = torch.randn(B, HV, K, **fp32)
        k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v = torch.randn(B, HV, V, **fp32)
        alpha = torch.full((B, HV), 0.995, **fp32)
        beta = torch.rand(B, HV, **fp32).sigmoid()
        h0 = torch.randn(B, HV, K, V, **fp32) * 0.1

        print(f"\n=== batch={B}  HV={HV} K={K} V={V}  "
              f"(state {B*HV*K*V*4/MB:.2f} MB fp32) ===")

        # ---------------- FP32 baseline (fused Triton) ----------------
        h_a = h0.contiguous()
        h_b = torch.empty_like(h_a)
        o = torch.empty(B, HV, V, **fp32)

        hr, orr = kernel.gdn_step_fp32_torch(h0, q, k, v, alpha, beta)
        kernel._gdn_fp32_kernel[(B, HV, kernel._cdiv(V, BLOCK_V))](
            h_a, q, k, v, alpha, beta, o, h_b, HV, K=K, V=V, BLOCK_V=BLOCK_V)
        torch.cuda.synchronize()
        err = (h_b - hr).abs().max().item()
        assert err < 1e-3, f"fp32 kernel mismatch: {err}"

        bufs = [h_a, h_b]

        def loop_fp32():
            kernel._gdn_fp32_kernel[(B, HV, kernel._cdiv(V, BLOCK_V))](
                bufs[0], q, k, v, alpha, beta, o, bufs[1],
                HV, K=K, V=V, BLOCK_V=BLOCK_V)
            bufs[0], bufs[1] = bufs[1], bufs[0]

        for _ in range(warmup):
            loop_fp32()
        t_fp32 = time_kernel(loop_fp32, steps)
        b_fp32 = traffic_fp32(B, HV, K, V)
        bw_fp32 = b_fp32["total"] / (t_fp32 * 1e-3) / 1e9  # GB/s

        print(f"FP32 fused:      {t_fp32*1e3:8.1f} us/token  "
              f"traffic {b_fp32['total']/MB:6.2f} MB  -> {bw_fp32:6.1f} GB/s")

        # ---------------- INT8 + EF, sweep residual precision ----------------
        for res_fmt in (kernel.RES_FP32, kernel.RES_FP16, kernel.RES_FP8,
                        kernel.RES_INT8):
            label = kernel.residual_label(res_fmt)
            h_int8, scale, e0, es0 = kernel.init_quant_state(h0, 8, res_fmt)
            s = {"h": h_int8.contiguous(), "h2": torch.empty_like(h_int8),
                 "s": scale.contiguous(), "s2": torch.empty_like(scale),
                 "e": e0.contiguous(), "e2": torch.empty_like(e0)}
            if res_fmt == kernel.RES_INT8:
                s["es"] = es0.contiguous()
                s["es2"] = torch.empty_like(es0)
            else:
                s["es"] = torch.zeros(B, HV, V, **fp32)
                s["es2"] = torch.empty(B, HV, V, **fp32)

            err_h, err_o = _validate(
                B, HV, K, V, s["h"], s["s"], s["e"], s["es"], q, k, v, alpha,
                beta, res_fmt, o, s["h2"], s["s2"], s["e2"], s["es2"], BLOCK_V)
            assert err_o < 1e-4, f"int8 kernel o mismatch ({label}): {err_o}"
            assert err_h < 1e-2, f"int8 kernel h mismatch ({label}): {err_h}"

            def loop_quant():
                kernel._gdn_quant_kernel[(B, HV, kernel._cdiv(V, BLOCK_V))](
                    s["h"], s["s"], s["e"], s["es"], q, k, v, alpha, beta,
                    o, s["h2"], s["s2"], s["e2"], s["es2"], HV, K=K, V=V,
                    QMAX=127.0, BLOCK_V=BLOCK_V, RES_FMT=res_fmt)
                s["h"], s["h2"] = s["h2"], s["h"]
                s["s"], s["s2"] = s["s2"], s["s"]
                s["e"], s["e2"] = s["e2"], s["e"]
                s["es"], s["es2"] = s["es2"], s["es"]

            for _ in range(warmup):
                loop_quant()
            t_q = time_kernel(loop_quant, steps)
            b_q = traffic_quant(B, HV, K, V, res_fmt)
            bw_q = b_q["total"] / (t_q * 1e-3) / 1e9
            speedup = t_fp32 / t_q
            compress = b_fp32["total"] / b_q["total"]

            print(f"INT8+EF [{label:<5}]: {t_q*1e3:8.1f} us/token  "
                  f"state {b_q['state_rw']/MB:5.2f} MB "
                  f"+resid {b_q['resid_rw']/MB:5.2f} MB "
                  f"total {b_q['total']/MB:6.2f} MB  -> {bw_q:6.1f} GB/s   "
                  f"speedup {speedup:5.2f}x  compress {compress:5.2f}x")

            results[f"B{B}_{label}"] = {
                "fp32_us": t_fp32 * 1e3, "int8_us": t_q * 1e3,
                "fp32_MB": b_fp32["total"] / MB, "int8_MB": b_q["total"] / MB,
                "fp32_GBs": bw_fp32, "int8_GBs": bw_q,
                "speedup": speedup, "compression": compress,
            }

    return results


def main():
    assert kernel.HAS_TRITON and torch.cuda.is_available(), \
        "kernel_bench requires CUDA + Triton"
    return run()


if __name__ == "__main__":
    main()
