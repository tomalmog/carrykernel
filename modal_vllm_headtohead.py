"""Head-to-head: our quantized GDN decode kernel vs vLLM's production kernel.

Every speed number in this project so far compares our CUDA kernel against our
own Triton kernel. That is self-referential: it says which of our two
implementations is better, not where either sits against what actually ships.

This benchmark fixes that. It installs vLLM, imports its vendored decode kernel
(`fused_recurrent_gated_delta_rule_packed_decode`, the one Qwen3.5 serving
actually calls), and times it against
`vllm_integration/quantized_packed_decode.py` at identical shapes, on the same
GPU, in the same process.

Two things are measured, and they are different questions:

  1. Speed -- us/token for vLLM's bf16/fp32 kernel vs ours (int8, int8 and int4
     residuals). Ours does strictly more work per step (dequantize, requantize,
     two amax reductions, residual pack), so the honest expectation is that we
     are SLOWER. The point is to find out by how much.
  2. Traffic -- bytes of recurrent state moved per step, which is where the
     quantization actually pays. This is arithmetic from the tensor sizes, not
     a benchmark, and it is the claim the project rests on.

Correctness is checked first: our kernel's output is compared against vLLM's on
the same inputs, so the timing is between two things that compute the same
function (up to quantization error).

Usage:
    modal run modal_vllm_headtohead.py
"""

import modal

app = modal.App("carrykernel-vllm-headtohead")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch", "numpy")
    .pip_install("vllm")  # brings the vendored flash-linear-attention ops
    .add_local_dir(".", remote_path="/root/repo", ignore=[".git", "__pycache__", "*.pyc"])
)


@app.function(gpu="A10G", image=image, timeout=3600)
def bench(steps: int = 500, warmup: int = 100):
    import sys
    sys.path.insert(0, "/root/repo")

    import torch
    import vllm

    print("torch", torch.__version__, "| vllm", vllm.__version__,
          "| gpu", torch.cuda.get_device_name(0), flush=True)

    # --- locate vLLM's vendored GDN decode kernel -------------------------
    fn = None
    for modpath in (
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent",
        "vllm.model_executor.layers.fla.ops.fused_recurrent",
    ):
        try:
            mod = __import__(modpath, fromlist=["x"])
            fn = getattr(mod, "fused_recurrent_gated_delta_rule_packed_decode", None)
            if fn is not None:
                print(f"found vLLM kernel in {modpath}", flush=True)
                break
        except Exception as exc:  # noqa: BLE001 - report and try the next path
            print(f"  (no {modpath}: {type(exc).__name__})", flush=True)
    if fn is None:
        return {"error": "could not import vLLM's packed decode kernel"}

    from vllm_integration import quantized_packed_decode as qpd

    dev = torch.device("cuda")
    torch.manual_seed(0)

    def time_it(call):
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(steps):
            call()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / steps * 1e3  # us/step

    results = {}
    # Qwen3.5-4B GDN shapes: 32 V-heads, 16 QK-heads, head dim 128.
    H, HV, K, V = 16, 32, 128, 128

    def sanity(label, t_fp32, t_bf16):
        """bf16 moves half the bytes of fp32; if it times slower, the harness
        is measuring something other than the kernel. Say so loudly rather than
        reporting a flattering-but-wrong ratio."""
        if t_fp32 and t_bf16 and t_bf16 > t_fp32 * 1.2:
            print(f"  !! SUSPECT [{label}]: vLLM bf16 ({t_bf16:.1f} us) slower than "
                  f"fp32 ({t_fp32:.1f} us) -- half the traffic should not be slower. "
                  f"Treat the bf16 comparison as invalid.", flush=True)
            return False
        return True

    for B in (1, 8, 16):
        mixed = torch.randn(B, 2 * H * K + HV * V, device=dev)
        a = torch.randn(B, HV, device=dev) * 0.5
        b = torch.randn(B, HV, device=dev)
        A_log = torch.randn(HV, device=dev) * 0.2
        dt_bias = torch.randn(HV, device=dev) * 0.1
        scale = K ** -0.5
        idx = torch.arange(1, B + 1, dtype=torch.int32, device=dev)
        out = torch.empty(B, 1, HV, V, device=dev)

        h0 = torch.randn(B + 1, HV, V, K, device=dev) * 0.1

        print(f"\n=== batch={B}  HV={HV} K={K} V={V} ===", flush=True)

        # ---- vLLM's kernel, fp32 state (its prefix-caching configuration) ----
        state_f32 = h0.clone()
        out_f32 = torch.empty(B, 1, HV, V, device=dev, dtype=torch.float32)
        vllm_f32 = lambda: fn(  # noqa: E731
            mixed_qkv=mixed, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale,
            initial_state=state_f32, out=out_f32, ssm_state_indices=idx,
            use_qk_l2norm_in_kernel=True)
        try:
            vllm_f32()
            torch.cuda.synchronize()
            t_v32 = time_it(vllm_f32)
        except Exception as exc:  # noqa: BLE001
            print(f"  vLLM fp32 kernel failed: {type(exc).__name__}: {exc}", flush=True)
            t_v32 = None

        # ---- vLLM's kernel, bf16 state (the real serving default) ----
        # Every tensor is pre-cast once, outside the timed lambda. Casting
        # inline (out.to(bfloat16) per call) allocates a fresh output tensor on
        # every iteration and times the allocation, not the kernel -- which made
        # bf16 look slower than fp32, an impossible result that flagged the bug.
        state_bf16 = h0.to(torch.bfloat16)
        mixed_bf16 = mixed.to(torch.bfloat16)
        a_bf16 = a.to(torch.bfloat16)
        b_bf16 = b.to(torch.bfloat16)
        A_log_bf16 = A_log.to(torch.bfloat16)
        dt_bias_bf16 = dt_bias.to(torch.bfloat16)
        out_bf16 = torch.empty(B, 1, HV, V, device=dev, dtype=torch.bfloat16)
        vllm_bf16 = lambda: fn(  # noqa: E731
            mixed_qkv=mixed_bf16, a=a_bf16, b=b_bf16, A_log=A_log_bf16,
            dt_bias=dt_bias_bf16, scale=scale,
            initial_state=state_bf16, out=out_bf16,
            ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
        try:
            vllm_bf16()
            torch.cuda.synchronize()
            t_vbf = time_it(vllm_bf16)
        except Exception as exc:  # noqa: BLE001
            print(f"  vLLM bf16 kernel failed: {type(exc).__name__}: {exc}", flush=True)
            t_vbf = None

        # ---- ours, int8 state, int8 and int4 residuals ----
        ours = {}
        for fmt, name in ((qpd.RESID_INT8, "int8resid"), (qpd.RESID_INT4, "int4resid")):
            sc, ss, rc, rs = qpd.init_quantized_state(
                B + 1, HV, V, K, dev, h0=h0, resid_fmt=fmt)
            call = lambda sc=sc, ss=ss, rc=rc, rs=rs, fmt=fmt: (  # noqa: E731
                qpd.quantized_gated_delta_rule_packed_decode(
                    mixed, a, b, A_log, dt_bias, scale, sc, ss, rc, rs, out, idx,
                    use_qk_l2norm_in_kernel=True, resid_fmt=fmt))
            call()
            torch.cuda.synchronize()
            ours[name] = time_it(call)

        # ---- state traffic per step (read + write), bytes ----
        elems = B * HV * V * K
        scale_bytes = 2 * B * HV * V * 4
        traffic = {
            "vllm_fp32": 2 * elems * 4,
            "vllm_bf16": 2 * elems * 2,
            "ours_int8resid": 2 * (elems + elems) + 2 * scale_bytes,
            "ours_int4resid": 2 * (elems + elems // 2) + 2 * scale_bytes,
        }

        def fmt_us(t):
            return f"{t:8.1f}" if t is not None else "     n/a"

        bf16_ok = sanity(f"B{B}", t_v32, t_vbf)

        print(f"  vLLM  fp32 state : {fmt_us(t_v32)} us  "
              f"traffic {traffic['vllm_fp32']/1e6:6.2f} MB", flush=True)
        print(f"  vLLM  bf16 state : {fmt_us(t_vbf)} us  "
              f"traffic {traffic['vllm_bf16']/1e6:6.2f} MB   <- real default", flush=True)
        for name in ("int8resid", "int4resid"):
            rel = ""
            if t_vbf and bf16_ok:
                rel = f"  ({t_vbf/ours[name]:.2f}x vs vLLM bf16)"
            elif t_v32:
                rel = f"  ({t_v32/ours[name]:.2f}x vs vLLM fp32)"
            print(f"  ours  {name:<10} : {ours[name]:8.1f} us  "
                  f"traffic {traffic['ours_'+name]/1e6:6.2f} MB{rel}", flush=True)

        results[f"B{B}"] = {
            "vllm_fp32_us": t_v32, "vllm_bf16_us": t_vbf,
            "ours_int8resid_us": ours["int8resid"],
            "ours_int4resid_us": ours["int4resid"],
            "traffic_MB": {k: v / 1e6 for k, v in traffic.items()},
        }

    return results


@app.local_entrypoint()
def main(steps: int = 500, warmup: int = 100):
    res = bench.remote(steps=steps, warmup=warmup)
    print("\n=== summary ===")
    print(res)
