"""Correctness test for the fused GDN + INT8 + error-feedback kernel.

Oracle: `statequant/reference.py` (exact GDN recurrence) + `statequant/quant.py`
(`ErrorFeedback` + `per_axis_quantize(dim=-2)` = per-V-channel). The fused
implementation must reproduce this oracle bit-close.

Runs on CPU (the torch reference path). If a CUDA GPU + Triton are available it
also validates the Triton kernel against the torch reference, for every residual
precision (fp32 / fp16 / fp8 / int8).
"""

import torch

from statequant.reference import gdn_step_batched
from statequant.quant import ErrorFeedback, per_axis_quantize, uniform_quantize
from statequant import kernel


def rel(x, y):
    return ((x - y).norm() / (y.norm() + 1e-12)).item()


def make_inputs(T, B, HV, K, V, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(T, B, HV, K)
    k = torch.randn(T, B, HV, K)
    k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v = torch.randn(T, B, HV, V)
    alpha = torch.full((T, B, HV), 0.995)
    beta = torch.rand(T, B, HV).sigmoid()
    h0 = torch.randn(B, HV, K, V) * 0.1
    return h0, q, k, v, alpha, beta


def oracle_decode(h0, q, k, v, alpha, beta, bits=8):
    """Ground-truth decode loop: reference GDN + ErrorFeedback + per-V quantize."""
    B, HV, K, V = h0.shape
    assert B == 1, "oracle is written for the unbatched reference; keep B=1"
    T = q.shape[0]
    _, _, deq0 = kernel.per_v_channel_quantize(h0, bits)  # init: quantize h0
    h = deq0[0].clone()
    fb = ErrorFeedback((HV, K, V))
    fb.e = torch.zeros(HV, K, V)
    qf = lambda x: per_axis_quantize(x, bits, -2)  # per-V-channel (reduce over K)
    os = []
    for t in range(T):
        h, o = gdn_step_batched(h, q[t, 0], k[t, 0], v[t, 0], alpha[t, 0], beta[t, 0])
        h = fb.step(h, qf)
        os.append(o)
    return torch.stack(os).unsqueeze(1), h.unsqueeze(0), fb.e.unsqueeze(0)


def fused_decode(h0, q, k, v, alpha, beta, bits=8, res_fmt=kernel.RES_FP32):
    h_int8, scale, e, e_scale = kernel.init_quant_state(h0, bits, res_fmt)
    T = q.shape[0]
    os = []
    for t in range(T):
        o, h_int8, scale, e, e_scale = kernel.gdn_quant_step_torch(
            h_int8, scale, e, e_scale, q[t], k[t], v[t], alpha[t], beta[t],
            bits, res_fmt)
        os.append(o)
    h_deq = h_int8.float() * scale[..., None, :]
    e_deq = kernel._dequant_residual_torch(e, e_scale, res_fmt)
    return torch.stack(os), h_deq, e_deq


def test_quant_matches_quantpy():
    x = torch.randn(3, 128, 128) * 0.7
    _, _, deq_mine = kernel.per_v_channel_quantize(x, 8)
    deq_ref = per_axis_quantize(x, 8, -2)
    assert torch.allclose(deq_mine, deq_ref), "per-V quantize mismatch vs quant.py"
    # sanity: uniform (per-head) must be *different* (we are per-V, not per-head)
    assert not torch.allclose(deq_mine, uniform_quantize(x, 8)[0]), "unexpectedly uniform"


def gpu_check(h0, q, k, v, alpha, beta, T):
    """Compare the Triton kernels against the torch reference on GPU."""
    dev = "cuda"
    h0c = h0.to(dev)
    qc, kc, vc = q.to(dev), k.to(dev), v.to(dev)
    ac, bc = alpha.to(dev), beta.to(dev)

    # ---- fp32 baseline kernel vs torch reference (no quantization) ----
    h_ref, o_ref = kernel.gdn_step_fp32_torch(h0, q[0], k[0], v[0], alpha[0], beta[0])
    h_tri, o_tri = kernel.gdn_step_fused_fp32(h0c, qc[0], kc[0], vc[0], ac[0], bc[0])
    err_h = (h_tri - h_ref.to(dev)).abs().max().item()
    err_o = (o_tri - o_ref.to(dev)).abs().max().item()
    print(f"  fp32 kernel  h err = {err_h:.3e}  o err = {err_o:.3e}")
    assert err_h < 1e-3 and err_o < 1e-3, "fp32 kernel mismatch"

    # ---- INT8+EF kernel vs torch reference, per residual format ----
    # The Triton kernel uses a different fp32 reduction order than torch's
    # einsum, so at quantization boundaries the int8 code occasionally flips by
    # 1 LSB.  Tolerances are set to the INT8 quantization noise floor (measured:
    # full-decode o/h vs the unquantized FP32 recurrence is ~0.13 / ~0.010).
    for res_fmt in (kernel.RES_FP32, kernel.RES_FP16, kernel.RES_FP8, kernel.RES_INT8):
        label = kernel.residual_label(res_fmt)
        # torch-reference full decode
        o_ref, h_ref, _ = fused_decode(h0, q, k, v, alpha, beta, 8, res_fmt)
        # triton full decode
        o_t, h_t, s_t, e_t, es_t = kernel.gdn_quant_decode_fused(
            h0c, qc, kc, vc, ac, bc, T, bits=8, res_fmt=res_fmt)
        h_t_deq = h_t.float() * s_t[..., None, :]
        err_o = (o_t - o_ref.to(dev)).abs().max().item()
        err_h = (h_t_deq - h_ref.to(dev)).abs().max().item()
        print(f"  int8+EF [{label:<5}]  o max err = {err_o:.3e}  h max err = {err_h:.3e}")
        assert err_o < 0.2 and err_h < 0.05, f"int8+EF [{label}] mismatch"


def main():
    B, HV, K, V, T = 1, 32, 128, 128, 64
    h0, q, k, v, alpha, beta = make_inputs(T, B, HV, K, V)

    test_quant_matches_quantpy()

    o_oracle, h_oracle, e_oracle = oracle_decode(h0, q, k, v, alpha, beta)
    o_fused, h_fused, e_fused = fused_decode(h0, q, k, v, alpha, beta)

    o_err = (o_fused - o_oracle).abs().max().item()
    h_err = (h_fused - h_oracle).abs().max().item()
    e_err = (e_fused - e_oracle).abs().max().item()
    o_rel = rel(o_fused, o_oracle)
    h_rel = rel(h_fused, h_oracle)

    print("=== torch-reference fused kernel vs reference.py oracle ===")
    print(f"o   max abs err = {o_err:.3e}   rel = {o_rel:.3e}")
    print(f"h   max abs err = {h_err:.3e}   rel = {h_rel:.3e}")
    print(f"e   max abs err = {e_err:.3e}")

    # The fused path uses the batched einsum (bhkv,bhk->bhv) while the oracle uses
    # the unbatched (hkv,hk->hv).  These are bit-identical on torch 2.8 but differ
    # by ~1e-6/step on newer torch (2.14) where einsum changed reduction order; the
    # near-unstable recurrence (alpha=0.995) amplifies that to ~5e-4 relative over
    # 64 steps.  A relative bound separates that from a real bug (wrong einsum /
    # decay / scale -> rel >> 1e-2).
    ok = o_rel < 5e-3 and h_rel < 5e-3 and e_err < 1e-1
    print("PASS" if ok else "FAIL")

    # ---- Triton kernel validation (only if CUDA + Triton available) ----
    if kernel.HAS_TRITON and torch.cuda.is_available():
        print("\n=== Triton kernels vs torch reference (GPU) ===")
        gpu_check(h0, q, k, v, alpha, beta, T)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
