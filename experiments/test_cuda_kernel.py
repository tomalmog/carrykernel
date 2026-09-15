"""Correctness test for the CUDA fused GDN + INT8 + error-feedback kernel.

Same oracle as ``experiments/test_kernel.py``: the exact GDN recurrence in
``statequant/reference.py`` composed with ``statequant/quant.py``'s
``ErrorFeedback`` + ``per_axis_quantize(dim=-2)`` (per-V-channel). The CUDA
kernel must reproduce that oracle, and must also agree with the Triton kernel
and the torch reference.

Requires a CUDA GPU with a working nvcc (run it on Modal via
``modal run modal_cuda.py``). Checks, for every residual precision:

  1. CUDA one-step vs the torch reference        (tight: same algorithm)
  2. CUDA full decode vs the reference.py oracle (INT8 noise floor)
  3. CUDA vs Triton full decode                  (two kernels, one spec)
  4. round-half-to-even is exact at .5 boundaries
"""

import torch

from statequant import kernel as tk
from statequant.quant import ErrorFeedback, per_axis_quantize
from statequant.reference import gdn_step_batched
from cuda import binding


def rel(x, y):
    return ((x - y).norm() / (y.norm() + 1e-12)).item()


def make_inputs(T, B, HV, K, V, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(T, B, HV, K)
    k = torch.randn(T, B, HV, K)
    k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v = torch.randn(T, B, HV, V)
    alpha = torch.full((T, B, HV), 0.995)  # slow decay: the compounding regime
    beta = torch.rand(T, B, HV).sigmoid()
    h0 = torch.randn(B, HV, K, V) * 0.1
    return h0, q, k, v, alpha, beta


def oracle_decode(h0, q, k, v, alpha, beta, bits=8):
    """Ground truth: reference GDN + ErrorFeedback + per-V-channel quantize."""
    B, HV, K, V = h0.shape
    assert B == 1, "oracle is the unbatched reference; keep B=1"
    T = q.shape[0]
    _, _, deq0 = tk.per_v_channel_quantize(h0, bits)
    h = deq0[0].clone()
    fb = ErrorFeedback((HV, K, V))
    qf = lambda x: per_axis_quantize(x, bits, -2)
    os_ = []
    for t in range(T):
        h, o = gdn_step_batched(h, q[t, 0], k[t, 0], v[t, 0], alpha[t, 0], beta[t, 0])
        h = fb.step(h, qf)
        os_.append(o)
    return torch.stack(os_).unsqueeze(1), h.unsqueeze(0), fb.e.unsqueeze(0)


def test_round_half_to_even(dev):
    """The kernel must use torch.round semantics (banker's rounding).

    A naive floor(x+0.5) drifts the error-feedback residual over long decodes,
    so pin it directly: build a state whose codes land exactly on .5 boundaries
    and require the CUDA codes to match torch.round exactly.
    """
    B, HV, K, V = 1, 1, 32, 8
    qmax = 127.0
    # The kernel derives scale = amax_K(|target|)/qmax itself, so to land codes
    # exactly on .5 boundaries we pick the *codes* first and invert: with
    # amax = qmax the scale is exactly 1.0, so target == code.
    #
    # Codes ..., -2.5, -1.5, -0.5, 0.5, 1.5, ... exercise both tie directions:
    # round-half-to-even sends 0.5->0 and 1.5->2, whereas floor(x+0.5) (the bug)
    # sends 0.5->1 and 1.5->2. Row 0 is pinned to +qmax to fix scale == 1.0.
    codes = (torch.arange(K, dtype=torch.float32) % 8) - 3.5   # ..., -3.5, ..., 3.5
    target = codes[:, None].repeat(1, V).contiguous()
    target[0, :] = qmax          # pin amax so scale == 1.0 exactly
    scale = target.abs().amax(dim=0).clamp(min=1e-12) / qmax
    assert torch.allclose(scale, torch.ones_like(scale)), "scale must be exactly 1.0"
    codes_ref = torch.round(target / scale[None, :]).clamp(-qmax, qmax)

    # Drive one kernel step with alpha=0, beta=0: h decays to 0, dv = 0, so
    # target == e, i.e. the residual we inject is exactly what gets quantized.
    h_int8 = torch.zeros(B, HV, K, V, dtype=torch.int8, device=dev)
    sc = torch.ones(B, HV, V, device=dev)
    e = target[None, None].to(dev).contiguous()
    q = torch.zeros(B, HV, K, device=dev)
    k = torch.zeros(B, HV, K, device=dev)
    v = torch.zeros(B, HV, V, device=dev)
    alpha = torch.zeros(B, HV, device=dev)
    beta = torch.zeros(B, HV, device=dev)

    _, h_out, s_out, _, _ = binding.gdn_quant_step_cuda(
        h_int8, sc, e, None, q, k, v, alpha, beta, 8, tk.RES_FP32)
    torch.cuda.synchronize()

    got = h_out[0, 0].float().cpu()
    exp = codes_ref
    mismatch = (got - exp).abs().max().item()
    ratios = (target / scale[None, :]).abs()
    n_half = int(((ratios % 1.0) == 0.5).sum())
    # Guard against the test going vacuous again: if the construction stops
    # producing ties it proves nothing, so fail loudly rather than pass.
    assert n_half > 0, "rounding test is vacuous: no exact .5 boundaries constructed"

    # Show that this input actually discriminates the two rounding rules.
    naive = torch.floor(target / scale[None, :] + 0.5).clamp(-qmax, qmax)
    n_diff = int((naive != exp).sum())
    assert n_diff > 0, "test cannot distinguish round-half-even from floor(x+0.5)"

    print(f"  round-half-to-even: {n_half} exact .5 boundaries "
          f"({n_diff} where floor(x+0.5) would differ), max code diff = {mismatch:.1f}")
    assert mismatch == 0.0, "rounding does not match torch.round (round-half-to-even)"


def check_one_step(h0, q, k, v, alpha, beta, dev):
    """CUDA one step vs the torch reference, per residual format (tight bound)."""
    for res_fmt in (tk.RES_FP32, tk.RES_FP16, tk.RES_FP8, tk.RES_INT8):
        label = tk.residual_label(res_fmt)
        h_int8, scale, e, e_scale = tk.init_quant_state(h0, 8, res_fmt)
        args_cpu = (h_int8, scale, e, e_scale, q[0], k[0], v[0], alpha[0], beta[0])
        o_ref, h_ref, s_ref, e_ref, es_ref = tk.gdn_quant_step_torch(*args_cpu, 8, res_fmt)

        cu = [x.to(dev) if torch.is_tensor(x) else x for x in args_cpu]
        o_c, h_c, s_c, e_c, es_c = binding.gdn_quant_step_cuda(*cu, 8, res_fmt)
        torch.cuda.synchronize()

        err_o = (o_c.cpu() - o_ref).abs().max().item()
        # compare dequantized state (code+scale together is the meaningful value)
        h_c_deq = h_c.float().cpu() * s_c.cpu()[..., None, :]
        h_r_deq = h_ref.float() * s_ref[..., None, :]
        err_h = (h_c_deq - h_r_deq).abs().max().item()

        # A handful of codes differ by 1 LSB: the kernel reduces h^T k over K in
        # a different order than torch's einsum, so a value sitting within one
        # float ULP of a rounding tie can land either side. Assert the shape of
        # that disagreement (never more than 1 LSB, and vanishingly rare) rather
        # than just its existence -- a real bug (wrong decay/scale/rounding)
        # would move many codes, and by more than one step.
        code_diff = (h_c.cpu().int() - h_ref.int()).abs()
        n_code_diff = int((code_diff > 0).sum())
        max_code_diff = int(code_diff.max())
        frac = n_code_diff / code_diff.numel()
        print(f"  [{label:<5}] one-step  o err={err_o:.3e}  h err={err_h:.3e}  "
              f"codes differing={n_code_diff}/{code_diff.numel()} "
              f"({frac:.2e}, max {max_code_diff} LSB)")
        assert max_code_diff <= 1, f"codes differ by >1 LSB ({label}): {max_code_diff}"
        assert frac < 1e-4, f"too many differing codes ({label}): {frac:.2e}"
        assert err_o < 1e-3, f"one-step o mismatch ({label}): {err_o}"
        assert err_h < 1e-2, f"one-step h mismatch ({label}): {err_h}"


def check_decode_vs_oracle(h0, q, k, v, alpha, beta, T, dev):
    """CUDA full decode vs the reference.py oracle, and vs Triton."""
    o_oracle, h_oracle, _ = oracle_decode(h0, q, k, v, alpha, beta)

    for res_fmt in (tk.RES_FP32, tk.RES_FP16, tk.RES_FP8, tk.RES_INT8):
        label = tk.residual_label(res_fmt)
        o_c, h_c, s_c, _, _ = binding.gdn_quant_decode_cuda(
            h0.to(dev), q.to(dev), k.to(dev), v.to(dev),
            alpha.to(dev), beta.to(dev), T, 8, res_fmt)
        torch.cuda.synchronize()
        h_c_deq = (h_c.float() * s_c[..., None, :]).cpu()

        o_rel = rel(o_c.cpu(), o_oracle)
        h_rel = rel(h_c_deq, h_oracle)
        print(f"  [{label:<5}] decode T={T} vs oracle   o rel={o_rel:.3e}  h rel={h_rel:.3e}")
        # fp32/fp16 residuals track the oracle closely; fp8/int8 residuals are a
        # deliberate approximation of it, so only the fp32 residual gets the
        # tight bound (the oracle itself carries an fp32 residual).
        bound = 5e-3 if res_fmt in (tk.RES_FP32, tk.RES_FP16) else 1.5e-1
        assert o_rel < bound and h_rel < bound, f"decode mismatch vs oracle ({label})"

        # CUDA vs Triton: two independent implementations of one spec.
        if tk.HAS_TRITON:
            o_t, h_t, s_t, _, _ = tk.gdn_quant_decode_fused(
                h0.to(dev), q.to(dev), k.to(dev), v.to(dev),
                alpha.to(dev), beta.to(dev), T, bits=8, res_fmt=res_fmt)
            torch.cuda.synchronize()
            h_t_deq = h_t.float() * s_t[..., None, :]
            o_rel_t = rel(o_c, o_t)
            h_rel_t = rel(h_c.float() * s_c[..., None, :], h_t_deq)
            print(f"  [{label:<5}] decode T={T} vs Triton   o rel={o_rel_t:.3e}  "
                  f"h rel={h_rel_t:.3e}")
            assert o_rel_t < 5e-3 and h_rel_t < 5e-3, f"CUDA vs Triton mismatch ({label})"


def check_fp32_baseline(h0, q, k, v, alpha, beta, dev):
    h_ref, o_ref = tk.gdn_step_fp32_torch(h0, q[0], k[0], v[0], alpha[0], beta[0])
    h_c, o_c = binding.gdn_fp32_step_cuda(
        h0.to(dev), q[0].to(dev), k[0].to(dev), v[0].to(dev),
        alpha[0].to(dev), beta[0].to(dev))
    torch.cuda.synchronize()
    err_h = (h_c.cpu() - h_ref).abs().max().item()
    err_o = (o_c.cpu() - o_ref).abs().max().item()
    print(f"  fp32 baseline  h err={err_h:.3e}  o err={err_o:.3e}")
    assert err_h < 1e-3 and err_o < 1e-3, "fp32 CUDA kernel mismatch"


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device (run this on Modal: modal run modal_cuda.py)")
        return 0
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print("building CUDA extension...")
    binding.load_extension(verbose=False)
    print("built OK\n")

    B, HV, K, V, T = 1, 32, 128, 128, 64
    h0, q, k, v, alpha, beta = make_inputs(T, B, HV, K, V)

    print("=== rounding semantics ===")
    test_round_half_to_even(dev)

    print("\n=== FP32 baseline kernel vs torch ===")
    check_fp32_baseline(h0, q, k, v, alpha, beta, dev)

    print("\n=== CUDA one step vs torch reference ===")
    check_one_step(h0, q, k, v, alpha, beta, dev)

    print("\n=== CUDA full decode vs oracle / Triton ===")
    check_decode_vs_oracle(h0, q, k, v, alpha, beta, T, dev)

    # batch>1 shape coverage (the oracle is unbatched, so compare to torch ref)
    print("\n=== batch=8 one step vs torch reference ===")
    h0b, qb, kb, vb, ab, bb = make_inputs(1, 8, 32, 128, 128, seed=1)
    check_one_step(h0b, qb, kb, vb, ab, bb, dev)

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
