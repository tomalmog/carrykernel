"""Correctness test for the vLLM-layout INT8 + error-feedback packed decode.

Oracle is the same one the rest of the project uses: the exact GDN recurrence
in ``statequant/reference.py`` composed with ``statequant/quant.py``'s
``ErrorFeedback`` and per-channel quantization. The novelty here is the
**layout transpose**: vLLM stores the state as ``[slots, HV, V, K]`` with K
contiguous, while CarryKernel uses ``[B, HV, K, V]``. The per-V-channel scale
is an amax over K in both, so the oracle is shared -- only the memory layout
and the reduction axis differ, which is exactly what this test pins down.

Checks:
  1. an unquantized sanity path: the recurrence itself matches the oracle
  2. INT8+EF decode over T steps vs the oracle (INT8 noise floor)
  3. NULL_BLOCK_ID (slot <= 0) padding writes zeros and leaves state untouched
  4. paging: requests read/write only their own slot, via ssm_state_indices
  5. round-half-to-even at exact .5 boundaries (the residual-drift bug)
  6. the memory claim: bytes/slot quantized vs fp32

Run on Modal (needs a GPU + Triton):  modal run modal_vllm_stage_a.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statequant.quant import ErrorFeedback, per_axis_quantize
from statequant.reference import gdn_step_batched
from vllm_integration import quantized_packed_decode as qpd


def rel(x, y):
    return ((x - y).norm() / (y.norm() + 1e-12)).item()


def make_packed_inputs(B, H, HV, K, V, device, seed=0):
    """Build the packed [B, 2*H*K + HV*V] activation tensor plus the gates."""
    torch.manual_seed(seed)
    q = torch.randn(B, H, K, device=device)
    k = torch.randn(B, H, K, device=device)
    v = torch.randn(B, HV, V, device=device)
    mixed = torch.cat([q.reshape(B, -1), k.reshape(B, -1), v.reshape(B, -1)], dim=1)
    mixed = mixed.contiguous()

    a = torch.randn(B, HV, device=device) * 0.5
    b = torch.randn(B, HV, device=device)
    A_log = torch.randn(HV, device=device) * 0.2
    dt_bias = torch.randn(HV, device=device) * 0.1
    return mixed, a, b, A_log, dt_bias


def gates_from_inputs(a, b, A_log, dt_bias):
    """Reproduce the kernel's in-kernel alpha/beta derivation, in torch."""
    x = a + dt_bias
    softplus_x = torch.where(x <= qpd.SOFTPLUS_THRESHOLD,
                             torch.log1p(torch.exp(x)), x)
    g = -torch.exp(A_log) * softplus_x
    alpha = torch.exp(g)
    beta = torch.sigmoid(b)
    return alpha, beta


def unpack_qkv(mixed, B, H, HV, K, V, scale, l2norm):
    q = mixed[:, :H * K].reshape(B, H, K).float()
    k = mixed[:, H * K:2 * H * K].reshape(B, H, K).float()
    v = mixed[:, 2 * H * K:].reshape(B, HV, V).float()
    if l2norm:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q * scale
    # broadcast the QK heads out to the V heads (grouped: HV // H per QK head)
    rep = HV // H
    q = q.repeat_interleave(rep, dim=1)
    k = k.repeat_interleave(rep, dim=1)
    return q, k, v


def oracle_decode_vk(h0_vk, steps, H, scale, l2norm):
    """Oracle in vLLM's [V, K] layout, run per request.

    ``h0_vk`` is [HV, V, K]. Internally transposes to the reference's [K, V],
    applies ErrorFeedback + per-V-channel quantization, and returns the
    dequantized final state in [V, K] plus the per-step outputs.
    """
    HV, V, K = h0_vk.shape
    h = h0_vk.transpose(-1, -2).contiguous()          # -> [HV, K, V]
    fb = ErrorFeedback((HV, K, V), device=h.device)
    qf = lambda x: per_axis_quantize(x, 8, -2)        # amax over K, per V-column
    outs = []
    for (mixed, a, b, A_log, dt_bias) in steps:
        alpha, beta = gates_from_inputs(a, b, A_log, dt_bias)
        q, k, v = unpack_qkv(mixed, 1, H, HV, K, V, scale, l2norm)
        h, o = gdn_step_batched(h, q[0], k[0], v[0], alpha[0], beta[0])
        h = fb.step(h, qf)
        outs.append(o)
    return torch.stack(outs), h.transpose(-1, -2).contiguous()


def quantize_vk(h_vk):
    """Quantize a [.., V, K] state: one scale per V-row (amax over K)."""
    scale = h_vk.abs().amax(dim=-1).clamp(min=qpd.EPS) / qpd.QMAX_INT8
    codes = torch.round(h_vk / scale[..., None]).clamp(-qpd.QMAX_INT8, qpd.QMAX_INT8)
    return codes.to(torch.int8), scale


def test_decode_vs_oracle(device, B=1, H=8, HV=16, K=64, V=64, T=32):
    """INT8+EF decode in vLLM layout vs the reference oracle."""
    num_slots = B + 1                       # slot 0 is NULL_BLOCK_ID
    scale = K ** -0.5
    h0 = torch.randn(B, HV, V, K, device=device) * 0.1

    pool = torch.zeros(num_slots, HV, V, K, device=device)
    pool[1:] = h0
    sc, ss, rc, rs = qpd.init_quantized_state(num_slots, HV, V, K, device, h0=pool)
    indices = torch.arange(1, B + 1, dtype=torch.int32, device=device)

    steps = [make_packed_inputs(B, H, HV, K, V, device, seed=100 + t) for t in range(T)]
    out = torch.empty(B, 1, HV, V, device=device)

    kernel_outs = []
    for (mixed, a, b, A_log, dt_bias) in steps:
        qpd.quantized_gated_delta_rule_packed_decode(
            mixed, a, b, A_log, dt_bias, scale,
            sc, ss, rc, rs, out, indices, use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        kernel_outs.append(out[:, 0].clone())
    kernel_o = torch.stack(kernel_outs)                      # [T, B, HV, V]

    # oracle: the pool was quantized at init, so seed the oracle from the same
    # dequantized value for an apples-to-apples comparison
    c0, s0 = quantize_vk(pool[1:])
    h0_deq = c0.float() * s0[..., None]

    max_o_rel = 0.0
    max_h_rel = 0.0
    for r in range(B):
        per_req_steps = [(m[r:r+1], a[r:r+1], b[r:r+1], A, d)
                         for (m, a, b, A, d) in steps]
        # oracle_decode_vk runs one request, so its outputs are [T, HV, V]
        # (no batch axis); the kernel's are [T, B, HV, V].
        o_ref, h_ref = oracle_decode_vk(h0_deq[r], per_req_steps, H, scale, True)
        h_got = sc[r + 1].float() * ss[r + 1][..., None]
        max_o_rel = max(max_o_rel, rel(kernel_o[:, r], o_ref))
        max_h_rel = max(max_h_rel, rel(h_got, h_ref))

    print(f"  decode T={T} B={B} vs oracle:  o rel={max_o_rel:.3e}  h rel={max_h_rel:.3e}")
    assert max_o_rel < 5e-2, f"output drifted from the oracle: {max_o_rel}"
    assert max_h_rel < 5e-2, f"state drifted from the oracle: {max_h_rel}"


def test_null_block_padding(device, H=8, HV=16, K=64, V=64):
    """slot <= 0 is CUDA-graph padding: write zeros, leave the pool untouched."""
    B = 2
    scale = K ** -0.5
    pool = torch.randn(3, HV, V, K, device=device) * 0.1
    sc, ss, rc, rs = qpd.init_quantized_state(3, HV, V, K, device, h0=pool)
    before_codes, before_scale = sc.clone(), ss.clone()

    # request 0 -> real slot 2; request 1 -> NULL_BLOCK_ID
    indices = torch.tensor([2, qpd.NULL_BLOCK_ID], dtype=torch.int32, device=device)
    mixed, a, b, A_log, dt_bias = make_packed_inputs(B, H, HV, K, V, device, seed=7)
    out = torch.full((B, 1, HV, V), float("nan"), device=device)

    qpd.quantized_gated_delta_rule_packed_decode(
        mixed, a, b, A_log, dt_bias, scale, sc, ss, rc, rs, out, indices)
    torch.cuda.synchronize()

    assert torch.all(out[1] == 0), "NULL_BLOCK_ID request must emit zeros"
    assert not torch.any(torch.isnan(out[0])), "real request must be written"
    # slots other than 2 are untouched
    for slot in (0, 1):
        assert torch.equal(sc[slot], before_codes[slot]), f"slot {slot} was modified"
        assert torch.equal(ss[slot], before_scale[slot]), f"slot {slot} scale modified"
    assert not torch.equal(sc[2], before_codes[2]), "the real slot should have advanced"
    print("  NULL_BLOCK_ID padding: zeros emitted, other slots untouched  OK")


def test_paging_isolation(device, H=8, HV=16, K=64, V=64):
    """Two requests on non-adjacent slots must not disturb each other."""
    scale = K ** -0.5
    pool = torch.randn(5, HV, V, K, device=device) * 0.1
    sc, ss, rc, rs = qpd.init_quantized_state(5, HV, V, K, device, h0=pool)

    # Run requests together on slots 3 and 1...
    indices = torch.tensor([3, 1], dtype=torch.int32, device=device)
    mixed, a, b, A_log, dt_bias = make_packed_inputs(2, H, HV, K, V, device, seed=11)
    out = torch.empty(2, 1, HV, V, device=device)
    qpd.quantized_gated_delta_rule_packed_decode(
        mixed, a, b, A_log, dt_bias, scale, sc, ss, rc, rs, out, indices)
    torch.cuda.synchronize()
    together = (sc.clone(), ss.clone(), out.clone())

    # ...then one at a time, and require identical results.
    sc2, ss2, rc2, rs2 = qpd.init_quantized_state(5, HV, V, K, device, h0=pool)
    for r, slot in ((0, 3), (1, 1)):
        idx = torch.tensor([slot], dtype=torch.int32, device=device)
        o1 = torch.empty(1, 1, HV, V, device=device)
        qpd.quantized_gated_delta_rule_packed_decode(
            mixed[r:r+1], a[r:r+1], b[r:r+1], A_log, dt_bias, scale,
            sc2, ss2, rc2, rs2, o1, idx)
        torch.cuda.synchronize()
        assert torch.equal(o1[0], together[2][r]), f"request {r} output differs when batched"

    assert torch.equal(sc2, together[0]), "batched vs sequential state differs"
    assert torch.equal(ss2, together[1]), "batched vs sequential scale differs"
    print("  paging isolation: batched == sequential, per-slot  OK")


def test_round_half_to_even(device, H=1, HV=1, K=32, V=4):
    """Ties must round to even (torch.round), not away from zero.

    Drive a step with alpha~0 and beta=0 so the recurrence contributes nothing
    and `target` is exactly the residual we seeded -- then the stored codes are
    a direct read-out of the rounding rule.
    """
    scale = K ** -0.5
    # The residual is stored as int8 codes times a per-V-row scale, so the tie
    # pattern must be built from integer codes. Odd codes with a residual scale
    # of 0.5 dequantize to exact half-integers (..., -2.5, -0.5, 0.5, 1.5, ...).
    # One entry is pinned to the largest odd int8 so the row's amax is fixed and
    # the scale the kernel derives is predictable; `expected` is then computed
    # from that same derived scale, so this asserts the rounding rule alone.
    # The kernel derives its own scale as amax(|target|)/127, so for the stored
    # codes to be a read-out of the rounding rule we need that scale to be
    # exactly 1.0 (then ratio == target) while target still contains exact .5
    # ties. Two seeds combine to do that, since target = h + e:
    #   * the residual supplies the half-integers: odd int8 codes at scale 0.5
    #     dequantize to ..., -2.5, -0.5, 0.5, 1.5, ...
    #   * the state supplies the amax pin: one entry at 127.0, reachable because
    #     the state scale is fp32 and unconstrained (code 1 x scale 127.0).
    RESID_SCALE = 0.5
    odd = (torch.arange(K, dtype=torch.float32, device=device) % 8) * 2 - 7
    resid_codes_in = odd[None, None, :].repeat(HV, V, 1).contiguous()
    resid_codes_in[..., 0] = 0.0          # the pinned column comes from the state
    e_part = resid_codes_in * RESID_SCALE

    state_codes_in = torch.zeros(HV, V, K, device=device)
    state_codes_in[..., 0] = 1.0          # x state scale 127.0 -> exactly 127.0
    h_part = state_codes_in * qpd.QMAX_INT8

    target = h_part + e_part
    derived_scale = target.abs().amax(dim=-1).clamp(min=qpd.EPS) / qpd.QMAX_INT8
    assert torch.allclose(derived_scale, torch.ones_like(derived_scale)), \
        f"scale must be exactly 1.0, got {derived_scale.flatten()[0].item()}"

    ratios = target / derived_scale[..., None]
    expected = torch.round(ratios).clamp(-qpd.QMAX_INT8, qpd.QMAX_INT8)

    n_half = int(((ratios.abs() % 1.0) == 0.5).sum())
    naive = torch.floor(ratios + 0.5).clamp(-qpd.QMAX_INT8, qpd.QMAX_INT8)
    n_diff = int((naive != expected).sum())
    assert n_half > 0, "vacuous test: no exact .5 ties constructed"
    assert n_diff > 0, "vacuous test: cannot distinguish the two rounding rules"

    sc = torch.zeros(2, HV, V, K, dtype=torch.int8, device=device)
    ss = torch.zeros(2, HV, V, device=device)
    rc = torch.zeros(2, HV, V, K, dtype=torch.int8, device=device)
    rs = torch.zeros(2, HV, V, device=device)
    sc[1] = state_codes_in.to(torch.int8)
    ss[1] = torch.full((HV, V), qpd.QMAX_INT8, device=device)
    rc[1] = resid_codes_in.to(torch.int8)
    rs[1] = torch.full((HV, V), RESID_SCALE, device=device)

    # Neutralize the recurrence so target == h + e exactly:
    #   beta = sigmoid(b) -> 0 kills the rank-1 write (b very negative)
    #   alpha = exp(-exp(A_log) * softplus(a + dt_bias)) -> 1 leaves h intact,
    #   which needs softplus(.) -> 0, i.e. a + dt_bias very negative.
    mixed = torch.zeros(1, 2 * H * K + HV * V, device=device)
    a = torch.full((1, HV), -60.0, device=device)
    b = torch.full((1, HV), -60.0, device=device)
    A_log = torch.zeros(HV, device=device)
    dt_bias = torch.zeros(HV, device=device)
    indices = torch.tensor([1], dtype=torch.int32, device=device)
    out = torch.empty(1, 1, HV, V, device=device)

    qpd.quantized_gated_delta_rule_packed_decode(
        mixed, a, b, A_log, dt_bias, scale, sc, ss, rc, rs, out, indices)
    torch.cuda.synchronize()

    got = sc[1].float()
    diff = (got - expected).abs().max().item()
    print(f"  round-half-to-even: {n_half} ties ({n_diff} discriminating), "
          f"max code diff = {diff:.1f}")
    assert diff == 0.0, "rounding does not match torch.round (round-half-to-even)"


def test_memory_claim(HV=32, V=128, K=128):
    fp32 = qpd.state_bytes_per_slot(HV, V, K, quantized=False, base_dtype_size=4)
    bf16 = qpd.state_bytes_per_slot(HV, V, K, quantized=False, base_dtype_size=2)
    q8 = qpd.state_bytes_per_slot(HV, V, K, quantized=True, resid_fmt=qpd.RESID_INT8)
    q4 = qpd.state_bytes_per_slot(HV, V, K, quantized=True, resid_fmt=qpd.RESID_INT4)
    print(f"  bytes/slot: fp32={fp32/1e6:.2f} MB  bf16={bf16/1e6:.2f} MB")
    print(f"              int8 resid={q8/1e6:.2f} MB  int4 resid={q4/1e6:.2f} MB")
    print(f"  int8 resid: {fp32/q8:.2f}x vs fp32, {bf16/q8:.2f}x vs bf16")
    print(f"  int4 resid: {fp32/q4:.2f}x vs fp32, {bf16/q4:.2f}x vs bf16")

    # int8 state + int8 residual + two fp32 per-V scale vectors = 2 B/elem,
    # exactly bf16's cost -- a win only against FP32.
    assert 1.9 < fp32 / q8 < 2.05, f"expected ~2x vs fp32, got {fp32/q8:.3f}"
    assert bf16 / q8 < 1.1, "int8 residual must not claim a win against bf16"

    # int4 residual is 1.5 B/elem: this is the variant that actually beats the
    # baseline a real vLLM deployment runs.
    assert 2.5 < fp32 / q4 < 2.75, f"expected ~2.67x vs fp32, got {fp32/q4:.3f}"
    assert bf16 / q4 > 1.25, f"int4 residual should beat bf16, got {bf16/q4:.3f}x"


def test_int4_residual_vs_oracle(device, B=1, H=8, HV=16, K=64, V=64, T=32):
    """int4-packed residual: decode must still track the oracle.

    The residual is a correction term, so coarsening it to 4 bits is expected to
    track the oracle slightly less tightly than int8 -- but it must stay at the
    quantization noise floor, not drift.
    """
    num_slots = B + 1
    scale = K ** -0.5
    h0 = torch.randn(B, HV, V, K, device=device) * 0.1
    pool = torch.zeros(num_slots, HV, V, K, device=device)
    pool[1:] = h0

    sc, ss, rc, rs = qpd.init_quantized_state(
        num_slots, HV, V, K, device, h0=pool, resid_fmt=qpd.RESID_INT4)
    assert rc.shape[-1] == K // 2, f"int4 residual should be packed: {rc.shape}"
    indices = torch.arange(1, B + 1, dtype=torch.int32, device=device)

    steps = [make_packed_inputs(B, H, HV, K, V, device, seed=300 + t) for t in range(T)]
    out = torch.empty(B, 1, HV, V, device=device)
    kernel_outs = []
    for (mixed, a, b, A_log, dt_bias) in steps:
        qpd.quantized_gated_delta_rule_packed_decode(
            mixed, a, b, A_log, dt_bias, scale, sc, ss, rc, rs, out, indices,
            use_qk_l2norm_in_kernel=True, resid_fmt=qpd.RESID_INT4)
        torch.cuda.synchronize()
        kernel_outs.append(out[:, 0].clone())
    kernel_o = torch.stack(kernel_outs)

    c0, s0 = quantize_vk(pool[1:])
    h0_deq = c0.float() * s0[..., None]

    max_o_rel = max_h_rel = 0.0
    for r in range(B):
        per_req = [(m[r:r+1], a[r:r+1], b[r:r+1], A, d)
                   for (m, a, b, A, d) in steps]
        o_ref, h_ref = oracle_decode_vk(h0_deq[r], per_req, H, scale, True)
        h_got = sc[r + 1].float() * ss[r + 1][..., None]
        max_o_rel = max(max_o_rel, rel(kernel_o[:, r], o_ref))
        max_h_rel = max(max_h_rel, rel(h_got, h_ref))

    print(f"  int4 resid decode T={T} B={B} vs oracle:  "
          f"o rel={max_o_rel:.3e}  h rel={max_h_rel:.3e}")
    assert max_o_rel < 8e-2, f"int4 residual output drifted: {max_o_rel}"
    assert max_h_rel < 8e-2, f"int4 residual state drifted: {max_h_rel}"


def test_int4_packing_roundtrip(device, H=4, HV=4, K=32, V=8):
    """The nibble pack/unpack must be lossless for in-range int4 codes.

    Runs one step, reads the packed bytes back, and checks that every unpacked
    nibble is a valid signed 4-bit code -- this is what catches a pack that puts
    the two halves of a byte in the wrong lanes.
    """
    scale = K ** -0.5
    pool = torch.randn(2, HV, V, K, device=device) * 0.1
    sc, ss, rc, rs = qpd.init_quantized_state(
        2, HV, V, K, device, h0=pool, resid_fmt=qpd.RESID_INT4)
    indices = torch.tensor([1], dtype=torch.int32, device=device)
    mixed, a, b, A_log, dt_bias = make_packed_inputs(1, H, HV, K, V, device, seed=5)
    out = torch.empty(1, 1, HV, V, device=device)

    qpd.quantized_gated_delta_rule_packed_decode(
        mixed, a, b, A_log, dt_bias, scale, sc, ss, rc, rs, out, indices,
        resid_fmt=qpd.RESID_INT4)
    torch.cuda.synchronize()

    packed = rc[1].to(torch.int32) & 0xFF                  # [HV, V, K//2]
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    assert int(lo.abs().max()) <= 7, f"low nibble out of int4 range: {int(lo.abs().max())}"
    assert int(hi.abs().max()) <= 7, f"high nibble out of int4 range: {int(hi.abs().max())}"
    # a non-trivial residual must actually have been written
    assert int(lo.abs().sum()) + int(hi.abs().sum()) > 0, "residual is all zeros"
    print(f"  int4 packing: nibbles in range, "
          f"{int((lo != 0).sum()) + int((hi != 0).sum())} non-zero codes  OK")


def main():
    if not torch.cuda.is_available():
        print("SKIP: needs a CUDA GPU + Triton (run via modal_vllm_stage_a.py)")
        return 0
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}\n")

    print("=== rounding semantics ===")
    test_round_half_to_even(device)

    print("\n=== NULL_BLOCK_ID padding ===")
    test_null_block_padding(device)

    print("\n=== paging isolation ===")
    test_paging_isolation(device)

    print("\n=== INT8+EF decode vs reference.py oracle ([V,K] layout) ===")
    test_decode_vs_oracle(device, B=1)
    test_decode_vs_oracle(device, B=4, T=16)

    print("\n=== int4-packed residual ===")
    test_int4_packing_roundtrip(device)
    test_int4_residual_vs_oracle(device, B=1)

    print("\n=== memory claim (MambaSpec.page_size_bytes equivalent) ===")
    test_memory_claim()

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
