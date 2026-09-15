"""INT8 + error-feedback recurrent state for vLLM's GDN packed-decode kernel.

Stage A of the vLLM integration: a drop-in quantized variant of vLLM's vendored
kernel ``fused_recurrent_gated_delta_rule_packed_decode``
(``vllm/third_party/flash_linear_attention/ops/fused_recurrent.py``), written
against that kernel's exact invocation contract so it can be swapped in without
touching the call site's shapes or semantics.

Why this is not a copy of ``statequant/kernel.py``
--------------------------------------------------
vLLM's GDN state is ``[num_slots, HV, V, K]`` -- **transposed** relative to
CarryKernel's ``[B, HV, K, V]``, and K (not V) is the contiguous axis. That
changes the kernel's shape in two ways, both favourable:

* CarryKernel's per-V-channel scale is ``amax`` over K. In our layout K is the
  strided axis; in vLLM's it is contiguous, so the same reduction becomes a
  row-wise reduction along contiguous memory.
* The vendored kernel forces ``BK = next_pow2(K)`` with ``NK == 1``, so one
  program instance holds a complete ``[BV, K]`` tile. The per-V amax therefore
  reduces entirely **within one program, in registers** -- no cross-program
  reduction, no atomics, no second pass. (CarryKernel's CUDA kernel needs a
  multi-pass design precisely because it lacks this property.)

The state pool is paged: ``ssm_state_indices[i_n]`` selects a slot, and slot
``<= 0`` is ``NULL_BLOCK_ID`` padding for CUDA-graph capture, which must write
zeros to ``out`` and leave the state untouched.

Quantized state representation (per slot, per value-head, per V-row)
--------------------------------------------------------------------
* ``state_codes``  int8   ``[num_slots, HV, V, K]``  the stored state
* ``state_scale``  fp32   ``[num_slots, HV, V]``     one scale per V-row
* ``resid_codes``  int8   ``[num_slots, HV, V, K]``  error-feedback residual
* ``resid_scale``  fp32   ``[num_slots, HV, V]``     one scale per V-row

Per step, matching ``statequant.quant.ErrorFeedback`` semantics exactly
(``target = h + e; hq = quant(target); e = target - hq``)::

    h      = state_codes * state_scale        # dequantize
    h      = h * exp(g)                       # decay
    h      = h + beta * (v - h^T k) k^T       # delta-rule rank-1 write
    o      = h^T q
    target = h + resid_codes * resid_scale    # residual added pre-quantization
    s'     = amax_K(|target|) / 127
    codes  = round_half_even(target / s')
    e'     = target - codes * s'

Rounding is round-half-to-even (``libdevice.nearbyint``) to match
``torch.round``; ``floor(x + 0.5)`` drifts the residual over long decodes.

Status: standalone and correctness-tested against CarryKernel's oracle. Not yet
wired into vLLM's cache allocation -- that is Stage B (extend ``MambaDType`` in
``vllm/config/cache.py``, thread the extra tensors through ``MambaSpec``, and
relax ``FUSED_GDN_STATE_DTYPES`` in
``vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py``).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only hosts
    triton = None
    tl = None
    libdevice = None
    HAS_TRITON = False

QMAX_INT8 = 127.0
QMAX_INT4 = 7.0
EPS = 1e-12
SOFTPLUS_THRESHOLD = 20.0
NULL_BLOCK_ID = 0

# Residual precision. int8 costs 1 B/element, which makes the whole scheme
# 2 B/element -- exactly bf16's cost, so it is break-even against vLLM's real
# default. int4 packs two codes per byte for 1.5 B/element, an actual win.
RESID_INT8 = 0
RESID_INT4 = 1


if HAS_TRITON:
    # Triton kernels can only read globals declared as tl.constexpr, so the
    # host-side constants above are mirrored here for use inside @triton.jit.
    _QMAX = tl.constexpr(QMAX_INT8)
    _QMAX4 = tl.constexpr(QMAX_INT4)
    _EPS = tl.constexpr(EPS)
    _SOFTPLUS_THRESHOLD = tl.constexpr(SOFTPLUS_THRESHOLD)
    _RESID_INT4 = tl.constexpr(RESID_INT4)

    @triton.jit
    def _round_half_to_even(x):
        """torch.round semantics. libdevice.nearbyint uses the current rounding
        mode, which is round-to-nearest-even."""
        return libdevice.nearbyint(x)

    @triton.jit
    def _quant_int8_kernel(
        # packed inputs, exactly as the vendored kernel receives them
        mixed_qkv, a, b, A_log, dt_bias, scale,
        # quantized state pool (replaces `initial_state`)
        state_codes, state_scale, resid_codes, resid_scale,
        out, ssm_state_indices,
        stride_mixed_qkv_tok, stride_a_tok, stride_b_tok,
        stride_state_slot, stride_resid_slot, stride_scale_slot, stride_indices_seq,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr,
        USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
        SPLIT_BATCH_HEAD_GRID: tl.constexpr,
        RESID_FMT: tl.constexpr,
    ):
        if SPLIT_BATCH_HEAD_GRID:
            i_v, i_hv, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        else:
            i_v, i_nh = tl.program_id(0), tl.program_id(1)
            i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
        p_o = out + (i_n * HV + i_hv) * V + o_v

        # NULL_BLOCK_ID padding (CUDA-graph capture): emit zeros, touch nothing.
        if state_idx <= 0:
            zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
            tl.store(p_o, zero, mask=mask_v)
            return

        # --- state pointers: [slot, HV, V, K], K contiguous ---
        p_state = (state_codes + state_idx * stride_state_slot
                   + i_hv * V * K + o_v[:, None] * K + o_k[None, :])
        # int4 residuals pack two codes per byte along K, so the row is half as
        # wide and each element needs a nibble select.
        if RESID_FMT == _RESID_INT4:
            o_kb = o_k // 2
            p_resid = (resid_codes + state_idx * stride_resid_slot
                       + i_hv * V * (K // 2) + o_v[:, None] * (K // 2) + o_kb[None, :])
        else:
            p_resid = (resid_codes + state_idx * stride_resid_slot
                       + i_hv * V * K + o_v[:, None] * K + o_k[None, :])
        # one scale per V-row
        p_sscale = state_scale + state_idx * stride_scale_slot + i_hv * V + o_v
        p_rscale = resid_scale + state_idx * stride_scale_slot + i_hv * V + o_v

        b_sscale = tl.load(p_sscale, mask=mask_v, other=0.0).to(tl.float32)
        b_rscale = tl.load(p_rscale, mask=mask_v, other=0.0).to(tl.float32)

        # dequantize the state: codes are [BV, K], scale is per V-row
        b_h = tl.load(p_state, mask=mask_h, other=0).to(tl.float32) * b_sscale[:, None]

        # --- q / k / v out of the packed activation tensor ---
        p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
        q_off = i_h * K + o_k
        k_off = (H * K) + i_h * K + o_k
        v_off = (2 * H * K) + i_hv * V + o_v
        b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        # --- gates: alpha and beta are derived in-kernel, as upstream does ---
        a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
        b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
        A_log_val = tl.load(A_log + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= _SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        g_val = -tl.exp(A_log_val) * softplus_x
        beta_val = tl.sigmoid(b_val)

        # --- the GDN recurrence (identical to upstream) ---
        b_h *= tl.exp(g_val)
        b_v -= tl.sum(b_h * b_k[None, :], 1)      # reduce over K (contiguous)
        b_v *= beta_val
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # --- error feedback + requantization ---
        # The whole [BV, K] tile is resident, so amax over K is an in-register
        # row reduction: one scale per V-row, no cross-program communication.
        if RESID_FMT == _RESID_INT4:
            packed = tl.load(p_resid, mask=mask_h, other=0).to(tl.int32) & 0xFF
            # low nibble for even K, high nibble for odd K; sign-extend from 4 bits
            nib = tl.where((o_k[None, :] % 2) == 0, packed & 0xF, (packed >> 4) & 0xF)
            nib = tl.where(nib > 7, nib - 16, nib)
            b_e = nib.to(tl.float32) * b_rscale[:, None]
        else:
            b_e = tl.load(p_resid, mask=mask_h, other=0).to(tl.float32) * b_rscale[:, None]
        target = b_h + b_e

        amax = tl.max(tl.abs(target), 1)                    # [BV]
        s_new = tl.maximum(amax / _QMAX, _EPS)
        codes = _round_half_to_even(target / s_new[:, None])
        codes = tl.minimum(tl.maximum(codes, -_QMAX), _QMAX)
        e_new = target - codes * s_new[:, None]

        e_qmax = _QMAX4 if RESID_FMT == _RESID_INT4 else _QMAX
        e_amax = tl.max(tl.abs(e_new), 1)
        es_new = tl.maximum(e_amax / e_qmax, _EPS)
        e_codes = _round_half_to_even(e_new / es_new[:, None])
        e_codes = tl.minimum(tl.maximum(e_codes, -e_qmax), e_qmax)

        tl.store(p_state, codes.to(tl.int8), mask=mask_h)
        tl.store(p_sscale, s_new, mask=mask_v)
        tl.store(p_rscale, es_new, mask=mask_v)

        if RESID_FMT == _RESID_INT4:
            # Repack pairs of nibbles. Each byte is written once, by the even-K
            # lane, which reads its odd partner's code via a shifted load of the
            # same register tile -- so no cross-lane traffic and no read-modify-
            # write race between the two halves of a byte.
            ec = e_codes.to(tl.int32) & 0xF
            lo = tl.where((o_k[None, :] % 2) == 0, ec, 0)
            hi = tl.where((o_k[None, :] % 2) == 1, ec << 4, 0)
            # sum adjacent K pairs: reshape [BV, K] -> [BV, K//2, 2] and reduce
            byte_vals = tl.sum(tl.reshape(lo + hi, (BV, K // 2, 2)), 2)
            o_kb2 = tl.arange(0, BK // 2)
            mask_kb = o_kb2 < (K // 2)
            p_resid_w = (resid_codes + state_idx * stride_resid_slot
                         + i_hv * V * (K // 2) + o_v[:, None] * (K // 2)
                         + o_kb2[None, :])
            tl.store(p_resid_w, byte_vals.to(tl.int8),
                     mask=mask_v[:, None] & mask_kb[None, :])
        else:
            tl.store(p_resid, e_codes.to(tl.int8), mask=mask_h)


def quantized_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    state_codes: torch.Tensor,
    state_scale: torch.Tensor,
    resid_codes: torch.Tensor,
    resid_scale: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    resid_fmt: int = RESID_INT8,
):
    """INT8 + error-feedback drop-in for ``fused_recurrent_gated_delta_rule_packed_decode``.

    Signature matches the vendored kernel except that the single fp32/bf16
    ``initial_state`` is replaced by the four-tensor quantized representation
    (codes + per-V-row scale, for both the state and the error-feedback
    residual). Updates the state in place and returns ``out``.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for the quantized packed decode")

    if mixed_qkv.dim() != 2:
        raise ValueError(f"mixed_qkv must be 2-D, got {tuple(mixed_qkv.shape)}")
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("mixed_qkv must be contiguous in its last dimension")
    if state_codes.dim() != 4:
        raise ValueError(f"state_codes must be 4-D, got {tuple(state_codes.shape)}")
    if state_codes.stride(-1) != 1:
        raise ValueError("state_codes must be contiguous in its last dimension")
    if state_codes.dtype != torch.int8 or resid_codes.dtype != torch.int8:
        raise ValueError("state_codes and resid_codes must be int8 storage")
    if resid_fmt == RESID_INT4 and resid_codes.shape[-1] * 2 != state_codes.shape[-1]:
        raise ValueError(
            "int4 residual must be packed two codes per byte: expected last dim "
            f"{state_codes.shape[-1] // 2}, got {resid_codes.shape[-1]}")
    if resid_fmt == RESID_INT8 and resid_codes.shape != state_codes.shape:
        raise ValueError("int8 residual must have the same shape as the state")
    if ssm_state_indices.dim() != 1:
        raise ValueError("ssm_state_indices must be 1-D")
    if ssm_state_indices.dtype != torch.int32:
        raise ValueError("ssm_state_indices must be int32")

    B = mixed_qkv.shape[0]
    _, HV, V, K = state_codes.shape
    if out.shape != (B, 1, HV, V):
        raise ValueError(f"out must be {(B, 1, HV, V)}, got {tuple(out.shape)}")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")

    # H is inferred from the packed width, exactly as upstream does.
    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(f"cannot infer H from mixed_qkv width {qkv_dim}")
    H = (qk_dim // 2) // K

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(f"packed decode only supports NK=1 (K={K}, BK={BK})")
    if resid_fmt == RESID_INT4 and K % 2 != 0:
        raise ValueError(f"int4 residual packing needs an even K, got {K}")
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)

    split = B * HV > 65535
    grid = (NV, HV, B) if split else (NV, B * HV)

    _quant_int8_kernel[grid](
        mixed_qkv, a, b, A_log, dt_bias, scale,
        state_codes, state_scale, resid_codes, resid_scale,
        out, ssm_state_indices,
        mixed_qkv.stride(0), a.stride(0), b.stride(0),
        state_codes.stride(0), resid_codes.stride(0), state_scale.stride(0),
        ssm_state_indices.stride(0),
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        SPLIT_BATCH_HEAD_GRID=split,
        RESID_FMT=resid_fmt,
        num_warps=1, num_stages=3,
    )
    return out


def init_quantized_state(num_slots: int, HV: int, V: int, K: int, device,
                         h0: torch.Tensor | None = None,
                         resid_fmt: int = RESID_INT8):
    """Allocate the quantized state pool, optionally seeding slot contents.

    ``h0``, when given, is ``[num_slots, HV, V, K]`` fp32 and is quantized into
    the pool; the residual starts at zero. Returns
    ``(state_codes, state_scale, resid_codes, resid_scale)``. With
    ``resid_fmt=RESID_INT4`` the residual is packed two codes per byte, so its
    last dimension is ``K // 2``.
    """
    resid_k = K // 2 if resid_fmt == RESID_INT4 else K
    state_codes = torch.zeros((num_slots, HV, V, K), dtype=torch.int8, device=device)
    state_scale = torch.zeros((num_slots, HV, V), dtype=torch.float32, device=device)
    resid_codes = torch.zeros((num_slots, HV, V, resid_k), dtype=torch.int8, device=device)
    resid_scale = torch.zeros((num_slots, HV, V), dtype=torch.float32, device=device)

    if h0 is not None:
        scale = h0.abs().amax(dim=-1).clamp(min=EPS) / QMAX_INT8      # [slots, HV, V]
        codes = torch.round(h0 / scale[..., None]).clamp(-QMAX_INT8, QMAX_INT8)
        state_codes.copy_(codes.to(torch.int8))
        state_scale.copy_(scale)
    return state_codes, state_scale, resid_codes, resid_scale


def state_bytes_per_slot(HV: int, V: int, K: int, quantized: bool,
                         base_dtype_size: int = 4, resid_fmt: int = RESID_INT8):
    """Bytes of recurrent state per request slot, for the memory claim.

    Mirrors what vLLM's ``MambaSpec.page_size_bytes`` computes -- deterministic,
    not a benchmark. Note the baseline matters: an int8 state with an int8
    residual is 2 B/element, exactly bf16's cost, so it only beats an FP32
    baseline. The int4 residual (1.5 B/element) is what beats bf16.
    """
    if not quantized:
        return HV * V * K * base_dtype_size
    codes = HV * V * K                                    # int8 state
    resid = HV * V * (K // 2 if resid_fmt == RESID_INT4 else K)
    scales = 2 * HV * V * 4                               # fp32 state + resid scales
    return codes + resid + scales
