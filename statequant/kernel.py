"""Fused GDN recurrent-state update + INT8 quantization with error feedback.

One decode step of the exact Gated DeltaNet recurrence (matching
``statequant/reference.py``), fused with per-V-channel INT8 state quantization
plus LoRC-style error feedback (matching ``statequant/quant.py``'s
``ErrorFeedback``).

State representation between steps (per value-head ``hv``, ``[K, V]`` matrix):

  * ``h_int8``  : packed int8 state  ``[B, HV, K, V]``  (the stored state)
  * ``scale``   : fp32 per-V-channel scale  ``[B, HV, V]``  (one per V-column)
  * ``e``       : error-feedback residual  ``[B, HV, K, V]``, stored at a chosen
    precision ``res_fmt`` (fp32 / fp16 / fp8 / int8). For ``int8`` an extra
    per-V-channel residual scale ``e_scale`` ``[B, HV, V]`` is kept.

Per step the fused op is::

    h     = dequantize(h_int8)          # h = h_int8.float() * scale  (NOT + e)
    h     = alpha * h                   # decay
    read  = h^T k                       # [V]
    dv    = beta * (v - read)
    h     = h + k dv^T                  # rank-1 write
    o     = h^T q                       # output read
    # requantize with error feedback (residual is added *before* quantizing):
    target    = h + e
    scale_new = amax(|target|, over K) / qmax        # per-V-channel
    h_int8    = clamp(round_half_to_even(target / scale_new))
    e_new     = target - h_int8 * scale_new          # carried at res_fmt precision

The residual ``e`` is added to the freshly-updated state immediately before
quantization (not to the recurrence input), which is exactly ``ErrorFeedback.step``
semantics: ``target = x + e; xq = quant(target); e = target - xq``.

Rounding uses round-half-to-even (``torch.round`` semantics) so the Triton kernel
is bit-comparable to the pure-PyTorch reference.

The Triton kernel needs a CUDA GPU to run. On CPU / no-Triton environments the
wrapper transparently falls back to a pure-PyTorch reference with identical math,
so the module is importable and testable everywhere.
"""

from __future__ import annotations

import torch

try:  # Triton is Linux/CUDA-only; not installable on this macOS box.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised on CPU-only hosts
    triton = None
    tl = None
    HAS_TRITON = False

QMAX_INT8 = 127.0
EPS = 1e-12

# Residual (error-feedback) storage formats.
RES_FP32 = 0
RES_FP16 = 1
RES_FP8 = 2
RES_INT8 = 3

_RES_BYTES = {RES_FP32: 4, RES_FP16: 2, RES_FP8: 1, RES_INT8: 1}
_RES_LABEL = {RES_FP32: "fp32", RES_FP16: "fp16", RES_FP8: "fp8", RES_INT8: "int8"}


def residual_bytes(fmt: int) -> int:
    """Bytes per residual element for a given residual format."""
    return _RES_BYTES[fmt]


def residual_label(fmt: int) -> str:
    return _RES_LABEL[fmt]


def residual_dtype(fmt: int):
    if fmt == RES_FP32:
        return torch.float32
    if fmt == RES_FP16:
        return torch.float16
    if fmt == RES_FP8:
        return torch.float8_e5m2
    if fmt == RES_INT8:
        return torch.int8
    raise ValueError(f"unknown residual format {fmt}")


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# --------------------------------------------------------------------------
# Pure-PyTorch references (ground-truth-matching, usable on CPU)
# --------------------------------------------------------------------------


def per_v_channel_quantize(x: torch.Tensor, bits: int = 8):
    """Symmetric per-V-channel quantization: one scale per V-column.

    ``x`` is ``[..., K, V]``; the scale is computed as ``amax(|x|, over K)/qmax``
    so there is one scale per V-column (``[..., V]``). Matches
    ``statequant.quant.per_axis_quantize(x, bits, dim=-2)`` for the dequantized
    result, but also returns the int8 codes and the scale tensor.

    Returns ``(q_int8 [..., K, V], scale [..., V], dequantized [..., K, V])``.
    """
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-2, keepdim=True).clamp(min=EPS) / qmax  # [..., 1, V]
    q = torch.round(x / scale).clamp(-qmax, qmax)
    deq = q * scale
    return q.to(torch.int8), scale.squeeze(-2), deq


def gdn_step_fp32_torch(h, q, k, v, alpha, beta):
    """Fused FP32 GDN step (batched), no quantization. Returns ``(h, o)``."""
    h = alpha[..., None, None] * h
    read = torch.einsum("bhkv,bhk->bhv", h, k)
    dv = beta[..., None] * (v - read)
    h = h + torch.einsum("bhk,bhv->bhkv", k, dv)
    o = torch.einsum("bhkv,bhk->bhv", h, q)
    return h, o


def _dequant_residual_torch(e, e_scale, res_fmt):
    """Convert the stored residual ``e`` back to fp32."""
    if res_fmt == RES_INT8:
        return e.float() * e_scale[..., None, :]
    return e.float()


def _quant_residual_torch(e_fp32, res_fmt):
    """Quantize a fp32 residual to ``res_fmt``. Returns ``(e_stored, e_scale)``."""
    if res_fmt == RES_FP32:
        return e_fp32, None
    if res_fmt == RES_FP16:
        return e_fp32.to(torch.float16), None
    if res_fmt == RES_FP8:
        return e_fp32.to(torch.float8_e5m2), None
    if res_fmt == RES_INT8:
        qmax = QMAX_INT8
        scale = e_fp32.abs().amax(dim=-2).clamp(min=EPS) / qmax  # [..., V]
        q = torch.round(e_fp32 / scale[..., None, :]).clamp(-qmax, qmax)
        return q.to(torch.int8), scale
    raise ValueError(f"unknown residual format {res_fmt}")


def gdn_quant_step_torch(h_int8, scale, e, e_scale, q, k, v, alpha, beta,
                         bits: int = 8, res_fmt: int = RES_FP32):
    """Fused GDN + INT8 quantize + error feedback, one step (batched).

    Inputs:
      h_int8  [B, HV, K, V] int8      stored quantized state
      scale   [B, HV, V]    fp32      per-V-channel scale
      e       [B, HV, K, V]           error-feedback residual (res_fmt dtype)
      e_scale [B, HV, V]    fp32|None residual scale (int8 residual only)
      q, k    [B, HV, K]    fp32      queries / keys
      v       [B, HV, V]    fp32      values
      alpha, beta [B, HV]   fp32      decay / write strength

    Returns ``(o, h_int8_new, scale_new, e_new, e_scale_new)``.
    """
    qmax = 2 ** (bits - 1) - 1
    h = h_int8.float() * scale[..., None, :]              # dequantize (per-V)
    h = alpha[..., None, None] * h                        # decay
    read = torch.einsum("bhkv,bhk->bhv", h, k)            # h^T k
    dv = beta[..., None] * (v - read)                     # delta-rule correction
    h = h + torch.einsum("bhk,bhv->bhkv", k, dv)          # rank-1 write
    o = torch.einsum("bhkv,bhk->bhv", h, q)               # h^T q

    target = h + _dequant_residual_torch(e, e_scale, res_fmt)
    scale_new = target.abs().amax(dim=-2).clamp(min=EPS) / qmax  # [B, HV, V]
    q_int = torch.round(target / scale_new[..., None, :]).clamp(-qmax, qmax)
    h_int8_new = q_int.to(torch.int8)
    deq = q_int.float() * scale_new[..., None, :]
    e_new, e_scale_new = _quant_residual_torch(target - deq, res_fmt)
    return o, h_int8_new, scale_new, e_new, e_scale_new


def init_state(h0: torch.Tensor, bits: int = 8):
    """Quantize an fp32 ``h0 [..., K, V]`` into ``(h_int8, scale, e=0)``.

    The first fused step will dequantize back to ``quantize(h0)`` (the closest
    representable state), so caller must feed the *same* dequantized value to any
    reference oracle for an apples-to-apples comparison.
    """
    h_int8, scale, _ = per_v_channel_quantize(h0, bits)
    e = torch.zeros_like(h0)
    return h_int8, scale, e


def init_quant_state(h0: torch.Tensor, bits: int = 8, res_fmt: int = RES_FP32):
    """Init the full INT8+EF state ``(h_int8, scale, e, e_scale)``."""
    h_int8, scale, _ = per_v_channel_quantize(h0, bits)
    if res_fmt == RES_INT8:
        e = torch.zeros(h0.shape, dtype=torch.int8, device=h0.device)
        e_scale = torch.zeros((*h0.shape[:-2], h0.shape[-1]),
                              dtype=torch.float32, device=h0.device)
        return h_int8, scale, e, e_scale
    e = torch.zeros(h0.shape, dtype=residual_dtype(res_fmt), device=h0.device)
    return h_int8, scale, e, None


# --------------------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _gdn_fp32_kernel(
        h_ptr, q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr,
        o_ptr, h_out_ptr,
        HV, K: tl.constexpr, V: tl.constexpr, BLOCK_V: tl.constexpr,
    ):
        """Fused FP32 GDN step. One program per (b, hv, V-tile); full K in regs."""
        pid_b = tl.program_id(0)
        pid_hv = tl.program_id(1)
        pid_v = tl.program_id(2)

        offs_k = tl.arange(0, K)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V

        a = tl.load(alpha_ptr + pid_b * HV + pid_hv)
        b = tl.load(beta_ptr + pid_b * HV + pid_hv)

        qbase = pid_b * HV * K + pid_hv * K
        q = tl.load(q_ptr + qbase + offs_k)
        k = tl.load(k_ptr + qbase + offs_k)

        vbase = pid_b * HV * V + pid_hv * V
        v = tl.load(v_ptr + vbase + offs_v, mask=mask_v, other=0.0)

        hbase = pid_b * HV * K * V + pid_hv * K * V
        h_ptrs = hbase + offs_k[:, None] * V + offs_v[None, :]
        mask_kv = mask_v[None, :]
        h = tl.load(h_ptr + h_ptrs, mask=mask_kv, other=0.0)

        h = h * a
        read = tl.sum(h * k[:, None], axis=0)
        dv = b * (v - read)
        h = h + k[:, None] * dv[None, :]
        o = tl.sum(h * q[:, None], axis=0)

        tl.store(o_ptr + vbase + offs_v, o, mask=mask_v)
        tl.store(h_out_ptr + h_ptrs, h, mask=mask_kv)

    @triton.jit
    def _round_half_to_even(r):
        """torch.round semantics: round half to even (banker's rounding)."""
        flr = tl.floor(r)
        frac = r - flr
        is_even = (flr.to(tl.int32) & 1) == 0
        return tl.where(frac < 0.5, flr,
                        tl.where(frac > 0.5, flr + 1.0,
                                 tl.where(is_even, flr, flr + 1.0)))

    @triton.jit
    def _gdn_quant_kernel(
        h_ptr, scale_ptr, e_ptr, e_scale_ptr,
        q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr,
        o_ptr, h_out_ptr, scale_out_ptr, e_out_ptr, e_scale_out_ptr,
        HV, K: tl.constexpr, V: tl.constexpr,
        QMAX: tl.constexpr, BLOCK_V: tl.constexpr, RES_FMT: tl.constexpr,
    ):
        """Fused GDN step + per-V-channel INT8 quantization + error feedback.

        ``RES_FMT`` selects the residual storage precision:
        0=fp32, 1=fp16, 2=fp8(e5m2), 3=int8(per-V scale, uses ``e_scale_ptr``).
        """
        pid_b = tl.program_id(0)
        pid_hv = tl.program_id(1)
        pid_v = tl.program_id(2)

        offs_k = tl.arange(0, K)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V

        a = tl.load(alpha_ptr + pid_b * HV + pid_hv)
        b = tl.load(beta_ptr + pid_b * HV + pid_hv)

        qbase = pid_b * HV * K + pid_hv * K
        q = tl.load(q_ptr + qbase + offs_k)
        k = tl.load(k_ptr + qbase + offs_k)

        vbase = pid_b * HV * V + pid_hv * V
        v = tl.load(v_ptr + vbase + offs_v, mask=mask_v, other=0.0)
        scale = tl.load(scale_ptr + vbase + offs_v, mask=mask_v, other=1.0)

        hbase = pid_b * HV * K * V + pid_hv * K * V
        h_ptrs = hbase + offs_k[:, None] * V + offs_v[None, :]
        mask_kv = mask_v[None, :]

        h_int = tl.load(h_ptr + h_ptrs, mask=mask_kv, other=0).to(tl.float32)
        h = h_int * scale[None, :]                      # dequantize (per-V)

        # dequantize the residual  (RES_FMT: 0=fp32 1=fp16 2=fp8 3=int8)
        if RES_FMT == 3:
            e_int = tl.load(e_ptr + h_ptrs, mask=mask_kv, other=0).to(tl.float32)
            e_sc = tl.load(e_scale_ptr + vbase + offs_v, mask=mask_v, other=0.0)
            e = e_int * e_sc[None, :]
        else:
            e = tl.load(e_ptr + h_ptrs, mask=mask_kv, other=0.0).to(tl.float32)

        # --- GDN update ---
        h = h * a
        read = tl.sum(h * k[:, None], axis=0)
        dv = b * (v - read)
        h = h + k[:, None] * dv[None, :]
        o = tl.sum(h * q[:, None], axis=0)
        tl.store(o_ptr + vbase + offs_v, o, mask=mask_v)

        # --- requantize with error feedback ---
        target = h + e
        amax_abs = tl.max(tl.abs(target), axis=0)       # over K, per V-column
        scale_new = tl.maximum(amax_abs / QMAX, 1e-12)
        r = target / scale_new[None, :]
        q_int = _round_half_to_even(r)
        q_int = tl.minimum(tl.maximum(q_int, -QMAX), QMAX)
        deq = q_int * scale_new[None, :]
        e_new = target - deq

        tl.store(h_out_ptr + h_ptrs, q_int.to(tl.int8), mask=mask_kv)
        tl.store(scale_out_ptr + vbase + offs_v, scale_new, mask=mask_v)

        if RES_FMT == 3:
            e_amax = tl.max(tl.abs(e_new), axis=0)
            e_scale_new = tl.maximum(e_amax / QMAX, 1e-12)
            er = e_new / e_scale_new[None, :]
            e_q = _round_half_to_even(er)
            e_q = tl.minimum(tl.maximum(e_q, -QMAX), QMAX)
            tl.store(e_out_ptr + h_ptrs, e_q.to(tl.int8), mask=mask_kv)
            tl.store(e_scale_out_ptr + vbase + offs_v, e_scale_new, mask=mask_v)
        elif RES_FMT == 1:
            tl.store(e_out_ptr + h_ptrs, e_new.to(tl.float16), mask=mask_kv)
        elif RES_FMT == 2:
            tl.store(e_out_ptr + h_ptrs, e_new.to(tl.float8e5), mask=mask_kv)
        else:
            tl.store(e_out_ptr + h_ptrs, e_new, mask=mask_kv)


# --------------------------------------------------------------------------
# Public fused wrappers (Triton on GPU, torch reference otherwise)
# --------------------------------------------------------------------------


def _prep(h_int8, scale, e, e_scale, q, k, v, alpha, beta, res_fmt):
    """Cast/contiguity-check inputs and derive the output shapes/dtypes."""
    assert h_int8.dtype == torch.int8, f"h_int8 must be int8, got {h_int8.dtype}"
    assert h_int8.dim() == 4, f"expected [B, HV, K, V], got {h_int8.shape}"
    B, HV, K, V = h_int8.shape
    dev = h_int8.device
    fp32 = dict(device=dev, dtype=torch.float32)

    scale = scale.contiguous().float()
    e = e.contiguous()
    if e_scale is not None:
        e_scale = e_scale.contiguous().float()
    q = q.contiguous().float()
    k = k.contiguous().float()
    v = v.contiguous().float()
    alpha = alpha.contiguous().float()
    beta = beta.contiguous().float()
    h_int8 = h_int8.contiguous()

    o = torch.empty((B, HV, V), **fp32)
    h_out = torch.empty((B, HV, K, V), dtype=torch.int8, device=dev)
    scale_out = torch.empty((B, HV, V), **fp32)
    e_out = torch.empty((B, HV, K, V), dtype=e.dtype, device=dev)
    e_scale_out = None
    if res_fmt == RES_INT8:
        e_scale_out = torch.empty((B, HV, V), **fp32)
    return B, HV, K, V, dev, (h_int8, scale, e, e_scale, q, k, v, alpha, beta), (
        o, h_out, scale_out, e_out, e_scale_out)


def gdn_quant_step_fused(h_int8, scale, e, e_scale, q, k, v, alpha, beta,
                         bits: int = 8, res_fmt: int = RES_FP32, BLOCK_V: int = 32):
    """Fused GDN + INT8 quantize + error feedback (one decode step).

    Uses the Triton kernel when a CUDA GPU is available; otherwise falls back to
    the exact-math PyTorch reference. ``res_fmt`` selects the residual precision
    (``RES_FP32``/``RES_FP16``/``RES_FP8``/``RES_INT8``).
    """
    B, HV, K, V, dev, inputs, outputs = _prep(
        h_int8, scale, e, e_scale, q, k, v, alpha, beta, res_fmt)
    o, h_out, scale_out, e_out, e_scale_out = outputs
    if not HAS_TRITON or not dev.type == "cuda":
        h_int8, scale, e, e_scale, q, k, v, alpha, beta = inputs
        o, h_out, scale_out, e_out, e_scale_out = gdn_quant_step_torch(
            h_int8, scale, e, e_scale, q, k, v, alpha, beta, bits, res_fmt)
        return o, h_out, scale_out, e_out, e_scale_out

    h_int8, scale, e, e_scale, q, k, v, alpha, beta = inputs
    if e_scale is None:
        e_scale = torch.zeros((B, HV, V), device=dev, dtype=torch.float32)
    if e_scale_out is None:
        e_scale_out = torch.empty((B, HV, V), device=dev, dtype=torch.float32)
    grid = (B, HV, _cdiv(V, BLOCK_V))
    _gdn_quant_kernel[grid](
        h_int8, scale, e, e_scale, q, k, v, alpha, beta,
        o, h_out, scale_out, e_out, e_scale_out,
        HV, K=K, V=V, QMAX=float(2 ** (bits - 1) - 1),
        BLOCK_V=BLOCK_V, RES_FMT=res_fmt,
    )
    return o, h_out, scale_out, e_out, e_scale_out if res_fmt == RES_INT8 else None


def gdn_step_fused_fp32(h, q, k, v, alpha, beta, BLOCK_V: int = 32):
    """Fused FP32 GDN step (baseline). Returns ``(h, o)``."""
    assert h.dim() == 4
    B, HV, K, V = h.shape
    dev = h.device
    fp32 = dict(device=dev, dtype=torch.float32)
    h = h.contiguous().float()
    q = q.contiguous().float()
    k = k.contiguous().float()
    v = v.contiguous().float()
    alpha = alpha.contiguous().float()
    beta = beta.contiguous().float()
    o = torch.empty((B, HV, V), **fp32)
    h_out = torch.empty((B, HV, K, V), **fp32)
    if not HAS_TRITON or not dev.type == "cuda":
        h_out, o = gdn_step_fp32_torch(h, q, k, v, alpha, beta)
        return h_out, o
    grid = (B, HV, _cdiv(V, BLOCK_V))
    _gdn_fp32_kernel[grid](
        h, q, k, v, alpha, beta, o, h_out,
        HV, K=K, V=V, BLOCK_V=BLOCK_V,
    )
    return h_out, o


def gdn_quant_decode_fused(h0, q, k, v, alpha, beta, T, bits: int = 8,
                           res_fmt: int = RES_FP32, BLOCK_V: int = 32):
    """Run the fused INT8+EF decode over T steps (batched, ``[T, B, HV, *]``).

    ``h0`` is ``[B, HV, K, V]`` fp32; ``q/k`` are ``[T, B, HV, K]``,
    ``v`` is ``[T, B, HV, V]``, ``alpha/beta`` are ``[T, B, HV]``.

    Returns ``(o [T, B, HV, V], h_int8 [B, HV, K, V], scale [B, HV, V], e, e_scale)``.
    """
    h_int8, scale, e, e_scale = init_quant_state(h0, bits, res_fmt)
    os = []
    for t in range(T):
        o, h_int8, scale, e, e_scale = gdn_quant_step_fused(
            h_int8, scale, e, e_scale, q[t], k[t], v[t], alpha[t], beta[t],
            bits=bits, res_fmt=res_fmt, BLOCK_V=BLOCK_V)
        os.append(o)
    return torch.stack(os), h_int8, scale, e, e_scale
