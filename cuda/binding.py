"""Thin Python wrapper around the CUDA fused GDN-state kernel.

JIT-compiles ``gdn_state_kernel.cu`` via ``torch.utils.cpp_extension.load`` on
first use and exposes the same call signature as the Triton path in
``statequant/kernel.py``, so the two can be swapped in the tests and benchmark.

The extension needs a CUDA GPU *and* a real ``nvcc``. Where either is missing
(e.g. a macOS dev box, or a torch wheel image with no CUDA toolkit) importing
this module still works; calling into it raises :class:`CudaUnavailable`.
"""

from __future__ import annotations

import os
import threading

import torch

from statequant import kernel as triton_kernel

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, "gdn_state_kernel.cu")

# sm_86 = A10G (the benchmark GPU); sm_80 = A100, sm_90 = H100.
_DEFAULT_ARCHS = ("80", "86", "90")

_ext = None
_ext_lock = threading.Lock()


class CudaUnavailable(RuntimeError):
    """Raised when the CUDA extension cannot be built or run here."""


def _gencode_flags(archs=_DEFAULT_ARCHS):
    flags = []
    for a in archs:
        flags += ["-gencode", f"arch=compute_{a},code=sm_{a}"]
    return flags


def load_extension(verbose: bool = False, archs=None):
    """Build (once) and return the compiled CUDA extension module."""
    global _ext
    if _ext is not None:
        return _ext
    with _ext_lock:
        if _ext is not None:
            return _ext
        if not torch.cuda.is_available():
            raise CudaUnavailable("no CUDA device available")
        from torch.utils.cpp_extension import load

        if archs is None:
            # Compile for the device we are actually on, plus the common ones.
            major, minor = torch.cuda.get_device_capability(0)
            archs = tuple(dict.fromkeys((f"{major}{minor}",) + _DEFAULT_ARCHS))
        try:
            _ext = load(
                name="carrykernel_gdn_cuda",
                sources=[_SOURCE],
                extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + _gencode_flags(archs),
                extra_cflags=["-O3"],
                verbose=verbose,
            )
        except Exception as exc:  # compile failure -> a clear, actionable error
            raise CudaUnavailable(f"failed to build the CUDA extension: {exc}") from exc
        return _ext


def _empty_like_residual(e, B, HV, V, res_fmt, dev):
    """Allocate the output residual + residual-scale buffers."""
    e_out = torch.empty_like(e)
    if res_fmt == triton_kernel.RES_INT8:
        e_scale_out = torch.empty((B, HV, V), device=dev, dtype=torch.float32)
    else:
        e_scale_out = torch.empty((B, HV, V), device=dev, dtype=torch.float32)
    return e_out, e_scale_out


def gdn_quant_step_cuda(h_int8, scale, e, e_scale, q, k, v, alpha, beta,
                        bits: int = 8, res_fmt: int = triton_kernel.RES_FP32,
                        BLOCK_V: int = 128):
    """One fused GDN + INT8-quantize + error-feedback step, in CUDA.

    Mirrors ``statequant.kernel.gdn_quant_step_fused``. Returns
    ``(o, h_int8_new, scale_new, e_new, e_scale_new)``; ``e_scale_new`` is
    ``None`` for every residual format except int8.
    """
    ext = load_extension()
    assert h_int8.dtype == torch.int8, f"h_int8 must be int8, got {h_int8.dtype}"
    assert h_int8.dim() == 4, f"expected [B, HV, K, V], got {tuple(h_int8.shape)}"
    B, HV, K, V = h_int8.shape
    dev = h_int8.device

    h_int8 = h_int8.contiguous()
    scale = scale.contiguous().float()
    e = e.contiguous()
    q = q.contiguous().float()
    k = k.contiguous().float()
    v = v.contiguous().float()
    alpha = alpha.contiguous().float()
    beta = beta.contiguous().float()

    # The kernel always reads an e_scale buffer; for non-int8 residuals it is
    # ignored, so a zero tensor keeps the signature uniform.
    if e_scale is None:
        e_scale = torch.zeros((B, HV, V), device=dev, dtype=torch.float32)
    else:
        e_scale = e_scale.contiguous().float()

    o = torch.empty((B, HV, V), device=dev, dtype=torch.float32)
    h_out = torch.empty((B, HV, K, V), device=dev, dtype=torch.int8)
    scale_out = torch.empty((B, HV, V), device=dev, dtype=torch.float32)
    e_out, e_scale_out = _empty_like_residual(e, B, HV, V, res_fmt, dev)

    ext.gdn_quant_step(
        h_int8, scale, e, e_scale, q, k, v, alpha, beta,
        o, h_out, scale_out, e_out, e_scale_out,
        float(2 ** (bits - 1) - 1), int(res_fmt), int(BLOCK_V),
    )
    return (o, h_out, scale_out, e_out,
            e_scale_out if res_fmt == triton_kernel.RES_INT8 else None)


def gdn_fp32_step_cuda(h, q, k, v, alpha, beta, BLOCK_V: int = 128):
    """Fused FP32 GDN step (baseline), in CUDA. Returns ``(h_out, o)``."""
    ext = load_extension()
    assert h.dim() == 4, f"expected [B, HV, K, V], got {tuple(h.shape)}"
    B, HV, K, V = h.shape
    dev = h.device

    h = h.contiguous().float()
    q = q.contiguous().float()
    k = k.contiguous().float()
    v = v.contiguous().float()
    alpha = alpha.contiguous().float()
    beta = beta.contiguous().float()

    o = torch.empty((B, HV, V), device=dev, dtype=torch.float32)
    h_out = torch.empty((B, HV, K, V), device=dev, dtype=torch.float32)
    ext.gdn_fp32_step(h, q, k, v, alpha, beta, o, h_out, int(BLOCK_V))
    return h_out, o


def gdn_quant_decode_cuda(h0, q, k, v, alpha, beta, T, bits: int = 8,
                          res_fmt: int = triton_kernel.RES_FP32, BLOCK_V: int = 128):
    """Run the CUDA fused INT8+EF decode over T steps (``[T, B, HV, *]`` inputs)."""
    h_int8, scale, e, e_scale = triton_kernel.init_quant_state(h0, bits, res_fmt)
    os_ = []
    for t in range(T):
        o, h_int8, scale, e, e_scale = gdn_quant_step_cuda(
            h_int8, scale, e, e_scale, q[t], k[t], v[t], alpha[t], beta[t],
            bits=bits, res_fmt=res_fmt, BLOCK_V=BLOCK_V)
        os_.append(o)
    return torch.stack(os_), h_int8, scale, e, e_scale
