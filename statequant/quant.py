"""Recurrent-state quantization schemes.

The state `h` (fp32, [.., K, V]) is the thing being stored between decode
steps. We quantize it to a low-bit representation for storage, and dequantize
back to fp32 for the actual recurrence compute. This is exactly the transfer
point DAMP targets (their "recurrent-state quantization").

Schemes:
  - uniform:    single scale per head (DAMP's "uniform quantization" baseline,
                which they report collapses to near-zero accuracy at INT4).
  - blockscale: per-block scale (NVFP4-style, as used by Minima for weights).
  - error-feedback: uniform/block + LoRC-style residual carry (SAGE lineage).
"""

import torch


def _levels(bits: int) -> int:
    # symmetric signed range: [-2^(b-1), 2^(b-1)-1]
    return 2 ** (bits - 1) - 1  # max magnitude, e.g. 7 for int4


def uniform_quantize(x: torch.Tensor, bits: int, dim=(-1, -2)):
    """Symmetric affine quantization with one scale per head (last two dims).

    For a state `x` of shape [.., K, V], `dim=(-1,-2)` collapses K*V into one
    scale per (..., head). This is DAMP's "uniform quantization" baseline.
    Returns (dequantized, scale).
    """
    qmax = _levels(bits)
    scale = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-12) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    xq = q * scale
    return xq, scale


def per_axis_quantize(x: torch.Tensor, bits: int, dim: int):
    """One scale per slice along `dim` (per-K or per-V channel)."""
    qmax = _levels(bits)
    scale = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-12) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale


def blockscale_quantize(x: torch.Tensor, bits: int, block: int):
    """Per-block scale over the last axis (or 2D blocks for [K,V] state).

    For a [.., K, V] state we block over the trailing (K, V) matrix in tiles of
    `block` x `block` values. Returns dequantized tensor + per-block scale.
    """
    qmax = _levels(bits)
    *lead, K, V = x.shape
    # pad to block multiple
    kp = (K + block - 1) // block * block
    vp = (V + block - 1) // block * block
    xp = torch.zeros(*lead, kp, vp, dtype=x.dtype, device=x.device)
    xp[..., :K, :V] = x
    xr = xp.view(*lead, kp // block, block, vp // block, block)
    scale = xr.abs().amax(dim=(-1, -3), keepdim=True).clamp(min=1e-12) / qmax
    q = torch.round(xr / scale).clamp(-qmax, qmax)
    xq = (q * scale).view(*lead, kp, vp)[..., :K, :V]
    return xq, scale


class ErrorFeedback:
    """LoRC-style residual carry for recurrent state quantization.

    Instead of quantizing h directly, quantize (h + e_prev) and carry the
    residual e = (h + e_prev) - dequant(h + e_prev) to the next step.
    """

    def __init__(self, shape, dtype=torch.float32, device="cpu"):
        self.e = torch.zeros(shape, dtype=dtype, device=device)

    def step(self, x: torch.Tensor, quant_fn):
        """`quant_fn` returns the dequantized tensor. Carries the residual."""
        target = x + self.e
        xq = quant_fn(target)
        self.e = target - xq
        return xq
