"""Experiment 2: bits sweep + error feedback, at worst-case (slow) decay.

Maps the INT-k error floor for recurrent-state quantization, with and without
error feedback, at alpha=0.995 (the compounding regime). Also tests per-channel
scaling (DAMP's "high-risk channel" intuition) vs. per-head uniform.
"""

import torch

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statequant.reference import gdn_decode_batched, gdn_step_batched
from statequant.quant import uniform_quantize, ErrorFeedback

torch.manual_seed(0)


def rel(x, y):
    return (x - y).norm() / (y.norm() + 1e-12)


def per_axis_quantize(x, bits, dim):
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-12) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale


def run(h0, q, k, v, alpha, beta, quant_fn, use_ef=False):
    T = q.shape[0]
    h = h0.clone()
    fb = ErrorFeedback(h0.shape) if use_ef else None
    os = []
    for t in range(T):
        h, o = gdn_step_batched(h, q[t], k[t], v[t], alpha[t], beta[t])
        if use_ef:
            h = fb.step(h, quant_fn)
        else:
            h = quant_fn(h)
        os.append(o)
    return torch.stack(os)


def main():
    HV, K, V, T = 8, 128, 128, 2000
    q = torch.randn(T, HV, K)
    k = torch.randn(T, HV, K)
    k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v = torch.randn(T, HV, V)
    alpha = torch.full((T, HV), 0.995)
    beta = torch.rand(T, HV).sigmoid()
    h0 = torch.randn(HV, K, V) * 0.1
    o_ref, _ = gdn_decode_batched(h0, q, k, v, alpha, beta)

    print("=== alpha=0.995 (worst-case compounding regime) ===")
    print(f"{'scheme':<28} {'@500':>10} {'@2000':>10}  bits/val (state)")
    rows = []
    for bits in (3, 4, 5, 6, 8):
        for ef in (False, True):
            qf = lambda x, b=bits: uniform_quantize(x, b)[0]
            o_q = run(h0, q, k, v, alpha, beta, qf, use_ef=ef)
            name = f"int{bits}-uniform" + ("+EF" if ef else "")
            e500, e2000 = rel(o_q[500], o_ref[500]).item(), rel(o_q[1999], o_ref[1999]).item()
            print(f"{name:<28} {e500:>10.2e} {e2000:>10.2e}")
    # per-channel scaling (DAMP intuition)
    for dim, dname in [(-2, "per-K"), (-1, "per-V")]:
        qf = lambda x, b=4, d=dim: per_axis_quantize(x, b, d)
        o_q = run(h0, q, k, v, alpha, beta, qf)
        print(f"{'int4-' + dname + '-channel':<28} {rel(o_q[500], o_ref[500]).item():>10.2e} {rel(o_q[1999], o_ref[1999]).item():>10.2e}")
        o_q = run(h0, q, k, v, alpha, beta, qf, use_ef=True)
        print(f"{'int4-' + dname + '-channel+EF':<28} {rel(o_q[500], o_ref[500]).item():>10.2e} {rel(o_q[1999], o_ref[1999]).item():>10.2e}")


if __name__ == "__main__":
    main()
