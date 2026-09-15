"""Experiment 2b: clean comparison of quantization granularity x error feedback.

At worst-case alpha=0.995. Reports steady-state (t=2000) output error for:
  granularity: per-head, per-K, per-V, block16, block32
  x error-feedback: off / on
across int3..int8.
"""

import torch

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statequant.reference import gdn_step_batched, gdn_decode_batched
from statequant.quant import uniform_quantize, per_axis_quantize, blockscale_quantize, ErrorFeedback

torch.manual_seed(0)


def rel(x, y):
    return (x - y).norm() / (y.norm() + 1e-12)


def make_qf(gran, bits):
    if gran == "per-head":
        return lambda x: uniform_quantize(x, bits)[0]
    if gran == "per-K":
        return lambda x: per_axis_quantize(x, bits, -2)
    if gran == "per-V":
        return lambda x: per_axis_quantize(x, bits, -1)
    if gran.startswith("block"):
        b = int(gran[5:])
        return lambda x: blockscale_quantize(x, bits, b)[0]
    raise ValueError(gran)


def run(h0, q, k, v, alpha, beta, qf, ef):
    T = q.shape[0]
    h = h0.clone()
    fb = ErrorFeedback(h0.shape) if ef else None
    for t in range(T):
        h, o = gdn_step_batched(h, q[t], k[t], v[t], alpha[t], beta[t])
        h = fb.step(h, qf) if ef else qf(h)
    return h


def main():
    HV, K, V, T = 8, 128, 128, 2000
    q = torch.randn(T, HV, K)
    k = torch.randn(T, HV, K); k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v = torch.randn(T, HV, V)
    alpha = torch.full((T, HV), 0.995)
    beta = torch.rand(T, HV).sigmoid()
    h0 = torch.randn(HV, K, V) * 0.1
    _, h_ref = gdn_decode_batched(h0, q, k, v, alpha, beta)

    print("=== alpha=0.995, final-state error at t=2000 ===")
    print(f"{'scheme':<26} {'state_err':>12} {'gran x EF'}")
    for gran in ["per-head", "per-K", "per-V", "block16", "block32"]:
        for bits in (4,):
            for ef in (False, True):
                qf = make_qf(gran, bits)
                h_q = run(h0, q, k, v, alpha, beta, qf, ef)
                print(f"int{bits}-{gran:<8}{'+EF' if ef else '':<4} {rel(h_q, h_ref).item():>12.2e}")
    print()
    for bits in (3, 4, 5, 6, 8):
        for ef in (False, True):
            qf = make_qf("per-head", bits)
            h_q = run(h0, q, k, v, alpha, beta, qf, ef)
            print(f"int{bits}-per-head {'+EF' if ef else '   '}  {rel(h_q, h_ref).item():>12.2e}")


if __name__ == "__main__":
    main()
