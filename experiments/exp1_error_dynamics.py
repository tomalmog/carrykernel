"""Experiment 1: does recurrent-state quantization error compound or wash out?

Decisive CPU test of the DAMP vs. Minima tension:

  - DAMP (2608.27513):  INT4 state quantization -> "near zero" accuracy.
  - Minima (2609.04098): "holds injected noise at a flat plateau ... forgets a
    state impulse within hundreds of steps."

Run the exact GDN recurrence, quantize the state between steps, track error.
Flat error -> Minima right, low-bit state plausible. Growing error -> DAMP's
failure mode is real; then block-scaling / error-feedback must rescue it.
"""

import torch

from statequant.reference import gdn_decode_batched, gdn_step_batched
from statequant.quant import uniform_quantize, blockscale_quantize, ErrorFeedback

torch.manual_seed(0)


def make_inputs(T, HV, K, V, alpha_level):
    q = torch.randn(T, HV, K)
    k = torch.randn(T, HV, K)
    k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # L2-normed keys
    v = torch.randn(T, HV, V)
    alpha = torch.full((T, HV), alpha_level)
    beta = torch.rand(T, HV).sigmoid()
    return q, k, v, alpha, beta


def rel(x, y):
    return (x - y).norm() / (y.norm() + 1e-12)


def run_quantized(h0, q, k, v, alpha, beta, quant_fn):
    HV, K, V = h0.shape
    T = q.shape[0]
    h = h0.clone()
    os = []
    for t in range(T):
        h, o = gdn_step_batched(h, q[t], k[t], v[t], alpha[t], beta[t])
        h = quant_fn(h)
        os.append(o)
    return torch.stack(os)


def error_trajectory(o_q, o_ref, steps=(10, 100, 500, 1999)):
    return [rel(o_q[t], o_ref[t]).item() for t in steps]


def main():
    HV, K, V, T = 8, 128, 128, 2000
    for alpha_level in (0.90, 0.99, 0.995):
        q, k, v, alpha, beta = make_inputs(T, HV, K, V, alpha_level)
        h0 = torch.randn(HV, K, V) * 0.1
        o_ref, _ = gdn_decode_batched(h0, q, k, v, alpha, beta)

        schemes = {
            "fp16": lambda x: x.half().float(),
            "int8-uniform": lambda x: uniform_quantize(x, 8)[0],
            "int4-uniform": lambda x: uniform_quantize(x, 4)[0],
            "int4-block16": lambda x: blockscale_quantize(x, 4, 16)[0],
            "int4-block32": lambda x: blockscale_quantize(x, 4, 32)[0],
        }
        print(f"\n=== alpha={alpha_level} (forgetting ~{1/(1-alpha_level):.0f} steps) ===")
        print(f"{'scheme':<16} {'@10':>10} {'@100':>10} {'@500':>10} {'@2000':>10}")
        for name, qf in schemes.items():
            o_q = run_quantized(h0, q, k, v, alpha, beta, qf)
            errs = error_trajectory(o_q, o_ref)
            print(f"{name:<16} " + " ".join(f"{e:>10.2e}" for e in errs))

        for name, bits, blk in [("ef-int4-uniform", 4, None), ("ef-int4-block16", 4, 16)]:
            qf = (lambda x: uniform_quantize(x, bits)[0]) if blk is None else (
                lambda x: blockscale_quantize(x, bits, blk)[0])
            h = h0.clone()
            fb = ErrorFeedback(h0.shape)
            os = []
            for t in range(T):
                h, o = gdn_step_batched(h, q[t], k[t], v[t], alpha[t], beta[t])
                h = fb.step(h, qf)
                os.append(o)
            errs = error_trajectory(torch.stack(os), o_ref)
            print(f"{name:<16} " + " ".join(f"{e:>10.2e}" for e in errs))

    # impulse forgetting test (Minima's claim)
    print("\n=== impulse forgetting (perturb h0 by +5.0, then exact) ===")
    q, k, v, alpha, beta = make_inputs(T, HV, K, V, 0.995)
    h0 = torch.randn(HV, K, V) * 0.1
    o_ref, _ = gdn_decode_batched(h0, q, k, v, alpha, beta)
    h_imp = h0.clone() + torch.randn(HV, K, V) * 5.0
    o_imp, _ = gdn_decode_batched(h_imp, q, k, v, alpha, beta)
    for t in (1, 10, 50, 100, 200, 500, 1000, 1999):
        print(f"  t={t:<5} out_err={rel(o_imp[t], o_ref[t]).item():.3e}")


if __name__ == "__main__":
    main()
