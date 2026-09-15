"""Exact Gated DeltaNet (GDN) recurrent state update.

Mirrors fla's `fused_recurrent_gated_delta_rule` kernel, default layout
(`state_v_first=False`, state is ``[K, V]``), so our reference is bit-comparable
to what Qwen3.5-family models actually execute during decode.

Per step (per head):
    h = alpha * h                              # decay  (alpha = exp(g_log))
    read = h^T k                               # [V]
    dv   = beta * (v - read)                   # delta-rule correction
    h   += k dv^T                              # rank-1 write (outer product)
    o    = h^T q                               # read

where h is [K, V], k/q are [K], v is [V], alpha/beta are scalars per head.
"""

import torch


def gdn_step_batched(
    h: torch.Tensor,  # [HV, K, V] fp32
    q: torch.Tensor,  # [HV, K]
    k: torch.Tensor,  # [HV, K]
    v: torch.Tensor,  # [HV, V]
    alpha: torch.Tensor,  # [HV]  decay factor in (0, 1]
    beta: torch.Tensor,  # [HV]  write strength in (0, 1] (post-sigmoid)
) -> tuple[torch.Tensor, torch.Tensor]:
    """One GDN recurrence step, batched over value-heads HV. Returns (h, o)."""
    h = alpha[:, None, None] * h
    read = torch.einsum("hkv,hk->hv", h, k)  # h^T k
    dv = beta[:, None] * (v - read)          # [HV, V]
    h = h + torch.einsum("hk,hv->hkv", k, dv)  # rank-1 write
    o = torch.einsum("hkv,hk->hv", h, q)     # h^T q
    return h, o


def gdn_decode_batched(
    h0: torch.Tensor,  # [HV, K, V]
    q: torch.Tensor,  # [T, HV, K]
    k: torch.Tensor,  # [T, HV, K]
    v: torch.Tensor,  # [T, HV, V]
    alpha: torch.Tensor,  # [T, HV]
    beta: torch.Tensor,  # [T, HV]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full decode over T steps. Returns (o [T,HV,V], h_final [HV,K,V])."""
    HV, K, V = h0.shape
    T = q.shape[0]
    h = h0.clone()
    os = []
    for t in range(T):
        h, o = gdn_step_batched(h, q[t], k[t], v[t], alpha[t], beta[t])
        os.append(o)
    return torch.stack(os), h


def validate_reference(seed: int = 0, HV: int = 4, K: int = 16, V: int = 16, T: int = 32):
    """Naive per-head loop vs. batched einsum — must match exactly."""
    torch.manual_seed(seed)
    h0 = torch.randn(HV, K, V)
    q = torch.randn(T, HV, K)
    k = torch.randn(T, HV, K)
    v = torch.randn(T, HV, V)
    alpha = torch.rand(T, HV) * 0.5 + 0.5
    beta = torch.rand(T, HV).sigmoid()

    # batched
    o_b, h_b = gdn_decode_batched(h0, q, k, v, alpha, beta)

    # naive per head
    o_n = torch.zeros_like(o_b)
    h_n = h0.clone()
    for t in range(T):
        for hv in range(HV):
            h_hv = h_n[hv].clone()
            h_hv = alpha[t, hv] * h_hv
            read = h_hv.T @ k[t, hv]
            dv = beta[t, hv] * (v[t, hv] - read)
            h_hv = h_hv + torch.outer(k[t, hv], dv)
            o_n[t, hv] = h_hv.T @ q[t, hv]
            h_n[hv] = h_hv

    assert torch.allclose(o_b, o_n, atol=1e-5), "output mismatch"
    assert torch.allclose(h_b, h_n, atol=1e-5), "state mismatch"
    return True
