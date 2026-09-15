// Fused GDN recurrent-state update + INT8 quantization + error feedback (CUDA).
//
// CUDA/C++ port of the Triton kernel in `statequant/kernel.py`, implementing the
// identical math so both are validated against the same oracle
// (`statequant/reference.py` + `statequant/quant.py`).
//
// One decode step, per (batch b, value-head hv), over the [K, V] state matrix:
//
//     h      = h_int8 * scale            // dequantize (per-V-channel scale)
//     h      = alpha * h                 // decay
//     read   = h^T k                     // [V]
//     dv     = beta * (v - read)
//     h      = h + k dv^T                // rank-1 write
//     o      = h^T q                     // [V]
//     target = h + e                     // error feedback: residual added pre-quant
//     scale' = amax_K(|target|) / QMAX   // per-V-channel
//     h_int8 = clamp(rint_half_even(target / scale'))
//     e'     = target - h_int8 * scale'  // new residual, stored at res_fmt
//
// ---------------------------------------------------------------------------
// Parallel decomposition and why it is shaped this way
// ---------------------------------------------------------------------------
// The state is [K, V] row-major: V is the CONTIGUOUS axis. Every reduction the
// math needs (h^T k, h^T q, amax over K) reduces along K -- the *strided* axis.
// That tension is the whole design problem:
//
//   * Assign one thread per K (reduce inside a warp, cheap shuffles) and each
//     lane touches addresses V floats apart -> every global access is a
//     separate 32-byte sector. On a memory-bound kernel that is fatal.
//   * Assign one thread per V (perfectly coalesced, 128-byte transactions) and
//     the K-reduction becomes a loop *inside* each thread -- sequential, but
//     needing no cross-lane communication at all.
//
// This kernel takes the second option. A block owns one (b, hv) pair and a tile
// of BLOCK_V columns; thread `t` owns column `v_tile + t` and walks all K rows
// of it. Consecutive threads therefore touch consecutive V addresses on every
// single load and store, so each warp's access to an int8 row coalesces into
// one 32-byte transaction and an fp32 row into one 128-byte transaction.
//
// The K-reduction (h^T k, h^T q, amax) is then a plain per-thread accumulator
// over the K loop -- no shuffles, no shared memory, no __syncthreads on the
// reduction path. `k` and `q` are indexed by K (shared by every column), so
// they are staged in shared memory once per block and reused K times.
//
// `read = h^T k` must be fully reduced before `dv` -- and hence the updated
// state -- is known, so the state is necessarily touched more than once per
// step. The question is whether the later touches re-read global memory and
// recompute, or whether the column is held locally between them.
//
// Both were built and measured on an A10G (batch 8, int8 residual):
//
//   multi-pass recompute (this version) : 136 us, 10.06M inst, 10.4 MB written
//   per-thread `float h_reg[128]` cache :  266 us,  5.53M inst, 71.1 MB written
//
// The register-cached version halves the instruction count and is 2x SLOWER,
// because 128 floats/thread (512 B) overflows the register budget and nvcc
// spills the array to local memory -- which is DRAM-backed, so `dram__bytes_
// write` went from 10.4 MB to 71.1 MB, ~7x the traffic model. On a
// memory-bound kernel that trade is strictly bad: redundant arithmetic is
// cheaper than spilled traffic.
//
// So the state is deliberately NOT cached per thread and NOT staged in shared
// memory (at K=V=128 a full fp32 tile is 64 KB, over the 48 KB limit, and each
// element is read once and written once per step -- there is no reuse for SMEM
// to exploit). Instead the state streams through global memory and the few
// values that ARE reused get recomputed:
//
//   pass 1: load h, dequantize, decay, accumulate read = h^T k
//   pass 2: reload h (L2-resident), rank-1 write, o = h^T q, +residual, amax
//   pass 3: reload h, requantize, store codes + residual
//
// Passes 2 and 3 hit L2 rather than HBM (lts__t_sector_hit_rate ~31-50%),
// which is why measured HBM traffic stays close to the 1-read + 1-write model
// (38.3 MB read vs 34.1 MB modelled) despite three logical passes.
//
// Shared memory holds only `k` and `q` (2*K floats), which every column in the
// block reuses K times.
//
// ---------------------------------------------------------------------------
// Numerics
// ---------------------------------------------------------------------------
// Rounding must be round-half-to-EVEN to match `torch.round`; a naive
// floor(x+0.5) drifts the error-feedback residual over thousands of steps
// (a real bug found in the Triton version). CUDA's `rintf` is round-to-nearest-
// even under the default rounding mode, so it is exactly right.
//
// Reduction order differs from torch's einsum, so results are bit-close rather
// than bit-identical; the tests assert relative error at the INT8 noise floor
// and that no state code differs by more than 1 LSB.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace {

constexpr float EPSF = 1e-12f;

// Residual storage formats -- must match statequant/kernel.py.
constexpr int RES_FP32 = 0;
constexpr int RES_FP16 = 1;
constexpr int RES_FP8 = 2;  // e5m2, matching the Triton kernel's tl.float8e5
constexpr int RES_INT8 = 3;

// round-half-to-even + clamp to the symmetric signed range. `rintf` uses the
// current rounding mode (round-to-nearest-even by default) == torch.round.
__device__ __forceinline__ float quantize_code(float x, float qmax) {
  return fminf(fmaxf(rintf(x), -qmax), qmax);
}

template <int RES_FMT>
__device__ __forceinline__ float load_residual(const void* e, long idx, float e_scale) {
  if (RES_FMT == RES_FP32) return __ldg(reinterpret_cast<const float*>(e) + idx);
  if (RES_FMT == RES_FP16)
    return __half2float(__ldg(reinterpret_cast<const __half*>(e) + idx));
  if (RES_FMT == RES_FP8) {
    __nv_fp8_e5m2 raw = reinterpret_cast<const __nv_fp8_e5m2*>(e)[idx];
    return static_cast<float>(raw);
  }
  return static_cast<float>(__ldg(reinterpret_cast<const int8_t*>(e) + idx)) * e_scale;
}

template <int RES_FMT>
__device__ __forceinline__ void store_residual(void* e, long idx, float val) {
  if (RES_FMT == RES_FP32) {
    reinterpret_cast<float*>(e)[idx] = val;
  } else if (RES_FMT == RES_FP16) {
    reinterpret_cast<__half*>(e)[idx] = __float2half(val);
  } else if (RES_FMT == RES_FP8) {
    reinterpret_cast<__nv_fp8_e5m2*>(e)[idx] = __nv_fp8_e5m2(val);
  }
  // RES_INT8 is written by the caller (it needs the second amax pass).
}

// ---------------------------------------------------------------------------
// Fused INT8 + error-feedback kernel
// ---------------------------------------------------------------------------
// grid  = (V / BLOCK_V, HV, B);  block = BLOCK_V threads, one per V-column.
template <int RES_FMT>
__global__ __launch_bounds__(256) void gdn_quant_step_kernel(
    const int8_t* __restrict__ h_in,      // [B, HV, K, V]
    const float* __restrict__ scale_in,   // [B, HV, V]
    const void* __restrict__ e_in,        // [B, HV, K, V] @ res_fmt
    const float* __restrict__ e_scale_in, // [B, HV, V]   (int8 residual only)
    const float* __restrict__ q,          // [B, HV, K]
    const float* __restrict__ k,          // [B, HV, K]
    const float* __restrict__ v,          // [B, HV, V]
    const float* __restrict__ alpha,      // [B, HV]
    const float* __restrict__ beta,       // [B, HV]
    float* __restrict__ o,                // [B, HV, V]
    int8_t* __restrict__ h_out,           // [B, HV, K, V]
    float* __restrict__ scale_out,        // [B, HV, V]
    void* __restrict__ e_out,             // [B, HV, K, V] @ res_fmt
    float* __restrict__ e_scale_out,      // [B, HV, V]
    int HV, int K, int V, float qmax) {
  extern __shared__ float smem[];   // [K] keys, then [K] queries
  float* sk = smem;
  float* sq = smem + K;

  const int tid = threadIdx.x;
  const int vcol = blockIdx.x * blockDim.x + tid;
  const int hv = blockIdx.y;
  const int b = blockIdx.z;

  const long hv_base = (long)b * HV + hv;
  const long state_base = hv_base * (long)K * V;
  const long vec_base = hv_base * (long)V;
  const long kq_base = hv_base * (long)K;

  // Stage k and q: indexed by K, so every column in this block reuses them.
  for (int i = tid; i < K; i += blockDim.x) {
    sk[i] = k[kq_base + i];
    sq[i] = q[kq_base + i];
  }
  __syncthreads();

  if (vcol >= V) return;

  const float a = alpha[hv_base];
  const float bta = beta[hv_base];
  const float sc = scale_in[vec_base + vcol];
  const float e_sc = (RES_FMT == RES_INT8) ? e_scale_in[vec_base + vcol] : 0.f;

  // ---- pass 1: read = (alpha * h)^T k -----------------------------------
  // Coalesced: lane t reads column (v_tile + t) of row kk, so a warp's 32
  // lanes touch 32 consecutive int8 -> one 32-byte sector.
  float read = 0.f;
  for (int kk = 0; kk < K; ++kk) {
    const float hval = static_cast<float>(__ldg(h_in + state_base + (long)kk * V + vcol));
    read += hval * sk[kk];
  }
  read *= a * sc;   // fold the decay and the dequant scale into the reduction

  const float dv = bta * (v[vec_base + vcol] - read);

  // ---- pass 2: rank-1 write, output read, error feedback, amax ----------
  float ovals = 0.f;
  float amax = 0.f;
  for (int kk = 0; kk < K; ++kk) {
    const long gidx = state_base + (long)kk * V + vcol;
    const float hval = static_cast<float>(__ldg(h_in + gidx)) * sc * a + sk[kk] * dv;
    ovals += hval * sq[kk];
    const float tgt = hval + load_residual<RES_FMT>(e_in, gidx, e_sc);
    amax = fmaxf(amax, fabsf(tgt));
  }
  o[vec_base + vcol] = ovals;

  const float scale_new = fmaxf(amax / qmax, EPSF);
  scale_out[vec_base + vcol] = scale_new;
  const float inv_scale = 1.0f / scale_new;

  // ---- pass 3: quantize, store codes, form the new residual -------------
  float eamax = 0.f;
  for (int kk = 0; kk < K; ++kk) {
    const long gidx = state_base + (long)kk * V + vcol;
    const float hval = static_cast<float>(__ldg(h_in + gidx)) * sc * a + sk[kk] * dv;
    const float tgt = hval + load_residual<RES_FMT>(e_in, gidx, e_sc);
    const float code = quantize_code(tgt * inv_scale, qmax);
    h_out[gidx] = static_cast<int8_t>(code);
    const float enew = tgt - code * scale_new;
    if (RES_FMT == RES_INT8) {
      eamax = fmaxf(eamax, fabsf(enew));
    } else {
      store_residual<RES_FMT>(e_out, gidx, enew);
    }
  }

  // The int8 residual needs its own amax before it can be encoded, so it costs
  // one more pass over the (L2-resident) state.
  if (RES_FMT == RES_INT8) {
    const float e_scale_new = fmaxf(eamax / qmax, EPSF);
    e_scale_out[vec_base + vcol] = e_scale_new;
    const float inv_e = 1.0f / e_scale_new;
    for (int kk = 0; kk < K; ++kk) {
      const long gidx = state_base + (long)kk * V + vcol;
      const float hval = static_cast<float>(__ldg(h_in + gidx)) * sc * a + sk[kk] * dv;
      const float tgt = hval + load_residual<RES_INT8>(e_in, gidx, e_sc);
      const float code = quantize_code(tgt * inv_scale, qmax);
      const float enew = tgt - code * scale_new;
      reinterpret_cast<int8_t*>(e_out)[gidx] =
          static_cast<int8_t>(quantize_code(enew * inv_e, qmax));
    }
  }
}

// ---------------------------------------------------------------------------
// FP32 baseline kernel (same decomposition, no quantization)
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void gdn_fp32_step_kernel(
    const float* __restrict__ h_in, const float* __restrict__ q,
    const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ alpha, const float* __restrict__ beta,
    float* __restrict__ o, float* __restrict__ h_out,
    int HV, int K, int V) {
  extern __shared__ float smem[];
  float* sk = smem;
  float* sq = smem + K;

  const int tid = threadIdx.x;
  const int vcol = blockIdx.x * blockDim.x + tid;
  const int hv = blockIdx.y;
  const int b = blockIdx.z;

  const long hv_base = (long)b * HV + hv;
  const long state_base = hv_base * (long)K * V;
  const long vec_base = hv_base * (long)V;
  const long kq_base = hv_base * (long)K;

  for (int i = tid; i < K; i += blockDim.x) {
    sk[i] = k[kq_base + i];
    sq[i] = q[kq_base + i];
  }
  __syncthreads();

  if (vcol >= V) return;

  const float a = alpha[hv_base];
  const float bta = beta[hv_base];

  float read = 0.f;
  for (int kk = 0; kk < K; ++kk)
    read += __ldg(h_in + state_base + (long)kk * V + vcol) * sk[kk];
  read *= a;

  const float dv = bta * (v[vec_base + vcol] - read);

  float ovals = 0.f;
  for (int kk = 0; kk < K; ++kk) {
    const long gidx = state_base + (long)kk * V + vcol;
    const float hval = __ldg(h_in + gidx) * a + sk[kk] * dv;
    h_out[gidx] = hval;
    ovals += hval * sq[kk];
  }
  o[vec_base + vcol] = ovals;
}

void check_state(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

void gdn_quant_step_cuda(torch::Tensor h_in, torch::Tensor scale_in, torch::Tensor e_in,
                         torch::Tensor e_scale_in, torch::Tensor q, torch::Tensor k,
                         torch::Tensor v, torch::Tensor alpha, torch::Tensor beta,
                         torch::Tensor o, torch::Tensor h_out, torch::Tensor scale_out,
                         torch::Tensor e_out, torch::Tensor e_scale_out,
                         double qmax, int64_t res_fmt, int64_t block_v) {
  check_state(h_in, "h_in");
  check_state(e_in, "e_in");
  TORCH_CHECK(h_in.dtype() == torch::kInt8, "h_in must be int8");
  TORCH_CHECK(h_in.dim() == 4, "h_in must be [B, HV, K, V]");

  const int B = h_in.size(0), HV = h_in.size(1), K = h_in.size(2), V = h_in.size(3);
  int BV = static_cast<int>(block_v);
  TORCH_CHECK(BV > 0 && BV <= 1024, "block_v must be in (0, 1024]");

  const at::cuda::CUDAGuard guard(h_in.device());
  const dim3 grid((V + BV - 1) / BV, HV, B);
  const size_t smem = (size_t)2 * K * sizeof(float);  // k and q
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH(FMT)                                                                \
  gdn_quant_step_kernel<FMT><<<grid, BV, smem, stream>>>(                          \
      h_in.data_ptr<int8_t>(), scale_in.data_ptr<float>(), e_in.data_ptr(),        \
      e_scale_in.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(),      \
      v.data_ptr<float>(), alpha.data_ptr<float>(), beta.data_ptr<float>(),        \
      o.data_ptr<float>(), h_out.data_ptr<int8_t>(), scale_out.data_ptr<float>(),  \
      e_out.data_ptr(), e_scale_out.data_ptr<float>(), HV, K, V,                   \
      static_cast<float>(qmax))

  switch (res_fmt) {
    case RES_FP32: LAUNCH(RES_FP32); break;
    case RES_FP16: LAUNCH(RES_FP16); break;
    case RES_FP8:  LAUNCH(RES_FP8);  break;
    case RES_INT8: LAUNCH(RES_INT8); break;
    default: TORCH_CHECK(false, "unknown res_fmt ", res_fmt);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_fp32_step_cuda(torch::Tensor h_in, torch::Tensor q, torch::Tensor k,
                        torch::Tensor v, torch::Tensor alpha, torch::Tensor beta,
                        torch::Tensor o, torch::Tensor h_out, int64_t block_v) {
  check_state(h_in, "h_in");
  TORCH_CHECK(h_in.dtype() == torch::kFloat32, "h_in must be fp32");
  TORCH_CHECK(h_in.dim() == 4, "h_in must be [B, HV, K, V]");

  const int B = h_in.size(0), HV = h_in.size(1), K = h_in.size(2), V = h_in.size(3);
  const int BV = static_cast<int>(block_v);
  TORCH_CHECK(BV > 0 && BV <= 1024, "block_v must be in (0, 1024]");

  const at::cuda::CUDAGuard guard(h_in.device());
  const dim3 grid((V + BV - 1) / BV, HV, B);
  const size_t smem = (size_t)2 * K * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  gdn_fp32_step_kernel<<<grid, BV, smem, stream>>>(
      h_in.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(),
      v.data_ptr<float>(), alpha.data_ptr<float>(), beta.data_ptr<float>(),
      o.data_ptr<float>(), h_out.data_ptr<float>(), HV, K, V);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_quant_step", &gdn_quant_step_cuda,
        "Fused GDN state update + INT8 quantize + error feedback (CUDA)");
  m.def("gdn_fp32_step", &gdn_fp32_step_cuda, "Fused FP32 GDN state update (CUDA)");
}
