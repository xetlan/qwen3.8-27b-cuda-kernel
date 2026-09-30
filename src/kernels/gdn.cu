// Gated DeltaNet (linear attention) kernels for Qwen3.5-architecture layers.
//
//   gdn_ab          a/b gate projections (bf16 weights, 96 rows): SIMT GEMV (decode) or tiled
//                   SIMT GEMM (prefill), with the folded input RMSNorm row scale
//   gdn_conv        causal depthwise conv (width 4) + SiLU over [q|k|v], q/k L2-normalisation,
//                   conv-state carry across steps
//   gdn_recurrence  gated delta rule, one CTA per (sequence, value head) with the 128x128 fp32
//                   state held in registers (4 lanes per state row), fused gated RMSNorm * SiLU(z)
//
// The recurrence is sequential over tokens (the prefill cost is a few percent of the GEMMs on
// this model); the chunked WY formulation is listed in docs/OPTIMIZATION_PLAN.md.
#include "qwen38/common.h"
#include "kernels/kernels.h"
#include "kernels/ptx.cuh"

namespace qwen38::kernels {
namespace {

using ptx::as_bf162;

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

__device__ __forceinline__ float row_scale(const float* partials, int blocks, float inv_hidden, float eps) {
  float s = 0.0f;
  for (int b = 0; b < blocks; ++b) s += partials[b];
  return rsqrtf(s * inv_hidden + eps);
}

__device__ __forceinline__ float dot8(uint4 a, uint4 b) {
  const uint32_t av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 x = __bfloat1622float2(as_bf162(av[i])), y = __bfloat1622float2(as_bf162(bv[i]));
    s = fmaf(x.x, y.x, s);
    s = fmaf(x.y, y.y, s);
  }
  return s;
}

// ---------------------------------------------------------------------------------------------
// a/b projections
// ---------------------------------------------------------------------------------------------
constexpr int kAbSmallM = 16;

// One warp per output row; the row is streamed once and dotted with all m (<= 16) tokens.
__global__ void __launch_bounds__(256) gdn_ab_small_kernel(const __nv_bfloat16* __restrict__ x, int m, int hidden,
                                                           const __nv_bfloat16* __restrict__ w, int rows,
                                                           const float* __restrict__ partials, float eps,
                                                           float* __restrict__ out) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= rows) return;
  float acc[kAbSmallM];
#pragma unroll
  for (int t = 0; t < kAbSmallM; ++t) acc[t] = 0.0f;
  const uint4* wr = reinterpret_cast<const uint4*>(w + size_t(row) * hidden);
  for (int c = lane; c < hidden / 8; c += 32) {
    const uint4 wv = wr[c];
#pragma unroll
    for (int t = 0; t < kAbSmallM; ++t) {
      if (t < m) acc[t] += dot8(wv, reinterpret_cast<const uint4*>(x + size_t(t) * hidden)[c]);
    }
  }
  const int blocks = hidden / 128;
  const float inv_hidden = 1.0f / float(hidden);
#pragma unroll
  for (int t = 0; t < kAbSmallM; ++t) {
    if (t >= m) break;
    float v = acc[t];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) out[size_t(t) * rows + row] = v * row_scale(partials + size_t(t) * blocks, blocks, inv_hidden, eps);
  }
}

// 64 tokens x all rows (<= 96) per CTA, K in chunks of 64 staged through shared memory.
__global__ void __launch_bounds__(256) gdn_ab_tiled_kernel(const __nv_bfloat16* __restrict__ x, int m, int hidden,
                                                           const __nv_bfloat16* __restrict__ w, int rows,
                                                           const float* __restrict__ partials, float eps,
                                                           float* __restrict__ out) {
  __shared__ float xs[64][65];
  __shared__ float ws[96][65];
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int t0 = blockIdx.x * 64;
  float acc[4][6];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 6; ++j) acc[i][j] = 0.0f;
  for (int k0 = 0; k0 < hidden; k0 += 64) {
    for (int e = tid; e < 64 * 64; e += 256) {
      const int r = e >> 6, c = e & 63;
      xs[r][c] = t0 + r < m ? __bfloat162float(x[size_t(t0 + r) * hidden + k0 + c]) : 0.0f;
    }
    for (int e = tid; e < 96 * 64; e += 256) {
      const int r = e >> 6, c = e & 63;
      ws[r][c] = r < rows ? __bfloat162float(w[size_t(r) * hidden + k0 + c]) : 0.0f;
    }
    __syncthreads();
#pragma unroll 8
    for (int kk = 0; kk < 64; ++kk) {
      float xv[4], wv[6];
#pragma unroll
      for (int i = 0; i < 4; ++i) xv[i] = xs[tx + 16 * i][kk];
#pragma unroll
      for (int j = 0; j < 6; ++j) wv[j] = ws[ty + 16 * j][kk];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 6; ++j) acc[i][j] = fmaf(xv[i], wv[j], acc[i][j]);
    }
    __syncthreads();
  }
  const int blocks = hidden / 128;
  const float inv_hidden = 1.0f / float(hidden);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int t = t0 + tx + 16 * i;
    if (t >= m) continue;
    const float rs = row_scale(partials + size_t(t) * blocks, blocks, inv_hidden, eps);
#pragma unroll
    for (int j = 0; j < 6; ++j) {
      const int r = ty + 16 * j;
      if (r < rows) out[size_t(t) * rows + r] = acc[i][j] * rs;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Causal conv + SiLU + q/k L2 norm
// ---------------------------------------------------------------------------------------------
constexpr int kConvChunk = 32;

__global__ void __launch_bounds__(128) gdn_conv_kernel(GdnConvParams p, GdnDims d, BatchDesc batch,
                                                       GdnStateDesc st) {
  __shared__ float ys[kConvChunk][129];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int head_block = blockIdx.x;        // 128 channels
  const int c = head_block * 128 + tid;     // channel in [q | k | v]
  const int b = blockIdx.y;
  const int s0 = batch.seq_start[b], s1 = batch.seq_start[b + 1];
  const int slot = batch.seq_slot[b];
  const bool is_qk = head_block < 2 * d.num_k_heads;
  const float out_scale = head_block < d.num_k_heads ? rsqrtf(128.0f) : 1.0f;

  __nv_bfloat16* state = st.conv + (size_t(slot) * st.num_gdn_layers + p.gdn_layer) * 3 * d.conv_dim + c;
  float x0 = __bfloat162float(state[0]);
  float x1 = __bfloat162float(state[d.conv_dim]);
  float x2 = __bfloat162float(state[2 * size_t(d.conv_dim)]);
  const float4 wt = *reinterpret_cast<const float4*>(p.conv_w + size_t(c) * 4);

  for (int t0 = s0; t0 < s1; t0 += kConvChunk) {
    const int n = min(kConvChunk, s1 - t0);
    for (int i = 0; i < n; ++i) {
      const float x = __bfloat162float(p.proj[size_t(t0 + i) * d.ld_proj + c]);
      const float y = silu(wt.x * x0 + wt.y * x1 + wt.z * x2 + wt.w * x);
      x0 = x1;
      x1 = x2;
      x2 = x;
      if (is_qk) {
        ys[i][tid] = y;
      } else {
        p.qkv[size_t(t0 + i) * d.conv_dim + c] = y;
      }
    }
    if (is_qk) {  // block-uniform
      __syncthreads();
      for (int i = warp; i < n; i += 4) {
        float v[4], ss = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = ys[i][lane + 32 * e];
          ss += v[e] * v[e];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
        const float r = rsqrtf(ss + 1e-6f) * out_scale;
#pragma unroll
        for (int e = 0; e < 4; ++e) ys[i][lane + 32 * e] = v[e] * r;
      }
      __syncthreads();
      for (int i = 0; i < n; ++i) p.qkv[size_t(t0 + i) * d.conv_dim + c] = ys[i][tid];
      __syncthreads();
    }
  }
  state[0] = __float2bfloat16_rn(x0);
  state[d.conv_dim] = __float2bfloat16_rn(x1);
  state[2 * size_t(d.conv_dim)] = __float2bfloat16_rn(x2);
}

// ---------------------------------------------------------------------------------------------
// Gated delta rule recurrence + gated RMSNorm
// ---------------------------------------------------------------------------------------------
constexpr int kRecTokens = 16;   // tokens staged per chunk (= warps, one per token in the norm stage)
constexpr int kRecThreads = 512; // 128 state rows (dv) x 4 lanes (dk quarters of 32)
constexpr int kQuarterStride = 36;  // padded fp32 stride of a 32-value quarter (bank spread)

struct RecSmem {
  float q[2][kRecTokens][4 * kQuarterStride];
  float k[2][kRecTokens][4 * kQuarterStride];
  float v[2][kRecTokens][128];
  float decay[2][kRecTokens];
  float beta[2][kRecTokens];
  float o[kRecTokens][128];
};

__device__ __forceinline__ void rec_load_chunk(RecSmem& sm, int buf, const GdnRecurrenceParams& p,
                                               const GdnDims& d, int t0, int n, int head, int khead) {
  const int tid = threadIdx.x;
  // q, k: n tokens x 128 floats = 32 x 16-byte chunks each; v: same
  for (int e = tid; e < n * 32 * 3; e += kRecThreads) {
    const int which = e / (n * 32);  // 0 q, 1 k, 2 v
    const int r = e % (n * 32);
    const int t = r >> 5, chunk = r & 31;
    const size_t row = size_t(t0 + t) * d.conv_dim;
    if (which == 0) {
      ptx::cp_async_16(&sm.q[buf][t][(chunk >> 3) * kQuarterStride + (chunk & 7) * 4],
                       p.qkv + row + khead * 128 + chunk * 4);
    } else if (which == 1) {
      ptx::cp_async_16(&sm.k[buf][t][(chunk >> 3) * kQuarterStride + (chunk & 7) * 4],
                       p.qkv + row + (d.num_k_heads + khead) * 128 + chunk * 4);
    } else {
      ptx::cp_async_16(&sm.v[buf][t][chunk * 4], p.qkv + row + (2 * d.num_k_heads + head) * 128 + chunk * 4);
    }
  }
  if (tid < n) {
    const float* ab = p.ab + size_t(t0 + tid) * 2 * d.num_v_heads;
    sm.beta[buf][tid] = sigmoid(ab[head]);
    sm.decay[buf][tid] = __expf(p.A[head] * softplus(ab[d.num_v_heads + head] + p.dt_bias[head]));
  }
}

__global__ void __launch_bounds__(kRecThreads, 1) gdn_recurrence_kernel(GdnRecurrenceParams p, GdnDims d,
                                                                        BatchDesc batch, GdnStateDesc st) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  RecSmem& sm = *reinterpret_cast<RecSmem*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int head = blockIdx.x;
  const int khead = head / (d.num_v_heads / d.num_k_heads);
  const int b = blockIdx.y;
  const int s0 = batch.seq_start[b], s1 = batch.seq_start[b + 1];
  const int slot = batch.seq_slot[b];
  const int row = warp * 8 + (lane >> 2);  // dv index this thread owns
  const int quarter = lane & 3;            // dk range [32 quarter, 32 quarter + 32)

  float* state = st.recurrent +
                 ((size_t(slot) * st.num_gdn_layers + p.gdn_layer) * d.num_v_heads + head) * 128 * 128 +
                 size_t(row) * 128 + quarter * 32;
  float S[32];
#pragma unroll
  for (int i = 0; i < 32; i += 4) {
    const float4 v = *reinterpret_cast<const float4*>(state + i);
    S[i] = v.x; S[i + 1] = v.y; S[i + 2] = v.z; S[i + 3] = v.w;
  }

  const int total = s1 - s0;
  const int chunks = (total + kRecTokens - 1) / kRecTokens;
  if (chunks > 0) rec_load_chunk(sm, 0, p, d, s0, min(kRecTokens, total), head, khead);
  ptx::cp_async_commit();
  for (int ci = 0; ci < chunks; ++ci) {
    const int buf = ci & 1;
    const int t0 = s0 + ci * kRecTokens;
    const int n = min(kRecTokens, s1 - t0);
    if (ci + 1 < chunks) {
      const int t1 = t0 + kRecTokens;
      rec_load_chunk(sm, buf ^ 1, p, d, t1, min(kRecTokens, s1 - t1), head, khead);
    }
    ptx::cp_async_commit();
    ptx::cp_async_wait<1>();
    __syncthreads();

    for (int t = 0; t < n; ++t) {
      const float decay = sm.decay[buf][t], beta = sm.beta[buf][t];
      const float* kq = &sm.k[buf][t][quarter * kQuarterStride];
      const float* qq = &sm.q[buf][t][quarter * kQuarterStride];
      float kv[4] = {0.f, 0.f, 0.f, 0.f};
      float kr[32];
#pragma unroll
      for (int i = 0; i < 32; i += 4) {
        const float4 kk = *reinterpret_cast<const float4*>(kq + i);
        kr[i] = kk.x; kr[i + 1] = kk.y; kr[i + 2] = kk.z; kr[i + 3] = kk.w;
      }
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        S[i] *= decay;
        kv[i & 3] = fmaf(S[i], kr[i], kv[i & 3]);
      }
      float kvs = (kv[0] + kv[1]) + (kv[2] + kv[3]);
      kvs += __shfl_xor_sync(0xffffffffu, kvs, 1);
      kvs += __shfl_xor_sync(0xffffffffu, kvs, 2);
      const float delta = beta * (sm.v[buf][t][row] - kvs);
      float o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int i = 0; i < 32; i += 4) {
        const float4 qv = *reinterpret_cast<const float4*>(qq + i);
        S[i] = fmaf(delta, kr[i], S[i]);
        S[i + 1] = fmaf(delta, kr[i + 1], S[i + 1]);
        S[i + 2] = fmaf(delta, kr[i + 2], S[i + 2]);
        S[i + 3] = fmaf(delta, kr[i + 3], S[i + 3]);
        o[0] = fmaf(S[i], qv.x, o[0]);
        o[1] = fmaf(S[i + 1], qv.y, o[1]);
        o[2] = fmaf(S[i + 2], qv.z, o[2]);
        o[3] = fmaf(S[i + 3], qv.w, o[3]);
      }
      float os = (o[0] + o[1]) + (o[2] + o[3]);
      os += __shfl_xor_sync(0xffffffffu, os, 1);
      os += __shfl_xor_sync(0xffffffffu, os, 2);
      if (quarter == 0) sm.o[t][row] = os;
    }
    __syncthreads();

    // gated RMSNorm: warp w finishes token w of the chunk
    if (warp < n) {
      const int m = t0 + warp;
      float v[4], ss = 0.0f;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = sm.o[warp][lane + 32 * e];
        ss += v[e] * v[e];
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
      const float r = rsqrtf(ss * (1.0f / 128.0f) + p.eps);
      const __nv_bfloat16* z = p.z + size_t(m) * p.ld_z + head * 128;
      __nv_bfloat16* out = p.out + size_t(m) * (d.num_v_heads * 128) + head * 128;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int dd = lane + 32 * e;
        out[dd] = __float2bfloat16_rn(v[e] * r * p.norm_w[dd] * silu(__bfloat162float(z[dd])));
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 32; i += 4) *reinterpret_cast<float4*>(state + i) = make_float4(S[i], S[i + 1], S[i + 2], S[i + 3]);
}

}  // namespace

void gdn_ab(const __nv_bfloat16* xbf, int m, int hidden, const __nv_bfloat16* w_ab, int rows,
            const float* partials_in, float eps, float* ab_out, cudaStream_t stream) {
  QWEN38_CHECK(hidden % 128 == 0 && rows <= 96, "gdn_ab: shape");
  if (m <= kAbSmallM) {
    gdn_ab_small_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(xbf, m, hidden, w_ab, rows, partials_in, eps, ab_out);
  } else {
    gdn_ab_tiled_kernel<<<(m + 63) / 64, 256, 0, stream>>>(xbf, m, hidden, w_ab, rows, partials_in, eps, ab_out);
  }
}

void gdn_conv(const GdnConvParams& p, const GdnDims& d, const BatchDesc& batch, const GdnStateDesc& st,
              cudaStream_t stream) {
  QWEN38_CHECK(d.conv_dim % 128 == 0, "gdn_conv: conv_dim");
  const dim3 grid(d.conv_dim / 128, batch.num_seqs);
  gdn_conv_kernel<<<grid, 128, 0, stream>>>(p, d, batch, st);
}

void gdn_recurrence(const GdnRecurrenceParams& p, const GdnDims& d, const BatchDesc& batch, const GdnStateDesc& st,
                    cudaStream_t stream) {
  static const bool attr = [] {
    cudaFuncSetAttribute(gdn_recurrence_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(RecSmem)));
    return true;
  }();
  (void)attr;
  const dim3 grid(d.num_v_heads, batch.num_seqs);
  gdn_recurrence_kernel<<<grid, kRecThreads, sizeof(RecSmem), stream>>>(p, d, batch, st);
}

}  // namespace qwen38::kernels
