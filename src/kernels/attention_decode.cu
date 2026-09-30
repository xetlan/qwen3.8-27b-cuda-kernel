// Decode attention: one new token per sequence against its paged KV cache (GQA, head_dim 256).
//
// Split-KV flash-decoding on CUDA cores. With GQA group 6 and int8 KV the arithmetic intensity is
// ~12 FLOP/byte, well below what the SMs sustain, so the kernel is organised for memory
// throughput: grid (splits, kv_head, sequence); each warp streams 4 consecutive tokens at a time,
// lane l owning dims [8l, 8l+8) of all GROUP query heads of its KV head. Partial dot products
// are combined with a transpose-reduce (each shuffle round halves the values per lane), softmax
// is online in the log2 domain, and int8 is widened with a PRMT/FADD magic-number conversion
// (no I2F). Per-split (max, sum, acc) go to scratch; attention_combine merges splits and applies
// the sigmoid output gate.
#include <algorithm>

#include "qwen38/common.h"
#include "kernels/kernels.h"
#include "kernels/ptx.cuh"

namespace qwen38::kernels {
namespace {

constexpr int kDecWarps = 4;
constexpr int kTokPerWarp = 4;
constexpr float kLog2e = 1.4426950408889634f;

// ---- transpose-reduce -------------------------------------------------------------------------
template <int N>
struct TrPlan {
  static constexpr int halvings = (N % 32 == 0) ? 5 : (N % 16 == 0) ? 4 : (N % 8 == 0) ? 3 : (N % 4 == 0) ? 2 : (N % 2 == 0) ? 1 : 0;
  static constexpr int rest = N >> halvings;  // values per lane afterwards
};

template <int N>
__device__ __forceinline__ void tr_halve(float* v, int lane, int off) {
  const bool upper = (lane & off) != 0;
#pragma unroll
  for (int i = 0; i < N / 2; ++i) {
    const float send = upper ? v[i] : v[i + N / 2];
    const float keep = upper ? v[i + N / 2] : v[i];
    v[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
  }
}

// Sum N per-lane values over the warp. On return lane l holds the totals of indices
// [R * (l >> (5 - H)), R * (l >> (5 - H)) + R) in v[0..R), where H = halvings, R = rest.
template <int N>
__device__ __forceinline__ void transpose_reduce(float (&v)[N], int lane) {
  constexpr int H = TrPlan<N>::halvings;
  if constexpr (H >= 1) tr_halve<N>(v, lane, 16);
  if constexpr (H >= 2) tr_halve<(N >> 1)>(v, lane, 8);
  if constexpr (H >= 3) tr_halve<(N >> 2)>(v, lane, 4);
  if constexpr (H >= 4) tr_halve<(N >> 3)>(v, lane, 2);
  if constexpr (H >= 5) tr_halve<(N >> 4)>(v, lane, 1);
  constexpr int R = TrPlan<N>::rest;
  if constexpr (H < 5) {
#pragma unroll
    for (int off = (1 << (4 - H)); off >= 1; off >>= 1) {
#pragma unroll
      for (int i = 0; i < R; ++i) v[i] += __shfl_xor_sync(0xffffffffu, v[i], off);
    }
  }
}

template <int N>
__device__ __forceinline__ float tr_fetch(const float (&v)[N], int idx_const) {
  constexpr int H = TrPlan<N>::halvings, R = TrPlan<N>::rest;
  const int src_lane = (idx_const / R) << (5 - H);
  return __shfl_sync(0xffffffffu, v[idx_const % R], src_lane);
}

// ---- int8 widening ----------------------------------------------------------------------------
// word holds 4 signed bytes; returns them as exact floats (PRMT + FADD each)
__device__ __forceinline__ void i8x4_to_f32(uint32_t word, float* out) {
  const uint32_t u = word ^ 0x80808080u;  // offset binary: x + 128
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint32_t bits = ptx::prmt(u, 0x4B000000u, 0x7540u | k);
    out[k] = __uint_as_float(bits) - 8388736.0f;  // 2^23 + 128
  }
}

template <KvDtype DT>
__device__ __forceinline__ void load_row8(const uint8_t* row, int lane, float* out) {
  if constexpr (DT == KvDtype::Int8) {
    const uint2 w = *reinterpret_cast<const uint2*>(row + lane * 8);
    i8x4_to_f32(w.x, out);
    i8x4_to_f32(w.y, out + 4);
  } else {
    const uint4 w = *reinterpret_cast<const uint4*>(row + lane * 16);
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      out[2 * i] = __uint_as_float(ws[i] << 16);
      out[2 * i + 1] = __uint_as_float(ws[i] & 0xffff0000u);
    }
  }
}

template <int GROUP, KvDtype DT>
__global__ void __launch_bounds__(kDecWarps * 32) attn_decode_kernel(AttentionParams p, BatchDesc batch,
                                                                     KvCacheDesc kv, int splits) {
  __shared__ float red_acc[kDecWarps][GROUP][kHeadDim];
  __shared__ float red_ml[kDecWarps][GROUP][2];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int split = blockIdx.x, kvh = blockIdx.y, b = blockIdx.z;
  const int L = batch.seq_pos0[b] + 1;  // the new token is already in the cache
  const int slot = batch.seq_slot[b];
  const int m = batch.seq_start[b];
  const int per = ((L + splits - 1) / splits + 63) / 64 * 64;
  const int t_begin = split * per;
  const int t_end = min(L, t_begin + per);

  // q for the GROUP heads of this KV head, pre-scaled for exp2 softmax
  float q[GROUP][8];
  const float qscale = kLog2e / 16.0f;  // 1/sqrt(256) * log2(e)
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    const __half* src = p.q + (size_t(m) * p.num_q_heads + kvh * GROUP + g) * kHeadDim + lane * 8;
    const uint4 raw = *reinterpret_cast<const uint4*>(src);
    const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      __half2 h2;
      __builtin_memcpy(&h2, &w[i], 4);
      const float2 f = __half22float2(h2);
      q[g][2 * i] = f.x * qscale;
      q[g][2 * i + 1] = f.y * qscale;
    }
  }

  float mx[GROUP], l[GROUP], acc[GROUP][8];
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    mx[g] = -INFINITY;
    l[g] = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[g][e] = 0.0f;
  }

  const uint8_t* layer = kv.pool + size_t(p.attn_layer) * kv.layer_stride;
  const int* bt = kv.block_table + size_t(slot) * kv.max_pages_per_seq;
  for (int base = t_begin + warp * kTokPerWarp; base < t_end; base += kDecWarps * kTokPerWarp) {
    const int nvalid = min(kTokPerWarp, t_end - base);
    float part[GROUP * kTokPerWarp];
    float kscale[kTokPerWarp], vscale[kTokPerWarp];
    const uint8_t* pages[kTokPerWarp];
    int in_page[kTokPerWarp];
#pragma unroll
    for (int i = 0; i < kTokPerWarp; ++i) {
      const int t = min(base + i, t_end - 1);  // clamp: invalid slots re-read a valid row, masked below
      pages[i] = layer + size_t(bt[t / kPageTokens]) * kv.page_bytes;
      in_page[i] = t % kPageTokens;
      if constexpr (DT == KvDtype::Int8) {
        kscale[i] = __half2float(reinterpret_cast<const __half*>(pages[i] + kv.ks_offset(kvh, in_page[i]))[lane >> 3]);
        vscale[i] = __half2float(reinterpret_cast<const __half*>(pages[i] + kv.vs_offset(kvh, in_page[i]))[lane >> 3]);
      } else {
        kscale[i] = 1.0f;
        vscale[i] = 1.0f;
      }
      float kf[8];
      load_row8<DT>(pages[i] + kv.k_offset(kvh, in_page[i]), lane, kf);
#pragma unroll
      for (int g = 0; g < GROUP; ++g) {
        float s = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) s = fmaf(q[g][e], kf[e], s);
        part[g * kTokPerWarp + i] = s * kscale[i];
      }
    }
    transpose_reduce<GROUP * kTokPerWarp>(part, lane);
    float s[GROUP][kTokPerWarp];
#pragma unroll
    for (int g = 0; g < GROUP; ++g)
#pragma unroll
      for (int i = 0; i < kTokPerWarp; ++i) {
        const float val = tr_fetch<GROUP * kTokPerWarp>(part, g * kTokPerWarp + i);
        s[g][i] = i < nvalid ? val : -INFINITY;
      }
    float pv[GROUP][kTokPerWarp];
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      float mnew = mx[g];
#pragma unroll
      for (int i = 0; i < kTokPerWarp; ++i) mnew = fmaxf(mnew, s[g][i]);
      const float corr = exp2f(mx[g] - mnew);  // mx = -inf on the first block -> 0
      float sum = 0.0f;
#pragma unroll
      for (int i = 0; i < kTokPerWarp; ++i) {
        pv[g][i] = exp2f(s[g][i] - mnew);
        sum += pv[g][i];
      }
      l[g] = l[g] * corr + sum;
      mx[g] = mnew;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[g][e] *= corr;
    }
#pragma unroll
    for (int i = 0; i < kTokPerWarp; ++i) {
      float vf[8];
      load_row8<DT>(pages[i] + kv.v_offset(kvh, in_page[i]), lane, vf);
#pragma unroll
      for (int g = 0; g < GROUP; ++g) {
        const float w = pv[g][i] * vscale[i];
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[g][e] = fmaf(w, vf[e], acc[g][e]);
      }
    }
  }

  // merge the warps of this CTA
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
#pragma unroll
    for (int e = 0; e < 8; ++e) red_acc[warp][g][lane * 8 + e] = acc[g][e];
    if (lane == 0) {
      red_ml[warp][g][0] = mx[g];
      red_ml[warp][g][1] = l[g];
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < GROUP * kHeadDim; idx += kDecWarps * 32) {
    const int g = idx / kHeadDim, dd = idx % kHeadDim;
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < kDecWarps; ++w) M = fmaxf(M, red_ml[w][g][0]);
    float a = 0.0f, lsum = 0.0f;
#pragma unroll
    for (int w = 0; w < kDecWarps; ++w) {
      const float f = M == -INFINITY ? 0.0f : exp2f(red_ml[w][g][0] - M);
      a += f * red_acc[w][g][dd];
      lsum += f * red_ml[w][g][1];
    }
    float* dst = p.split_scratch + ((size_t(b) * p.num_q_heads + kvh * GROUP + g) * p.max_splits + split) * (kHeadDim + 2);
    dst[dd] = a;
    if (dd == 0) {
      dst[kHeadDim] = M;
      dst[kHeadDim + 1] = lsum;
    }
  }
}

// grid (num_q_heads, B), 256 threads: merge splits, normalise, apply sigmoid(gate)
__global__ void __launch_bounds__(256) attn_combine_kernel(AttentionParams p, BatchDesc batch, int splits) {
  const int h = blockIdx.x, b = blockIdx.y, dd = threadIdx.x;
  const float* src = p.split_scratch + (size_t(b) * p.num_q_heads + h) * p.max_splits * (kHeadDim + 2);
  float M = -INFINITY;
  for (int s = 0; s < splits; ++s) M = fmaxf(M, src[s * (kHeadDim + 2) + kHeadDim]);
  float a = 0.0f, lsum = 0.0f;
  for (int s = 0; s < splits; ++s) {
    const float ms = src[s * (kHeadDim + 2) + kHeadDim];
    const float f = ms == -INFINITY ? 0.0f : exp2f(ms - M);
    a += f * src[s * (kHeadDim + 2) + dd];
    lsum += f * src[s * (kHeadDim + 2) + kHeadDim + 1];
  }
  const int m = batch.seq_start[b];
  const float gate = __bfloat162float(p.gate[size_t(m) * p.ld_gate + h * kHeadDim + dd]);
  const float y = (a / lsum) * (1.0f / (1.0f + __expf(-gate)));
  p.out[(size_t(m) * p.num_q_heads + h) * kHeadDim + dd] = __float2bfloat16_rn(y);
}

template <int GROUP>
void launch_group(const AttentionParams& p, const BatchDesc& batch, const KvCacheDesc& kv, int splits,
                  cudaStream_t stream) {
  const dim3 grid(splits, p.num_kv_heads, batch.num_seqs);
  if (kv.dtype == KvDtype::Int8) {
    attn_decode_kernel<GROUP, KvDtype::Int8><<<grid, kDecWarps * 32, 0, stream>>>(p, batch, kv, splits);
  } else {
    attn_decode_kernel<GROUP, KvDtype::BF16><<<grid, kDecWarps * 32, 0, stream>>>(p, batch, kv, splits);
  }
}

}  // namespace

int attention_decode_splits(int batch_seqs, int kv_heads, int num_sms, int max_splits) {
  // enough CTAs for ~2 waves; each split handles a whole number of 64-token pages
  const int ctas_per_split = std::max(1, batch_seqs * kv_heads);
  int splits = (2 * num_sms + ctas_per_split - 1) / ctas_per_split;
  return std::max(1, std::min(splits, max_splits));
}

void attention_decode(const AttentionParams& p, const BatchDesc& batch, const KvCacheDesc& kv, cudaStream_t stream) {
  QWEN38_CHECK(p.num_q_heads % p.num_kv_heads == 0, "attention: GQA heads");
  QWEN38_CHECK(p.split_scratch && p.max_splits > 0, "attention_decode: split scratch");
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int splits = attention_decode_splits(batch.num_seqs, p.num_kv_heads, sms, p.max_splits);
  switch (p.num_q_heads / p.num_kv_heads) {
    case 1: launch_group<1>(p, batch, kv, splits, stream); break;
    case 2: launch_group<2>(p, batch, kv, splits, stream); break;
    case 3: launch_group<3>(p, batch, kv, splits, stream); break;
    case 4: launch_group<4>(p, batch, kv, splits, stream); break;
    case 6: launch_group<6>(p, batch, kv, splits, stream); break;
    case 8: launch_group<8>(p, batch, kv, splits, stream); break;
    default: fail("attention_decode: unsupported GQA group size");
  }
  attn_combine_kernel<<<dim3(p.num_q_heads, batch.num_seqs), kHeadDim, 0, stream>>>(p, batch, splits);
}

}  // namespace qwen38::kernels
