// Fused q/k offset-RMSNorm + partial RoPE + KV-cache append (int8 per 64-dim group, or bf16).
//
// One CTA per token, one warp per head: q heads, then k heads, then v heads. Lane l owns dims
// {l, l+32, ..., l+224}, so both halves of every rotated pair (i, i+32), i < 32, sit in lane i.
#include "qwen38/common.h"
#include "kernels/kernels.h"
#include "kernels/ptx.cuh"

namespace qwen38::kernels {
namespace {

__global__ void __launch_bounds__(1024) qk_norm_rope_kernel(QkNormRopeParams p, BatchDesc batch, KvCacheDesc kv) {
  __shared__ float cos_s[32], sin_s[32];
  const int m = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int pos = batch.tok_pos[m];
  if (threadIdx.x < 32) {
    const float inv_freq = 1.0f / powf(p.rope_theta, float(2 * threadIdx.x) / 64.0f);
    float s, c;
    sincosf(float(pos) * inv_freq, &s, &c);
    cos_s[threadIdx.x] = c;
    sin_s[threadIdx.x] = s;
  }
  __syncthreads();

  const int nq = p.num_q_heads, nkv = p.num_kv_heads;
  if (warp >= nq + 2 * nkv) return;
  const bool is_q = warp < nq;
  const bool is_k = !is_q && warp < nq + nkv;
  const int head = is_q ? warp : is_k ? warp - nq : warp - nq - nkv;
  const __nv_bfloat16* src = p.proj + size_t(m) * p.ld_proj +
                             (is_q ? size_t(head) * kHeadDim
                                   : size_t(2 * nq) * kHeadDim + size_t(is_k ? head : nkv + head) * kHeadDim);
  float y[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) y[e] = __bfloat162float(src[lane + 32 * e]);

  if (is_q || is_k) {
    float ss = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) ss += y[e] * y[e];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float r = rsqrtf(ss * (1.0f / kHeadDim) + p.eps);
    const float* w = is_q ? p.q_norm : p.k_norm;
#pragma unroll
    for (int e = 0; e < 8; ++e) y[e] = y[e] * r * w[lane + 32 * e];
    const float c = cos_s[lane], s = sin_s[lane];
    const float x1 = y[0], x2 = y[1];
    y[0] = x1 * c - x2 * s;
    y[1] = x2 * c + x1 * s;
  }

  if (is_q) {
    __half* dst = p.q_out + (size_t(m) * nq + head) * kHeadDim;
#pragma unroll
    for (int e = 0; e < 8; ++e) dst[lane + 32 * e] = __float2half_rn(y[e]);
    return;
  }

  const int b = batch.tok_seq[m];
  const int slot = batch.seq_slot[b];
  const int page = kv.block_table[size_t(slot) * kv.max_pages_per_seq + pos / kPageTokens];
  const int tok = pos % kPageTokens;
  uint8_t* base = kv.pool + size_t(p.attn_layer) * kv.layer_stride + size_t(page) * kv.page_bytes;
  const size_t off = is_k ? kv.k_offset(head, tok) : kv.v_offset(head, tok);
  if (kv.dtype == KvDtype::BF16) {
    auto* dst = reinterpret_cast<__nv_bfloat16*>(base + off);
#pragma unroll
    for (int e = 0; e < 8; ++e) dst[lane + 32 * e] = __float2bfloat16_rn(y[e]);
    return;
  }
  auto* dst = reinterpret_cast<int8_t*>(base + off);
  auto* scales = reinterpret_cast<__half*>(base + (is_k ? kv.ks_offset(head, tok) : kv.vs_offset(head, tok)));
#pragma unroll
  for (int grp = 0; grp < 4; ++grp) {
    float amax = fmaxf(fabsf(y[2 * grp]), fabsf(y[2 * grp + 1]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const __half scale_h = __float2half_rn(amax / 127.0f);
    const float scale = __half2float(scale_h);
#pragma unroll
    for (int e = 2 * grp; e < 2 * grp + 2; ++e) {
      float q = 0.0f;
      if (scale > 0.0f) q = fminf(fmaxf(rintf(y[e] / scale), -127.0f), 127.0f);
      dst[lane + 32 * e] = int8_t(q);
    }
    if (lane == 0) scales[grp] = scale_h;
  }
}

}  // namespace

void qk_norm_rope_append(const QkNormRopeParams& p, const BatchDesc& batch, const KvCacheDesc& kv,
                         cudaStream_t stream) {
  const int warps = p.num_q_heads + 2 * p.num_kv_heads;
  QWEN38_CHECK(warps <= 32, "qk_norm_rope: too many heads for one CTA");
  if (batch.num_tokens == 0) return;
  qk_norm_rope_kernel<<<batch.num_tokens, warps * 32, 0, stream>>>(p, batch, kv);
}

}  // namespace qwen38::kernels
