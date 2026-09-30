// Token embedding gather + residual stream initialisation.
//
// The bf16 embedding table (248320 x 5120, 2.5 GB) normally stays in pinned host memory and is
// read zero-copy over PCIe: a decode step needs one 10 KB row per sequence, so keeping it off
// the GPU frees 2.5 GB of VRAM for KV cache at negligible cost.
#include "qwen38/common.h"
#include "kernels/kernels.h"
#include "kernels/ptx.cuh"

namespace qwen38::kernels {
namespace {

__global__ void __launch_bounds__(128) embed_kernel(const __nv_bfloat16* __restrict__ table,
                                                    const int32_t* __restrict__ tokens, int hidden,
                                                    float* __restrict__ residual, __nv_bfloat16* __restrict__ xbf,
                                                    float* __restrict__ partials) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int token = tokens[row];
  const __nv_bfloat16* src = table + size_t(token) * hidden;
  const int blocks = hidden / 128;
  for (int b = warp; b < blocks; b += 4) {
    const int col = b * 128 + lane * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(src + col);
    const float2 lo = __bfloat1622float2(ptx::as_bf162(raw.x));
    const float2 hi = __bfloat1622float2(ptx::as_bf162(raw.y));
    *reinterpret_cast<float4*>(residual + size_t(row) * hidden + col) = make_float4(lo.x, lo.y, hi.x, hi.y);
    *reinterpret_cast<uint2*>(xbf + size_t(row) * hidden + col) = raw;
    float ss = lo.x * lo.x + lo.y * lo.y + hi.x * hi.x + hi.y * hi.y;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    if (lane == 0) partials[size_t(row) * blocks + b] = ss;
  }
}

}  // namespace

void embed(const __nv_bfloat16* table, const int32_t* tokens, int m, int hidden, float* residual,
           __nv_bfloat16* xbf, float* partials, cudaStream_t stream) {
  QWEN38_CHECK(hidden % 128 == 0, "embed: hidden must be a multiple of 128");
  if (m == 0) return;
  embed_kernel<<<m, 128, 0, stream>>>(table, tokens, hidden, residual, xbf, partials);
}

}  // namespace qwen38::kernels
