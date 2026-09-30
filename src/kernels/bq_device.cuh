// In-register dequantization of BQ weights (docs/FORMAT.md section 2).
//
// The bit manipulation is written in plain C++ (nvcc turns each `(x >> s) & m | c` into
// SHF + LOP3) and is compiled for the host as well, so tests/host/test_bq_layout.cpp can run the
// exact device bit logic against the independent spec decoder in src/bq_host.cpp.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define QWEN38_HD __host__ __device__ __forceinline__
#else
#define QWEN38_HD inline
#endif

namespace qwen38::bqdev {

// One lane's share of a 16 (k) x 64 (n) slab: 16 pairs of codes.
struct SlabLane {
  uint32_t lo[4];
  uint32_t hi[4];  // bq5: hi[0]; bq6: hi[0..1]; bq8: hi[0..3] (second nibble plane)
};

constexpr int hi_slab_bytes(int bits) { return bits == 4 ? 0 : bits == 5 ? 128 : bits == 6 ? 256 : 512; }

// bf16 constants
constexpr uint32_t kBf16x2_128 = 0x43004300u;  // 128.0
constexpr uint32_t kBf16x2_16 = 0x41804180u;   // 16.0
constexpr uint32_t bias_bits(int bits) {       // 128 + 2^(bits-1)
  return bits == 4 ? 0x43084308u : bits == 5 ? 0x43104310u : bits == 6 ? 0x43204320u : 0x43084308u;
}

// bf16x2 bit pattern (128 + u_first, 128 + u_second) of pair q (0..15), b = 4, 5, 6.
// For b = 8 this is the low nibble only (see pair_hi_nibble).
template <int BITS>
QWEN38_HD uint32_t pair_magic(const SlabLane& s, int q) {
  const int w = q >> 2, p = q & 3;
  uint32_t v = ((s.lo[w] >> (4 * p)) & 0x000F000Fu) | kBf16x2_128;
  if constexpr (BITS == 5) {
    const uint32_t h = s.hi[0];
    const uint32_t moved = q >= 4 ? (h >> (q - 4)) : (h << (4 - q));
    v |= moved & 0x00100010u;
  } else if constexpr (BITS == 6) {
    const uint32_t h = s.hi[q >> 3];
    const int r2 = 2 * (q & 7);
    const uint32_t moved = r2 >= 4 ? (h >> (r2 - 4)) : (h << (4 - r2));
    v |= moved & 0x00300030u;
  }
  return v;
}

// bq8: (128 + hi_first, 128 + hi_second) from the second nibble plane.
QWEN38_HD uint32_t pair_hi_nibble(const SlabLane& s, int q) {
  const int w = q >> 2, p = q & 3;
  return ((s.hi[w] >> (4 * p)) & 0x000F000Fu) | kBf16x2_128;
}

}  // namespace qwen38::bqdev

#if defined(__CUDACC__)
#include <cuda_bf16.h>

#include "kernels/ptx.cuh"

namespace qwen38::bqdev {

// Signed codes of pair q as exact bf16x2 values.
template <int BITS>
__device__ __forceinline__ __nv_bfloat162 pair_codes(const SlabLane& s, int q) {
  using ptx::as_bf162;
  if constexpr (BITS == 8) {
    const __nv_bfloat162 lo = __hsub2(as_bf162(pair_magic<4>(s, q)), as_bf162(kBf16x2_128));  // [0, 15]
    const __nv_bfloat162 hi = __hsub2(as_bf162(pair_hi_nibble(s, q)), as_bf162(0x43084308u));  // hi - 8
    return __hfma2(hi, as_bf162(kBf16x2_16), lo);  // 16 * (hi - 8) + lo = u - 128, exact
  } else {
    return __hsub2(as_bf162(pair_magic<BITS>(s, q)), as_bf162(bias_bits(BITS)));
  }
}

// Dequantized weights bf16(code * scale) of pair q, as an mma B-fragment register.
template <int BITS>
__device__ __forceinline__ uint32_t pair_weights(const SlabLane& s, int q, __nv_bfloat162 scale) {
  return ptx::as_u32(__hmul2(pair_codes<BITS>(s, q), scale));
}

}  // namespace qwen38::bqdev
#endif
