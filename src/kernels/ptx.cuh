// Inline-PTX primitives for sm_80+ (Ampere), used by the hand-written kernels.
//
//   cp.async        16-byte global->shared copies with zero-fill predicate and L2 cache hints
//   ldmatrix        8x8 b16 matrix loads from shared memory into mma fragments
//   mma.sync        m16n8k16 bf16/f16 tensor-core MMA with fp32 accumulate
//
// Fragment layouts follow the PTX ISA ("Matrix Fragments for mma.m16n8k16"); with
// g = lane >> 2 and t = lane & 3:
//   A (16x16, row):  a0a1 (g, 2t..2t+1)  a2a3 (g+8, 2t..)  a4a5 (g, 2t+8..)  a6a7 (g+8, 2t+8..)
//   B (16x8, col):   b0b1 (k 2t..2t+1, n g)  b2b3 (k 2t+8..2t+9, n g)
//   C (16x8, f32):   c0c1 (g, 2t..2t+1)  c2c3 (g+8, 2t..2t+1)
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace qwen38::ptx {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---- cp.async ---------------------------------------------------------------------------------

// 16 bytes global -> shared, cached in L2 only (.cg). When `pred` is false nothing is read from
// global memory and the destination is zero-filled.
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, bool pred = true) {
  const int src_bytes = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(smem)), "l"(gmem),
               "r"(src_bytes));
}

// Same, with an L2 cache policy (e.g. evict_first for weights that are streamed once per step).
__device__ __forceinline__ void cp_async_16_hint(void* smem, const void* gmem, uint64_t policy) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(smem_u32(smem)),
               "l"(gmem), "l"(policy));
}

__device__ __forceinline__ uint64_t l2_policy_evict_first() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
  return policy;
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ---- ldmatrix ---------------------------------------------------------------------------------

// Four 8x8 b16 matrices. Lane l supplies the row address for row (l % 8) of matrix (l / 8);
// register i receives matrix i in the mma fragment distribution (thread g,t holds row g,
// elements 2t, 2t+1).
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem_row) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(smem_row)));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* smem_row) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(smem_u32(smem_row)));
}

// Transposed variant: thread g,t receives column g, rows 2t, 2t+1 of each 8x8 matrix.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem_row) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(smem_row)));
}

// ---- mma.sync ---------------------------------------------------------------------------------

__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---- bit tricks -------------------------------------------------------------------------------

// d = f(a, b, c) for the 8-bit truth table LUT (a = 0xF0, b = 0xCC, c = 0xAA).
template <uint32_t LUT>
__device__ __forceinline__ uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c), "n"(LUT));
  return d;
}

// (a & b) | c
__device__ __forceinline__ uint32_t and_or(uint32_t a, uint32_t b, uint32_t c) {
  return lop3<(0xF0 & 0xCC) | 0xAA>(a, b, c);
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t selector) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(selector));
  return d;
}

// ---- packed 16-bit helpers --------------------------------------------------------------------

__device__ __forceinline__ __nv_bfloat162 as_bf162(uint32_t u) {
  __nv_bfloat162 v;
  static_assert(sizeof(v) == 4);
  __builtin_memcpy(&v, &u, 4);
  return v;
}

__device__ __forceinline__ uint32_t as_u32(__nv_bfloat162 v) {
  uint32_t u;
  __builtin_memcpy(&u, &v, 4);
  return u;
}

__device__ __forceinline__ uint32_t as_u32(__half2 v) {
  uint32_t u;
  __builtin_memcpy(&u, &v, 4);
  return u;
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  return as_u32(__floats2bfloat162_rn(lo, hi));
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) { return as_u32(__floats2half2_rn(lo, hi)); }

}  // namespace qwen38::ptx
