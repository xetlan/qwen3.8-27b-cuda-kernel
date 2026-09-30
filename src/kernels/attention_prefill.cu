// Prefill attention (causal, GQA, head_dim 256) over the paged KV cache, FlashAttention-2 schedule.
//
// CuTe carries the tensor-core part: a TiledMMA of SM80 m16n8k16 fp16 atoms (4 warps along M),
// swizzled shared-memory layouts (Swizzle<3,3,3> atoms: conflict-free ldmatrix and 16-byte
// stores), ldmatrix copy atoms (non-transposed for Q/K, transposed for V), and the accumulator
// layout algebra that turns S into P register fragments without a trip through shared memory.
//
// K/V tiles come from the paged cache (int8 + per-64-dim scales, or bf16): cp.async stages the raw
// tile, one conversion pass dequantizes it into the fp16 swizzled tile, and loads of the next
// tile overlap the current tile's math (K(n-1) streams during P*V(n), V(n) during Q*K(n)).
// Queries are this step's new tokens; keys are every cached position up to the query position.
#include <cute/tensor.hpp>  // must come first: the CuTe headers are not individually self-contained
#include <cutlass/numeric_types.h>

#include "qwen38/common.h"
#include "kernels/kernels.h"
#include "kernels/ptx.cuh"

namespace qwen38::kernels {
namespace {

using namespace cute;

constexpr int kBlockM = 64;
constexpr int kBlockN = 32;
constexpr int kNWarps = 4;
constexpr int kNThreads = kNWarps * 32;
constexpr float kLog2e = 1.4426950408889634f;

using Element = cutlass::half_t;
using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>, Layout<Shape<Int<kNWarps>, _1, _1>>,
                          Tile<Int<16 * kNWarps>, _16, _16>>;
using SmemLayoutAtom = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
using SmemLayoutVt =
    decltype(composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
using SmemLayoutVtNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVt{}));
using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using SmemCopyAtomT = Copy_Atom<SM75_U16x8_LDSM_T, Element>;

struct PrefillSmem {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutKV>> k;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutKV>> v;
  alignas(16) uint8_t stage_k[kBlockN * kHeadDim * 2];  // raw tile (bf16 is the larger case)
  alignas(16) uint8_t stage_v[kBlockN * kHeadDim * 2];
  alignas(16) __half scale_k[kBlockN * 4];
  alignas(16) __half scale_v[kBlockN * 4];
};

// (MMA=4, MMA_M, MMA_N) -> (nrow = (2, MMA_M), ncol = (2, MMA_N))
template <class Layout>
__device__ __forceinline__ auto acc_rowcol(Layout acc_layout) {
  auto l = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
}

// (MMA=4, MMA_M, MMA_N) -> ((4, 2), MMA_M, MMA_N / 2): accumulator of S reused as A operand of P*V
template <class Layout>
__device__ __forceinline__ auto acc_as_a(Layout acc_layout) {
  auto l = logical_divide(acc_layout, Shape<Underscore, Underscore, _2>{});
  return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
}

template <class TA, class TB, class TSA, class TSB, class TC, class CA, class CB, class TCA, class TCB>
__device__ __forceinline__ void gemm_ss(TC& acc, TA& tCrA, TB& tCrB, const TSA& tCsA, const TSB& tCsB,
                                        const TiledMma& mma, const CA& copy_a, const CB& copy_b,
                                        const TCA& thr_copy_a, const TCB& thr_copy_b) {
  Tensor tCrA_view = thr_copy_a.retile_D(tCrA);
  Tensor tCrB_view = thr_copy_b.retile_D(tCrB);
  cute::copy(copy_a, tCsA(_, _, _0{}), tCrA_view(_, _, _0{}));
  cute::copy(copy_b, tCsB(_, _, _0{}), tCrB_view(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(copy_a, tCsA(_, _, i + 1), tCrA_view(_, _, i + 1));
      cute::copy(copy_b, tCsB(_, _, i + 1), tCrB_view(_, _, i + 1));
    }
    cute::gemm(mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

template <class TA, class TB, class TSB, class TC, class CB, class TCB>
__device__ __forceinline__ void gemm_rs(TC& acc, TA& tCrA, TB& tCrB, const TSB& tCsB, const TiledMma& mma,
                                        const CB& copy_b, const TCB& thr_copy_b) {
  Tensor tCrB_view = thr_copy_b.retile_D(tCrB);
  cute::copy(copy_b, tCsB(_, _, _0{}), tCrB_view(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) cute::copy(copy_b, tCsB(_, _, i + 1), tCrB_view(_, _, i + 1));
    cute::gemm(mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

// cp.async the raw K or V tile (kBlockN tokens of one head, all inside one page) + its scales.
template <KvDtype DT>
__device__ __forceinline__ void stage_tile(uint8_t* stage, __half* scales, const uint8_t* page, size_t row_off,
                                           size_t scale_off) {
  constexpr int kRowBytes = kHeadDim * (DT == KvDtype::Int8 ? 1 : 2);
  constexpr int kChunks = kBlockN * kRowBytes / 16;
#pragma unroll
  for (int c = threadIdx.x; c < kChunks; c += kNThreads) ptx::cp_async_16(stage + c * 16, page + row_off + c * 16);
  if constexpr (DT == KvDtype::Int8) {
    if (threadIdx.x < kBlockN * 8 / 16)
      ptx::cp_async_16(reinterpret_cast<uint8_t*>(scales) + threadIdx.x * 16, page + scale_off + threadIdx.x * 16);
  }
}

// raw tile -> fp16 swizzled tile; key rows at or beyond `valid` are zeroed (stale cache slots)
template <KvDtype DT, class SmemTensor>
__device__ __forceinline__ void convert_tile(const uint8_t* stage, const __half* scales, SmemTensor& dst, int valid) {
  for (int c = threadIdx.x; c < kBlockN * kHeadDim / 8; c += kNThreads) {
    const int r = c / (kHeadDim / 8), c0 = (c % (kHeadDim / 8)) * 8;
    uint4 out = make_uint4(0u, 0u, 0u, 0u);
    if (r < valid) {
      if constexpr (DT == KvDtype::Int8) {
        const uint2 raw = *reinterpret_cast<const uint2*>(stage + r * kHeadDim + c0);
        const __half2 s2 = __half2half2(scales[r * 4 + c0 / 64]);
        const uint32_t w[2] = {raw.x ^ 0x80808080u, raw.y ^ 0x80808080u};  // offset binary
        uint32_t o[4];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          // bytes -> (1024 + u) half pairs, minus 1152 -> exact signed values
          const uint32_t lo = ptx::prmt(w[i], 0x64646464u, 0x5140u);
          const uint32_t hi = ptx::prmt(w[i], 0x64646464u, 0x7362u);
          __half2 a, b;
          __builtin_memcpy(&a, &lo, 4);
          __builtin_memcpy(&b, &hi, 4);
          const __half2 bias = __float2half2_rn(1152.0f);
          a = __hmul2(__hsub2(a, bias), s2);
          b = __hmul2(__hsub2(b, bias), s2);
          __builtin_memcpy(&o[2 * i], &a, 4);
          __builtin_memcpy(&o[2 * i + 1], &b, 4);
        }
        out = make_uint4(o[0], o[1], o[2], o[3]);
      } else {
        const uint4 raw = *reinterpret_cast<const uint4*>(stage + (r * kHeadDim + c0) * 2);
        const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
        uint32_t o[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float x0 = fminf(fmaxf(__uint_as_float(w[i] << 16), -65504.0f), 65504.0f);
          const float x1 = fminf(fmaxf(__uint_as_float(w[i] & 0xffff0000u), -65504.0f), 65504.0f);
          o[i] = ptx::pack_f16x2(x0, x1);
        }
        out = make_uint4(o[0], o[1], o[2], o[3]);
      }
    }
    *reinterpret_cast<uint4*>(&dst(r, c0)) = out;
  }
}

template <KvDtype DT>
__global__ void __launch_bounds__(kNThreads, 1) attn_prefill_kernel(AttentionParams p, BatchDesc batch,
                                                                    KvCacheDesc kv) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  PrefillSmem& sm = *reinterpret_cast<PrefillSmem*>(smem_raw);
  const int tid = threadIdx.x;
  const int m_block = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int s0 = batch.seq_start[b];
  const int n_new = batch.seq_start[b + 1] - s0;
  const int row0 = m_block * kBlockM;
  if (row0 >= n_new) return;
  const int pos0 = batch.seq_pos0[b];
  const int slot = batch.seq_slot[b];
  const int kvh = h / (p.num_q_heads / p.num_kv_heads);
  const int L = pos0 + n_new;  // keys [0, L)
  const int q_first = pos0 + row0;
  const int q_last = pos0 + min(row0 + kBlockM, n_new) - 1;
  const int n_blocks = q_last / kBlockN + 1;

  const uint8_t* layer = kv.pool + size_t(p.attn_layer) * kv.layer_stride;
  const int* bt = kv.block_table + size_t(slot) * kv.max_pages_per_seq;
  auto page_of = [&](int n0) { return layer + size_t(bt[n0 / kPageTokens]) * kv.page_bytes; };

  Tensor sQ = make_tensor(make_smem_ptr(sm.q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(sm.k.data()), SmemLayoutKV{});
  Tensor sV = make_tensor(make_smem_ptr(sm.v.data()), SmemLayoutKV{});
  Tensor sVt = make_tensor(make_smem_ptr(sm.v.data()), SmemLayoutVt{});
  Tensor sVtNoSwizzle = make_tensor(make_smem_ptr(sm.v.data()), SmemLayoutVtNoSwizzle{});

  // Q tile (rows beyond the sequence zero-filled)
  for (int c = tid; c < kBlockM * kHeadDim / 8; c += kNThreads) {
    const int r = c / (kHeadDim / 8), c0 = (c % (kHeadDim / 8)) * 8;
    const bool ok = row0 + r < n_new;
    const __half* src = p.q + (size_t(s0 + (ok ? row0 + r : 0)) * p.num_q_heads + h) * kHeadDim + c0;
    ptx::cp_async_16(&sQ(r, c0), src, ok);
  }
  {
    const int n0 = (n_blocks - 1) * kBlockN;
    stage_tile<DT>(sm.stage_k, sm.scale_k, page_of(n0), kv.k_offset(kvh, n0 % kPageTokens),
                   kv.ks_offset(kvh, n0 % kPageTokens));
  }
  ptx::cp_async_commit();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
  Tensor tSrK = thr_mma.partition_fragment_B(sK);
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
  Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  clear(acc_o);

  auto copy_q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto thr_copy_q = copy_q.get_thread_slice(tid);
  Tensor tSsQ = thr_copy_q.partition_S(sQ);
  auto copy_k = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto thr_copy_k = copy_k.get_thread_slice(tid);
  Tensor tSsK = thr_copy_k.partition_S(sK);
  auto copy_v = make_tiled_copy_B(SmemCopyAtomT{}, tiled_mma);
  auto thr_copy_v = copy_v.get_thread_slice(tid);
  Tensor tOsVt = thr_copy_v.partition_S(sVt);

  Tensor tScS = thr_mma.partition_C(make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{}));
  Tensor tOcO = thr_mma.partition_C(make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{}));

  constexpr float kScale = kLog2e / 16.0f;  // 1/sqrt(256) in the log2 domain
  Tensor acc_o_rc = make_tensor(acc_o.data(), acc_rowcol(acc_o.layout()));
  constexpr int kRows = decltype(size<0>(acc_o_rc))::value;
  float row_max[kRows], row_sum[kRows];
#pragma unroll
  for (int i = 0; i < kRows; ++i) {
    row_max[i] = -INFINITY;
    row_sum[i] = 0.0f;
  }

  for (int nb = n_blocks - 1; nb >= 0; --nb) {
    const int n0 = nb * kBlockN;
    ptx::cp_async_wait<0>();
    __syncthreads();
    convert_tile<DT>(sm.stage_k, sm.scale_k, sK, L - n0);
    __syncthreads();
    stage_tile<DT>(sm.stage_v, sm.scale_v, page_of(n0), kv.v_offset(kvh, n0 % kPageTokens),
                   kv.vs_offset(kvh, n0 % kPageTokens));
    ptx::cp_async_commit();

    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);
    gemm_ss(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, copy_q, copy_k, thr_copy_q, thr_copy_k);

    if (n0 + kBlockN - 1 > q_first) {  // block touches the diagonal: causal mask
#pragma unroll
      for (int i = 0; i < size(acc_s); ++i) {
        const int key = n0 + get<1>(tScS(i));
        const int qpos = q_first + get<0>(tScS(i));
        if (key > qpos) acc_s(i) = -INFINITY;
      }
    }

    // online softmax (rows spread over quads of lanes)
    Tensor s_rc = make_tensor(acc_s.data(), acc_rowcol(acc_s.layout()));
#pragma unroll
    for (int mi = 0; mi < kRows; ++mi) {
      float mx = row_max[mi];
#pragma unroll
      for (int ni = 0; ni < size<1>(s_rc); ++ni) mx = fmaxf(mx, s_rc(mi, ni));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float mx_scaled = mx == -INFINITY ? 0.0f : mx * kScale;
      const float corr = row_max[mi] == -INFINITY ? 0.0f : exp2f(row_max[mi] * kScale - mx_scaled);
      row_max[mi] = mx;
      float sum = 0.0f;
#pragma unroll
      for (int ni = 0; ni < size<1>(s_rc); ++ni) {
        const float e = exp2f(s_rc(mi, ni) * kScale - mx_scaled);
        s_rc(mi, ni) = e;
        sum += e;
      }
      row_sum[mi] = row_sum[mi] * corr + sum;
#pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rc); ++ni) acc_o_rc(mi, ni) *= corr;
    }

    ptx::cp_async_wait<0>();
    __syncthreads();
    convert_tile<DT>(sm.stage_v, sm.scale_v, sV, L - n0);
    __syncthreads();
    if (nb > 0) {
      const int nn = n0 - kBlockN;
      stage_tile<DT>(sm.stage_k, sm.scale_k, page_of(nn), kv.k_offset(kvh, nn % kPageTokens),
                     kv.ks_offset(kvh, nn % kPageTokens));
    }
    ptx::cp_async_commit();

    Tensor rP = make_tensor<Element>(acc_s.layout());
#pragma unroll
    for (int i = 0; i < size(acc_s); ++i) rP(i) = Element(acc_s(i));
    Tensor tOrP = make_tensor(rP.data(), acc_as_a(rP.layout()));
    gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, copy_v, thr_copy_v);
  }

  // epilogue: normalise, gate, store bf16
#pragma unroll
  for (int mi = 0; mi < kRows; ++mi) {
    float sum = row_sum[mi];
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
#pragma unroll
    for (int ni = 0; ni < size<1>(acc_o_rc); ++ni) acc_o_rc(mi, ni) *= inv;
  }
#pragma unroll
  for (int i = 0; i < size(acc_o); ++i) {
    const int r = get<0>(tOcO(i)), c = get<1>(tOcO(i));
    if (row0 + r < n_new) {
      const size_t m = size_t(s0 + row0 + r);
      const float gate = __bfloat162float(p.gate[m * p.ld_gate + h * kHeadDim + c]);
      const float y = acc_o(i) * (1.0f / (1.0f + __expf(-gate)));
      p.out[(m * p.num_q_heads + h) * kHeadDim + c] = __float2bfloat16_rn(y);
    }
  }
}

}  // namespace

void attention_prefill(const AttentionParams& p, const BatchDesc& batch, const KvCacheDesc& kv, cudaStream_t stream) {
  QWEN38_CHECK(p.num_q_heads % p.num_kv_heads == 0, "attention: GQA heads");
  constexpr int kSmem = int(sizeof(PrefillSmem));
  static_assert(sizeof(PrefillSmem) <= 99 * 1024, "prefill attention shared memory");
  const dim3 grid((batch.max_tokens_per_seq + kBlockM - 1) / kBlockM, p.num_q_heads, batch.num_seqs);
  if (kv.dtype == KvDtype::Int8) {
    static const bool attr =
        cudaFuncSetAttribute(attn_prefill_kernel<KvDtype::Int8>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) ==
        cudaSuccess;
    (void)attr;
    attn_prefill_kernel<KvDtype::Int8><<<grid, kNThreads, kSmem, stream>>>(p, batch, kv);
  } else {
    static const bool attr =
        cudaFuncSetAttribute(attn_prefill_kernel<KvDtype::BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) ==
        cudaSuccess;
    (void)attr;
    attn_prefill_kernel<KvDtype::BF16><<<grid, kNThreads, kSmem, stream>>>(p, batch, kv);
  }
}

}  // namespace qwen38::kernels
