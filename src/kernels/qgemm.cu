// Quantized GEMM: C[M][N] = A[M][K] (bf16) * W^T (BQ 4/5/6/8-bit, group 64), fp32 accumulate.
//
// Structure (per CTA):
//   * multi-stage cp.async pipeline: A tile (TM x TK, XOR-swizzled 16-byte chunks), the lanes'
//     pre-permuted weight slabs (lo/hi planes) and bf16 scales, STAGES deep
//   * warp tile = (MT x m16) x n64 over one k16 slab per step; warps may also split the
//     stage's k-slabs (WK > 1) and are reduced deterministically in shared memory
//   * A fragments via ldmatrix.x4, weights dequantized in registers (bq_device.cuh), scaled in
//     bf16 and fed to mma.sync.m16n8k16 (bf16 x bf16 -> fp32)
//   * work decomposition: data-parallel tiles with grouped rasterization for large M, or
//     stream-K over (tile, k-iteration) space for small M so every SM streams weights; partial
//     tiles are combined by the last-arriving CTA in fixed segment order (deterministic)
//   * fused epilogues: folded-RMSNorm row scale, SwiGLU, residual add (+ bf16 copy + norm
//     partials), bf16 or fp32 store
#include <algorithm>
#include <array>
#include <mutex>

#include "qwen38/common.h"
#include "kernels/bq_device.cuh"
#include "kernels/kernels.h"
#include "kernels/ptx.cuh"

namespace qwen38::kernels {
namespace {

using namespace qwen38::ptx;
using bqdev::SlabLane;

// ---------------------------------------------------------------------------------------------
// Tile configurations
// ---------------------------------------------------------------------------------------------
template <int WM_, int WN_, int WK_, int MT_, int TK_, int STAGES_, int MIN_BLOCKS_>
struct Cfg {
  static constexpr int WM = WM_, WN = WN_, WK = WK_, MT = MT_, TK = TK_, STAGES = STAGES_;
  static constexpr int MIN_BLOCKS = MIN_BLOCKS_;
  static constexpr int kWarps = WM * WN * WK;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int TM = WM * MT * 16;
  static constexpr int TN = WN * 64;
  static constexpr int kSlabs = TK / 16;       // k16 slabs per stage
  static constexpr int kGroups = TK / 64;      // scale groups per stage
  static constexpr int kAChunks = TK / 8;      // 16-byte chunks per A row
  static constexpr int kCStride = TN + 8;      // padded fp32 row stride of the C tile in smem
  static_assert(TK % 64 == 0 && kSlabs % WK == 0);
  static_assert(TN == 128 || TN == 256, "epilogue assumes 4 or 8 columns per lane");
};

//                    WM WN WK MT  TK STG MINB
using CfgM16  = Cfg<1, 2, 4, 1, 64, 4, 2>;   // decode, M <= 16
using CfgM32  = Cfg<1, 2, 4, 2, 64, 4, 1>;   // M <= 32 (2 CTAs/SM would spill at 128 regs)
using CfgM64  = Cfg<2, 2, 2, 2, 64, 4, 1>;   // M <= 64
using CfgM128 = Cfg<4, 2, 1, 2, 64, 4, 1>;   // prefill
using CfgM64K128 = Cfg<2, 2, 2, 2, 128, 3, 1>;  // TK=128: half the barriers per K

template <int BITS, class C>
struct Smem {
  static constexpr int kA = C::TM * C::TK * 2;
  static constexpr int kLo = C::WN * C::kSlabs * 512;
  static constexpr int kHi = C::WN * C::kSlabs * bqdev::hi_slab_bytes(BITS);
  static constexpr int kSc = C::WN * C::kGroups * 128;
  static constexpr int kStage = kA + kLo + kHi + kSc;
  static constexpr int kPipe = C::STAGES * kStage;
  static constexpr int kCTile = C::TM * C::kCStride * 4;
  static constexpr int kMain = kPipe > kCTile ? kPipe : kCTile;
  static constexpr int kTotal = kMain + C::TM * 4 + 16;  // + row scales + flag
  static_assert(kTotal <= 99 * 1024, "exceeds the sm_86 per-block shared memory limit");
};

// ---------------------------------------------------------------------------------------------
// Kernel parameters
// ---------------------------------------------------------------------------------------------
struct Params {
  const __nv_bfloat16* a;
  int m, n, k;
  const uint8_t* lo;
  const uint8_t* hi;
  const uint8_t* scales;
  // epilogue
  int epilogue;
  void* out;
  int ldo;
  int n_store;
  float* residual;
  __nv_bfloat16* xbf;
  float* partials_out;
  const float* partials_in;
  int norm_blocks;
  float inv_hidden;
  float eps;
  // schedule
  int tiles_m, tiles_n, iters_per_tile;
  int streamk;
  long long total_iters;
  int grid;
  float* workspace;
  int* counters;
};

__device__ __forceinline__ int swz(int row, int chunk) { return chunk ^ (row & 7); }

// Grouped rasterization (GROUP_M m-tiles share each n-tile's weights while they are hot in L2).
__device__ __forceinline__ void raster(int tile, int tiles_m, int tiles_n, int& mt, int& nt) {
  constexpr int kGroupM = 8;
  const int group_size = kGroupM * tiles_n;
  const int group = tile / group_size;
  const int first_m = group * kGroupM;
  const int gm = min(tiles_m - first_m, kGroupM);
  const int in_group = tile % group_size;
  mt = first_m + in_group % gm;
  nt = in_group / gm;
}

// CTA whose stream-K range contains global iteration i.
__device__ __forceinline__ int cta_of(long long i, long long total, long long grid) {
  return int(((i + 1) * grid - 1) / total);
}

// ---------------------------------------------------------------------------------------------
// Mainloop
// ---------------------------------------------------------------------------------------------
template <int BITS, class C>
__device__ __forceinline__ void load_stage(uint8_t* stage, const Params& p, int m0, int n0, int kit,
                                           uint64_t policy) {
  using S = Smem<BITS, C>;
  const int tid = threadIdx.x;
  // A: TM rows x TK columns
  auto* As = reinterpret_cast<__nv_bfloat16*>(stage);
  constexpr int kAChunksTotal = C::TM * C::kAChunks;
#pragma unroll
  for (int c = tid; c < kAChunksTotal; c += C::kThreads) {
    const int row = c / C::kAChunks, cc = c % C::kAChunks;
    const int grow = m0 + row;
    const bool ok = grow < p.m;
    const __nv_bfloat16* src = p.a + size_t(ok ? grow : 0) * p.k + size_t(kit) * C::TK + cc * 8;
    cp_async_16(As + row * C::TK + swz(row, cc) * 8, src, ok);
  }
  const int nb0 = n0 / 64;
  const size_t slabs_k = size_t(p.k) / 16;
  // lo plane: WN blocks of kSlabs contiguous 512-byte slabs
  uint8_t* lo_s = stage + S::kA;
  constexpr int kLoChunks = C::kSlabs * 32;
#pragma unroll
  for (int c = tid; c < C::WN * kLoChunks; c += C::kThreads) {
    const int wn = c / kLoChunks, off = c % kLoChunks;
    const uint8_t* src = p.lo + ((size_t(nb0 + wn) * slabs_k + size_t(kit) * C::kSlabs) * 512) + off * 16;
    cp_async_16_hint(lo_s + wn * C::kSlabs * 512 + off * 16, src, policy);
  }
  if constexpr (BITS != 4) {
    constexpr int kHiSlab = bqdev::hi_slab_bytes(BITS);
    uint8_t* hi_s = stage + S::kA + S::kLo;
    constexpr int kHiChunks = C::kSlabs * kHiSlab / 16;
#pragma unroll
    for (int c = tid; c < C::WN * kHiChunks; c += C::kThreads) {
      const int wn = c / kHiChunks, off = c % kHiChunks;
      const uint8_t* src =
          p.hi + ((size_t(nb0 + wn) * slabs_k + size_t(kit) * C::kSlabs) * kHiSlab) + off * 16;
      cp_async_16_hint(hi_s + wn * C::kSlabs * kHiSlab + off * 16, src, policy);
    }
  }
  // scales: WN blocks of kGroups x 128 bytes
  uint8_t* sc_s = stage + S::kA + S::kLo + S::kHi;
  constexpr int kScChunks = C::kGroups * 8;
  const size_t groups_k = size_t(p.k) / 64;
#pragma unroll
  for (int c = tid; c < C::WN * kScChunks; c += C::kThreads) {
    const int wn = c / kScChunks, off = c % kScChunks;
    const uint8_t* src = p.scales + ((size_t(nb0 + wn) * groups_k + size_t(kit) * C::kGroups) * 128) + off * 16;
    cp_async_16_hint(sc_s + wn * C::kGroups * 128 + off * 16, src, policy);
  }
}

template <int BITS, class C>
__device__ __forceinline__ void compute_stage(const uint8_t* stage, float (&acc)[C::MT][8][4], int wm, int wn,
                                              int wk, int lane) {
  using S = Smem<BITS, C>;
  const auto* As = reinterpret_cast<const __nv_bfloat16*>(stage);
  const uint8_t* lo_s = stage + S::kA;
  const uint8_t* hi_s = lo_s + S::kLo;
  const uint8_t* sc_s = hi_s + S::kHi;
#pragma unroll
  for (int si = 0; si < C::kSlabs / C::WK; ++si) {
    const int s = si * C::WK + wk;
    const int grp = s >> 2;
    // eight bf16 scales, entry j = column 8j + (lane >> 2)
    const uint4 sc4 = *reinterpret_cast<const uint4*>(sc_s + (wn * C::kGroups + grp) * 128 + (lane >> 2) * 16);
    const uint32_t sc[4] = {sc4.x, sc4.y, sc4.z, sc4.w};
    SlabLane sl;
    {
      const uint4 v = *reinterpret_cast<const uint4*>(lo_s + (wn * C::kSlabs + s) * 512 + lane * 16);
      sl.lo[0] = v.x; sl.lo[1] = v.y; sl.lo[2] = v.z; sl.lo[3] = v.w;
    }
    if constexpr (BITS == 5) {
      sl.hi[0] = *reinterpret_cast<const uint32_t*>(hi_s + (wn * C::kSlabs + s) * 128 + lane * 4);
    } else if constexpr (BITS == 6) {
      const uint2 v = *reinterpret_cast<const uint2*>(hi_s + (wn * C::kSlabs + s) * 256 + lane * 8);
      sl.hi[0] = v.x; sl.hi[1] = v.y;
    } else if constexpr (BITS == 8) {
      const uint4 v = *reinterpret_cast<const uint4*>(hi_s + (wn * C::kSlabs + s) * 512 + lane * 16);
      sl.hi[0] = v.x; sl.hi[1] = v.y; sl.hi[2] = v.z; sl.hi[3] = v.w;
    }
    uint32_t af[C::MT][4];
#pragma unroll
    for (int i = 0; i < C::MT; ++i) {
      const int row = (wm * C::MT + i) * 16 + (lane & 15);
      const int chunk = s * 2 + (lane >> 4);
      ldmatrix_x4(af[i], As + row * C::TK + swz(row, chunk) * 8);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const __nv_bfloat162 scale = as_bf162(prmt(sc[j >> 1], 0u, (j & 1) ? 0x3232u : 0x1010u));
      const uint32_t b0 = bqdev::pair_weights<BITS>(sl, 2 * j, scale);
      const uint32_t b1 = bqdev::pair_weights<BITS>(sl, 2 * j + 1, scale);
#pragma unroll
      for (int i = 0; i < C::MT; ++i) mma_bf16_16816(acc[i][j], af[i], b0, b1);
    }
  }
}

template <int BITS, class C>
__device__ __forceinline__ void mainloop(uint8_t* smem, const Params& p, int m0, int n0, int kb, int ke,
                                         float (&acc)[C::MT][8][4], int wm, int wn, int wk, int lane,
                                         uint64_t policy) {
  using S = Smem<BITS, C>;
#pragma unroll
  for (int i = 0; i < C::MT; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

  const int iters = ke - kb;
#pragma unroll
  for (int s = 0; s < C::STAGES - 1; ++s) {
    if (s < iters) load_stage<BITS, C>(smem + s * S::kStage, p, m0, n0, kb + s, policy);
    cp_async_commit();
  }
  for (int i = 0; i < iters; ++i) {
    cp_async_wait<C::STAGES - 2>();
    __syncthreads();  // stage i visible to all; every warp is done with stage i-1
    const int pf = i + C::STAGES - 1;
    if (pf < iters) load_stage<BITS, C>(smem + (pf % C::STAGES) * S::kStage, p, m0, n0, kb + pf, policy);
    cp_async_commit();
    compute_stage<BITS, C>(smem + (i % C::STAGES) * S::kStage, acc, wm, wn, wk, lane);
  }
  cp_async_wait<0>();
  __syncthreads();  // pipeline buffers free for the C tile
}

// Warps write (WK > 1: sum, in fixed wk order) their fragments into the fp32 C tile in smem.
template <class C>
__device__ __forceinline__ void accumulate_ctile(float* ctile, const float (&acc)[C::MT][8][4], int wm, int wn,
                                                 int wk, int lane) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int kk = 0; kk < C::WK; ++kk) {
    if (wk == kk) {
#pragma unroll
      for (int i = 0; i < C::MT; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int r0 = (wm * C::MT + i) * 16 + g;
          const int c = wn * 64 + j * 8 + 2 * t;
          float2* p0 = reinterpret_cast<float2*>(ctile + r0 * C::kCStride + c);
          float2* p1 = reinterpret_cast<float2*>(ctile + (r0 + 8) * C::kCStride + c);
          if (kk == 0) {
            *p0 = make_float2(acc[i][j][0], acc[i][j][1]);
            *p1 = make_float2(acc[i][j][2], acc[i][j][3]);
          } else {
            float2 a0 = *p0, a1 = *p1;
            a0.x += acc[i][j][0]; a0.y += acc[i][j][1];
            a1.x += acc[i][j][2]; a1.y += acc[i][j][3];
            *p0 = a0;
            *p1 = a1;
          }
        }
      }
    }
    __syncthreads();
  }
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

template <class C>
__device__ void epilogue(const Params& p, const float* ctile, float* rowscale, int m0, int n0) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (p.partials_in) {
    for (int r = tid; r < C::TM; r += C::kThreads) {
      const int row = m0 + r;
      float s = 0.0f;
      if (row < p.m) {
        const float* pr = p.partials_in + size_t(row) * p.norm_blocks;
        for (int b = 0; b < p.norm_blocks; ++b) s += pr[b];
      }
      rowscale[r] = rsqrtf(s * p.inv_hidden + p.eps);
    }
    __syncthreads();
  }
  constexpr int kCols = C::TN / 32;  // columns per lane: 4 or 8
  for (int r = warp; r < C::TM; r += C::kWarps) {
    const int row = m0 + r;
    if (row >= p.m) break;  // rows ascend with r; warp-uniform
    const float rs = p.partials_in ? rowscale[r] : 1.0f;
    const int c0 = lane * kCols;
    const int col = n0 + c0;
    float v[kCols];
#pragma unroll
    for (int i = 0; i < kCols; i += 4) {
      const float4 x = *reinterpret_cast<const float4*>(ctile + r * C::kCStride + c0 + i);
      v[i] = x.x; v[i + 1] = x.y; v[i + 2] = x.z; v[i + 3] = x.w;
    }
    const bool in_range = col < p.n_store;
    switch (p.epilogue) {
      case int(Epilogue::StoreBF16): {
        if (in_range) {
          uint32_t packed[kCols / 2];
#pragma unroll
          for (int i = 0; i < kCols / 2; ++i) packed[i] = pack_bf16x2(v[2 * i] * rs, v[2 * i + 1] * rs);
          auto* dst = reinterpret_cast<__nv_bfloat16*>(p.out) + size_t(row) * p.ldo + col;
          if constexpr (kCols == 4) {
            *reinterpret_cast<uint2*>(dst) = make_uint2(packed[0], packed[1]);
          } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
          }
        }
        break;
      }
      case int(Epilogue::StoreF32): {
        if (in_range) {
          float* dst = reinterpret_cast<float*>(p.out) + size_t(row) * p.ldo + col;
#pragma unroll
          for (int i = 0; i < kCols; i += 4)
            *reinterpret_cast<float4*>(dst + i) = make_float4(v[i] * rs, v[i + 1] * rs, v[i + 2] * rs, v[i + 3] * rs);
        }
        break;
      }
      case int(Epilogue::SwiGLU): {
        if (in_range) {
          uint32_t packed[kCols / 4];
#pragma unroll
          for (int i = 0; i < kCols / 4; ++i) {
            const float y0 = silu(v[4 * i] * rs) * (v[4 * i + 1] * rs);
            const float y1 = silu(v[4 * i + 2] * rs) * (v[4 * i + 3] * rs);
            packed[i] = pack_bf16x2(y0, y1);
          }
          auto* dst = reinterpret_cast<__nv_bfloat16*>(p.out) + size_t(row) * p.ldo + col / 2;
          if constexpr (kCols == 4) {
            *reinterpret_cast<uint32_t*>(dst) = packed[0];
          } else {
            *reinterpret_cast<uint2*>(dst) = make_uint2(packed[0], packed[1]);
          }
        }
        break;
      }
      case int(Epilogue::Residual): {
        float* res = p.residual + size_t(row) * p.n + col;
        float ss = 0.0f;
        uint32_t packed[kCols / 2];
#pragma unroll
        for (int i = 0; i < kCols; i += 4) {
          float4 x = *reinterpret_cast<const float4*>(res + i);
          x.x += v[i]; x.y += v[i + 1]; x.z += v[i + 2]; x.w += v[i + 3];
          *reinterpret_cast<float4*>(res + i) = x;
          ss += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
          packed[i / 2] = pack_bf16x2(x.x, x.y);
          packed[i / 2 + 1] = pack_bf16x2(x.z, x.w);
        }
        auto* xb = p.xbf + size_t(row) * p.n + col;
        if constexpr (kCols == 4) {
          *reinterpret_cast<uint2*>(xb) = make_uint2(packed[0], packed[1]);
        } else {
          *reinterpret_cast<uint4*>(xb) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        }
        // one partial per 128 columns: 32 lanes (TN=128) or 16 lanes (TN=256)
        constexpr int kLanesPerBlock = 128 / kCols;
#pragma unroll
        for (int off = kLanesPerBlock / 2; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
        if ((lane % kLanesPerBlock) == 0) p.partials_out[size_t(row) * p.norm_blocks + col / 128] = ss;
        break;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------------------------
template <int BITS, class C>
__global__ void __launch_bounds__(C::kThreads, C::MIN_BLOCKS) qgemm_kernel(const Params p) {
  using S = Smem<BITS, C>;
  extern __shared__ __align__(128) uint8_t smem[];
  float* ctile = reinterpret_cast<float*>(smem);
  float* rowscale = reinterpret_cast<float*>(smem + S::kMain);
  int* flag = reinterpret_cast<int*>(smem + S::kMain + C::TM * 4);

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int wm = warp % C::WM, wn = (warp / C::WM) % C::WN, wk = warp / (C::WM * C::WN);
  const uint64_t policy = l2_policy_evict_first();
  float acc[C::MT][8][4];

  if (!p.streamk) {
    int mt, nt;
    raster(blockIdx.x, p.tiles_m, p.tiles_n, mt, nt);
    mainloop<BITS, C>(smem, p, mt * C::TM, nt * C::TN, 0, p.iters_per_tile, acc, wm, wn, wk, lane, policy);
    accumulate_ctile<C>(ctile, acc, wm, wn, wk, lane);
    epilogue<C>(p, ctile, rowscale, mt * C::TM, nt * C::TN);
    return;
  }

  const long long total = p.total_iters, grid = p.grid;
  long long it = (long long)blockIdx.x * total / grid;
  const long long end = ((long long)blockIdx.x + 1) * total / grid;
  const int first_tile = int(it / p.iters_per_tile);
  while (it < end) {
    const int t = int(it / p.iters_per_tile);
    const int kb = int(it % p.iters_per_tile);
    const int ke = int(min((long long)p.iters_per_tile, kb + (end - it)));
    int mt, nt;
    raster(t, p.tiles_m, p.tiles_n, mt, nt);
    const int m0 = mt * C::TM, n0 = nt * C::TN;
    mainloop<BITS, C>(smem, p, m0, n0, kb, ke, acc, wm, wn, wk, lane, policy);
    accumulate_ctile<C>(ctile, acc, wm, wn, wk, lane);
    if (kb == 0 && ke == p.iters_per_tile) {
      epilogue<C>(p, ctile, rowscale, m0, n0);
    } else {
      const long long tile_begin = (long long)t * p.iters_per_tile;
      const int first = cta_of(tile_begin, total, grid);
      const int last = cta_of(tile_begin + p.iters_per_tile - 1, total, grid);
      const int nseg = last - first + 1;
      const int slot = t == first_tile ? 0 : 1;
      float* mine = p.workspace + (size_t(blockIdx.x) * 2 + slot) * (C::TM * C::TN);
      for (int e = threadIdx.x * 4; e < C::TM * C::TN; e += C::kThreads * 4) {
        const int r = e / C::TN, c = e % C::TN;
        *reinterpret_cast<float4*>(mine + e) = *reinterpret_cast<const float4*>(ctile + r * C::kCStride + c);
      }
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) {
        const int prev = atomicAdd(p.counters + t, 1);
        const int is_last = prev == nseg - 1;
        if (is_last) p.counters[t] = 0;  // every segment has arrived; reset for the next launch
        *flag = is_last;
      }
      __syncthreads();
      if (*flag) {
        __threadfence();
        for (int e = threadIdx.x * 4; e < C::TM * C::TN; e += C::kThreads * 4) {
          float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
          for (int sgm = 0; sgm < nseg; ++sgm) {
            const int cta = first + sgm;
            const int cta_first_tile = int(((long long)cta * total / grid) / p.iters_per_tile);
            const int cslot = t == cta_first_tile ? 0 : 1;
            const float4 x =
                __ldcg(reinterpret_cast<const float4*>(p.workspace + (size_t(cta) * 2 + cslot) * (C::TM * C::TN) + e));
            sum.x += x.x; sum.y += x.y; sum.z += x.z; sum.w += x.w;
          }
          const int r = e / C::TN, c = e % C::TN;
          *reinterpret_cast<float4*>(ctile + r * C::kCStride + c) = sum;
        }
        __syncthreads();
        epilogue<C>(p, ctile, rowscale, m0, n0);
      }
    }
    it += ke - kb;
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------
struct ConfigInfo {
  const char* name;
  int tm, tn;
};

constexpr std::array<ConfigInfo, 5> kConfigs = {{
    {"m16_n128_k64_s4", 16, 128},
    {"m32_n128_k64_s4", 32, 128},
    {"m64_n128_k64_s4", 64, 128},
    {"m128_n128_k64_s4", 128, 128},
    {"m64_n128_k128_s3", 64, 128},
}};

GemmTuningLookup g_tuning = nullptr;

struct LaunchInfo {
  int smem = 0;
  int threads = 0;
  int resident = 0;  // CTAs per SM
  int tm = 0, tn = 0;
};

int g_num_sms = 0;

int num_sms() {
  if (g_num_sms == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev);
  }
  return g_num_sms;
}

template <int BITS, class C>
const LaunchInfo& launch_info() {
  static LaunchInfo info = [] {
    LaunchInfo li;
    li.smem = Smem<BITS, C>::kTotal;
    li.threads = C::kThreads;
    li.tm = C::TM;
    li.tn = C::TN;
    auto kernel = qgemm_kernel<BITS, C>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, li.smem);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&li.resident, kernel, li.threads, li.smem);
    if (li.resident < 1) li.resident = 1;
    return li;
  }();
  return info;
}

template <int BITS, class C>
void launch(Params p, const GemmWorkspace& ws, cudaStream_t stream) {
  const LaunchInfo& li = launch_info<BITS, C>();
  p.tiles_m = int(ceil_div(p.m, C::TM));
  p.tiles_n = p.n / C::TN;
  p.iters_per_tile = p.k / C::TK;
  const long long tiles = (long long)p.tiles_m * p.tiles_n;
  const long long resident = (long long)li.resident * num_sms();
  int grid;
  if (tiles >= 3 * resident || ws.partials == nullptr) {
    p.streamk = 0;
    grid = int(tiles);
  } else {
    p.streamk = 1;
    p.total_iters = tiles * p.iters_per_tile;
    grid = int(std::min<long long>(resident, p.total_iters));
    p.grid = grid;
    p.workspace = ws.partials;
    p.counters = ws.counters;
    QWEN38_CHECK(size_t(grid) * 2 * C::TM * C::TN <= ws.partial_floats, "qgemm: stream-K workspace too small");
    QWEN38_CHECK(tiles <= ws.num_counters, "qgemm: too few tile counters");
  }
  qgemm_kernel<BITS, C><<<grid, C::kThreads, li.smem, stream>>>(p);
}

template <class C>
void launch_bits(int bits, const Params& p, const GemmWorkspace& ws, cudaStream_t stream) {
  switch (bits) {
    case 4: launch<4, C>(p, ws, stream); break;
    case 5: launch<5, C>(p, ws, stream); break;
    case 6: launch<6, C>(p, ws, stream); break;
    case 8: launch<8, C>(p, ws, stream); break;
    default: fail("qgemm: unsupported bit width " + std::to_string(bits));
  }
}

int default_config(int m) {
  if (m <= 16) return 0;
  if (m <= 32) return 1;
  if (m <= 64) return 2;
  return 3;
}

}  // namespace

int qgemm_num_configs() { return int(kConfigs.size()); }
const char* qgemm_config_name(int config) { return kConfigs.at(config).name; }
int qgemm_config_tile_m(int config) { return kConfigs.at(config).tm; }
void set_qgemm_tuning(GemmTuningLookup lookup) { g_tuning = lookup; }

void qgemm(const __nv_bfloat16* a, int m, const BQWeight& w, const GemmEpilogue& ep, const GemmWorkspace& ws,
           cudaStream_t stream, int config) {
  QWEN38_CHECK(m > 0, "qgemm: m must be positive");
  QWEN38_CHECK(w.n % 128 == 0 && w.k % 64 == 0, "qgemm: weight shape");
  Params p{};
  p.a = a;
  p.m = m;
  p.n = w.n;
  p.k = w.k;
  p.lo = w.lo;
  p.hi = w.hi;
  p.scales = reinterpret_cast<const uint8_t*>(w.scales);
  p.epilogue = int(ep.kind);
  p.out = ep.out;
  p.ldo = ep.ldo;
  p.n_store = ep.n_store > 0 ? ep.n_store : w.n;
  p.residual = ep.residual;
  p.xbf = ep.xbf;
  p.partials_out = ep.partials_out;
  p.partials_in = ep.partials_in;
  p.norm_blocks = ep.hidden / 128;
  p.inv_hidden = ep.hidden > 0 ? 1.0f / float(ep.hidden) : 0.0f;
  p.eps = ep.eps;
  if (ep.kind == Epilogue::Residual) {
    QWEN38_CHECK(ep.residual && ep.xbf && ep.partials_out, "qgemm: residual epilogue needs its buffers");
    p.norm_blocks = w.n / 128;
    p.n_store = w.n;
  }
  if (ep.partials_in) QWEN38_CHECK(ep.hidden % 128 == 0, "qgemm: norm hidden size");

  if (config == kGemmConfigAuto && g_tuning) config = g_tuning(w.bits, w.n, w.k, m);
  if (config == kGemmConfigAuto) config = default_config(m);
  switch (config) {
    case 0: launch_bits<CfgM16>(w.bits, p, ws, stream); break;
    case 1: launch_bits<CfgM32>(w.bits, p, ws, stream); break;
    case 2: launch_bits<CfgM64>(w.bits, p, ws, stream); break;
    case 3: launch_bits<CfgM128>(w.bits, p, ws, stream); break;
    case 4:
      QWEN38_CHECK(w.k % 128 == 0, "qgemm: config 4 needs K % 128 == 0");
      launch_bits<CfgM64K128>(w.bits, p, ws, stream);
      break;
    default: fail("qgemm: unknown config " + std::to_string(config));
  }
}

}  // namespace qwen38::kernels
