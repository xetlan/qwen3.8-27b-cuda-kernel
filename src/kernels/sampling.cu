// Token sampling over the full vocabulary, one CTA (1024 threads) per sequence.
//
//   penalties     presence / repetition penalty over the sequence's generated tokens, each unique
//                 token penalised once (shared-memory bitmap)
//   greedy        temperature <= 0: argmax, lowest id on ties
//   top-k         exact k-th largest logit by 4-pass radix select (ties at the threshold kept)
//   min-p         p_i >= min_p * p_max  <=>  z_i >= log(min_p)
//   top-p         smallest nucleus with mass >= top_p, radix select on probability mass
//   sample        index-order inverse CDF with a counter-based RNG (seed, step)
//
// All sums of probability mass use 2^-40 fixed point in 64-bit integers, so the chosen token is a
// deterministic function of (logits, params, seed, step): no float atomics anywhere.
#include <cstdint>

#include "qwen38/common.h"
#include "kernels/kernels.h"

namespace qwen38::kernels {
namespace {

constexpr int kThreads = 1024;
constexpr double kFix = 1099511627776.0;  // 2^40

__device__ __forceinline__ uint32_t ordered_key(float x) {
  const uint32_t u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

struct SampleSmem {
  uint32_t bitmap[(256 * 1024) / 32];  // history de-duplication, vocab <= 262144
  unsigned long long hist[256];
  unsigned long long scan[kThreads / 32];
  float red_f[kThreads / 32];
  int red_i[kThreads / 32];
  unsigned long long red_u[kThreads / 32];
  uint32_t sel_prefix;
  unsigned long long sel_above;
};

__device__ __forceinline__ float block_max(float v, SampleSmem& sm) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  if (lane == 0) sm.red_f[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = sm.red_f[lane];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (lane == 0) sm.red_f[0] = v;
  }
  __syncthreads();
  v = sm.red_f[0];
  __syncthreads();
  return v;
}

__device__ __forceinline__ unsigned long long block_sum_u64(unsigned long long v, SampleSmem& sm) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if (lane == 0) sm.red_u[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = sm.red_u[lane];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) sm.red_u[0] = v;
  }
  __syncthreads();
  v = sm.red_u[0];
  __syncthreads();
  return v;
}

// argmax with lowest-index tie break
__device__ __forceinline__ int block_argmax(float v, int idx, SampleSmem& sm) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
    if (ov > v || (ov == v && oi < idx)) {
      v = ov;
      idx = oi;
    }
  }
  if (lane == 0) {
    sm.red_f[warp] = v;
    sm.red_i[warp] = idx;
  }
  __syncthreads();
  if (warp == 0) {
    v = sm.red_f[lane];
    idx = sm.red_i[lane];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, v, o);
      const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
      if (ov > v || (ov == v && oi < idx)) {
        v = ov;
        idx = oi;
      }
    }
    if (lane == 0) sm.red_i[0] = idx;
  }
  __syncthreads();
  idx = sm.red_i[0];
  __syncthreads();
  return idx;
}

// After the histogram of one radix digit is complete, pick the digit whose cumulative weight
// (from the top) first reaches `target`. Returns via sm.sel_prefix / sm.sel_above.
__device__ __forceinline__ void pick_digit(SampleSmem& sm, unsigned long long target, unsigned long long above,
                                           uint32_t prefix, int shift) {
  if (threadIdx.x == 0) {
    unsigned long long cum = above;
    int d = 255;
    for (; d > 0; --d) {
      if (cum + sm.hist[d] >= target) break;
      cum += sm.hist[d];
    }
    sm.sel_prefix = prefix | (uint32_t(d) << shift);
    sm.sel_above = cum;
  }
  __syncthreads();
}

__global__ void __launch_bounds__(kThreads, 1) sample_kernel(SamplingArgs a) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  SampleSmem& sm = *reinterpret_cast<SampleSmem*>(smem_raw);
  const int b = blockIdx.x, tid = threadIdx.x;
  const SamplingParamsDev prm = a.params[b];
  float* logits = a.logits + size_t(b) * a.ld;
  const int V = a.vocab;
  const int slot = a.row_slot ? a.row_slot[b] : b;
  // appends the chosen token to the slot's history ring (called by exactly one thread)
  auto commit = [&](int tok, float logprob) {
    a.tokens_out[b] = tok;
    if (a.logprob_out) a.logprob_out[b] = logprob;
    if (a.history) {
      const int len = a.history_len[slot];
      a.history[size_t(slot) * a.history_cap + (len % a.history_cap)] = tok;
      a.history_len[slot] = len + 1;
    }
  };

  // ---- penalties ----------------------------------------------------------------------------
  const bool penalize = prm.presence_penalty != 0.0f || prm.repetition_penalty != 1.0f;
  if (penalize && a.history) {
    const int words = (V + 31) / 32;
    for (int i = tid; i < words; i += kThreads) sm.bitmap[i] = 0u;
    __syncthreads();
    const int n = min(a.history_len[slot], a.history_cap);
    const int32_t* hist = a.history + size_t(slot) * a.history_cap;
    for (int i = tid; i < n; i += kThreads) {
      const int t = hist[i];
      if (t < 0 || t >= V) continue;
      const uint32_t bit = 1u << (t & 31);
      if (atomicOr(&sm.bitmap[t >> 5], bit) & bit) continue;  // already penalised
      float l = logits[t];
      if (prm.repetition_penalty != 1.0f) l = l > 0.0f ? l / prm.repetition_penalty : l * prm.repetition_penalty;
      logits[t] = l - prm.presence_penalty;
    }
    __syncthreads();
  }

  // ---- greedy ---------------------------------------------------------------------------------
  float best = -INFINITY;
  int best_i = 0x7fffffff;
  for (int i = tid; i < V; i += kThreads) {
    const float l = logits[i];
    if (l > best) {
      best = l;
      best_i = i;
    }
  }
  if (prm.temperature <= 0.0f) {
    const int tok = block_argmax(best, best_i, sm);
    if (tid == 0) commit(tok, 0.0f);  // logprob not computed for greedy
    return;
  }
  const float M = block_max(best, sm);
  const float inv_t = 1.0f / prm.temperature;

  // ---- thresholds on the ordered key of the raw logit ----------------------------------------
  uint32_t thresh = 0u;  // keep keys >= thresh
  if (prm.top_k > 0 && prm.top_k < V) {
    uint32_t prefix = 0u, mask = 0u;
    unsigned long long above = 0ull;
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int i = tid; i < 256; i += kThreads) sm.hist[i] = 0ull;
      __syncthreads();
      for (int i = tid; i < V; i += kThreads) {
        const uint32_t key = ordered_key(logits[i]);
        if ((key & mask) == prefix) atomicAdd(&sm.hist[(key >> shift) & 255u], 1ull);
      }
      __syncthreads();
      pick_digit(sm, (unsigned long long)prm.top_k, above, prefix, shift);
      prefix = sm.sel_prefix;
      above = sm.sel_above;
      mask |= 255u << shift;
      __syncthreads();
    }
    thresh = prefix;
  }
  if (prm.min_p > 0.0f) {
    // p_i / p_max = exp((l_i - M) / T) >= min_p  <=>  l_i >= M + T * log(min_p)
    const float lmin = M + prm.temperature * logf(prm.min_p);
    thresh = max(thresh, ordered_key(lmin));
  }

  // probability weights of the current candidates, 2^-40 fixed point
  auto weight = [&](float l) -> unsigned long long {
    return (unsigned long long)(double(__expf((l - M) * inv_t)) * kFix);
  };

  if (prm.top_p < 1.0f) {
    unsigned long long local = 0ull;
    for (int i = tid; i < V; i += kThreads) {
      const float l = logits[i];
      if (ordered_key(l) >= thresh) local += weight(l);
    }
    const unsigned long long W = block_sum_u64(local, sm);
    const unsigned long long target = (unsigned long long)(double(prm.top_p) * double(W));
    uint32_t prefix = 0u, mask = 0u;
    unsigned long long above = 0ull;
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int i = tid; i < 256; i += kThreads) sm.hist[i] = 0ull;
      __syncthreads();
      for (int i = tid; i < V; i += kThreads) {
        const float l = logits[i];
        const uint32_t key = ordered_key(l);
        if (key >= thresh && (key & mask) == prefix) atomicAdd(&sm.hist[(key >> shift) & 255u], weight(l));
      }
      __syncthreads();
      pick_digit(sm, target > 0 ? target : 1ull, above, prefix, shift);
      prefix = sm.sel_prefix;
      above = sm.sel_above;
      mask |= 255u << shift;
      __syncthreads();
    }
    thresh = max(thresh, prefix);
  }

  // ---- inverse CDF in index order over contiguous per-thread chunks ---------------------------
  const int chunk = (V + kThreads - 1) / kThreads;
  const int i0 = tid * chunk, i1 = min(V, i0 + chunk);
  unsigned long long own = 0ull, all = 0ull;
  for (int i = i0; i < i1; ++i) {
    const float l = logits[i];
    const unsigned long long w = weight(l);
    all += w;
    if (ordered_key(l) >= thresh) own += w;
  }
  // exclusive scan of `own`
  const int lane = tid & 31, warp = tid >> 5;
  unsigned long long incl = own;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned long long y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  if (lane == 31) sm.scan[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    unsigned long long w = sm.scan[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned long long y = __shfl_up_sync(0xffffffffu, w, o);
      if (lane >= o) w += y;
    }
    sm.scan[lane] = w;  // inclusive per-warp totals
  }
  __syncthreads();
  const unsigned long long total = sm.scan[31];
  const unsigned long long before = (warp > 0 ? sm.scan[warp - 1] : 0ull) + incl - own;
  const unsigned long long all_sum = block_sum_u64(all, sm);

  const int step = a.step ? a.step[b] : 0;
  const uint64_t r = splitmix64(prm.seed ^ (0xD1B54A32D192ED03ull * uint64_t(step + 1)));
  const double u01 = double(r >> 11) * (1.0 / 9007199254740992.0);  // [0, 1)
  const unsigned long long u = (unsigned long long)(u01 * double(total));

  if (own > 0 && u >= before && u < before + own) {
    unsigned long long cum = before;
    int tok = -1, last = -1;
    for (int i = i0; i < i1; ++i) {
      const float l = logits[i];
      if (ordered_key(l) < thresh) continue;
      last = i;
      cum += weight(l);
      if (u < cum) {
        tok = i;
        break;
      }
    }
    if (tok < 0) tok = last;
    const double p_tok = double(weight(logits[tok])) / double(all_sum > 0 ? all_sum : 1ull);
    commit(tok, float(log(p_tok > 0.0 ? p_tok : 1e-300)));
  }
}

__global__ void __launch_bounds__(256) target_logprob_kernel(const float* __restrict__ logits, int ld, int vocab,
                                                            const int32_t* __restrict__ targets, float* __restrict__ out) {
  __shared__ float red[8];
  const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* row = logits + size_t(r) * ld;
  float mx = -INFINITY;
  for (int i = tid; i < vocab; i += 256) mx = fmaxf(mx, row[i]);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  mx = red[0];
  for (int w = 1; w < 8; ++w) mx = fmaxf(mx, red[w]);
  __syncthreads();
  double sum = 0.0;
  for (int i = tid; i < vocab; i += 256) sum += exp(double(row[i] - mx));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  __shared__ double red_d[8];
  if (lane == 0) red_d[warp] = sum;
  __syncthreads();
  if (tid == 0) {
    double total = 0.0;
    for (int w = 0; w < 8; ++w) total += red_d[w];
    const int t = targets[r];
    out[r] = (t >= 0 && t < vocab) ? float(double(row[t] - mx) - log(total)) : 0.0f;
  }
}

}  // namespace

void target_logprobs(const float* logits, int ld, int vocab, const int32_t* targets, int rows, float* out,
                     cudaStream_t stream) {
  if (rows == 0) return;
  target_logprob_kernel<<<rows, 256, 0, stream>>>(logits, ld, vocab, targets, out);
}

void sample(const SamplingArgs& a, int batch, cudaStream_t stream) {
  QWEN38_CHECK(a.vocab > 0 && a.vocab <= 256 * 1024, "sample: vocab size");
  if (batch == 0) return;
  static const bool attr = cudaFuncSetAttribute(sample_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                int(sizeof(SampleSmem))) == cudaSuccess;
  (void)attr;
  sample_kernel<<<batch, kThreads, sizeof(SampleSmem), stream>>>(a);
}

}  // namespace qwen38::kernels
