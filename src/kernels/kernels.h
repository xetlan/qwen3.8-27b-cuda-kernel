// Host-side launch API of all CUDA kernels (internal to the engine).
//
// Conventions: row-major activations, M = number of tokens in the step. All pointers are device
// pointers unless stated otherwise. Every launcher is asynchronous on `stream`, performs no
// allocation and no host synchronization, and is safe to capture into a CUDA graph (all
// data-dependent sizes are read from device memory).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace qwen38::kernels {

// ---------------------------------------------------------------------------------------------
// Batch description of one engine step. A step runs M tokens that belong to B sequences; each
// sequence contributes a contiguous run of tokens (a prefill chunk, or one token when decoding).
// ---------------------------------------------------------------------------------------------
struct BatchDesc {
  int num_seqs = 0;               // B (host value; fixed per captured graph)
  int num_tokens = 0;             // M (host value; fixed per captured graph)
  int max_tokens_per_seq = 0;     // upper bound of the per-sequence run length (host value)
  const int* seq_start = nullptr; // [B + 1] offsets of each sequence's tokens
  const int* seq_slot = nullptr;  // [B] state slot = block-table row
  const int* seq_pos0 = nullptr;  // [B] position of the sequence's first token in this step
  const int* tok_pos = nullptr;   // [M] absolute position of every token
  const int* tok_seq = nullptr;   // [M] sequence index (0..B-1) of every token
};

// ---------------------------------------------------------------------------------------------
// Paged KV cache (docs/FORMAT.md 3.1). One pool; page ids are shared by all attention layers.
// ---------------------------------------------------------------------------------------------
enum class KvDtype : int { Int8 = 0, BF16 = 1 };

constexpr int kPageTokens = 64;
constexpr int kHeadDim = 256;

struct KvCacheDesc {
  uint8_t* pool = nullptr;        // base of the pool
  KvDtype dtype = KvDtype::Int8;
  int kv_heads = 0;
  int num_pages = 0;
  size_t page_bytes = 0;          // bytes per (layer, page)
  size_t layer_stride = 0;        // bytes per attention layer = num_pages * page_bytes
  const int* block_table = nullptr;  // [slots][max_pages_per_seq]
  int max_pages_per_seq = 0;

  // byte offsets inside one (layer, page) block
  __host__ __device__ size_t elem_bytes() const { return dtype == KvDtype::Int8 ? 1 : 2; }
  __host__ __device__ size_t k_offset(int head, int tok) const {
    return (size_t(head) * kPageTokens + tok) * kHeadDim * elem_bytes();
  }
  __host__ __device__ size_t v_offset(int head, int tok) const {
    return size_t(kv_heads) * kPageTokens * kHeadDim * elem_bytes() + k_offset(head, tok);
  }
  // int8 only: 4 fp16 scales (one per 64-dim group) per (head, token)
  __host__ __device__ size_t ks_offset(int head, int tok) const {
    return 2 * size_t(kv_heads) * kPageTokens * kHeadDim + (size_t(head) * kPageTokens + tok) * 4 * 2;
  }
  __host__ __device__ size_t vs_offset(int head, int tok) const {
    return ks_offset(head, tok) + size_t(kv_heads) * kPageTokens * 4 * 2;
  }
  static size_t page_bytes_for(KvDtype d, int kv_heads) {
    return d == KvDtype::Int8 ? 2 * size_t(kv_heads) * kPageTokens * (kHeadDim + 4 * 2)
                              : 2 * size_t(kv_heads) * kPageTokens * kHeadDim * 2;
  }
};

// ---------------------------------------------------------------------------------------------
// Quantized GEMM (BQ weights, bf16 activations, fp32 accumulate) with fused epilogues.
// ---------------------------------------------------------------------------------------------
enum class Epilogue : int {
  StoreBF16 = 0,  // out = bf16(C * rs)
  SwiGLU = 1,     // columns (2i, 2i+1) = (gate, up): out[i] = bf16(silu(gate * rs) * up * rs)
  Residual = 2,   // residual += C; xbf = bf16(residual); norm partials of the new residual
  StoreF32 = 3,   // out = C * rs
};

struct BQWeight {
  const uint8_t* lo = nullptr;
  const uint8_t* hi = nullptr;
  const __nv_bfloat16* scales = nullptr;
  int bits = 4;
  int n = 0;     // padded rows (multiple of 128)
  int k = 0;     // multiple of 64
  int rows = 0;  // logical rows
};

struct GemmEpilogue {
  Epilogue kind = Epilogue::StoreBF16;
  void* out = nullptr;              // bf16 (StoreBF16, SwiGLU) or float (StoreF32)
  int ldo = 0;                      // elements
  int n_store = 0;                  // columns of C to store (0 = all); SwiGLU counts C columns
  float* residual = nullptr;        // Residual: [M][n]
  __nv_bfloat16* xbf = nullptr;     // Residual: [M][n]
  float* partials_out = nullptr;    // Residual: [M][n / 128]
  // Row scale of a folded RMSNorm: rs = rsqrt(sum(partials_in[row]) / hidden + eps); null = 1.
  const float* partials_in = nullptr;
  int hidden = 0;
  float eps = 1e-6f;
};

struct GemmWorkspace {
  float* partials = nullptr;  // stream-K partial tiles
  size_t partial_floats = 0;
  int* counters = nullptr;    // per-tile arrival counters, zero-initialized, self-resetting
  int num_counters = 0;
};

// Compiled tile configurations (see qgemm.cu); kAuto picks by M or from the tuning table.
constexpr int kGemmConfigAuto = -1;
int qgemm_num_configs();
const char* qgemm_config_name(int config);
int qgemm_config_tile_m(int config);

// C[M][N] = A[M][K] * W^T, A bf16 row-major with leading dimension K.
void qgemm(const __nv_bfloat16* a, int m, const BQWeight& w, const GemmEpilogue& ep, const GemmWorkspace& ws,
           cudaStream_t stream, int config = kGemmConfigAuto);

// Tuning table hook: returns the config for (bits, n, k, m) or kGemmConfigAuto.
using GemmTuningLookup = int (*)(int bits, int n, int k, int m);
void set_qgemm_tuning(GemmTuningLookup lookup);

// ---------------------------------------------------------------------------------------------
// Embedding, residual stream initialisation
// ---------------------------------------------------------------------------------------------

// residual[m] = table[tokens[m]] (bf16 -> fp32), xbf = bf16 copy, partials = sum of squares per
// 128 columns. `table` may be device memory or mapped pinned host memory (zero-copy).
void embed(const __nv_bfloat16* table, const int32_t* tokens, int m, int hidden, float* residual,
           __nv_bfloat16* xbf, float* partials, cudaStream_t stream);

// ---------------------------------------------------------------------------------------------
// Full-attention layers
// ---------------------------------------------------------------------------------------------

// q/k offset-RMSNorm, partial RoPE (first 64 dims, rotate-half pairs (i, i+32)), q -> fp16,
// k/v -> paged cache (int8 per 64-group or bf16).
struct QkNormRopeParams {
  const __nv_bfloat16* proj = nullptr;  // [M][ld_proj]: [q nq*256 | gate nq*256 | k nkv*256 | v nkv*256]
  int ld_proj = 0;
  const float* q_norm = nullptr;        // [256], already 1 + w
  const float* k_norm = nullptr;
  float eps = 1e-6f;
  float rope_theta = 1e7f;
  int num_q_heads = 0;
  int num_kv_heads = 0;
  __half* q_out = nullptr;              // [M][nq][256]
  int attn_layer = 0;                   // index among attention layers
};
void qk_norm_rope_append(const QkNormRopeParams& p, const BatchDesc& batch, const KvCacheDesc& kv,
                         cudaStream_t stream);

struct AttentionParams {
  const __half* q = nullptr;            // [M][nq][256]
  const __nv_bfloat16* gate = nullptr;  // proj + nq*256, row stride ld_gate
  int ld_gate = 0;
  __nv_bfloat16* out = nullptr;         // [M][nq*256]
  int num_q_heads = 0;
  int num_kv_heads = 0;
  int attn_layer = 0;
  // split-KV decode scratch: [B][nq][max_splits][256 + 2] floats
  float* split_scratch = nullptr;
  int max_splits = 0;
};

// Decode: one token per sequence (M == B). Split-KV flash-decoding on CUDA cores.
// attention_decode_splits() is the split count the launcher will use (size scratch with it).
int attention_decode_splits(int batch_seqs, int kv_heads, int num_sms, int max_splits);
void attention_decode(const AttentionParams& p, const BatchDesc& batch, const KvCacheDesc& kv,
                      cudaStream_t stream);

// Prefill: causal attention of each sequence's new tokens over its cached prefix + themselves
// (the new K/V must already be appended). Tensor cores (CuTe, FlashAttention-2 schedule).
void attention_prefill(const AttentionParams& p, const BatchDesc& batch, const KvCacheDesc& kv,
                       cudaStream_t stream);

// ---------------------------------------------------------------------------------------------
// Gated DeltaNet layers
// ---------------------------------------------------------------------------------------------
struct GdnDims {
  int num_k_heads = 0;   // 16
  int num_v_heads = 0;   // 48
  int conv_dim = 0;      // 2 * nk * 128 + nv * 128
  int ld_proj = 0;       // row stride of the qkvz projection output
};

struct GdnStateDesc {
  float* recurrent = nullptr;      // [slots][gdn_layers][nv][128 dv][128 dk]
  __nv_bfloat16* conv = nullptr;   // [slots][gdn_layers][3][conv_dim]
  int num_gdn_layers = 0;
};

// a/b projection (bf16 weights [2*nv][H], norm folded): ab[m][j] (fp32), j < nv -> b, else a.
void gdn_ab(const __nv_bfloat16* xbf, int m, int hidden, const __nv_bfloat16* w_ab, int rows,
            const float* partials_in, float eps, float* ab_out, cudaStream_t stream);

// Causal depthwise conv (width 4) + SiLU over [q|k|v], q/k L2 norm (q also scaled by 1/sqrt(128)).
// Updates the conv state. Output: qkv [M][conv_dim] fp32 = [q nk*128 | k nk*128 | v nv*128].
struct GdnConvParams {
  const __nv_bfloat16* proj = nullptr;  // [M][ld_proj]
  const float* conv_w = nullptr;        // [conv_dim][4]
  float* qkv = nullptr;                 // [M][conv_dim]
  int gdn_layer = 0;
};
void gdn_conv(const GdnConvParams& p, const GdnDims& d, const BatchDesc& batch, const GdnStateDesc& st,
              cudaStream_t stream);

// Gates beta = sigmoid(b), decay = exp(A * softplus(a + dt_bias)); recurrence
// S = decay*S + beta*(v - decay*S k) k^T, o = S q; gated RMSNorm y = rmsnorm(o) * norm_w * silu(z)
// -> bf16 [M][nv*128]. Updates the recurrent state.
struct GdnRecurrenceParams {
  const float* qkv = nullptr;        // [M][conv_dim] from gdn_conv
  const float* ab = nullptr;         // [M][2 * nv]: b then a
  const float* A = nullptr;          // [nv]  (-exp(A_log))
  const float* dt_bias = nullptr;    // [nv]
  const __nv_bfloat16* z = nullptr;  // proj + conv_dim, row stride ld_z
  int ld_z = 0;
  const float* norm_w = nullptr;     // [128]
  float eps = 1e-6f;
  __nv_bfloat16* out = nullptr;      // [M][nv*128]
  int gdn_layer = 0;
};
void gdn_recurrence(const GdnRecurrenceParams& p, const GdnDims& d, const BatchDesc& batch,
                    const GdnStateDesc& st, cudaStream_t stream);

// ---------------------------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------------------------
struct SamplingParamsDev {
  float temperature;   // 0 = greedy
  int top_k;           // 0 = off
  float top_p;         // 1 = off
  float min_p;         // 0 = off
  float presence_penalty;
  float repetition_penalty;  // 1 = off
  uint64_t seed;
};

struct SamplingArgs {
  float* logits = nullptr;                    // [B][ld], modified in place (penalties)
  int ld = 0;
  int vocab = 0;                              // valid ids are [0, vocab)
  const SamplingParamsDev* params = nullptr;  // [B]
  const int* row_slot = nullptr;              // [B] slot owning each row's history
  int32_t* history = nullptr;                 // [slots][history_cap] generated tokens (ring); the
  int* history_len = nullptr;                 // [slots]  sampled token is appended by the kernel
  int history_cap = 0;
  const int* step = nullptr;                  // [B] per-sequence sample counter (RNG offset)
  int32_t* tokens_out = nullptr;              // [B]
  float* logprob_out = nullptr;               // [B] log-probability of the chosen token (optional)
};
void sample(const SamplingArgs& a, int batch, cudaStream_t stream);

// out[r] = log softmax(logits[r])[targets[r]] (natural log), for perplexity / scoring.
void target_logprobs(const float* logits, int ld, int vocab, const int32_t* targets, int rows, float* out,
                     cudaStream_t stream);

}  // namespace qwen38::kernels
