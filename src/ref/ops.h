// CPU reference implementations of every engine operation.
//
// These are the oracle for the CUDA kernels (tests/gpu) and the building blocks of the
// reference model (ref_model.h). Each function mirrors one kernel's contract, including where
// the engine rounds to bf16/fp16, unless `exact` is requested (then everything stays fp32,
// which is what the Hugging Face comparison in tests/host uses).
#pragma once

#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "qwen38/bq.h"
#include "qwen38/config.h"

namespace qwen38::ref {

enum class KvDtype { F32, BF16, Int8 };

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// Row scale of the folded RMSNorm: rsqrt(sum(partials) / H + eps).
float norm_row_scale(const float* partials, int blocks, int hidden, float eps);

// Sum of squares of `row` over 128-column blocks -> partials[H / 128].
void norm_partials(const float* row, int hidden, float* partials);

// C[m][n] = sum_k A[m][k] * W[n][k], fp32 accumulation (double internally for determinism).
// `w` is a dequantized [n_rows, K] matrix.
void matmul_nt(const float* a, int m, int k, const float* w, int n, float* c);

enum class Epilogue { StoreBF16, SwiGLU, Residual, StoreF32 };

struct GemmOut {
  // StoreBF16 / SwiGLU / StoreF32: out [M][ldo]
  float* out = nullptr;
  int ldo = 0;
  // Residual: residual [M][H] (updated in place), xbf [M][H], partials [M][H/128]
  float* residual = nullptr;
  float* xbf = nullptr;
  float* partials = nullptr;
};

// Applies one quantized projection with the engine's fused epilogue.
// a: [M][K] (values already bf16-exact in engine mode), row_scale: per-row norm scale or null.
void qgemm(const float* a, int m, const std::vector<float>& w_dequant, int n, int k,
           const float* row_scale, Epilogue ep, const GemmOut& out, bool exact);

// ---- attention ------------------------------------------------------------------------------

struct KvStore {
  // Dequantized cache contents for one sequence and one attention layer:
  // k/v [kv_heads][capacity][head_dim] (values as the attention kernel sees them).
  int kv_heads = 0, head_dim = 0, capacity = 0;
  std::vector<float> k, v;
  void resize(int heads, int dim, int cap);
  float* k_at(int head, int pos) { return k.data() + (size_t(head) * capacity + pos) * head_dim; }
  float* v_at(int head, int pos) { return v.data() + (size_t(head) * capacity + pos) * head_dim; }
};

// Symmetric int8 quantize-dequantize per 64-element group (FORMAT.md 3.1), in place.
// Scales are rounded to fp16 like the cache stores them.
void int8_group_roundtrip(float* x, int n);

// q/k RMSNorm (offset weights already 1+w), partial RoPE, KV append.
// proj row layout: [q nq*hd | gate nq*hd | k nkv*hd | v nkv*hd].
// q_out: [nq][hd]. Writes position `pos` of `kv`.
void qk_norm_rope_append(const ModelConfig& cfg, const float* proj_row, int pos, const float* q_norm,
                         const float* k_norm, float* q_out, KvStore& kv, KvDtype kv_dtype, bool exact);

// Causal GQA attention for one query token at position `pos` over kv[0..pos],
// followed by the output gate: out[h][d] = attn * sigmoid(gate).
void attention_token(const ModelConfig& cfg, const float* q /*[nq][hd]*/, const float* gate /*[nq*hd]*/,
                     int pos, const KvStore& kv, float* out /*[nq*hd]*/, bool exact);

// ---- Gated DeltaNet -------------------------------------------------------------------------

struct GdnState {
  // recurrent [v_heads][dv][dk], conv [taps-1][conv_dim] (oldest first)
  std::vector<float> s;
  std::vector<float> conv;
};

// Full GDN mixer for one token (conv, gates, l2norm, recurrence, gated norm).
// proj_row: [q | k | v | z] (bf16-exact values), ab_row: [b (lv) | a (lv) | pad] fp32.
// out: [lv * dv] (engine mode: bf16-rounded).
void gdn_token(const ModelConfig& cfg, const float* proj_row, const float* ab_row, const float* conv_w,
               const float* A, const float* dt_bias, const float* norm_w, GdnState& st, float* out,
               bool exact);

// ---- sampling -------------------------------------------------------------------------------

struct SamplingParams {
  float temperature = 1.0f;  // 0 = greedy
  int top_k = 0;             // 0 = disabled
  float top_p = 1.0f;
  float min_p = 0.0f;
  float presence_penalty = 0.0f;
  float repetition_penalty = 1.0f;
  uint64_t seed = 0;
};

}  // namespace qwen38::ref
