// ModelRunner: executes Qwen3.5-architecture forward steps on the GPU.
//
// It owns the weights, the paged KV cache, the GDN state pool, all activation buffers and the
// CUDA graphs used for decode. It knows nothing about requests: the Engine (engine.h) decides which
// sequence runs in which slot and calls prefill()/decode().
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "qwen38/device.h"
#include "qwen38/kv_cache.h"
#include "qwen38/model_file.h"
#include "qwen38/weights.h"
#include "kernels/kernels.h"

namespace qwen38 {

struct SamplingParams {
  float temperature = 1.0f;  // 0 = greedy
  int top_k = 20;            // 0 = off
  float top_p = 0.95f;       // 1 = off
  float min_p = 0.0f;
  float presence_penalty = 0.0f;
  float repetition_penalty = 1.0f;
  uint64_t seed = 0;
};

struct RunnerOptions {
  int max_batch = 8;              // concurrent sequences (each owns a 154 MB GDN state slot)
  int prefill_chunk = 1024;       // tokens per prefill step
  kernels::KvDtype kv_dtype = kernels::KvDtype::Int8;
  int64_t kv_tokens = 0;          // total KV capacity in tokens; 0 = everything that fits
  int max_context = 262144;       // per-sequence limit
  size_t reserve_mib = 640;       // VRAM headroom kept free (CUDA context growth, graphs, display)
  EmbeddingPlacement embedding = EmbeddingPlacement::Host;
  bool cuda_graphs = true;
  int history_cap = 1024;         // generated tokens remembered per sequence for penalties
  bool verbose = true;
};

struct DecodeRow {
  int slot = 0;
  int position = 0;       // position of `token`
  int32_t token = 0;
  SamplingParams sampling;
  int step = 0;           // sample counter of the sequence (RNG stream offset)
};

struct StepResult {
  std::vector<int32_t> tokens;
  std::vector<float> logprobs;
};

struct MemoryPlan {
  size_t free_at_start = 0;
  size_t weights = 0;
  size_t embedding_host = 0;
  size_t workspace = 0;
  size_t state = 0;
  size_t kv = 0;
  size_t reserve = 0;
  int kv_pages = 0;
  int64_t kv_tokens = 0;
  std::string describe() const;
};

// Accumulated GPU time per operation class (profile mode only).
struct OpProfile {
  std::map<std::string, double> ms;
  std::map<std::string, int> calls;
};

class ModelRunner {
 public:
  ModelRunner(const ModelFile& file, const RunnerOptions& options);
  ~ModelRunner();
  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;

  const ModelConfig& config() const;
  const RunnerOptions& options() const;
  const MemoryPlan& memory() const;
  const DeviceInfo& device() const;
  int num_slots() const;  // usable slots: [0, max_batch)
  KvCache& kv();
  const ModelWeights& weights() const;
  cudaStream_t stream() const;
  const kernels::GemmWorkspace& gemm_workspace() const;

  // Prepare `slot` for a new sequence (zero GDN state and penalty history).
  void reset_slot(int slot);

  // One prefill chunk: tokens occupy positions [pos0, pos0 + n) of the sequence in `slot`
  // (KV pages must already be reserved). If `sample` is given, the logits of the last token are
  // sampled and the result holds one token.
  StepResult prefill(int slot, int pos0, std::span<const int32_t> tokens, const SamplingParams* sample, int step);

  // One decode step: every row consumes one token and samples the next. Uses a CUDA graph per
  // batch-size bucket (rows are padded to the bucket with a private dummy sequence).
  StepResult decode(std::span<const DecodeRow> rows);

  // Teacher forcing for tests / perplexity: runs `tokens` as prefill chunks from position `pos0`
  // and returns log p(targets[i] | tokens[..i]) for every position.
  std::vector<float> score(int slot, int pos0, std::span<const int32_t> tokens, std::span<const int32_t> targets);

  // Full logits [n][vocab] of a prefill chunk (small models / tests only).
  std::vector<float> prefill_logits(int slot, int pos0, std::span<const int32_t> tokens);

  // Profile mode: per-op GPU timings recorded around every kernel of eager (non-graph) steps.
  void set_profiling(bool on);
  const OpProfile& profile() const;
  void clear_profile();

  // Force eager execution of decode steps (for profiling or debugging graphs).
  void set_graphs_enabled(bool on);

  struct Impl;  // defined in model_runner.cpp

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace qwen38
