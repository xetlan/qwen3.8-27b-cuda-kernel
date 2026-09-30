// Engine: request queue + continuous-batching scheduler on top of ModelRunner.
//
// Every scheduler iteration runs at most one prefill chunk (for the oldest sequence still
// consuming its prompt) and then one batched decode step for every sequence that is generating.
// Long prompts therefore never stall running generations for more than one chunk. Sequences
// join and leave the decode batch between iterations; the decode step is a replayed CUDA graph
// for the current batch-size bucket.
#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "qwen38/model_file.h"
#include "qwen38/model_runner.h"

namespace qwen38 {

struct GenerationConfig {
  SamplingParams sampling;
  int max_new_tokens = 4096;
  std::vector<int32_t> stop_tokens;  // empty = model EOS ids
  bool ignore_eos = false;           // benchmarks: always generate max_new_tokens
};

enum class FinishReason { None, Stop, Length, Error };
const char* finish_reason_name(FinishReason r);

struct RequestResult {
  int id = -1;
  std::vector<int32_t> prompt;
  std::vector<int32_t> tokens;
  std::vector<float> logprobs;
  FinishReason finish = FinishReason::None;
  std::string error;
  double submit_s = 0, first_token_s = 0, done_s = 0;  // engine clock, seconds
};

// Called for every generated token (including the stop token) on the scheduler thread.
using TokenCallback = std::function<void(int request_id, int32_t token, bool finished)>;

struct EngineOptions {
  RunnerOptions runner;
  uint64_t seed = 0x5eed;  // default per-request seed = seed + request id
};

struct EngineStats {
  int64_t prefill_tokens = 0;
  double prefill_s = 0;
  int64_t decode_tokens = 0;
  double decode_s = 0;
  int64_t decode_steps = 0;
  int64_t requests_done = 0;
  double tokens_per_s_decode() const { return decode_s > 0 ? double(decode_tokens) / decode_s : 0; }
  double tokens_per_s_prefill() const { return prefill_s > 0 ? double(prefill_tokens) / prefill_s : 0; }
  double avg_batch() const { return decode_steps ? double(decode_tokens) / double(decode_steps) : 0; }
};

class Engine {
 public:
  Engine(const ModelFile& file, const EngineOptions& options);
  ~Engine();

  int submit(std::vector<int32_t> prompt, GenerationConfig config, TokenCallback on_token = nullptr);
  bool step();  // one scheduler iteration; false when there is nothing to do
  void run();   // until every submitted request has finished
  std::vector<RequestResult> take_finished();
  bool idle() const;

  const EngineStats& stats() const { return stats_; }
  void reset_stats() { stats_ = EngineStats{}; }
  ModelRunner& runner() { return *runner_; }
  const ModelConfig& config() const { return runner_->config(); }

 private:
  struct Sequence;
  void admit();
  void finish(Sequence& s, FinishReason reason, const std::string& error = {});
  double now() const;

  EngineOptions opt_;
  std::unique_ptr<ModelRunner> runner_;
  std::vector<int32_t> eos_;
  std::deque<std::unique_ptr<Sequence>> waiting_;
  std::vector<std::unique_ptr<Sequence>> active_;
  std::vector<int> free_slots_;
  std::vector<RequestResult> finished_;
  EngineStats stats_;
  int next_id_ = 0;
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace qwen38
