#include "qwen38/engine.h"

#include <algorithm>

#include "qwen38/common.h"

namespace qwen38 {

const char* finish_reason_name(FinishReason r) {
  switch (r) {
    case FinishReason::None: return "none";
    case FinishReason::Stop: return "stop";
    case FinishReason::Length: return "length";
    case FinishReason::Error: return "error";
  }
  return "?";
}

struct Engine::Sequence {
  RequestResult result;
  GenerationConfig config;
  TokenCallback on_token;
  int slot = -1;
  int prefilled = 0;       // prompt tokens already in the cache
  int position = 0;        // next position to write
  int32_t pending = -1;    // last sampled token (input of the next decode step)
  int samples = 0;
  bool decoding() const { return prefilled == int(result.prompt.size()); }
};

Engine::Engine(const ModelFile& file, const EngineOptions& options)
    : opt_(options), t0_(std::chrono::steady_clock::now()) {
  runner_ = std::make_unique<ModelRunner>(file, opt_.runner);
  eos_ = file.config().eos_token_ids;
  for (int s = runner_->num_slots() - 1; s >= 0; --s) free_slots_.push_back(s);
}

Engine::~Engine() = default;

double Engine::now() const {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
}

int Engine::submit(std::vector<int32_t> prompt, GenerationConfig config, TokenCallback on_token) {
  auto s = std::make_unique<Sequence>();
  s->result.id = next_id_++;
  s->result.prompt = std::move(prompt);
  s->result.submit_s = now();
  if (config.sampling.seed == 0) config.sampling.seed = opt_.seed + uint64_t(s->result.id) * 0x9E3779B97F4A7C15ull;
  if (config.stop_tokens.empty() && !config.ignore_eos) config.stop_tokens = eos_;
  s->config = std::move(config);
  s->on_token = std::move(on_token);
  const int id = s->result.id;
  if (s->result.prompt.empty()) {
    s->result.finish = FinishReason::Error;
    s->result.error = "empty prompt";
    finished_.push_back(std::move(s->result));
  } else {
    waiting_.push_back(std::move(s));
  }
  return id;
}

bool Engine::idle() const { return waiting_.empty() && active_.empty(); }

void Engine::admit() {
  const int max_ctx = runner_->options().max_context;
  while (!waiting_.empty() && !free_slots_.empty()) {
    Sequence& s = *waiting_.front();
    const int prompt = int(s.result.prompt.size());
    const int want = std::min<int64_t>(int64_t(prompt) + s.config.max_new_tokens, max_ctx);
    if (prompt >= max_ctx) {
      auto seq = std::move(waiting_.front());
      waiting_.pop_front();
      finish(*seq, FinishReason::Error, "prompt longer than the context limit");
      continue;
    }
    const int slot = free_slots_.back();
    // reserve the whole sequence up front: no preemption is ever needed
    if (!runner_->kv().ensure(slot, want)) {
      if (active_.empty()) {
        auto seq = std::move(waiting_.front());
        waiting_.pop_front();
        finish(*seq, FinishReason::Error, "request needs more KV cache than the engine has");
        continue;
      }
      break;  // wait for running sequences to free pages
    }
    free_slots_.pop_back();
    s.slot = slot;
    runner_->reset_slot(slot);
    active_.push_back(std::move(waiting_.front()));
    waiting_.pop_front();
  }
}

void Engine::finish(Sequence& s, FinishReason reason, const std::string& error) {
  s.result.finish = reason;
  s.result.error = error;
  s.result.done_s = now();
  if (s.slot >= 0) {
    runner_->kv().release(s.slot);
    free_slots_.push_back(s.slot);
    s.slot = -1;
  }
  ++stats_.requests_done;
  finished_.push_back(std::move(s.result));
}

bool Engine::step() {
  admit();
  if (active_.empty()) return !waiting_.empty();

  auto on_sample = [&](Sequence& s, int32_t tok, float logprob) -> bool {
    s.result.tokens.push_back(tok);
    s.result.logprobs.push_back(logprob);
    if (s.result.first_token_s == 0) s.result.first_token_s = now();
    ++s.samples;
    s.pending = tok;
    const bool stop = std::find(s.config.stop_tokens.begin(), s.config.stop_tokens.end(), tok) !=
                      s.config.stop_tokens.end();
    // s.position is where `tok` would be written by the next decode step
    const bool length = int(s.result.tokens.size()) >= s.config.max_new_tokens ||
                        s.position >= runner_->options().max_context;
    if (s.on_token) s.on_token(s.result.id, tok, stop || length);
    if (stop || length) {
      finish(s, stop ? FinishReason::Stop : FinishReason::Length);
      return true;
    }
    return false;
  };

  // 1) at most one prefill chunk
  for (auto& sp : active_) {
    Sequence& s = *sp;
    if (s.decoding()) continue;
    const int prompt = int(s.result.prompt.size());
    const int n = std::min(runner_->options().prefill_chunk, prompt - s.prefilled);
    const bool last = s.prefilled + n == prompt;
    const auto t = now();
    const StepResult r = runner_->prefill(
        s.slot, s.prefilled, std::span<const int32_t>(s.result.prompt).subspan(size_t(s.prefilled), size_t(n)),
        last ? &s.config.sampling : nullptr, s.samples);
    stats_.prefill_s += now() - t;
    stats_.prefill_tokens += n;
    s.prefilled += n;
    s.position = s.prefilled;
    if (last) on_sample(s, r.tokens[0], r.logprobs[0]);
    break;
  }
  active_.erase(std::remove_if(active_.begin(), active_.end(), [](const auto& p) { return p->slot < 0; }),
                active_.end());

  // 2) one decode step over every generating sequence (up to the batch capacity)
  std::vector<DecodeRow> rows;
  std::vector<Sequence*> seqs;
  for (auto& sp : active_) {
    Sequence& s = *sp;
    if (!s.decoding() || s.pending < 0 || int(rows.size()) == runner_->num_slots()) continue;
    DecodeRow r;
    r.slot = s.slot;
    r.position = s.position;  // = prompt + generated - 1: the pending token goes here
    r.token = s.pending;
    r.sampling = s.config.sampling;
    r.step = s.samples;
    rows.push_back(r);
    seqs.push_back(&s);
  }
  if (!rows.empty()) {
    const auto t = now();
    const StepResult r = runner_->decode(rows);
    stats_.decode_s += now() - t;
    stats_.decode_tokens += int64_t(rows.size());
    ++stats_.decode_steps;
    for (size_t i = 0; i < seqs.size(); ++i) {
      Sequence& s = *seqs[i];
      s.position += 1;  // the pending token is now in the cache
      on_sample(s, r.tokens[i], r.logprobs[i]);
    }
    active_.erase(std::remove_if(active_.begin(), active_.end(), [](const auto& p) { return p->slot < 0; }),
                  active_.end());
  }
  return true;
}

void Engine::run() {
  while (!idle()) step();
}

std::vector<RequestResult> Engine::take_finished() {
  std::vector<RequestResult> out;
  out.swap(finished_);
  return out;
}

}  // namespace qwen38
