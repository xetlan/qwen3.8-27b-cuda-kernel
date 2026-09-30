// End-to-end GPU test on the tiny model: ModelRunner (all kernels, chunked prefill, CUDA-graph
// decode, continuous batching bookkeeping) against the CPU reference model in engine mode.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "qwen38/common.h"
#include "qwen38/engine.h"
#include "qwen38/model_file.h"
#include "qwen38/model_runner.h"
#include "ref/ref_model.h"
#include "test_util.h"

using namespace qwen38;

namespace {

std::string tiny_model() {
  const char* env = std::getenv("QWEN38_TINY_MODEL");
  return env ? env : QWEN38_TESTDATA_DIR "/tiny/tiny-default.q38";
}

double kl(const float* p_logits, const float* q_logits, int n) {  // KL(softmax(p) || softmax(q))
  const double mp = *std::max_element(p_logits, p_logits + n), mq = *std::max_element(q_logits, q_logits + n);
  double zp = 0, zq = 0;
  for (int i = 0; i < n; ++i) {
    zp += std::exp(double(p_logits[i]) - mp);
    zq += std::exp(double(q_logits[i]) - mq);
  }
  double k = 0;
  for (int i = 0; i < n; ++i) {
    const double lp = double(p_logits[i]) - mp - std::log(zp), lq = double(q_logits[i]) - mq - std::log(zq);
    k += std::exp(lp) * (lp - lq);
  }
  return k;
}

RunnerOptions tiny_options(kernels::KvDtype kv) {
  RunnerOptions o;
  o.max_batch = 4;
  o.prefill_chunk = 128;
  o.kv_dtype = kv;
  o.kv_tokens = 8192;
  o.max_context = 4096;
  o.embedding = EmbeddingPlacement::Device;
  o.verbose = false;
  return o;
}

}  // namespace

QWEN38_TEST(e2e_prefill_logits_match_reference) {
  auto file = ModelFile::open(tiny_model());
  for (auto kv : {kernels::KvDtype::Int8, kernels::KvDtype::BF16}) {
    ModelRunner runner(*file, tiny_options(kv));
    ref::RefOptions ro;
    ro.kv_dtype = kv == kernels::KvDtype::Int8 ? ref::KvDtype::Int8 : ref::KvDtype::BF16;
    ref::RefModel ref(*file, ro);
    std::mt19937 rng(21);
    std::uniform_int_distribution<int> tok(0, 999);
    std::vector<int32_t> tokens(120);
    for (auto& t : tokens) t = tok(rng);
    runner.reset_slot(0);
    const auto got = runner.prefill_logits(0, 0, tokens);
    auto seq = ref.new_sequence();
    const auto want = ref.forward(seq, tokens, true);
    const int V = file->config().vocab_size;
    double kl_max = 0, kl_sum = 0;
    int top1 = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
      const float* g = got.data() + i * V;
      const float* w = want.data() + i * V;
      const double k = kl(w, g, V);
      kl_max = std::max(kl_max, k);
      kl_sum += k;
      top1 += int(std::max_element(g, g + V) - g) == int(std::max_element(w, w + V) - w);
    }
    std::printf("  kv=%s: KL mean %.2e max %.2e, top-1 agreement %d/%zu\n", kv == kernels::KvDtype::Int8 ? "int8" : "bf16",
                kl_sum / tokens.size(), kl_max, top1, tokens.size());
    CHECK(kl_sum / tokens.size() < 2e-3);
    CHECK(top1 >= int(tokens.size() * 0.95));
    runner.kv().release(0);
  }
}

QWEN38_TEST(e2e_chunked_prefill_and_decode_are_consistent) {
  // The same tokens through (a) one prefill chunk and (b) several chunks + graph decode steps
  // must give the same next-token predictions (greedy) up to accumulated rounding noise.
  auto file = ModelFile::open(tiny_model());
  ModelRunner runner(*file, tiny_options(kernels::KvDtype::Int8));
  std::mt19937 rng(8);
  std::uniform_int_distribution<int> tok(0, 999);
  std::vector<int32_t> tokens(100);
  for (auto& t : tokens) t = tok(rng);
  const int V = file->config().vocab_size;

  runner.reset_slot(0);
  const auto full = runner.prefill_logits(0, 0, tokens);
  runner.kv().release(0);

  // slot 1: prefill 60 tokens (chunks of 37 + 23), then feed the remaining 40 one by one
  runner.reset_slot(1);
  QWEN38_CHECK(runner.kv().ensure(1, 101), "pages");
  SamplingParams greedy;
  greedy.temperature = 0.0f;
  runner.prefill(1, 0, std::span<const int32_t>(tokens).subspan(0, 37), nullptr, 0);
  auto r = runner.prefill(1, 37, std::span<const int32_t>(tokens).subspan(37, 23), &greedy, 0);
  int agree = 0, total = 0;
  auto argmax_full = [&](int pos) {
    const float* row = full.data() + size_t(pos) * V;
    return int(std::max_element(row, row + V) - row);
  };
  agree += r.tokens[0] == argmax_full(59);
  ++total;
  for (int pos = 60; pos < 100; ++pos) {
    DecodeRow row;
    row.slot = 1;
    row.position = pos;
    row.token = tokens[size_t(pos)];
    row.sampling = greedy;
    row.step = pos;
    const auto d = runner.decode(std::span<const DecodeRow>(&row, 1));
    agree += d.tokens[0] == argmax_full(pos);
    ++total;
  }
  std::printf("  chunked prefill + decode vs single prefill: %d/%d greedy tokens agree\n", agree, total);
  CHECK(agree >= total - 2);
}

QWEN38_TEST(e2e_engine_batches_independent_sequences) {
  // Four requests decoded together must produce what each produces alone (greedy).
  auto file = ModelFile::open(tiny_model());
  EngineOptions eo;
  eo.runner = tiny_options(kernels::KvDtype::Int8);
  std::mt19937 rng(4);
  std::uniform_int_distribution<int> tok(0, 999);
  std::vector<std::vector<int32_t>> prompts(4);
  for (size_t i = 0; i < prompts.size(); ++i) {
    prompts[i].resize(20 + 30 * i);
    for (auto& t : prompts[i]) t = tok(rng);
  }
  GenerationConfig gen;
  gen.sampling.temperature = 0.0f;
  gen.max_new_tokens = 24;
  gen.ignore_eos = true;

  std::vector<std::vector<int32_t>> alone(prompts.size());
  {
    Engine engine(*file, eo);
    for (size_t i = 0; i < prompts.size(); ++i) {
      engine.submit(prompts[i], gen);
      engine.run();
      alone[i] = engine.take_finished().at(0).tokens;
    }
  }
  Engine engine(*file, eo);
  for (const auto& p : prompts) engine.submit(p, gen);
  engine.run();
  auto done = engine.take_finished();
  std::sort(done.begin(), done.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
  CHECK(done.size() == prompts.size());
  int same = 0, total = 0;
  for (size_t i = 0; i < done.size(); ++i) {
    CHECK(done[i].tokens.size() == size_t(gen.max_new_tokens));
    for (size_t t = 0; t < std::min(done[i].tokens.size(), alone[i].size()); ++t) {
      same += done[i].tokens[t] == alone[i][t];
      ++total;
      if (done[i].tokens[t] != alone[i][t]) break;  // after a divergence the rest may differ legitimately
    }
  }
  std::printf("  batched vs alone: %d/%d tokens identical before first divergence; avg batch %.2f\n", same, total,
              engine.stats().avg_batch());
  CHECK(engine.stats().avg_batch() > 1.5);
  // batch composition only changes reduction order inside GEMM tiles (stream-K splits per bucket),
  // so rare near-tie flips are possible; require long identical prefixes
  CHECK(same >= int(0.9 * double(prompts.size() * gen.max_new_tokens)));
}
