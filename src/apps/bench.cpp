// qwen38-bench: throughput, per-op profile and GEMM autotuning on the real model.
//
//   qwen38-bench --model M.q38                         decode (B = 1,2,4,8,16) + prefill sweep
//   qwen38-bench --model M.q38 --decode 1,8 --context 32768 --gen 128
//   qwen38-bench --model M.q38 --profile               per-op GPU time of decode and prefill steps
//   qwen38-bench --model M.q38 --tune tuning.json      sweep GEMM tile configs, write tuning table
//   qwen38-bench --model M.q38 --json results.json     machine-readable output
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "qwen38/common.h"
#include "qwen38/device.h"
#include "qwen38/model_file.h"
#include "qwen38/model_runner.h"
#include "qwen38/tuning.h"

namespace {

using namespace qwen38;
using Clock = std::chrono::steady_clock;

std::vector<int> parse_list(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) out.push_back(std::stoi(item));
  return out;
}

double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

std::vector<int32_t> random_tokens(int n, std::mt19937& rng, int vocab) {
  std::uniform_int_distribution<int> d(1000, std::min(vocab, 150000) - 1);  // ordinary text tokens
  std::vector<int32_t> t(static_cast<size_t>(n));
  for (auto& x : t) x = d(rng);
  return t;
}

// Prefill `tokens` into `slot` in chunks; returns the first sampled token.
int32_t prefill_all(ModelRunner& r, int slot, const std::vector<int32_t>& tokens, const SamplingParams& sp) {
  const int chunk = r.options().prefill_chunk;
  int32_t first = 0;
  for (size_t off = 0; off < tokens.size(); off += size_t(chunk)) {
    const size_t n = std::min(tokens.size() - off, size_t(chunk));
    const bool last = off + n == tokens.size();
    auto res = r.prefill(slot, int(off), std::span<const int32_t>(tokens).subspan(off, n), last ? &sp : nullptr, 0);
    if (last) first = res.tokens[0];
  }
  return first;
}

struct DecodeResult {
  int batch;
  int context;
  double tok_s;       // aggregate
  double ms_per_step;
};

DecodeResult bench_decode(ModelRunner& r, int batch, int context, int gen, std::mt19937& rng,
                          bool profile = false) {
  SamplingParams sp;
  sp.temperature = 0.0f;  // greedy keeps runs reproducible
  const int warm = 8;
  std::vector<DecodeRow> rows(static_cast<size_t>(batch));
  for (int b = 0; b < batch; ++b) {
    r.reset_slot(b);
    QWEN38_CHECK(r.kv().ensure(b, context + warm + gen + 1), "KV cache too small for decode benchmark (batch ", batch,
               ", context ", context, ")");
    const auto prompt = random_tokens(context, rng, r.config().vocab_size);
    rows[size_t(b)].slot = b;
    rows[size_t(b)].position = context;
    rows[size_t(b)].token = prefill_all(r, b, prompt, sp);
    rows[size_t(b)].sampling = sp;
  }
  auto step = [&]() {
    auto res = r.decode(rows);
    for (int b = 0; b < batch; ++b) {
      rows[size_t(b)].position += 1;
      rows[size_t(b)].token = res.tokens[size_t(b)];
      rows[size_t(b)].step += 1;
    }
  };
  for (int i = 0; i < warm; ++i) step();
  if (profile) {
    r.clear_profile();
    r.set_profiling(true);
  }
  const auto t0 = Clock::now();
  for (int i = 0; i < gen; ++i) step();
  const double s = seconds_since(t0);
  if (profile) r.set_profiling(false);
  for (int b = 0; b < batch; ++b) r.kv().release(b);
  return {batch, context, double(batch) * gen / s, 1e3 * s / gen};
}

double bench_prefill(ModelRunner& r, int length, std::mt19937& rng) {
  SamplingParams sp;
  sp.temperature = 0.0f;
  r.reset_slot(0);
  QWEN38_CHECK(r.kv().ensure(0, length + 1), "KV cache too small for a ", length, "-token prefill");
  const auto prompt = random_tokens(length, rng, r.config().vocab_size);
  const auto t0 = Clock::now();
  prefill_all(r, 0, prompt, sp);
  const double s = seconds_since(t0);
  r.kv().release(0);
  return double(length) / s;
}

void print_profile(const OpProfile& p, const char* title, int steps) {
  double total = 0;
  for (const auto& [k, v] : p.ms) total += v;
  std::vector<std::pair<std::string, double>> v(p.ms.begin(), p.ms.end());
  std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
  std::printf("\n%s (per step, %d steps averaged)\n", title, steps);
  std::printf("  %-26s %10s %8s %8s\n", "op", "ms/step", "share", "calls");
  for (const auto& [name, ms] : v)
    std::printf("  %-26s %10.3f %7.1f%% %8d\n", name.c_str(), ms / steps, 100.0 * ms / total,
                p.calls.at(name) / std::max(1, steps));
  std::printf("  %-26s %10.3f\n", "total", total / steps);
}

// ---- GEMM autotuning ---------------------------------------------------------------------------
std::vector<TuningEntry> tune(ModelRunner& r, int max_m) {
  const ModelWeights& W = r.weights();
  std::map<std::tuple<int, int, int>, const kernels::BQWeight*> shapes;
  auto add = [&](const kernels::BQWeight& w) { shapes.emplace(std::make_tuple(w.bits, w.n, w.k), &w); };
  for (const auto& L : W.layers()) {
    add(L.in_proj);
    add(L.out_proj);
    add(L.gate_up);
    add(L.down);
  }
  add(W.lm_head());
  int max_n = 0, max_k = 0;
  for (const auto& [key, w] : shapes) {
    max_n = std::max(max_n, w->n);
    max_k = std::max(max_k, w->k);
  }
  DeviceBuffer a(size_t(max_m) * max_k * 2);
  DeviceBuffer out(size_t(max_m) * max_n * 2);
  QWEN38_CUDA(cudaMemset(a.get(), 0x3c, a.bytes()));  // ~1.0 in bf16 halves: finite inputs
  cudaStream_t s = r.stream();
  Event e0(true), e1(true);
  std::vector<TuningEntry> entries;
  std::vector<int> ladder;
  for (int m : {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096})
    if (m <= max_m) ladder.push_back(m);
  for (const auto& [key, w] : shapes) {
    const auto [bits, n, k] = key;
    for (int m : ladder) {
      TuningEntry best;
      best.bits = bits;
      best.n = n;
      best.k = k;
      best.m_max = m;
      for (int c = 0; c < kernels::qgemm_num_configs(); ++c) {
        if (c == 4 && k % 128) continue;
        kernels::GemmEpilogue ep;
        ep.kind = kernels::Epilogue::StoreBF16;
        ep.out = out.get();
        ep.ldo = n;
        for (int i = 0; i < 3; ++i) kernels::qgemm(a.as<__nv_bfloat16>(), m, *w, ep, r.gemm_workspace(), s, c);
        const int reps = m <= 64 ? 20 : 5;
        e0.record(s);
        for (int i = 0; i < reps; ++i) kernels::qgemm(a.as<__nv_bfloat16>(), m, *w, ep, r.gemm_workspace(), s, c);
        e1.record(s);
        e1.sync();
        const double us = 1e3 * e1.ms_since(e0) / reps;
        if (best.config < 0 || us < best.us) {
          best.config = c;
          best.us = us;
        }
      }
      const double bytes = double(w->n) * w->k * w->bits / 8.0;
      std::printf("  bq%d n=%6d k=%6d m=%5d  -> %-20s %9.1f us  %7.1f GB/s  %6.1f TFLOP/s\n", bits, n, k, m,
                  kernels::qgemm_config_name(best.config), best.us, bytes / best.us / 1e3,
                  2.0 * m * n * k / best.us / 1e6);
      std::fflush(stdout);
      entries.push_back(best);
    }
  }
  return entries;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string model, json_path, tune_path, tuning_in;
    std::vector<int> decode_batches = {1, 2, 4, 8, 16}, prefill_lengths = {512, 2048, 8192};
    int context = 1024, gen = 64, tune_max_m = 0;
    bool profile = false, run_decode = true, run_prefill = true;
    RunnerOptions ro;
    ro.verbose = true;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw Error("missing value for " + a);
        return argv[++i];
      };
      if (a == "--model") model = next();
      else if (a == "--decode") decode_batches = parse_list(next());
      else if (a == "--prefill") prefill_lengths = parse_list(next());
      else if (a == "--no-decode") run_decode = false;
      else if (a == "--no-prefill") run_prefill = false;
      else if (a == "--context") context = std::stoi(next());
      else if (a == "--gen") gen = std::stoi(next());
      else if (a == "--profile") profile = true;
      else if (a == "--tune") tune_path = next();
      else if (a == "--tune-max-m") tune_max_m = std::stoi(next());
      else if (a == "--tuning") tuning_in = next();
      else if (a == "--json") json_path = next();
      else if (a == "--max-batch") ro.max_batch = std::stoi(next());
      else if (a == "--prefill-chunk") ro.prefill_chunk = std::stoi(next());
      else if (a == "--kv") ro.kv_dtype = next() == "bf16" ? kernels::KvDtype::BF16 : kernels::KvDtype::Int8;
      else if (a == "--kv-tokens") ro.kv_tokens = std::stoll(next());
      else if (a == "--max-context") ro.max_context = std::stoi(next());
      else if (a == "--embed") ro.embedding = next() == "device" ? EmbeddingPlacement::Device : EmbeddingPlacement::Host;
      else if (a == "--reserve-mib") ro.reserve_mib = std::stoul(next());
      else if (a == "--no-graphs") ro.cuda_graphs = false;
      else throw Error("unknown argument " + a);
    }
    QWEN38_CHECK(!model.empty(), "usage: qwen38-bench --model FILE.q38 [--decode 1,8] [--prefill 2048] [--context N] "
                               "[--gen N] [--profile] [--tune OUT.json] [--json OUT.json]");
    if (!decode_batches.empty()) ro.max_batch = std::max(ro.max_batch, *std::max_element(decode_batches.begin(), decode_batches.end()));
    auto file = ModelFile::open(model);
    if (!tuning_in.empty()) load_tuning(tuning_in);
    ModelRunner runner(*file, ro);
    const DeviceInfo& dev = runner.device();
    std::printf("device: %s, %d SMs, %.1f GiB | model: %s | recipe %s | weights %.2f GiB | KV %s %lld tokens\n",
                dev.name.c_str(), dev.num_sms, double(dev.total_mem) / (1 << 30),
                file->config().summary().c_str(), file->recipe_name().c_str(),
                double(runner.memory().weights) / (1 << 30),
                ro.kv_dtype == kernels::KvDtype::Int8 ? "int8" : "bf16", (long long)runner.memory().kv_tokens);
    nlohmann::json results;
    results["device"] = dev.name;
    results["model"] = model;
    results["recipe"] = file->recipe_name();
    results["weights_gib"] = double(runner.memory().weights) / (1 << 30);
    results["kv_tokens"] = runner.memory().kv_tokens;
    std::mt19937 rng(1234);

    if (!tune_path.empty()) {
      std::printf("\nGEMM tuning sweep\n");
      const auto entries = tune(runner, tune_max_m > 0 ? tune_max_m : ro.prefill_chunk);
      save_tuning(tune_path, entries, dev.name);
      install_tuning(entries);
      std::printf("wrote %s (%zu entries)\n", tune_path.c_str(), entries.size());
    }

    if (profile) {
      for (int b : {1, ro.max_batch}) {
        bench_decode(runner, b, context, 4, rng, /*profile=*/true);
        char title[96];
        std::snprintf(title, sizeof title, "decode profile, batch %d, context %d (eager, no graph)", b, context);
        print_profile(runner.profile(), title, 4);
      }
      runner.clear_profile();
      runner.set_profiling(true);
      bench_prefill(runner, ro.prefill_chunk, rng);
      runner.set_profiling(false);
      print_profile(runner.profile(), "prefill profile, one chunk", 1);
    }

    if (run_decode) {
      std::printf("\ndecode throughput (context %d, %d measured steps, CUDA graphs %s)\n", context, gen,
                  ro.cuda_graphs ? "on" : "off");
      std::printf("  %6s %14s %14s %12s\n", "batch", "tok/s (sum)", "tok/s (seq)", "ms/step");
      for (int b : decode_batches) {
        const auto d = bench_decode(runner, b, context, gen, rng);
        std::printf("  %6d %14.1f %14.1f %12.2f\n", b, d.tok_s, d.tok_s / b, d.ms_per_step);
        std::fflush(stdout);
        results["decode"].push_back({{"batch", b}, {"context", context}, {"tok_s", d.tok_s}, {"ms_per_step", d.ms_per_step}});
      }
    }
    if (run_prefill) {
      std::printf("\nprefill throughput (chunk %d)\n", ro.prefill_chunk);
      for (int len : prefill_lengths) {
        const double tps = bench_prefill(runner, len, rng);
        std::printf("  %8d tokens: %10.1f tok/s\n", len, tps);
        std::fflush(stdout);
        results["prefill"].push_back({{"tokens", len}, {"tok_s", tps}});
      }
    }
    if (!json_path.empty()) {
      std::ofstream(json_path) << results.dump(1) << "\n";
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "qwen38-bench: error: %s\n", e.what());
    return 1;
  }
  return 0;
}
