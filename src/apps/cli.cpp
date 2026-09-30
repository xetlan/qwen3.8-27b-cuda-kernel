// qwen38: command-line inference for Qwen3.8-27B on one RTX 3090.
//
//   qwen38 --model qwen3.8-27b.q38 --prompt "Explain paged attention."
//   qwen38 --model qwen3.8-27b.q38 --chat
//   qwen38 --model qwen3.8-27b.q38 --batch prompts.jsonl --out completions.jsonl
//
// Answer text goes to stdout; reasoning (thinking) and statistics go to stderr.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "qwen38/chat.h"
#include "qwen38/common.h"
#include "qwen38/engine.h"
#include "qwen38/model_file.h"
#include "qwen38/tokenizer.h"
#include "qwen38/tuning.h"

namespace {

using namespace qwen38;

struct CliOptions {
  std::string model, prompt, system, batch_in, batch_out, tuning, tokenizer, score_path;
  int score_ctx = 4096;
  bool chat = false, raw = false, show_think = true, stats = false;
  ChatOptions chat_opt;
  GenerationConfig gen;
  EngineOptions engine;
};

void usage() {
  std::cerr <<
      R"(usage: qwen38 --model FILE.q38 (--prompt TEXT | --chat | --batch IN.jsonl --out OUT.jsonl) [options]

generation
  --system TEXT            system message
  --no-think               disable thinking (template's enable_thinking=false)
  --effort xhigh|medium|low  reasoning effort (default xhigh, as the chat template)
  --raw                    feed --prompt verbatim, no chat template
  --max-new N              max generated tokens (default 4096)
  --temperature F          (default 1.0)   --top-k N (20)   --top-p F (0.95)   --min-p F (0)
  --presence-penalty F (0) --repetition-penalty F (1)  --seed N   --greedy
  --hide-think             do not print the reasoning stream
  --stats                  print timing and memory statistics
  --score FILE             perplexity of a text file (teacher forcing, windows of --score-ctx tokens)
  --score-ctx N            scoring window (default 4096)

engine
  --max-batch N            concurrent sequences (default 8; 154 MB of GDN state each)
  --prefill-chunk N        tokens per prefill step (default 1024)
  --kv int8|bf16           KV cache precision (default int8)
  --kv-tokens N            KV capacity in tokens (default: all free VRAM)
  --max-context N          per-sequence context limit (default 262144)
  --embed host|device      embedding table placement (default host: zero-copy, saves 2.5 GB VRAM)
  --reserve-mib N          VRAM left free (default 640)
  --no-graphs              run decode eagerly (debugging)
  --tuning FILE            GEMM tuning table from qwen38-bench --tune
  --tokenizer FILE         tokenizer blob (default: embedded in the model file)
)";
}

CliOptions parse(int argc, char** argv) {
  CliOptions o;
  o.gen.sampling = SamplingParams{};  // generation_config.json: T=1.0, top_k=20, top_p=0.95
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw Error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--model") o.model = next();
    else if (a == "--prompt") o.prompt = next();
    else if (a == "--chat") o.chat = true;
    else if (a == "--batch") o.batch_in = next();
    else if (a == "--out") o.batch_out = next();
    else if (a == "--system") o.system = next();
    else if (a == "--no-think") o.chat_opt.enable_thinking = false;
    else if (a == "--effort") o.chat_opt.reasoning_effort = next();
    else if (a == "--raw") o.raw = true;
    else if (a == "--max-new") o.gen.max_new_tokens = std::stoi(next());
    else if (a == "--temperature") o.gen.sampling.temperature = std::stof(next());
    else if (a == "--top-k") o.gen.sampling.top_k = std::stoi(next());
    else if (a == "--top-p") o.gen.sampling.top_p = std::stof(next());
    else if (a == "--min-p") o.gen.sampling.min_p = std::stof(next());
    else if (a == "--presence-penalty") o.gen.sampling.presence_penalty = std::stof(next());
    else if (a == "--repetition-penalty") o.gen.sampling.repetition_penalty = std::stof(next());
    else if (a == "--seed") o.gen.sampling.seed = std::stoull(next());
    else if (a == "--greedy") o.gen.sampling.temperature = 0.0f;
    else if (a == "--hide-think") o.show_think = false;
    else if (a == "--stats") o.stats = true;
    else if (a == "--score") o.score_path = next();
    else if (a == "--score-ctx") o.score_ctx = std::stoi(next());
    else if (a == "--max-batch") o.engine.runner.max_batch = std::stoi(next());
    else if (a == "--prefill-chunk") o.engine.runner.prefill_chunk = std::stoi(next());
    else if (a == "--kv") {
      const std::string v = next();
      if (v == "int8") o.engine.runner.kv_dtype = kernels::KvDtype::Int8;
      else if (v == "bf16") o.engine.runner.kv_dtype = kernels::KvDtype::BF16;
      else throw Error("--kv must be int8 or bf16");
    } else if (a == "--kv-tokens") o.engine.runner.kv_tokens = std::stoll(next());
    else if (a == "--max-context") o.engine.runner.max_context = std::stoi(next());
    else if (a == "--embed") {
      const std::string v = next();
      o.engine.runner.embedding = v == "device" ? EmbeddingPlacement::Device : EmbeddingPlacement::Host;
    } else if (a == "--reserve-mib") o.engine.runner.reserve_mib = std::stoul(next());
    else if (a == "--no-graphs") o.engine.runner.cuda_graphs = false;
    else if (a == "--tuning") o.tuning = next();
    else if (a == "--tokenizer") o.tokenizer = next();
    else if (a == "-h" || a == "--help") {
      usage();
      std::exit(0);
    } else {
      throw Error("unknown argument " + a);
    }
  }
  if (o.model.empty() || (o.prompt.empty() && !o.chat && o.batch_in.empty() && o.score_path.empty())) {
    usage();
    std::exit(2);
  }
  if (!o.batch_in.empty() && o.batch_out.empty()) throw Error("--batch needs --out");
  return o;
}

Tokenizer load_tokenizer(const ModelFile& file, const std::string& override_path) {
  if (!override_path.empty()) return Tokenizer::from_file(override_path);
  const auto blob = file.resource("tokenizer");
  QWEN38_CHECK(!blob.empty(), "model file has no tokenizer; pass --tokenizer");
  return Tokenizer::from_blob(blob.data(), blob.size());
}

// Streams one request's output, routing reasoning and answer to different sinks.
class Printer {
 public:
  Printer(const Tokenizer& tok, bool thinking, bool show_think)
      : tok_(tok), in_think_(thinking), show_think_(show_think) {}
  void token(int32_t id) {
    if (id == kThinkCloseId) {
      flush_think();
      in_think_ = false;
      return;
    }
    if (id == kThinkOpenId) {
      in_think_ = true;
      return;
    }
    if (tok_.is_special(id)) return;  // <|im_end|> etc.
    std::string text = (in_think_ ? think_ : answer_).push(tok_.token_bytes(id));
    if (in_think_) {
      reasoning += text;
      if (show_think_) std::cerr << "\x1b[2m" << text << "\x1b[0m" << std::flush;
    } else {
      answer += text;
      std::cout << text << std::flush;
    }
  }
  void finish() {
    flush_think();
    const std::string rest = answer_.flush();
    answer += rest;
    std::cout << rest << "\n" << std::flush;
  }
  std::string reasoning, answer;

 private:
  void flush_think() {
    const std::string rest = think_.flush();
    reasoning += rest;
    if (show_think_) std::cerr << "\x1b[2m" << rest << "\x1b[0m\n" << std::flush;
  }
  const Tokenizer& tok_;
  bool in_think_;
  bool show_think_;
  Utf8StreamDecoder think_, answer_;
};

void print_stats(const Engine& engine, const RequestResult& r) {
  const auto& st = engine.stats();
  std::fprintf(stderr,
               "[qwen38] prompt %zu tok, generated %zu tok (%s) | prefill %.1f tok/s | decode %.1f tok/s | TTFT %.2f s\n",
               r.prompt.size(), r.tokens.size(), finish_reason_name(r.finish), st.tokens_per_s_prefill(),
               st.tokens_per_s_decode(), r.first_token_s - r.submit_s);
}

int run_single(Engine& engine, const Tokenizer& tok, const CliOptions& o) {
  std::vector<int32_t> prompt;
  bool thinking = false;
  if (o.raw) {
    prompt = tok.encode(o.prompt);
  } else {
    std::vector<ChatMessage> msgs;
    if (!o.system.empty()) msgs.push_back({"system", o.system, ""});
    msgs.push_back({"user", o.prompt, ""});
    prompt = tok.encode(render_chat(msgs, o.chat_opt));
    thinking = o.chat_opt.enable_thinking;
  }
  Printer printer(tok, thinking, o.show_think);
  engine.submit(prompt, o.gen, [&](int, int32_t t, bool) { printer.token(t); });
  engine.run();
  printer.finish();
  auto done = engine.take_finished();
  if (o.stats && !done.empty()) print_stats(engine, done[0]);
  return done.empty() || done[0].finish == FinishReason::Error ? 1 : 0;
}

int run_chat(Engine& engine, const Tokenizer& tok, const CliOptions& o) {
  std::vector<ChatMessage> history;
  if (!o.system.empty()) history.push_back({"system", o.system, ""});
  std::string line;
  while (true) {
    std::cerr << "\n> " << std::flush;
    if (!std::getline(std::cin, line)) break;
    if (line == "/exit" || line == "/quit") break;
    if (line == "/reset") {
      history.resize(o.system.empty() ? 0 : 1);
      continue;
    }
    if (line.empty()) continue;
    history.push_back({"user", line, ""});
    const auto prompt = tok.encode(render_chat(history, o.chat_opt));
    Printer printer(tok, o.chat_opt.enable_thinking, o.show_think);
    engine.submit(prompt, o.gen, [&](int, int32_t t, bool) { printer.token(t); });
    engine.run();
    printer.finish();
    auto done = engine.take_finished();
    if (o.stats && !done.empty()) print_stats(engine, done[0]);
    history.push_back({"assistant", printer.answer, printer.reasoning});
    engine.reset_stats();
  }
  return 0;
}

// JSONL in: {"id": .., "prompt": "..."} or {"messages": [{"role","content"}, ...]}
//           optional: max_tokens, temperature, top_p, top_k, min_p, presence_penalty, seed,
//                     enable_thinking, reasoning_effort, raw
// JSONL out: {"id", "text", "reasoning", "finish_reason", "prompt_tokens", "completion_tokens"}
int run_batch(Engine& engine, const Tokenizer& tok, const CliOptions& o) {
  std::ifstream in(o.batch_in);
  QWEN38_CHECK(in.good(), "cannot open ", o.batch_in);
  struct Item {
    nlohmann::json id;
    bool thinking;
  };
  std::vector<Item> items;
  std::string line;
  while (std::getline(in, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    const auto j = nlohmann::json::parse(line);
    GenerationConfig gen = o.gen;
    gen.max_new_tokens = j.value("max_tokens", gen.max_new_tokens);
    gen.sampling.temperature = j.value("temperature", gen.sampling.temperature);
    gen.sampling.top_p = j.value("top_p", gen.sampling.top_p);
    gen.sampling.top_k = j.value("top_k", gen.sampling.top_k);
    gen.sampling.min_p = j.value("min_p", gen.sampling.min_p);
    gen.sampling.presence_penalty = j.value("presence_penalty", gen.sampling.presence_penalty);
    gen.sampling.seed = j.value("seed", gen.sampling.seed);
    ChatOptions co = o.chat_opt;
    co.enable_thinking = j.value("enable_thinking", co.enable_thinking);
    co.reasoning_effort = j.value("reasoning_effort", co.reasoning_effort);
    std::string text;
    bool thinking = false;
    if (j.value("raw", o.raw)) {
      text = j.at("prompt").get<std::string>();
    } else {
      std::vector<ChatMessage> msgs;
      if (j.contains("messages")) {
        for (const auto& m : j["messages"]) msgs.push_back({m.at("role"), m.value("content", ""), m.value("reasoning", "")});
      } else {
        if (!o.system.empty()) msgs.push_back({"system", o.system, ""});
        msgs.push_back({"user", j.at("prompt").get<std::string>(), ""});
      }
      text = render_chat(msgs, co);
      thinking = co.enable_thinking;
    }
    items.push_back({j.value("id", nlohmann::json(int(items.size()))), thinking});
    engine.submit(tok.encode(text), gen);
  }
  std::fprintf(stderr, "[qwen38] %zu requests queued\n", items.size());
  std::ofstream out(o.batch_out);
  int done_count = 0;
  while (!engine.idle()) {
    engine.step();
    for (auto& r : engine.take_finished()) {
      // split reasoning / answer at </think>
      std::vector<int32_t> think, answer;
      bool in_think = items[r.id].thinking;
      for (int32_t t : r.tokens) {
        if (t == kThinkCloseId) {
          in_think = false;
          continue;
        }
        if (tok.is_special(t) || t == kThinkOpenId) continue;
        (in_think ? think : answer).push_back(t);
      }
      nlohmann::json j;
      j["id"] = items[r.id].id;
      j["text"] = tok.decode(answer);
      j["reasoning"] = tok.decode(think);
      j["finish_reason"] = finish_reason_name(r.finish);
      if (!r.error.empty()) j["error"] = r.error;
      j["prompt_tokens"] = r.prompt.size();
      j["completion_tokens"] = r.tokens.size();
      out << j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << "\n" << std::flush;
      ++done_count;
      if (o.stats)
        std::fprintf(stderr, "[qwen38] %d/%zu done, decode %.1f tok/s aggregate (avg batch %.1f)\n", done_count,
                     items.size(), engine.stats().tokens_per_s_decode(), engine.stats().avg_batch());
    }
  }
  return 0;
}

// Perplexity over non-overlapping windows; prints PPL and mean log-prob (nats/token).
int run_score(Engine& engine, const Tokenizer& tok, const CliOptions& o) {
  std::ifstream in(o.score_path, std::ios::binary);
  QWEN38_CHECK(in.good(), "cannot open ", o.score_path);
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto ids = tok.encode(text, /*allow_added_tokens=*/false);
  ModelRunner& runner = engine.runner();
  double nll = 0;
  int64_t count = 0;
  for (size_t w0 = 0; w0 + 1 < ids.size(); w0 += size_t(o.score_ctx)) {
    const size_t n = std::min(size_t(o.score_ctx), ids.size() - 1 - w0);
    std::vector<int32_t> inputs(ids.begin() + ptrdiff_t(w0), ids.begin() + ptrdiff_t(w0 + n));
    std::vector<int32_t> targets(ids.begin() + ptrdiff_t(w0 + 1), ids.begin() + ptrdiff_t(w0 + n + 1));
    runner.reset_slot(0);
    const auto lp = runner.score(0, 0, inputs, targets);
    runner.kv().release(0);
    for (float x : lp) nll -= x;
    count += int64_t(lp.size());
    std::fprintf(stderr, "[qwen38] scored %lld / %zu tokens, running PPL %.4f\n", (long long)count, ids.size() - 1,
                 std::exp(nll / double(count)));
  }
  std::printf("tokens %lld  mean_nll %.6f  ppl %.6f\n", (long long)count, nll / double(count),
              std::exp(nll / double(count)));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions o = parse(argc, argv);
    auto file = ModelFile::open(o.model);
    std::fprintf(stderr, "[qwen38] %s: %s, recipe %s\n", o.model.c_str(), file->config().summary().c_str(),
                 file->recipe_name().c_str());
    const Tokenizer tok = load_tokenizer(*file, o.tokenizer);
    if (!o.tuning.empty()) load_tuning(o.tuning);
    Engine engine(*file, o.engine);
    if (!o.score_path.empty()) return run_score(engine, tok, o);
    if (!o.batch_in.empty()) return run_batch(engine, tok, o);
    if (o.chat) return run_chat(engine, tok, o);
    return run_single(engine, tok, o);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "qwen38: error: %s\n", e.what());
    return 1;
  }
}
