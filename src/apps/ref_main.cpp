// qwen38-ref: run the CPU reference model on a token file and dump logits.
//
//   qwen38-ref --model tiny.q38 --tokens tokens.i32 --out logits.f32 [--exact] [--kv f32|bf16|int8]
//            [--all-logits]
//   qwen38-ref --model tiny-vocab.q38 --prompt "Hello" [--max-new 8]   (chat template + tokenizer
//            from the model file, greedy generation, detokenized output: a host-only smoke test)
//
// tokens.i32: little-endian int32 token ids. logits.f32: float32 [n_out][vocab].
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "qwen38/chat.h"
#include "qwen38/common.h"
#include "qwen38/model_file.h"
#include "qwen38/tokenizer.h"
#include "ref/ref_model.h"

namespace {

std::vector<int32_t> read_tokens(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  QWEN38_CHECK(in.good(), "cannot open ", path);
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  QWEN38_CHECK(bytes.size() % 4 == 0, "token file size must be a multiple of 4");
  std::vector<int32_t> tokens(bytes.size() / 4);
  std::memcpy(tokens.data(), bytes.data(), bytes.size());
  return tokens;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, tokens_path, out_path, kv = "int8", prompt;
  int max_new = 8;
  bool exact = false, all_logits = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw qwen38::Error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--model") model = next();
    else if (a == "--tokens") tokens_path = next();
    else if (a == "--out") out_path = next();
    else if (a == "--kv") kv = next();
    else if (a == "--exact") exact = true;
    else if (a == "--all-logits") all_logits = true;
    else if (a == "--prompt") prompt = next();
    else if (a == "--max-new") max_new = std::stoi(next());
    else {
      std::cerr << "unknown argument " << a << "\n";
      return 2;
    }
  }
  if (!model.empty() && !prompt.empty()) {
    try {
      auto file = qwen38::ModelFile::open(model);
      const auto blob = file->resource("tokenizer");
      QWEN38_CHECK(!blob.empty(), "model file has no tokenizer");
      const auto tok = qwen38::Tokenizer::from_blob(blob.data(), blob.size());
      const std::string text = qwen38::render_chat({{"user", prompt, ""}}, qwen38::ChatOptions{});
      auto ids = tok.encode(text);
      std::cerr << "prompt: " << ids.size() << " tokens\n";
      qwen38::ref::RefOptions opt;
      opt.max_positions = int(ids.size()) + max_new + 1;
      qwen38::ref::RefModel ref(*file, opt);
      auto seq = ref.new_sequence();
      auto logits = ref.forward(seq, ids, false);
      qwen38::Utf8StreamDecoder dec;
      const int V = file->config().vocab_size;
      for (int i = 0; i < max_new; ++i) {
        const int next = int(std::max_element(logits.begin(), logits.begin() + V) - logits.begin());
        std::cout << "[" << next << "]" << dec.push(tok.token_bytes(next)) << std::flush;
        const int32_t t = next;
        logits = ref.forward(seq, std::span<const int32_t>(&t, 1), false);
      }
      std::cout << dec.flush() << "\n";
      return 0;
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return 1;
    }
  }
  if (model.empty() || tokens_path.empty() || out_path.empty()) {
    std::cerr << "usage: qwen38-ref --model M.q38 --tokens T.i32 --out L.f32 [--exact] [--kv f32|bf16|int8] "
                 "[--all-logits]\n";
    return 2;
  }
  try {
    auto file = qwen38::ModelFile::open(model);
    qwen38::ref::RefOptions opt;
    opt.exact = exact;
    opt.kv_dtype = kv == "f32" ? qwen38::ref::KvDtype::F32 : kv == "bf16" ? qwen38::ref::KvDtype::BF16
                                                                         : qwen38::ref::KvDtype::Int8;
    auto tokens = read_tokens(tokens_path);
    opt.max_positions = int(tokens.size()) + 1;
    qwen38::ref::RefModel ref(*file, opt);
    auto seq = ref.new_sequence();
    auto logits = ref.forward(seq, tokens, all_logits);
    std::ofstream out(out_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * sizeof(float)));
    std::cerr << "qwen38-ref: " << tokens.size() << " tokens, " << file->config().summary() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
