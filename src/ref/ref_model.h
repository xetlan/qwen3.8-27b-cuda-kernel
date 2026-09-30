// CPU reference model: a straightforward, token-at-a-time implementation of the full
// Qwen3.5-architecture forward pass on top of ref/ops.h.
//
// Intended for small test models (all BQ weights are dequantized to fp32 at construction).
// Engine mode reproduces the CUDA engine's rounding points; exact mode keeps every activation
// in fp32 and is compared against Hugging Face transformers.
#pragma once

#include <memory>
#include <span>
#include <vector>

#include "qwen38/model_file.h"
#include "ref/ops.h"

namespace qwen38::ref {

struct RefOptions {
  bool exact = false;
  KvDtype kv_dtype = KvDtype::Int8;
  int max_positions = 4096;
};

class RefModel {
 public:
  RefModel(const ModelFile& file, RefOptions options);

  struct Sequence {
    int pos = 0;
    std::vector<KvStore> kv;     // per attention layer
    std::vector<GdnState> gdn;   // per GDN layer
  };

  Sequence new_sequence() const;

  // Appends `tokens` to the sequence. Returns logits [n_out][vocab], n_out = tokens.size() if
  // all_logits, else 1 (the last token).
  std::vector<float> forward(Sequence& seq, std::span<const int32_t> tokens, bool all_logits = false) const;

  const ModelConfig& config() const { return cfg_; }

 private:
  struct Layer {
    bool attention = false;
    std::vector<float> w_in;      // attn.qkvg or gdn.qkvz (dequantized, padded rows)
    int in_rows = 0;
    std::vector<float> w_ab;      // gdn.ab
    int ab_rows = 0;
    std::vector<float> w_out;     // attn.o or gdn.out
    std::vector<float> w_gate_up;
    std::vector<float> w_down;
    std::vector<float> q_norm, k_norm;
    std::vector<float> conv, A, dt_bias, gdn_norm;
    int mixer_width = 0;          // K of the out projection
  };

  RefOptions opt_;
  ModelConfig cfg_;
  std::vector<Layer> layers_;
  std::vector<float> lm_head_;
  std::span<const uint16_t> embed_;  // bf16 bits [vocab][hidden]
};

}  // namespace qwen38::ref
