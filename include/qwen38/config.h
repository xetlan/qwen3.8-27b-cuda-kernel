// Model configuration as stored in the .q38 header (docs/FORMAT.md 1.1).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qwen38 {

struct ModelConfig {
  int hidden_size = 0;
  int num_layers = 0;
  int vocab_size = 0;
  int intermediate_size = 0;
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int head_dim = 0;
  int rotary_dim = 0;
  double rope_theta = 0;
  int linear_num_key_heads = 0;
  int linear_num_value_heads = 0;
  int linear_key_head_dim = 0;
  int linear_value_head_dim = 0;
  int linear_conv_kernel_dim = 0;
  float rms_norm_eps = 1e-6f;
  int max_position_embeddings = 0;
  bool norm_folded = true;
  std::vector<uint8_t> is_attention;  // per layer: 1 = full attention, 0 = Gated DeltaNet
  std::vector<int32_t> eos_token_ids;

  int num_attention_layers() const;
  int num_gdn_layers() const;
  int attn_index(int layer) const;  // index among attention layers, -1 for GDN layers
  int gdn_index(int layer) const;   // index among GDN layers, -1 for attention layers

  int q_dim() const { return num_attention_heads * head_dim; }
  int kv_dim() const { return num_key_value_heads * head_dim; }
  int attn_group() const { return num_attention_heads / num_key_value_heads; }
  int qkvg_rows() const { return 2 * q_dim() + 2 * kv_dim(); }
  int gdn_key_dim() const { return linear_num_key_heads * linear_key_head_dim; }
  int gdn_value_dim() const { return linear_num_value_heads * linear_value_head_dim; }
  int gdn_conv_dim() const { return 2 * gdn_key_dim() + gdn_value_dim(); }
  int gdn_qkvz_rows() const { return gdn_conv_dim() + gdn_value_dim(); }
  int gdn_v_per_k() const { return linear_num_value_heads / linear_num_key_heads; }
  int norm_blocks() const { return hidden_size / 128; }  // residual sum-of-squares partials

  // Throws qwen38::Error if the configuration is outside what the kernels implement.
  void validate() const;
  std::string summary() const;
};

}  // namespace qwen38
