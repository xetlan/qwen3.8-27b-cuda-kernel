// Device-resident model weights, uploaded once from a .q38 file into a single arena.
#pragma once

#include <cuda_bf16.h>

#include <vector>

#include "qwen38/device.h"
#include "qwen38/model_file.h"
#include "kernels/kernels.h"

namespace qwen38 {

enum class EmbeddingPlacement {
  Host,    // pinned, mapped host memory read zero-copy by the embed kernel (saves 2.5 GB VRAM)
  Device,  // bf16 table in VRAM
};

struct LayerWeights {
  bool attention = false;
  kernels::BQWeight in_proj;   // attn.qkvg or gdn.qkvz
  kernels::BQWeight out_proj;  // attn.o or gdn.out
  kernels::BQWeight gate_up;
  kernels::BQWeight down;
  // attention
  const float* q_norm = nullptr;
  const float* k_norm = nullptr;
  // Gated DeltaNet
  const __nv_bfloat16* ab = nullptr;
  int ab_rows = 0;
  const float* conv = nullptr;
  const float* A = nullptr;
  const float* dt_bias = nullptr;
  const float* gdn_norm = nullptr;
};

class ModelWeights {
 public:
  ModelWeights(const ModelFile& file, EmbeddingPlacement placement);

  const std::vector<LayerWeights>& layers() const { return layers_; }
  const kernels::BQWeight& lm_head() const { return lm_head_; }
  const __nv_bfloat16* embedding() const { return embedding_; }  // device-accessible
  EmbeddingPlacement embedding_placement() const { return placement_; }
  size_t device_bytes() const { return arena_.used(); }
  size_t host_bytes() const { return embed_host_.bytes(); }

  // Bytes of device memory the weights will occupy for a given file/placement (for planning).
  static size_t required_device_bytes(const ModelFile& file, EmbeddingPlacement placement);

 private:
  kernels::BQWeight upload_bq(const ModelFile& file, const std::string& name);
  template <typename T>
  const T* upload_direct(const ModelFile& file, const std::string& name);

  DeviceArena arena_;
  PinnedBuffer embed_host_;
  EmbeddingPlacement placement_;
  std::vector<LayerWeights> layers_;
  kernels::BQWeight lm_head_;
  const __nv_bfloat16* embedding_ = nullptr;
};

}  // namespace qwen38
