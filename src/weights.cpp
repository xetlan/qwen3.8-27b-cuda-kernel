#include "qwen38/weights.h"

#include <cstring>

#include "qwen38/common.h"

namespace qwen38 {

namespace {

constexpr size_t kAlign = 256;

size_t padded(size_t bytes) { return size_t(align_up(int64_t(bytes), int64_t(kAlign))); }

}  // namespace

size_t ModelWeights::required_device_bytes(const ModelFile& file, EmbeddingPlacement placement) {
  size_t total = 0;
  for (const auto& [name, t] : file.tensors()) {
    if (name == "embed" && placement == EmbeddingPlacement::Host) continue;
    if (is_bq(t.dtype)) {
      total += padded(size_t(t.lo_bytes)) + padded(size_t(t.hi_bytes)) + padded(size_t(t.sc_bytes));
    } else {
      total += padded(size_t(t.bytes));
    }
  }
  return total + kAlign;
}

kernels::BQWeight ModelWeights::upload_bq(const ModelFile& file, const std::string& name) {
  const TensorInfo& t = file.tensor(name);
  QWEN38_CHECK(is_bq(t.dtype), name, " must be a BQ tensor");
  kernels::BQWeight w;
  w.bits = bq_bits(t.dtype);
  w.n = int(t.shape[0]);
  w.k = int(t.shape[1]);
  w.rows = int(t.rows);
  auto copy = [&](int64_t off, int64_t bytes) -> void* {
    if (bytes == 0) return nullptr;
    void* dst = arena_.allocate(size_t(bytes), kAlign);
    QWEN38_CUDA(cudaMemcpy(dst, file.data(off, bytes).data(), size_t(bytes), cudaMemcpyHostToDevice));
    return dst;
  };
  w.lo = static_cast<const uint8_t*>(copy(t.lo_offset, t.lo_bytes));
  w.hi = static_cast<const uint8_t*>(copy(t.hi_offset, t.hi_bytes));
  w.scales = static_cast<const __nv_bfloat16*>(copy(t.sc_offset, t.sc_bytes));
  return w;
}

template <typename T>
const T* ModelWeights::upload_direct(const ModelFile& file, const std::string& name) {
  const TensorInfo& t = file.tensor(name);
  void* dst = arena_.allocate(size_t(t.bytes), kAlign);
  QWEN38_CUDA(cudaMemcpy(dst, file.data(t.offset, t.bytes).data(), size_t(t.bytes), cudaMemcpyHostToDevice));
  return static_cast<const T*>(dst);
}

ModelWeights::ModelWeights(const ModelFile& file, EmbeddingPlacement placement)
    : arena_(required_device_bytes(file, placement)), placement_(placement) {
  const ModelConfig& cfg = file.config();
  const TensorInfo& emb = file.tensor("embed");
  QWEN38_CHECK(emb.dtype == DType::BF16, "embedding must be bf16");
  if (placement == EmbeddingPlacement::Host) {
    embed_host_ = PinnedBuffer(size_t(emb.bytes), /*mapped=*/true);
    std::memcpy(embed_host_.get(), file.data(emb.offset, emb.bytes).data(), size_t(emb.bytes));
    embedding_ = static_cast<const __nv_bfloat16*>(embed_host_.device_ptr());
  } else {
    embedding_ = upload_direct<__nv_bfloat16>(file, "embed");
  }

  layers_.resize(cfg.num_layers);
  for (int i = 0; i < cfg.num_layers; ++i) {
    LayerWeights& L = layers_[i];
    const std::string p = std::to_string(i) + ".";
    L.attention = cfg.is_attention[i] != 0;
    if (L.attention) {
      L.in_proj = upload_bq(file, p + "attn.qkvg");
      L.out_proj = upload_bq(file, p + "attn.o");
      L.q_norm = upload_direct<float>(file, p + "attn.q_norm");
      L.k_norm = upload_direct<float>(file, p + "attn.k_norm");
    } else {
      L.in_proj = upload_bq(file, p + "gdn.qkvz");
      L.out_proj = upload_bq(file, p + "gdn.out");
      L.ab = upload_direct<__nv_bfloat16>(file, p + "gdn.ab");
      L.ab_rows = int(file.tensor(p + "gdn.ab").shape[0]);
      L.conv = upload_direct<float>(file, p + "gdn.conv");
      L.A = upload_direct<float>(file, p + "gdn.A");
      L.dt_bias = upload_direct<float>(file, p + "gdn.dt_bias");
      L.gdn_norm = upload_direct<float>(file, p + "gdn.norm");
    }
    L.gate_up = upload_bq(file, p + "mlp.gate_up");
    L.down = upload_bq(file, p + "mlp.down");
  }
  lm_head_ = upload_bq(file, "lm_head");
}

}  // namespace qwen38
