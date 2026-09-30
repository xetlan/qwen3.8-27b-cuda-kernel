#include "qwen38/kv_cache.h"

#include <algorithm>
#include <cstring>

#include "qwen38/common.h"

namespace qwen38 {

size_t KvCache::page_bytes_all_layers(const ModelConfig& cfg, kernels::KvDtype dtype) {
  return kernels::KvCacheDesc::page_bytes_for(dtype, cfg.num_key_value_heads) * size_t(cfg.num_attention_layers());
}

KvCache::KvCache(const ModelConfig& cfg, kernels::KvDtype dtype, int num_pages, int num_slots, int max_pages_per_seq)
    : num_pages_(num_pages),
      num_slots_(num_slots),
      max_pages_per_seq_(max_pages_per_seq),
      dtype_(dtype),
      kv_heads_(cfg.num_key_value_heads) {
  QWEN38_CHECK(cfg.head_dim == kernels::kHeadDim, "KV cache: head_dim");
  page_bytes_ = kernels::KvCacheDesc::page_bytes_for(dtype, kv_heads_);
  layer_stride_ = page_bytes_ * size_t(num_pages);
  const size_t layers = size_t(std::max(1, cfg.num_attention_layers()));
  pool_ = DeviceBuffer(layer_stride_ * layers);
  // Zero once: pages recycled later may hold stale data, which the kernels never read before
  // writing (they mask positions >= the sequence length), but a clean start keeps tools like
  // compute-sanitizer quiet.
  QWEN38_CUDA(cudaMemset(pool_.get(), 0, pool_.bytes()));
  const size_t table_bytes = size_t(num_slots) * max_pages_per_seq * sizeof(int);
  table_dev_ = DeviceBuffer(table_bytes);
  table_host_ = PinnedBuffer(table_bytes, false);
  std::memset(table_host_.get(), 0, table_bytes);
  QWEN38_CUDA(cudaMemset(table_dev_.get(), 0, table_bytes));
  free_.resize(num_pages);
  for (int i = 0; i < num_pages; ++i) free_[i] = num_pages - 1 - i;  // pop_back hands out page 0 first
  owned_.assign(num_slots, {});
  dirty_.assign(num_slots, false);
}

kernels::KvCacheDesc KvCache::desc() const {
  kernels::KvCacheDesc d;
  d.pool = static_cast<uint8_t*>(pool_.get());
  d.dtype = dtype_;
  d.kv_heads = kv_heads_;
  d.num_pages = num_pages_;
  d.page_bytes = page_bytes_;
  d.layer_stride = layer_stride_;
  d.block_table = table_dev_.as<int>();
  d.max_pages_per_seq = max_pages_per_seq_;
  return d;
}

bool KvCache::ensure(int slot, int tokens) {
  QWEN38_CHECK(slot >= 0 && slot < num_slots_, "KV cache: slot out of range");
  const int need = int(ceil_div(tokens, kernels::kPageTokens));
  QWEN38_CHECK(need <= max_pages_per_seq_, "sequence exceeds the per-sequence context limit (", tokens, " tokens)");
  auto& pages = owned_[slot];
  const int extra = need - int(pages.size());
  if (extra <= 0) return true;
  if (extra > int(free_.size())) return false;
  int* row = table_host_.as<int>() + size_t(slot) * max_pages_per_seq_;
  for (int i = 0; i < extra; ++i) {
    const int page = free_.back();
    free_.pop_back();
    row[pages.size()] = page;
    pages.push_back(page);
  }
  dirty_[slot] = true;
  return true;
}

void KvCache::release(int slot) {
  auto& pages = owned_[slot];
  for (auto it = pages.rbegin(); it != pages.rend(); ++it) free_.push_back(*it);
  pages.clear();
}

void KvCache::sync(cudaStream_t stream) {
  for (int s = 0; s < num_slots_; ++s) {
    if (!dirty_[s]) continue;
    const size_t off = size_t(s) * max_pages_per_seq_;
    const size_t bytes = owned_[s].size() * sizeof(int);
    if (bytes)
      QWEN38_CUDA(cudaMemcpyAsync(table_dev_.as<int>() + off, table_host_.as<int>() + off, bytes,
                                cudaMemcpyHostToDevice, stream));
    dirty_[s] = false;
  }
}

size_t GdnStatePool::bytes_per_slot(const ModelConfig& cfg) {
  const size_t layers = size_t(cfg.num_gdn_layers());
  const size_t rec = layers * cfg.linear_num_value_heads * cfg.linear_value_head_dim * cfg.linear_key_head_dim * 4;
  const size_t conv = layers * size_t(cfg.linear_conv_kernel_dim - 1) * cfg.gdn_conv_dim() * 2;
  return rec + conv;
}

GdnStatePool::GdnStatePool(const ModelConfig& cfg, int num_slots) : num_gdn_layers_(cfg.num_gdn_layers()) {
  const size_t layers = size_t(num_gdn_layers_);
  recurrent_per_slot_ = layers * cfg.linear_num_value_heads * cfg.linear_value_head_dim * cfg.linear_key_head_dim * 4;
  conv_per_slot_ = layers * size_t(cfg.linear_conv_kernel_dim - 1) * cfg.gdn_conv_dim() * 2;
  recurrent_ = DeviceBuffer(std::max<size_t>(recurrent_per_slot_ * num_slots, 256));
  conv_ = DeviceBuffer(std::max<size_t>(conv_per_slot_ * num_slots, 256));
  QWEN38_CUDA(cudaMemset(recurrent_.get(), 0, recurrent_.bytes()));
  QWEN38_CUDA(cudaMemset(conv_.get(), 0, conv_.bytes()));
}

kernels::GdnStateDesc GdnStatePool::desc() const {
  kernels::GdnStateDesc d;
  d.recurrent = recurrent_.as<float>();
  d.conv = conv_.as<__nv_bfloat16>();
  d.num_gdn_layers = num_gdn_layers_;
  return d;
}

void GdnStatePool::reset(int slot, cudaStream_t stream) {
  if (num_gdn_layers_ == 0) return;
  QWEN38_CUDA(cudaMemsetAsync(recurrent_.as<uint8_t>() + recurrent_per_slot_ * slot, 0, recurrent_per_slot_, stream));
  QWEN38_CUDA(cudaMemsetAsync(conv_.as<uint8_t>() + conv_per_slot_ * slot, 0, conv_per_slot_, stream));
}

}  // namespace qwen38
