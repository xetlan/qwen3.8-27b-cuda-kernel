// Paged KV cache for the full-attention layers and the Gated DeltaNet state pool.
//
// KV memory is one device allocation split into 64-token pages; a page id is valid in every
// attention layer at once (layer-major pool, docs/FORMAT.md 3.1). Sequences own "slots": a slot
// indexes a block-table row (its page list) and a GDN state image. Pages are handed out from a
// free list, so there is no fragmentation however sequences come and go.
#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "qwen38/config.h"
#include "qwen38/device.h"
#include "kernels/kernels.h"

namespace qwen38 {

class KvCache {
 public:
  KvCache(const ModelConfig& cfg, kernels::KvDtype dtype, int num_pages, int num_slots, int max_pages_per_seq);

  // Bytes of one page across all attention layers.
  static size_t page_bytes_all_layers(const ModelConfig& cfg, kernels::KvDtype dtype);

  kernels::KvCacheDesc desc() const;
  int num_pages() const { return num_pages_; }
  int free_pages() const { return int(free_.size()); }
  int pages_of(int slot) const { return int(owned_[slot].size()); }
  int max_pages_per_seq() const { return max_pages_per_seq_; }

  // Make sure `slot` has pages for positions [0, tokens). Returns false (and changes nothing)
  // if the pool cannot provide them.
  bool ensure(int slot, int tokens);
  void release(int slot);
  // Upload modified block-table rows (async on `stream`).
  void sync(cudaStream_t stream);

 private:
  int num_pages_, num_slots_, max_pages_per_seq_;
  kernels::KvDtype dtype_;
  int kv_heads_;
  size_t page_bytes_ = 0;
  size_t layer_stride_ = 0;
  DeviceBuffer pool_;
  DeviceBuffer table_dev_;
  PinnedBuffer table_host_;
  std::vector<int> free_;
  std::vector<std::vector<int>> owned_;
  std::vector<bool> dirty_;
};

// Recurrent + conv state images of the GDN layers, one per slot.
class GdnStatePool {
 public:
  GdnStatePool(const ModelConfig& cfg, int num_slots);
  static size_t bytes_per_slot(const ModelConfig& cfg);
  kernels::GdnStateDesc desc() const;
  void reset(int slot, cudaStream_t stream);  // zero state for a new sequence

 private:
  size_t recurrent_per_slot_ = 0, conv_per_slot_ = 0;
  int num_gdn_layers_ = 0;
  DeviceBuffer recurrent_, conv_;
};

}  // namespace qwen38
