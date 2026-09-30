// Reader for .q38 model files (docs/FORMAT.md). The data region is memory-mapped on POSIX
// systems and read into memory elsewhere.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "qwen38/config.h"

namespace qwen38 {

enum class DType : uint8_t { BQ4, BQ5, BQ6, BQ8, BF16, F32 };

const char* dtype_name(DType dtype);
inline bool is_bq(DType d) { return d == DType::BQ4 || d == DType::BQ5 || d == DType::BQ6 || d == DType::BQ8; }
inline int bq_bits(DType d) {
  switch (d) {
    case DType::BQ4: return 4;
    case DType::BQ5: return 5;
    case DType::BQ6: return 6;
    case DType::BQ8: return 8;
    default: return 0;
  }
}

struct TensorInfo {
  std::string name;
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  int64_t rows = 0;  // logical rows (differs from shape[0] for padded BQ tensors)
  // direct tensors
  int64_t offset = 0;
  int64_t bytes = 0;
  // BQ planes (offsets relative to the data region; hi_bytes == 0 for 4-bit)
  int64_t lo_offset = 0, lo_bytes = 0;
  int64_t hi_offset = 0, hi_bytes = 0;
  int64_t sc_offset = 0, sc_bytes = 0;

  int64_t numel() const;
  int64_t storage_bytes() const { return is_bq(dtype) ? lo_bytes + hi_bytes + sc_bytes : bytes; }
};

class ModelFile {
 public:
  static std::unique_ptr<ModelFile> open(const std::string& path);
  ~ModelFile();
  ModelFile(const ModelFile&) = delete;
  ModelFile& operator=(const ModelFile&) = delete;

  const ModelConfig& config() const { return config_; }
  const std::string& path() const { return path_; }
  const std::string& recipe_name() const { return recipe_; }
  const std::string& header_json() const { return header_; }
  uint64_t data_start() const { return data_start_; }

  bool has_tensor(const std::string& name) const { return tensors_.count(name) != 0; }
  const TensorInfo& tensor(const std::string& name) const;
  const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }

  // Bytes of the data region, [offset, offset + size) relative to data_start().
  std::span<const uint8_t> data(int64_t offset, int64_t size) const;
  // Resource payload ("tokenizer", "chat_template"); empty span if absent.
  std::span<const uint8_t> resource(const std::string& name) const;

  template <typename T>
  std::span<const T> direct(const std::string& name) const {
    const TensorInfo& t = tensor(name);
    auto d = data(t.offset, t.bytes);
    return {reinterpret_cast<const T*>(d.data()), d.size() / sizeof(T)};
  }

  int64_t total_weight_bytes() const;

 private:
  ModelFile() = default;
  std::string path_;
  std::string header_;
  std::string recipe_;
  ModelConfig config_;
  std::map<std::string, TensorInfo> tensors_;
  std::map<std::string, std::pair<int64_t, int64_t>> resources_;
  uint64_t data_start_ = 0;
  uint64_t file_size_ = 0;
  // mapping
  const uint8_t* base_ = nullptr;  // start of the file mapping (or buffer)
  size_t mapped_size_ = 0;
  std::vector<uint8_t> buffer_;  // non-POSIX fallback
  int fd_ = -1;
};

}  // namespace qwen38
