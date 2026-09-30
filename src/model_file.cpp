#include "qwen38/model_file.h"

#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>

#include "qwen38/common.h"

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define QWEN38_HAVE_MMAP 1
#endif

namespace qwen38 {

// ---- common.h ------------------------------------------------------------------------------

void fail(const std::string& message) { throw Error(message); }

std::string format_check_failure(const char* expr, const char* file, int line, const std::string& msg) {
  std::string out = std::string(file) + ":" + std::to_string(line) + ": check failed: " + expr;
  if (!msg.empty()) out += ": " + msg;
  return out;
}

// ---- config.h ------------------------------------------------------------------------------

int ModelConfig::num_attention_layers() const {
  int n = 0;
  for (auto a : is_attention) n += a ? 1 : 0;
  return n;
}

int ModelConfig::num_gdn_layers() const { return num_layers - num_attention_layers(); }

int ModelConfig::attn_index(int layer) const {
  if (!is_attention[layer]) return -1;
  int idx = 0;
  for (int i = 0; i < layer; ++i) idx += is_attention[i] ? 1 : 0;
  return idx;
}

int ModelConfig::gdn_index(int layer) const {
  if (is_attention[layer]) return -1;
  int idx = 0;
  for (int i = 0; i < layer; ++i) idx += is_attention[i] ? 0 : 1;
  return idx;
}

void ModelConfig::validate() const {
  QWEN38_CHECK(num_layers > 0 && int(is_attention.size()) == num_layers, "layer_types size");
  QWEN38_CHECK(hidden_size % 128 == 0, "hidden_size must be a multiple of 128, got ", hidden_size);
  QWEN38_CHECK(vocab_size % 128 == 0, "vocab_size must be a multiple of 128, got ", vocab_size);
  QWEN38_CHECK(intermediate_size % 64 == 0, "intermediate_size must be a multiple of 64");
  if (num_attention_layers() > 0) {
    QWEN38_CHECK(head_dim == 256, "attention kernels implement head_dim 256, got ", head_dim);
    QWEN38_CHECK(rotary_dim == 64, "RoPE kernel implements rotary_dim 64, got ", rotary_dim);
    QWEN38_CHECK(num_key_value_heads > 0 && num_attention_heads % num_key_value_heads == 0, "GQA heads");
    QWEN38_CHECK(attn_group() <= 8, "GQA group > 8 is not implemented");
  }
  if (num_gdn_layers() > 0) {
    QWEN38_CHECK(linear_key_head_dim == 128 && linear_value_head_dim == 128,
               "GDN kernels implement 128-dim heads");
    QWEN38_CHECK(linear_conv_kernel_dim == 4, "GDN conv kernel implements width 4");
    QWEN38_CHECK(linear_num_value_heads % linear_num_key_heads == 0, "GDN head ratio");
  }
  QWEN38_CHECK(norm_folded, "model files without folded norms are not supported");
}

std::string ModelConfig::summary() const {
  std::ostringstream os;
  os << "hidden " << hidden_size << ", layers " << num_layers << " (" << num_attention_layers()
     << " attention / " << num_gdn_layers() << " GDN), vocab " << vocab_size << ", ffn "
     << intermediate_size << ", attn " << num_attention_heads << "q/" << num_key_value_heads << "kv x"
     << head_dim << ", gdn " << linear_num_key_heads << "k/" << linear_num_value_heads << "v x"
     << linear_key_head_dim;
  return os.str();
}

// ---- model file ----------------------------------------------------------------------------

const char* dtype_name(DType dtype) {
  switch (dtype) {
    case DType::BQ4: return "bq4";
    case DType::BQ5: return "bq5";
    case DType::BQ6: return "bq6";
    case DType::BQ8: return "bq8";
    case DType::BF16: return "bf16";
    case DType::F32: return "f32";
  }
  return "?";
}

static DType parse_dtype(const std::string& s) {
  if (s == "bq4") return DType::BQ4;
  if (s == "bq5") return DType::BQ5;
  if (s == "bq6") return DType::BQ6;
  if (s == "bq8") return DType::BQ8;
  if (s == "bf16") return DType::BF16;
  if (s == "f32") return DType::F32;
  fail("unknown dtype " + s);
}

int64_t TensorInfo::numel() const {
  int64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

static ModelConfig parse_config(const nlohmann::json& j) {
  ModelConfig c;
  c.hidden_size = j.at("hidden_size");
  c.num_layers = j.at("num_layers");
  c.vocab_size = j.at("vocab_size");
  c.intermediate_size = j.at("intermediate_size");
  c.num_attention_heads = j.at("num_attention_heads");
  c.num_key_value_heads = j.at("num_key_value_heads");
  c.head_dim = j.at("head_dim");
  c.rotary_dim = j.at("rotary_dim");
  c.rope_theta = j.at("rope_theta");
  c.linear_num_key_heads = j.at("linear_num_key_heads");
  c.linear_num_value_heads = j.at("linear_num_value_heads");
  c.linear_key_head_dim = j.at("linear_key_head_dim");
  c.linear_value_head_dim = j.at("linear_value_head_dim");
  c.linear_conv_kernel_dim = j.at("linear_conv_kernel_dim");
  c.rms_norm_eps = j.at("rms_norm_eps");
  c.max_position_embeddings = j.at("max_position_embeddings");
  c.norm_folded = j.value("norm_folded", false);
  for (const auto& t : j.at("layer_types")) {
    const std::string kind = t;
    QWEN38_CHECK(kind == "full_attention" || kind == "linear_attention", "layer type ", kind);
    c.is_attention.push_back(kind == "full_attention" ? 1 : 0);
  }
  for (const auto& e : j.value("eos_token_ids", nlohmann::json::array())) c.eos_token_ids.push_back(e);
  return c;
}

std::unique_ptr<ModelFile> ModelFile::open(const std::string& path) {
  std::unique_ptr<ModelFile> mf(new ModelFile());
  mf->path_ = path;

  std::ifstream in(path, std::ios::binary);
  QWEN38_CHECK(in.good(), "cannot open ", path);
  char magic[8];
  in.read(magic, 8);
  QWEN38_CHECK(in.good() && std::memcmp(magic, "QW38MDL1", 8) == 0, path, " is not a .q38 model file");
  uint64_t header_len = 0;
  in.read(reinterpret_cast<char*>(&header_len), 8);
  QWEN38_CHECK(header_len > 0 && header_len < (1ull << 30), "bad header length");
  mf->header_.resize(header_len);
  in.read(mf->header_.data(), std::streamsize(header_len));
  QWEN38_CHECK(in.good(), "truncated header");
  in.seekg(0, std::ios::end);
  mf->file_size_ = uint64_t(in.tellg());
  mf->data_start_ = uint64_t(align_up(int64_t(16 + header_len), 4096));

  const auto j = nlohmann::json::parse(mf->header_);
  QWEN38_CHECK(j.at("format") == "qwen38-model", "not a qwen38 model header");
  QWEN38_CHECK(j.at("format_version") == 1, "unsupported format_version ", j.at("format_version").dump());
  mf->config_ = parse_config(j.at("config"));
  mf->recipe_ = j.contains("recipe") ? j["recipe"].value("name", "") : "";

  for (const auto& [name, rec] : j.at("tensors").items()) {
    TensorInfo t;
    t.name = name;
    t.dtype = parse_dtype(rec.at("dtype"));
    for (const auto& d : rec.at("shape")) t.shape.push_back(d);
    t.rows = t.shape.empty() ? 1 : t.shape[0];
    if (rec.contains("rows")) t.rows = rec["rows"];
    if (is_bq(t.dtype)) {
      QWEN38_CHECK(t.shape.size() == 2, name, ": BQ tensors are rank 2");
      QWEN38_CHECK(rec.value("group_size", 0) == 64, name, ": group_size must be 64");
      QWEN38_CHECK(t.shape[0] % 128 == 0 && t.shape[1] % 64 == 0, name, ": BQ shape constraints");
      const auto& planes = rec.at("planes");
      t.lo_offset = planes.at("lo")[0];
      t.lo_bytes = planes.at("lo")[1];
      if (planes.contains("hi")) {
        t.hi_offset = planes["hi"][0];
        t.hi_bytes = planes["hi"][1];
      }
      t.sc_offset = planes.at("scales")[0];
      t.sc_bytes = planes.at("scales")[1];
      const int bits = bq_bits(t.dtype);
      QWEN38_CHECK(t.lo_bytes == t.shape[0] * t.shape[1] / 2, name, ": lo plane size");
      QWEN38_CHECK(t.hi_bytes == t.shape[0] * t.shape[1] * (bits - 4) / 8, name, ": hi plane size");
      QWEN38_CHECK(t.sc_bytes == t.shape[0] * (t.shape[1] / 64) * 2, name, ": scale plane size");
    } else {
      t.offset = rec.at("offset");
      t.bytes = rec.at("bytes");
      const int64_t esize = t.dtype == DType::BF16 ? 2 : 4;
      QWEN38_CHECK(t.bytes == t.numel() * esize, name, ": size mismatch");
    }
    mf->tensors_.emplace(name, std::move(t));
  }
  if (j.contains("resources")) {
    for (const auto& [name, rec] : j["resources"].items()) {
      mf->resources_[name] = {rec.at("offset").get<int64_t>(), rec.at("bytes").get<int64_t>()};
    }
  }
  in.close();

#if QWEN38_HAVE_MMAP
  mf->fd_ = ::open(path.c_str(), O_RDONLY);
  QWEN38_CHECK(mf->fd_ >= 0, "cannot open ", path);
  void* p = ::mmap(nullptr, mf->file_size_, PROT_READ, MAP_PRIVATE, mf->fd_, 0);
  QWEN38_CHECK(p != MAP_FAILED, "mmap failed for ", path);
  mf->base_ = static_cast<const uint8_t*>(p);
  mf->mapped_size_ = mf->file_size_;
#else
  mf->buffer_.resize(mf->file_size_);
  std::ifstream all(path, std::ios::binary);
  all.read(reinterpret_cast<char*>(mf->buffer_.data()), std::streamsize(mf->file_size_));
  QWEN38_CHECK(all.good(), "failed reading ", path);
  mf->base_ = mf->buffer_.data();
  mf->mapped_size_ = mf->file_size_;
#endif

  // bounds check every object once
  for (const auto& [name, t] : mf->tensors_) {
    auto check = [&](int64_t off, int64_t size) {
      QWEN38_CHECK(off >= 0 && size >= 0 && mf->data_start_ + uint64_t(off + size) <= mf->file_size_,
                 name, ": object outside the file (truncated download?)");
    };
    if (is_bq(t.dtype)) {
      check(t.lo_offset, t.lo_bytes);
      check(t.hi_offset, t.hi_bytes);
      check(t.sc_offset, t.sc_bytes);
    } else {
      check(t.offset, t.bytes);
    }
  }
  mf->config_.validate();
  return mf;
}

ModelFile::~ModelFile() {
#if QWEN38_HAVE_MMAP
  if (base_ && mapped_size_) ::munmap(const_cast<uint8_t*>(base_), mapped_size_);
  if (fd_ >= 0) ::close(fd_);
#endif
}

const TensorInfo& ModelFile::tensor(const std::string& name) const {
  auto it = tensors_.find(name);
  QWEN38_CHECK(it != tensors_.end(), "tensor '", name, "' not found in ", path_);
  return it->second;
}

std::span<const uint8_t> ModelFile::data(int64_t offset, int64_t size) const {
  QWEN38_CHECK(offset >= 0 && size >= 0 && data_start_ + uint64_t(offset + size) <= file_size_, "data range");
  return {base_ + data_start_ + offset, size_t(size)};
}

std::span<const uint8_t> ModelFile::resource(const std::string& name) const {
  auto it = resources_.find(name);
  if (it == resources_.end()) return {};
  return data(it->second.first, it->second.second);
}

int64_t ModelFile::total_weight_bytes() const {
  int64_t total = 0;
  for (const auto& [name, t] : tensors_) total += t.storage_bytes();
  return total;
}

}  // namespace qwen38
