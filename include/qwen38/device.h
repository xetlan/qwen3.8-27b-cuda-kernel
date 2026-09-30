// RAII ownership of CUDA resources. Every device allocation in the engine is owned by one of
// these types; nothing is freed by hand.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace qwen38 {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);
#define QWEN38_CUDA(expr) ::qwen38::cuda_check((expr), #expr, __FILE__, __LINE__)

// Device memory block.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes);
  ~DeviceBuffer();
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)), bytes_(std::exchange(o.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* get() const { return ptr_; }
  template <typename T>
  T* as() const { return static_cast<T*>(ptr_); }
  size_t bytes() const { return bytes_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

// Page-locked host memory; `mapped` makes it addressable from kernels (zero-copy).
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  PinnedBuffer(size_t bytes, bool mapped);
  ~PinnedBuffer();
  PinnedBuffer(PinnedBuffer&& o) noexcept
      : ptr_(std::exchange(o.ptr_, nullptr)), bytes_(std::exchange(o.bytes_, 0)), mapped_(o.mapped_) {}
  PinnedBuffer& operator=(PinnedBuffer&& o) noexcept;
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  void* get() const { return ptr_; }
  template <typename T>
  T* as() const { return static_cast<T*>(ptr_); }
  void* device_ptr() const;  // valid for mapped buffers
  size_t bytes() const { return bytes_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
  bool mapped_ = false;
};

// One large device allocation carved into aligned sub-ranges (weights, workspace). Sub-ranges
// live as long as the arena; there is no per-allocation free, so no fragmentation.
class DeviceArena {
 public:
  DeviceArena() = default;
  explicit DeviceArena(size_t capacity) : block_(capacity) {}
  void* allocate(size_t bytes, size_t align = 256);
  template <typename T>
  T* allocate_array(size_t count, size_t align = 256) {
    return static_cast<T*>(allocate(count * sizeof(T), align));
  }
  size_t used() const { return used_; }
  size_t capacity() const { return block_.bytes(); }

 private:
  DeviceBuffer block_;
  size_t used_ = 0;
};

class Stream {
 public:
  Stream();
  ~Stream();
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  cudaStream_t get() const { return s_; }
  void sync() const;

 private:
  cudaStream_t s_ = nullptr;
};

class Event {
 public:
  explicit Event(bool timing = false);
  ~Event();
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  cudaEvent_t get() const { return e_; }
  void record(cudaStream_t s) const;
  void sync() const;
  float ms_since(const Event& start) const;

 private:
  cudaEvent_t e_ = nullptr;
};

// Instantiated CUDA graph.
class Graph {
 public:
  Graph() = default;
  ~Graph();
  Graph(Graph&& o) noexcept : exec_(std::exchange(o.exec_, nullptr)) {}
  Graph& operator=(Graph&& o) noexcept;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  void begin_capture(cudaStream_t s);
  void end_capture(cudaStream_t s);
  void launch(cudaStream_t s) const;
  bool valid() const { return exec_ != nullptr; }

 private:
  cudaGraphExec_t exec_ = nullptr;
};

struct DeviceInfo {
  int device = 0;
  std::string name;
  int sm_major = 0, sm_minor = 0;
  int num_sms = 0;
  size_t total_mem = 0;
  size_t free_mem = 0;
  size_t l2_bytes = 0;
  int max_smem_per_block_optin = 0;
  bool can_map_host = false;
};

DeviceInfo query_device(int device = 0);

}  // namespace qwen38
