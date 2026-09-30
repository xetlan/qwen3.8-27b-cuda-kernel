#include "qwen38/device.h"

#include "qwen38/common.h"

namespace qwen38 {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
  if (err == cudaSuccess) return;
  throw Error(std::string(file) + ":" + std::to_string(line) + ": " + expr + " failed: " +
              cudaGetErrorName(err) + " (" + cudaGetErrorString(err) + ")");
}

DeviceBuffer::DeviceBuffer(size_t bytes) : bytes_(bytes) {
  if (bytes) QWEN38_CUDA(cudaMalloc(&ptr_, bytes));
}

DeviceBuffer::~DeviceBuffer() {
  if (ptr_) cudaFree(ptr_);
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& o) noexcept {
  if (this != &o) {
    if (ptr_) cudaFree(ptr_);
    ptr_ = std::exchange(o.ptr_, nullptr);
    bytes_ = std::exchange(o.bytes_, 0);
  }
  return *this;
}

PinnedBuffer::PinnedBuffer(size_t bytes, bool mapped) : bytes_(bytes), mapped_(mapped) {
  if (bytes) QWEN38_CUDA(cudaHostAlloc(&ptr_, bytes, mapped ? cudaHostAllocMapped : cudaHostAllocDefault));
}

PinnedBuffer::~PinnedBuffer() {
  if (ptr_) cudaFreeHost(ptr_);
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& o) noexcept {
  if (this != &o) {
    if (ptr_) cudaFreeHost(ptr_);
    ptr_ = std::exchange(o.ptr_, nullptr);
    bytes_ = std::exchange(o.bytes_, 0);
    mapped_ = o.mapped_;
  }
  return *this;
}

void* PinnedBuffer::device_ptr() const {
  QWEN38_CHECK(mapped_, "device_ptr() on a non-mapped pinned buffer");
  void* d = nullptr;
  QWEN38_CUDA(cudaHostGetDevicePointer(&d, ptr_, 0));
  return d;
}

void* DeviceArena::allocate(size_t bytes, size_t align) {
  const size_t offset = size_t(align_up(int64_t(used_), int64_t(align)));
  QWEN38_CHECK(offset + bytes <= block_.bytes(), "device arena exhausted: need ", offset + bytes, " of ",
             block_.bytes(), " bytes");
  used_ = offset + bytes;
  return static_cast<uint8_t*>(block_.get()) + offset;
}

Stream::Stream() { QWEN38_CUDA(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking)); }
Stream::~Stream() {
  if (s_) cudaStreamDestroy(s_);
}
void Stream::sync() const { QWEN38_CUDA(cudaStreamSynchronize(s_)); }

Event::Event(bool timing) {
  QWEN38_CUDA(cudaEventCreateWithFlags(&e_, timing ? cudaEventDefault : cudaEventDisableTiming));
}
Event::~Event() {
  if (e_) cudaEventDestroy(e_);
}
void Event::record(cudaStream_t s) const { QWEN38_CUDA(cudaEventRecord(e_, s)); }
void Event::sync() const { QWEN38_CUDA(cudaEventSynchronize(e_)); }
float Event::ms_since(const Event& start) const {
  float ms = 0.0f;
  QWEN38_CUDA(cudaEventElapsedTime(&ms, start.e_, e_));
  return ms;
}

Graph::~Graph() {
  if (exec_) cudaGraphExecDestroy(exec_);
}

Graph& Graph::operator=(Graph&& o) noexcept {
  if (this != &o) {
    if (exec_) cudaGraphExecDestroy(exec_);
    exec_ = std::exchange(o.exec_, nullptr);
  }
  return *this;
}

void Graph::begin_capture(cudaStream_t s) { QWEN38_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal)); }

void Graph::end_capture(cudaStream_t s) {
  cudaGraph_t graph = nullptr;
  QWEN38_CUDA(cudaStreamEndCapture(s, &graph));
  if (exec_) {
    cudaGraphExecDestroy(exec_);
    exec_ = nullptr;
  }
  const cudaError_t err = cudaGraphInstantiateWithFlags(&exec_, graph, 0);
  cudaGraphDestroy(graph);
  QWEN38_CUDA(err);
}

void Graph::launch(cudaStream_t s) const { QWEN38_CUDA(cudaGraphLaunch(exec_, s)); }

DeviceInfo query_device(int device) {
  DeviceInfo d;
  d.device = device;
  cudaDeviceProp prop{};
  QWEN38_CUDA(cudaGetDeviceProperties(&prop, device));
  d.name = prop.name;
  d.sm_major = prop.major;
  d.sm_minor = prop.minor;
  d.num_sms = prop.multiProcessorCount;
  d.total_mem = prop.totalGlobalMem;
  d.l2_bytes = size_t(prop.l2CacheSize);
  d.max_smem_per_block_optin = int(prop.sharedMemPerBlockOptin);
  d.can_map_host = prop.canMapHostMemory != 0;
  QWEN38_CUDA(cudaSetDevice(device));
  size_t free_b = 0, total_b = 0;
  QWEN38_CUDA(cudaMemGetInfo(&free_b, &total_b));
  d.free_mem = free_b;
  return d;
}

}  // namespace qwen38
