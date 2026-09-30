# ---------------------------------------------------------------------------------------------
# Host-only libraries (build everywhere, including machines without CUDA)
# ---------------------------------------------------------------------------------------------

# Byte-level BPE tokenizer (Qwen3.x): NFC normalizer, regex pre-tokenizer, BPE merges,
# added/special tokens. Data comes from a blob produced by tools/tokenizer_export.py and
# embedded in the model file.
add_library(qwen38_tokenizer STATIC tokenizer.cpp)
target_include_directories(qwen38_tokenizer PUBLIC ${PROJECT_SOURCE_DIR}/include)
qwen38_host_flags(qwen38_tokenizer)

add_library(qwen38_core STATIC
  model_file.cpp
  bq_host.cpp
  chat.cpp
  ref/ops.cpp
  ref/ref_model.cpp)
target_include_directories(qwen38_core PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(qwen38_core PUBLIC qwen38_tokenizer nlohmann_json::nlohmann_json)
qwen38_host_flags(qwen38_core)

# CPU reference runner: loads a .q38 file and runs the reference forward pass.
add_executable(qwen38-ref apps/ref_main.cpp)
target_link_libraries(qwen38-ref PRIVATE qwen38_core)
qwen38_host_flags(qwen38-ref)

# ---------------------------------------------------------------------------------------------
# CUDA engine
# ---------------------------------------------------------------------------------------------
if(QWEN38_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_library(qwen38_kernels STATIC
    kernels/qgemm.cu
    kernels/attention_prefill.cu
    kernels/attention_decode.cu
    kernels/qk_norm_rope.cu
    kernels/gdn.cu
    kernels/embed.cu
    kernels/sampling.cu)
  target_include_directories(qwen38_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(qwen38_kernels PUBLIC CUDA::cudart qwen38_cutlass)
  qwen38_cuda_flags(qwen38_kernels)

  add_library(qwen38_engine STATIC
    device.cpp
    weights.cpp
    kv_cache.cpp
    model_runner.cpp
    engine.cpp
    tuning.cpp)
  target_link_libraries(qwen38_engine PUBLIC qwen38_kernels qwen38_core CUDA::cudart)
  qwen38_host_flags(qwen38_engine)

  add_executable(qwen38 apps/cli.cpp)
  target_link_libraries(qwen38 PRIVATE qwen38_engine)
  qwen38_host_flags(qwen38)

  add_executable(qwen38-bench apps/bench.cpp)
  target_link_libraries(qwen38-bench PRIVATE qwen38_engine)
  qwen38_host_flags(qwen38-bench)
endif()
