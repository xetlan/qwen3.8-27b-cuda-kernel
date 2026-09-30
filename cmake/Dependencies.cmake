# Third-party dependencies, fetched at configure time and pinned by release tag.
#
#   nlohmann/json  v3.11.3  model-file header, batch JSONL I/O
#   CUTLASS        v3.9.2   CuTe layouts / MMA and copy atoms (header-only), CUDA builds only
#
# Set FETCHCONTENT_SOURCE_DIR_<NAME> to use a local checkout (offline builds).

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

if(QWEN38_ENABLE_CUDA)
  # Header-only use: SOURCE_SUBDIR points at a directory without a CMakeLists.txt, so
  # MakeAvailable only downloads and never configures CUTLASS's own (very large) build.
  FetchContent_Declare(
    cutlass
    URL https://github.com/NVIDIA/cutlass/archive/refs/tags/v3.9.2.tar.gz
    URL_HASH SHA256=4b97bd6cece9701664eec3a634a1f2f2061d85bf76d843fa5799e1a692b4db0d
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    SOURCE_SUBDIR include)
  FetchContent_MakeAvailable(cutlass)
  add_library(qwen38_cutlass INTERFACE)
  target_include_directories(qwen38_cutlass SYSTEM INTERFACE
    ${cutlass_SOURCE_DIR}/include
    ${cutlass_SOURCE_DIR}/tools/util/include)
endif()
