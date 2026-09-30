cmake_minimum_required(VERSION 3.24)
project(qwen38 VERSION 0.1.0 LANGUAGES CXX)

# qwen3.8-27b-cuda-kernel: Qwen3.8-27B inference engine for a single RTX 3090 (sm_86).
#
#   QWEN38_ENABLE_CUDA=ON   full engine (Linux or WSL2, CUDA >= 12.4)
#   QWEN38_ENABLE_CUDA=OFF  host-only build: model file reader, tokenizer, CPU reference model,
#                         host tests. Used for development and verification on machines
#                         without an NVIDIA GPU.

option(QWEN38_ENABLE_CUDA "Build the CUDA engine (nvcc, sm_86)" ON)
option(QWEN38_BUILD_TESTS "Build tests" ON)
option(QWEN38_PTXAS_VERBOSE "Print per-kernel register/spill statistics from ptxas" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(QWEN38_ENABLE_CUDA)
  # sm_86 only: GA102 (RTX 3090). "86-real" emits SASS for sm_86 and no PTX, so a
  # wrong GPU fails loudly at load time instead of JIT-compiling a slow fallback.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86-real)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.4)
    message(FATAL_ERROR "CUDA >= 12.4 is required (found ${CMAKE_CUDA_COMPILER_VERSION})")
  endif()
endif()

include(cmake/Dependencies.cmake)
include(cmake/CompilerFlags.cmake)

add_subdirectory(src)

if(QWEN38_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
