# Compiler flags shared by all targets.

function(qwen38_host_flags target)
  if(MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers>)
  endif()
endfunction()

# nvcc flags for sm_86 device code.
#
#   --expt-relaxed-constexpr   CuTe uses constexpr host functions from device code
#   -lineinfo                  source correlation in Nsight Compute without -G
#   -ftz=true                  flush fp32 denormals (softmax / GDN decay underflow to 0 anyway)
#   -Xptxas=-warn-spills       every register spill becomes a visible warning; the
#   -Xptxas=-warn-lmem-usage   `tools/check_spills.sh` gate turns them into errors
#
# Deliberately NOT --use_fast_math: it swaps sincosf for __sincosf, whose error is large for
# RoPE angles up to 2.6e5 rad. Fast intrinsics are used explicitly where they are safe.
function(qwen38_cuda_flags target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr
      -lineinfo
      -ftz=true
      -Xptxas=-warn-spills
      -Xptxas=-warn-lmem-usage
      -Xcompiler=-Wall
      $<$<BOOL:${QWEN38_PTXAS_VERBOSE}>:-Xptxas=-v>
    >)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
endfunction()
