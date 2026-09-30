add_library(qwen38_test_main OBJECT test_main.cpp)
target_include_directories(qwen38_test_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(QWEN38_TESTDATA ${PROJECT_SOURCE_DIR}/testdata)

# qwen38_add_test(<name> SOURCES ... LIBS ...)
function(qwen38_add_test name)
  cmake_parse_arguments(T "" "" "SOURCES;LIBS" ${ARGN})
  add_executable(${name} ${T_SOURCES} $<TARGET_OBJECTS:qwen38_test_main>)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(${name} PRIVATE ${T_LIBS})
  target_compile_definitions(${name} PRIVATE QWEN38_TESTDATA_DIR="${QWEN38_TESTDATA}")
  qwen38_host_flags(${name})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# ---- host tests (no GPU). Test data: see tests/README.md -----------------------------------------
qwen38_add_test(test_bq_layout SOURCES host/test_bq_layout.cpp LIBS qwen38_core)
qwen38_add_test(test_tokenizer SOURCES host/test_tokenizer.cpp LIBS qwen38_tokenizer)
set_tests_properties(test_tokenizer PROPERTIES ENVIRONMENT
  "QWEN38_TOKENIZER_BLOB=${QWEN38_TESTDATA}/qwen3.8/tokenizer.bin;QWEN38_TOKENIZER_CASES=${QWEN38_TESTDATA}/qwen3.8/tokenizer_cases.bin")
qwen38_add_test(test_chat SOURCES host/test_chat.cpp LIBS qwen38_core)

# ---- GPU tests (RTX 3090) --------------------------------------------------------------------------
if(QWEN38_ENABLE_CUDA)
  qwen38_add_test(test_kernels SOURCES gpu/test_kernels.cu LIBS qwen38_engine)
  qwen38_cuda_flags(test_kernels)
  qwen38_add_test(test_e2e SOURCES gpu/test_e2e.cpp LIBS qwen38_engine)
endif()
