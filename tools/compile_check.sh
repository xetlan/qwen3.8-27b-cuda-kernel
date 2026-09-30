#!/usr/bin/env bash
# Full CUDA build (sm_86) + register-spill gate. Needs nvcc but no GPU, so it also runs on
# development machines through the compile container:
#   docker build -t qwen38-compile -f docker/Dockerfile.compile docker
#   docker run --rm -v "$PWD":/src -w /src qwen38-compile tools/compile_check.sh
set -euo pipefail
BUILD=${BUILD:-build-cuda}
# every kernel must go through ptxas for the spill gate: always build from scratch
rm -rf "$BUILD"
EXTRA=()
if [ -d build-deps/cutlass-3.9.2/include ]; then
  EXTRA+=("-DFETCHCONTENT_SOURCE_DIR_CUTLASS=$PWD/build-deps/cutlass-3.9.2")
fi
cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DQWEN38_PTXAS_VERBOSE=ON "${EXTRA[@]}" > /dev/null
set +e
cmake --build "$BUILD" -j "$(nproc)" 2>&1 | tee "$BUILD/build.log" | grep -E "error|FAILED|warning: " | grep -v "Wunused"
status=${PIPESTATUS[0]}
set -e
if [ "$status" -ne 0 ]; then
  echo "compile_check: build FAILED"
  exit 1
fi
# ptxas prints "N bytes stack frame, S bytes spill stores, L bytes spill loads" per kernel
spills=$(grep -E "bytes spill stores" "$BUILD/build.log" | grep -vE " 0 bytes spill stores, 0 bytes spill loads" || true)
if [ -n "$spills" ]; then
  echo "compile_check: register spills found:"
  grep -B1 -E "bytes spill stores" "$BUILD/build.log" | grep -vE " 0 bytes spill stores, 0 bytes spill loads" | head -40
  exit 1
fi
kernels=$(grep -c "Compiling entry function" "$BUILD/build.log" || true)
echo "compile_check: OK ($kernels kernel entry points compiled for sm_86, no register spills)"
