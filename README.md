# qwen3.8-27b-cuda-kernel

A from-scratch C++20/CUDA inference engine for **Qwen3.8-27B** on a single **RTX 3090**
(sm_86, 24 GB). One model, one GPU, built for maximum tokens/s and maximum context in 24 GB.

* **Weights**: fused 4/5/6/8-bit group-wise format (BQ), pre-permuted into tensor-core fragment
  order and dequantized in registers inside the GEMM. The default recipe uses 14.8 GiB of VRAM;
  the embedding table stays in pinned host memory (zero-copy).
* **GEMM**: hand-written `cp.async` / `ldmatrix` / `mma.sync` pipeline with XOR-swizzled shared
  memory, stream-K scheduling for decode, and fused epilogues. The epilogues cover the folded
  RMSNorm row scale, the residual add plus the next norm's statistics, SwiGLU, and bf16/fp32 stores.
  There is no standalone normalization kernel.
* **Attention**: paged int8 KV cache (64-token pages). Prefill uses a FlashAttention-2 kernel
  written with CuTe from CUTLASS 3.9.2. Decode uses GQA-aware split-KV flash-decoding with fused
  RoPE, q/k-norm and KV quantization, and the output gate is applied in the merge step.
* **Gated DeltaNet** (48 of the 64 layers): fused causal conv + L2 norm, and a register-resident
  128x128 recurrent state with the gated RMSNorm fused in.
* **Serving**: continuous batching (chunked prefill interleaved with batched decode) and
  CUDA-graph decode per batch bucket. The sampler (temperature, top-k, top-p, min-p, presence and
  repetition penalties) is deterministic.
* **Context** (computed from the memory plan; the engine prints the real figure at startup): about
  205K tokens of int8 KV at batch capacity 8 with the default recipe. The full 262K native
  context fits with the `lean` recipe.

Files: [docs/DESIGN.md](docs/DESIGN.md) (architecture, memory plan, kernels),
[docs/FORMAT.md](docs/FORMAT.md) (file and weight layout), [docs/QUALITY.md](docs/QUALITY.md)
(quantization and the 96% criterion), [docs/OPTIMIZATION_PLAN.md](docs/OPTIMIZATION_PLAN.md)
(Nsight Compute targets, autotuning, register pressure, CUDA Graphs, roadmap).

## Status

Everything was written and checked on a machine **without an NVIDIA GPU**, so its status splits
three ways:

| | status |
|---|---|
| converter, file format, CPU reference model vs Hugging Face transformers | **verified**: max relative logit error 4e-6, 100% top-1 on a tiny random model with Qwen3.8's exact geometry |
| in-register dequant bit logic vs the format spec (every slab and lane, 4/5/6/8-bit) | **verified** |
| tokenizer vs HF `tokenizers` (15,002 adversarial cases + 9 MB of text, ids/decode/streaming); chat prompt vs the model's Jinja template | **verified** |
| every CUDA kernel: compiles for sm_86 with nvcc 12.8, zero register spills, SASS uses LDGSTS/LDSM/HMMA | **verified** (`tools/compile_check.sh`) |
| kernels vs CPU reference; tiny-model end to end; real-model speed and quality | **not run yet**: needs the RTX 3090 (`ctest`, `qwen38-bench`, docs/QUALITY.md) |

Expect the first GPU run to find bugs. The GPU tests compare each kernel against the verified
CPU reference, so failures point at one kernel and one configuration.

## 1. Build (on the 3090 machine)

Linux or WSL2, CUDA 12.4-12.9, CMake >= 3.24, GCC >= 11 (or Clang >= 16), Ninja. CUTLASS 3.9.2 and
nlohmann/json are downloaded at configure time.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Targets: `build/src/qwen38` (CLI), `build/src/qwen38-bench` (benchmarks, profiling, tuning),
`build/src/qwen38-ref` (CPU reference), tests in `build/tests/`. Device code is compiled for
`sm_86` only (`-DCMAKE_CUDA_ARCHITECTURES=86-real`).

## 2. Tests

Python 3.12, 3.13 or 3.14 (the tokenizer exporter pins the Unicode tables it was verified
against: `unicodedata` 15.0-16.0).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
# tiny test models + the Hugging Face comparison (also verifies the converter)
.venv/bin/python tests/host/test_ref_vs_hf.py --qwen38-ref build/src/qwen38-ref
.venv/bin/python tools/convert.py --model testdata/tiny/hf --out testdata/tiny/tiny-quality.q38 \
    --recipe quality --no-tokenizer
# tokenizer test data: the real tokenizer.json, its blob, and 15,002 HF-encoded cases
mkdir -p testdata/qwen3.8
hf download Qwen/Qwen3.8-27B tokenizer.json --local-dir testdata/qwen3.8
.venv/bin/python tools/tokenizer_export.py testdata/qwen3.8/tokenizer.json testdata/qwen3.8/tokenizer.bin
.venv/bin/python tests/host/tokenizer_cases.py testdata/qwen3.8/tokenizer.json testdata/qwen3.8/tokenizer_cases.bin
ctest --test-dir build --output-on-failure
```

`test_kernels` checks every kernel against the CPU reference: all GEMM configurations x bit widths
x stream-K/data-parallel x epilogues (plus bitwise determinism), q/k-norm + RoPE + KV append,
decode and prefill attention at page-boundary lengths (int8 and bf16 KV), GDN over chunked
prefill + decode, and the sampler (filters, empirical distribution, determinism). `test_e2e`
runs the tiny model end to end against the reference model, including chunked prefill vs one-shot
prefill and batched vs unbatched generation.

## 3. Convert the model

Needs the BF16 checkpoint (56 GB on disk), about 17 GB for the output, and 32 GB of RAM or more.
Quantization runs on the GPU if PyTorch sees one; it also works on the CPU, just slowly.

```bash
hf download Qwen/Qwen3.8-27B --local-dir models/Qwen3.8-27B
.venv/bin/python tools/convert.py --model models/Qwen3.8-27B --out models/qwen3.8-27b.q38
# sizes of any recipe without converting:
.venv/bin/python tools/convert.py --model models/Qwen3.8-27B --out x --recipe lean --dry-run
```

Recipes: `default` (4.64 bits/weight body, 8-bit head, 14.81 GiB on GPU), `lean` (all 4-bit,
6-bit head, 13.01 GiB), `quality` (6-bit value path, 15.71 GiB). `--bits mlp.down=6` or
`--bits 63.mlp.down=8` override single tensors. `python tools/q38file.py FILE.q38` summarises a
converted file.

## 4. Run

```bash
B=./build/src/qwen38; M=models/qwen3.8-27b.q38
$B --model $M --prompt "Write a CUDA kernel that transposes a matrix."   # thinking on (template default)
$B --model $M --prompt "2+2?" --no-think --greedy
$B --model $M --chat --stats                                             # interactive, /reset, /exit
$B --model $M --batch requests.jsonl --out completions.jsonl --max-batch 8   # continuous batching
$B --model $M --score code.txt                                           # perplexity
```

Sampling defaults come from the model's `generation_config.json`: temperature 1.0, top-k 20,
top-p 0.95. Thinking and effort follow the chat template (`--no-think`, `--effort low|medium|xhigh`).
Useful engine flags: `--kv int8|bf16`, `--kv-tokens N`, `--max-context N`, `--max-batch N`,
`--prefill-chunk N`, `--embed device` (if the machine cannot pin 2.5 GB), `--tuning FILE`.
The reasoning stream goes to stderr (dim) and the answer to stdout.

## 5. Benchmark, profile, tune

```bash
./build/src/qwen38-bench --model $M --tune tuning.json          # GEMM tile autotuning (once per GPU)
./build/src/qwen38-bench --model $M --tuning tuning.json        # decode B=1..16 @1K context + prefill sweep
./build/src/qwen38-bench --model $M --decode 1 --context 131072 --gen 64 --no-prefill
./build/src/qwen38-bench --model $M --profile --no-decode --no-prefill   # per-op time split
```

Nsight Compute recipes and targets are in [docs/OPTIMIZATION_PLAN.md](docs/OPTIMIZATION_PLAN.md).

## Layout

```
include/qwen38/      public headers: model file, config, device RAII, weights, KV cache, runner, engine, tokenizer, chat
src/kernels/       CUDA: ptx.cuh, bq_device.cuh, qgemm.cu, attention_{prefill,decode}.cu, qk_norm_rope.cu, gdn.cu, embed.cu, sampling.cu
src/ref/           CPU reference ops and model (the oracle)
src/apps/          qwen38, qwen38-bench, qwen38-ref
tools/             convert.py, bqpack.py, q38file.py, tokenizer_export.py, make_tiny_model.py, compile_check.sh, ptxas_report.py, eval/
tests/host         tests that run anywhere; tests/gpu: tests that need the 3090
docker/            nvcc compile-check image for machines without a GPU
```

## Prompt requirements, and where they are met

| requirement | where |
|---|---|
| C++20, RAII for all VRAM, pool allocators for the KV cache | `include/qwen38/device.h` (`DeviceBuffer`, `PinnedBuffer`, `DeviceArena`, `Graph`), `kv_cache.h` (page pool + free list) |
| CUTLASS 3.x / CuTe layouts, tiling, warp-level tensor-core math | `src/kernels/attention_prefill.cu` (TiledMMA, Swizzle, LDSM atoms, fragment relayout) |
| inline PTX `cp.async` / `mma.sync`, swizzled shared memory | `src/kernels/ptx.cuh`, `qgemm.cu` (A tile `chunk ^ (row & 7)`) |
| fused RMSNorm / RoPE / (de)quantization / GEMM | GEMM epilogues (norm row scale, residual + norm statistics, SwiGLU), `qk_norm_rope.cu` (norm + RoPE + KV quantize), in-register weight dequant |
| FlashAttention-style tiled attention, no materialized score matrix | `attention_prefill.cu`, `attention_decode.cu` |
| 4-bit quantization, >= 96% of base quality, never below Q3 | `tools/bqpack.py`, `docs/QUALITY.md` (evidence + measurement protocol) |
| `CMakeLists.txt` with `sm_86` nvcc flags | `CMakeLists.txt`, `cmake/CompilerFlags.cmake` |
| ncu targets, autotuning, register pressure, CUDA Graphs plan | `docs/OPTIMIZATION_PLAN.md`; autotuner in `qwen38-bench --tune`, graphs in `model_runner.cpp` |
