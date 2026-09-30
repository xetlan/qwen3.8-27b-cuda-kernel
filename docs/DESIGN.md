# Design

This is a single-model, single-GPU inference engine: Qwen3.8-27B (text) on one RTX 3090
(GA102, sm_86, 82 SMs, 24 GB GDDR6X at 936 GB/s, 6 MB L2). It is written from scratch in
C++20/CUDA; CUTLASS 3.9.2 is used header-only for CuTe.

## 1. The model, as the engine sees it

Qwen3.8-27B is a *hybrid* dense model (Hugging Face architecture `Qwen3_5ForCausalLM`), verified
against the official `config.json` and the transformers reference implementation:

| | |
|---|---|
| layers | 64: 48 Gated DeltaNet (linear attention) + 16 gated full attention (every 4th) |
| hidden / FFN | 5120 / 17408 (SwiGLU) |
| full attention | 24 Q heads, 4 KV heads (GQA 6), head dim 256, output gate `sigmoid(gate)`, q/k RMSNorm, partial RoPE (first 64 dims, theta 1e7) |
| Gated DeltaNet | 16 QK heads, 48 V heads, dims 128, causal conv (width 4) + SiLU, L2-normalised q/k, recurrent state 128x128 per V head (fp32), gated RMSNorm * SiLU(z) |
| norms | "offset" RMSNorm `(1 + w) * x / rms(x)` everywhere except the GDN output norm |
| vocabulary | 248,320 rows (248,077 real ids), untied embedding and LM head |
| context | 262,144 native |

Consequences that shape the engine:

* Only 16 of 64 layers keep a KV cache. At 4 KV heads x 256 dims that is 32 KiB/token in int8
  (plus 1 KiB of scales), so the whole native context of 262K tokens needs 8.3 GiB.
* The 48 GDN layers carry a fixed-size state instead: 144 MiB (fp32) + 2.9 MiB conv history per
  sequence, independent of length. Concurrency is bounded by this state as much as by KV.
* Decode is memory-bound: every token streams all ~15 GiB of weights once. At batch 1 the ceiling
  is ~936 / 15.9 GB = 59 tok/s; everything in the decode path is organised around keeping DRAM busy.

## 2. Memory plan (default recipe, 24 GiB card)

| item | size | notes |
|---|---:|---|
| weights on GPU | 14.81 GiB | 4/5/8-bit BQ, see QUALITY.md; `lean` recipe: 13.01 GiB |
| embedding | 2.37 GiB **host** | pinned + mapped, read zero-copy (one 10 KB row per token) |
| workspace | ~0.25 GiB | activations for a 1024-token prefill chunk, logits, split-KV scratch |
| GDN state | 154 MiB x (max_batch + 1) | fp32 recurrent state + bf16 conv history per slot |
| headroom | 640 MiB | CUDA context growth, graphs, a display on the same GPU |
| KV cache | everything left | 64-token pages; int8 = 33,792 B/token (all 16 layers) |

`ModelRunner` computes this at startup from `cudaMemGetInfo` (`MemoryPlan::describe()` prints it).
Examples on a headless 3090 (~23.5 GiB free after context creation):

| recipe | max_batch | KV tokens (int8) |
|---|---:|---:|
| default | 8 | ~205K |
| default | 1 | ~240K |
| lean | 1 | ~290K (full 262K context fits) |

## 3. Weight format (BQ) and the fused dequant-GEMM

Every projection is stored as signed 4/5/6/8-bit codes with one bf16 scale per 64 inputs,
pre-permuted offline into the exact `mma.sync.m16n8k16` B-fragment order (docs/FORMAT.md). A lane
loads its 32 codes of a 16x64 slab with **one 16-byte shared-memory load** (plus 4-16 bytes of high
bits for 5/6/8-bit) and turns each code pair into a bf16x2 fragment register with a shift, a
`LOP3` that injects the nibble into the mantissa of 128.0, and an exact `HSUB2`; one `HMUL2`
applies the scale. There is no dequantization pass and no bf16 copy of any weight.

`qgemm` (src/kernels/qgemm.cu):

* **pipeline**: STAGES-deep `cp.async` ring (default 4) carrying the A tile (TM x TK bf16,
  16-byte chunks XOR-swizzled with `chunk ^ (row & 7)`, conflict-free for both the async writes and
  `ldmatrix.x4`), the lanes' weight slabs (contiguous 512-byte runs, written and read linearly: no
  swizzle needed) and the scales. Weights use an L2 `evict_first` cache policy; they are read once
  per step and should not evict activations or KV pages.
* **warp tiling**: warp tile = (MT x m16) x n64 over one k16 slab; WK > 1 splits the stage's slabs
  between warps (decode: 4 warps per n64 column) with a deterministic shared-memory reduction.
* **configs** (compile-time, `qwen38-bench --tune` picks per shape and M):
  `m16_n128_k64_s4` (decode), `m32_n128_k64_s4`, `m64_n128_k64_s4`, `m128_n128_k64_s4` (prefill),
  `m64_n128_k128_s3`.
* **work decomposition**: when the tile count is small relative to the resident CTAs (all
  decode GEMMs), **stream-K**: the (tile, k-iteration) space is split evenly over exactly
  `SMs x occupancy` CTAs, so every SM streams weights until the end. Partial tiles are written to a
  workspace; the last-arriving CTA (atomic counter, self-resetting for graph replay) sums them in
  segment order: results are bit-reproducible. Large-M (prefill) GEMMs run data-parallel tiles with
  grouped (GROUP_M = 8) rasterization for L2 reuse.
* **fused epilogues** (from an fp32 C tile in shared memory, vectorized stores):
  * folded RMSNorm: the norm weights `(1 + w)` are multiplied into the next projection's columns
    at conversion time, and the per-row `rsqrt(mean(x^2) + eps)` is applied in the epilogue;
  * `Residual`: `residual += C`, writes the bf16 copy the next GEMM reads **and** the per-128-column
    sums of squares that the next GEMM's epilogue turns into its RMSNorm row scale. There is no
    standalone RMSNorm kernel anywhere in the model.
  * `SwiGLU`: gate/up rows are interleaved at conversion, so `silu(g) * u` is computed on
    adjacent accumulator registers; the 34,816-wide intermediate never exists in memory.
  * bf16 / fp32 stores (the LM head writes fp32 logits).

## 4. Per-layer kernel sequence

Decode step (B sequences, one token each; captured once per batch-size bucket as a CUDA graph):

```
embed (zero-copy gather, residual init + norm partials)
for each of 64 layers:
  qgemm in-projection      [norm row scale -> bf16]   attention: q|gate|k|v   GDN: q|k|v|z
  attention layer:  qk_norm_rope_append (q/k RMSNorm, partial RoPE, int8 KV quantize + paged write)
                    attention_decode (split-KV flash-decoding) + combine (softmax merge, * sigmoid(gate))
  GDN layer:        gdn_ab (a/b gate projection, bf16 GEMV)
                    gdn_conv (causal conv + SiLU + q/k L2 norm, conv-state carry)
                    gdn_recurrence (decay, delta rule, readout, gated RMSNorm * SiLU(z))
  qgemm out-projection     [residual epilogue]
  qgemm gate_up            [norm row scale + SwiGLU]
  qgemm down               [residual epilogue]
qgemm LM head (fp32 logits) -> sample (penalties, temperature, top-k, top-p, min-p)
```

Prefill uses the same sequence on a chunk of up to 1024 tokens, with the tensor-core
`attention_prefill` (CuTe, FlashAttention-2 schedule) instead of the split-KV decoder, and runs
eagerly (the kernels are long enough that launch overhead is negligible).

## 5. Attention

* **KV cache**: one pool, 64-token pages, page ids shared by all 16 attention layers (block table
  per sequence slot). int8 with one fp16 scale per (token, head, 64 dims) (the rotary 64 dims get
  their own scale), or bf16. Pages come from a free list; a request reserves its whole
  `prompt + max_new_tokens` at admission, so generation never runs out of pages mid-way and no
  preemption logic is needed.
* **decode** (`attention_decode.cu`): grid (splits, KV head, sequence); each CTA processes all 6
  query heads of its KV head, so every K/V byte is read once per group. Lanes own 8 consecutive
  dims; 4 tokens per warp iteration; dot products are finished with a transpose-reduce (each
  shuffle round halves the values per lane: 48 shuffles for 6 heads x 4 tokens instead of 120);
  int8 is widened with `PRMT` into the mantissa of 2^23 plus one `FADD` (no `I2F`); online softmax
  in the log2 domain. Split count targets ~2 waves of CTAs; a second kernel merges splits and
  applies the output gate.
* **prefill** (`attention_prefill.cu`): FlashAttention-2 forward in CuTe: `TiledMMA` of
  `SM80_16x8x16_F32F16F16F32_TN` over 4 warps (64 query rows x 32 keys), swizzled
  (`Swizzle<3,3,3>`) Q/K/V tiles, `SM75_U32x4_LDSM_N` for Q/K and `SM75_U16x8_LDSM_T` for V, and the
  accumulator-to-A-fragment relayout that feeds P to the second GEMM without shared memory.
  K/V tiles are staged raw with `cp.async`, dequantized once into fp16 swizzled tiles, and the next
  tile's load overlaps the current tile's math. Causal masking only on diagonal blocks; stale cache
  slots beyond the sequence are zeroed during dequantization (never multiplied into P*V).

## 6. Gated DeltaNet

* `gdn_conv`: one CTA per (128-channel head block, sequence) slides the width-4 window through all
  of the sequence's new tokens, so the conv state is carried in registers and written back once;
  q/k blocks are L2-normalised 32 tokens at a time through shared memory.
* `gdn_recurrence`: one CTA per (V head, sequence), 512 threads. The 128x128 fp32 state lives in
  registers: 4 lanes per state row (dk quarters of 32), so `S k` and `S q` are 32-term dot products
  plus two shuffles. q/k/v for 16 tokens are double-buffered through shared memory with `cp.async`
  (q/k quarters padded to 36 floats to spread banks). After each 16-token chunk, 16 warps apply the
  gated RMSNorm to 16 tokens in parallel. The recurrence is sequential in time: a few percent of
  prefill time on this model; the chunked WY/UT tensor-core form is in the optimization plan.

## 7. Scheduler (continuous batching)

`Engine` (src/engine.cpp) keeps a FIFO of requests and up to `max_batch` active sequences, each
owning a *slot* (block-table row + GDN state image + penalty history). Every iteration:

1. admit waiting requests while a slot and the full KV reservation are available;
2. run **one** prefill chunk (up to 1024 tokens) of the oldest sequence still reading its prompt;
3. run **one** decode step for every generating sequence (CUDA graph for the batch bucket
   1/2/4/8/12/16/...; padding rows run a private dummy sequence);
4. apply stop conditions (EOS ids, max tokens, context limit), stream tokens, free slots.

Long prompts therefore delay running generations by at most one chunk (~1 s at 1024 tokens).

## 8. Numerics

| value | precision |
|---|---|
| residual stream | fp32 |
| GEMM inputs | bf16 activations x bf16(code * scale) weights, fp32 accumulate |
| q, k, v, P in attention | fp16 (int8 or bf16 KV dequantized to fp16 tiles), fp32 softmax and O |
| GDN state, conv, gates, recurrence | fp32 |
| logits, sampling | fp32; probability masses in 2^-40 fixed point (deterministic top-p) |

The CPU reference (`src/ref/`) mirrors every rounding point ("engine mode") so kernel tests can
use tight tolerances; its "exact mode" matches Hugging Face transformers to fp32 rounding.

## 9. What is verified, where

| check | where | status |
|---|---|---|
| converter + BQ packing + norm folding + tensor mapping + full layer math vs HF transformers | `tests/host/test_ref_vs_hf.py` (tiny random model, same geometry) | passes on macOS: max relative logit error 4e-6, top-1 100% |
| device dequant bit logic vs independent spec decoder, all slabs/lanes, 4/5/6/8-bit | `tests/host/test_bq_layout.cpp` | passes |
| tokenizer vs HF `tokenizers` | `tests/host/test_tokenizer.cpp` | passes |
| chat formatting vs the model's Jinja template | `tests/host/test_chat.cpp` | passes |
| every CUDA kernel compiles for sm_86, no register spills, expected SASS (LDGSTS/LDSM/HMMA) | `tools/compile_check.sh` in `docker/Dockerfile.compile` | passes (nvcc 12.8) |
| every kernel vs the CPU reference; GEMM determinism | `tests/gpu/test_kernels.cu` | **needs the RTX 3090** |
| tiny model end-to-end, chunked prefill + graph decode, batched = unbatched | `tests/gpu/test_e2e.cpp` | **needs the RTX 3090** |
| throughput, memory, quality on the real 27B model | `qwen38-bench`, `qwen38 --score`, tools/eval | **needs the RTX 3090** |
