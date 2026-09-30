# Iterative optimization plan

This is the follow-up plan requested in the prompt: what to measure with Nsight Compute, how tile
sizes are autotuned for the 3090, how register pressure is controlled, how CUDA Graphs are used,
and the ordered list of larger optimizations. It starts from what exists now (numbers below are
from `tools/ptxas_report.py` on the sm_86 build) and from the roofline of the hardware.

## 0. Performance model: what "maximum" means on this card

| resource | RTX 3090 (GA102) | consequence |
|---|---|---|
| DRAM | 936 GB/s (~850 sustainable) | decode reads ~15.9 GB/token at batch 1 (default recipe) |
| bf16/fp16 MMA, fp32 accumulate | 71 TFLOPS (half of the fp16-accumulate rate on GeForce Ampere) | prefill of 27B: ~50 GFLOP/token |
| int8 MMA | 284 TOPS (4x the line above) | the big lever for prefill (section 5) |
| L2 | 6 MB | a GEMM's A tile set for 2048 tokens is 21 MB: rasterization matters |
| SMs / smem | 82 SMs, 99 KB per block, 100 KB per SM | 1 prefill GEMM CTA per SM, 2 decode CTAs per SM |

Ceilings for the default recipe (14.8 GiB of weights on the GPU):

| phase | bound | ceiling | first target |
|---|---|---:|---:|
| decode, batch 1, short context | DRAM | ~53 tok/s at 850 GB/s | 45 tok/s (85%) |
| decode, batch 8 | DRAM (weights once per step) + state/KV per sequence | ~330 tok/s aggregate | 250 tok/s |
| decode, batch 1, 128K context | DRAM: +4.2 GB of int8 KV per step | ~42 tok/s | 35 tok/s |
| prefill, 1024-token chunks | tensor cores (bf16, fp32 acc) | ~1400 tok/s at 71 TFLOPS | 900 tok/s |

Every optimization below is judged against these ceilings, not against another engine.

## 1. Nsight Compute targets

Profile one decode step and one prefill chunk with CUDA graphs off (`qwen38-bench --no-graphs`), one
kernel instance per class:

```
ncu --set full --target-processes all -k regex:"qgemm|attn_|gdn_|qk_norm|sample" \
    --launch-skip 200 --launch-count 40 -o decode \
    ./build/src/qwen38-bench --model M.q38 --no-prefill --decode 1 --gen 4 --no-graphs
ncu --import decode.ncu-rep --page details --csv > decode.csv
```

`qwen38-bench --profile` gives the per-op time split first, so that ncu time goes to the kernels
that dominate.

### 1.1 Decode GEMMs (`qgemm` m16 config, stream-K): memory-bound

| metric | target | if missed |
|---|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | >= 85% | more bytes in flight: STAGES 4 -> 5/6, TK 64 -> 128, occupancy 2 -> 3 CTAs/SM |
| `dram__bytes_read.sum` / (weight bytes of the launch) | <= 1.03 | A tile or partial-tile traffic leaking to DRAM (should hit L2) |
| `lts__t_sector_hit_rate.pct` (A tile, partials) | >= 90% for non-weight traffic | check evict_first policy is applied to weights only |
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | dominant stall is fine; `stall_barrier` should be < 15% | fewer `__syncthreads` per byte: TK 128 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | >= 25% (16 of 48 warps) | registers: see section 3 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | ~0 | A-tile swizzle regression |
| `smsp__inst_executed_pipe_lsu.avg` vs `..._pipe_alu` | dequant ALU well under issue limit | fewer shifts: pre-shifted nibble planes |
| `launch__grid_size` | = SMs x occupancy | stream-K grid misconfigured |
| stream-K fixup share: time of the partial-tile store/reduce | < 3% of kernel time | fewer, larger segments (grid = SMs x 1) |

### 1.2 Prefill GEMMs (`qgemm` m128 config, data-parallel): tensor-bound

| metric | target | if missed |
|---|---|---|
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | >= 60% | more MMAs per ldmatrix/dequant: 64x... warp tiles, B fragment reuse across more m16 tiles |
| `smsp__inst_executed_op_ldsm.sum` / HMMA | <= 0.25 | larger warp tile M |
| `smsp__warp_issue_stalled_math_pipe_throttle` | high is good | |
| `smsp__warp_issue_stalled_barrier` | < 10% | deeper pipeline or fewer syncs (TK 128) |
| `lts__t_bytes.sum` / FLOP | minimise | GROUP_M rasterization sweep (4, 8, 16) |

### 1.3 Attention

| kernel | metric | target |
|---|---|---|
| `attn_decode` | `dram__throughput` pct of peak (long contexts) | >= 75% at 32K+ context |
| `attn_decode` | achieved occupancy (209 registers -> 8 warps/SM) | raise to 16 warps (section 3) |
| `attn_prefill` | HMMA pipe active | >= 45% (FA2 hdim256 on sm86 typically 40-50%) |
| `attn_prefill` | conversion pass time / total | < 15%; else convert in registers (int8 ldmatrix) |

### 1.4 GDN

| kernel | metric | target |
|---|---|---|
| `gdn_recurrence` (decode) | DRAM throughput while loading/storing the 64 KB state per CTA | >= 60% (only 48 x B CTAs) |
| `gdn_recurrence` (prefill) | time per token per layer | <= 0.5 us |
| `gdn_conv`, `gdn_ab` | total share of a decode step | < 3% combined |

### 1.5 Whole-step checks

* Nsight Systems on a graph-replayed decode step: GPU idle between steps (host scheduling) < 1%;
  gaps between kernels inside the graph < 2 us.
* `compute-sanitizer --tool racecheck/memcheck` on `test_kernels` and `test_e2e` once per change
  to a kernel's synchronisation.

## 2. Autotuning tile sizes (82 SMs, 6 MB L2)

**Implemented.** `qgemm` is a template over `(BITS, WM, WN, WK, MT, TK, STAGES, MIN_BLOCKS)`,
i.e. BLOCK_M = WM x MT x 16, BLOCK_N = WN x 64, BLOCK_K = TK, plus the intra-CTA split-K factor WK
and pipeline depth. The compiled set lives in one list in `src/kernels/qgemm.cu`:

```cpp
//                    WM WN WK MT  TK STG MINB
using CfgM16  = Cfg<1, 2, 4, 1, 64, 4, 2>;   // decode, M <= 16
using CfgM32  = Cfg<1, 2, 4, 2, 64, 4, 1>;
using CfgM64  = Cfg<2, 2, 2, 2, 64, 4, 1>;
using CfgM128 = Cfg<4, 2, 1, 2, 64, 4, 1>;   // prefill
using CfgM64K128 = Cfg<2, 2, 2, 2, 128, 3, 1>;
```

`qwen38-bench --tune tuning.json` times every config for every `(bits, N, K)` of the loaded model at
M in {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024} and writes the winners;
`qwen38 --tuning tuning.json` installs them (smallest `m_max >= M` wins, else the built-in M
heuristic). Adding a candidate is one `using` line, one `kConfigs` entry and one `case` in
`qgemm()`; `static_assert`s reject configs that exceed 99 KB of shared memory.

**Sweep plan for the 3090** (in this order, each measured with the tuner, not guessed):

1. decode (M <= 16): `(WK, STAGES, TK) in {2,4} x {3,4,5,6} x {64,128}`, `MIN_BLOCKS in {2,3}`: the
   goal is bytes in flight per SM (>= 24 KB) without dropping below 2 CTAs/SM. Grid is always
   `SMs x occupancy` (stream-K), so 82 SMs never idle behind a wave tail.
2. prefill (M >= 256): BLOCK_M x BLOCK_N in {128x128, 128x256, 256x128} with `MT` in {2, 4};
   STAGES 3/4; GROUP_M (L2 rasterization) in {4, 8, 16}. For N=34816, K=5120, a GROUP_M=8 group
   of 128-row tiles keeps 8 x 1.3 MB of A plus the current B strip in the 6 MB L2 only partially;
   GROUP_M=4 may win for the widest GEMMs and 16 for the narrow ones: that is exactly what the sweep
   decides per shape.
3. the stream-K / data-parallel switch point (`tiles >= 3 x resident`) per shape: add it to the
   tuning entry if the sweep shows shape dependence.

## 3. Register pressure

Current (ptxas, sm_86, zero spills everywhere):

| kernel | regs | CTAs/SM | warps/SM | note |
|---|---:|---:|---:|---|
| qgemm m16 (decode) | 128 | 2 | 16 | capped by `__launch_bounds__(256, 2)`; cap is binding |
| qgemm m32 | 178-196 | 1 | 8 | would spill at 128 |
| qgemm m128 (prefill) | 186-207 | 1 | 8 | 64 fp32 accumulators + fragments |
| attn_prefill | 242-246 | 1 | 4 | 128 fp32 O accumulators per thread (hdim 256) |
| attn_decode group 6 | 190-209 | 2 | 8 | q (48) + acc (48) per lane |
| gdn_recurrence | 106 | 1 | 16 | 32 state + 32 k values per lane |

Strategies, per kernel:

* **Gate every build on zero spills**: `tools/compile_check.sh` fails on any spill store/load.
  Every `__launch_bounds__` carries an explicit min-blocks value so ptxas targets a known budget.
* **qgemm decode**: the 128-register cap binds. If ncu shows long-scoreboard stalls dominating,
  try `MIN_BLOCKS=3` (85 registers) with `MT=1, WK=8`: fewer accumulators per warp (each warp owns
  fewer n8 subtiles) in exchange for more warps.
* **qgemm prefill**: do not raise occupancy: 8 warps with 16 independent MMAs each saturate the
  tensor pipe. If the HMMA pipe is under 60%, the fix is fewer auxiliary instructions per MMA
  (larger warp tiles with register reuse of B fragments), not more warps.
* **attn_decode**: split the 6 heads of a group over 2 lane groups (3 heads x 8 dims) or move the
  kernel to tensor cores (section 5, item 4); both halve q/acc registers and double warps/SM.
* **attn_prefill**: FA2's hdim-256 layout is inherently 128 accumulators/thread. The lever is
  keeping Q in registers only for the current k-slice (already the case) and letting V fragments be
  loaded per n-tile pair instead of per k-slice if ncu reports spills after changes.
* **Never index register arrays dynamically**: all fragment loops are `#pragma unroll` with
  compile-time trip counts; `cuobjdump -sass | grep -c "LDL\|STL"` must stay 0 (checked: 0).

## 4. CUDA Graphs

**Implemented.** Decode steps are captured once per batch-size bucket (1, 2, 4, 8, 12, 16, 24, 32,
48, 64; capped at `max_batch`) and replayed: ~530 kernels per step become one launch. Everything a
kernel needs that changes between steps lives in device memory at fixed addresses: tokens,
positions, slots, block tables, sampling parameters, RNG step counters, penalty history. The graph
itself contains the host-to-device copy of the step metadata (a memcpy node reading a fixed pinned
buffer), and the device-to-host copy of the sampled tokens. Split-KV attention picks its split
length on the device from the sequence length, so one graph serves every context length. Stream-K
tile counters reset themselves, so replays need no memsets. Padding rows (batch 5 in bucket 8) run a
private dummy sequence.

Next steps:

1. **Asynchronous stepping**: launch step t+1's graph before the host has consumed step t's
   tokens (tokens stay on the device; a small kernel builds the next metadata). Removes the host
   round trip (~20-50 us/step, 0.1-0.3%) and becomes necessary for MTP (section 5).
2. **Prefill graphs** for the fixed chunk size: saves ~1-2 ms of launch overhead per chunk
   (0.1%); low priority.
3. **Graph memory**: `cudaGraphInstantiate` per bucket costs a few MB; measure with `nvidia-smi`
   and drop unused buckets if VRAM is tight.
4. Ampere has no programmatic dependent launch (sm_90+), so inter-kernel bubbles are removed by
   fusion (section 5, item 5), not by PDL.

## 5. Larger optimizations, in order of expected value

1. **MTP speculative decoding** (single-stream decode x1.6-2.3). Qwen3.8 ships a one-layer MTP
   head (attention + MLP, ~0.42B params, ~0.25 GB at 5 bits). Draft 3 tokens, verify 4 in one
   batched step (the GEMMs are memory-bound, so verifying 4 tokens costs almost the same as 1).
   Required pieces: MTP weights in the converter, a verify step with M = 4 per sequence, and GDN
   state rollback for rejected drafts. Keep the recurrent state of the last accepted position by
   recording per-token inputs during verification and re-applying the accepted prefix (as NInfer's
   ReplaySSM does) instead of snapshotting 144 MB per draft position. NInfer measured 136-196 tok/s
   at batch 1 on a 5090 with MTP3 versus 84 without; the ratio carries to the 3090.
2. **INT8 tensor-core prefill** (prefill x2-3): quantize activations per token to int8 on the fly
   (smoothing the few outlier channels into the weights at conversion) and use
   `mma.sync.m16n8k32.s8.s8.s32`, 4x the fp32-accumulate bf16 rate on GA102. Weights stay 4/5-bit
   and are widened to int8 in registers exactly like today. Gate on the perplexity delta
   (docs/QUALITY.md 4.1).
3. **Chunked GDN prefill** (WY/UT transform, tensor cores): replaces 48 x T sequential steps
   with chunk-parallel matmuls; matters for prompts over ~32K tokens where the recurrence becomes
   ~5-8% of prefill.
4. **Tensor-core decode attention**: int8 K/V fed to `mma` via `ldmatrix` on byte quads with a
   consistent permutation of head dims (dot products are permutation-invariant), V stored
   transposed per page. Halves registers and makes long-context decode reach >= 85% of bandwidth.
5. **Fusion**: `gdn_ab` into `gdn_conv` (decode), attention split-merge into the o-projection's A
   loader, embedding gather into the first GEMM. ~2 us saved per removed kernel, ~100 kernels per
   step.
6. **fp16-accumulate MMA for prefill** (x2 tensor rate on GeForce): accumulate each 64-wide group
   in fp16, promote to fp32 per group. Needs the activation outlier audit first.
7. **Prefix caching**: reuse KV pages and GDN state snapshots for shared prompt prefixes (system
   prompts, multi-turn chat). GDN state must be snapshotted at the prefix boundary (144 MB), so
   store snapshots only at chunk boundaries of long shared prefixes.
