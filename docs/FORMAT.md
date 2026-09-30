# Storage formats

This file is the contract between the converter (`tools/convert.py`), the CPU reference
(`src/ref/`), the CUDA kernels (`src/kernels/`) and the tokenizer. If code and this file
disagree, the code has a bug.

## 1. Model file (`*.q38`)

```
offset 0      8 bytes   magic "QW38MDL1"
offset 8      8 bytes   u64 little-endian: length L of the JSON header
offset 16     L bytes   JSON header (UTF-8)
              padding   zero bytes up to the next multiple of 4096
data_start    ...       object payloads; every object offset is relative to data_start
                        and aligned to 256 bytes
```

JSON header:

```json
{
  "format": "qwen38-model", "format_version": 1,
  "config": { ... model configuration, see 1.1 ... },
  "recipe": { "name": "default", "tensors": { "<name>": "q5", ... } },
  "tensors": { "<name>": <tensor record>, ... },
  "resources": { "tokenizer": { "offset": 0, "bytes": 0 } },
  "provenance": { "source": "Qwen/Qwen3.8-27B", "revision": "...", "converter": "..." }
}
```

### 1.1 `config`

| key | Qwen3.8-27B | meaning |
|---|---|---|
| `hidden_size` | 5120 | H |
| `num_layers` | 64 | |
| `layer_types` | `["linear_attention" x3, "full_attention", ...]` | per layer |
| `vocab_size` | 248320 | embedding / LM-head rows (padded vocabulary) |
| `intermediate_size` | 17408 | dense MLP width |
| `num_attention_heads` | 24 | Q heads |
| `num_key_value_heads` | 4 | KV heads |
| `head_dim` | 256 | |
| `rotary_dim` | 64 | partial RoPE width (`head_dim * partial_rotary_factor`) |
| `rope_theta` | 1e7 | |
| `linear_num_key_heads` | 16 | GDN Q/K heads |
| `linear_num_value_heads` | 48 | GDN V heads |
| `linear_key_head_dim` | 128 | |
| `linear_value_head_dim` | 128 | |
| `linear_conv_kernel_dim` | 4 | |
| `rms_norm_eps` | 1e-6 | |
| `max_position_embeddings` | 262144 | |
| `norm_folded` | true | input/post-attention/final RMSNorm weights are folded into the following projections (1.3) |
| `eos_token_ids` | `[248046, 248044]` | |

### 1.2 Tensor record

```json
{ "dtype": "bq4" | "bq5" | "bq6" | "bq8" | "bf16" | "f32",
  "shape": [N, K],                    // logical shape, row-major, N = output rows
  "offset": 0, "bytes": 0,            // bf16 / f32: one contiguous little-endian array
  "planes": { "lo": [off, bytes], "hi": [off, bytes], "scales": [off, bytes] },  // bq* only
  "group_size": 64 }                  // bq* only
```

`bq*` tensors require `K % 64 == 0` and `N % 128 == 0`. The converter pads `N` with zero
rows (codes 0, scales 0) and records the logical row count in `"rows"`.

### 1.3 Tensor names and semantics

`L` is a layer index. Weights are `[out, in]` like `torch.nn.Linear`.

| name | dtype (default recipe) | shape | content |
|---|---|---|---|
| `embed` | bf16 | [V, H] | `embed_tokens` |
| `L.attn.qkvg` | bq5 | [14336, H] | rows `[q: 24x256 | gate: 24x256 | k: 4x256 | v: 4x256]`, input norm folded |
| `L.attn.o` | bq5 | [H, 6144] | `o_proj` |
| `L.attn.q_norm`, `L.attn.k_norm` | f32 | [256] | **`1 + w`** (offset RMSNorm multiplier) |
| `L.gdn.qkvz` | bq5 | [16384, H] | rows `[q 2048 | k 2048 | v 6144 | z 6144]`, input norm folded |
| `L.gdn.ab` | bf16 | [96, H] | rows `[b 48 | a 48]`, input norm folded (small; computed by the GDN gate kernel) |
| `L.gdn.conv` | f32 | [10240, 4] | depthwise causal conv taps over `[q | k | v]` channels |
| `L.gdn.A` | f32 | [48] | `-exp(A_log)` |
| `L.gdn.dt_bias` | f32 | [48] | |
| `L.gdn.norm` | f32 | [128] | gated RMSNorm weight (plain, **not** `1 + w`) |
| `L.gdn.out` | bq5 | [H, 6144] | `out_proj` |
| `L.mlp.gate_up` | bq4 | [2 x 17408, H] | row `2r` = `gate_proj[r]`, row `2r+1` = `up_proj[r]`; post-attention norm folded |
| `L.mlp.down` | bq5 | [H, 17408] | `down_proj` |
| `lm_head` | bq8 | [V, H] | final norm folded |

Norm folding: for a projection `W` that consumes `offset_rmsnorm(x, w)`, the file stores
`W'[n, k] = W[n, k] * (1 + w[k])` (computed in fp32 before quantization). The runtime computes
`y = rsqrt(mean(x^2) + eps) * (bf16(x) @ W'^T)`.

## 2. BQ weight format (`bq4`, `bq5`, `bq6`, `bq8`)

### 2.1 Numbers

Group size G = 64 along K. For bit width b, each weight has a signed code
`c in [-2^(b-1), 2^(b-1) - 1]` and each (row n, group g = k / 64) has a bf16 scale `s[n][g]`:

```
W[n][k] = bf16( float(c[n][k]) * float(s[n][k / 64]) )
```

The multiply result is rounded to bf16 (round-to-nearest-even); this is exactly what the kernels
compute (`hmul2` on bf16 pairs). The stored unsigned code is `u = c + 2^(b-1)`, `0 <= u < 2^b`.
`lo = u & 15`, `hi = u >> 4` (`b - 4` bits, absent for b = 4).

### 2.2 Slabs and lanes

The matrix is cut into slabs of 16 (k) x 64 (n): slab `(kb, nb)` covers
`k in [16 kb, 16 kb + 16)`, `n in [64 nb, 64 nb + 64)`. Slabs are stored `nb`-major:

```
slab_index(kb, nb) = nb * (K / 16) + kb
```

Inside a slab, lane `L in [0, 32)` (one warp) owns 32 codes. With `g = L >> 2`, `t = L & 3`,
the lane owns, for n8 sub-tile `j in [0, 8)` (column `n = 64 nb + 8 j + g`), the four rows

```
k in { 16 kb + 2t, 16 kb + 2t + 1, 16 kb + 2t + 8, 16 kb + 2t + 9 }
```

which is exactly the B-operand fragment of `mma.sync.m16n8k16` for sub-tile `j`.

The lane's 32 codes are organised as 16 **pairs** `q in [0, 16)`: `j = q >> 1`,
`h = q & 1`; pair q is `(k0, k0 + 1)` with `k0 = 16 kb + 2t + 8h`, column `n = 64 nb + 8 j + g`.
The first element of a pair goes to the low 16 bits of a bf16x2 register, the second to the high.

### 2.3 `lo` plane (all b)

Per slab 512 bytes; lane L's 16 bytes at `16 L`, as four little-endian u32 words
`W[0..3]`. Pair q lives in word `w = q >> 2`, position `p = q & 3`:

```
bits [4p, 4p + 4)       = lo of the first element of pair q
bits [16 + 4p, 20 + 4p) = lo of the second element of pair q
```

Extraction: `((W[w] >> 4p) & 0x000F000F) | 0x43004300` is the bf16x2 `(128 + lo0, 128 + lo1)`.

### 2.4 `hi` plane

* **b = 4**: absent.
* **b = 5**: per slab 128 bytes, lane L's u32 `H` at `4 L`. Bit `q` = hi bit of the first
  element of pair q, bit `16 + q` = hi bit of the second.
* **b = 6**: per slab 256 bytes, lane L's two u32 `H[0..1]` at `8 L`. Pair q uses word
  `q >> 3`, `r = q & 7`: bits `[2r, 2r + 2)` = hi bits of the first element,
  bits `[16 + 2r, 18 + 2r)` = hi bits of the second.
* **b = 8**: a second nibble plane with exactly the `lo` layout, holding `hi = u >> 4`.

For b = 5 and 6 the hi bits are moved to mantissa bits 4.. of the bf16 halves:
`v = lo_ext | (shift(H, to bit 4) & mask)` with mask `0x00100010` (b=5) or `0x00300030` (b=6),
giving `(128 + u0, 128 + u1)`; then `c = v - (128 + 2^(b-1))` exactly.
For b = 8: `c = 16 * ((128 + hi) - 136) + ((128 + lo) - 128)`, exact in bf16.

Slab order and slab size define plane offsets: `hi` slab bytes are 0 / 128 / 256 / 512.

### 2.5 `scales` plane

bf16, blocks of 64 columns per (nb, group):

```
block_index(nb, g) = nb * (K / 64) + g          // 128 bytes per block
position of column n_local = 8 j + g_lane  ->  g_lane * 8 + j
```

so lane L reads its eight scales (j = 0..7) as one 16-byte load at `block + 16 * (L >> 2)`.

### 2.6 Planes in the file

`lo`, `hi` (if any) and `scales` are separate byte ranges, each 256-byte aligned.
Sizes: lo `N*K/2`, hi `N*K*(b-4)/8`, scales `N*(K/64)*2` bytes.

### 2.7 Quantizer (default)

Per (row, group) the converter picks the scale minimising the squared error over a grid of
candidates, including the signed-extreme mapping `s = x_ext / -2^(b-1)` (the value with the
largest magnitude maps onto the negative end, as llama.cpp Q4_0) and `s = amax / (2^(b-1) - 1)`.
The chosen scale is rounded to bf16 first; codes are `clamp(rint(x / s_bf16))` against that stored
scale, so scale rounding introduces no bias.

## 3. Runtime layouts (device memory, not in the file)

### 3.1 KV cache (full-attention layers only)

Page = 64 tokens. One pool per engine; page ids are shared by all attention layers.
Per attention layer and page:

| dtype | K | V | K scales | V scales |
|---|---|---|---|---|
| int8 | `[kv_head][64 tok][256] i8` | same | `[kv_head][64 tok][4] f16` | same |
| bf16 | `[kv_head][64 tok][256] bf16` | same | none | none |

INT8 quantization is symmetric per (token, head, 64-dim group): `scale = amax / 127`,
`q = rint(x / scale)`; K is stored after q/k-norm and RoPE, V as projected.

### 3.2 GDN state (per sequence slot)

* recurrent: `f32 [layer 48][v_head 48][dv 128][dk 128]` (`S[dv][dk]`, i.e. transposed w.r.t.
  Hugging Face's `[dk][dv]`)
* conv: `bf16 [layer 48][tap 3][channel 10240]`, tap 0 = oldest

### 3.3 Activation conventions

* residual stream: f32 `[M][H]`
* `xbf`: bf16 copy of the residual, the A operand of every normed projection
* norm partials: f32 `[M][H / 128]`, sum of squares of the residual over each 128-column block
