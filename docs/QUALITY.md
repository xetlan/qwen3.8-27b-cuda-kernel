# Quantization quality

The prompt asks for a 4-bit format that "mathematically preserves at least 96% of the base
model's intelligence on standard coding benchmarks", never below Q3. This document says what can
and cannot be claimed about that, what the default recipe is and why, and how to measure the 96%
on the RTX 3090.

## 1. What "mathematically preserves" can mean

No quantization format *mathematically guarantees* a benchmark score. Rounding error per weight
is bounded (for BQ: at most half a quantization step, a step being `|scale|`), and one can bound
the error of a single matrix product, but a 64-layer network sampled for thousands of reasoning
tokens is not a function whose benchmark accuracy can be bounded from weight error. Benchmark
retention is an **empirical** property of (model, format, bit allocation, quantizer, kernels,
sampling settings). What this project provides instead:

1. a format and bit allocation chosen on published evidence for this exact model (section 3);
2. a quantizer that is better than the one behind that evidence (section 2);
3. exactness guarantees for everything *except* the chosen rounding: the kernels compute exactly
   `bf16(code * scale)` weights, verified bit-for-bit against the format specification, and the
   whole layer stack is verified against Hugging Face transformers (tests/host/test_ref_vs_hf.py);
4. a measurement protocol with a pass/fail threshold (section 4), to be run on the 3090.

## 2. Format: BQ group-wise integer, 4/5/6/8 bits

* Symmetric signed codes, **group size 64** along the input dimension, one **bf16 scale** per
  group (4.25 / 5.25 / 6.25 / 8.25 bits per weight). The same idea as GPTQ/AWQ "W4A16 g64", in a
  layout pre-permuted for the Ampere tensor-core fragments (docs/FORMAT.md).
* 5/6-bit codes add a separate high-bit plane, which is what makes an EXL2-style *mixed* bit
  allocation possible without giving up the one-16-byte-load-per-lane fast path of 4-bit.
* **Quantizer**: per group, an MSE search over 48 candidate scales (the signed extreme mapped to
  either end of the code range, stretched over [0.80, 1.06]), each followed by a least-squares
  refit of the scale for the chosen codes; the scale is then rounded to bf16 and the codes are
  recomputed against the rounded scale. On real weights it has 16% lower error than the plain
  absmax rounding used by the reference evaluation below (table).
* **Norm folding**: the offset-RMSNorm weights `(1 + w)` are multiplied into the following
  projection before quantization (the standard preprocessing of QuaRot/SpinQuant). Channels that
  the norm amplifies get proportionally finer steps, the AWQ direction.

Measured on real Qwen3.8-27B weights (layer 0: GDN + MLP, layer 3: attention + MLP, fetched with
`tools/eval/quant_study.py`), relative Frobenius error `||W_q - W|| / ||W||` after norm folding,
parameter-weighted over the 12 matrices:

| bits | plain absmax (fp16 scale) | MSE search, this repo (bf16 scale) | change |
|---:|---:|---:|---:|
| 4 | 0.1107 | 0.0929 | -16.1% |
| 5 | 0.0516 | 0.0465 | -10.0% |
| 6 | 0.0250 | 0.0231 | -7.4% |
| 8 | 0.0061 | 0.0059 | -3.4% |

The spread across individual tensors is small (4-bit: -15.8% to -16.4%), so the gain is not
carried by a few matrices. For the projections the published artifact kept at 4 bits and this recipe
keeps at 5 (attention and GDN q/k), the error drops from 0.111-0.118 to 0.047-0.050.

What is *not* done (candidates if measurement shows a gap, section 5): activation-aware scaling
(AWQ), Hessian-based error compensation (GPTQ), importance-weighted MSE from a calibration set.
All of them fit the same file format, so they are converter-only changes.

## 3. Bit allocation (the `default` recipe) and the evidence for it

| projection | bits | why |
|---|---:|---|
| MLP gate/up | 4 | 42% of all weights; feeds SiLU(gate) * up, errors are damped |
| MLP down | 5 | writes straight into the residual stream |
| attention q/gate/k/v (fused) | 5 | |
| attention o | 5 | residual stream |
| GDN q/k/v/z (fused) | 5 | v and z are on the value path |
| GDN out | 5 | residual stream |
| GDN a/b gate projections | bf16 | 96 rows per layer; they drive exp/softplus decay gates |
| LM head | 8 | one output per vocabulary entry; top-k/top-p see its errors directly |
| embedding | bf16 (host) | exact; costs no VRAM |

Average over the decoder body: **~4.64 bits/weight**, 14.81 GiB on the GPU.

**Published evidence for this allocation.** The NInfer project published evaluations of
Qwen3.8-27B converted with a group-64 absmax Q4/Q5 recipe (Q4: attention q/k, GDN q/k, MLP
gate/up; Q5: everything else; Q8 group-32 embedding and head), served with thinking enabled and
an INT8 group-64 KV cache, i.e. the same KV format this engine uses by default. Their numbers, single
sample, next to the official BF16 figures from the Qwen model card (the protocols are not
identical):

| benchmark | Q4/Q5 artifact | official BF16 | ratio |
|---|---:|---:|---:|
| GPQA-Diamond | 87.37 | 89.2 | 97.9% |
| IFBench (prompt, strict) | 77.67 | 79.5 | 97.7% |
| RealWorldQA (vision) | 82.22 | 85.9 | 95.7% |
| ERQA (vision) | 66.25 | 65.5 | 101.1% |
| AIME 2025 / 2026 | 96.67 / 96.67 | n/a | |

The default recipe differs from that artifact only in the direction of more precision: q/k at 5
instead of 4 bits, a quantizer with 10-16% lower weight error at the same bit width (section 2), a
bf16 embedding; the head is 8-bit with group 64 instead of 32 (a negligible difference at 8 bits). That makes the published numbers *evidence* for the
default recipe, not proof: there is no published coding benchmark for it, and the one vision
result under 96% is a reminder that single-sample scores are noisy at this level.

Recipes: `default` (above), `lean` (everything 4-bit, 6-bit head: 13.01 GiB, fastest, unvalidated),
`quality` (6-bit down/o/out: 15.71 GiB). Nothing below 4 bits exists, so the "never below Q3"
constraint holds by construction.

## 4. Measuring the 96% on the 3090

The official BF16 coding scores to compare against (Qwen3.8-27B model card, thinking mode):
**LiveCodeBench v6 90.3**, SWE-bench Pro 61.7, Terminal-Bench 2.1 73.0. 96% of LCB v6 is **86.7**.

### 4.1 Fast proxy (minutes): perplexity across recipes

```
./build/src/qwen38 --model models/qwen3.8-27b-quality.q38 --score code.txt --score-ctx 4096
./build/src/qwen38 --model models/qwen3.8-27b.q38         --score code.txt --score-ctx 4096
./build/src/qwen38 --model models/qwen3.8-27b-lean.q38    --score code.txt --score-ctx 4096
```

Use a code corpus the model has not memorised (for example recent commits of a repository created
after the model's training cutoff, ~200K tokens). Rule of thumb from the quantization literature:
a perplexity increase under ~1% relative to a near-lossless reference goes with benchmark losses
within noise; above ~3% expect measurable losses. The `quality` recipe is the closest thing to a
reference that fits in 24 GB; a true BF16 reference needs a larger GPU (vLLM or transformers) or
~64 GB of system RAM for transformers on the CPU.

### 4.2 The gate: LiveCodeBench v6 with the official sampling settings

`qwen38 --batch` turns any benchmark into files: one JSON request per line in, one completion per
line out, with continuous batching for throughput. `tools/eval/humaneval_plus.py` is a complete
worked example (HumanEval+ via `evalplus`); LiveCodeBench works the same way with its
`lcb_runner` generation/evaluation split:

1. export the LCB v6 problems (release `v6`, the window the model card reports) as JSONL prompts in
   LCB's own prompt format, with `temperature 1.0, top_p 0.95, top_k 20`, thinking on, and a
   generous `max_tokens` (the model card uses long reasoning budgets; start with 32768);
2. `qwen38 --model ... --batch lcb.jsonl --out lcb_out.jsonl --max-batch 8`;
3. convert `lcb_out.jsonl` to `lcb_runner`'s custom-output format and run its evaluator.

**Pass criterion: pass@1 >= 86.7.** Because the 3090 cannot run the BF16 model, the comparison
is against the published 90.3; if the reproduction protocol differs from Qwen's, run the same
protocol on the `quality` recipe as a second reference and require the default recipe to be within
4% of it as well.

Budget: LCB v6 with long reasoning is expensive. At ~45 tok/s single stream, or ~250 tok/s
aggregate at batch 8, 175 problems x ~15K tokens is ~3 hours at batch 8. Run it once per recipe
change, not per build.

## 5. If the default recipe misses the bar

In order of cost:
1. raise the most sensitive tensors by one bit (`--bits mlp.down=6`, `--bits gdn.qkvz=6`) and
   measure again; the dry run (`tools/convert.py --dry-run`) prints the VRAM cost;
2. per-layer allocation: the first and last layers are usually the most sensitive
   (`--bits 0.mlp.down=8 --bits 63.mlp.down=8`);
3. calibration-based quantizers (AWQ scaling, GPTQ) in the converter: same file format, no engine
   change;
4. bf16 KV cache (`--kv bf16`) to separate KV quantization error from weight error.
