#!/usr/bin/env python3
"""End-to-end check of converter + file format + CPU reference model against Hugging Face.

1. Builds the tiny random Qwen3.5-architecture model (tools/make_tiny_model.py) if missing.
2. Converts it with tools/convert.py (default recipe: 4/5/8-bit BQ weights).
3. Builds the "HF-equivalent" model: the same HF module, loaded with exactly the dequantized
   weights from the .q38 file and the folded norms set to identity.
4. Runs HF (fp32, eager attention) and `qwen38-ref --exact` on the same tokens: these must agree
   to fp32 rounding, which validates the packing, the norm folding, the tensor mapping and all
   of the model math (gated attention, partial RoPE, GDN conv/gates/recurrence/gated norm, MLP).
5. Runs `qwen38-ref` in engine mode (bf16 activations, int8 KV) and reports how far the engine's
   rounding points move the logits, plus the quantization effect vs the original weights.

    python tests/host/test_ref_vs_hf.py --qwen38-ref build-host/src/qwen38-ref
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from q38file import Q38File  # noqa: E402
from transformers import Qwen3_5ForCausalLM  # noqa: E402


def run(cmd: list[str]) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


@torch.no_grad()
def hf_equivalent(model_dir: Path, mf: Q38File) -> Qwen3_5ForCausalLM:
    m = Qwen3_5ForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager")
    c = mf.config
    H, nq, nkv, hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    lk, lv, dk, dv = (c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"],
                      c["linear_value_head_dim"])
    conv_dim = 2 * lk * dk + lv * dv
    for i, layer in enumerate(m.model.layers):
        layer.input_layernorm.weight.zero_()
        layer.post_attention_layernorm.weight.zero_()
        if c["layer_types"][i] == "full_attention":
            w = mf.dequant(f"{i}.attn.qkvg")
            q, gate, k, v = torch.split(w, [nq * hd, nq * hd, nkv * hd, nkv * hd])
            q_proj = torch.stack([q.view(nq, hd, H), gate.view(nq, hd, H)], dim=1).reshape(2 * nq * hd, H)
            a = layer.self_attn
            a.q_proj.weight.copy_(q_proj)
            a.k_proj.weight.copy_(k)
            a.v_proj.weight.copy_(v)
            a.o_proj.weight.copy_(mf.dequant(f"{i}.attn.o"))
            a.q_norm.weight.copy_(mf.direct(f"{i}.attn.q_norm") - 1.0)
            a.k_norm.weight.copy_(mf.direct(f"{i}.attn.k_norm") - 1.0)
        else:
            g = layer.linear_attn
            w = mf.dequant(f"{i}.gdn.qkvz")
            g.in_proj_qkv.weight.copy_(w[:conv_dim])
            g.in_proj_z.weight.copy_(w[conv_dim:])
            ab = mf.direct(f"{i}.gdn.ab").to(torch.float32)
            g.in_proj_b.weight.copy_(ab[:lv])
            g.in_proj_a.weight.copy_(ab[lv:2 * lv])
            g.out_proj.weight.copy_(mf.dequant(f"{i}.gdn.out"))
            # stored exactly: check instead of copying
            assert torch.equal(mf.direct(f"{i}.gdn.conv"), g.conv1d.weight.reshape(conv_dim, -1))
            assert torch.equal(mf.direct(f"{i}.gdn.A"), -torch.exp(g.A_log.float()))
            assert torch.equal(mf.direct(f"{i}.gdn.dt_bias"), g.dt_bias.float())
            assert torch.equal(mf.direct(f"{i}.gdn.norm"), g.norm.weight.float())
        gu = mf.dequant(f"{i}.mlp.gate_up")
        layer.mlp.gate_proj.weight.copy_(gu[0::2])
        layer.mlp.up_proj.weight.copy_(gu[1::2])
        layer.mlp.down_proj.weight.copy_(mf.dequant(f"{i}.mlp.down"))
    m.model.norm.weight.zero_()
    m.lm_head.weight.copy_(mf.dequant("lm_head"))
    emb = mf.direct("embed").to(torch.float32)
    assert torch.equal(emb, m.model.embed_tokens.weight.to(torch.bfloat16).to(torch.float32))
    m.model.embed_tokens.weight.copy_(emb)
    return m.eval()


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: [T, V] logits. KL(softmax(b) || softmax(a)) per token, top-1 agreement."""
    diff = np.abs(a - b)
    la = torch.log_softmax(torch.from_numpy(a).double(), -1)
    lb = torch.log_softmax(torch.from_numpy(b).double(), -1)
    kl = (lb.exp() * (lb - la)).sum(-1)
    return {
        "max_abs": float(diff.max()),
        "rel_max": float(diff.max() / np.abs(b).max()),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "top1": float((a.argmax(-1) == b.argmax(-1)).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen38-ref", default=str(ROOT / "build-host/src/qwen38-ref"))
    ap.add_argument("--workdir", default=str(ROOT / "testdata/tiny"))
    ap.add_argument("--recipe", default="default")
    ap.add_argument("--tokens", type=int, default=150)
    args = ap.parse_args()

    work = Path(args.workdir)
    model_dir = work / "hf"
    if not (model_dir / "config.json").exists():
        run([sys.executable, str(ROOT / "tools/make_tiny_model.py"), "--out", str(model_dir)])
    q38_path = work / f"tiny-{args.recipe}.q38"
    run([sys.executable, str(ROOT / "tools/convert.py"), "--model", str(model_dir), "--out", str(q38_path),
         "--recipe", args.recipe, "--no-tokenizer", "--device", "cpu"])

    rng = np.random.default_rng(7)
    tokens = rng.integers(0, 1000, size=args.tokens).astype("<i4")
    tok_path = work / "tokens.i32"
    tokens.tofile(tok_path)

    mf = Q38File(q38_path)
    V = mf.config["vocab_size"]
    with torch.no_grad():
        hf_eq = hf_equivalent(model_dir, mf)
        ref_logits = hf_eq(torch.from_numpy(tokens.astype(np.int64))[None]).logits[0].numpy()
        orig = Qwen3_5ForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager")
        orig_logits = orig(torch.from_numpy(tokens.astype(np.int64))[None]).logits[0].numpy()

    results = {}
    for mode, extra in (("exact", ["--exact", "--kv", "f32"]), ("engine-int8kv", ["--kv", "int8"]),
                        ("engine-bf16kv", ["--kv", "bf16"])):
        out = work / f"logits-{mode}.f32"
        run([args.qwen38_ref, "--model", str(q38_path), "--tokens", str(tok_path), "--out", str(out),
             "--all-logits", *extra])
        got = np.fromfile(out, dtype="<f4").reshape(-1, V)
        results[mode] = stats(got, ref_logits)
    results["quantization (hf-equivalent vs original)"] = stats(ref_logits, orig_logits)

    for k, v in results.items():
        print(f"{k:45s} " + "  ".join(f"{a}={b:.3g}" for a, b in v.items()))

    ok = True
    ex = results["exact"]
    if not (ex["rel_max"] < 2e-4 and ex["top1"] == 1.0):
        print("FAIL: exact reference does not match Hugging Face")
        ok = False
    for mode in ("engine-int8kv", "engine-bf16kv"):
        if not (results[mode]["top1"] >= 0.95 and results[mode]["kl_mean"] < 5e-3):
            print(f"FAIL: {mode} deviates too much from the HF-equivalent model")
            ok = False
    print("PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
