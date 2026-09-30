#!/usr/bin/env python3
"""Quantization error of real Qwen3.8 weights: the MSE-searched quantizer (tools/bqpack.py) vs plain absmax rounding.

Fetches individual tensors from the Hugging Face safetensors shards with HTTP range requests
(no full-shard download), folds the RMSNorm weights exactly like tools/convert.py, and reports the
relative Frobenius error ||W_q - W|| / ||W|| per tensor and bit width.

    python tools/eval/quant_study.py --layers 0,3 --cache testdata/real
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import bqpack as bq  # noqa: E402

REPO = "Qwen/Qwen3.8-27B"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
PREFIX = "model.language_model."


def http_range(url: str, start: int, end_inclusive: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_inclusive}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


class RemoteShards:
    def __init__(self, index: dict, cache: Path):
        self.weight_map = index["weight_map"]
        self.cache = cache
        self.headers: dict[str, tuple[int, dict]] = {}

    def header(self, shard: str):
        if shard not in self.headers:
            n = struct.unpack("<Q", http_range(BASE + shard, 0, 7))[0]
            meta = json.loads(http_range(BASE + shard, 8, 8 + n - 1))
            self.headers[shard] = (8 + n, meta)
        return self.headers[shard]

    def get(self, name: str) -> torch.Tensor:
        path = self.cache / (name + ".npy")
        if path.exists():
            return torch.from_numpy(np.load(path)).view(torch.bfloat16)
        shard = self.weight_map[name]
        base, meta = self.header(shard)
        info = meta[name]
        assert info["dtype"] == "BF16", info
        b, e = info["data_offsets"]
        raw = http_range(BASE + shard, base + b, base + e - 1)
        arr = np.frombuffer(raw, dtype="<i2").reshape(info["shape"]).copy()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
        return torch.from_numpy(arr).view(torch.bfloat16)


def absmax_quant(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Plain group-64 absmax rounding with an fp16 scale (the method of the published Q4/Q5 artifact)."""
    qmax = (1 << (bits - 1)) - 1
    g = w.reshape(-1, 64)
    s = (g.abs().amax(dim=1) / qmax).to(torch.float16).to(torch.float32)
    safe = torch.where(s > 0, s, torch.ones_like(s))
    c = torch.clamp(torch.round(g / safe[:, None]), -qmax - (1 if bits == 8 else 1), qmax)
    return (c * s[:, None]).reshape(w.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--cache", default=str(ROOT / "testdata/real"))
    ap.add_argument("--index", default=str(ROOT / "testdata/qwen3.8/model.safetensors.index.json"))
    ap.add_argument("--bits", default="4,5,6,8")
    args = ap.parse_args()
    shards = RemoteShards(json.loads(Path(args.index).read_text()), Path(args.cache))
    cfg = json.loads((ROOT / "testdata/qwen3.8/config.json").read_text())["text_config"]
    bits_list = [int(b) for b in args.bits.split(",")]
    rows = []
    for layer in [int(x) for x in args.layers.split(",")]:
        p = f"{PREFIX}layers.{layer}."
        g_in = 1.0 + shards.get(p + "input_layernorm.weight").float()
        g_post = 1.0 + shards.get(p + "post_attention_layernorm.weight").float()
        tensors = {}
        if cfg["layer_types"][layer] == "full_attention":
            tensors["attn.q_proj(+gate)"] = shards.get(p + "self_attn.q_proj.weight").float() * g_in
            tensors["attn.k_proj"] = shards.get(p + "self_attn.k_proj.weight").float() * g_in
            tensors["attn.v_proj"] = shards.get(p + "self_attn.v_proj.weight").float() * g_in
            tensors["attn.o_proj"] = shards.get(p + "self_attn.o_proj.weight").float()
        else:
            tensors["gdn.in_proj_qkv"] = shards.get(p + "linear_attn.in_proj_qkv.weight").float() * g_in
            tensors["gdn.in_proj_z"] = shards.get(p + "linear_attn.in_proj_z.weight").float() * g_in
            tensors["gdn.out_proj"] = shards.get(p + "linear_attn.out_proj.weight").float()
        tensors["mlp.gate_proj"] = shards.get(p + "mlp.gate_proj.weight").float() * g_post
        tensors["mlp.up_proj"] = shards.get(p + "mlp.up_proj.weight").float() * g_post
        tensors["mlp.down_proj"] = shards.get(p + "mlp.down_proj.weight").float()
        for name, w in tensors.items():
            norm = w.norm().item()
            for bits in bits_list:
                t0 = time.time()
                codes, scales = bq.quantize_matrix(w, bits)
                t_q = time.time() - t0
                ours = (bq.dequant_codes(codes, scales) - w).norm().item() / norm
                base = (absmax_quant(w, bits) - w).norm().item() / norm
                rows.append((layer, name, bits, base, ours, t_q, w.numel()))
                print(f"L{layer:<2d} {name:22s} bq{bits}: absmax {base:.4f}  mse-search {ours:.4f}  "
                      f"(error {100 * (ours / base - 1):+.1f}%)  {w.numel() / t_q / 1e6:.1f} Mw/s", flush=True)
    # summary per bit width (parameter-weighted mean of relative errors)
    print("\nweighted mean relative Frobenius error")
    for bits in bits_list:
        sel = [r for r in rows if r[2] == bits]
        wsum = sum(r[6] for r in sel)
        base = sum(r[3] * r[6] for r in sel) / wsum
        ours = sum(r[4] * r[6] for r in sel) / wsum
        print(f"  bq{bits}: absmax {base:.4f}  mse-search {ours:.4f}  (error {100 * (ours / base - 1):+.1f}%)")


if __name__ == "__main__":
    main()
