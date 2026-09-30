#!/usr/bin/env python3
"""Convert a Hugging Face Qwen3.5-architecture checkpoint (e.g. Qwen/Qwen3.8-27B) to a .q38 file.

    python tools/convert.py --model /path/to/Qwen3.8-27B --out models/qwen3.8-27b.q38

The layout of the output is specified in docs/FORMAT.md. Conversion streams one tensor at a
time (peak host RAM is a few GB) and quantizes on the GPU when one is available.

Recipes pick the BQ bit width per projection. `default` keeps the value path (attention gate/v/o,
GDN v/z/out, MLP down) at 5 bits, the MLP gate/up at 4 bits and the LM head at 8 bits; the GDN
gate projections (96 rows per layer) stay bf16. Override single tensors with --bits, e.g.
`--bits mlp.down=6 --bits 63.mlp.down=8`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bqpack as bq  # noqa: E402

MAGIC = b"QW38MDL1"
ALIGN = 256
HEADER_ALIGN = 4096

RECIPES: dict[str, dict[str, int]] = {
    # ~4.6 bits/weight in the decoder body. Value-path projections at 5 bits follow the
    # Q4/Q5 split that has published evaluations on Qwen3.8-27B (see docs/QUALITY.md).
    # The GDN gate projections (a, b; 96 rows per layer) always stay bf16.
    "default": {"attn.qkvg": 5, "attn.o": 5, "gdn.qkvz": 5, "gdn.out": 5,
                "mlp.gate_up": 4, "mlp.down": 5, "lm_head": 8},
    # everything 4-bit except the gates and a 6-bit head: fastest decode, most KV room,
    # quality not validated
    "lean": {"attn.qkvg": 4, "attn.o": 4, "gdn.qkvz": 4, "gdn.out": 4,
             "mlp.gate_up": 4, "mlp.down": 4, "lm_head": 6},
    # more bits on the value path; ~0.9 GB larger than default
    "quality": {"attn.qkvg": 5, "attn.o": 6, "gdn.qkvz": 5, "gdn.out": 6,
                "mlp.gate_up": 4, "mlp.down": 6, "lm_head": 8},
    # near-lossless, only for small test models (27B would not fit in 24 GB)
    "q8": {"attn.qkvg": 8, "attn.o": 8, "gdn.qkvz": 8, "gdn.out": 8,
           "mlp.gate_up": 8, "mlp.down": 8, "lm_head": 8},
}


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


# ---------------------------------------------------------------------------------------------
# Source checkpoint
# ---------------------------------------------------------------------------------------------

class Source:
    """Reads tensors from a (sharded) safetensors checkpoint by logical HF name."""

    def __init__(self, model_dir: Path):
        self.dir = model_dir
        index = model_dir / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
        else:
            files = sorted(model_dir.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"no safetensors files in {model_dir}")
            weight_map = {}
            for f in files:
                with safe_open(f, "pt") as h:
                    for k in h.keys():
                        weight_map[k] = f.name
        self.weight_map = weight_map
        self._handles: dict[str, object] = {}
        # Qwen3_5ForConditionalGeneration nests the text model; Qwen3_5ForCausalLM does not.
        if any(k.startswith("model.language_model.") for k in weight_map):
            self.prefix = "model.language_model."
        else:
            self.prefix = "model."

    def _handle(self, fname: str):
        if fname not in self._handles:
            self._handles[fname] = safe_open(self.dir / fname, "pt")
        return self._handles[fname]

    def get(self, name: str) -> torch.Tensor:
        """name relative to the text model, e.g. 'layers.0.mlp.down_proj.weight'; or 'lm_head.weight'."""
        full = name if name.startswith("lm_head") else self.prefix + name
        if full not in self.weight_map:
            raise KeyError(f"tensor {full} not found in checkpoint")
        return self._handle(self.weight_map[full]).get_tensor(full)


def text_config(model_dir: Path) -> dict:
    raw = json.loads((model_dir / "config.json").read_text())
    tc = raw.get("text_config", raw)
    rope = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
    partial = rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25))
    theta = rope.get("rope_theta", tc.get("rope_theta", 10_000_000))
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("only default (unscaled) RoPE is supported; YaRN is a runtime follow-up")
    n_layers = tc["num_hidden_layers"]
    layer_types = tc.get("layer_types")
    if layer_types is None:
        interval = tc.get("full_attention_interval", 4)
        layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                       for i in range(n_layers)]
    head_dim = tc["head_dim"]
    eos = []
    gen = model_dir / "generation_config.json"
    if gen.exists():
        g = json.loads(gen.read_text()).get("eos_token_id", [])
        eos = g if isinstance(g, list) else [g]
    if not eos:
        e = tc.get("eos_token_id", [])
        eos = e if isinstance(e, list) else [e]
    if tc.get("hidden_act", "silu") != "silu" or tc.get("attention_bias", False):
        raise ValueError("unsupported activation or attention bias")
    if tc.get("attn_output_gate", True) is not True:
        raise ValueError("expected gated attention (attn_output_gate=true)")
    return {
        "architecture": "qwen3_5_dense",
        "hidden_size": tc["hidden_size"],
        "num_layers": n_layers,
        "layer_types": layer_types,
        "vocab_size": tc["vocab_size"],
        "intermediate_size": tc["intermediate_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": head_dim,
        "rotary_dim": int(head_dim * partial),
        "rope_theta": float(theta),
        "linear_num_key_heads": tc["linear_num_key_heads"],
        "linear_num_value_heads": tc["linear_num_value_heads"],
        "linear_key_head_dim": tc["linear_key_head_dim"],
        "linear_value_head_dim": tc["linear_value_head_dim"],
        "linear_conv_kernel_dim": tc["linear_conv_kernel_dim"],
        "rms_norm_eps": float(tc.get("rms_norm_eps", 1e-6)),
        "max_position_embeddings": tc["max_position_embeddings"],
        "tie_word_embeddings": bool(raw.get("tie_word_embeddings", tc.get("tie_word_embeddings", False))),
        "norm_folded": True,
        "eos_token_ids": eos,
    }


# ---------------------------------------------------------------------------------------------
# Planned objects
# ---------------------------------------------------------------------------------------------

@dataclass
class Obj:
    name: str
    dtype: str                  # bq4/bq5/bq6/bq8/bf16/f32/bytes
    shape: list[int]
    produce: Callable[[], object]
    rows: int | None = None     # logical rows for padded bq tensors
    record: dict = field(default_factory=dict)

    def sizes(self) -> list[int]:
        if self.dtype.startswith("bq"):
            bits = int(self.dtype[2:])
            return list(bq.plane_sizes(self.shape[0], self.shape[1], bits))
        if self.dtype == "bf16":
            return [math.prod(self.shape) * 2]
        if self.dtype == "f32":
            return [math.prod(self.shape) * 4]
        raise ValueError(self.dtype)


class Converter:
    def __init__(self, args):
        self.args = args
        self.model_dir = Path(args.model)
        self.src = Source(self.model_dir)
        self.cfg = text_config(self.model_dir)
        if args.layers is not None:
            self.cfg["num_layers"] = args.layers
            self.cfg["layer_types"] = self.cfg["layer_types"][: args.layers]
        self.recipe = dict(RECIPES[args.recipe])
        self.overrides: dict[str, int] = {}
        for spec in args.bits or []:
            key, value = spec.split("=")
            self.overrides[key] = int(value)
        if args.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = args.device
        self.report: dict[str, dict] = {}

    def bits_for(self, layer: int | None, role: str) -> int:
        if layer is not None and f"{layer}.{role}" in self.overrides:
            return self.overrides[f"{layer}.{role}"]
        if role in self.overrides:
            return self.overrides[role]
        return self.recipe[role]

    # -- tensor helpers ---------------------------------------------------------------------
    def load(self, name: str) -> torch.Tensor:
        return self.src.get(name).to(torch.float32)

    def gamma(self, name: str) -> torch.Tensor:
        # offset RMSNorm multiplier (1 + w), fp32
        return 1.0 + self.load(name)

    def quantized(self, name: str, build: Callable[[], torch.Tensor], bits: int, shape: list[int],
                  rows: int | None = None) -> Obj:
        def produce():
            w = build()
            n = w.shape[0]
            if n % 128:
                w = torch.cat([w, torch.zeros(align_up(n, 128) - n, w.shape[1])], dim=0)
            # Row blocks keep host/GPU memory bounded (the LM head is 248320 x 5120): the BQ planes
            # are nb-major, so the bytes of consecutive 64-row blocks are consecutive in every plane.
            lo, hi, sc = [], [], []
            sq_err = 0.0
            max_err = 0.0
            for r0 in range(0, w.shape[0], self.args.block_rows):
                blk = w[r0:r0 + self.args.block_rows].to(self.device)
                codes, scales = bq.quantize_matrix(blk, bits, grid=self.args.grid)
                err = bq.dequant_codes(codes, scales) - blk.to(torch.float32)
                sq_err += float((err.double() ** 2).sum())
                max_err = max(max_err, float(err.abs().max()))
                packed = bq.pack_bq(codes, scales, bits)
                lo.append(packed.lo)
                hi.append(packed.hi)
                sc.append(packed.scales)
            norm = float(w.double().norm())
            self.report[name] = {"rel_fro_err": (sq_err ** 0.5) / max(norm, 1e-30), "max_abs_err": max_err,
                                 "bits": bits}
            return [b"".join(lo), b"".join(hi), b"".join(sc)]
        return Obj(name, f"bq{bits}", shape, produce, rows=rows)

    def direct(self, name: str, dtype: str, build: Callable[[], torch.Tensor], shape: list[int]) -> Obj:
        def produce():
            t = build()
            assert list(t.shape) == shape, (name, list(t.shape), shape)
            if dtype == "bf16":
                return [t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()]
            return [t.to(torch.float32).contiguous().numpy().astype("<f4").tobytes()]
        return Obj(name, dtype, shape, produce)

    # -- plan -------------------------------------------------------------------------------
    def plan(self) -> list[Obj]:
        c = self.cfg
        H = c["hidden_size"]
        nq, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        lk, lv = c["linear_num_key_heads"], c["linear_num_value_heads"]
        dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        inter = c["intermediate_size"]
        objs: list[Obj] = []

        objs.append(self.direct("embed", "bf16", lambda: self.src.get("embed_tokens.weight"),
                                [c["vocab_size"], H]))

        for i in range(c["num_layers"]):
            p = f"layers.{i}."
            kind = c["layer_types"][i]
            if kind == "full_attention":
                def qkvg(p=p):
                    g = self.gamma(p + "input_layernorm.weight")
                    q_proj = self.load(p + "self_attn.q_proj.weight").view(nq, 2 * hd, H)
                    q, gate = q_proj[:, :hd, :].reshape(nq * hd, H), q_proj[:, hd:, :].reshape(nq * hd, H)
                    k = self.load(p + "self_attn.k_proj.weight")
                    v = self.load(p + "self_attn.v_proj.weight")
                    return torch.cat([q, gate, k, v], dim=0) * g[None, :]
                rows = 2 * nq * hd + 2 * nkv * hd
                objs.append(self.quantized(f"{i}.attn.qkvg", qkvg, self.bits_for(i, "attn.qkvg"),
                                           [align_up(rows, 128), H], rows))
                objs.append(self.quantized(
                    f"{i}.attn.o", lambda p=p: self.load(p + "self_attn.o_proj.weight"),
                    self.bits_for(i, "attn.o"), [H, nq * hd]))
                objs.append(self.direct(f"{i}.attn.q_norm", "f32",
                                        lambda p=p: self.gamma(p + "self_attn.q_norm.weight"), [hd]))
                objs.append(self.direct(f"{i}.attn.k_norm", "f32",
                                        lambda p=p: self.gamma(p + "self_attn.k_norm.weight"), [hd]))
            elif kind == "linear_attention":
                conv_dim = 2 * lk * dk + lv * dv

                def qkvz(p=p):
                    g = self.gamma(p + "input_layernorm.weight")
                    qkv = self.load(p + "linear_attn.in_proj_qkv.weight")
                    z = self.load(p + "linear_attn.in_proj_z.weight")
                    return torch.cat([qkv, z], dim=0) * g[None, :]

                def ab(p=p):
                    g = self.gamma(p + "input_layernorm.weight")
                    b = self.load(p + "linear_attn.in_proj_b.weight")
                    a = self.load(p + "linear_attn.in_proj_a.weight")
                    return torch.cat([b, a], dim=0) * g[None, :]

                rows = conv_dim + lv * dv
                objs.append(self.quantized(f"{i}.gdn.qkvz", qkvz, self.bits_for(i, "gdn.qkvz"),
                                           [align_up(rows, 128), H], rows))
                objs.append(self.direct(f"{i}.gdn.ab", "bf16", ab, [2 * lv, H]))
                objs.append(self.direct(
                    f"{i}.gdn.conv", "f32",
                    lambda p=p: self.load(p + "linear_attn.conv1d.weight").reshape(conv_dim, -1),
                    [conv_dim, c["linear_conv_kernel_dim"]]))
                objs.append(self.direct(
                    f"{i}.gdn.A", "f32", lambda p=p: -torch.exp(self.load(p + "linear_attn.A_log")), [lv]))
                objs.append(self.direct(
                    f"{i}.gdn.dt_bias", "f32", lambda p=p: self.load(p + "linear_attn.dt_bias"), [lv]))
                objs.append(self.direct(
                    f"{i}.gdn.norm", "f32", lambda p=p: self.load(p + "linear_attn.norm.weight"), [dv]))
                objs.append(self.quantized(
                    f"{i}.gdn.out", lambda p=p: self.load(p + "linear_attn.out_proj.weight"),
                    self.bits_for(i, "gdn.out"), [H, lv * dv]))
            else:
                raise ValueError(kind)

            def gate_up(p=p):
                g = self.gamma(p + "post_attention_layernorm.weight")
                gate = self.load(p + "mlp.gate_proj.weight")
                up = self.load(p + "mlp.up_proj.weight")
                inter_rows = torch.stack([gate, up], dim=1).reshape(2 * inter, H)  # row 2r gate, 2r+1 up
                return inter_rows * g[None, :]
            objs.append(self.quantized(f"{i}.mlp.gate_up", gate_up, self.bits_for(i, "mlp.gate_up"),
                                       [2 * inter, H]))
            objs.append(self.quantized(
                f"{i}.mlp.down", lambda p=p: self.load(p + "mlp.down_proj.weight"),
                self.bits_for(i, "mlp.down"), [H, inter]))

        def head():
            g = self.gamma("norm.weight")
            name = "embed_tokens.weight" if c["tie_word_embeddings"] else "lm_head.weight"
            w = self.src.get(name).to(torch.float32)
            return w * g[None, :]
        objs.append(self.quantized("lm_head", head, self.bits_for(None, "lm_head"),
                                   [c["vocab_size"], H]))
        return objs

    # -- write ------------------------------------------------------------------------------
    def dry_run(self):
        """Sizes only (no tensor data is read): per-role bytes and the total device footprint."""
        objs = self.plan()
        by_role: dict[str, int] = {}
        for o in objs:
            role = o.name.split(".", 1)[1] if o.name[0].isdigit() else o.name
            key = f"{role} ({o.dtype})"
            by_role[key] = by_role.get(key, 0) + sum(align_up(x, ALIGN) for x in o.sizes())
        total = sum(by_role.values())
        emb = by_role.get("embed (bf16)", 0)
        print(f"recipe {self.args.recipe}: {len(objs)} tensors")
        for k, v in sorted(by_role.items(), key=lambda kv: -kv[1]):
            print(f"  {k:28s} {v / 2**30:8.3f} GiB")
        print(f"  {'total file':28s} {total / 2**30:8.3f} GiB")
        print(f"  {'on GPU (embedding on host)':28s} {(total - emb) / 2**30:8.3f} GiB")
        return total

    def run(self):
        objs = self.plan()
        resources: dict[str, bytes] = {}
        if not self.args.no_tokenizer:
            tok_json = self.model_dir / "tokenizer.json"
            from tokenizer_export import export_tokenizer_blob  # written alongside this tool
            resources["tokenizer"] = export_tokenizer_blob(str(tok_json))
        template = self.model_dir / "chat_template.jinja"
        if template.exists():
            resources["chat_template"] = template.read_bytes()

        # layout
        offset = 0
        tensors = {}
        for o in objs:
            sizes = o.sizes()
            if o.dtype.startswith("bq"):
                planes = {}
                for plane, size in zip(("lo", "hi", "scales"), sizes):
                    if size == 0:
                        continue
                    offset = align_up(offset, ALIGN)
                    planes[plane] = [offset, size]
                    offset += size
                o.record = {"dtype": o.dtype, "shape": o.shape, "group_size": bq.GROUP, "planes": planes}
                if o.rows is not None and o.rows != o.shape[0]:
                    o.record["rows"] = o.rows
            else:
                offset = align_up(offset, ALIGN)
                o.record = {"dtype": o.dtype, "shape": o.shape, "offset": offset, "bytes": sizes[0]}
                offset += sizes[0]
            tensors[o.name] = o.record
        res_records = {}
        for key, blob in resources.items():
            offset = align_up(offset, ALIGN)
            res_records[key] = {"offset": offset, "bytes": len(blob)}
            offset += len(blob)
        total_data = offset

        header = {
            "format": "qwen38-model",
            "format_version": 1,
            "config": self.cfg,
            "recipe": {"name": self.args.recipe, "overrides": self.overrides, "grid": self.args.grid},
            "tensors": tensors,
            "resources": res_records,
            "provenance": {
                "source": str(self.args.source_id or self.model_dir.name),
                "converter": "tools/convert.py",
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }
        header_bytes = json.dumps(header, indent=1).encode("utf-8")
        data_start = align_up(16 + len(header_bytes), HEADER_ALIGN)

        out = Path(self.args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".partial")
        t0 = time.time()
        with open(tmp, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"\0" * (data_start - f.tell()))
            for idx, o in enumerate(objs):
                parts = o.produce()
                if o.dtype.startswith("bq"):
                    for plane, blob in zip(("lo", "hi", "scales"), parts):
                        if plane not in o.record["planes"]:
                            assert len(blob) == 0
                            continue
                        off, size = o.record["planes"][plane]
                        assert len(blob) == size, (o.name, plane, len(blob), size)
                        f.seek(data_start + off)
                        f.write(blob)
                else:
                    f.seek(data_start + o.record["offset"])
                    assert len(parts[0]) == o.record["bytes"]
                    f.write(parts[0])
                if self.args.verbose or idx % 10 == 0:
                    rep = self.report.get(o.name)
                    extra = f" rel_err={rep['rel_fro_err']:.4f}" if rep else ""
                    print(f"[{idx + 1}/{len(objs)}] {o.name} {o.dtype} {o.shape}{extra} "
                          f"({time.time() - t0:.0f}s)", flush=True)
            for key, blob in resources.items():
                f.seek(data_start + res_records[key]["offset"])
                f.write(blob)
            f.seek(data_start + total_data)
            f.truncate()
        os.replace(tmp, out)
        report_path = out.with_suffix(out.suffix + ".report.json")
        report_path.write_text(json.dumps(self.report, indent=1))
        size_gb = out.stat().st_size / 1e9
        print(f"wrote {out} ({size_gb:.2f} GB) in {time.time() - t0:.0f}s; report {report_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face checkpoint directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", default="default", choices=sorted(RECIPES))
    ap.add_argument("--bits", action="append", help="override, e.g. mlp.down=6 or 12.attn.o=8")
    ap.add_argument("--grid", type=int, default=24, help="scale candidates per mapping in the MSE search")
    ap.add_argument("--block-rows", type=int, default=8192, help="rows quantized/packed at a time (multiple of 128)")
    ap.add_argument("--layers", type=int, default=None, help="convert only the first N layers (testing)")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--source-id", default=None, help="provenance string, e.g. Qwen/Qwen3.8-27B@<rev>")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the size plan without reading weights")
    args = ap.parse_args()
    conv = Converter(args)
    conv.dry_run() if args.dry_run else conv.run()


if __name__ == "__main__":
    main()
