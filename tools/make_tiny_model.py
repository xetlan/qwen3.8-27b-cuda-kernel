#!/usr/bin/env python3
"""Create a small random model with the exact Qwen3.8 architecture (Qwen3_5ForCausalLM) for tests.

Same head geometry as the 27B model (attention head_dim 256, rotary 64, GDN 128-dim heads,
V/K head ratio 3, conv width 4) but narrow and shallow, so the CPU reference and Hugging Face
can run it in seconds. Weights are drawn with magnitudes that keep activations O(1) and make
every nonlinearity (gates, softplus, SiLU, norms) matter.

    python tools/make_tiny_model.py --out testdata/tiny
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


def tiny_config(layers: int = 8, vocab: int = 1024) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=vocab,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=layers,
        num_attention_heads=6,
        num_key_value_heads=1,
        head_dim=256,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [11, 11, 10], "mrope_interleaved": True},
        eos_token_id=[1000, 1001],
    )


@torch.no_grad()
def randomize(model: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)

    def normal(t: torch.Tensor, std: float, mean: float = 0.0):
        t.copy_(torch.randn(t.shape, generator=g) * std + mean)

    for name, p in model.named_parameters():
        if name.endswith("embed_tokens.weight"):
            normal(p, 1.0)
        elif name.endswith("lm_head.weight"):
            normal(p, 3.0 / math.sqrt(p.shape[1]))
        elif name.endswith("A_log"):
            p.copy_(torch.log(torch.empty(p.shape).uniform_(0.5, 8.0, generator=g)))
        elif name.endswith("dt_bias"):
            normal(p, 0.5)
        elif name.endswith("conv1d.weight"):
            normal(p, 0.4)
        elif name.endswith("linear_attn.norm.weight"):
            normal(p, 0.2, mean=1.0)  # plain RMSNorm weight
        elif "norm" in name:
            normal(p, 0.2)  # offset RMSNorm: multiplier is 1 + w
        elif p.dim() == 2:
            normal(p, 1.0 / math.sqrt(p.shape[1]))
        else:
            raise ValueError(f"unhandled parameter {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="testdata/tiny")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--vocab", type=int, default=1024,
                    help="248320 + --tokenizer-dir gives a CLI smoke-test model with the real tokenizer")
    ap.add_argument("--tokenizer-dir", default=None,
                    help="copy tokenizer.json / chat_template.jinja from here (e.g. testdata/qwen3.8)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = tiny_config(args.layers, args.vocab)
    model = Qwen3_5ForCausalLM(cfg).to(torch.float32)
    randomize(model, args.seed)
    model.save_pretrained(out, safe_serialization=True)
    eos = [248046, 248044] if args.vocab >= 248077 else [1000, 1001]
    (out / "generation_config.json").write_text(json.dumps({"eos_token_id": eos}))
    if args.tokenizer_dir:
        import shutil
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            src = Path(args.tokenizer_dir) / name
            if src.exists():
                shutil.copy(src, out / name)
    print(f"wrote tiny model to {out}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
          f"layer_types={cfg.layer_types}")


if __name__ == "__main__":
    main()
