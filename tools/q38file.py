"""Python reader for .q38 model files (docs/FORMAT.md). Used by tests and for inspection:

    python tools/q38file.py models/qwen3.8-27b.q38          # summary
"""

from __future__ import annotations

import json
import mmap
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bqpack as bq  # noqa: E402


class Q38File:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        magic = self._f.read(8)
        if magic != b"QW38MDL1":
            raise ValueError(f"{path} is not a .q38 file")
        (hlen,) = struct.unpack("<Q", self._f.read(8))
        self.header = json.loads(self._f.read(hlen))
        self.data_start = (16 + hlen + 4095) // 4096 * 4096
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.config = self.header["config"]
        self.tensors = self.header["tensors"]

    def _bytes(self, off: int, size: int) -> bytes:
        start = self.data_start + off
        return self._mm[start:start + size]

    def direct(self, name: str) -> torch.Tensor:
        rec = self.tensors[name]
        raw = self._bytes(rec["offset"], rec["bytes"])
        if rec["dtype"] == "bf16":
            arr = np.frombuffer(raw, dtype="<i2").copy()
            return torch.from_numpy(arr).view(torch.bfloat16).reshape(rec["shape"])
        if rec["dtype"] == "f32":
            return torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy()).reshape(rec["shape"])
        raise ValueError(f"{name}: {rec['dtype']} is not a direct dtype")

    def packed(self, name: str) -> bq.PackedBQ:
        rec = self.tensors[name]
        bits = int(rec["dtype"][2:])
        planes = rec["planes"]
        lo = self._bytes(*planes["lo"])
        hi = self._bytes(*planes["hi"]) if "hi" in planes else b""
        sc = self._bytes(*planes["scales"])
        n, k = rec["shape"]
        return bq.PackedBQ(bits=bits, n=n, k=k, lo=lo, hi=hi, scales=sc)

    def dequant(self, name: str, logical_rows: bool = True) -> torch.Tensor:
        """Dequantized weight [rows, K] as fp32 (values are bf16-exact)."""
        codes, scales = bq.unpack_bq(self.packed(name))
        w = bq.dequant_codes(codes, scales)
        rows = self.tensors[name].get("rows", w.shape[0])
        return w[:rows] if logical_rows else w

    def resource(self, name: str) -> bytes:
        rec = self.header.get("resources", {}).get(name)
        return b"" if rec is None else self._bytes(rec["offset"], rec["bytes"])


def main():
    f = Q38File(sys.argv[1])
    c = f.config
    print(f"{f.path}: recipe={f.header.get('recipe', {}).get('name')}  hidden={c['hidden_size']} "
          f"layers={c['num_layers']} vocab={c['vocab_size']}")
    total = 0
    by_dtype: dict[str, int] = {}
    for name, rec in f.tensors.items():
        size = sum(p[1] for p in rec["planes"].values()) if "planes" in rec else rec["bytes"]
        total += size
        by_dtype[rec["dtype"]] = by_dtype.get(rec["dtype"], 0) + size
    for d, s in sorted(by_dtype.items()):
        print(f"  {d:5s} {s / 2**30:8.3f} GiB")
    print(f"  total {total / 2**30:8.3f} GiB")


if __name__ == "__main__":
    main()
