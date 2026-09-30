"""BQ weight quantization and packing (see docs/FORMAT.md, section 2).

Everything here is vectorised torch code so the converter can run on the GPU (fast) or the
CPU (slow but works). The packing functions are the single source of truth for the byte
layout; tests unpack with `unpack_bq` and compare against the C++ and CUDA readers.

Numbers: codes c in [-2^(b-1), 2^(b-1)-1], one bf16 scale per (row, 64-wide K group),
dequantised weight = bf16(float(c) * float(scale)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

GROUP = 64
SLAB_K = 16
SLAB_N = 64
BITS_SUPPORTED = (4, 5, 6, 8)


def to_bf16_f32(x: torch.Tensor) -> torch.Tensor:
    """Round an fp32 tensor to bf16 (RNE) and return it as fp32."""
    return x.to(torch.bfloat16).to(torch.float32)


def dequant_codes(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """codes [N, K] int, scales [N, K/64] fp32 (bf16-representable) -> bf16(c * s) as fp32."""
    n, k = codes.shape
    s = scales.to(torch.float32).repeat_interleave(GROUP, dim=1)
    return to_bf16_f32(codes.to(torch.float32) * s)


# ---------------------------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def quantize_groups(x: torch.Tensor, bits: int, grid: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE-searched symmetric quantization of rows of 64 values.

    x: [G, 64] fp32.  Returns (codes int16 [G, 64], scale fp32 [G], bf16-representable).

    Candidates (llama.cpp make_qx_quants style): scales that map the signed extreme value onto
    -2^(b-1) (the negative end has one more level) or onto +(2^(b-1)-1), each stretched over a
    small grid, followed by a least-squares refit of the scale for the chosen codes. The final
    scale is rounded to bf16 and the codes are recomputed against the rounded scale, so the stored
    (codes, scale) pair has no systematic bias from scale rounding.
    """
    assert x.dim() == 2 and x.shape[1] == GROUP
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    x = x.to(torch.float32)
    idx = x.abs().argmax(dim=1, keepdim=True)
    ext = x.gather(1, idx).squeeze(1)  # signed extreme, |ext| = amax
    nonzero = ext != 0
    safe_ext = torch.where(nonzero, ext, torch.ones_like(ext))

    best_err = torch.full_like(ext, float("inf"))
    best_scale = torch.zeros_like(ext)
    factors = torch.linspace(0.80, 1.06, grid, device=x.device)
    for target in (float(qmin), float(qmax)):
        for f in factors:
            s = safe_ext / (target * f)
            c = torch.clamp(torch.round(x / s[:, None]), qmin, qmax)
            # least-squares refit of the scale for these codes
            num = (c * x).sum(dim=1)
            den = (c * c).sum(dim=1)
            s_ls = torch.where(den > 0, num / den.clamp_min(1e-30), s)
            s_ls = to_bf16_f32(s_ls)
            s_ls = torch.where(s_ls == 0, to_bf16_f32(s), s_ls)
            c2 = torch.clamp(torch.round(x / s_ls[:, None]), qmin, qmax)
            err = ((c2 * s_ls[:, None] - x) ** 2).sum(dim=1)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, s_ls, best_scale)

    best_scale = torch.where(nonzero, best_scale, torch.zeros_like(best_scale))
    safe = torch.where(best_scale != 0, best_scale, torch.ones_like(best_scale))
    codes = torch.clamp(torch.round(x / safe[:, None]), qmin, qmax)
    codes = torch.where(best_scale[:, None] != 0, codes, torch.zeros_like(codes))
    return codes.to(torch.int16), best_scale


@torch.no_grad()
def quantize_matrix(w: torch.Tensor, bits: int, grid: int = 24, row_chunk: int = 2048
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """w: [N, K] float -> (codes int16 [N, K], scales fp32 [N, K/64])."""
    assert bits in BITS_SUPPORTED
    n, k = w.shape
    assert k % GROUP == 0, f"K={k} must be a multiple of {GROUP}"
    codes = torch.empty((n, k), dtype=torch.int16, device=w.device)
    scales = torch.empty((n, k // GROUP), dtype=torch.float32, device=w.device)
    for r0 in range(0, n, row_chunk):
        r1 = min(n, r0 + row_chunk)
        block = w[r0:r1].to(torch.float32).reshape(-1, GROUP)
        c, s = quantize_groups(block, bits, grid)
        codes[r0:r1] = c.reshape(r1 - r0, k)
        scales[r0:r1] = s.reshape(r1 - r0, k // GROUP)
    return codes, scales


# ---------------------------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------------------------

@dataclass
class PackedBQ:
    bits: int
    n: int
    k: int
    lo: bytes
    hi: bytes  # b"" for 4-bit
    scales: bytes


def _lane_pair_view(u: np.ndarray) -> np.ndarray:
    """u: [N, K] uint8 unsigned codes -> [nb, kb, lane(32), pair(16), elem(2)] in slab order.

    Pair q of lane L (g = L >> 2, t = L & 3): j = q >> 1, h = q & 1,
      column n = 64 nb + 8 j + g, rows k0 = 16 kb + 2t + 8h and k0 + 1.
    """
    n, k = u.shape
    nb, kb = n // SLAB_N, k // SLAB_K
    # [nb, j(8), g(8), kb, h(2), t(4), e(2)]  from  n = 64nb + 8j + g,  k = 16kb + 8h + 2t + e
    v = u.reshape(nb, 8, 8, kb, 2, 4, 2)
    # -> [nb, kb, g, t, j, h, e]
    v = v.transpose(0, 3, 2, 5, 1, 4, 6)
    # lane = 4g + t, pair = 2j + h
    return v.reshape(nb, kb, 32, 16, 2)


def pack_bq(codes: torch.Tensor, scales: torch.Tensor, bits: int) -> PackedBQ:
    """codes [N, K] signed ints, scales [N, K/64] fp32 (bf16-representable)."""
    assert bits in BITS_SUPPORTED
    n, k = codes.shape
    assert n % 128 == 0 and k % GROUP == 0, (n, k)
    c = codes.to(torch.int32).cpu().numpy()
    u = (c + (1 << (bits - 1))).astype(np.uint32)
    assert u.min() >= 0 and u.max() < (1 << bits)
    pv = _lane_pair_view(u)  # [nb, kb, 32, 16, 2]
    nb, kb = pv.shape[0], pv.shape[1]

    def nibble_plane(vals: np.ndarray) -> bytes:
        # vals: [nb, kb, 32, 16, 2] values < 16 ; word w = q >> 2, position p = q & 3
        v = vals.reshape(nb, kb, 32, 4, 4, 2).astype(np.uint32)
        p = np.arange(4, dtype=np.uint32)
        words = ((v[..., 0] << (4 * p)) | (v[..., 1] << (16 + 4 * p))).sum(axis=-1, dtype=np.uint32)
        # words: [nb, kb, 32, 4]
        return words.astype("<u4").tobytes()

    lo = nibble_plane(pv & 15)
    hi_vals = pv >> 4
    if bits == 4:
        hi = b""
    elif bits == 5:
        q = np.arange(16, dtype=np.uint32)
        h = hi_vals.astype(np.uint32)  # [nb, kb, 32, 16, 2], values 0..1
        word = ((h[..., 0] << q) | (h[..., 1] << (16 + q))).sum(axis=-1, dtype=np.uint32)
        hi = word.astype("<u4").tobytes()  # [nb, kb, 32] one word per lane
    elif bits == 6:
        h = hi_vals.astype(np.uint32).reshape(nb, kb, 32, 2, 8, 2)  # [.., word, r, e], values 0..3
        r = np.arange(8, dtype=np.uint32)
        words = ((h[..., 0] << (2 * r)) | (h[..., 1] << (16 + 2 * r))).sum(axis=-1, dtype=np.uint32)
        hi = words.astype("<u4").tobytes()  # [nb, kb, 32, 2]
    else:  # 8
        hi = nibble_plane(hi_vals)

    # scales: [N, K/64] -> [nb, g, 64 columns permuted: position = g_lane * 8 + j for n_local = 8 j + g_lane]
    s = scales.to(torch.float32).cpu()
    s16 = s.to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)  # exact bf16 bits
    groups = k // GROUP
    s16 = s16.reshape(nb, 8, 8, groups)          # [nb, j, g_lane, group]
    s16 = s16.transpose(0, 3, 2, 1)              # [nb, group, g_lane, j]
    scales_bytes = np.ascontiguousarray(s16).astype("<u2").tobytes()
    return PackedBQ(bits=bits, n=n, k=k, lo=lo, hi=hi, scales=scales_bytes)


def unpack_bq(p: PackedBQ) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of pack_bq -> (codes int16 [N, K], scales fp32 [N, K/64])."""
    n, k, bits = p.n, p.k, p.bits
    nb, kb = n // SLAB_N, k // SLAB_K

    def nibbles(buf: bytes) -> np.ndarray:
        words = np.frombuffer(buf, dtype="<u4").reshape(nb, kb, 32, 4).astype(np.uint32)
        pos = np.arange(4, dtype=np.uint32)
        e0 = (words[..., None] >> (4 * pos)) & 15            # [nb, kb, 32, w, p]
        e1 = (words[..., None] >> (16 + 4 * pos)) & 15
        return np.stack([e0, e1], axis=-1).reshape(nb, kb, 32, 16, 2)

    u = nibbles(p.lo)
    if bits == 5:
        word = np.frombuffer(p.hi, dtype="<u4").reshape(nb, kb, 32).astype(np.uint32)
        q = np.arange(16, dtype=np.uint32)
        h0 = (word[..., None] >> q) & 1
        h1 = (word[..., None] >> (16 + q)) & 1
        u = u | (np.stack([h0, h1], axis=-1) << 4)
    elif bits == 6:
        words = np.frombuffer(p.hi, dtype="<u4").reshape(nb, kb, 32, 2).astype(np.uint32)
        r = np.arange(8, dtype=np.uint32)
        h0 = (words[..., None] >> (2 * r)) & 3            # [nb, kb, 32, word, r]
        h1 = (words[..., None] >> (16 + 2 * r)) & 3
        h = np.stack([h0, h1], axis=-1).reshape(nb, kb, 32, 16, 2)
        u = u | (h << 4)
    elif bits == 8:
        u = u | (nibbles(p.hi) << 4)

    # invert _lane_pair_view: [nb, kb, g, t, j, h, e] -> [nb, j, g, kb, h, t, e]
    v = u.reshape(nb, kb, 8, 4, 8, 2, 2).transpose(0, 4, 2, 1, 5, 3, 6)
    u_nk = v.reshape(n, k).astype(np.int32)
    codes = torch.from_numpy(u_nk - (1 << (bits - 1))).to(torch.int16)

    groups = k // GROUP
    s16 = np.frombuffer(p.scales, dtype="<u2").reshape(nb, groups, 8, 8)  # [nb, group, g_lane, j]
    s16 = s16.transpose(0, 3, 2, 1).reshape(n, groups)                     # [n = 64nb + 8j + g, group]
    scales = torch.from_numpy(s16.astype(np.int16).copy()).view(torch.bfloat16).to(torch.float32)
    return codes, scales


def plane_sizes(n: int, k: int, bits: int) -> tuple[int, int, int]:
    lo = n * k // 2
    hi = n * k * (bits - 4) // 8
    sc = n * (k // GROUP) * 2
    return lo, hi, sc


def quantization_report(w: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor) -> dict:
    wq = dequant_codes(codes, scales)
    w = w.to(torch.float32)
    err = (wq - w)
    rel = (err.norm() / w.norm().clamp_min(1e-30)).item()
    return {"rel_fro_err": rel, "max_abs_err": err.abs().max().item()}
