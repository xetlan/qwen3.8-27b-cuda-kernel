#!/usr/bin/env python3
"""Export a Hugging Face byte-level BPE tokenizer (Qwen3.x) into the engine's tokenizer blob.

    python tools/tokenizer_export.py <tokenizer.json> <out.bin>            # write the blob
    python tools/tokenizer_export.py <tokenizer.json> <out.bin> --verify   # ... and cross-check
                                                                          # the Unicode tables
                                                                          # against the installed
                                                                          # HF `tokenizers`

`export_tokenizer_blob(path) -> bytes` is the library entry point (used by the model converter,
which embeds the blob in the .q38 file). Only the Python standard library is needed to export;
`--verify` additionally needs `tokenizers`.

The C++ reader is `src/tokenizer.cpp` (`qwen38::Tokenizer::from_blob`). It implements the Qwen3
split regex with a hand-written scanner, so the exporter refuses any tokenizer.json whose
normalizer / pre-tokenizer / model configuration differs from what that scanner implements.


Blob layout (format version 1, all integers little-endian)
---------------------------------------------------------

    offset 0   8 bytes  magic b"QW38TOK\\0"
    offset 8   u32      format version (1)
    offset 12  u32      section count S
    offset 16  S x 24   section directory: { u8 tag[4]; u32 reserved = 0; u64 offset; u64 size }
                        (offset from the start of the blob, every section starts 8-byte aligned,
                        gaps are zero padding)

Sections (each tag appears once; all are required unless noted):

    META  UTF-8 text, one "key=value" per line: provenance (source sha256, unidata_version,
          which HF behaviour the Unicode tables were patched to), normalizer, ignore_merges, ...
    PATN  UTF-8 split regex of the pre-tokenizer; the C++ side checks that it is exactly the
          pattern its scanner implements.
    VOCB  u32 n_ids; u32 n_base; u32 offsets[n_ids + 1]; u8 bytes[offsets[n_ids]]
          Raw bytes of every addressable id (GPT-2 bytes_to_unicode already undone). Ids
          [0, n_base) are the BPE model vocabulary; ids >= n_base are added tokens (their content
          bytes). An id that is not assigned has an empty entry.
    MERG  u32 n; n x { u32 left_id; u32 right_id; u32 result_id }
          BPE merges in priority order: record i has rank i (lower merges first).
          result_id is the id of bytes(left) + bytes(right).
    ADDT  u32 n; n x { u32 id; u32 flags; u32 offset; u32 length }; u8 pool[]
          Added tokens, content = pool[offset, offset + length) (raw UTF-8, before any
          normalization). flags: 1 special, 2 lstrip, 4 rstrip, 8 single_word, 16 normalized.
    UCLS  u32 n; n x { u32 first; u32 last; u32 flags }   sorted, disjoint, inclusive ranges
          Character classes of the split regex: 1 = \\p{L}, 2 = \\p{M}, 4 = \\p{N},
          8 = \\s (Oniguruma: White_Space = Zs | Zl | Zp | U+0009..U+000D | U+0085; also what
          added-token lstrip/rstrip strip), 16 = \\w (the word test of single_word added tokens).
          Code points not covered have no class.
    UCCC  u32 n; n x { u32 first; u32 last; u32 ccc }   sorted, disjoint; canonical combining
          class for the NFC normalizer (absent = 0).
    UDEC  u32 n; n x { u32 cp; u32 offset; u32 length }; u32 pool[]   sorted by cp
          Full canonical decomposition (recursively expanded, canonically ordered) of every code
          point that has one, excluding Hangul syllables U+AC00..U+D7A3, which the reader
          decomposes and composes algorithmically (UAX #15 / Unicode ch. 3.12).
    UCMP  u32 n; n x { u32 first; u32 second; u32 composite }   sorted by (first, second)
          Canonical composition pairs of all primary composites (i.e. composition exclusions,
          singletons and non-starter decompositions already removed), Hangul excluded.
    UEXC  u32 n; u32 cp[n]   sorted. Informational: code points with a canonical decomposition
          that NFC never recomposes (the full composition exclusion set). Not needed to encode.


Unicode data provenance
-----------------------

All Unicode tables are generated from Python's `unicodedata` (its `unidata_version` is recorded
in META) and then patched to reproduce what Hugging Face `tokenizers` actually does, because the
two Rust components HF uses carry their own, different Unicode versions:

  * The split regex runs on Oniguruma (crate onig 6.5.x), whose property tables are
    Unicode 16.0.0. Python 3.12 ships 15.0.0, so the 5055 code points that 16.0 assigns as
    letters, marks or numbers are added (_REGEX_UNICODE16_ADDITIONS, exact categories taken from
    UCD 16.0.0). No other code point differs; `--verify` probes every code point through HF.
  * NFC runs on unicode-normalization-alignments 0.1.12, whose tables are Unicode 9.0.0: the 108
    combining marks encoded after 9.0 are starters (ccc 0) there, and U+11938 (13.0) has no
    decomposition, so the pair U+11935 U+11930 does not compose (_NFC_CCC_ZERO,
    _NFC_NO_DECOMPOSITION). Everything else agrees with 15.0 (normalization stability).

Both patch tables were derived by probing HF tokenizers 0.23.2 over all 1,112,064 scalar values;
`--verify` repeats the probe against whatever version is installed. They also carry the Unicode
15.1 / 16.0 differences, so Python versions shipping those (3.13, 3.14) export identical tables;
any other `unidata_version` is refused.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import unicodedata

MAGIC = b"QW38TOK\x00"
FORMAT_VERSION = 1

# The only pre-tokenizer split pattern the C++ scanner implements (Qwen3.5 / Qwen3.8).
QWEN_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

CLS_L, CLS_M, CLS_N, CLS_WS, CLS_WORD = 1, 2, 4, 8, 16
ADDED_SPECIAL, ADDED_LSTRIP, ADDED_RSTRIP, ADDED_SINGLE_WORD, ADDED_NORMALIZED = 1, 2, 4, 8, 16

HANGUL_S_BASE, HANGUL_S_COUNT = 0xAC00, 11172
HANGUL_L_BASE, HANGUL_V_BASE, HANGUL_T_BASE = 0x1100, 0x1161, 0x11A7
HANGUL_L_COUNT, HANGUL_V_COUNT, HANGUL_T_COUNT = 19, 21, 28
HANGUL_N_COUNT = HANGUL_V_COUNT * HANGUL_T_COUNT

# Letters / marks / numbers that Unicode 16.0.0 assigns and Python's 15.0.0 tables do not know
# (all were Cn), with their 16.0.0 general category. Oniguruma in HF tokenizers matches them with
# \p{L} / \p{M} / \p{N} (and \w).
_REGEX_UNICODE16_ADDITIONS = [
    (0x00897, 0x00897, "Mn"), (0x01C89, 0x01C89, "Lu"), (0x01C8A, 0x01C8A, "Ll"), (0x0A7CB, 0x0A7CC, "Lu"),
    (0x0A7CD, 0x0A7CD, "Ll"), (0x0A7DA, 0x0A7DA, "Lu"), (0x0A7DB, 0x0A7DB, "Ll"), (0x0A7DC, 0x0A7DC, "Lu"),
    (0x105C0, 0x105F3, "Lo"), (0x10D40, 0x10D49, "Nd"), (0x10D4A, 0x10D4D, "Lo"), (0x10D4E, 0x10D4E, "Lm"),
    (0x10D4F, 0x10D4F, "Lo"), (0x10D50, 0x10D65, "Lu"), (0x10D69, 0x10D6D, "Mn"), (0x10D6F, 0x10D6F, "Lm"),
    (0x10D70, 0x10D85, "Ll"), (0x10EC2, 0x10EC4, "Lo"), (0x10EFC, 0x10EFC, "Mn"), (0x11380, 0x11389, "Lo"),
    (0x1138B, 0x1138B, "Lo"), (0x1138E, 0x1138E, "Lo"), (0x11390, 0x113B5, "Lo"), (0x113B7, 0x113B7, "Lo"),
    (0x113B8, 0x113BA, "Mc"), (0x113BB, 0x113C0, "Mn"), (0x113C2, 0x113C2, "Mc"), (0x113C5, 0x113C5, "Mc"),
    (0x113C7, 0x113CA, "Mc"), (0x113CC, 0x113CD, "Mc"), (0x113CE, 0x113CE, "Mn"), (0x113CF, 0x113CF, "Mc"),
    (0x113D0, 0x113D0, "Mn"), (0x113D1, 0x113D1, "Lo"), (0x113D2, 0x113D2, "Mn"), (0x113D3, 0x113D3, "Lo"),
    (0x113E1, 0x113E2, "Mn"), (0x116D0, 0x116E3, "Nd"), (0x11BC0, 0x11BE0, "Lo"), (0x11BF0, 0x11BF9, "Nd"),
    (0x11F5A, 0x11F5A, "Mn"), (0x13460, 0x143FA, "Lo"), (0x16100, 0x1611D, "Lo"), (0x1611E, 0x16129, "Mn"),
    (0x1612A, 0x1612C, "Mc"), (0x1612D, 0x1612F, "Mn"), (0x16130, 0x16139, "Nd"), (0x16D40, 0x16D42, "Lm"),
    (0x16D43, 0x16D6A, "Lo"), (0x16D6B, 0x16D6C, "Lm"), (0x16D70, 0x16D79, "Nd"), (0x18CFF, 0x18CFF, "Lo"),
    (0x1CCF0, 0x1CCF9, "Nd"), (0x1E5D0, 0x1E5ED, "Lo"), (0x1E5EE, 0x1E5EF, "Mn"), (0x1E5F0, 0x1E5F0, "Lo"),
    (0x1E5F1, 0x1E5FA, "Nd"), (0x2EBF0, 0x2EE5D, "Lo"),
]

# Symbols (gc=So) with the Other_Alphabetic property: Oniguruma's \w includes them.
_WORD_OTHER_ALPHABETIC_SO = [(0x024B6, 0x024E9), (0x1F130, 0x1F149), (0x1F150, 0x1F169), (0x1F170, 0x1F189)]

# Combining marks encoded after Unicode 9.0: non-zero ccc in Python's tables, starters in HF's NFC.
_NFC_CCC_ZERO = [
    (0x007FD, 0x007FD), (0x00898, 0x0089F), (0x008CA, 0x008D3), (0x009FE, 0x009FE), (0x00C3C, 0x00C3C),
    (0x00D3B, 0x00D3C), (0x00EBA, 0x00EBA), (0x01715, 0x01715), (0x01ABF, 0x01ACE), (0x01DF6, 0x01DFA),
    (0x0A82C, 0x0A82C), (0x10D24, 0x10D27), (0x10EAB, 0x10EAC), (0x10EFD, 0x10EFF), (0x10F46, 0x10F50),
    (0x10F82, 0x10F85), (0x11070, 0x11070), (0x1133B, 0x1133B), (0x1145E, 0x1145E), (0x11839, 0x1183A),
    (0x1193D, 0x1193E), (0x11943, 0x11943), (0x119E0, 0x119E0), (0x11A34, 0x11A34), (0x11A47, 0x11A47),
    (0x11A99, 0x11A99), (0x11D42, 0x11D42), (0x11D44, 0x11D45), (0x11D97, 0x11D97), (0x11F41, 0x11F42),
    (0x16FF0, 0x16FF1), (0x1E08F, 0x1E08F), (0x1E130, 0x1E136), (0x1E2AE, 0x1E2AE), (0x1E2EC, 0x1E2EF),
    (0x1E4EC, 0x1E4EF),
    # new in Unicode 16.0 (only matter when Python's unicodedata is 16.0)
    (0x00897, 0x00897), (0x10D69, 0x10D6D), (0x113CE, 0x113D0), (0x1612F, 0x1612F), (0x1E5EE, 0x1E5EF),
]
# Canonical decompositions HF's NFC does not know (and therefore never composes back): U+11938
# (Unicode 13.0) and the 20 decomposable characters of Unicode 16.0.
_NFC_NO_DECOMPOSITION = [
    0x11938,
    0x105C9, 0x105E4, 0x11383, 0x11385, 0x1138E, 0x11391, 0x113C5, 0x113C7, 0x113C8, 0x16121, 0x16122,
    0x16123, 0x16124, 0x16125, 0x16126, 0x16127, 0x16128, 0x16D68, 0x16D69, 0x16D6A,
]

# Python unicodedata versions for which the patch tables above give exactly HF's behaviour
# (checked: every exported Unicode section is byte-identical for 15.0.0, 15.1.0 and 16.0.0).
# A newer version assigns characters HF does not know; re-derive the tables with --verify.
_SUPPORTED_UNIDATA = ("15.0.0", "15.1.0", "16.0.0")

_HF_REFERENCE = (
    "tokenizers 0.23.2 (regex: onig 6.5.3 / Oniguruma, Unicode 16.0.0; "
    "NFC: unicode-normalization-alignments 0.1.12, Unicode 9.0.0)"
)


# ---------------------------------------------------------------------------------------------
# GPT-2 byte <-> unicode mapping
# ---------------------------------------------------------------------------------------------

def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 / HF ByteLevel table: byte value -> printable unicode character."""
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))  # printable Latin-1
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_UNICODE_TO_BYTE = {u: b for b, u in bytes_to_unicode().items()}


def _bytelevel_to_raw(token: str) -> bytes:
    try:
        return bytes(_UNICODE_TO_BYTE[ch] for ch in token)
    except KeyError as e:
        raise ValueError(f"vocab token {token!r} contains a character outside the ByteLevel alphabet") from e


# ---------------------------------------------------------------------------------------------
# tokenizer.json validation
# ---------------------------------------------------------------------------------------------

def _validate_config(tj: dict) -> dict:
    """Check that the tokenizer is the byte-level BPE the C++ side implements; return options."""
    opts = {}
    norm = tj.get("normalizer")
    if norm is None:
        opts["normalizer"] = "none"
    elif norm == {"type": "NFC"}:
        opts["normalizer"] = "NFC"
    else:
        raise ValueError(f"unsupported normalizer {norm!r} (only NFC or none)")

    pre = tj.get("pre_tokenizer") or {}
    seq = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
    if not seq or len(seq) != 2:
        raise ValueError(f"unsupported pre_tokenizer {pre!r}: expected Sequence[Split, ByteLevel]")
    split, bl = seq
    if (split.get("type") != "Split" or split.get("behavior") != "Isolated" or split.get("invert")
            or (split.get("pattern") or {}).get("Regex") != QWEN_SPLIT_PATTERN):
        raise ValueError(f"unsupported Split pre-tokenizer {split!r}: the C++ scanner implements "
                         f"exactly the Qwen3 pattern {QWEN_SPLIT_PATTERN!r} (Isolated, invert=false)")
    if bl.get("type") != "ByteLevel" or bl.get("add_prefix_space") or bl.get("use_regex"):
        raise ValueError(f"unsupported ByteLevel pre-tokenizer {bl!r} (need add_prefix_space=false, "
                         f"use_regex=false)")

    dec = tj.get("decoder") or {}
    if dec.get("type") != "ByteLevel":
        raise ValueError(f"unsupported decoder {dec!r} (need ByteLevel)")

    model = tj.get("model") or {}
    if model.get("type") != "BPE":
        raise ValueError(f"unsupported model type {model.get('type')!r}")
    if model.get("dropout") not in (None, 0, 0.0):
        raise ValueError("BPE dropout is not supported")
    if model.get("byte_fallback"):
        raise ValueError("byte_fallback is not supported (byte-level BPE never needs it)")
    if model.get("continuing_subword_prefix") not in (None, ""):
        raise ValueError("continuing_subword_prefix is not supported")
    if model.get("end_of_word_suffix") not in (None, ""):
        raise ValueError("end_of_word_suffix is not supported")
    opts["ignore_merges"] = bool(model.get("ignore_merges", False))
    return opts


# ---------------------------------------------------------------------------------------------
# Unicode tables
# ---------------------------------------------------------------------------------------------

def _ranges(values: dict[int, int]) -> list[tuple[int, int, int]]:
    """{cp: v} (v != 0) -> sorted maximal ranges (first, last, v)."""
    out: list[list[int]] = []
    for cp in sorted(values):
        v = values[cp]
        if out and out[-1][1] == cp - 1 and out[-1][2] == v:
            out[-1][1] = cp
        else:
            out.append([cp, cp, v])
    return [tuple(r) for r in out]


def _all_scalars():
    return (cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF)


def _check_unidata(ud) -> None:
    if ud.unidata_version not in _SUPPORTED_UNIDATA:
        raise ValueError(f"unicodedata {ud.unidata_version} is not one of {_SUPPORTED_UNIDATA}, the versions the "
                         f"HF patch tables were verified for; re-derive them (see --verify) before exporting")


def build_regex_classes(ud=unicodedata) -> dict[int, int]:
    """{cp: class flags} for the split regex (and \\w), as HF's Oniguruma sees it."""
    _check_unidata(ud)
    cats = {cp: ud.category(chr(cp)) for cp in _all_scalars()}
    for first, last, cat in _REGEX_UNICODE16_ADDITIONS:
        for cp in range(first, last + 1):
            if cats[cp] == "Cn":
                cats[cp] = cat
            elif cats[cp] != cat:
                raise AssertionError(f"U+{cp:04X} is {cats[cp]}, expected {cat}; patch table is stale")
    other_alpha_so = {cp for a, b in _WORD_OTHER_ALPHABETIC_SO for cp in range(a, b + 1)}
    cls: dict[int, int] = {}
    for cp, cat in cats.items():
        f = {"L": CLS_L, "M": CLS_M, "N": CLS_N}.get(cat[0], 0)
        if cat in ("Zs", "Zl", "Zp") or 0x09 <= cp <= 0x0D or cp == 0x85:
            f |= CLS_WS
        # Oniguruma / UTS #18 \w: Alphabetic | Mark | Decimal_Number | Connector_Punctuation |
        # Join_Control. Alphabetic = L | Nl | Other_Alphabetic (the non-L/M ones are the So below).
        if (cat[0] in "LM" or cat in ("Nd", "Nl", "Pc") or cp in (0x200C, 0x200D)
                or cp in other_alpha_so):
            f |= CLS_WORD
        if f:
            cls[cp] = f
    return cls


class NfcTables:
    """Canonical normalization data restricted to what HF's NFC implements."""

    def __init__(self, ud=unicodedata) -> None:
        _check_unidata(ud)
        ccc_zero = {cp for a, b in _NFC_CCC_ZERO for cp in range(a, b + 1)}
        no_dec = set(_NFC_NO_DECOMPOSITION)
        self.ccc: dict[int, int] = {}
        for cp in _all_scalars():
            c = ud.combining(chr(cp))
            if c and cp not in ccc_zero:
                self.ccc[cp] = c
        # one-level canonical decompositions (Hangul is algorithmic)
        one: dict[int, list[int]] = {}
        for cp in _all_scalars():
            if HANGUL_S_BASE <= cp < HANGUL_S_BASE + HANGUL_S_COUNT or cp in no_dec:
                continue
            d = ud.decomposition(chr(cp))
            if d and not d.startswith("<"):
                one[cp] = [int(x, 16) for x in d.split()]

        def full(cp: int) -> list[int]:
            if cp not in one:
                return [cp]
            return [x for part in one[cp] for x in full(part)]

        self.decomp: dict[int, list[int]] = {}
        for cp in one:
            seq = full(cp)
            self.decomp[cp] = self._reorder(seq)
        # primary composites: canonical decomposition that NFC maps back to the character.
        # Python's NFC is the oracle; the patched-out code points are never composites.
        self.pairs: dict[tuple[int, int], int] = {}
        self.exclusions: list[int] = []
        for cp, parts in sorted(one.items()):
            if ud.normalize("NFC", chr(cp)) == chr(cp):
                if len(parts) != 2:
                    raise AssertionError(f"primary composite U+{cp:04X} with {len(parts)} parts")
                key = (parts[0], parts[1])
                if key in self.pairs:
                    raise AssertionError(f"duplicate composition pair {key}")
                self.pairs[key] = cp
            else:
                self.exclusions.append(cp)

    def ccc_of(self, cp: int) -> int:
        return self.ccc.get(cp, 0)

    def _reorder(self, seq: list[int]) -> list[int]:
        out = list(seq)
        i = 0
        while i < len(out):
            if self.ccc_of(out[i]) == 0:
                i += 1
                continue
            j = i
            while j < len(out) and self.ccc_of(out[j]) != 0:
                j += 1
            out[i:j] = sorted(out[i:j], key=self.ccc_of)  # stable
            i = j
        return out

    def _compose_pair(self, a: int, b: int) -> int | None:
        if HANGUL_L_BASE <= a < HANGUL_L_BASE + HANGUL_L_COUNT and \
                HANGUL_V_BASE <= b < HANGUL_V_BASE + HANGUL_V_COUNT:
            return HANGUL_S_BASE + ((a - HANGUL_L_BASE) * HANGUL_V_COUNT + (b - HANGUL_V_BASE)) * HANGUL_T_COUNT
        if HANGUL_S_BASE <= a < HANGUL_S_BASE + HANGUL_S_COUNT and (a - HANGUL_S_BASE) % HANGUL_T_COUNT == 0 \
                and HANGUL_T_BASE < b < HANGUL_T_BASE + HANGUL_T_COUNT:
            return a + (b - HANGUL_T_BASE)
        return self.pairs.get((a, b))

    def nfc(self, s: str) -> str:
        """Reference NFC with these tables (used by --verify only)."""
        buf: list[int] = []
        for ch in s:
            cp = ord(ch)
            if HANGUL_S_BASE <= cp < HANGUL_S_BASE + HANGUL_S_COUNT:
                si = cp - HANGUL_S_BASE
                buf.append(HANGUL_L_BASE + si // HANGUL_N_COUNT)
                buf.append(HANGUL_V_BASE + (si % HANGUL_N_COUNT) // HANGUL_T_COUNT)
                if si % HANGUL_T_COUNT:
                    buf.append(HANGUL_T_BASE + si % HANGUL_T_COUNT)
            else:
                buf.extend(self.decomp.get(cp, (cp,)))
        buf = self._reorder(buf)
        if not buf:
            return ""
        out = [buf[0]]
        starter = 0
        last_class = self.ccc_of(buf[0])
        if last_class != 0:
            last_class = 256  # no starter yet: nothing can compose
        for cp in buf[1:]:
            c = self.ccc_of(cp)
            comp = self._compose_pair(out[starter], cp) if (last_class < c or last_class == 0) else None
            if comp is not None:
                out[starter] = comp
                continue
            if c == 0:
                starter = len(out)
            last_class = c
            out.append(cp)
        return "".join(map(chr, out))


# ---------------------------------------------------------------------------------------------
# Blob writer
# ---------------------------------------------------------------------------------------------

def _u32s(*vals: int) -> bytes:
    return struct.pack(f"<{len(vals)}I", *vals)


def _pack_records(records, fields: int) -> bytes:
    flat = [v for r in records for v in r]
    assert len(flat) == len(records) * fields
    return _u32s(len(records)) + struct.pack(f"<{len(flat)}I", *flat)


def export_tokenizer_blob(tokenizer_json_path: str, unicode_db=unicodedata) -> bytes:
    """Build the tokenizer blob for a HF tokenizer.json (layout: see module docstring).

    `unicode_db` is the unicodedata-compatible module the tables are generated from (tests use
    the `unicodedata2` package to check that other Unicode versions export identical tables).
    """
    ud = unicode_db
    with open(tokenizer_json_path, "rb") as f:
        raw_json = f.read()
    tj = json.loads(raw_json.decode("utf-8"))
    opts = _validate_config(tj)
    model = tj["model"]

    # --- vocabulary ------------------------------------------------------------------------
    vocab: dict[str, int] = model["vocab"]
    n_base = len(vocab)
    if sorted(vocab.values()) != list(range(n_base)):
        raise ValueError("BPE vocab ids are not the contiguous range [0, len(vocab))")
    id_bytes: dict[int, bytes] = {}
    seen: dict[bytes, int] = {}
    for tok, i in vocab.items():
        rb = _bytelevel_to_raw(tok)
        if rb in seen:
            raise ValueError(f"vocab ids {seen[rb]} and {i} have the same bytes {rb!r}")
        seen[rb] = i
        id_bytes[i] = rb
    for b in range(256):
        if bytes([b]) not in seen:
            raise ValueError(f"byte 0x{b:02X} has no single-byte token")

    # --- added tokens ----------------------------------------------------------------------
    # HF does not trust the "id" fields: it re-adds the tokens in order, giving a token whose
    # content is a vocab key that vocab id and every other one the next free id. Refuse files
    # where the two disagree instead of silently picking one.
    added = []
    hf_ids: dict[str, int] = {}
    for t in tj.get("added_tokens", []):
        if t["content"] in hf_ids:
            continue
        if t["content"] in vocab:
            hf_ids[t["content"]] = vocab[t["content"]]
        else:
            top = max(hf_ids.values(), default=-1)
            hf_ids[t["content"]] = top + 1 if top >= n_base else n_base
        if hf_ids[t["content"]] != int(t["id"]):
            raise ValueError(f"added token {t['content']!r} has id {t['id']} in tokenizer.json but HF assigns "
                             f"{hf_ids[t['content']]}")
    for t in tj.get("added_tokens", []):
        content = t["content"].encode("utf-8")
        if not content:
            raise ValueError(f"empty added token {t!r}")
        i = int(t["id"])
        if i < n_base and id_bytes[i] != content:
            raise ValueError(f"added token {t!r} reuses id {i} of a different vocab token")
        if i >= n_base:
            if i in id_bytes and id_bytes[i] != content:
                raise ValueError(f"two added tokens share id {i}")
            id_bytes[i] = content
        flags = ((ADDED_SPECIAL if t.get("special") else 0) | (ADDED_LSTRIP if t.get("lstrip") else 0)
                 | (ADDED_RSTRIP if t.get("rstrip") else 0)
                 | (ADDED_SINGLE_WORD if t.get("single_word") else 0)
                 | (ADDED_NORMALIZED if t.get("normalized") else 0))
        added.append((i, flags, content))
    n_ids = max(id_bytes) + 1

    offsets = [0]
    pool = bytearray()
    for i in range(n_ids):
        pool += id_bytes.get(i, b"")
        offsets.append(len(pool))
    vocb = _u32s(n_ids, n_base) + struct.pack(f"<{n_ids + 1}I", *offsets) + bytes(pool)

    # --- merges ----------------------------------------------------------------------------
    merges = []
    seen_pairs = set()
    for m in model["merges"]:
        a, b = m.split(" ") if isinstance(m, str) else m
        if a not in vocab or b not in vocab or a + b not in vocab:
            raise ValueError(f"merge {m!r} references tokens outside the vocab")
        pair = (vocab[a], vocab[b])
        if pair in seen_pairs:
            raise ValueError(f"duplicate merge {m!r}")
        seen_pairs.add(pair)
        merges.append((vocab[a], vocab[b], vocab[a + b]))
    merg = _pack_records(merges, 3)

    apool = bytearray()
    arecs = []
    for i, flags, content in added:
        arecs.append((i, flags, len(apool), len(content)))
        apool += content
    addt = _pack_records(arecs, 4) + bytes(apool)

    # --- Unicode ---------------------------------------------------------------------------
    cls = build_regex_classes(ud)
    ucls = _pack_records(_ranges(cls), 3)
    nfc = NfcTables(ud)
    uccc = _pack_records(_ranges(nfc.ccc), 3)
    drecs, dpool = [], []
    for cp in sorted(nfc.decomp):
        seq = nfc.decomp[cp]
        drecs.append((cp, len(dpool), len(seq)))
        dpool.extend(seq)
    udec = _pack_records(drecs, 3) + struct.pack(f"<{len(dpool)}I", *dpool)
    ucmp = _pack_records(sorted((a, b, c) for (a, b), c in nfc.pairs.items()), 3)
    uexc = _u32s(len(nfc.exclusions)) + struct.pack(f"<{len(nfc.exclusions)}I", *nfc.exclusions)

    n_special = sum(1 for a in added if a[1] & ADDED_SPECIAL)
    meta_items = [
        ("format", "qwen38-tokenizer"),
        ("format_version", str(FORMAT_VERSION)),
        ("exporter", "tools/tokenizer_export.py"),
        ("source_sha256", hashlib.sha256(raw_json).hexdigest()),
        ("model", "BPE"),
        ("normalizer", opts["normalizer"]),
        ("pre_tokenizer", "Split(Isolated) + ByteLevel(add_prefix_space=false, use_regex=false)"),
        ("ignore_merges", "true" if opts["ignore_merges"] else "false"),
        ("n_ids", str(n_ids)),
        ("n_base", str(n_base)),
        ("n_merges", str(len(merges))),
        ("n_added", str(len(added))),
        ("n_special", str(n_special)),
        ("unidata_version", ud.unidata_version),
        ("regex_unicode", "unicodedata %s + the letters/marks/numbers of Unicode 16.0.0 (HF Oniguruma)"
         % ud.unidata_version),
        ("nfc_unicode", "unicodedata %s restricted to Unicode 9.0.0 behaviour (HF "
         "unicode-normalization-alignments): ccc 0 for marks encoded later, no decomposition for %s "
         "and the Unicode 16.0 decompositions" % (ud.unidata_version,
                                                   ",".join(f"U+{cp:04X}" for cp in _NFC_NO_DECOMPOSITION[:1]))),
        ("hf_reference", _HF_REFERENCE),
    ]
    for k, v in meta_items:
        if "\n" in k + v or "=" in k:
            raise AssertionError("bad META entry")
    meta = "".join(f"{k}={v}\n" for k, v in meta_items).encode("utf-8")

    sections = [
        (b"META", meta), (b"PATN", QWEN_SPLIT_PATTERN.encode("utf-8")), (b"VOCB", vocb),
        (b"MERG", merg), (b"ADDT", addt), (b"UCLS", ucls), (b"UCCC", uccc), (b"UDEC", udec),
        (b"UCMP", ucmp), (b"UEXC", uexc),
    ]
    header_len = 16 + 24 * len(sections)
    pos = (header_len + 7) & ~7
    directory = bytearray()
    body = bytearray(pos - header_len)
    for tag, data in sections:
        directory += tag + struct.pack("<IQQ", 0, pos, len(data))
        body += data
        pos += len(data)
        pad = (-pos) % 8
        body += b"\0" * pad
        pos += pad
    return MAGIC + _u32s(FORMAT_VERSION, len(sections)) + bytes(directory) + bytes(body)


# ---------------------------------------------------------------------------------------------
# Verification against the installed HF tokenizers (optional)
# ---------------------------------------------------------------------------------------------

def verify_against_hf(tokenizer_json_path: str, verbose: bool = True) -> int:
    """Probe HF's regex and NFC for every scalar value; return the number of disagreements."""
    import random

    from tokenizers import Regex, normalizers, pre_tokenizers

    def log(msg: str) -> None:
        if verbose:
            print(msg, file=sys.stderr)

    with open(tokenizer_json_path, encoding="utf-8") as f:
        tj = json.load(f)
    _validate_config(tj)
    errors = 0

    # 1. split-regex character classes. Probe A = "a" c "a", probe B = "!" c "!", separated by
    # a digit (always its own piece). The piece counts identify the class of c.
    split = pre_tokenizers.Split(Regex(QWEN_SPLIT_PATTERN), behavior="isolated", invert=False)
    cls = build_regex_classes()

    def expected_sig(cp: int) -> tuple[int, int]:
        f = cls.get(cp, 0)
        if cp in (0x0A, 0x0D):
            return (3, 2)
        if cp == 0x20:
            return (2, 2)
        if f & (CLS_L | CLS_M):
            return (1, 2)
        if f & CLS_N:
            return (3, 3)
        if f & CLS_WS:
            return (2, 3)
        return (2, 1)

    cps = list(_all_scalars())
    bad_cls = []
    for i in range(0, len(cps), 50000):
        chunk = cps[i:i + 50000]
        probes, bounds, pos = [], [], 0
        for cp in chunk:
            for p in ("a" + chr(cp) + "a", "!" + chr(cp) + "!"):
                probes.append(p)
                bounds.append((pos, pos + len(p)))
                pos += len(p) + 1
        counts = [0] * len(probes)
        bi = 0
        for _, (a, _b) in split.pre_tokenize_str("1".join(probes)):
            while bi < len(bounds) and bounds[bi][1] <= a:
                bi += 1
            if bi < len(bounds) and bounds[bi][0] <= a:
                counts[bi] += 1
        for k, cp in enumerate(chunk):
            if (counts[2 * k], counts[2 * k + 1]) != expected_sig(cp):
                bad_cls.append(cp)
    log(f"regex classes: {len(bad_cls)} disagreements"
        + (": " + " ".join(f"U+{c:04X}" for c in bad_cls[:20]) if bad_cls else ""))
    errors += len(bad_cls)

    # contractions: (?i:...) accepts exactly these case variants (U+017F folds to 's').
    probes = {"'" + c + "a": ["'" + c, "a"] for c in "sStTmMdD\U0000017f"}
    probes.update({"'" + a + b + "a": ["'" + a + b, "a"] for a in "rRvV" for b in "eE"})
    probes.update({"'" + a + b + "a": ["'" + a + b, "a"] for a in "lL" for b in "lL"})
    probes.update({"'" + c + "a": ["'" + c + "a"] for c in "kKx\U000000df\U00000130\U00000131"})  # letters that must not match
    bad = [p for p, want in probes.items() if [x for x, _ in split.pre_tokenize_str(p)] != want]
    log(f"contraction case folding: {len(bad)} disagreements {bad[:10]}")
    errors += len(bad)

    # added-token flags: the \w test of single_word and the whitespace stripped by lstrip/rstrip,
    # probed with synthetic added tokens on a copy of the tokenizer.
    from tokenizers import AddedToken, Tokenizer
    tok = Tokenizer.from_file(tokenizer_json_path)
    tok.add_tokens([AddedToken("qqzz", single_word=True, normalized=False),
                    AddedToken("ll~~", lstrip=True, normalized=False),
                    AddedToken("rr~~", rstrip=True, normalized=False)])
    sw, ls, rs = tok.token_to_id("qqzz"), tok.token_to_id("ll~~"), tok.token_to_id("rr~~")
    chars = [chr(cp) for cp in cps if cp != 0]
    enc = tok.encode_batch([c + "qqzz " for c in chars], add_special_tokens=False)
    bad = [c for c, e in zip(chars, enc) if (sw not in e.ids) != bool(cls.get(ord(c), 0) & CLS_WORD)]
    enc = tok.encode_batch([" qqzz" + c for c in chars], add_special_tokens=False)
    bad += [c for c, e in zip(chars, enc) if (sw not in e.ids) != bool(cls.get(ord(c), 0) & CLS_WORD)]
    log(f"single_word \\w class: {len(bad)} disagreements {[hex(ord(c)) for c in bad[:10]]}")
    errors += len(bad)
    enc = tok.encode_batch(["x" + c + "ll~~" for c in chars], add_special_tokens=False)
    bad = [c for c, e in zip(chars, enc)
           if (e.offsets[e.ids.index(ls)][0] == 1) != bool(cls.get(ord(c), 0) & CLS_WS)]
    enc = tok.encode_batch(["rr~~" + c + "x" for c in chars], add_special_tokens=False)
    bad += [c for c, e in zip(chars, enc)
            if (e.offsets[e.ids.index(rs)][1] > 4) != bool(cls.get(ord(c), 0) & CLS_WS)]
    log(f"lstrip/rstrip whitespace: {len(bad)} disagreements {[hex(ord(c)) for c in bad[:10]]}")
    errors += len(bad)

    # 2. NFC
    hf_nfc = normalizers.NFC()
    nfc = NfcTables()
    sep = "\u0000"

    def run(strs: list[str], what: str) -> int:
        n_bad = 0
        for i in range(0, len(strs), 100000):
            part = strs[i:i + 100000]
            got = hf_nfc.normalize_str(sep.join(part)).split(sep)
            assert len(got) == len(part)
            for s, g in zip(part, got):
                if g != nfc.nfc(s):
                    n_bad += 1
                    if n_bad <= 10:
                        log(f"  NFC {what}: {[hex(ord(c)) for c in s]} hf={[hex(ord(c)) for c in g]} "
                            f"ref={[hex(ord(c)) for c in nfc.nfc(s)]}")
        log(f"NFC {what}: {n_bad} disagreements over {len(strs)} probes")
        return n_bad

    chars = [chr(cp) for cp in cps if cp != 0]
    errors += run(chars, "single")
    errors += run(["a" + c for c in chars], "after 'a'")
    errors += run(["\U00000345" + c for c in chars], "after U+0345 (ccc 240)")
    errors += run([c + "\U00000334" for c in chars], "before U+0334 (ccc 1)")
    reps: dict[int, int] = {}
    for cp, c in sorted(nfc.ccc.items()):
        if cp not in nfc.decomp:
            reps.setdefault(c, cp)
    marks = sorted(set(nfc.ccc) | {cp for a, b in _NFC_CCC_ZERO for cp in range(a, b + 1)})
    errors += run([chr(m) + chr(r) for m in marks for r in reps.values()], "mark order")
    errors += run([chr(a) + chr(b) for (a, b) in nfc.pairs] + [chr(0x11935) + chr(0x11930)], "pairs")
    rng = random.Random(1234)
    pool = [chr(c) for c in list(nfc.ccc)[:400]] + [chr(c) for c in range(0x1100, 0x1200)] + list(
        "aeiouAEIOUsSkK\U00000b47\U00000b3e\U00000cc6\U00000cd5\U00000dd9\U00000dcf\U00001025\U0000102e\U0000ac00\U0000ac01") + [
        chr(c) for c in list(nfc.decomp)[:300]]
    errors += run(["".join(rng.choice(pool) for _ in range(rng.randint(1, 12))) for _ in range(200000)],
                  "random sequences")
    log(f"verify: {errors} disagreements in total")
    return errors


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}
    if len(args) != 2 or flags - {"--verify"}:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        print("usage: tokenizer_export.py <tokenizer.json> <out.bin> [--verify]", file=sys.stderr)
        return 2
    blob = export_tokenizer_blob(args[0])
    with open(args[1], "wb") as f:
        f.write(blob)
    print(f"wrote {args[1]}: {len(blob)} bytes (unidata {unicodedata.unidata_version})")
    if "--verify" in flags:
        return 1 if verify_against_hf(args[0]) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
