#!/usr/bin/env python3
"""Per-kernel register / shared-memory / spill table from a build log made with -DQWEN38_PTXAS_VERBOSE=ON.

    python tools/ptxas_report.py build-cuda/build.log
"""
import re
import sys


def short_name(name: str) -> str:
    m = re.search(r"qgemm_kernelILi(\d)ENS1_3CfgILi(\d)ELi(\d)ELi(\d)ELi(\d)ELi(\d+)ELi(\d)ELi(\d)", name)
    if m:
        b, wm, wn, wk, mt, tk, st, _ = m.groups()
        return f"qgemm bq{b} m{int(wm) * int(mt) * 16}_n{int(wn) * 64}_k{tk}_s{st} (WK{wk})"
    for key in ["attn_prefill_kernel", "attn_decode_kernel", "attn_combine_kernel", "gdn_recurrence_kernel",
                "gdn_conv_kernel", "gdn_ab_small_kernel", "gdn_ab_tiled_kernel", "sample_kernel",
                "target_logprob_kernel", "qk_norm_rope_kernel", "embed_kernel"]:
        if key in name:
            out = key.replace("_kernel", "")
            g = re.search(r"attn_decode_kernelILi(\d+)ELNS0_7KvDtypeE(\d)", name)
            if g:
                return f"{out} group{g.group(1)} {'int8' if g.group(2) == '0' else 'bf16'}"
            g = re.search(r"attn_prefill_kernelILNS0_7KvDtypeE(\d)", name)
            if g:
                return f"{out} {'int8' if g.group(1) == '0' else 'bf16'}"
            return out
    return name


def main():
    lines = open(sys.argv[1]).read().splitlines()
    rows = {}
    current = None
    for line in lines:
        m = re.search(r"Compiling entry function '([^']+)' for 'sm_\d+'", line)
        if m:
            current = short_name(m.group(1))
            rows[current] = {}
            continue
        if current is None:
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m:
            rows[current]["stack"] = int(m.group(1))
            rows[current]["spill"] = int(m.group(2)) + int(m.group(3))
        m = re.search(r"Used (\d+) registers", line)
        if m:
            rows[current]["regs"] = int(m.group(1))
            s = re.search(r"(\d+) bytes smem", line)
            rows[current]["smem"] = int(s.group(1)) if s else 0
            current = None
    print("| kernel | registers | static smem (B) | stack (B) | spill (B) |")
    print("|---|---:|---:|---:|---:|")
    for k in sorted(rows):
        r = rows[k]
        print(f"| {k} | {r.get('regs', '?')} | {r.get('smem', 0)} | {r.get('stack', 0)} | {r.get('spill', 0)} |")


if __name__ == "__main__":
    main()
