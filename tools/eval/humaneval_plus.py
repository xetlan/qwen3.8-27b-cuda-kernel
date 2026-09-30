#!/usr/bin/env python3
"""HumanEval+ (evalplus) through `qwen38 --batch`: a complete example of the file-based evaluation flow.

    pip install evalplus
    python tools/eval/humaneval_plus.py prepare --out he_in.jsonl            # thinking on (default)
    ./build/src/qwen38 --model models/qwen3.8-27b.q38 --batch he_in.jsonl --out he_out.jsonl --max-batch 8
    python tools/eval/humaneval_plus.py score --completions he_out.jsonl --samples he_samples.jsonl
    evalplus.evaluate --dataset humaneval --samples he_samples.jsonl

Run the same three steps for a reference configuration (another recipe, or the BF16 model served
elsewhere with the same prompts) and compare pass@1: docs/QUALITY.md explains the 96% criterion.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

INSTRUCTION = (
    "Complete the following Python function. Reply with the complete function (including the "
    "signature and any imports it needs) in a single ```python code block.\n\n"
)


def prepare(args):
    from evalplus.data import get_human_eval_plus

    problems = get_human_eval_plus()
    with open(args.out, "w") as f:
        for task_id, p in problems.items():
            req = {
                "id": task_id,
                "prompt": INSTRUCTION + p["prompt"],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": 0.95,
                "top_k": 20,
                "enable_thinking": not args.no_think,
                "seed": 1234,
            }
            f.write(json.dumps(req) + "\n")
    print(f"wrote {len(problems)} requests to {args.out}")


CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        # the longest block is the solution; short ones are usually examples
        return max(blocks, key=len)
    return text


def score(args):
    n = 0
    finish = {}
    with open(args.completions) as f, open(args.samples, "w") as out:
        for line in f:
            r = json.loads(line)
            finish[r["finish_reason"]] = finish.get(r["finish_reason"], 0) + 1
            out.write(json.dumps({"task_id": r["id"], "solution": extract_code(r["text"])}) + "\n")
            n += 1
    print(f"wrote {n} samples to {args.samples}; finish reasons: {finish}")
    if finish.get("length"):
        print("note: some completions hit max_tokens; raise it (thinking mode reasons at length)",
              file=sys.stderr)
    print(f"now run: evalplus.evaluate --dataset humaneval --samples {args.samples}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--no-think", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--completions", required=True)
    s.add_argument("--samples", required=True)
    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else score(args)


if __name__ == "__main__":
    main()
