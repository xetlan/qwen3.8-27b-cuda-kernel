#!/usr/bin/env python3
"""Generate tokenizer test cases with Hugging Face `tokenizers` as the ground truth.

    python tests/host/tokenizer_cases.py <tokenizer.json> <out.bin> [--seed N]

The corpus is deterministic for a given seed and Python installation (a few long documents are
taken from the Python standard library sources). Categories: prose, code (Python, C++, JS/TS,
Rust, Go, shell, JSON, YAML, TOML, Markdown, diff, SQL, HTML/CSS, LaTeX, logs, ...) with
tab / CRLF / trailing-space variants, ~50 languages and scripts (plus NFD variants), emoji,
Unicode normalization edge cases, numbers, contractions, whitespace, added tokens, invalid UTF-8,
huge single pre-tokens, documents > 50 KB, thousands of seeded random strings, and a sweep that
puts every code point of planes 0-3 and 14 (and a sample of the others) through the pipeline.

Cases are also run with allow_added_tokens = false; their expected ids come from the same
tokenizer with its added tokens removed (plain BPE over the text).

File format (little-endian), read by tests/host/test_tokenizer.cpp:

    8 bytes   magic b"QW38TCS1"
    u32       format version (1)
    u32       number of cases
    per case:
      u32               flags: bit 0 = allow_added_tokens
      u32 n, n bytes    label (ASCII, "category/detail")
      u32 n, n bytes    input bytes (may be invalid UTF-8)
      u32 n, n x i32    expected ids = HF encode(input.decode("utf-8", "replace"),
                        add_special_tokens=False).ids
      u32 n, n bytes    expected decode(ids): the input after U+FFFD replacement and NFC, where
                        NFC is applied per segment between added tokens (as HF does)
"""

from __future__ import annotations

import difflib
import json
import os
import random
import re
import struct
import sys
import sysconfig
import time
import unicodedata

MAGIC = b"QW38TCS1"
VERSION = 1


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))  # printable Latin-1
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


# =============================================================================================
# Corpus: prose
# =============================================================================================

PROSE = [
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs!",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age "
    "of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season "
    "of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair.",
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, "
    "conceived in Liberty, and dedicated to the proposition that all men are created equal.",
    "We hold these truths to be self-evident, that all men are created equal, that they are endowed "
    "by their Creator with certain unalienable Rights, that among these are Life, Liberty and the "
    "pursuit of Happiness.",
    "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my "
    "purse, and nothing particular to interest me on shore, I thought I would sail about a little "
    "and see the watery part of the world.",
    "In 2026, the team shipped v3.2.1 of the inference engine; throughput rose by 37.5% (from 812 "
    "tok/s to 1,116 tok/s) on a single RTX 3090, while p99 latency fell to 41 ms.",
    "Dr. Smith, Ph.D., arrived at 9:30 a.m. on Sept. 30th, i.e. yesterday, e.g. before the U.S.A. "
    "delegation; see pp. 12-15 & Fig. 3(b) for details.",
    "\"Well,\" she said, 'I don't know what you're talking about.' He'd already left; they'll "
    "follow, we've waited, and I'm tired. Isn't it odd? Wouldn't you agree? You'd think so.",
    "“Smart quotes” and ‘single ones’ aren’t the same as \"straight\" quotes—nor is an en dash "
    "(–) the same as a hyphen (-) or a minus sign (−). Ellipsis… or three dots...",
    "Please e-mail john.doe+filter@example.co.uk or visit https://www.example.com/path/to/page"
    "?query=hello%20world&lang=en#section-2 before 11:59 PM (UTC-05:00).",
    "ALL CAPS SENTENCES ARE LOUD. Title Case Sentences Look Like This. mIxEd CaSe iS aNnOyInG.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut "
    "labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco.",
    "The tokenizer splits text into pieces:   multiple   spaces,\ttabs,\nnewlines,\n\nand\r\nCRLF "
    "line endings   \n   should all survive a round trip.",
    "Q: What's 2+2? A: 4. Q: What's √16? A: 4 as well. Q: And 10^3? A: 1000 (one thousand).",
    "Once upon a time, in a land far, far away, there lived a dragon named Ember. Ember's scales "
    "shimmered like molten copper, and her roar could be heard from the mountains to the sea.",
    "BREAKING: Markets rallied 2.3% on Tuesday after the central bank held rates steady at 4.25%; "
    "the S&P 500 closed at 5,847.12, the Nasdaq at 18,652.30 and the Dow at 42,114.40.",
    "Step 1) Preheat the oven to 180°C (356°F). Step 2) Mix 250 g flour, 2 eggs & ½ cup milk. "
    "Step 3) Bake for 25–30 min. Serves 4–6 people.",
    "Hmm... well... I mean—okay, fine!!! Are you sure??? Yes!?! No?!",
    "   Indented paragraph with leading spaces and trailing spaces.   ",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
]

CONTRACTIONS = [
    "I'm you're he's she's it's we're they're I've you've we've they've I'd you'd he'd she'd we'd "
    "they'd I'll you'll he'll she'll we'll they'll isn't aren't wasn't weren't don't doesn't didn't",
    "I'M YOU'RE HE'S IT'S WE'RE THEY'VE I'D YOU'LL ISN'T DON'T CAN'T WON'T SHOULDN'T",
    "I'M You'Re He'S It'S we'RE they'Ve i'D you'Ll YOU'lL we'lL isn'T DoN't",
    "'s 't 're 've 'm 'll 'd 'S 'T 'RE 'VE 'M 'LL 'D 'Re 'rE 'Ve 'vE 'Ll 'lL",
    "'sa 'ta 'rea 'vea 'ma 'lla 'da 's1 't! 're? 'll. 'd\n 'm\t",
    "it’s don’t they’re we’ll I’m (curly apostrophes are not contractions)",
    "it‘s don`t they´re we′ll I＇m",
    "rock'n'roll o'clock 'twas y'all'd've fo'c's'le ma'am ne'er-do-well",
    "The dog's bone, the dogs' bones, James's book, the children's toys.",
    "'quoted' ''double'' '''triple''' ' ' '' ''' 'a'b'c'",
    "Long s: it'ſ, IT'ſ, 'ſa, ſ's; Kelvin: '\u212a 'k 'K; dotted: 'İ 'ı",
    "Yáʼátʼééh ʻŌlelo Hawaiʻi (modifier letters are letters, not apostrophes)",
    "'", "''", "'s", "'S", "'ll", "'LL", "'l", "'r", "'v", "'re", "x'", "x's", "x'S!",
    "don't\n'tis\n'Tis\n'TIS",
    "12's 3'd 4'll a'1 é's ü'LL",
]

# =============================================================================================
# Corpus: code
# =============================================================================================

CODE = {
    "python": '''#!/usr/bin/env python3
"""Tiny LRU cache with statistics."""
from __future__ import annotations

import functools
from collections import OrderedDict
from typing import Callable, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int = 128) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity!r}")
        self._data: OrderedDict[K, V] = OrderedDict()
        self.capacity = capacity
        self.hits = self.misses = 0

    def get(self, key: K, default: V | None = None) -> V | None:
        try:
            value = self._data.pop(key)
        except KeyError:
            self.misses += 1
            return default
        self._data[key] = value  # move to the end
        self.hits += 1
        return value

    def put(self, key: K, value: V) -> None:
        self._data.pop(key, None)
        self._data[key] = value
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def __repr__(self) -> str:
        ratio = self.hits / max(1, self.hits + self.misses)
        return f"<LRUCache {len(self._data)}/{self.capacity} hit={ratio:.2%}>"


def memoize(fn: Callable[..., V]) -> Callable[..., V]:
    cache: dict[tuple, V] = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if (hit := cache.get(key)) is not None:
            return hit
        cache[key] = result = fn(*args, **kwargs)
        return result

    return wrapper


async def fetch_all(urls: list[str]) -> dict[str, int]:
    results = {u: len(u) for u in urls if u.startswith(("http://", "https://"))}
    squares = [x ** 2 for x in range(10) if x % 2 == 0]
    café = "naïve résumé"  # unicode identifiers and strings
    print(f"{café!r:>20} {squares} {results}", end="\\n", flush=True)
    return results


if __name__ == "__main__":
    c = LRUCache[str, int](capacity=2)
    for k, v in zip("abc", range(3)):
        c.put(k, v)
    assert c.get("a") is None and c.get("c") == 2, c
    print(c)
''',
    "cpp": r'''#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace demo {

template <typename T, std::size_t N = 16>
class RingBuffer {
 public:
  bool push(const T& v) noexcept {
    if (size_ == N) return false;
    data_[(head_ + size_++) % N] = v;
    return true;
  }
  [[nodiscard]] std::size_t size() const { return size_; }

 private:
  T data_[N]{};
  std::size_t head_ = 0, size_ = 0;
};

struct Node {
  int id;
  std::vector<std::pair<int, float>> edges;  // (target, weight)
};

inline uint64_t fnv1a(std::string_view s) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : s) { h ^= c; h *= 0x100000001b3ull; }
  return h;
}

}  // namespace demo

int main(int argc, char** argv) {
  std::unordered_map<std::string, int> counts;
  for (int i = 1; i < argc; ++i) ++counts[argv[i]];
  auto by_count = [](const auto& a, const auto& b) { return a.second > b.second; };
  std::vector<std::pair<std::string, int>> v(counts.begin(), counts.end());
  std::sort(v.begin(), v.end(), by_count);
  for (const auto& [word, n] : v) std::cout << word << ": " << n << '\n';
  auto p = std::make_unique<demo::RingBuffer<int>>();
  p->push(42);
  const char* raw = R"(raw "string" with \n no escapes)";
  std::cout << raw << " " << demo::fnv1a("hello") << std::endl;
  /* block comment
     spanning lines */
  return v.empty() ? 1 : 0;  // exit code
}
''',
    "javascript": '''// @ts-check
import { readFile } from "node:fs/promises";

export const debounce = (fn, ms = 250) => {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
};

export async function loadConfig(path) {
  const text = await readFile(path, "utf8");
  const cfg = JSON.parse(text);
  const port = cfg?.server?.port ?? 8080;
  const re = /^(?<user>[\\w.+-]+)@(?<host>[\\w-]+\\.[\\w.]+)$/u;
  if (!re.test(cfg.admin)) throw new Error(`bad admin e-mail: ${cfg.admin}`);
  return { ...cfg, port, url: `http://localhost:${port}/` };
}

class Counter extends EventTarget {
  #n = 0;
  increment(by = 1) { this.#n += by; this.dispatchEvent(new CustomEvent("change", { detail: this.#n })); }
  get value() { return this.#n; }
}

const App = ({ items }) => (
  <ul className="list">
    {items.map((it, i) => <li key={i}>{it.name} &mdash; {it.price.toFixed(2)} €</li>)}
  </ul>
);

console.log([1, 2, 3].reduce((a, b) => a + b, 0) === 6, typeof null, 0.1 + 0.2 !== 0.3);
''',
    "typescript": '''type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E };

interface User {
  readonly id: number;
  name: string;
  email?: string;
  roles: Array<"admin" | "viewer">;
}

export function parseUser(raw: unknown): Result<User> {
  if (typeof raw !== "object" || raw === null) return { ok: false, error: new TypeError("not an object") };
  const u = raw as Partial<User>;
  return u.id !== undefined && u.name ? { ok: true, value: u as User } : { ok: false, error: new Error("missing fields") };
}

export const enum Level { Debug = 0, Info, Warn, Error }
const byId = new Map<number, User>();
function assertNever(x: never): never { throw new Error(`unexpected: ${x}`); }
''',
    "rust": r'''use std::collections::HashMap;
use std::fmt::{self, Display};

#[derive(Debug, Clone, PartialEq)]
pub enum Token<'a> {
    Word(&'a str),
    Number(f64),
    Punct(char),
}

impl<'a> Display for Token<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Word(w) => write!(f, "{w}"),
            Token::Number(n) => write!(f, "{n:.3}"),
            Token::Punct(c) => write!(f, "'{c}'"),
        }
    }
}

pub fn count_words(text: &str) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for w in text.split_whitespace().map(|w| w.trim_matches(|c: char| !c.is_alphanumeric())) {
        *map.entry(w.to_lowercase()).or_insert(0) += 1;
    }
    map
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = b"\x00\xffraw bytes";
    let raw = r#"a "raw" string"#;
    let n: u64 = "42".parse()?;
    println!("{:?} {} {} {}", bytes, raw, n, '\n' as u32);
    let v: Vec<_> = (0..10).filter(|x| x % 3 == 0).collect();
    assert_eq!(v, vec![0, 3, 6, 9]);
    Ok(())
}
''',
    "go": '''package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
)

type Server struct {
	mu    sync.Mutex
	count map[string]int
}

func (s *Server) Handle(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.count[r.URL.Path]++
	if err := json.NewEncoder(w).Encode(s.count); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func main() {
	ch := make(chan int, 3)
	go func() { for i := 0; i < 3; i++ { ch <- i * i }; close(ch) }()
	for v := range ch {
		fmt.Printf("%d\\t%v\\n", v, v%2 == 0)
	}
	s := &Server{count: map[string]int{}}
	http.HandleFunc("/", s.Handle)
	_ = http.ListenAndServe(":8080", nil)
}
''',
    "shell": r'''#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

BUILD_DIR="${BUILD_DIR:-build}"
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu)

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

if [[ ! -d "$BUILD_DIR" ]]; then
  log "configuring into $BUILD_DIR"
  cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DQWEN38_ENABLE_CUDA=OFF
fi
cmake --build "$BUILD_DIR" -j"$JOBS" 2>&1 | tee build.log | grep -E 'warning|error' || true

for f in testdata/*.json; do
  [ -f "$f" ] || continue
  n=$(wc -l < "$f"); echo "$f: $n lines"
done

cat <<EOF > summary.txt
jobs=$JOBS
dir=$BUILD_DIR
EOF
awk -F, 'NR > 1 { s += $3 } END { print "total:", s }' data.csv | sed 's/total/TOTAL/'
find . -name '*.tmp' -mtime +7 -exec rm -f {} \;
echo "done: $? ~ ${#BUILD_DIR} chars" && exit 0
''',
    "json": json.dumps({
        "model": "Qwen3.8-27B", "max_tokens": 4096, "temperature": 0.7, "top_p": 0.95, "stream": True,
        "stop": ["<|im_end|>", "\n\n\n"], "logit_bias": {"248046": -100.0}, "seed": None,
        "messages": [{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": "Translate \"héllo wörld\" to 日本語, please. Tab:\tNL:\n"}],
        "numbers": [0, -1, 3.14159, 2.5e-10, 1e308, -0.0, 12345678901234567890],
        "nested": {"a": {"b": {"c": [True, False, None, [], {}]}}},
    }, indent=2, ensure_ascii=False),
    "json_compact": json.dumps({"id": "chatcmpl-9x8y7z", "object": "chat.completion", "created": 1759248000,
                                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Sure! é中\U0001F600"},
                                             "finish_reason": "stop"}], "usage": {"prompt_tokens": 17, "completion_tokens": 42}},
                               separators=(",", ":")),
    "json_ascii": json.dumps({"text": "naïve café — 中文 😀", "esc": "quote\" backslash\\ slash/ \b\f\n\r\t"}),
    "yaml": '''# CI pipeline
name: build-and-test
on:
  push:
    branches: [main, "release/**"]
  pull_request: {}
jobs:
  test:
    runs-on: ubuntu-24.04
    strategy:
      matrix:
        compiler: [gcc-13, clang-18]
        build_type: [Debug, Release]
    steps:
      - uses: actions/checkout@v4
      - name: Configure
        run: |
          cmake -S . -B build \\
            -DCMAKE_CXX_COMPILER=${{ matrix.compiler }} \\
            -DQWEN38_ENABLE_CUDA=OFF
      - run: ctest --test-dir build --output-on-failure  # run all tests
        env: { QWEN38_TOKENIZER_BLOB: testdata/qwen3.8/tokenizer.bin, OMP_NUM_THREADS: "4" }
anchors: &defaults
  retries: 3
  timeout: 1.5e3
other:
  <<: *defaults
  empty: ~
  multi: >
    folded text
    continues here
''',
    "toml": '''[package]
name = "qwen38-tools"
version = "0.1.0"
edition = "2021"
authors = ["Jane Doe <jane@example.com>"]

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", default-features = false, features = ["rt-multi-thread", "macros"] }

[profile.release]
lto = "fat"
codegen-units = 1
opt-level = 3
dates = 1979-05-27T07:32:00-08:00
float = +1.0e-3
hex = 0xDEAD_BEEF
''',
    "markdown": '''# qwen38 — inference engine

> **Status:** experimental. _Use at your own risk._

## Features

- [x] Byte-level BPE tokenizer (`tokenizer.cpp`)
- [ ] CUDA graphs for decode
  1. capture
  2. replay

| Kernel | Time (µs) | Speed-up |
|:-------|----------:|:--------:|
| qgemm  |     123.4 |   2.1×   |
| attn   |      56.7 |   1.4×   |

```bash
$ cmake -S . -B build && cmake --build build -j
```

See [the docs](https://example.com/docs "Docs") or ![logo](img/logo.png).
Inline `code`, **bold**, *italic*, ~~strike~~, <kbd>Ctrl</kbd>+<kbd>C</kbd>, &amp; &lt;tag&gt;.

Footnote[^1].

[^1]: A footnote with a URL <https://example.org>.

---
''',
    "diff": '''diff --git a/src/tokenizer.cpp b/src/tokenizer.cpp
index 3f2a1b4..9c8d7e6 100644
--- a/src/tokenizer.cpp
+++ b/src/tokenizer.cpp
@@ -120,7 +120,9 @@ size_t Tokenizer::Impl::match_at(const uint8_t* p, size_t n, size_t i) const {
   const char32_t c = decode_at(p, i, len);
-  if (c == '\\'') {
+  if (c == '\\'' && j < n) {
+    // contractions
     uint32_t l1;
+    const char32_t c1 = fold(decode_at(p, j, l1));
   }
\\ No newline at end of file
''',
    "sql": '''-- monthly active users
WITH active AS (
  SELECT u.id, date_trunc('month', e.created_at) AS month
  FROM users u
  JOIN events e ON e.user_id = u.id AND e.kind IN ('login', 'purchase')
  WHERE e.created_at >= TIMESTAMP '2026-01-01 00:00:00' AND u.email NOT LIKE '%@test.%'
)
SELECT month, COUNT(DISTINCT id) AS mau, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM active GROUP BY month HAVING COUNT(*) > 10 ORDER BY month DESC LIMIT 12;
INSERT INTO t (a, b) VALUES (1, 'it''s'), (2, NULL);
''',
    "html": '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Test &mdash; Page</title>
  <style>
    body { font: 16px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 42rem; }
    .btn:hover, .btn:focus-visible { background: #0af; color: rgb(255 255 255 / 90%); }
    @media (max-width: 600px) { body { padding: 0 1em; } }
  </style>
</head>
<body>
  <!-- navigation -->
  <nav><a href="/" class="btn">Home</a> | <a href="/about?x=1&amp;y=2">About</a></nav>
  <p>Caf&eacute; &copy; 2026 &#8212; &#x1F600;&nbsp;done.</p>
  <script>document.querySelectorAll("a").forEach(a => a.addEventListener("click", e => e.preventDefault()));</script>
</body>
</html>
''',
    "latex": r'''\documentclass{article}
\usepackage{amsmath}
\begin{document}
\section{Attention}
The scaled dot-product attention is
\begin{equation}
  \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V ,
\end{equation}
where $Q \in \mathbb{R}^{n \times d_k}$ and $\alpha_i = e^{x_i} / \sum_{j=1}^{n} e^{x_j}$.
% a comment
\end{document}
''',
    "makefile": "CXX ?= g++\nCXXFLAGS := -std=c++20 -O2 -Wall -Wextra\n\nall: test_tokenizer\n\n"
                "test_tokenizer: src/tokenizer.o tests/host/test_tokenizer.o\n\t$(CXX) $(CXXFLAGS) -o $@ $^\n\n"
                "%.o: %.cpp\n\t$(CXX) $(CXXFLAGS) -Iinclude -c $< -o $@\n\n.PHONY: clean\nclean:\n\trm -f *.o test_tokenizer\n",
    "dockerfile": "FROM nvidia/cuda:12.6.2-devel-ubuntu24.04 AS build\nARG DEBIAN_FRONTEND=noninteractive\n"
                  "RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build \\\n"
                  "    && rm -rf /var/lib/apt/lists/*\nWORKDIR /src\nCOPY . .\n"
                  "RUN cmake -G Ninja -S . -B build && cmake --build build\nENTRYPOINT [\"/src/build/qwen38\"]\n",
    "log": "2026-09-30T12:34:56.789Z INFO  [main] server listening on 0.0.0.0:8080 (pid=12345)\n"
           "2026-09-30T12:34:57.001Z WARN  [pool-1-thread-3] slow request: GET /api/v1/items?id=42&page=3 took 1532ms\n"
           "2026-09-30T12:35:00.000Z ERROR [worker] java.lang.NullPointerException: Cannot invoke \"String.length()\"\n"
           "\tat com.example.App.run(App.java:42)\n\tat java.base/java.lang.Thread.run(Thread.java:1583)\n"
           "127.0.0.1 - - [30/Sep/2026:12:35:01 +0000] \"POST /login HTTP/1.1\" 302 0 \"-\" \"Mozilla/5.0\"\n",
    "traceback": 'Traceback (most recent call last):\n  File "/app/main.py", line 17, in <module>\n    main()\n'
                 '  File "/app/main.py", line 12, in main\n    x = compute(data["key"])\n'
                 '                        ~~~~^^^^^^^\nKeyError: \'key\'\n',
    "csv": 'id,name,price,tags,notes\n1,"Widget, large",12.50,"a;b;c","He said ""hi"""\n'
           '2,Gadget,7,,\n3,"Thing\nwith newline",0.99,x,"é ü 中"\n',
    "minified_js": '!function(e,t){"object"==typeof exports&&"undefined"!=typeof module?module.exports=t():'
                   '"function"==typeof define&&define.amd?define(t):(e=e||self).lib=t()}(this,function(){'
                   '"use strict";var e=function(e){return e&&e.__esModule?e:{default:e}};return{a:e,b:1e3}});',
    "base64_hex": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==\n"
                  "00000000: 7f45 4c46 0201 0100 0000 0000 0000 0000  .ELF............\n"
                  "00000010: 0300 3e00 0100 0000 1010 0000 0000 0000  ..>.............\n"
                  "sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n",
    "regex": r"^(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+$"
             "\n(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*\n",
}

# =============================================================================================
# Corpus: languages and scripts
# =============================================================================================

LANGUAGES = {
    "german": "Größenwahn ist keine Lösung. Die Straße führt über die Brücke nach Köln; Fußgänger "
              "müssen auf dem Gehweg bleiben. ÄÖÜ äöü ß ẞ.",
    "french": "L'été dernier, nous sommes allés à Montréal. « Bonjour », dit-il, « ça va ? » "
              "Il n'y a pas de quoi. Œuvre, cœur, Noël, naïf.",
    "spanish": "¿Dónde está el baño? ¡Qué sorpresa! El niño comió piña y jamón en el año 2025.",
    "portuguese": "Não há razão para preocupação; a situação está sob controle. Ação, coração, pão, avô.",
    "italian": "Perché l'università è chiusa? Più tardi andrò là, però non so se c'è qualcuno.",
    "polish": "Zażółć gęślą jaźń. Chrząszcz brzmi w trzcinie w Szczebrzeszynie.",
    "czech": "Příliš žluťoučký kůň úpěl ďábelské ódy.",
    "turkish": "İstanbul'da ılık bir akşam. Şişli'de çay içtik. IĞDIR, ığdır, İi Iı.",
    "vietnamese": "Tiếng Việt có nhiều dấu thanh: sắc, huyền, hỏi, ngã, nặng. Người ta nói rằng Hà Nội "
                  "rất đẹp vào mùa thu. Phở bò, bánh mì, cà phê sữa đá.",
    "greek": "Η γρήγορη καφέ αλεπού πηδάει πάνω από τον τεμπέλη σκύλο. Ἐν ἀρχῇ ἦν ὁ λόγος; ᾯ ῼ ΐ.",
    "russian": "Съешь же ещё этих мягких французских булок, да выпей чаю. Ёжик в тумане.",
    "ukrainian": "Гей, хлопці, вставайте! Їжак, ґанок, єнот, ї.",
    "arabic": "مرحبا بالعالم. اللغة العربية جميلة جداً. كَتَبَ الطَّالِبُ الدَّرْسَ في ١٢ سبتمبر ٢٠٢٥.",
    "persian": "زبان فارسی بسیار زیباست. می\u200cخواهم به ایران سفر کنم. ۱۴۰۴",
    "urdu": "اردو ایک خوبصورت زبان ہے۔ میں پاکستان سے ہوں۔",
    "hebrew": "שָׁלוֹם עֲלֵיכֶם. עברית היא שפה עתיקה. בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם.",
    "hindi": "नमस्ते दुनिया। हिन्दी भारत की राजभाषा है। क़लम, ज़रूर, फ़िल्म, ड़, ढ़। ॐ १२३४",
    "bengali": "আমি বাংলায় গান গাই। বাংলা ভাষা সুন্দর। ড় ঢ় য়, কো কৌ।",
    "tamil": "தமிழ் ஒரு பழமையான மொழி. வணக்கம்! கொ கோ கௌ ௐ",
    "telugu": "తెలుగు ఒక ద్రావిడ భాష. నమస్కారం! కై",
    "kannada": "ಕನ್ನಡ ಭಾಷೆ ಸುಂದರವಾಗಿದೆ. ಕೀ ಕೇ ಕೊ ಕೋ",
    "malayalam": "മലയാളം ഒരു ദ്രാവിഡ ഭാഷയാണ്. കൊ കോ കൌ",
    "sinhala": "සිංහල භාෂාව ලස්සනයි. කෙ කේ කො කෝ කෞ",
    "gujarati": "ગુજરાતી ભાષા. નમસ્તે!",
    "punjabi": "ਪੰਜਾਬੀ ਭਾਸ਼ਾ। ਸਤ ਸ੍ਰੀ ਅਕਾਲ। ਖ਼ ਗ਼ ਜ਼ ਫ਼",
    "odia": "ଓଡ଼ିଆ ଭାଷା। କୈ କୋ କୌ",
    "thai": "สวัสดีครับ ภาษาไทยไม่มีการเว้นวรรคระหว่างคำ ผู้ใหญ่หาผ้าใหม่ให้สะใภ้ใช้คล้องคอ ๑๒๓",
    "lao": "ສະບາຍດີ ພາສາລາວ ໜ້າ",
    "khmer": "សួស្តី ភាសាខ្មែរ ជាភាសាផ្លូវការ។",
    "myanmar": "မင်္ဂလာပါ မြန်မာဘာသာ ဦး ၁၂၃",
    "tibetan": "བོད་སྐད་ནི་སྐད་ཡིག་ཅིག་རེད། ཧཱུྃ ཀྵ ཱི ཱུ",
    "georgian": "ქართული ენა ძალიან ლამაზია. Ა Ბ Გ",
    "armenian": "Հայերեն լեզուն գեղեցիկ է։ Բարև ձեզ։",
    "amharic": "ሰላም ለዓለም። የአማርኛ ቋንቋ። ፩፪፫",
    "chinese_simplified": "我们今天去公园散步，天气非常好。人工智能正在改变世界！你知道吗？“引号”与（括号）。",
    "chinese_traditional": "臺灣的夜市非常熱鬧，小吃種類繁多：蚵仔煎、珍珠奶茶、臭豆腐。",
    "japanese": "今日はいい天気ですね。私は東京に住んでいます。カタカナとひらがなを混ぜて書きます。"
                "ｶﾀｶﾅ（半角）もあります。「こんにちは」と言った。ヴァイオリン、が、ぱ。",
    "korean": "안녕하세요. 한국어는 한글로 씁니다. 대한민국의 수도는 서울입니다. 똠방각하, 뷁!",
    "mongolian": "Монгол хэл. ᠮᠣᠩᠭᠣᠯ ᠬᠡᠯᠡ ᠠ\u180b ᠨ\u180eᠠ",
    "yoruba": "Ẹ káàárọ̀. Ọmọ náà ń ṣeré. Ògún, Ọ̀ṣun.",
    "icelandic": "Þetta er íslenska. Ævar ók Ðví. Jökulsárlón.",
    "hawaiian": "ʻŌlelo Hawaiʻi. Aloha kākou!",
    "maltese": "Il-Malti għandu ħafna ittri speċjali: ċ, ġ, ħ, ż.",
    "esperanto": "Ĉu vi parolas Esperanton? Eĥoŝanĝo ĉiuĵaŭde.",
    "cherokee": "ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ. ᎣᏏᏲ!",
    "inuktitut": "ᐃᓄᒃᑎᑐᑦ ᑭᓯᐊᓂ.",
    "syriac": "ܠܫܢܐ ܣܘܪܝܝܐ ܫܠܡܐ ܥܠܡܐ",
    "thaana": "ދިވެހި ބަސް",
    "nko": "ߒߞߏ ߞߊ߲",
    "javanese": "ꦲꦏ꧀ꦱꦫꦗꦮ",
    "balinese": "ᬅᬓ᭄ᬱᬭᬩᬮᬶ ᬆ ᬈ ᬊ ᬌ ᬎ ᬒ",
    "math_alnum": "𝐇𝐞𝐥𝐥𝐨 𝕎𝕠𝕣𝕝𝕕 𝓢𝓬𝓻𝓲𝓹𝓽 𝟏𝟐𝟑 𝔄𝔟𝔠",
    "ancient": "𐌷𐌰𐌻𐌻𐍉 𐐶𐐯𐑊𐐬 𒀀𒀁𒀂 𓀀𓀁𓂀 𐎀𐎁 ᚠᚢᚦᚨᚱᚲ",
    "zalgo": "Z̴̢̛̖͕a̷̧̪͚l̶̰͊g̸̳̋o̵̬͗ t̸̨̛e̵͍͠x̴̣̀t̷̞́",
    "unicode16": "\U000105c0\U000105c1\U000105c2 \U00010d40\U00010d41 \U00010d50\U00010d51\U00010d52 \U00011380\U00011381\U000113b8\U000113c2 \U00011bc0\U00011bc1 \U00016100\U00016101\U0001611e \U00016d40\U00016d41 \U0001e5d0\U0001e5d1 \U0002ebf0\U0002ebf1 \u0897 \u1c89\u1c8a \ua7cb\ua7cd",
    "unicode10_marks": "a\u1abf b\u1ac0\u0301 c\u1df6 d\u0301\u1df6\u0316 क\u093c\u1df6 \u0d3b\u0d3c x\U00010d24 y\U0001e130 z\u0301\U0001e4ec\u0316 \U00011935\U00011930",
    "mixed_script": "Hello мир 世界 🌍 مرحبا שלום नमस्ते こんにちは 안녕 ¡Hola! Ωμέγα 123 ٤٥٦ ७८९",
    "rtl_with_numbers": "המחיר הוא 1,234.56 ₪ והמשלוח 12.5% — \u200f(RTL mark)\u200f السعر 99$ \u202bembedded\u202c",
}

EMOJI = [
    "👨\u200d👩\u200d👧\u200d👦 👩🏽\u200d💻 🏳\ufe0f\u200d🌈 🏴\u200d☠\ufe0f 🇩🇪🇺🇸🇯🇵 🇦 👍🏻👍🏼👍🏽👍🏾👍🏿 #\ufe0f\u20e3 1\ufe0f\u20e3 *\ufe0f\u20e3 ❤\ufe0f ❤ ☺\ufe0f 🧑🏿\u200d🤝\u200d🧑🏻",
    "🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f 🫠 🫶🏼 🐈\u200d⬛ 😀😃😄😁😆😅🤣😂 🙂🙃🫃🏾 🧔🏻\u200d♀\ufe0f 👁\ufe0f\u200d🗨\ufe0f 🇺🇳🇪🇺🇦🇶",
    "a😀b 1😀2 !😀? don't😀 😀\u0301 😀' 's😀 \n😀\n 😀  😀\t😀",
    "🇩🇪🇩🇪🇩🇪🇩 (odd number of regional indicators) 🏻 (modifier alone) \u200d\u200d (bare ZWJs) \ufe0f",
    "Emoji text: I ❤ NY ☕ ✓ ✔\ufe0f ✗ ★ ☆ ♠ ♣ ♥ ♦ ☀\ufe0f ☁ ⚡ ❄\ufe0f ⛄ ✈\ufe0f ⚽ ♻\ufe0f ⚠\ufe0f ⏰ ⌚",
    "Keycaps: 0\ufe0f\u20e31\ufe0f\u20e32\ufe0f\u20e33\ufe0f\u20e34\ufe0f\u20e35\ufe0f\u20e36\ufe0f\u20e37\ufe0f\u20e38\ufe0f\u20e39\ufe0f\u20e3🔟 #\u20e3 *\u20e3",
]

UNICODE_EDGE = [
    # NFD input that NFC must recompose
    unicodedata.normalize("NFD", "é è ê ë ñ ç å ü ö ä Å Ç Ñ Ö Ü ǖ ǘ ǚ ǜ ṩ ệ ở ự ỹ Ǻ ǻ ώ ΐ ᾯ"),
    "e\u0301 E\u0301 a\u0308\u0304 u\u0308\u0301 s\u0323\u0307 s\u0307\u0323 A\u030a\u0301 o\u031b\u0303",
    "e\u0327\u0301 e\u0301\u0327 a\u0316\u0301\u0302 a\u0302\u0316\u0301 x\u0301\u0316\u0302\u0317",
    unicodedata.normalize("NFD", "안녕하세요 한국어 똠방각하 뷁"),
    "각 가 힣 각 가ᆧ 각ᆨ ᄀᆨ ᄓᅡ",
    unicodedata.normalize("NFD", "Tiếng Việt có nhiều dấu thanh"),
    unicodedata.normalize("NFD", "か゛が ぱ ヴ ボ プ"),
    "か\u3099 ハ\u309a ウ\u3099 ｶﾞ",
    # compatibility characters that NFC keeps
    "ﬁ ﬂ ﬀ ﬃ ﬄ ﬅ ﬆ ＡＢＣ１２３！？ ①②③ ²³¹ ½¼¾ ℃℉ ㎏㎡ ⅠⅡⅢⅫ ㈱ ｶﾞｷﾞ ǅ ŉ ſt ™ ℡ ﷺ ㍿",
    # singletons and exclusions that NFC changes
    "Ω K Å \u0340 \u0341 \u0343 \u0344 ʹ ; · ΅ ´ \u2000 \u2001",
    "豈更﨎郞 \U0002f800\U0002f801\U0002fa1d क़ख़य़ ড়ঢ় ਲ਼ ਖ਼ ଡ଼",
    "གྷཌྷ\u0f73\u0f75\u0f76\u0f78\u0f81\u0f93 יִײַשׁפֿ ⫝̸ \U0001d15e\U0001d164 \U0001d1bb\U0001d1c0",
    "क\u093c न\u093c र\u093c ळ\u093c",
    # starter + starter compositions and blocked compositions
    "\u0b47\u0b3e \u0b47\u0b56 \u0b47\u0b57 \u0bc6\u0bbe \u0bc7\u0bbe \u0bc6\u0bd7 \u0cbf\u0cd5 \u0cc6\u0cd5 \u0cc6\u0cc2 \u0d46\u0d3e \u0dd9\u0dca \u0dd9\u0dcf ဥ\u102e",
    "\u0b47\u0301\u0b3e \u0b47\u0b3e\u0b3e e\u0301\u0301 e\u034f\u0301 a\u0301\u0308",
    "\U00011935\U00011930 \U00011938 \U00011099\U000110ba \U0001109b\U000110ba \U00011131\U00011127 \U00011347\U0001133e \U000114b9\U000114ba \U000115b8\U000115af",
    # combining marks in odd places
    "\u0301start \u0301\u0301\u0301 1\u0301 !\u0301 \u0301' ' \u0301 \t\u0301 \n\u0301 \u3000\u0301 😀\u0301 中\u0301",
    "a\u0345\u0301 α\u0345\u0301 α\u0301\u0345 ᾳ\u0301 \u0345",
    # format and invisible characters
    "\ufeffBOM at start, \ufeff in the middle, zero\u200bwidth, non\u200cjoiner, joiner\u200d, soft\u00adhyphen, "
    "word\u2060joiner, \u200eLRM \u200fRLM \u202aLRE\u202c \u2066LRI\u2069 \u061c ALM \u180e MVS",
    "variation\ufe00selectors\ufe0f and \U000e0100 tags \U000e0001\U000e0065\U000e006e\U000e007f",
    # noncharacters, private use, unassigned
    "\ufdd0 \ufdef \ufffe \uffff \U0001fffe \U0010fffe \U0010ffff \ue000 \uf8ff \U000f0000 \U0010fffd \u0378 ࡰ \U0001ff00",
    "\u0000 NUL \u0001 SOH \u001f US \u007f DEL \u0080 PAD \u009f APC \u001c\u001d\u001e FS GS RS",
]

NUMBERS = [
    "0 1 12 123 1234 12345 123456 1234567 12345678 123456789 1234567890",
    "3.14159 2.718281828 -1.5e-10 +6.022e23 1E6 0.000001 .5 5. 1e+308 -0.0 NaN inf",
    "1,234,567.89 1.234.567,89 1 234 567 1\u202f234\u202f567 12'345'678 $1,000.00 €5 ¥100 ₹2,50,000 ₿0.001 £9.99",
    "2026-09-30 30/09/2026 09.30.2026 12:34:56.789 T23:59:59Z +05:30 1759248000",
    "192.168.0.1 10.0.0.255/24 ::1 2001:db8::8a2e:370:7334 fe80::1%eth0 :8080",
    "+49 (0) 30 1234-5678 +1-800-555-0199 (555) 123 4567",
    "0xDEADBEEF 0b10101010 0o755 1_000_000 0x1.8p3 42u 3.0f 7L 99n",
    "v1.2.3-rc.1+build.5 Python 3.12.14 CUDA 12.6 sm_86 RTX 3090 x86_64 arm64",
    "１２３ ４５６ ٠١٢٣٤٥٦٧٨٩ ۰۱۲۳ ०१२३ ০১২৩ ๐๑๒๓ 〇一二三 ⅰⅱⅲ Ⅻ Ⅿ ① ⑳ ㉑ ⒈ ⓵",
    "x² H₂O E=mc² 10⁻³ ½ ⅓ ¾ 1⁄2 ⅞ 99.9% 100‰ 3×4=12 8÷2 ±5 ≈ ≠ ≤ ≥ ∞ √2 ∑∫∂∇",
    "π≈3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706",
    "1" * 200,
    "9" * 1000,
    "".join(str(i) for i in range(400)),
    "12ab34cd56 a1b2c3 x86_64 h264 mp3 1st 2nd 3rd 4th 21st 100km 5kg 3.5GHz 16GB 4K 1080p",
]

WHITESPACE = [
    " ", "  ", "   ", "    ", "\t", "\t\t", "\n", "\n\n", "\n\n\n", "\r", "\r\n", "\r\n\r\n", "\n\r", "\r\r",
    " \n", "\n ", " \n ", "  \n  ", "\t \n", "\n \t", " \t ", "\t\n\t", "\v", "\f", "\x85", "\u00a0", "\u3000",
    "\u2028", "\u2029", "\u1680", "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a",
    "\u202f", "\u205f", " a", "  a", "   a", "a ", "a  ", "a   ", " a ", "a  b", "a   b", "a    b",
    "a\tb", "a\t\tb", "a\nb", "a\n\nb", "a\n\n\nb", "a\r\nb", "a\rb", "a \n b", "a  \n  b", "a\u00a0b",
    "a\u3000b", "a\u2028b", "a \u00a0b", "a\u00a0 b", "a\u00a0\u00a0b", "  !", " !", "\t!", "\u00a0!",
    "\u3000!", " 1", "  1", "\t1", "\n1", " \u0301", "  \u0301", "x  ", "x\n  ", "x  \n", "x \r\n",
    "   leading and trailing   ", "\n\nleading newlines", "trailing newlines\n\n", "\t\tindented\n\t\t\tmore",
    "mixed \t \n \r\n \u00a0 \u3000 \u2028 whitespace   end", "a" + " " * 50 + "b", " " * 100,
    "\n" * 30, "\r\n" * 20, "\t" * 16 + "x", " \n" * 20, "x" + " " * 7 + "\n" + " " * 7 + "y",
    "!!! \n  ??? \n\n", ";\n\n\t", "}\r\n\r\n", ")\n    ", "\" \n", "a,\u00a0b", "end.\u2029Next",
]

ADDED = [
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\nThe user asks a simple sum.\n"
    "</think>\n\n2+2 = **4**.<|im_end|>\n<|endoftext|>",
    "<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Berlin\"}}\n"
    "</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{\"temp\": 21.5, \"unit\": \"°C\"}\n"
    "</tool_response><|im_end|>\n",
    "<|fim_prefix|>def add(a, b):\n    <|fim_suffix|>\n    return c<|fim_middle|>c = a + b",
    "<|repo_name|>qwen38\n<|file_sep|>src/main.cpp\nint main() {}\n<|file_sep|>README.md\n# qwen38\n<|fim_pad|>",
    "<|vision_start|>" + "<|image_pad|>" * 64 + "<|vision_end|>Describe the image.",
    "<|vision_start|><|video_pad|><|video_pad|><|vision_end|><|vision_pad|><|box_start|>(10,20),(30,40)<|box_end|>"
    "<|object_ref_start|>the cat<|object_ref_end|><|quad_start|>quad<|quad_end|>",
    "<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|><tts_pad><tts_text_bos>hello<tts_text_eod>"
    "<tts_text_bos_single>x<tts_text_bos",
    "<|im_start|><|im_end|><|im_start|><|im_end|><|endoftext|><|endoftext|><think></think><think><think>",
    "abc<think>def</think>ghi 中文<|im_end|>中文 1<|im_start|>2 !<tool_call>! 'll<|im_end|>'s",
    "<|im_start <|im_start| |im_start|> <|im_start |> <| im_start|> <|IM_START|> <|Im_Start|> < think> <think >"
    " <Think> <THINK> </ think> <|endoftext| <|endoftext <|end of text|> <tool_call > <tool_call",
    "<<|im_start|>> <|<|im_end|>|> <<think>> <|im_end|>< <<<|endoftext|>>> <|im_start|<|im_start|>",
    "e<think>\u0301 e\u0301<think>e\u0301 <|im_start|>\u0301 \u0301<|im_end|> 한<think>ᆨ ᄀ<|im_end|>ᅡ",
    "  <|im_start|>  \n\n<|im_end|>\n\n  <think>   </think>  \t<tool_call>\r\n",
    "<think>\n\n</think>\n\n",
    "<|im_start|>user\n" + "Hello! " * 50 + "<|im_end|>\n<|im_start|>assistant\n",
]
# every added token alone and in context
_ADDED_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
                 "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
                 "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
                 "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>",
                 "<|file_sep|>", "<tool_response>", "</tool_response>", "<think>", "</think>", "<|audio_start|>",
                 "<|audio_end|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>",
                 "<|audio_pad|>"]
for _t in _ADDED_TOKENS:
    ADDED += [_t, " " + _t + " ", "x" + _t + "y", "\n" + _t + "\n", _t + _t, _t[:-1], _t[1:], _t[:-1] + _t]


# =============================================================================================
# Random strings
# =============================================================================================

def _r(*pairs):
    out = []
    for p in pairs:
        if isinstance(p, int):
            out.append((p, p))
        else:
            out.append(p)
    return out


RANDOM_POOLS = [
    (20, _r((0x41, 0x5A), (0x61, 0x7A))),
    (8, _r((0x30, 0x39))),
    (10, _r((0x21, 0x2F), (0x3A, 0x40), (0x5B, 0x60), (0x7B, 0x7E))),
    (10, _r(0x20)),
    (4, _r(0x09, 0x0A, 0x0D, 0x0B, 0x0C, 0x85, 0xA0, 0x1680, (0x2000, 0x200A), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000)),
    (3, _r(0x27)),
    (6, _r((0xC0, 0x24F))),
    (6, _r((0x300, 0x36F))),
    (3, _r((0x483, 0x489), (0x591, 0x5C7), (0x610, 0x61A), (0x64B, 0x65F), (0x900, 0x903), (0x93A, 0x94F),
           (0x1AB0, 0x1AFF), (0x1DC0, 0x1DFF), (0x20D0, 0x20FF), (0xFE20, 0xFE2F), (0x1D165, 0x1D189))),
    (2, _r(0x7FD, (0x898, 0x89F), (0x8CA, 0x8D3), 0x9FE, 0xC3C, (0xD3B, 0xD3C), 0xEBA, 0x1715, (0x1ABF, 0x1ACE),
           (0x1DF6, 0x1DFA), 0xA82C, (0x10D24, 0x10D27), (0x10EFD, 0x10EFF), (0x10F46, 0x10F50), (0x1E130, 0x1E136),
           (0x16FF0, 0x16FF1), (0x1E4EC, 0x1E4EF), 0x11938, 0x11935, 0x11930)),
    (4, _r((0x370, 0x3FF), (0x400, 0x52F), (0x1F00, 0x1FFF))),
    (4, _r((0x590, 0x7FF))),
    (4, _r((0x900, 0xDFF))),
    (3, _r((0xE00, 0x109F))),
    (3, _r((0x1100, 0x11FF), (0xA960, 0xA97F), (0xD7B0, 0xD7FF))),
    (3, _r((0xAC00, 0xD7A3))),
    (5, _r((0x4E00, 0x9FFF), (0x3400, 0x4DBF))),
    (3, _r((0x3040, 0x30FF), (0xFF65, 0xFF9F))),
    (2, _r((0xFF01, 0xFF5E))),
    (4, _r((0x2000, 0x2BFF))),
    (4, _r((0x1F300, 0x1FAFF), (0x2600, 0x27BF))),
    (1, _r((0x1F1E6, 0x1F1FF), (0x1F3FB, 0x1F3FF))),
    (2, _r(0x200D, 0x200C, 0xFE0F, (0xFE00, 0xFE0E), (0xE0100, 0xE01EF), 0x20E3)),
    (1, _r((0xE0000, 0xE007F))),
    (2, _r((0x10000, 0x1BFFF))),
    (1, _r((0x1D400, 0x1D7FF))),
    (1, _r((0x20000, 0x2FA1F), (0x30000, 0x3134F))),
    (2, _r(0x0897, (0x1C89, 0x1C8A), (0xA7CB, 0xA7CD), (0x105C0, 0x105F3), (0x10D40, 0x10D85), (0x11380, 0x113E2),
           (0x116D0, 0x116E3), (0x11BC0, 0x11BF9), (0x13460, 0x143FA), (0x16100, 0x16139), (0x16D40, 0x16D79),
           (0x1CCF0, 0x1CCF9), (0x1E5D0, 0x1E5FA), (0x2EBF0, 0x2EE5D))),
    (1, _r((0xE000, 0xF8FF), (0xF0000, 0x10FFFD), (0xFDD0, 0xFDEF), 0xFFFE, 0xFFFF, 0x1FFFE, 0x10FFFF, 0x0)),
    (1, _r((0xF900, 0xFAFF), (0x2F800, 0x2FA1D), 0x2126, 0x212A, 0x212B, (0x340, 0x344), 0x374, 0x37E, 0x387)),
    (1, _r((0x958, 0x95F), (0x9DC, 0x9DF), 0xA33, 0xA36, (0xA59, 0xA5E), 0xB5C, 0xB5D, (0xF43, 0xF81),
           (0xFB1D, 0xFB4E), 0x2ADC, (0x1D15E, 0x1D164), (0x1D1BB, 0x1D1C0))),
    (3, _r((0x1E00, 0x1EFF))),
    (2, _r((0x0, 0x10FFFF))),
]

WORDS = ["the", "The", "THE", "and", "tokenizer", "Hello", "world", "don't", "it's", "we'll", "I'm", "you've",
         "café", "naïve", "über", "中文", "日本語", "한국어", "Привет", "مرحبا", "שלום", "नमस\u094dत\u0947", "ไทย", "😀",
         "def", "return", "std::vector<int>", "->", "==", "!=", "...", "http://x.y/z?a=1", "3.14", "1e-5",
         "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>", "<tool_call>", "<|im_", "_start|>",
         "<thi", "nk>", "'s", "'LL", "'Re", "\r\n", "\n\n", "    ", "\t", " \n ", "123456", "_", "__init__"]


def random_string(rng: random.Random) -> str:
    weights = [w for w, _ in RANDOM_POOLS]
    n = rng.randint(1, 40) if rng.random() < 0.9 else rng.randint(40, 400)
    out = []
    while len(out) < n:
        if rng.random() < 0.15:
            out.append(rng.choice(WORDS))
            continue
        _, ranges = rng.choices(RANDOM_POOLS, weights)[0]
        total = sum(b - a + 1 for a, b in ranges)
        k = rng.randrange(total)
        for a, b in ranges:
            if k <= b - a:
                cp = a + k
                break
            k -= b - a + 1
        if 0xD800 <= cp <= 0xDFFF:
            continue
        out.append(chr(cp))
    return "".join(out)


def random_codepoints(rng: random.Random) -> str:
    out = []
    for _ in range(rng.randint(1, 30)):
        cp = rng.randrange(0x110000)
        if 0xD800 <= cp <= 0xDFFF:
            cp = 0xFFFD
        out.append(chr(cp))
    return "".join(out)


# =============================================================================================
# Invalid UTF-8
# =============================================================================================

INVALID_UTF8 = [
    b"\xff", b"\xfe", b"\x80", b"\xbf", b"\xc0", b"\xc1", b"\xc2", b"\xe2\x82", b"\xf0\x9f\x98",
    b"abc\xe2\x82", b"abc\xe2\x82 def", b"\xc0\xaf", b"\xe0\x80\xaf", b"\xf0\x80\x80\xaf", b"\xc0\x80",
    b"\xed\xa0\x80", b"\xed\xbf\xbf", b"\xed\xa0\xbd\xed\xb8\x80", b"\xf4\x90\x80\x80", b"\xf5\x80\x80\x80",
    b"\xf8\x88\x80\x80\x80", b"\xfc\x84\x80\x80\x80\x80", b"\x80\x80\x80", b"\xe0\xa0", b"\xe0\xa0A",
    b"\xf0\x90\x80A", b"\xf1\x80\x80\xc0", b"hello \xff world", b"<|im_\xffstart|>", b"<|im_start|>\xff<|im_end|>",
    b"\xef\xbb\xbfBOM", b"\xef\xbb", b"caf\xe9", b"na\xefve", b"\xe4\xb8\xad\xe6\x96", b"\xe4\xb8\xad\xe6\x96\x87",
    b"it\xe2\x80s", b"'\xc3", b"'\xc5\xbf", b" \xa0 ", b"\r\n\xff\r\n", b"a\xcc", b"a\xcc\x81", b"1\xef\xbc",
]


def mutate_bytes(rng: random.Random, b: bytes) -> bytes:
    ba = bytearray(b)
    for _ in range(rng.randint(1, 3)):
        op = rng.randrange(4)
        pos = rng.randrange(len(ba) + 1)
        if op == 0 and ba:
            ba[min(pos, len(ba) - 1)] = rng.randrange(256)
        elif op == 1:
            ba.insert(pos, rng.choice([0x80, 0xBF, 0xC0, 0xC3, 0xE2, 0xED, 0xF0, 0xF4, 0xF5, 0xFF, rng.randrange(256)]))
        elif op == 2 and ba:
            del ba[min(pos, len(ba) - 1)]
        else:
            ba = ba[:pos]
    return bytes(ba)


# =============================================================================================
# Long documents and huge pre-tokens
# =============================================================================================

def stdlib_text(names: list[str], min_bytes: int) -> str | None:
    base = sysconfig.get_paths()["stdlib"]
    parts = []
    for name in names:
        try:
            with open(os.path.join(base, name), encoding="utf-8") as f:
                parts.append(f.read())
        except OSError:
            continue
        if sum(len(p) for p in parts) >= min_bytes:
            break
    text = "\n".join(parts)
    return text if len(text.encode()) >= min_bytes else None


def long_documents(rng: random.Random) -> list[tuple[str, str]]:
    docs = []
    py = stdlib_text(["difflib.py", "textwrap.py", "json/decoder.py", "argparse.py"], 60_000)
    if py:
        docs.append(("long/python_stdlib", py))
        docs.append(("long/python_stdlib_crlf_tabs", py.replace("    ", "\t").replace("\n", "\r\n")))
        old = py.splitlines(keepends=True)[:1500]
        new = [ln.replace("self", "this").replace("return", "yield") if i % 7 == 0 else ln for i, ln in enumerate(old)]
        new.insert(100, "# inserted line with unicode: é 中 😀\n")
        docs.append(("long/diff", "".join(difflib.unified_diff(old, new, "a/difflib.py", "b/difflib.py"))))
    code = "\n\n".join(CODE.values())
    docs.append(("long/code_mix", (code + "\n") * (60_000 // len(code.encode()) + 1)))
    paras = list(LANGUAGES.values()) + PROSE + EMOJI
    out = []
    while sum(len(p.encode()) for p in out) < 80_000:
        out.append(rng.choice(paras))
        out.append(rng.choice([" ", "\n", "\n\n", "  ", "\t", "\r\n", " — ", ""]))
    docs.append(("long/multilingual", "".join(out)))
    chat = []
    while sum(len(c.encode()) for c in chat) < 60_000:
        user = " ".join(rng.choice(PROSE).split()[: rng.randint(3, 30)])
        reply = rng.choice(list(LANGUAGES.values()) + list(CODE.values()))
        chat.append(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n{rng.choice(PROSE)}\n"
                    f"</think>\n\n{reply}<|im_end|>\n")
    docs.append(("long/chat_transcript", "".join(chat)))
    rnd = []
    while sum(len(s.encode()) for s in rnd) < 60_000:
        rnd.append(random_string(rng))
    docs.append(("long/random", "".join(rnd)))
    docs.append(("long/markdown", (CODE["markdown"] + "\n") * 40))
    return docs


def huge_pretokens(rng: random.Random) -> list[tuple[str, str]]:
    lower = "abcdefghijklmnopqrstuvwxyz"
    letters = lower + lower.upper()
    return [
        ("huge/random_lowercase_100k", "".join(rng.choice(lower) for _ in range(100_000))),
        ("huge/random_letters_60k", "".join(rng.choice(letters) for _ in range(60_000))),
        ("huge/same_letter_50k", "a" * 50_000),
        ("huge/abab_40k", "ab" * 20_000),
        ("huge/cjk_40k", "".join(chr(rng.randint(0x4E00, 0x9FFF)) for _ in range(40_000))),
        ("huge/common_cjk_40k", "".join(rng.choice("的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长")
                                          for _ in range(40_000))),
        ("huge/punct_50k", "=" * 25_000 + "-" * 25_000),
        ("huge/spaces_50k", " " * 50_000 + "x"),
        ("huge/newlines_30k", "\n" * 30_000),
        ("huge/mixed_ws_30k", "".join(rng.choice(" \t\n\r\u00a0\u3000") for _ in range(30_000))),
        ("huge/letters_with_marks_20k", "".join(rng.choice("aeoun") + rng.choice(["\u0301", "\u0300\u0323", "\u0308",
                                                                                  "\u0327\u0301", "\u0302\u0303"])
                                                for _ in range(10_000))),
        ("huge/hangul_jamo_20k", "".join("각"[i % 3] if rng.random() < 0.8 else "ᅡ" for i in range(20_000))),
    ]


def sweep_cases() -> list[tuple[str, str]]:
    cps = [cp for cp in range(0x40000) if not 0xD800 <= cp <= 0xDFFF]
    cps += list(range(0xE0000, 0xF0000))
    cps += list(range(0x40000, 0xE0000, 257)) + list(range(0xF0000, 0x110000, 257)) + [0x10FFFE, 0x10FFFF]
    out = []
    for i in range(0, len(cps), 256):
        chunk = cps[i:i + 256]
        text = "".join(f"a{chr(c)} {chr(c)}1{chr(c)}\n" for c in chunk)
        out.append((f"sweep/U+{chunk[0]:05X}", text))
    return out


# =============================================================================================
# Assembly
# =============================================================================================

def variants(label: str, text: str) -> list[tuple[str, str]]:
    out = [(label, text)]
    if "\n" in text:
        out.append((label + "/crlf", text.replace("\n", "\r\n")))
    if "    " in text:
        out.append((label + "/tabs", text.replace("    ", "\t")))
    if "\n" in text:
        out.append((label + "/trailing_ws", "\n".join(ln + " " * (i % 3) for i, ln in enumerate(text.split("\n")))))
    return out


def build_corpus(seed: int) -> list[tuple[int, str, bytes]]:
    """Returns (flags, label, input bytes)."""
    rng = random.Random(seed)
    texts: list[tuple[str, str]] = [("empty", "")]
    for i, t in enumerate(PROSE):
        texts += variants(f"prose/{i}", t)
    texts.append(("prose/all", "\n\n".join(PROSE)))
    texts += [(f"contraction/{i}", t) for i, t in enumerate(CONTRACTIONS)]
    for name, t in CODE.items():
        texts += variants(f"code/{name}", t)
    for name, t in LANGUAGES.items():
        texts.append((f"lang/{name}", t))
        nfd = unicodedata.normalize("NFD", t)
        if nfd != t:
            texts.append((f"lang/{name}/nfd", nfd))
    texts.append(("lang/all", "\n".join(LANGUAGES.values())))
    texts += [(f"emoji/{i}", t) for i, t in enumerate(EMOJI)]
    texts += [(f"unicode/{i}", t) for i, t in enumerate(UNICODE_EDGE)]
    texts += [(f"number/{i}", t) for i, t in enumerate(NUMBERS)]
    texts += [(f"whitespace/{i}", t) for i, t in enumerate(WHITESPACE)]
    texts += [(f"added/{i}", t) for i, t in enumerate(ADDED)]
    texts += [(f"random/mix/{i}", random_string(rng)) for i in range(8000)]
    texts += [(f"random/codepoints/{i}", random_codepoints(rng)) for i in range(1500)]
    texts += long_documents(rng)
    texts += huge_pretokens(rng)
    texts += sweep_cases()

    cases = [(1, label, t.encode("utf-8")) for label, t in texts]
    cases += [(1, f"invalid/{i}", b) for i, b in enumerate(INVALID_UTF8)]
    cases += [(1, f"invalid/random/{i}", bytes(rng.choice([rng.randrange(256), rng.randrange(0x80, 0x100)])
                                               for _ in range(rng.randint(1, 40)))) for i in range(400)]
    cases += [(1, f"invalid/mutated/{i}", mutate_bytes(rng, random_string(rng).encode("utf-8"))) for i in range(600)]
    # allow_added_tokens = false
    plain = [c for c in cases if c[1].startswith(("added/", "prose/", "lang/", "code/", "long/chat", "invalid/"))]
    plain += rng.sample([c for c in cases if c[1].startswith("random/")], 1000)
    cases += [(0, "noadded/" + label, b) for _, label, b in plain]
    return cases


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    seed = 20260930
    for a in argv[1:]:
        if a.startswith("--seed="):
            seed = int(a.split("=", 1)[1])
    if len(args) != 2:
        print("usage: tokenizer_cases.py <tokenizer.json> <out.bin> [--seed=N]", file=sys.stderr)
        return 2
    from tokenizers import Tokenizer

    tj_path, out_path = args
    t0 = time.time()
    tok = Tokenizer.from_file(tj_path)
    with open(tj_path, encoding="utf-8") as f:
        tj = json.load(f)
    added_contents = [t["content"] for t in tj["added_tokens"]]
    if any(t.get(k) for t in tj["added_tokens"] for k in ("lstrip", "rstrip", "single_word", "normalized")):
        raise SystemExit("this generator's expected-decode model assumes plain added tokens")
    tj_plain = dict(tj)
    tj_plain["added_tokens"] = []
    tok_plain = Tokenizer.from_str(json.dumps(tj_plain))
    nfc = tok.normalizer

    u2b = {u: b for b, u in bytes_to_unicode().items()}
    raw = {i: bytes(u2b[c] for c in s) for s, i in tj["model"]["vocab"].items()}
    for t in tj["added_tokens"]:
        raw[t["id"]] = t["content"].encode("utf-8")

    # leftmost-longest matching: `re` alternation is leftmost-first, so list longer contents first
    added_re = re.compile("|".join(re.escape(a) for a in sorted(added_contents, key=len, reverse=True)))

    def expected_normalized(text: str, allow_added: bool) -> str:
        if not allow_added:
            return nfc.normalize_str(text)
        out, pos = [], 0
        for m in added_re.finditer(text):
            out.append(nfc.normalize_str(text[pos:m.start()]))
            out.append(m.group(0))
            pos = m.end()
        out.append(nfc.normalize_str(text[pos:]))
        return "".join(out)

    cases = build_corpus(seed)
    texts = [b.decode("utf-8", "replace") for _, _, b in cases]
    ids: list[list[int] | None] = [None] * len(cases)
    for flag, tk in ((1, tok), (0, tok_plain)):
        idx = [i for i, c in enumerate(cases) if c[0] == flag]
        for j in range(0, len(idx), 2000):
            part = idx[j:j + 2000]
            for i, enc in zip(part, tk.encode_batch([texts[i] for i in part], add_special_tokens=False)):
                ids[i] = enc.ids

    buf = bytearray(MAGIC + struct.pack("<II", VERSION, len(cases)))
    n_ids = 0
    for (flags, label, inp), text, case_ids in zip(cases, texts, ids):
        dec = expected_normalized(text, bool(flags & 1)).encode("utf-8")
        if b"".join(raw[i] for i in case_ids) != dec:
            raise SystemExit(f"internal check failed for {label}: HF ids do not decode to the per-segment NFC input")
        lb = label.encode("ascii")
        buf += struct.pack("<I", flags) + struct.pack("<I", len(lb)) + lb + struct.pack("<I", len(inp)) + inp
        buf += struct.pack("<I", len(case_ids)) + struct.pack(f"<{len(case_ids)}i", *case_ids)
        buf += struct.pack("<I", len(dec)) + dec
        n_ids += len(case_ids)
    with open(out_path, "wb") as f:
        f.write(buf)
    total_in = sum(len(c[2]) for c in cases)
    print(f"wrote {out_path}: {len(cases)} cases, {total_in / 1e6:.1f} MB input, {n_ids} ids, "
          f"{len(buf) / 1e6:.1f} MB file ({time.time() - t0:.1f} s, seed {seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
