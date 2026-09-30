// Byte-level BPE tokenizer for Qwen3.x, bit-exact with Hugging Face `tokenizers`:
//
//   Tokenizer::encode(text) == tokenizers.Tokenizer.from_file("tokenizer.json")
//                                  .encode(text, add_special_tokens=False).ids
//
// Pipeline (the same order as HF):
//   0. Invalid UTF-8 is replaced by U+FFFD, one per maximal subpart (identical to Python's
//      bytes.decode("utf-8", "replace")). HF only accepts valid UTF-8 (Python str), so for any
//      byte string b: encode(b) == HF.encode(b.decode("utf-8", "replace")).
//   1. Added tokens (<|im_start|>, <think>, ...) are cut out of the raw text: leftmost-longest
//      match, with HF's lstrip / rstrip / single_word / normalized semantics.
//   2. Every other segment is NFC-normalized with the tables HF uses (Unicode 9.0 behaviour).
//   3. The segment is split by the Qwen3 pre-tokenizer regex (hand-written scanner, character
//      classes of HF's Oniguruma = Unicode 16.0).
//   4. Each piece is BPE-encoded over its raw bytes: repeatedly merge the adjacent pair with the
//      lowest merge rank, leftmost on ties (HF's Word::merge_all). O(n log n) per piece.
//
// The data comes from the blob written by tools/tokenizer_export.py (layout documented there).
// A Tokenizer is immutable once built: every const method is thread-safe and copies are cheap
// (they share the loaded tables).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace qwen38 {

class Tokenizer {
 public:
  // Empty tokenizer: vocab_size() == 0, encode() throws std::logic_error. Assign a loaded one.
  Tokenizer();

  // Parses a tokenizer blob; copies what it needs. Throws std::runtime_error on malformed data
  // or on a blob whose pre-tokenizer pattern this implementation does not support.
  static Tokenizer from_blob(const void* data, size_t size);
  // Reads a blob file (throws std::runtime_error if it cannot be read or is malformed).
  static Tokenizer from_file(const std::string& path);

  // HF encode(text, add_special_tokens=False).ids. With allow_added_tokens == false the text of
  // added tokens is treated as plain text (no added-token matching at all, special or not).
  std::vector<int32_t> encode(std::string_view text, bool allow_added_tokens = true) const;
  // Same, appending to `out` (convenient when building prompts piece by piece).
  void encode(std::string_view text, std::vector<int32_t>& out, bool allow_added_tokens = true) const;

  // Concatenated raw token bytes (may be invalid UTF-8 if the ids are). Ids outside
  // [0, vocab_size()) contribute nothing.
  std::string decode(std::span<const int32_t> ids) const;
  // Raw bytes of one token; empty if out of range (or an unassigned id).
  std::string_view token_bytes(int32_t id) const;
  // Id of the vocab or added token whose raw bytes are exactly `token_text`, -1 if none.
  int32_t find_token(std::string_view token_text) const;
  // HF "special" flag of an added token (false for plain vocab ids and non-special added tokens).
  bool is_special(int32_t id) const;
  // True for every added token (special or not, e.g. <think>).
  bool is_added(int32_t id) const;
  // Number of addressable ids (base vocab + added tokens): 248077 for Qwen3.8.
  int32_t vocab_size() const;
  explicit operator bool() const { return impl_ != nullptr; }

  // Diagnostics (no added-token handling): the NFC normalizer alone, and normalizer + regex split.
  std::string normalize(std::string_view text) const;
  std::vector<std::string> pre_tokenize(std::string_view text) const;
  // Value of a META key of the blob (unidata_version, source_sha256, ...), "" if absent.
  std::string metadata(std::string_view key) const;

 private:
  struct Impl;
  explicit Tokenizer(std::shared_ptr<const Impl> impl);
  const Impl& impl() const;
  std::shared_ptr<const Impl> impl_;
};

// Incremental UTF-8 decoding for streaming output: push the bytes of each generated token and get
// back only complete characters. Incomplete trailing sequences are held back until the next push.
// Invalid sequences become U+FFFD (one per maximal subpart), so the concatenation of all push()
// results and flush() equals Python's bytes.decode("utf-8", "replace") of the whole stream.
class Utf8StreamDecoder {
 public:
  std::string push(std::string_view bytes);
  // Emits what is still pending (an incomplete sequence becomes U+FFFD) and resets the state.
  std::string flush();
  void reset() { pending_.clear(); }
  size_t pending_bytes() const { return pending_.size(); }

 private:
  std::string pending_;
};

}  // namespace qwen38
