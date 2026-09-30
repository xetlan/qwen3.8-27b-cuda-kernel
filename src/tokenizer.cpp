// Byte-level BPE tokenizer (Qwen3.x), see include/qwen38/tokenizer.h for the contract and
// tools/tokenizer_export.py for the blob layout.

#include "qwen38/tokenizer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qwen38 {
namespace {

// ---------------------------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------------------------

constexpr char kMagic[8] = {'Q', 'W', '3', '8', 'T', 'O', 'K', '\0'};
constexpr uint32_t kFormatVersion = 1;

// The pre-tokenizer regex implemented by Impl::match_at(). A blob with any other pattern is
// rejected instead of being tokenized wrongly.
constexpr std::string_view kSplitPattern =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N})re"
    R"re(| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";

constexpr char32_t kNumCodePoints = 0x110000;

// Per-code-point property byte. The low five bits come from the blob (UCLS), the high three are
// derived from the NFC tables at load time.
constexpr uint8_t kL = 1;     // \p{L}
constexpr uint8_t kM = 2;     // \p{M}
constexpr uint8_t kN = 4;     // \p{N}
constexpr uint8_t kWS = 8;    // \s (White_Space)
constexpr uint8_t kWord = 16; // \w (single_word test)
constexpr uint8_t kClassMask = kL | kM | kN | kWS | kWord;
// NFC maps the character to itself and it never combines with anything before it
// (ccc 0, no decomposition, not the second element of a composition).
constexpr uint8_t kNfcQuick = 32;
// NFC(text) == NFC(text[..i]) + NFC(text[i..]) whenever the character at i has this bit
// (ccc 0, not a second element, and its decomposition starts with such a character).
constexpr uint8_t kNfcBoundary = 64;
// Has an entry in the canonical decomposition table (Hangul syllables are algorithmic).
constexpr uint8_t kHasDecomp = 128;

constexpr uint32_t kAddedSpecial = 1, kAddedLstrip = 2, kAddedRstrip = 4, kAddedSingleWord = 8,
                   kAddedNormalized = 16;

// Hangul syllables (Unicode ch. 3.12).
constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr char32_t kLCount = 19, kVCount = 21, kTCount = 28;
constexpr char32_t kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

constexpr uint32_t kNoRank = std::numeric_limits<uint32_t>::max();

[[noreturn]] void malformed(const std::string& what) {
  throw std::runtime_error("qwen38::Tokenizer: malformed tokenizer blob: " + what);
}

// ---------------------------------------------------------------------------------------------
// UTF-8
// ---------------------------------------------------------------------------------------------

enum class Utf8Step { kValid, kInvalid, kTruncated };
struct Utf8Result {
  Utf8Step step;
  uint32_t len;  // kValid: sequence length; kInvalid: maximal subpart length; kTruncated: bytes seen
  char32_t cp;
};

// Decodes one sequence at p (n >= 1 bytes available) with the Unicode "maximal subpart" rule,
// which is also what CPython's "replace" error handler implements.
inline Utf8Result utf8_step(const uint8_t* p, size_t n) {
  const uint8_t b0 = p[0];
  if (b0 < 0x80) return {Utf8Step::kValid, 1, b0};
  uint32_t need;
  char32_t cp;
  uint8_t lo = 0x80, hi = 0xBF;
  if (b0 >= 0xC2 && b0 <= 0xDF) {
    need = 1;
    cp = b0 & 0x1F;
  } else if (b0 >= 0xE0 && b0 <= 0xEF) {
    need = 2;
    cp = b0 & 0x0F;
    if (b0 == 0xE0) lo = 0xA0;       // no overlongs
    else if (b0 == 0xED) hi = 0x9F;  // no surrogates
  } else if (b0 >= 0xF0 && b0 <= 0xF4) {
    need = 3;
    cp = b0 & 0x07;
    if (b0 == 0xF0) lo = 0x90;       // no overlongs
    else if (b0 == 0xF4) hi = 0x8F;  // <= U+10FFFF
  } else {
    return {Utf8Step::kInvalid, 1, 0};  // continuation byte, C0, C1, F5..FF
  }
  for (uint32_t k = 1; k <= need; ++k) {
    if (k >= n) return {Utf8Step::kTruncated, k, 0};
    const uint8_t b = p[k];
    if (b < lo || b > hi) return {Utf8Step::kInvalid, k, 0};
    lo = 0x80;
    hi = 0xBF;
    cp = (cp << 6) | (b & 0x3F);
  }
  return {Utf8Step::kValid, need + 1, cp};
}

bool utf8_valid(std::string_view s) {
  const auto* p = reinterpret_cast<const uint8_t*>(s.data());
  const size_t n = s.size();
  size_t i = 0;
  while (i < n) {
    if (n - i >= 8) {
      uint64_t w;
      std::memcpy(&w, p + i, 8);
      if ((w & 0x8080808080808080ull) == 0) {
        i += 8;
        continue;
      }
    }
    if (p[i] < 0x80) {
      ++i;
      continue;
    }
    const Utf8Result r = utf8_step(p + i, n - i);
    if (r.step != Utf8Step::kValid) return false;
    i += r.len;
  }
  return true;
}

void append_utf8(std::string& out, char32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    const char b[2] = {static_cast<char>(0xC0 | (cp >> 6)), static_cast<char>(0x80 | (cp & 0x3F))};
    out.append(b, 2);
  } else if (cp < 0x10000) {
    const char b[3] = {static_cast<char>(0xE0 | (cp >> 12)), static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
                       static_cast<char>(0x80 | (cp & 0x3F))};
    out.append(b, 3);
  } else {
    const char b[4] = {static_cast<char>(0xF0 | (cp >> 18)), static_cast<char>(0x80 | ((cp >> 12) & 0x3F)),
                       static_cast<char>(0x80 | ((cp >> 6) & 0x3F)), static_cast<char>(0x80 | (cp & 0x3F))};
    out.append(b, 4);
  }
}

constexpr std::string_view kReplacementUtf8 = "\xEF\xBF\xBD";

// Replaces every maximal invalid subpart by U+FFFD (bytes.decode("utf-8", "replace")).
std::string utf8_sanitize(std::string_view s) {
  const auto* p = reinterpret_cast<const uint8_t*>(s.data());
  const size_t n = s.size();
  std::string out;
  out.reserve(n + 16);
  size_t i = 0;
  while (i < n) {
    const Utf8Result r = utf8_step(p + i, n - i);
    if (r.step == Utf8Step::kValid) {
      out.append(s.data() + i, r.len);
    } else {
      out.append(kReplacementUtf8);  // kTruncated at the end of input is one maximal subpart
    }
    i += r.len;
  }
  return out;
}

// Decodes the code point at p[i] of a string known to be valid UTF-8.
inline char32_t decode_at(const uint8_t* p, size_t i, uint32_t& len) {
  const uint8_t b = p[i];
  if (b < 0x80) {
    len = 1;
    return b;
  }
  if (b < 0xE0) {
    len = 2;
    return (char32_t(b & 0x1F) << 6) | (p[i + 1] & 0x3F);
  }
  if (b < 0xF0) {
    len = 3;
    return (char32_t(b & 0x0F) << 12) | (char32_t(p[i + 1] & 0x3F) << 6) | (p[i + 2] & 0x3F);
  }
  len = 4;
  return (char32_t(b & 0x07) << 18) | (char32_t(p[i + 1] & 0x3F) << 12) | (char32_t(p[i + 2] & 0x3F) << 6) |
         (p[i + 3] & 0x3F);
}

// Start of the code point that ends at byte i (i > 0) of a valid UTF-8 string.
inline size_t prev_start(const uint8_t* p, size_t i) {
  do {
    --i;
  } while (i > 0 && (p[i] & 0xC0) == 0x80);
  return i;
}

// ---------------------------------------------------------------------------------------------
// Blob reading
// ---------------------------------------------------------------------------------------------

class Reader {
 public:
  Reader(const uint8_t* data, size_t size, std::string what) : p_(data), n_(size), what_(std::move(what)) {}

  uint32_t u32() {
    need(4);
    const uint8_t* q = p_ + pos_;
    pos_ += 4;
    return uint32_t(q[0]) | uint32_t(q[1]) << 8 | uint32_t(q[2]) << 16 | uint32_t(q[3]) << 24;
  }
  uint64_t u64() {
    const uint64_t lo = u32();
    const uint64_t hi = u32();
    return lo | hi << 32;
  }
  const uint8_t* bytes(size_t len) {
    need(len);
    const uint8_t* q = p_ + pos_;
    pos_ += len;
    return q;
  }
  // Reads a u32 record count and checks that count * record_bytes fits in the rest.
  uint32_t count(size_t record_bytes) {
    const uint32_t c = u32();
    if (uint64_t{c} * record_bytes > remaining()) fail("record count exceeds the section size");
    return c;
  }
  size_t remaining() const { return n_ - pos_; }
  [[noreturn]] void fail(const std::string& msg) const { malformed(what_ + ": " + msg); }

 private:
  void need(size_t len) {
    if (len > n_ - pos_) fail("truncated");
  }
  const uint8_t* p_;
  size_t n_;
  size_t pos_ = 0;
  std::string what_;
};

struct Range {
  uint32_t first, last, value;
};

std::vector<Range> read_ranges(Reader& r, uint32_t max_value) {
  const uint32_t n = r.count(12);
  std::vector<Range> out(n);
  for (uint32_t i = 0; i < n; ++i) {
    Range& g = out[i];
    g.first = r.u32();
    g.last = r.u32();
    g.value = r.u32();
    if (g.first > g.last || g.last >= kNumCodePoints) r.fail("bad code point range");
    if (i > 0 && g.first <= out[i - 1].last) r.fail("ranges not sorted / overlapping");
    if (g.value == 0 || g.value > max_value) r.fail("bad range value");
  }
  return out;
}

// Two-stage lookup table: one byte per code point, blocks of 256 deduplicated.
class CodePointMap {
 public:
  void build(const std::vector<uint8_t>& flat) {
    constexpr size_t kBlocks = kNumCodePoints >> 8;
    stage1_.assign(kBlocks, 0);
    stage2_.clear();
    std::unordered_map<std::string_view, uint16_t> seen;
    for (size_t b = 0; b < kBlocks; ++b) {
      const std::string_view block(reinterpret_cast<const char*>(flat.data()) + (b << 8), 256);
      auto it = seen.find(block);
      if (it == seen.end()) {
        const auto index = static_cast<uint16_t>(stage2_.size() >> 8);
        stage2_.insert(stage2_.end(), flat.begin() + static_cast<std::ptrdiff_t>(b << 8),
                       flat.begin() + static_cast<std::ptrdiff_t>((b + 1) << 8));
        it = seen.emplace(block, index).first;
      }
      stage1_[b] = it->second;
    }
  }
  uint8_t operator()(char32_t cp) const { return stage2_[(size_t{stage1_[cp >> 8]} << 8) | (cp & 0xFF)]; }

 private:
  std::vector<uint16_t> stage1_;
  std::vector<uint8_t> stage2_;
};

// Open-addressing map (left id, right id) -> (rank, merged id).
class MergeMap {
 public:
  void init(size_t n) {
    size_t cap = 16;
    int bits = 4;
    while (cap < 2 * n) {
      cap <<= 1;
      ++bits;
    }
    slots_.assign(cap, Slot{kEmpty, 0, 0});
    mask_ = cap - 1;
    shift_ = 64 - bits;
  }
  bool insert(uint32_t a, uint32_t b, uint32_t rank, int32_t result) {
    const uint64_t key = uint64_t{a} << 32 | b;
    size_t i = slot(key);
    while (slots_[i].key != kEmpty) {
      if (slots_[i].key == key) return false;
      i = (i + 1) & mask_;
    }
    slots_[i] = Slot{key, rank, result};
    return true;
  }
  // Rank of the merge (a, b) or kNoRank; *result receives the merged id.
  uint32_t find(int32_t a, int32_t b, int32_t* result) const {
    const uint64_t key = uint64_t{static_cast<uint32_t>(a)} << 32 | static_cast<uint32_t>(b);
    size_t i = slot(key);
    for (;;) {
      const Slot& s = slots_[i];
      if (s.key == key) {
        *result = s.result;
        return s.rank;
      }
      if (s.key == kEmpty) return kNoRank;
      i = (i + 1) & mask_;
    }
  }

 private:
  struct Slot {
    uint64_t key;
    uint32_t rank;
    int32_t result;
  };
  static constexpr uint64_t kEmpty = ~uint64_t{0};
  size_t slot(uint64_t key) const { return static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> shift_); }
  std::vector<Slot> slots_;
  size_t mask_ = 0;
  int shift_ = 64;
};

inline uint64_t hash_bytes(const uint8_t* p, size_t n) {
  uint64_t h = 0x9E3779B97F4A7C15ull ^ (n * 0xC2B2AE3D27D4EB4Full);
  while (n >= 8) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    h = (h ^ v) * 0xFF51AFD7ED558CCDull;
    h ^= h >> 32;
    p += 8;
    n -= 8;
  }
  uint64_t v = 0;
  std::memcpy(&v, p, n);
  h = (h ^ v) * 0xC4CEB9FE1A85EC53ull;
  h ^= h >> 29;
  return h;
}

// Per-encode() cache of BPE results keyed by piece bytes. Local to one call, so encode() needs
// no locking; when it fills up it is simply cleared.
class BpeCache {
 public:
  static constexpr size_t kMaxKey = 64;

  void prepare(size_t text_bytes) {
    size_t want = 256;
    while (want < text_bytes / 8 && want < (size_t{1} << 16)) want <<= 1;
    if (slots_.size() >= want) return;
    slots_.assign(want, Slot{});
    mask_ = want - 1;
    used_ = 0;
    keys_.clear();
    ids_.clear();
  }
  bool enabled() const { return !slots_.empty(); }

  const int32_t* find(const uint8_t* key, uint32_t len, uint64_t h, uint32_t* count) const {
    size_t i = static_cast<size_t>(h) & mask_;
    for (;;) {
      const Slot& s = slots_[i];
      if (s.key_len == 0) return nullptr;
      if (s.hash == h && s.key_len == len && std::memcmp(keys_.data() + s.key_off, key, len) == 0) {
        *count = s.ids_len;
        return ids_.data() + s.ids_off;
      }
      i = (i + 1) & mask_;
    }
  }
  void insert(const uint8_t* key, uint32_t len, uint64_t h, const int32_t* ids, uint32_t count) {
    if (2 * (used_ + 1) > slots_.size()) {
      std::fill(slots_.begin(), slots_.end(), Slot{});
      used_ = 0;
      keys_.clear();
      ids_.clear();
    }
    size_t i = static_cast<size_t>(h) & mask_;
    while (slots_[i].key_len != 0) i = (i + 1) & mask_;
    slots_[i] = Slot{h, static_cast<uint32_t>(keys_.size()), len, static_cast<uint32_t>(ids_.size()), count};
    keys_.append(reinterpret_cast<const char*>(key), len);
    ids_.insert(ids_.end(), ids, ids + count);
    ++used_;
  }

 private:
  struct Slot {
    uint64_t hash = 0;
    uint32_t key_off = 0, key_len = 0, ids_off = 0, ids_len = 0;
  };
  std::vector<Slot> slots_;
  std::string keys_;
  std::vector<int32_t> ids_;
  size_t mask_ = 0, used_ = 0;
};

struct HeapEntry {
  uint32_t rank;
  int32_t pos;
  int32_t result;
};
// std heap functions build a max-heap; "less" = larger (rank, pos), so the top is the minimum.
struct HeapLess {
  bool operator()(const HeapEntry& a, const HeapEntry& b) const {
    return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;
  }
};

// Scratch buffers of one encode() call.
struct EncodeState {
  std::string normalized;
  std::vector<char32_t> nfc_buf;
  std::vector<int32_t> ids, prev, next;
  std::vector<HeapEntry> heap;
  BpeCache cache;
};

struct AddedToken {
  int32_t id;
  uint32_t flags;
  std::string content;  // raw bytes; for normalized tokens the NFC of the content (as HF matches it)
};

struct AddedIndex {
  // Candidate tokens by first byte, longest first (so the first hit is the leftmost-longest one).
  std::array<std::vector<uint16_t>, 256> by_first;
  bool empty = true;
};

}  // namespace

// =============================================================================================
// Impl
// =============================================================================================

struct Tokenizer::Impl {
  // vocabulary
  int32_t n_ids = 0;
  int32_t n_base = 0;
  std::vector<uint32_t> tok_off;  // n_ids + 1
  std::string tok_bytes;
  std::vector<uint8_t> id_flags;  // 1 = added token, 2 = special
  std::array<int32_t, 256> byte_id{};
  MergeMap merges;
  std::vector<int32_t> sorted_ids;  // all ids with bytes, sorted by bytes (find_token)
  bool ignore_merges = false;
  bool nfc_enabled = true;

  // added tokens
  std::vector<AddedToken> added;
  AddedIndex raw_index, norm_index;

  // Unicode
  CodePointMap props;
  CodePointMap ccc;
  std::array<uint8_t, 128> ascii_props{};
  std::vector<uint32_t> dec_cp, dec_off, dec_len;
  std::vector<char32_t> dec_pool;
  std::vector<uint64_t> comp_key;  // (first << 32 | second), sorted
  std::vector<char32_t> comp_val;

  std::vector<std::pair<std::string, std::string>> meta;

  // ---- lookups ------------------------------------------------------------------------------
  uint8_t prop(char32_t cp) const { return cp < 128 ? ascii_props[cp] : props(cp); }
  std::string_view bytes_of(int32_t id) const {
    return std::string_view(tok_bytes).substr(tok_off[static_cast<size_t>(id)],
                                              tok_off[static_cast<size_t>(id) + 1] - tok_off[static_cast<size_t>(id)]);
  }
  int32_t find_bytes(std::string_view s) const {
    auto it = std::lower_bound(sorted_ids.begin(), sorted_ids.end(), s,
                               [this](int32_t id, std::string_view key) { return bytes_of(id) < key; });
    if (it != sorted_ids.end() && bytes_of(*it) == s) return *it;
    return -1;
  }

  // ---- loading ------------------------------------------------------------------------------
  void load(const uint8_t* data, size_t size);
  void build_unicode(const std::vector<Range>& cls, const std::vector<Range>& ccc_ranges);

  // ---- NFC ----------------------------------------------------------------------------------
  void decompose(char32_t cp, std::vector<char32_t>& out) const;
  char32_t compose(char32_t a, char32_t b) const;
  void normalize_chunk(const uint8_t* p, size_t n, std::vector<char32_t>& buf, std::string& out) const;
  bool nfc(std::string_view s, std::string& out, std::vector<char32_t>& buf) const;

  // ---- pre-tokenizer ------------------------------------------------------------------------
  size_t scan_letters(const uint8_t* p, size_t n, size_t i) const;
  size_t match_at(const uint8_t* p, size_t n, size_t i) const;

  // ---- BPE ----------------------------------------------------------------------------------
  void bpe_small(const uint8_t* p, size_t n, std::vector<int32_t>& out) const;
  void bpe_heap(const uint8_t* p, size_t n, std::vector<int32_t>& out, EncodeState& st) const;
  void encode_piece(const uint8_t* p, size_t n, std::vector<int32_t>& out, EncodeState& st) const;

  // ---- pipeline -----------------------------------------------------------------------------
  template <class SegmentFn, class TokenFn>
  void split_added(std::string_view text, const AddedIndex& index, SegmentFn&& segment, TokenFn&& token) const;
  void encode_pretokenized(std::string_view s, std::vector<int32_t>& out, EncodeState& st) const;
  void encode_segment(std::string_view seg, bool allow_added, std::vector<int32_t>& out, EncodeState& st) const;
  void encode(std::string_view text, bool allow_added, std::vector<int32_t>& out) const;
};

// ---------------------------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------------------------

void Tokenizer::Impl::load(const uint8_t* data, size_t size) {
  Reader header(data, size, "header");
  const uint8_t* magic = header.bytes(8);
  if (std::memcmp(magic, kMagic, 8) != 0) header.fail("bad magic (not a qwen38 tokenizer blob)");
  const uint32_t version = header.u32();
  if (version != kFormatVersion) {
    header.fail("unsupported format version " + std::to_string(version) + " (expected " +
                std::to_string(kFormatVersion) + ")");
  }
  const uint32_t n_sections = header.count(24);
  std::unordered_map<std::string, std::pair<const uint8_t*, size_t>> sections;
  for (uint32_t i = 0; i < n_sections; ++i) {
    const uint8_t* tag = header.bytes(4);
    header.u32();  // reserved
    const uint64_t off = header.u64();
    const uint64_t len = header.u64();
    if (off > size || len > size - off) header.fail("section outside the blob");
    std::string name(reinterpret_cast<const char*>(tag), 4);
    if (!sections.emplace(name, std::make_pair(data + off, static_cast<size_t>(len))).second) {
      header.fail("duplicate section " + name);
    }
  }
  auto section = [&](const char* name) {
    auto it = sections.find(name);
    if (it == sections.end()) malformed(std::string("missing section ") + name);
    return Reader(it->second.first, it->second.second, name);
  };

  // META
  {
    Reader r = section("META");
    const size_t n = r.remaining();
    const std::string_view text(reinterpret_cast<const char*>(r.bytes(n)), n);
    size_t pos = 0;
    while (pos < text.size()) {
      size_t eol = text.find('\n', pos);
      if (eol == std::string_view::npos) eol = text.size();
      const std::string_view line = text.substr(pos, eol - pos);
      const size_t eq = line.find('=');
      if (eq == std::string_view::npos) malformed("META line without '='");
      meta.emplace_back(std::string(line.substr(0, eq)), std::string(line.substr(eq + 1)));
      pos = eol + 1;
    }
    for (const auto& [k, v] : meta) {
      if (k == "normalizer") {
        if (v == "NFC") nfc_enabled = true;
        else if (v == "none") nfc_enabled = false;
        else malformed("unsupported normalizer '" + v + "'");
      } else if (k == "ignore_merges") {
        if (v != "true" && v != "false") malformed("bad ignore_merges value");
        ignore_merges = v == "true";
      }
    }
  }
  // PATN
  {
    Reader r = section("PATN");
    const size_t n = r.remaining();
    const std::string_view pattern(reinterpret_cast<const char*>(r.bytes(n)), n);
    if (pattern != kSplitPattern) {
      throw std::runtime_error("qwen38::Tokenizer: unsupported pre-tokenizer pattern (this scanner implements "
                               "only the Qwen3 split regex): " + std::string(pattern));
    }
  }
  // VOCB
  {
    Reader r = section("VOCB");
    const uint32_t n = r.u32();
    const uint32_t base = r.u32();
    if (n == 0 || n > static_cast<uint32_t>(std::numeric_limits<int32_t>::max() - 1)) r.fail("bad id count");
    if (base == 0 || base > n) r.fail("bad base vocab size");
    if (uint64_t{n + 1} * 4 > r.remaining()) r.fail("offset table truncated");
    tok_off.resize(size_t{n} + 1);
    for (auto& o : tok_off) o = r.u32();
    if (tok_off[0] != 0) r.fail("first offset is not 0");
    for (size_t i = 1; i < tok_off.size(); ++i) {
      if (tok_off[i] < tok_off[i - 1]) r.fail("offsets not monotonic");
    }
    const size_t pool = tok_off.back();
    tok_bytes.assign(reinterpret_cast<const char*>(r.bytes(pool)), pool);
    n_ids = static_cast<int32_t>(n);
    n_base = static_cast<int32_t>(base);
    id_flags.assign(n, 0);
    byte_id.fill(-1);
    for (int32_t id = 0; id < n_base; ++id) {
      const std::string_view b = bytes_of(id);
      if (b.empty()) r.fail("empty BPE vocab entry " + std::to_string(id));
      if (b.size() == 1) {
        auto& slot = byte_id[static_cast<uint8_t>(b[0])];
        if (slot >= 0) r.fail("duplicate single-byte token");
        slot = id;
      }
    }
    for (int b = 0; b < 256; ++b) {
      if (byte_id[static_cast<size_t>(b)] < 0) r.fail("byte " + std::to_string(b) + " has no token");
    }
    sorted_ids.reserve(n);
    for (int32_t id = 0; id < n_ids; ++id) {
      if (tok_off[static_cast<size_t>(id) + 1] != tok_off[static_cast<size_t>(id)]) sorted_ids.push_back(id);
    }
    std::sort(sorted_ids.begin(), sorted_ids.end(), [this](int32_t a, int32_t b) {
      const std::string_view x = bytes_of(a), y = bytes_of(b);
      return x != y ? x < y : a < b;
    });
  }
  // MERG
  {
    Reader r = section("MERG");
    const uint32_t n = r.count(12);
    merges.init(n);
    for (uint32_t rank = 0; rank < n; ++rank) {
      const uint32_t a = r.u32(), b = r.u32(), c = r.u32();
      if (a >= static_cast<uint32_t>(n_base) || b >= static_cast<uint32_t>(n_base) ||
          c >= static_cast<uint32_t>(n_base)) {
        r.fail("merge references an id outside the BPE vocab");
      }
      const std::string_view x = bytes_of(static_cast<int32_t>(a)), y = bytes_of(static_cast<int32_t>(b)),
                             z = bytes_of(static_cast<int32_t>(c));
      if (z.size() != x.size() + y.size() || z.substr(0, x.size()) != x || z.substr(x.size()) != y) {
        r.fail("merge result is not the concatenation of its parts (rank " + std::to_string(rank) + ")");
      }
      if (!merges.insert(a, b, rank, static_cast<int32_t>(c))) r.fail("duplicate merge");
    }
  }
  // Unicode tables (needed before the added tokens: normalized tokens are NFC'd)
  {
    Reader rc = section("UCLS");
    const std::vector<Range> cls = read_ranges(rc, kClassMask);
    Reader rcc = section("UCCC");
    const std::vector<Range> ccc_ranges = read_ranges(rcc, 254);

    Reader rd = section("UDEC");
    const uint32_t nd = rd.count(12);
    dec_cp.resize(nd);
    dec_off.resize(nd);
    dec_len.resize(nd);
    for (uint32_t i = 0; i < nd; ++i) {
      dec_cp[i] = rd.u32();
      dec_off[i] = rd.u32();
      dec_len[i] = rd.u32();
      if (dec_cp[i] >= kNumCodePoints || (i > 0 && dec_cp[i] <= dec_cp[i - 1])) {
        rd.fail("decomposition table not sorted");
      }
      if (dec_cp[i] - kSBase < kSCount) rd.fail("Hangul syllable in the decomposition table");
      if (dec_len[i] == 0 || dec_len[i] > 32) rd.fail("bad decomposition length");
    }
    const size_t pool_n = rd.remaining() / 4;
    dec_pool.resize(pool_n);
    for (auto& cp : dec_pool) {
      cp = rd.u32();
      if (cp >= kNumCodePoints || (cp >= 0xD800 && cp <= 0xDFFF)) rd.fail("bad code point in decomposition");
    }
    for (uint32_t i = 0; i < nd; ++i) {
      if (uint64_t{dec_off[i]} + dec_len[i] > pool_n) rd.fail("decomposition outside the pool");
    }

    Reader rp = section("UCMP");
    const uint32_t np = rp.count(12);
    comp_key.resize(np);
    comp_val.resize(np);
    for (uint32_t i = 0; i < np; ++i) {
      const uint32_t a = rp.u32(), b = rp.u32(), c = rp.u32();
      if (a >= kNumCodePoints || b >= kNumCodePoints || c >= kNumCodePoints || c == 0) rp.fail("bad composition");
      comp_key[i] = uint64_t{a} << 32 | b;
      comp_val[i] = c;
      if (i > 0 && comp_key[i] <= comp_key[i - 1]) rp.fail("composition table not sorted");
    }
    section("UEXC");  // informational; must exist
    build_unicode(cls, ccc_ranges);
  }
  // ADDT
  {
    Reader r = section("ADDT");
    const uint32_t n = r.count(16);
    if (n > 65535) r.fail("too many added tokens");
    struct Rec {
      uint32_t id, flags, off, len;
    };
    std::vector<Rec> recs(n);
    for (auto& rec : recs) {
      rec.id = r.u32();
      rec.flags = r.u32();
      rec.off = r.u32();
      rec.len = r.u32();
    }
    const size_t pool_n = r.remaining();
    const uint8_t* pool = r.bytes(pool_n);
    std::string scratch;
    std::vector<char32_t> buf;
    for (const Rec& rec : recs) {
      if (rec.id >= static_cast<uint32_t>(n_ids)) r.fail("added token id out of range");
      if (rec.flags & ~uint32_t{31}) r.fail("unknown added token flags");
      if (rec.len == 0 || uint64_t{rec.off} + rec.len > pool_n) r.fail("bad added token content");
      std::string content(reinterpret_cast<const char*>(pool) + rec.off, rec.len);
      if (!utf8_valid(content)) r.fail("added token content is not UTF-8");
      if (bytes_of(static_cast<int32_t>(rec.id)) != content) r.fail("added token content differs from its vocab bytes");
      id_flags[rec.id] |= 1;
      if (rec.flags & kAddedSpecial) id_flags[rec.id] |= 2;
      if ((rec.flags & kAddedNormalized) && nfc_enabled && nfc(content, scratch, buf)) content = scratch;
      added.push_back(AddedToken{static_cast<int32_t>(rec.id), rec.flags, std::move(content)});
    }
    // Index: by first byte, longest content first (ties: blob order, i.e. id order).
    std::vector<uint16_t> order(added.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<uint16_t>(i);
    std::stable_sort(order.begin(), order.end(),
                     [this](uint16_t a, uint16_t b) { return added[a].content.size() > added[b].content.size(); });
    for (uint16_t i : order) {
      AddedIndex& index = (added[i].flags & kAddedNormalized) ? norm_index : raw_index;
      index.by_first[static_cast<uint8_t>(added[i].content[0])].push_back(i);
      index.empty = false;
    }
  }
}

void Tokenizer::Impl::build_unicode(const std::vector<Range>& cls, const std::vector<Range>& ccc_ranges) {
  std::vector<uint8_t> flat(kNumCodePoints, 0);
  std::vector<uint8_t> ccc_flat(kNumCodePoints, 0);
  for (const Range& g : cls) {
    std::fill(flat.begin() + g.first, flat.begin() + g.last + 1, static_cast<uint8_t>(g.value));
  }
  for (const Range& g : ccc_ranges) {
    std::fill(ccc_flat.begin() + g.first, ccc_flat.begin() + g.last + 1, static_cast<uint8_t>(g.value));
  }
  for (uint32_t cp : dec_cp) flat[cp] |= kHasDecomp;

  std::vector<uint8_t> second(kNumCodePoints, 0);
  for (uint64_t key : comp_key) second[static_cast<uint32_t>(key)] = 1;
  for (char32_t v = kVBase; v < kVBase + kVCount; ++v) second[v] = 1;
  for (char32_t t = kTBase + 1; t < kTBase + kTCount; ++t) second[t] = 1;

  for (char32_t cp = 0; cp < kNumCodePoints; ++cp) {
    if (ccc_flat[cp] != 0 || second[cp]) continue;
    const bool hangul = cp - kSBase < kSCount;
    if (!(flat[cp] & kHasDecomp) && !hangul) {
      flat[cp] |= kNfcQuick | kNfcBoundary;
      continue;
    }
    char32_t first = hangul ? kLBase : 0;
    if (!hangul) {
      const auto it = std::lower_bound(dec_cp.begin(), dec_cp.end(), static_cast<uint32_t>(cp));
      first = dec_pool[dec_off[static_cast<size_t>(it - dec_cp.begin())]];
    }
    if (ccc_flat[first] == 0 && !second[first]) flat[cp] |= kNfcBoundary;
  }
  props.build(flat);
  ccc.build(ccc_flat);
  for (char32_t cp = 0; cp < 128; ++cp) ascii_props[cp] = flat[cp];
}

// ---------------------------------------------------------------------------------------------
// NFC (UAX #15: canonical decomposition, canonical ordering, canonical composition)
// ---------------------------------------------------------------------------------------------

void Tokenizer::Impl::decompose(char32_t cp, std::vector<char32_t>& out) const {
  if (cp - kSBase < kSCount) {
    const char32_t s = cp - kSBase;
    out.push_back(kLBase + s / kNCount);
    out.push_back(kVBase + (s % kNCount) / kTCount);
    if (s % kTCount) out.push_back(kTBase + s % kTCount);
    return;
  }
  if (!(prop(cp) & kHasDecomp)) {
    out.push_back(cp);
    return;
  }
  const auto it = std::lower_bound(dec_cp.begin(), dec_cp.end(), static_cast<uint32_t>(cp));
  const size_t k = static_cast<size_t>(it - dec_cp.begin());
  out.insert(out.end(), dec_pool.begin() + dec_off[k], dec_pool.begin() + dec_off[k] + dec_len[k]);
}

char32_t Tokenizer::Impl::compose(char32_t a, char32_t b) const {
  if (a - kLBase < kLCount && b - kVBase < kVCount) {
    return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
  }
  if (a - kSBase < kSCount && (a - kSBase) % kTCount == 0 && b - kTBase - 1 < kTCount - 1) {
    return a + (b - kTBase);
  }
  const uint64_t key = uint64_t{a} << 32 | b;
  const auto it = std::lower_bound(comp_key.begin(), comp_key.end(), key);
  if (it != comp_key.end() && *it == key) return comp_val[static_cast<size_t>(it - comp_key.begin())];
  return 0;
}

void Tokenizer::Impl::normalize_chunk(const uint8_t* p, size_t n, std::vector<char32_t>& buf, std::string& out) const {
  buf.clear();
  for (size_t i = 0; i < n;) {
    uint32_t len;
    const char32_t cp = decode_at(p, i, len);
    i += len;
    decompose(cp, buf);
  }
  // Canonical ordering: stable sort of every run of non-starters by combining class.
  const size_t m = buf.size();
  for (size_t i = 0; i < m;) {
    if (ccc(buf[i]) == 0) {
      ++i;
      continue;
    }
    size_t j = i + 1;
    while (j < m && ccc(buf[j]) != 0) ++j;
    if (j - i > 1) {
      auto by_ccc = [this](char32_t x, char32_t y) { return ccc(x) < ccc(y); };
      if (j - i <= 32) {
        for (size_t k = i + 1; k < j; ++k) {  // insertion sort (stable)
          const char32_t v = buf[k];
          size_t q = k;
          while (q > i && by_ccc(v, buf[q - 1])) {
            buf[q] = buf[q - 1];
            --q;
          }
          buf[q] = v;
        }
      } else {
        std::stable_sort(buf.begin() + static_cast<std::ptrdiff_t>(i), buf.begin() + static_cast<std::ptrdiff_t>(j),
                         by_ccc);
      }
    }
    i = j;
  }
  // Canonical composition.
  if (m > 0) {
    size_t starter = 0;
    size_t w = 1;
    int last_class = ccc(buf[0]);
    if (last_class != 0) last_class = 256;  // no starter yet: nothing may compose
    for (size_t i = 1; i < m; ++i) {
      const char32_t cp = buf[i];
      const int cls = ccc(cp);
      if (last_class < cls || last_class == 0) {
        const char32_t c = compose(buf[starter], cp);
        if (c != 0) {
          buf[starter] = c;
          continue;
        }
      }
      if (cls == 0) starter = w;
      last_class = cls;
      buf[w++] = cp;
    }
    buf.resize(w);
  }
  for (char32_t cp : buf) append_utf8(out, cp);
}

// Writes NFC(s) to `out` and returns true, or returns false if s is already NFC (common case).
bool Tokenizer::Impl::nfc(std::string_view s, std::string& out, std::vector<char32_t>& buf) const {
  const auto* p = reinterpret_cast<const uint8_t*>(s.data());
  const size_t n = s.size();
  size_t i = 0;
  uint32_t len = 0;
  for (;;) {
    while (i < n && p[i] < 0x80) ++i;  // ASCII is always quick
    if (i == n) return false;
    const char32_t cp = decode_at(p, i, len);
    if (!(prop(cp) & kNfcQuick)) break;
    i += len;
  }
  // The chunk containing i starts at i if that character is a boundary, else at the previous
  // character (which is quick, hence a boundary).
  size_t start = i;
  {
    const char32_t cp = decode_at(p, i, len);
    if (!(prop(cp) & kNfcBoundary) && i > 0) start = prev_start(p, i);
  }
  out.assign(s.data(), start);
  size_t j = start;
  while (j < n) {
    const size_t chunk = j;
    char32_t cp = decode_at(p, j, len);
    j += len;
    bool dirty = !(prop(cp) & kNfcQuick);
    while (j < n) {
      cp = decode_at(p, j, len);
      if (prop(cp) & kNfcBoundary) break;
      dirty = true;
      j += len;
    }
    if (dirty) {
      normalize_chunk(p + chunk, j - chunk, buf, out);
    } else {
      out.append(s.data() + chunk, j - chunk);
    }
  }
  return true;
}

// ---------------------------------------------------------------------------------------------
// Pre-tokenizer: hand-written equivalent of the Qwen3 split regex (Oniguruma, leftmost-first
// alternation with backtracking)
//
//   1 (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   2 [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//   3 \p{N}
//   4  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   5 \s*[\r\n]+
//   6 \s+(?!\S)
//   7 \s+
// ---------------------------------------------------------------------------------------------

inline char32_t fold_contraction(char32_t c) {
  if (c >= 'A' && c <= 'Z') return c + 32;
  if (c == 0x017F) return 's';  // LATIN SMALL LETTER LONG S case-folds to 's' in Oniguruma
  return c;
}

size_t Tokenizer::Impl::scan_letters(const uint8_t* p, size_t n, size_t i) const {
  while (i < n) {
    uint32_t len;
    const char32_t cp = decode_at(p, i, len);
    if (!(prop(cp) & (kL | kM))) break;
    i += len;
  }
  return i;
}

// End (byte offset) of the regex match that starts at byte i < n.
size_t Tokenizer::Impl::match_at(const uint8_t* p, size_t n, size_t i) const {
  uint32_t len;
  const char32_t c = decode_at(p, i, len);
  const uint8_t f = prop(c);
  const size_t j = i + len;

  // 1: contractions
  if (c == '\'' && j < n) {
    uint32_t l1;
    const char32_t c1 = fold_contraction(decode_at(p, j, l1));
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return j + l1;
    if ((c1 == 'r' || c1 == 'v' || c1 == 'l') && j + l1 < n) {
      uint32_t l2;
      const char32_t c2 = fold_contraction(decode_at(p, j + l1, l2));
      if (c1 == 'l' ? c2 == 'l' : c2 == 'e') return j + l1 + l2;
    }
  }
  // 2: optional prefix (anything but \r \n L N) followed by letters/marks. When the prefix
  // cannot be followed by a letter/mark, the regex backtracks to "no prefix".
  if (!(f & (kL | kN)) && c != '\r' && c != '\n' && j < n) {
    uint32_t l1;
    const char32_t c1 = decode_at(p, j, l1);
    if (prop(c1) & (kL | kM)) return scan_letters(p, n, j + l1);
  }
  if (f & (kL | kM)) return scan_letters(p, n, j);
  // 3: a single number character
  if (f & kN) return j;
  // 4: optional ASCII space, run of "other" characters, trailing CR/LF
  {
    size_t k = c == ' ' ? j : i;
    if (k < n) {
      uint32_t l1;
      const char32_t c1 = decode_at(p, k, l1);
      if (!(prop(c1) & (kL | kM | kN | kWS))) {
        k += l1;
        while (k < n) {
          uint32_t l2;
          const char32_t c2 = decode_at(p, k, l2);
          if (prop(c2) & (kL | kM | kN | kWS)) break;
          k += l2;
        }
        while (k < n && (p[k] == '\r' || p[k] == '\n')) ++k;
        return k;
      }
    }
  }
  if (!(f & kWS)) return j;  // unreachable: every character is L, M, N, \s or "other"
  // 5-7: whitespace run [i, e)
  size_t e = i, nl_end = 0, last_start = i;
  while (e < n) {
    uint32_t l1;
    const char32_t c1 = decode_at(p, e, l1);
    if (!(prop(c1) & kWS)) break;
    if (c1 == '\r' || c1 == '\n') nl_end = e + l1;
    last_start = e;
    e += l1;
  }
  if (nl_end != 0) return nl_end;       // 5: up to and including the last CR/LF of the run
  if (e == n) return e;                 // 6: the run ends the text
  if (last_start > i) return last_start;  // 6: all but the last whitespace character
  return e;                             // 7: a single whitespace character
}

// ---------------------------------------------------------------------------------------------
// BPE
// ---------------------------------------------------------------------------------------------

// Both variants apply, until none is left, the adjacent pair with the lowest rank (leftmost on
// ties) - the order HF's heap-based Word::merge_all produces, which matters because this merge
// list is not strictly "creation-ordered".
constexpr size_t kSmallWord = 48;

void Tokenizer::Impl::bpe_small(const uint8_t* p, size_t n, std::vector<int32_t>& out) const {
  int32_t id[kSmallWord], res[kSmallWord], prev[kSmallWord], next[kSmallWord];
  uint32_t rank[kSmallWord];
  const auto end = static_cast<int32_t>(n);
  for (int32_t k = 0; k < end; ++k) {
    id[k] = byte_id[p[k]];
    prev[k] = k - 1;
    next[k] = k + 1;
  }
  for (int32_t k = 0; k + 1 < end; ++k) rank[k] = merges.find(id[k], id[k + 1], &res[k]);
  rank[end - 1] = kNoRank;
  for (;;) {
    uint32_t best = kNoRank;
    int32_t bi = -1;
    for (int32_t k = 0; k != end; k = next[k]) {
      if (rank[k] < best) {
        best = rank[k];
        bi = k;
      }
    }
    if (bi < 0) break;
    const int32_t j = next[bi];
    id[bi] = res[bi];
    next[bi] = next[j];
    if (next[j] != end) prev[next[j]] = bi;
    rank[bi] = next[bi] != end ? merges.find(id[bi], id[next[bi]], &res[bi]) : kNoRank;
    if (prev[bi] >= 0) rank[prev[bi]] = merges.find(id[prev[bi]], id[bi], &res[prev[bi]]);
  }
  for (int32_t k = 0; k != end; k = next[k]) out.push_back(id[k]);
}

void Tokenizer::Impl::bpe_heap(const uint8_t* p, size_t n, std::vector<int32_t>& out, EncodeState& st) const {
  auto& id = st.ids;
  auto& prev = st.prev;
  auto& next = st.next;
  auto& heap = st.heap;
  const auto end = static_cast<int32_t>(n);
  id.resize(n);
  prev.resize(n);
  next.resize(n);
  heap.clear();
  for (int32_t k = 0; k < end; ++k) {
    id[static_cast<size_t>(k)] = byte_id[p[k]];
    prev[static_cast<size_t>(k)] = k - 1;
    next[static_cast<size_t>(k)] = k + 1;
  }
  int32_t res;
  for (int32_t k = 0; k + 1 < end; ++k) {
    const uint32_t r = merges.find(id[static_cast<size_t>(k)], id[static_cast<size_t>(k) + 1], &res);
    if (r != kNoRank) heap.push_back(HeapEntry{r, k, res});
  }
  std::make_heap(heap.begin(), heap.end(), HeapLess{});
  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), HeapLess{});
    const HeapEntry top = heap.back();
    heap.pop_back();
    const auto pos = static_cast<size_t>(top.pos);
    if (id[pos] < 0) continue;  // absorbed by its left neighbour
    const int32_t j = next[pos];
    if (j == end) continue;
    // Stale entry (a neighbour changed since it was pushed): the pair no longer merges into
    // the same token.
    if (merges.find(id[pos], id[static_cast<size_t>(j)], &res) == kNoRank || res != top.result) continue;
    id[pos] = top.result;
    id[static_cast<size_t>(j)] = -1;
    next[pos] = next[static_cast<size_t>(j)];
    if (next[pos] != end) prev[static_cast<size_t>(next[pos])] = top.pos;
    if (prev[pos] >= 0) {
      const auto q = static_cast<size_t>(prev[pos]);
      const uint32_t r = merges.find(id[q], id[pos], &res);
      if (r != kNoRank) {
        heap.push_back(HeapEntry{r, prev[pos], res});
        std::push_heap(heap.begin(), heap.end(), HeapLess{});
      }
    }
    if (next[pos] != end) {
      const uint32_t r = merges.find(id[pos], id[static_cast<size_t>(next[pos])], &res);
      if (r != kNoRank) {
        heap.push_back(HeapEntry{r, top.pos, res});
        std::push_heap(heap.begin(), heap.end(), HeapLess{});
      }
    }
  }
  for (int32_t k = 0; k != end; k = next[static_cast<size_t>(k)]) out.push_back(id[static_cast<size_t>(k)]);
}

void Tokenizer::Impl::encode_piece(const uint8_t* p, size_t n, std::vector<int32_t>& out, EncodeState& st) const {
  if (n == 1) {
    out.push_back(byte_id[p[0]]);
    return;
  }
  if (ignore_merges) {
    const int32_t id = find_bytes(std::string_view(reinterpret_cast<const char*>(p), n));
    if (id >= 0 && id < n_base) {
      out.push_back(id);
      return;
    }
  }
  const bool cacheable = st.cache.enabled() && n <= BpeCache::kMaxKey;
  uint64_t h = 0;
  if (cacheable) {
    h = hash_bytes(p, n);
    uint32_t count;
    if (const int32_t* hit = st.cache.find(p, static_cast<uint32_t>(n), h, &count)) {
      out.insert(out.end(), hit, hit + count);
      return;
    }
  }
  const size_t before = out.size();
  if (n <= kSmallWord) {
    bpe_small(p, n, out);
  } else {
    bpe_heap(p, n, out, st);
  }
  if (cacheable) {
    st.cache.insert(p, static_cast<uint32_t>(n), h, out.data() + before, static_cast<uint32_t>(out.size() - before));
  }
}

// ---------------------------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------------------------

// HF AddedVocabulary::find_matches: leftmost-longest matches of the automaton over the whole
// text, then single_word / lstrip / rstrip adjustments per match.
template <class SegmentFn, class TokenFn>
void Tokenizer::Impl::split_added(std::string_view text, const AddedIndex& index, SegmentFn&& segment,
                                  TokenFn&& token) const {
  const auto* p = reinterpret_cast<const uint8_t*>(text.data());
  const size_t n = text.size();
  size_t start_offset = 0;
  size_t i = 0;
  while (i < n) {
    const auto& candidates = index.by_first[p[i]];
    if (candidates.empty()) {
      ++i;
      continue;
    }
    const AddedToken* hit = nullptr;
    for (uint16_t k : candidates) {
      const AddedToken& t = added[k];
      if (t.content.size() <= n - i && std::memcmp(p + i, t.content.data(), t.content.size()) == 0) {
        hit = &t;
        break;
      }
    }
    if (hit == nullptr) {
      ++i;
      continue;
    }
    size_t start = i;
    size_t stop = i + hit->content.size();
    i = stop;
    if (hit->flags & kAddedSingleWord) {
      uint32_t len;
      const bool word_before = start > 0 && (prop(decode_at(p, prev_start(p, start), len)) & kWord);
      const bool word_after = stop < n && (prop(decode_at(p, stop, len)) & kWord);
      if (word_before || word_after) continue;
    }
    if (hit->flags & kAddedLstrip) {
      size_t s = start;
      uint32_t len;
      while (s > 0) {
        const size_t q = prev_start(p, s);
        if (!(prop(decode_at(p, q, len)) & kWS)) break;
        s = q;
      }
      start = std::max(s, start_offset);
    }
    if (hit->flags & kAddedRstrip) {
      uint32_t len;
      while (stop < n && (prop(decode_at(p, stop, len)) & kWS)) stop += len;
    }
    if (start_offset < start) segment(text.substr(start_offset, start - start_offset));
    token(hit->id);
    start_offset = stop;
  }
  if (start_offset < n) segment(text.substr(start_offset));
}

void Tokenizer::Impl::encode_pretokenized(std::string_view s, std::vector<int32_t>& out, EncodeState& st) const {
  const auto* p = reinterpret_cast<const uint8_t*>(s.data());
  const size_t n = s.size();
  for (size_t i = 0; i < n;) {
    const size_t e = match_at(p, n, i);
    encode_piece(p + i, e - i, out, st);
    i = e;
  }
}

void Tokenizer::Impl::encode_segment(std::string_view seg, bool allow_added, std::vector<int32_t>& out,
                                     EncodeState& st) const {
  std::string_view text = seg;
  if (nfc_enabled && nfc(seg, st.normalized, st.nfc_buf)) text = st.normalized;
  if (allow_added && !norm_index.empty) {
    split_added(
        text, norm_index, [&](std::string_view sub) { encode_pretokenized(sub, out, st); },
        [&](int32_t id) { out.push_back(id); });
  } else {
    encode_pretokenized(text, out, st);
  }
}

void Tokenizer::Impl::encode(std::string_view text, bool allow_added, std::vector<int32_t>& out) const {
  std::string sanitized;
  if (!utf8_valid(text)) {
    sanitized = utf8_sanitize(text);
    text = sanitized;
  }
  EncodeState st;
  if (text.size() >= 256) st.cache.prepare(text.size());
  if (allow_added && !raw_index.empty) {
    split_added(
        text, raw_index, [&](std::string_view seg) { encode_segment(seg, allow_added, out, st); },
        [&](int32_t id) { out.push_back(id); });
  } else {
    encode_segment(text, allow_added, out, st);
  }
}

// =============================================================================================
// Public API
// =============================================================================================

Tokenizer::Tokenizer() = default;
Tokenizer::Tokenizer(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

const Tokenizer::Impl& Tokenizer::impl() const {
  if (!impl_) throw std::logic_error("qwen38::Tokenizer: empty tokenizer (use from_blob / from_file)");
  return *impl_;
}

Tokenizer Tokenizer::from_blob(const void* data, size_t size) {
  if (data == nullptr && size != 0) throw std::runtime_error("qwen38::Tokenizer: null blob");
  auto impl = std::make_shared<Impl>();
  impl->load(static_cast<const uint8_t*>(data), size);
  return Tokenizer(std::move(impl));
}

Tokenizer Tokenizer::from_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("qwen38::Tokenizer: cannot open " + path);
  std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (f.bad()) throw std::runtime_error("qwen38::Tokenizer: cannot read " + path);
  return from_blob(data.data(), data.size());
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool allow_added_tokens) const {
  std::vector<int32_t> out;
  out.reserve(text.size() / 3 + 4);
  impl().encode(text, allow_added_tokens, out);
  return out;
}

void Tokenizer::encode(std::string_view text, std::vector<int32_t>& out, bool allow_added_tokens) const {
  impl().encode(text, allow_added_tokens, out);
}

std::string Tokenizer::decode(std::span<const int32_t> ids) const {
  std::string out;
  size_t total = 0;
  for (int32_t id : ids) total += token_bytes(id).size();
  out.reserve(total);
  for (int32_t id : ids) out.append(token_bytes(id));
  return out;
}

std::string_view Tokenizer::token_bytes(int32_t id) const {
  if (!impl_ || id < 0 || id >= impl_->n_ids) return {};
  return impl_->bytes_of(id);
}

int32_t Tokenizer::find_token(std::string_view token_text) const {
  if (!impl_ || token_text.empty()) return -1;
  return impl_->find_bytes(token_text);
}

bool Tokenizer::is_special(int32_t id) const {
  return impl_ && id >= 0 && id < impl_->n_ids && (impl_->id_flags[static_cast<size_t>(id)] & 2);
}

bool Tokenizer::is_added(int32_t id) const {
  return impl_ && id >= 0 && id < impl_->n_ids && (impl_->id_flags[static_cast<size_t>(id)] & 1);
}

int32_t Tokenizer::vocab_size() const { return impl_ ? impl_->n_ids : 0; }

std::string Tokenizer::normalize(std::string_view text) const {
  const Impl& im = impl();
  std::string sanitized;
  if (!utf8_valid(text)) {
    sanitized = utf8_sanitize(text);
    text = sanitized;
  }
  std::string out;
  std::vector<char32_t> buf;
  if (im.nfc_enabled && im.nfc(text, out, buf)) return out;
  return std::string(text);
}

std::vector<std::string> Tokenizer::pre_tokenize(std::string_view text) const {
  const Impl& im = impl();
  const std::string norm = normalize(text);
  const auto* p = reinterpret_cast<const uint8_t*>(norm.data());
  std::vector<std::string> pieces;
  for (size_t i = 0; i < norm.size();) {
    const size_t e = im.match_at(p, norm.size(), i);
    pieces.emplace_back(norm.substr(i, e - i));
    i = e;
  }
  return pieces;
}

std::string Tokenizer::metadata(std::string_view key) const {
  if (!impl_) return {};
  for (const auto& [k, v] : impl_->meta) {
    if (k == key) return v;
  }
  return {};
}

// ---------------------------------------------------------------------------------------------
// Utf8StreamDecoder
// ---------------------------------------------------------------------------------------------

std::string Utf8StreamDecoder::push(std::string_view bytes) {
  std::string joined;
  if (!pending_.empty()) {
    joined.swap(pending_);
    joined.append(bytes);
    bytes = joined;
  }
  const auto* p = reinterpret_cast<const uint8_t*>(bytes.data());
  const size_t n = bytes.size();
  std::string out;
  out.reserve(n);
  size_t i = 0;
  while (i < n) {
    if (p[i] < 0x80) {
      out.push_back(static_cast<char>(p[i]));
      ++i;
      continue;
    }
    const Utf8Result r = utf8_step(p + i, n - i);
    if (r.step == Utf8Step::kValid) {
      out.append(bytes.data() + i, r.len);
    } else if (r.step == Utf8Step::kInvalid) {
      out.append(kReplacementUtf8);
    } else {
      pending_.assign(bytes.data() + i, n - i);  // a valid but incomplete prefix
      break;
    }
    i += r.len;
  }
  return out;
}

std::string Utf8StreamDecoder::flush() {
  std::string out;
  if (!pending_.empty()) out.assign(kReplacementUtf8);  // one maximal subpart
  pending_.clear();
  return out;
}

}  // namespace qwen38
