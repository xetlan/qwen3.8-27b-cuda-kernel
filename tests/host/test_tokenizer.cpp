// Tokenizer tests with Hugging Face `tokenizers` as the ground truth.
//
// Environment:
//   QWEN38_TOKENIZER_BLOB   blob written by tools/tokenizer_export.py
//   QWEN38_TOKENIZER_CASES  cases written by tests/host/tokenizer_cases.py
//
// Regenerate both (from the repository root):
//   .venv/bin/python tools/tokenizer_export.py testdata/qwen3.8/tokenizer.json testdata/qwen3.8/tokenizer.bin
//   .venv/bin/python tests/host/tokenizer_cases.py testdata/qwen3.8/tokenizer.json
//       testdata/qwen3.8/tokenizer_cases.bin
//
// Build and run by hand (without CMake):
//   c++ -std=c++20 -O2 -Wall -Wextra -Iinclude -Itests src/tokenizer.cpp tests/host/test_tokenizer.cpp
//       tests/test_main.cpp -o /tmp/test_tokenizer
//   QWEN38_TOKENIZER_BLOB=testdata/qwen3.8/tokenizer.bin
//       QWEN38_TOKENIZER_CASES=testdata/qwen3.8/tokenizer_cases.bin /tmp/test_tokenizer

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "qwen38/tokenizer.h"
#include "test_util.h"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

std::string read_file(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error(std::string("cannot open ") + path);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

const char* env_path(const char* var, const char* what) {
  const char* v = std::getenv(var);
  if (v == nullptr || *v == '\0') {
    std::fprintf(stderr,
                 "%s is not set: point it at %s.\n"
                 "  generate: .venv/bin/python tools/tokenizer_export.py testdata/qwen3.8/tokenizer.json "
                 "testdata/qwen3.8/tokenizer.bin\n"
                 "            .venv/bin/python tests/host/tokenizer_cases.py testdata/qwen3.8/tokenizer.json "
                 "testdata/qwen3.8/tokenizer_cases.bin\n",
                 var, what);
    return nullptr;
  }
  return v;
}

struct Fixture {
  std::string blob;
  qwen38::Tokenizer tok;
};

const Fixture* fixture() {
  static std::unique_ptr<Fixture> fx;
  static bool tried = false;
  if (!tried) {
    tried = true;
    const char* path = env_path("QWEN38_TOKENIZER_BLOB", "the tokenizer blob (testdata/qwen3.8/tokenizer.bin)");
    if (path != nullptr) {
      auto f = std::make_unique<Fixture>();
      f->blob = read_file(path);
      const auto t0 = Clock::now();
      f->tok = qwen38::Tokenizer::from_blob(f->blob.data(), f->blob.size());
      std::printf("  loaded %s (%zu bytes) in %.1f ms\n", path, f->blob.size(), 1e3 * seconds_since(t0));
      fx = std::move(f);
    }
  }
  return fx.get();
}

struct Case {
  uint32_t flags = 0;
  std::string label, input, decoded;
  std::vector<int32_t> ids;
  bool allow_added() const { return flags & 1; }
  std::string category() const { return label.substr(0, label.find('/')); }
};

class CaseReader {
 public:
  explicit CaseReader(const std::string& data) : d_(data) {}
  uint32_t u32() {
    need(4);
    uint32_t v;
    std::memcpy(&v, d_.data() + pos_, 4);  // the case file is little-endian, as are all hosts we build on
    pos_ += 4;
    return v;
  }
  std::string bytes() {
    const uint32_t n = u32();
    need(n);
    std::string s = d_.substr(pos_, n);
    pos_ += n;
    return s;
  }
  std::vector<int32_t> ids() {
    const uint32_t n = u32();
    need(size_t{n} * 4);
    std::vector<int32_t> v(n);
    if (n != 0) std::memcpy(v.data(), d_.data() + pos_, size_t{n} * 4);
    pos_ += size_t{n} * 4;
    return v;
  }
  bool at_end() const { return pos_ == d_.size(); }

 private:
  void need(size_t n) const {
    if (n > d_.size() - pos_) throw std::runtime_error("tokenizer cases file is truncated");
  }
  const std::string& d_;
  size_t pos_ = 0;
};

const std::vector<Case>* cases() {
  static std::unique_ptr<std::vector<Case>> cs;
  static bool tried = false;
  if (!tried) {
    tried = true;
    const char* path = env_path("QWEN38_TOKENIZER_CASES", "the cases file (testdata/qwen3.8/tokenizer_cases.bin)");
    if (path != nullptr) {
      const std::string data = read_file(path);
      if (data.size() < 16 || data.compare(0, 8, "QW38TCS1") != 0) {
        throw std::runtime_error(std::string(path) + " is not a tokenizer cases file");
      }
      CaseReader r(data);
      r.u32();  // magic, first half
      r.u32();  // magic, second half
      if (r.u32() != 1) throw std::runtime_error("unsupported tokenizer cases version");
      const uint32_t n = r.u32();
      auto v = std::make_unique<std::vector<Case>>(n);
      for (Case& c : *v) {
        c.flags = r.u32();
        c.label = r.bytes();
        c.input = r.bytes();
        c.ids = r.ids();
        c.decoded = r.bytes();
      }
      if (!r.at_end()) throw std::runtime_error("trailing bytes in the tokenizer cases file");
      cs = std::move(v);
    }
  }
  return cs.get();
}

// Printable rendering of arbitrary bytes: ASCII as is, common escapes, other code points as \u{..},
// invalid bytes as \xNN.
std::string escape(std::string_view s, size_t max_chars = 160) {
  std::string out;
  size_t chars = 0;
  for (size_t i = 0; i < s.size() && chars < max_chars; ++chars) {
    const auto b = static_cast<unsigned char>(s[i]);
    char buf[16];
    if (b < 0x80) {
      if (b == '\n') out += "\\n";
      else if (b == '\r') out += "\\r";
      else if (b == '\t') out += "\\t";
      else if (b == '\\') out += "\\\\";
      else if (b == '"') out += "\\\"";
      else if (b < 0x20 || b == 0x7F) {
        std::snprintf(buf, sizeof buf, "\\x%02X", b);
        out += buf;
      } else {
        out += static_cast<char>(b);
      }
      ++i;
      continue;
    }
    const int len = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
    bool ok = len > 1 && i + static_cast<size_t>(len) <= s.size();
    uint32_t cp = b & (0x7F >> len);
    for (int k = 1; ok && k < len; ++k) {
      const auto c = static_cast<unsigned char>(s[i + static_cast<size_t>(k)]);
      ok = (c & 0xC0) == 0x80;
      cp = (cp << 6) | (c & 0x3F);
    }
    if (ok) {
      std::snprintf(buf, sizeof buf, "\\u{%X}", cp);
      i += static_cast<size_t>(len);
    } else {
      std::snprintf(buf, sizeof buf, "\\x%02X", b);
      ++i;
    }
    out += buf;
  }
  if (chars >= max_chars) out += "...";
  return out;
}

std::string ids_window(const qwen38::Tokenizer& tok, const std::vector<int32_t>& ids, size_t from, size_t to) {
  std::string out;
  for (size_t i = from; i < to && i < ids.size(); ++i) {
    out += std::to_string(ids[i]) + "\"" + escape(tok.token_bytes(ids[i]), 24) + "\" ";
  }
  return out;
}

void report_mismatch(const qwen38::Tokenizer& tok, size_t index, const Case& c, const std::vector<int32_t>& got,
                     const std::string& dec, const std::string& streamed) {
  std::fprintf(stderr, "---- case %zu [%s] allow_added_tokens=%d, input %zu bytes\n", index, c.label.c_str(),
               c.allow_added() ? 1 : 0, c.input.size());
  size_t diff = 0;
  while (diff < got.size() && diff < c.ids.size() && got[diff] == c.ids[diff]) ++diff;
  // show the input around the first differing token
  size_t offset = 0;
  for (size_t i = 0; i < diff; ++i) offset += tok.token_bytes(got[i]).size();
  const size_t ctx_from = offset > 60 ? offset - 60 : 0;
  std::fprintf(stderr, "  input: \"%s\"\n", escape(c.input, 200).c_str());
  if (got != c.ids) {
    const std::string_view near = std::string_view(dec).substr(std::min(ctx_from, dec.size()));
    std::fprintf(stderr, "  ids differ at index %zu (expected %zu ids, got %zu); normalized text near there: \"%s\"\n",
                 diff, c.ids.size(), got.size(), escape(near, 60).c_str());
    const size_t from = diff > 4 ? diff - 4 : 0;
    std::fprintf(stderr, "  expected[%zu..]: %s\n", from, ids_window(tok, c.ids, from, diff + 8).c_str());
    std::fprintf(stderr, "  got     [%zu..]: %s\n", from, ids_window(tok, got, from, diff + 8).c_str());
  }
  if (dec != c.decoded) {
    std::fprintf(stderr, "  decode mismatch: expected \"%s\"\n                   got      \"%s\"\n",
                 escape(c.decoded, 120).c_str(), escape(dec, 120).c_str());
  }
  if (streamed != c.decoded) std::fprintf(stderr, "  stream decode mismatch: \"%s\"\n", escape(streamed, 120).c_str());
}

std::string stream_decode(const qwen38::Tokenizer& tok, const std::vector<int32_t>& ids) {
  qwen38::Utf8StreamDecoder sd;
  std::string out;
  for (int32_t id : ids) out += sd.push(tok.token_bytes(id));
  out += sd.flush();
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------------------------

QWEN38_TEST(tokenizer_hf_cases) {
  const Fixture* fx = fixture();
  REQUIRE(fx != nullptr);
  const std::vector<Case>* cs = cases();
  REQUIRE(cs != nullptr);
  const qwen38::Tokenizer& tok = fx->tok;

  size_t ids_ok = 0, dec_ok = 0, stream_ok = 0, shown = 0, input_bytes = 0;
  std::map<std::string, std::pair<size_t, size_t>> by_category;  // category -> (passed, total)
  const auto t0 = Clock::now();
  for (size_t k = 0; k < cs->size(); ++k) {
    const Case& c = (*cs)[k];
    const std::vector<int32_t> got = tok.encode(c.input, c.allow_added());
    const std::string dec = tok.decode(got);
    const std::string streamed = stream_decode(tok, got);
    const bool a = got == c.ids, b = dec == c.decoded, s = streamed == c.decoded;
    ids_ok += a;
    dec_ok += b;
    stream_ok += s;
    input_bytes += c.input.size();
    auto& cat = by_category[c.allow_added() ? c.category() : "noadded"];
    cat.first += a && b && s;
    ++cat.second;
    if (!(a && b && s) && shown < 10) {
      report_mismatch(tok, k, c, got, dec, streamed);
      ++shown;
    }
  }
  const double dt = seconds_since(t0);
  std::printf("  %zu cases (%.1f MB): ids %zu/%zu, decode %zu/%zu, stream decode %zu/%zu  (%.2f s incl. decode)\n",
              cs->size(), input_bytes / 1e6, ids_ok, cs->size(), dec_ok, cs->size(), stream_ok, cs->size(), dt);
  std::string line = "  by category:";
  for (const auto& [name, pr] : by_category) {
    line += " " + name + " " + std::to_string(pr.first) + "/" + std::to_string(pr.second) + ",";
  }
  line.pop_back();
  std::printf("%s\n", line.c_str());
  CHECK(ids_ok == cs->size());
  CHECK(dec_ok == cs->size());
  CHECK(stream_ok == cs->size());
}

QWEN38_TEST(tokenizer_api) {
  const Fixture* fx = fixture();
  REQUIRE(fx != nullptr);
  const qwen38::Tokenizer& tok = fx->tok;

  CHECK(tok.vocab_size() == 248077);
  CHECK(std::to_string(tok.vocab_size()) == tok.metadata("n_ids"));
  CHECK(!tok.metadata("unidata_version").empty());
  CHECK(tok.metadata("no such key").empty());
  CHECK(tok.find_token("<|endoftext|>") == 248044);
  CHECK(tok.find_token("<|im_start|>") == 248045);
  CHECK(tok.find_token("<|im_end|>") == 248046);
  CHECK(tok.find_token("<think>") == 248068);
  CHECK(tok.find_token("</think>") == 248069);
  CHECK(tok.find_token("<|audio_pad|>") == 248076);
  CHECK(tok.is_special(248044) && tok.is_special(248046) && tok.is_special(248076));
  CHECK(!tok.is_special(248068) && tok.is_added(248068));  // <think>: added, not special
  CHECK(!tok.is_special(0) && !tok.is_added(0) && !tok.is_special(-1) && !tok.is_special(1 << 30));
  CHECK(tok.token_bytes(248068) == "<think>");
  CHECK(tok.token_bytes(-1).empty() && tok.token_bytes(tok.vocab_size()).empty());
  CHECK(tok.find_token("") == -1);
  CHECK(tok.find_token("this is certainly not a single token \xff") == -1);
  CHECK(tok.find_token(" the") >= 0 && tok.token_bytes(tok.find_token(" the")) == " the");
  CHECK(tok.find_token(std::string(1, '\0')) >= 0);
  CHECK(tok.find_token("\xff") >= 0);  // every byte is a token

  // find_token inverts token_bytes for every id
  int32_t bad = 0;
  for (int32_t id = 0; id < tok.vocab_size(); ++id) bad += tok.find_token(tok.token_bytes(id)) != id;
  CHECK_MSG(bad == 0, "%d ids do not round-trip through find_token", bad);

  const std::vector<int32_t> odd = {-5, 248046, 1 << 30, 248320};
  CHECK(tok.decode(odd) == "<|im_end|>");
  CHECK(tok.encode("").empty());
  CHECK(tok.encode("<|im_end|>") == std::vector<int32_t>{248046});
  const std::vector<int32_t> plain = tok.encode("<|im_end|>", false);
  CHECK(plain.size() > 1);
  for (int32_t id : plain) CHECK(!tok.is_added(id));
  CHECK(tok.decode(plain) == "<|im_end|>");

  std::vector<int32_t> out = {7};
  tok.encode("hello", out);
  const std::vector<int32_t> hello = tok.encode("hello");
  CHECK(out.size() == hello.size() + 1 && out[0] == 7 && std::equal(hello.begin(), hello.end(), out.begin() + 1));

  const qwen38::Tokenizer copy = tok;  // shares the tables
  CHECK(copy.encode("copy me <think>") == tok.encode("copy me <think>"));

  CHECK(tok.normalize("e\xCC\x81") == "\xC3\xA9");
  CHECK(tok.normalize("a\xFF") == "a\xEF\xBF\xBD");
  const std::vector<std::string> pieces = tok.pre_tokenize("Hello world's   end\n");
  const std::vector<std::string> want = {"Hello", " world", "'s", "  ", " end", "\n"};
  CHECK(pieces == want);

  const qwen38::Tokenizer empty;
  CHECK(!empty && empty.vocab_size() == 0 && empty.token_bytes(0).empty() && empty.find_token("a") == -1);
  CHECK(!empty.is_special(0) && empty.metadata("n_ids").empty());
  bool threw = false;
  try {
    (void)empty.encode("x");
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);
}

QWEN38_TEST(tokenizer_stream_decoder) {
  const std::string fffd = "\xEF\xBF\xBD";
  {
    qwen38::Utf8StreamDecoder d;
    CHECK(d.push("abc") == "abc");
    CHECK(d.push("\xF0") == "" && d.push("\x9F") == "" && d.push("\x98") == "" && d.pending_bytes() == 3);
    CHECK(d.push("\x80") == "\xF0\x9F\x98\x80");
    CHECK(d.flush().empty());
  }
  {
    qwen38::Utf8StreamDecoder d;
    CHECK(d.push("x\xE2\x82").size() == 1 && d.flush() == fffd && d.flush().empty());
  }
  struct Expect {
    std::string in, out;
  };
  const std::vector<Expect> expect = {
      {"a\xFF" "b", "a" + fffd + "b"},
      {"\xED\xA0\x80", fffd + fffd + fffd},                  // surrogate: every byte is a maximal subpart
      {"\xF0\x9F" "A", fffd + "A"},                          // valid prefix, then a non-continuation
      {"\xE0\x80", fffd + fffd},                             // E0 needs A0..BF
      {"\xC0\xAF", fffd + fffd},                             // overlong lead bytes are never valid
      {"\xF4\x90\x80\x80", fffd + fffd + fffd + fffd},       // > U+10FFFF
      {"\xF5\x80", fffd + fffd},
      {"\xE4\xB8\xAD\xE6\x96", "\xE4\xB8\xAD" + fffd},       // truncated at the end: one U+FFFD
      {"\x80\xBF", fffd + fffd},
      {"\xC3\xA9\xCC\x81", "\xC3\xA9\xCC\x81"},
  };
  for (const Expect& e : expect) {
    qwen38::Utf8StreamDecoder d;
    std::string got = d.push(e.in);
    got += d.flush();
    CHECK_MSG(got == e.out, "\"%s\" -> \"%s\"", escape(e.in).c_str(), escape(got).c_str());
    // byte by byte gives the same result
    qwen38::Utf8StreamDecoder d2;
    std::string got2;
    for (char ch : e.in) got2 += d2.push(std::string_view(&ch, 1));
    got2 += d2.flush();
    CHECK(got2 == e.out);
  }
  // Random streams: any chunking gives the same text as one push.
  std::mt19937 rng(12345);
  const char* pieces[] = {"a", "\xC3\xA9", "\xE4\xB8\xAD", "\xF0\x9F\x98\x80", "\xFF", "\x80", "\xE2\x82", "\xF0\x9F",
                          "\xED\xA0\x80", "\xC0", " ", "\n"};
  int mismatches = 0;
  for (int iter = 0; iter < 2000; ++iter) {
    std::string s;
    const int n = static_cast<int>(rng() % 20);
    for (int i = 0; i < n; ++i) s += pieces[rng() % (sizeof pieces / sizeof pieces[0])];
    qwen38::Utf8StreamDecoder whole;
    const std::string ref = whole.push(s) + whole.flush();
    qwen38::Utf8StreamDecoder chunked;
    std::string got;
    for (size_t i = 0; i < s.size();) {
      const size_t len = std::min<size_t>(s.size() - i, 1 + rng() % 5);
      got += chunked.push(std::string_view(s).substr(i, len));
      i += len;
    }
    got += chunked.flush();
    mismatches += got != ref;
  }
  CHECK_MSG(mismatches == 0, "%d chunked streams differ from one-shot decoding", mismatches);

  // Independent check against CPython's bytes.decode("utf-8", "replace"): for the invalid-UTF-8
  // cases encoded without added tokens, the expected text is NFC(replace-decoded input).
  const Fixture* fx = fixture();
  const std::vector<Case>* cs = cases();
  REQUIRE(fx != nullptr && cs != nullptr);
  size_t checked = 0, wrong = 0;
  for (const Case& c : *cs) {
    if (c.allow_added() || c.label.rfind("noadded/invalid/", 0) != 0) continue;
    qwen38::Utf8StreamDecoder d;
    std::string text;
    for (size_t i = 0; i < c.input.size(); i += 3) text += d.push(std::string_view(c.input).substr(i, 3));
    text += d.flush();
    ++checked;
    if (fx->tok.normalize(text) != c.decoded) {
      if (++wrong <= 3) std::fprintf(stderr, "  replace-decoding differs for [%s]\n", c.label.c_str());
    }
  }
  std::printf("  %zu invalid-UTF-8 streams checked against CPython's replace decoding\n", checked);
  CHECK(checked > 100);
  CHECK(wrong == 0);
}

QWEN38_TEST(tokenizer_malformed_blob) {
  const Fixture* fx = fixture();
  REQUIRE(fx != nullptr);
  const std::string& blob = fx->blob;

  auto rejects = [](const std::string& data) {
    try {
      (void)qwen38::Tokenizer::from_blob(data.data(), data.size());
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };
  auto rd32 = [&](const std::string& d, size_t off) {
    uint32_t v;
    std::memcpy(&v, d.data() + off, 4);
    return v;
  };
  auto wr32 = [](std::string& d, size_t off, uint32_t v) { std::memcpy(d.data() + off, &v, 4); };

  bool threw = false;
  try {
    (void)qwen38::Tokenizer::from_blob(nullptr, 0);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
  CHECK(rejects(""));
  CHECK(rejects("QW38TOK"));
  bool missing_file = false;
  try {
    (void)qwen38::Tokenizer::from_file("/nonexistent/qwen38/tokenizer.bin");
  } catch (const std::runtime_error&) {
    missing_file = true;
  }
  CHECK(missing_file);

  // Truncations anywhere before the end of the last section.
  const uint32_t n_sections = rd32(blob, 12);
  size_t data_end = 0;
  std::map<std::string, std::pair<size_t, size_t>> sec;
  for (uint32_t i = 0; i < n_sections; ++i) {
    const size_t e = 16 + 24 * size_t{i};
    uint64_t off, len;
    std::memcpy(&off, blob.data() + e + 8, 8);
    std::memcpy(&len, blob.data() + e + 16, 8);
    sec[blob.substr(e, 4)] = {static_cast<size_t>(off), static_cast<size_t>(len)};
    data_end = std::max<size_t>(data_end, static_cast<size_t>(off + len));
  }
  CHECK(sec.size() == 10);
  int kept = 0;
  for (size_t len = 0; len < data_end; len += 1 + data_end / 61) kept += !rejects(blob.substr(0, len));
  for (size_t len : {size_t{8}, size_t{15}, size_t{16}, 16 + 24 * size_t{n_sections} - 1, data_end - 1}) {
    kept += !rejects(blob.substr(0, len));
  }
  CHECK_MSG(kept == 0, "%d truncated blobs were accepted", kept);

  // Targeted corruptions.
  auto corrupt = [&](auto&& edit) {
    std::string d = blob;
    edit(d);
    return rejects(d);
  };
  CHECK(corrupt([](std::string& d) { d[0] = 'X'; }));
  CHECK(corrupt([&](std::string& d) { wr32(d, 8, 2); }));                     // version
  CHECK(corrupt([&](std::string& d) { wr32(d, 12, 0x7FFFFFFF); }));           // section count
  CHECK(corrupt([&](std::string& d) { wr32(d, 12, 0); }));                    // no sections
  CHECK(corrupt([](std::string& d) { d[16] = 'X'; }));                        // renamed first section
  CHECK(corrupt([&](std::string& d) { wr32(d, 16 + 8 + 4, 0x10); }));         // offset beyond the blob
  CHECK(corrupt([&](std::string& d) { std::memcpy(d.data() + 16 + 24, d.data() + 16, 4); }));  // duplicate tag
  CHECK(corrupt([&](std::string& d) { d[sec["PATN"].first + 2] ^= 1; }));     // unsupported pattern
  CHECK(corrupt([&](std::string& d) { wr32(d, sec["VOCB"].first, 0); }));     // no ids
  CHECK(corrupt([&](std::string& d) { wr32(d, sec["VOCB"].first + 4, 0x7FFFFFF0); }));  // base > ids
  CHECK(corrupt([&](std::string& d) { wr32(d, sec["VOCB"].first + 8 + 4 * 1000, 0xFFFFFF); }));  // offsets
  CHECK(corrupt([&](std::string& d) { wr32(d, sec["MERG"].first + 4, 300000); }));  // merge id out of range
  CHECK(corrupt([&](std::string& d) {  // merge result that is not left + right
    wr32(d, sec["MERG"].first + 4 + 8, rd32(d, sec["MERG"].first + 4 + 12 + 8));
  }));
  CHECK(corrupt([&](std::string& d) { wr32(d, sec["MERG"].first, 0x40000000); }));  // count
  CHECK(corrupt([&](std::string& d) { wr32(d, sec["ADDT"].first + 4, 248077); }));  // added id
  CHECK(corrupt([&](std::string& d) { wr32(d, sec["ADDT"].first + 4 + 8, 1u << 30); }));  // added offset
  CHECK(corrupt([&](std::string& d) { wr32(d, sec["UCLS"].first + 4, 0x110000); }));  // range
  CHECK(corrupt([&](std::string& d) { wr32(d, sec["UCLS"].first + 4 + 8, 0); }));  // empty class
  CHECK(corrupt([&](std::string& d) { wr32(d, sec["UCCC"].first + 4 + 8, 300); }));  // ccc > 254
  CHECK(corrupt([&](std::string& d) { wr32(d, sec["UDEC"].first + 4, 0xFFFFFFFF); }));  // cp
  CHECK(corrupt([&](std::string& d) { wr32(d, sec["UDEC"].first + 4 + 4, 1u << 30); }));  // pool offset
  CHECK(corrupt([&](std::string& d) { wr32(d, sec["UCMP"].first + 4, 0x7FFFFFFF); }));  // cp
  CHECK(corrupt([&](std::string& d) { d.replace(sec["META"].first, 10, "normalizer"); }));  // line without '='

  // Random garbage never crashes; random damage to the header / directory either loads or throws.
  std::mt19937 rng(777);
  int crashes_avoided = 0;
  for (int i = 0; i < 300; ++i) {
    std::string g(rng() % 400, '\0');
    for (auto& ch : g) ch = static_cast<char>(rng());
    if (i % 2 == 0 && g.size() >= 16) {
      std::memcpy(g.data(), blob.data(), 12);  // valid magic + version, garbage directory
    }
    crashes_avoided += rejects(g);
  }
  CHECK(crashes_avoided == 300);
  const size_t header_end = 16 + 24 * size_t{n_sections};
  for (int i = 0; i < 40; ++i) {
    std::string d = blob;
    d[rng() % header_end] ^= static_cast<char>(1u << (rng() % 8));
    try {
      const qwen38::Tokenizer t = qwen38::Tokenizer::from_blob(d.data(), d.size());
      CHECK(t.encode("still works?") == fx->tok.encode("still works?"));
    } catch (const std::runtime_error&) {
    }
  }
}

QWEN38_TEST(tokenizer_long_inputs) {
  const Fixture* fx = fixture();
  REQUIRE(fx != nullptr);
  const qwen38::Tokenizer& tok = fx->tok;
  constexpr size_t kMB = 1 << 20;
  std::mt19937 rng(4242);
  auto repeat = [](std::string_view unit, size_t bytes) {
    std::string s;
    s.reserve(bytes + unit.size());
    while (s.size() < bytes) s += unit;
    return s;
  };
  auto random_of = [&](const std::vector<std::string>& units, size_t bytes) {
    std::string s;
    while (s.size() < bytes) s += units[rng() % units.size()];
    return s;
  };
  std::vector<std::string> lower, cjk, marks;
  for (char c = 'a'; c <= 'z'; ++c) lower.emplace_back(1, c);
  for (const char* c : {"\xE7\x9A\x84", "\xE4\xB8\x80", "\xE6\x98\xAF", "\xE4\xB8\x8D", "\xE4\xBA\x86",
                        "\xE4\xBA\xBA", "\xE6\x88\x91", "\xE5\x9C\xA8", "\xE6\x9C\x89", "\xE4\xBB\x96",
                        "\xE8\xBF\x99", "\xE4\xB8\xAD"}) {
    cjk.emplace_back(c);
  }
  for (const char* c : {"\xCC\x81", "\xCC\x96", "\xCC\xA7", "\xCD\x85", "\xCC\x88", "\xD6\xB8"}) marks.emplace_back(c);
  std::string bytes(kMB, '\0');
  for (auto& ch : bytes) ch = static_cast<char>(rng());

  const std::vector<std::pair<std::string, std::string>> inputs = {
      {"random letters (one pre-token)", random_of(lower, kMB)},
      {"one repeated letter", std::string(kMB, 'a')},
      {"CJK without spaces", random_of(cjk, kMB)},
      {"spaces", std::string(kMB, ' ') + "x"},
      {"newlines", std::string(kMB, '\n')},
      {"punctuation run", std::string(kMB, '=')},
      {"letter + combining acute", repeat("a\xCC\x81", kMB)},
      {"combining marks only (mixed ccc)", random_of(marks, kMB)},
      {"random bytes (invalid UTF-8)", bytes},
      {"digits", random_of({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}, kMB)},
      {"apostrophes and letters", random_of({"'", "s", "ll", "re", "S", "x"}, kMB)},
      {"many added tokens", repeat("<|im_start|><think>", kMB)},
  };
  for (const auto& [name, text] : inputs) {
    const auto t0 = Clock::now();
    const std::vector<int32_t> ids = tok.encode(text);
    const double dt = seconds_since(t0);
    std::printf("  %-34s %7.2f MB -> %8zu ids in %6.1f ms\n", name.c_str(), text.size() / 1e6, ids.size(), 1e3 * dt);
    CHECK_MSG(dt < 5.0, "%s took %.2f s", name.c_str(), dt);
    CHECK(!ids.empty());
    // No added tokens in these inputs except the last one, whose tokens decode to themselves, so
    // the decoded text is the (sanitized) NFC of the input.
    CHECK_MSG(tok.decode(ids) == tok.normalize(text), "%s does not round-trip", name.c_str());
  }
}

QWEN38_TEST(tokenizer_throughput) {
  const Fixture* fx = fixture();
  const std::vector<Case>* cs = cases();
  REQUIRE(fx != nullptr && cs != nullptr);
  const qwen38::Tokenizer& tok = fx->tok;

  auto time_encode = [&](const std::string& text, size_t* n_ids) {
    double best = 1e9;
    for (int rep = 0; rep < 3; ++rep) {
      const auto t0 = Clock::now();
      *n_ids = tok.encode(text).size();
      best = std::min(best, seconds_since(t0));
    }
    return best;
  };
  (void)tok.encode(std::string_view((*cs)[0].input));  // warm up

  // 1. ~10 MB mixed document: the natural-text cases (prose, code, languages, chat transcripts,
  //    emoji, numbers) repeated. The repetition makes the per-call BPE cache hit more often than on
  //    fresh natural text, so this is an upper bound (fresh stdlib code + C headers + English +
  //    multilingual text measured about 30% lower on the same machine).
  const std::vector<std::string> keep = {"prose", "code", "lang", "long", "added", "emoji", "number", "contraction"};
  std::string mixed;
  while (mixed.size() < 10'000'000) {
    for (const Case& c : *cs) {
      if (c.allow_added() && std::find(keep.begin(), keep.end(), c.category()) != keep.end()) {
        mixed += c.input;
        mixed += '\n';
      }
    }
  }
  size_t n_ids = 0;
  double dt = time_encode(mixed, &n_ids);
  std::printf("  %.1f MB mixed document (natural-text cases, repeated): %6.1f MB/s  (%.2f bytes/token)\n",
              mixed.size() / 1e6, mixed.size() / 1e6 / dt,
              static_cast<double>(mixed.size()) / static_cast<double>(n_ids));

  // 2. Every case input once as one document: adversarial (code point sweep, random scripts,
  //    invalid UTF-8, huge single pre-tokens), little reuse.
  std::string all;
  for (const Case& c : *cs) all += c.input;
  dt = time_encode(all, &n_ids);
  std::printf("  %.1f MB adversarial document (all case inputs):        %6.1f MB/s\n", all.size() / 1e6,
              all.size() / 1e6 / dt);

  // 3. Many short calls (chat-sized messages): no BPE cache below 256 bytes.
  size_t bytes = 0, count = 0;
  const auto t0 = Clock::now();
  for (int rep = 0; rep < 3; ++rep) {
    for (const Case& c : *cs) {
      if (c.input.size() > 200 || !c.allow_added()) continue;
      count += tok.encode(c.input).size();
      bytes += c.input.size();
    }
  }
  dt = seconds_since(t0);
  std::printf("  short inputs (<= 200 B, one call each):             %6.1f MB/s\n", bytes / 1e6 / dt);
  CHECK(n_ids > 0 && count > 0);
}
