// Runs the device dequantization bit logic (src/kernels/bq_device.cuh, compiled for the host)
// over every slab and lane of real packed tensors and checks each decoded code and scale
// against the independent spec decoder (BQView) at the mma fragment position it must land in.
//
// Env: QWEN38_TEST_MODELS = ':'-separated .q38 files (default: the tiny test models).
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "qwen38/bq.h"
#include "qwen38/common.h"
#include "qwen38/model_file.h"
#include "kernels/bq_device.cuh"
#include "test_util.h"

namespace {

std::vector<std::string> model_paths() {
  const char* env = std::getenv("QWEN38_TEST_MODELS");
  std::string s = env ? env : QWEN38_TESTDATA_DIR "/tiny/tiny-default.q38:" QWEN38_TESTDATA_DIR "/tiny/tiny-quality.q38";
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ':'))
    if (!item.empty()) out.push_back(item);
  return out;
}

uint32_t load_u32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

// host emulation of the device's bf16 arithmetic on the magic values (exact integers)
int magic_to_u(uint16_t bits) { return int(qwen38::bf16_to_f32(bits)) - 128; }

template <int BITS>
int check_tensor(const qwen38::BQView& w, const char* name) {
  using namespace qwen38::bqdev;
  int bad = 0;
  const int64_t slabs_k = w.k / 16, blocks_n = w.n / 64;
  for (int64_t nb = 0; nb < blocks_n; ++nb) {
    for (int64_t kb = 0; kb < slabs_k; ++kb) {
      const int64_t slab = nb * slabs_k + kb;
      for (int lane = 0; lane < 32; ++lane) {
        SlabLane s{};
        for (int i = 0; i < 4; ++i) s.lo[i] = load_u32(w.lo + slab * 512 + lane * 16 + 4 * i);
        if (BITS == 5) s.hi[0] = load_u32(w.hi + slab * 128 + lane * 4);
        if (BITS == 6)
          for (int i = 0; i < 2; ++i) s.hi[i] = load_u32(w.hi + slab * 256 + lane * 8 + 4 * i);
        if (BITS == 8)
          for (int i = 0; i < 4; ++i) s.hi[i] = load_u32(w.hi + slab * 512 + lane * 16 + 4 * i);
        const int g = lane >> 2, t = lane & 3;
        for (int q = 0; q < 16; ++q) {
          const int j = q >> 1, h = q & 1;
          const int64_t n = nb * 64 + 8 * j + g;
          const int64_t k0 = kb * 16 + 2 * t + 8 * h;
          int c[2];
          if constexpr (BITS == 8) {
            const uint32_t lo = pair_magic<4>(s, q), hi = pair_hi_nibble(s, q);
            for (int e = 0; e < 2; ++e) {
              const int ulo = magic_to_u(uint16_t(lo >> (16 * e))), uhi = magic_to_u(uint16_t(hi >> (16 * e)));
              c[e] = 16 * (uhi - 8) + ulo;
            }
          } else {
            const uint32_t v = pair_magic<BITS>(s, q);
            for (int e = 0; e < 2; ++e) c[e] = magic_to_u(uint16_t(v >> (16 * e))) - (1 << (BITS - 1));
          }
          for (int e = 0; e < 2; ++e) {
            const int expect = w.code(n, k0 + e);
            if (c[e] != expect && bad++ < 5)
              std::fprintf(stderr, "%s: bits=%d nb=%lld kb=%lld lane=%d q=%d e=%d: got %d want %d\n", name, BITS,
                           (long long)nb, (long long)kb, lane, q, e, c[e], expect);
          }
        }
        // scales: lane reads 8 bf16 at block + 16 * g, entry j = column 8j + g
        for (int64_t grp = 0; grp < w.k / 64; ++grp) {
          const uint16_t* block = w.scales + (nb * (w.k / 64) + grp) * 64;
          for (int j = 0; j < 8; ++j) {
            const uint16_t got = block[g * 8 + j];
            const uint16_t want = w.scale_bits(nb * 64 + 8 * j + g, grp);
            if (got != want && bad++ < 5) std::fprintf(stderr, "%s: scale mismatch\n", name);
          }
        }
      }
    }
  }
  return bad;
}

}  // namespace

QWEN38_TEST(bq_device_bit_logic_matches_spec_decoder) {
  int checked = 0;
  bool seen[9] = {};
  for (const auto& path : model_paths()) {
    auto file = qwen38::ModelFile::open(path);
    for (const auto& [name, info] : file->tensors()) {
      if (!qwen38::is_bq(info.dtype)) continue;
      const auto w = qwen38::BQView::from_file(*file, name);
      int bad = 0;
      switch (w.bits) {
        case 4: bad = check_tensor<4>(w, name.c_str()); break;
        case 5: bad = check_tensor<5>(w, name.c_str()); break;
        case 6: bad = check_tensor<6>(w, name.c_str()); break;
        case 8: bad = check_tensor<8>(w, name.c_str()); break;
      }
      CHECK_MSG(bad == 0, "%s: %d mismatches", name.c_str(), bad);
      seen[w.bits] = true;
      ++checked;
    }
  }
  CHECK(checked > 0);
  CHECK_MSG(seen[4] && seen[5] && seen[6] && seen[8], "test models must cover 4/5/6/8-bit tensors");
}
