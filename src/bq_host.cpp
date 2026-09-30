#include "qwen38/bq.h"

#include "qwen38/common.h"

namespace qwen38 {

BQView BQView::from_file(const ModelFile& file, const std::string& name) {
  const TensorInfo& t = file.tensor(name);
  QWEN38_CHECK(is_bq(t.dtype), name, " is not a BQ tensor");
  BQView v;
  v.bits = bq_bits(t.dtype);
  v.n = t.shape[0];
  v.k = t.shape[1];
  v.rows = t.rows;
  v.lo = file.data(t.lo_offset, t.lo_bytes).data();
  v.hi = t.hi_bytes ? file.data(t.hi_offset, t.hi_bytes).data() : nullptr;
  v.scales = reinterpret_cast<const uint16_t*>(file.data(t.sc_offset, t.sc_bytes).data());
  return v;
}

namespace {

inline uint32_t load_u32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;  // little-endian host assumed (x86-64, aarch64)
}

struct Position {
  int64_t slab;  // slab index in plane order
  int lane;
  int pair;
  int elem;
};

// (row n, col k) -> slab / lane / pair / element, per FORMAT.md 2.2
inline Position locate(int64_t n, int64_t k, int64_t K) {
  const int64_t nb = n / 64, kb = k / 16;
  const int n_local = int(n % 64), k_local = int(k % 16);
  const int j = n_local / 8, g = n_local % 8;
  const int h = k_local / 8, t = (k_local % 8) / 2, e = k_local % 2;
  Position p;
  p.slab = nb * (K / 16) + kb;
  p.lane = 4 * g + t;
  p.pair = 2 * j + h;
  p.elem = e;
  return p;
}

inline unsigned nibble_at(const uint8_t* plane, const Position& p) {
  const uint8_t* lane_ptr = plane + p.slab * 512 + p.lane * 16;
  const uint32_t word = load_u32(lane_ptr + 4 * (p.pair >> 2));
  const int pos = p.pair & 3;
  return (word >> (p.elem ? 16 + 4 * pos : 4 * pos)) & 15u;
}

}  // namespace

int BQView::code(int64_t row, int64_t col) const {
  const Position p = locate(row, col, k);
  unsigned u = nibble_at(lo, p);
  if (bits == 5) {
    const uint32_t word = load_u32(hi + p.slab * 128 + p.lane * 4);
    u |= ((word >> (p.elem ? 16 + p.pair : p.pair)) & 1u) << 4;
  } else if (bits == 6) {
    const uint32_t word = load_u32(hi + p.slab * 256 + p.lane * 8 + 4 * (p.pair >> 3));
    const int r = p.pair & 7;
    u |= ((word >> (p.elem ? 16 + 2 * r : 2 * r)) & 3u) << 4;
  } else if (bits == 8) {
    u |= nibble_at(hi, p) << 4;
  }
  return int(u) - (1 << (bits - 1));
}

uint16_t BQView::scale_bits(int64_t row, int64_t group) const {
  const int64_t nb = row / 64;
  const int n_local = int(row % 64);
  const int j = n_local / 8, g = n_local % 8;
  const int64_t block = nb * (k / 64) + group;
  return scales[block * 64 + g * 8 + j];
}

float BQView::scale(int64_t row, int64_t group) const { return bf16_to_f32(scale_bits(row, group)); }

float BQView::weight(int64_t row, int64_t col) const {
  return round_bf16(float(code(row, col)) * scale(row, col / 64));
}

void BQView::dequant_row(int64_t row, float* out) const {
  for (int64_t col = 0; col < k; ++col) out[col] = weight(row, col);
}

std::vector<float> BQView::dequant_all() const {
  std::vector<float> w(size_t(n * k));
  for (int64_t r = 0; r < n; ++r) dequant_row(r, w.data() + r * k);
  return w;
}

}  // namespace qwen38
