// Shared host-side helpers: error handling, bf16/fp16 bit conversions, small utilities.
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace qwen38 {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& message);

#define QWEN38_CHECK(cond, ...)                                                          \
  do {                                                                                 \
    if (!(cond)) ::qwen38::fail(::qwen38::format_check_failure(#cond, __FILE__, __LINE__,  \
                                                           ::qwen38::strcat_args(__VA_ARGS__))); \
  } while (0)

std::string format_check_failure(const char* expr, const char* file, int line, const std::string& msg);

template <typename... Ts>
std::string strcat_args(const Ts&... parts) {
  std::ostringstream os;
  (os << ... << parts);
  return os.str();
}

// ---- 16-bit float formats (bit-exact, round-to-nearest-even) --------------------------------

inline float bf16_to_f32(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits) << 16); }

inline uint16_t f32_to_bf16(float f) {
  const uint32_t u = std::bit_cast<uint32_t>(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40u);  // quiet NaN
  const uint32_t rounding = 0x7fffu + ((u >> 16) & 1u);
  return uint16_t((u + rounding) >> 16);
}

inline float round_bf16(float f) { return bf16_to_f32(f32_to_bf16(f)); }

inline float f16_to_f32(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  const uint32_t man = h & 0x3ffu;
  if (exp == 0) {
    if (man == 0) return std::bit_cast<float>(sign);
    // subnormal: value = man * 2^-24
    const float v = std::ldexp(float(man), -24);
    return sign ? -v : v;
  }
  if (exp == 31) return std::bit_cast<float>(sign | 0x7f800000u | (man << 13));
  return std::bit_cast<float>(sign | ((exp + 112u) << 23) | (man << 13));
}

inline uint16_t f32_to_f16(float f) {
  const uint32_t u = std::bit_cast<uint32_t>(f);
  const uint32_t sign = (u >> 16) & 0x8000u;
  const uint32_t abs = u & 0x7fffffffu;
  if (abs > 0x7f800000u) return uint16_t(sign | 0x7e00u);  // NaN
  if (abs >= 0x477ff000u) return uint16_t(sign | 0x7c00u);  // overflow -> inf (>= 65520)
  if (abs < 0x38800000u) {                                  // below 2^-14: subnormal or zero
    // value / 2^-24, rounded to nearest even
    const float scaled = std::bit_cast<float>(abs) * 16777216.0f;  // exact (power of two)
    const float r = std::nearbyint(scaled);                          // default mode is RNE
    return uint16_t(sign | uint32_t(r));
  }
  const uint32_t rounding = 0xfffu + ((abs >> 13) & 1u);
  const uint32_t v = (abs + rounding) >> 13;  // exponent re-bias below
  return uint16_t(sign | (v - (112u << 10)));
}

inline float round_f16(float f) { return f16_to_f32(f32_to_f16(f)); }

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
inline int64_t align_up(int64_t a, int64_t b) { return ceil_div(a, b) * b; }

}  // namespace qwen38
