// Minimal test harness (no external dependency).
//
//   QWEN38_TEST(name) { CHECK(x == y); REQUIRE(ptr != nullptr); }
//
// Each test executable links tests/test_main.cpp, which runs every registered test (or those
// whose name contains argv[1]) and returns non-zero if any CHECK failed.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38::test {

struct Case {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<Case>& registry() {
  static std::vector<Case> cases;
  return cases;
}

inline int& failures() {
  static int count = 0;
  return count;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

struct RequireFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace qwen38::test

#define QWEN38_TEST(name)                                                 \
  static void name();                                                   \
  static ::qwen38::test::Registrar name##_registrar(#name, name);         \
  static void name()

#define CHECK(cond)                                                                    \
  do {                                                                                 \
    if (!(cond)) {                                                                     \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);    \
      ++::qwen38::test::failures();                                                      \
    }                                                                                  \
  } while (0)

#define CHECK_MSG(cond, ...)                                                           \
  do {                                                                                 \
    if (!(cond)) {                                                                     \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s: ", __FILE__, __LINE__, #cond);    \
      std::fprintf(stderr, __VA_ARGS__);                                               \
      std::fprintf(stderr, "\n");                                                      \
      ++::qwen38::test::failures();                                                      \
    }                                                                                  \
  } while (0)

#define REQUIRE(cond)                                                                  \
  do {                                                                                 \
    if (!(cond)) {                                                                     \
      std::fprintf(stderr, "%s:%d: REQUIRE failed: %s\n", __FILE__, __LINE__, #cond);  \
      ++::qwen38::test::failures();                                                      \
      throw ::qwen38::test::RequireFailure(#cond);                                       \
    }                                                                                  \
  } while (0)

// |a - b| <= atol + rtol * |b|
#define CHECK_NEAR(a, b, atol, rtol)                                                   \
  do {                                                                                 \
    const double qwen38_a_ = static_cast<double>(a);                                     \
    const double qwen38_b_ = static_cast<double>(b);                                     \
    if (!(std::fabs(qwen38_a_ - qwen38_b_) <= (atol) + (rtol) * std::fabs(qwen38_b_))) {     \
      std::fprintf(stderr, "%s:%d: CHECK_NEAR failed: %s = %.8g vs %s = %.8g\n",       \
                   __FILE__, __LINE__, #a, qwen38_a_, #b, qwen38_b_);                      \
      ++::qwen38::test::failures();                                                      \
    }                                                                                  \
  } while (0)
