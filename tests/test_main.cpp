#include <chrono>
#include <cstring>

#include "test_util.h"

int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int ran = 0;
  for (const auto& test_case : qwen38::test::registry()) {
    if (filter && !std::strstr(test_case.name, filter)) continue;
    const int before = qwen38::test::failures();
    const auto start = std::chrono::steady_clock::now();
    try {
      test_case.fn();
    } catch (const qwen38::test::RequireFailure&) {
      // already counted
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s: exception: %s\n", test_case.name, e.what());
      ++qwen38::test::failures();
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::printf("[%s] %s (%.1f ms)\n", qwen38::test::failures() == before ? " OK " : "FAIL",
                test_case.name, ms);
    ++ran;
  }
  std::printf("%d test(s), %d failure(s)\n", ran, qwen38::test::failures());
  return qwen38::test::failures() == 0 ? 0 : 1;
}
