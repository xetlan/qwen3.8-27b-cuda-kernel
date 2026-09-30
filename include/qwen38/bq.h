// Host-side view and decoder of BQ quantized weights (docs/FORMAT.md section 2).
//
// This is an independent implementation of the layout spec: the Python packer
// (tools/bqpack.py) writes it, this reads it, and tests check that both agree.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "qwen38/model_file.h"

namespace qwen38 {

struct BQView {
  int bits = 4;
  int64_t n = 0;     // padded rows (multiple of 128)
  int64_t k = 0;     // columns (multiple of 64)
  int64_t rows = 0;  // logical rows
  const uint8_t* lo = nullptr;
  const uint8_t* hi = nullptr;
  const uint16_t* scales = nullptr;  // bf16 bits

  static BQView from_file(const ModelFile& file, const std::string& name);

  int code(int64_t row, int64_t col) const;          // signed code
  uint16_t scale_bits(int64_t row, int64_t group) const;
  float scale(int64_t row, int64_t group) const;
  // bf16(code * scale), exactly what the kernels feed to the tensor cores
  float weight(int64_t row, int64_t col) const;
  // one full row of dequantized weights (length k)
  void dequant_row(int64_t row, float* out) const;
  std::vector<float> dequant_all() const;  // [n * k], tests / tiny models only
};

}  // namespace qwen38
