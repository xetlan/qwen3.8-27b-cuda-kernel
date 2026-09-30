// GEMM tile-configuration tuning table.
//
// `qwen38-bench --tune` times every compiled qgemm configuration for each (bits, N, K) of the model
// at a ladder of M values and writes the winners to JSON. The engine loads the table at startup
// (--tuning FILE) and the qgemm dispatcher consults it before its built-in M heuristic.
#pragma once

#include <string>
#include <vector>

namespace qwen38 {

struct TuningEntry {
  int bits = 0, n = 0, k = 0;
  int m_max = 0;   // entry applies to m <= m_max (smallest matching m_max wins)
  int config = -1;
  double us = 0;   // measured time of the winner
};

void load_tuning(const std::string& path);                 // installs the lookup into the kernels
void save_tuning(const std::string& path, const std::vector<TuningEntry>& entries, const std::string& device);
void install_tuning(std::vector<TuningEntry> entries);
int lookup_tuning(int bits, int n, int k, int m);

}  // namespace qwen38
