#include "qwen38/tuning.h"

#include <algorithm>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>

#include "qwen38/common.h"
#include "kernels/kernels.h"

namespace qwen38 {

namespace {

std::vector<TuningEntry>& table() {
  static std::vector<TuningEntry> t;
  return t;
}

int lookup_trampoline(int bits, int n, int k, int m) { return lookup_tuning(bits, n, k, m); }

}  // namespace

int lookup_tuning(int bits, int n, int k, int m) {
  const TuningEntry* best = nullptr;
  for (const auto& e : table()) {
    if (e.bits != bits || e.n != n || e.k != k || e.m_max < m) continue;
    if (!best || e.m_max < best->m_max) best = &e;
  }
  return best ? best->config : kernels::kGemmConfigAuto;
}

void install_tuning(std::vector<TuningEntry> entries) {
  table() = std::move(entries);
  kernels::set_qgemm_tuning(table().empty() ? nullptr : &lookup_trampoline);
}

void load_tuning(const std::string& path) {
  std::ifstream in(path);
  QWEN38_CHECK(in.good(), "cannot open tuning file ", path);
  const auto j = nlohmann::json::parse(in);
  std::vector<TuningEntry> entries;
  for (const auto& e : j.at("entries")) {
    TuningEntry t;
    t.bits = e.at("bits");
    t.n = e.at("n");
    t.k = e.at("k");
    t.m_max = e.at("m_max");
    t.config = e.at("config");
    t.us = e.value("us", 0.0);
    QWEN38_CHECK(t.config >= 0 && t.config < kernels::qgemm_num_configs(), "tuning file: bad config id");
    entries.push_back(t);
  }
  install_tuning(std::move(entries));
}

void save_tuning(const std::string& path, const std::vector<TuningEntry>& entries, const std::string& device) {
  nlohmann::json j;
  j["device"] = device;
  j["configs"] = nlohmann::json::array();
  for (int c = 0; c < kernels::qgemm_num_configs(); ++c) j["configs"].push_back(kernels::qgemm_config_name(c));
  j["entries"] = nlohmann::json::array();
  for (const auto& e : entries) {
    j["entries"].push_back({{"bits", e.bits}, {"n", e.n}, {"k", e.k}, {"m_max", e.m_max}, {"config", e.config},
                            {"name", kernels::qgemm_config_name(e.config)}, {"us", e.us}});
  }
  std::ofstream out(path);
  out << j.dump(1) << "\n";
  QWEN38_CHECK(out.good(), "failed writing ", path);
}

}  // namespace qwen38
