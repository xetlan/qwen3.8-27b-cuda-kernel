// GPU kernel tests: every kernel against the CPU reference ops (src/ref/ops.h), which are
// themselves verified against Hugging Face transformers (tests/host/test_ref_vs_hf.py).
//
// Weights come from the tiny test models (real packed BQ tensors, 4/5/6/8-bit):
//   QWEN38_TEST_MODELS=testdata/tiny/tiny-default.q38:testdata/tiny/tiny-quality.q38 (default)
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "qwen38/bq.h"
#include "qwen38/common.h"
#include "qwen38/device.h"
#include "qwen38/kv_cache.h"
#include "qwen38/model_file.h"
#include "qwen38/weights.h"
#include "kernels/kernels.h"
#include "ref/ops.h"
#include "test_util.h"

using namespace qwen38;

namespace {

std::vector<std::string> model_paths() {
  const char* env = std::getenv("QWEN38_TEST_MODELS");
  std::string s = env ? env
                      : QWEN38_TESTDATA_DIR "/tiny/tiny-default.q38:" QWEN38_TESTDATA_DIR "/tiny/tiny-quality.q38";
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ':'))
    if (!item.empty()) out.push_back(item);
  return out;
}

template <typename T>
DeviceBuffer upload(const std::vector<T>& v) {
  DeviceBuffer b(v.size() * sizeof(T));
  QWEN38_CUDA(cudaMemcpy(b.get(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return b;
}

template <typename T>
std::vector<T> download(const DeviceBuffer& b, size_t count) {
  std::vector<T> v(count);
  QWEN38_CUDA(cudaMemcpy(v.data(), b.get(), count * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}

std::vector<uint16_t> to_bf16_bits(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = f32_to_bf16(v[i]);
  return o;
}

std::vector<float> from_bf16_bits(const std::vector<uint16_t>& v) {
  std::vector<float> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = bf16_to_f32(v[i]);
  return o;
}

std::vector<float> random_bf16_values(size_t n, std::mt19937& rng, float scale = 1.0f) {
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = round_bf16(d(rng));
  return v;
}

struct ErrorStats {
  double max_abs = 0, max_rel_to_scale = 0;
  size_t bad = 0, count = 0;
};

// |got - want| <= atol + rtol * scale_i for every element
ErrorStats compare(const std::vector<float>& got, const std::vector<float>& want, const std::vector<float>& scale,
                   double atol, double rtol) {
  ErrorStats s;
  for (size_t i = 0; i < want.size(); ++i) {
    const double err = std::fabs(double(got[i]) - want[i]);
    const double sc = scale.empty() ? std::fabs(want[i]) : scale[i];
    s.max_abs = std::max(s.max_abs, err);
    if (sc > 0) s.max_rel_to_scale = std::max(s.max_rel_to_scale, err / sc);
    if (!(err <= atol + rtol * sc)) ++s.bad;  // also catches NaN
    ++s.count;
  }
  return s;
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// qgemm: all bit widths, all tile configs, stream-K and data-parallel, every epilogue
// ------------------------------------------------------------------------------------------------
QWEN38_TEST(qgemm_matches_reference) {
  std::mt19937 rng(42);
  int cases = 0;
  bool bits_seen[9] = {};
  for (const auto& path : model_paths()) {
    auto file = ModelFile::open(path);
    ModelWeights weights(*file, EmbeddingPlacement::Device);
    std::map<std::string, kernels::BQWeight> tensors;
    for (int i = 0; i < file->config().num_layers; ++i) {
      const auto& L = weights.layers()[size_t(i)];
      const std::string p = std::to_string(i) + ".";
      tensors[p + (L.attention ? "attn.qkvg" : "gdn.qkvz")] = L.in_proj;
      tensors[p + (L.attention ? "attn.o" : "gdn.out")] = L.out_proj;
      tensors[p + "mlp.gate_up"] = L.gate_up;
      tensors[p + "mlp.down"] = L.down;
    }
    tensors["lm_head"] = weights.lm_head();
    std::map<int, int> per_bits;  // test at most 2 tensors per bit width and file
    DeviceBuffer ws_partials(size_t(32) << 20), ws_counters(65536 * 4);
    QWEN38_CUDA(cudaMemset(ws_counters.get(), 0, ws_counters.bytes()));
    kernels::GemmWorkspace ws{ws_partials.as<float>(), ws_partials.bytes() / 4, ws_counters.as<int>(), 65536};
    for (const auto& [name, w] : tensors) {
      if (per_bits[w.bits]++ >= 2) continue;
      bits_seen[w.bits] = true;
      const std::vector<float> wd = BQView::from_file(*file, name).dequant_all();
      for (int m : {1, 3, 16, 17, 40, 64, 130}) {
        const auto a = random_bf16_values(size_t(m) * w.k, rng);
        auto a_dev = upload(to_bf16_bits(a));
        // reference C and per-element magnitude scale sum_k |a w|
        std::vector<float> c(size_t(m) * w.n), mag(size_t(m) * w.n);
        ref::matmul_nt(a.data(), m, w.k, wd.data(), w.n, c.data());
        for (int i = 0; i < m; ++i)
          for (int j = 0; j < w.n; ++j) {
            double s = 0;
            for (int t = 0; t < w.k; ++t) s += std::fabs(double(a[size_t(i) * w.k + t]) * wd[size_t(j) * w.k + t]);
            mag[size_t(i) * w.n + j] = float(s);
          }
        for (int cfg = 0; cfg < kernels::qgemm_num_configs(); ++cfg) {
          if (cfg == 4 && w.k % 128) continue;
          for (bool streamk : {true, false}) {
            kernels::GemmWorkspace use_ws = streamk ? ws : kernels::GemmWorkspace{};
            DeviceBuffer out(size_t(m) * w.n * 4);
            kernels::GemmEpilogue ep;
            ep.kind = kernels::Epilogue::StoreF32;
            ep.out = out.get();
            ep.ldo = w.n;
            kernels::qgemm(a_dev.as<__nv_bfloat16>(), m, w, ep, use_ws, nullptr, cfg);
            QWEN38_CUDA(cudaDeviceSynchronize());
            const auto got = download<float>(out, size_t(m) * w.n);
            const auto st = compare(got, c, mag, 1e-6, 2e-3);
            CHECK_MSG(st.bad == 0, "%s bits=%d m=%d cfg=%s streamk=%d: %zu/%zu bad, max_abs=%g", name.c_str(),
                      w.bits, m, kernels::qgemm_config_name(cfg), int(streamk), st.bad, st.count, st.max_abs);
            // determinism: a second run must be bit-identical
            kernels::qgemm(a_dev.as<__nv_bfloat16>(), m, w, ep, use_ws, nullptr, cfg);
            QWEN38_CUDA(cudaDeviceSynchronize());
            const auto again = download<float>(out, size_t(m) * w.n);
            CHECK_MSG(std::memcmp(got.data(), again.data(), got.size() * 4) == 0, "%s cfg=%d: not deterministic",
                      name.c_str(), cfg);
            ++cases;
          }
        }
      }
    }
  }
  CHECK(bits_seen[4] && bits_seen[5] && bits_seen[6] && bits_seen[8]);
  std::printf("  qgemm: %d cases\n", cases);
}

QWEN38_TEST(qgemm_epilogues) {
  std::mt19937 rng(7);
  auto file = ModelFile::open(model_paths()[0]);
  ModelWeights weights(*file, EmbeddingPlacement::Device);
  const ModelConfig& cfg = file->config();
  const int H = cfg.hidden_size;
  const auto& L0 = weights.layers()[0];
  DeviceBuffer ws_partials(size_t(32) << 20), ws_counters(65536 * 4);
  QWEN38_CUDA(cudaMemset(ws_counters.get(), 0, ws_counters.bytes()));
  kernels::GemmWorkspace ws{ws_partials.as<float>(), ws_partials.bytes() / 4, ws_counters.as<int>(), 65536};

  for (int m : {1, 5, 16, 77}) {
    // residual stream + norm partials as the engine keeps them
    std::vector<float> residual = random_bf16_values(size_t(m) * H, rng, 3.0f);
    std::vector<float> xbf(residual.size()), partials(size_t(m) * cfg.norm_blocks());
    for (size_t i = 0; i < residual.size(); ++i) xbf[i] = round_bf16(residual[i]);
    for (int i = 0; i < m; ++i) ref::norm_partials(residual.data() + size_t(i) * H, H, partials.data() + size_t(i) * cfg.norm_blocks());
    std::vector<float> rs(static_cast<size_t>(m));
    for (int i = 0; i < m; ++i)
      rs[size_t(i)] = ref::norm_row_scale(partials.data() + size_t(i) * cfg.norm_blocks(), cfg.norm_blocks(), H, cfg.rms_norm_eps);
    auto xbf_dev = upload(to_bf16_bits(xbf));
    auto partials_dev = upload(partials);

    // StoreBF16 with the folded-norm row scale (in-projection)
    {
      const auto& w = L0.in_proj;
      const auto wd = BQView::from_file(*file, "0." + std::string(L0.attention ? "attn.qkvg" : "gdn.qkvz")).dequant_all();
      std::vector<float> want(size_t(m) * w.n);
      ref::qgemm(xbf.data(), m, wd, w.n, w.k, rs.data(), ref::Epilogue::StoreBF16, ref::GemmOut{want.data(), w.n}, false);
      DeviceBuffer out(size_t(m) * w.n * 2);
      kernels::GemmEpilogue ep;
      ep.kind = kernels::Epilogue::StoreBF16;
      ep.out = out.get();
      ep.ldo = w.n;
      ep.partials_in = partials_dev.as<float>();
      ep.hidden = H;
      ep.eps = cfg.rms_norm_eps;
      kernels::qgemm(xbf_dev.as<__nv_bfloat16>(), m, w, ep, ws, nullptr);
      QWEN38_CUDA(cudaDeviceSynchronize());
      const auto got = from_bf16_bits(download<uint16_t>(out, size_t(m) * w.n));
      const auto st = compare(got, want, {}, 1e-3, 2e-2);
      CHECK_MSG(st.bad == 0, "StoreBF16+norm m=%d: %zu bad (max_abs %g)", m, st.bad, st.max_abs);
    }
    // SwiGLU (gate_up)
    {
      const auto& w = L0.gate_up;
      const auto wd = BQView::from_file(*file, "0.mlp.gate_up").dequant_all();
      std::vector<float> want(size_t(m) * w.n / 2);
      ref::qgemm(xbf.data(), m, wd, w.n, w.k, rs.data(), ref::Epilogue::SwiGLU, ref::GemmOut{want.data(), w.n / 2}, false);
      DeviceBuffer out(size_t(m) * w.n);
      kernels::GemmEpilogue ep;
      ep.kind = kernels::Epilogue::SwiGLU;
      ep.out = out.get();
      ep.ldo = w.n / 2;
      ep.partials_in = partials_dev.as<float>();
      ep.hidden = H;
      ep.eps = cfg.rms_norm_eps;
      kernels::qgemm(xbf_dev.as<__nv_bfloat16>(), m, w, ep, ws, nullptr);
      QWEN38_CUDA(cudaDeviceSynchronize());
      const auto got = from_bf16_bits(download<uint16_t>(out, size_t(m) * w.n / 2));
      std::vector<float> scale(want.size());
      float mx = 0;
      for (float x : want) mx = std::max(mx, std::fabs(x));
      for (size_t i = 0; i < want.size(); ++i) scale[i] = std::fabs(want[i]) + 1e-3f * mx;
      const auto st = compare(got, want, scale, 0, 2e-2);
      CHECK_MSG(st.bad == 0, "SwiGLU m=%d: %zu bad (max_abs %g)", m, st.bad, st.max_abs);
    }
    // Residual (down projection): residual += C, xbf, partials
    {
      const auto& w = L0.down;
      const auto wd = BQView::from_file(*file, "0.mlp.down").dequant_all();
      const auto act = random_bf16_values(size_t(m) * w.k, rng, 0.5f);
      std::vector<float> res_want = residual, xbf_want(residual.size()), part_want(partials.size());
      ref::qgemm(act.data(), m, wd, w.n, w.k, nullptr, ref::Epilogue::Residual,
                 ref::GemmOut{nullptr, 0, res_want.data(), xbf_want.data(), part_want.data()}, false);
      auto act_dev = upload(to_bf16_bits(act));
      auto res_dev = upload(residual);
      DeviceBuffer xbf_out(residual.size() * 2), part_out(partials.size() * 4);
      kernels::GemmEpilogue ep;
      ep.kind = kernels::Epilogue::Residual;
      ep.residual = res_dev.as<float>();
      ep.xbf = xbf_out.as<__nv_bfloat16>();
      ep.partials_out = part_out.as<float>();
      kernels::qgemm(act_dev.as<__nv_bfloat16>(), m, w, ep, ws, nullptr);
      QWEN38_CUDA(cudaDeviceSynchronize());
      const auto res_got = download<float>(res_dev, residual.size());
      const auto st = compare(res_got, res_want, {}, 1e-3, 1e-3);
      CHECK_MSG(st.bad == 0, "Residual m=%d: residual %zu bad (max_abs %g)", m, st.bad, st.max_abs);
      const auto xbf_got = from_bf16_bits(download<uint16_t>(xbf_out, residual.size()));
      std::vector<float> xbf_exp(res_got.size());
      for (size_t i = 0; i < res_got.size(); ++i) xbf_exp[i] = round_bf16(res_got[i]);
      CHECK_MSG(xbf_got == xbf_exp, "Residual m=%d: xbf is not bf16(residual)", m);
      const auto part_got = download<float>(part_out, partials.size());
      const auto st2 = compare(part_got, part_want, {}, 1e-3, 1e-4);
      CHECK_MSG(st2.bad == 0, "Residual m=%d: partials %zu bad (max_abs %g)", m, st2.bad, st2.max_abs);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// embedding
// ------------------------------------------------------------------------------------------------
QWEN38_TEST(embed_matches_reference) {
  auto file = ModelFile::open(model_paths()[0]);
  const ModelConfig& cfg = file->config();
  for (EmbeddingPlacement place : {EmbeddingPlacement::Device, EmbeddingPlacement::Host}) {
    ModelWeights weights(*file, place);
    const auto table = file->direct<uint16_t>("embed");
    std::vector<int32_t> tokens = {0, 5, 1023, 17, 17, 400};
    const int m = int(tokens.size()), H = cfg.hidden_size;
    auto tok_dev = upload(tokens);
    DeviceBuffer res(size_t(m) * H * 4), xbf(size_t(m) * H * 2), part(size_t(m) * cfg.norm_blocks() * 4);
    kernels::embed(weights.embedding(), tok_dev.as<int32_t>(), m, H, res.as<float>(), xbf.as<__nv_bfloat16>(),
                   part.as<float>(), nullptr);
    QWEN38_CUDA(cudaDeviceSynchronize());
    const auto r = download<float>(res, size_t(m) * H);
    const auto p = download<float>(part, size_t(m) * cfg.norm_blocks());
    for (int i = 0; i < m; ++i) {
      std::vector<float> row(static_cast<size_t>(H)), pw(static_cast<size_t>(cfg.norm_blocks()));
      for (int d = 0; d < H; ++d) row[size_t(d)] = bf16_to_f32(table[size_t(tokens[size_t(i)]) * H + d]);
      ref::norm_partials(row.data(), H, pw.data());
      CHECK(std::equal(row.begin(), row.end(), r.begin() + ptrdiff_t(i) * H));
      for (int b = 0; b < cfg.norm_blocks(); ++b) CHECK_NEAR(p[size_t(i) * cfg.norm_blocks() + b], pw[size_t(b)], 1e-4, 1e-5);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// attention: qk-norm/RoPE/KV-append, decode (split-KV) and prefill (CuTe) against the reference
// ------------------------------------------------------------------------------------------------
namespace {

struct AttnFixture {
  ModelConfig cfg;
  kernels::KvDtype dtype;
  std::unique_ptr<KvCache> kv;
  std::vector<float> q_norm, k_norm;
  DeviceBuffer q_norm_dev, k_norm_dev;
  AttnFixture(const ModelConfig& c, kernels::KvDtype d, int pages) : cfg(c), dtype(d) {
    kv = std::make_unique<KvCache>(cfg, dtype, pages, 4, pages);
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(1.0f, 0.2f);
    q_norm.resize(256);
    k_norm.resize(256);
    for (auto& x : q_norm) x = nd(rng);
    for (auto& x : k_norm) x = nd(rng);
    q_norm_dev = upload(q_norm);
    k_norm_dev = upload(k_norm);
  }
  // Read back one sequence's cache (layer 0) as the reference KvStore.
  ref::KvStore read_back(int slot, int len) {
    ref::KvStore st;
    st.resize(cfg.num_key_value_heads, 256, len);
    const auto d = kv->desc();
    std::vector<int> table(static_cast<size_t>(d.max_pages_per_seq));
    QWEN38_CUDA(cudaMemcpy(table.data(), d.block_table + size_t(slot) * d.max_pages_per_seq, table.size() * 4,
                         cudaMemcpyDeviceToHost));
    std::vector<uint8_t> page(d.page_bytes);
    for (int p = 0; p * 64 < len; ++p) {
      QWEN38_CUDA(cudaMemcpy(page.data(), d.pool + size_t(table[size_t(p)]) * d.page_bytes, d.page_bytes,
                           cudaMemcpyDeviceToHost));
      for (int t = 0; t < 64 && p * 64 + t < len; ++t) {
        for (int h = 0; h < cfg.num_key_value_heads; ++h) {
          for (int which = 0; which < 2; ++which) {
            float* dst = which ? st.v_at(h, p * 64 + t) : st.k_at(h, p * 64 + t);
            const size_t off = which ? d.v_offset(h, t) : d.k_offset(h, t);
            if (dtype == kernels::KvDtype::Int8) {
              const size_t soff = which ? d.vs_offset(h, t) : d.ks_offset(h, t);
              const auto* sc = reinterpret_cast<const uint16_t*>(page.data() + soff);
              const auto* q = reinterpret_cast<const int8_t*>(page.data() + off);
              for (int e = 0; e < 256; ++e) dst[e] = float(q[e]) * f16_to_f32(sc[e / 64]);
            } else {
              const auto* v = reinterpret_cast<const uint16_t*>(page.data() + off);
              for (int e = 0; e < 256; ++e) dst[e] = bf16_to_f32(v[e]);
            }
          }
        }
      }
    }
    return st;
  }
};

// Host-side batch metadata for one sequence run of n tokens starting at pos0.
struct HostBatch {
  std::vector<int> seq_start, seq_slot, seq_pos0, tok_pos, tok_seq;
  DeviceBuffer d_seq_start, d_seq_slot, d_seq_pos0, d_tok_pos, d_tok_seq;
  kernels::BatchDesc desc;
  void build(const std::vector<std::tuple<int, int, int>>& seqs /* slot, pos0, n */) {
    seq_start = {0};
    seq_slot.clear();
    seq_pos0.clear();
    tok_pos.clear();
    tok_seq.clear();
    int max_n = 0;
    for (size_t b = 0; b < seqs.size(); ++b) {
      const auto [slot, pos0, n] = seqs[b];
      seq_slot.push_back(slot);
      seq_pos0.push_back(pos0);
      for (int i = 0; i < n; ++i) {
        tok_pos.push_back(pos0 + i);
        tok_seq.push_back(int(b));
      }
      seq_start.push_back(seq_start.back() + n);
      max_n = std::max(max_n, n);
    }
    d_seq_start = upload(seq_start);
    d_seq_slot = upload(seq_slot);
    d_seq_pos0 = upload(seq_pos0);
    d_tok_pos = upload(tok_pos);
    d_tok_seq = upload(tok_seq);
    desc.num_seqs = int(seqs.size());
    desc.num_tokens = seq_start.back();
    desc.max_tokens_per_seq = max_n;
    desc.seq_start = d_seq_start.as<int>();
    desc.seq_slot = d_seq_slot.as<int>();
    desc.seq_pos0 = d_seq_pos0.as<int>();
    desc.tok_pos = d_tok_pos.as<int>();
    desc.tok_seq = d_tok_seq.as<int>();
  }
};

}  // namespace

QWEN38_TEST(attention_matches_reference) {
  auto file = ModelFile::open(model_paths()[0]);
  ModelConfig cfg = file->config();  // tiny: 6 q heads, 1 kv head, head_dim 256
  const int nq = cfg.num_attention_heads, nkv = cfg.num_key_value_heads, hd = 256;
  const int ld_proj = cfg.qkvg_rows();
  std::mt19937 rng(11);
  for (auto dtype : {kernels::KvDtype::Int8, kernels::KvDtype::BF16}) {
    // sequence layout: slot 0 prefilled in chunks (37, 64, 90), then decoded; slot 1 decoded only
    for (int prefix : {0, 1, 63, 64, 300, 2100}) {
      AttnFixture fx(cfg, dtype, 64);
      std::vector<std::tuple<int, int>> chunks;  // (pos0, n)
      {
        int pos = 0;
        const int sizes[] = {37, 64, 90, 211};
        for (int i = 0; pos < prefix; ++i) {
          const int n = std::min(sizes[i % 4], prefix - pos);
          chunks.emplace_back(pos, n);
          pos += n;
        }
      }
      chunks.emplace_back(prefix, 1);  // one decode token at the end
      chunks.emplace_back(prefix + 1, 1);
      QWEN38_CHECK(fx.kv->ensure(0, prefix + 2), "fixture pages");
      fx.kv->sync(nullptr);
      std::vector<float> ref_out, got_out;
      for (auto [pos0, n] : chunks) {
        const auto proj = random_bf16_values(size_t(n) * ld_proj, rng, 1.0f);
        auto proj_dev = upload(to_bf16_bits(proj));
        HostBatch hb;
        hb.build({{0, pos0, n}});
        DeviceBuffer q16(size_t(n) * nq * hd * 2), out(size_t(n) * nq * hd * 2);
        kernels::QkNormRopeParams qp;
        qp.proj = proj_dev.as<__nv_bfloat16>();
        qp.ld_proj = ld_proj;
        qp.q_norm = fx.q_norm_dev.as<float>();
        qp.k_norm = fx.k_norm_dev.as<float>();
        qp.eps = cfg.rms_norm_eps;
        qp.rope_theta = float(cfg.rope_theta);
        qp.num_q_heads = nq;
        qp.num_kv_heads = nkv;
        qp.q_out = q16.as<__half>();
        qp.attn_layer = 0;
        kernels::qk_norm_rope_append(qp, hb.desc, fx.kv->desc(), nullptr);
        DeviceBuffer split(size_t(4) * nq * 64 * 258 * 4);
        kernels::AttentionParams ap;
        ap.q = q16.as<__half>();
        ap.gate = proj_dev.as<__nv_bfloat16>() + size_t(nq) * hd;
        ap.ld_gate = ld_proj;
        ap.out = out.as<__nv_bfloat16>();
        ap.num_q_heads = nq;
        ap.num_kv_heads = nkv;
        ap.attn_layer = 0;
        ap.split_scratch = split.as<float>();
        ap.max_splits = 64;
        if (n == 1) kernels::attention_decode(ap, hb.desc, fx.kv->desc(), nullptr);
        else kernels::attention_prefill(ap, hb.desc, fx.kv->desc(), nullptr);
        QWEN38_CUDA(cudaDeviceSynchronize());
        // q/k-norm + RoPE check (q only; K/V are checked through the attention output)
        const auto q_got = download<uint16_t>(q16, size_t(n) * nq * hd);
        const auto kvs = fx.read_back(0, pos0 + n);
        for (int i = 0; i < n; ++i) {
          std::vector<float> q_ref(size_t(nq) * hd);
          ref::KvStore scratch;
          scratch.resize(nkv, hd, pos0 + n);
          ref::qk_norm_rope_append(cfg, proj.data() + size_t(i) * ld_proj, pos0 + i, fx.q_norm.data(),
                                   fx.k_norm.data(), q_ref.data(), scratch,
                                   dtype == kernels::KvDtype::Int8 ? ref::KvDtype::Int8 : ref::KvDtype::BF16, false);
          std::vector<float> qg(size_t(nq) * hd);
          for (size_t e = 0; e < qg.size(); ++e) qg[e] = f16_to_f32(q_got[size_t(i) * nq * hd + e]);
          const auto st = compare(qg, q_ref, {}, 2e-3, 2e-3);
          CHECK_MSG(st.bad == 0, "qk_norm_rope q pos=%d: %zu bad (max_abs %g)", pos0 + i, st.bad, st.max_abs);
          // K written by the kernel vs reference quantization (allow one int8 step)
          for (int h = 0; h < nkv; ++h) {
            double maxdiff = 0, amax = 0;
            for (int e = 0; e < hd; ++e) {
              maxdiff = std::max(maxdiff, double(std::fabs(kvs.k[(size_t(h) * kvs.capacity + pos0 + i) * hd + e] -
                                                            scratch.k[(size_t(h) * scratch.capacity + pos0 + i) * hd + e])));
              amax = std::max(amax, double(std::fabs(scratch.k[(size_t(h) * scratch.capacity + pos0 + i) * hd + e])));
            }
            CHECK_MSG(maxdiff <= amax / 100.0 + 1e-3, "K cache pos=%d head=%d: max diff %g (amax %g)", pos0 + i, h,
                      maxdiff, amax);
          }
          // attention output vs reference over the cache contents the kernel actually wrote
          std::vector<float> want(size_t(nq) * hd), gate(size_t(nq) * hd);
          for (size_t e = 0; e < gate.size(); ++e) gate[e] = proj[size_t(i) * ld_proj + size_t(nq) * hd + e];
          ref::attention_token(cfg, qg.data(), gate.data(), pos0 + i, kvs, want.data(), false);
          ref_out.insert(ref_out.end(), want.begin(), want.end());
        }
        const auto o = from_bf16_bits(download<uint16_t>(out, size_t(n) * nq * hd));
        got_out.insert(got_out.end(), o.begin(), o.end());
      }
      std::vector<float> scale(ref_out.size());
      float mx = 0;
      for (float x : ref_out) mx = std::max(mx, std::fabs(x));
      for (size_t i = 0; i < ref_out.size(); ++i) scale[i] = std::fabs(ref_out[i]) + 0.02f * mx;
      const auto st = compare(got_out, ref_out, scale, 0, 2e-2);
      CHECK_MSG(st.bad == 0, "attention dtype=%d prefix=%d: %zu/%zu bad (max_abs %g, max rel %g)", int(dtype), prefix,
                st.bad, st.count, st.max_abs, st.max_rel_to_scale);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Gated DeltaNet: a/b projection, conv, recurrence over chunked prefill + decode
// ------------------------------------------------------------------------------------------------
QWEN38_TEST(gdn_matches_reference) {
  auto file = ModelFile::open(model_paths()[0]);
  const ModelConfig& cfg = file->config();
  int layer = 0;
  while (cfg.is_attention[size_t(layer)]) ++layer;
  const std::string p = std::to_string(layer) + ".";
  const int H = cfg.hidden_size, nv = cfg.linear_num_value_heads, conv_dim = cfg.gdn_conv_dim();
  const int ld_proj = int(file->tensor(p + "gdn.qkvz").shape[0]);
  ModelWeights weights(*file, EmbeddingPlacement::Device);
  const LayerWeights& W = weights.layers()[size_t(layer)];
  const auto conv_w = file->direct<float>(p + "gdn.conv");
  const auto A = file->direct<float>(p + "gdn.A");
  const auto dt = file->direct<float>(p + "gdn.dt_bias");
  const auto norm_w = file->direct<float>(p + "gdn.norm");
  const auto ab_w = file->direct<uint16_t>(p + "gdn.ab");
  std::vector<float> ab_wf(ab_w.size());
  for (size_t i = 0; i < ab_w.size(); ++i) ab_wf[i] = bf16_to_f32(ab_w[i]);

  GdnStatePool pool(cfg, 3);
  kernels::GdnDims d{cfg.linear_num_key_heads, nv, conv_dim, ld_proj};
  ref::GdnState ref_state;
  ref_state.s.assign(size_t(nv) * 128 * 128, 0.0f);
  ref_state.conv.assign(size_t(3) * conv_dim, 0.0f);
  std::mt19937 rng(5);
  int pos = 0;
  for (int n : {37, 64, 1, 1, 100, 1, 17}) {
    const auto proj = random_bf16_values(size_t(n) * ld_proj, rng, 1.0f);
    auto x = random_bf16_values(size_t(n) * H, rng, 1.0f);
    std::vector<float> partials(size_t(n) * cfg.norm_blocks());
    for (int i = 0; i < n; ++i) ref::norm_partials(x.data() + size_t(i) * H, H, partials.data() + size_t(i) * cfg.norm_blocks());
    auto proj_dev = upload(to_bf16_bits(proj));
    auto x_dev = upload(to_bf16_bits(x));
    auto part_dev = upload(partials);
    DeviceBuffer ab(size_t(n) * 2 * nv * 4), qkv(size_t(n) * conv_dim * 4), out(size_t(n) * nv * 128 * 2);
    kernels::gdn_ab(x_dev.as<__nv_bfloat16>(), n, H, W.ab, W.ab_rows, part_dev.as<float>(), cfg.rms_norm_eps,
                    ab.as<float>(), nullptr);
    HostBatch hb;
    hb.build({{1, pos, n}});  // slot 1
    kernels::GdnConvParams cp;
    cp.proj = proj_dev.as<__nv_bfloat16>();
    cp.conv_w = W.conv;
    cp.qkv = qkv.as<float>();
    cp.gdn_layer = 0;
    kernels::gdn_conv(cp, d, hb.desc, pool.desc(), nullptr);
    kernels::GdnRecurrenceParams rp;
    rp.qkv = qkv.as<float>();
    rp.ab = ab.as<float>();
    rp.A = W.A;
    rp.dt_bias = W.dt_bias;
    rp.z = proj_dev.as<__nv_bfloat16>() + conv_dim;
    rp.ld_z = ld_proj;
    rp.norm_w = W.gdn_norm;
    rp.eps = cfg.rms_norm_eps;
    rp.out = out.as<__nv_bfloat16>();
    rp.gdn_layer = 0;
    kernels::gdn_recurrence(rp, d, hb.desc, pool.desc(), nullptr);
    QWEN38_CUDA(cudaDeviceSynchronize());
    // a/b check
    const auto ab_got = download<float>(ab, size_t(n) * 2 * nv);
    std::vector<float> ab_ref(size_t(n) * 2 * nv), rs(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) rs[size_t(i)] = ref::norm_row_scale(partials.data() + size_t(i) * cfg.norm_blocks(), cfg.norm_blocks(), H, cfg.rms_norm_eps);
    ref::qgemm(x.data(), n, ab_wf, 2 * nv, H, rs.data(), ref::Epilogue::StoreF32, ref::GemmOut{ab_ref.data(), 2 * nv}, false);
    const auto st_ab = compare(ab_got, ab_ref, {}, 1e-3, 1e-3);
    CHECK_MSG(st_ab.bad == 0, "gdn_ab n=%d: %zu bad (max_abs %g)", n, st_ab.bad, st_ab.max_abs);
    // mixer output, token by token through the reference (fed with the GPU's a/b)
    const auto got = from_bf16_bits(download<uint16_t>(out, size_t(n) * nv * 128));
    std::vector<float> want(size_t(n) * nv * 128);
    for (int i = 0; i < n; ++i)
      ref::gdn_token(cfg, proj.data() + size_t(i) * ld_proj, ab_got.data() + size_t(i) * 2 * nv, conv_w.data(), A.data(),
                     dt.data(), norm_w.data(), ref_state, want.data() + size_t(i) * nv * 128, false);
    std::vector<float> scale(want.size());
    for (size_t i = 0; i < want.size(); ++i) scale[i] = std::fabs(want[i]) + 0.05f;
    const auto st = compare(got, want, scale, 0, 2e-2);
    CHECK_MSG(st.bad == 0, "gdn n=%d pos=%d: %zu/%zu bad (max_abs %g)", n, pos, st.bad, st.count, st.max_abs);
    pos += n;
  }
  // final recurrent state
  const auto desc = pool.desc();
  std::vector<float> s_got(size_t(nv) * 128 * 128);
  QWEN38_CUDA(cudaMemcpy(s_got.data(), desc.recurrent + size_t(1) * desc.num_gdn_layers * nv * 128 * 128,
                       s_got.size() * 4, cudaMemcpyDeviceToHost));
  const auto st = compare(s_got, ref_state.s, {}, 1e-3, 1e-2);
  CHECK_MSG(st.bad == 0, "gdn final state: %zu bad (max_abs %g)", st.bad, st.max_abs);
}

// ------------------------------------------------------------------------------------------------
// sampling
// ------------------------------------------------------------------------------------------------
QWEN38_TEST(sampling_filters_and_determinism) {
  const int V = 248320, B = 4;
  std::mt19937 rng(9);
  std::normal_distribution<float> nd(0.0f, 2.0f);
  std::vector<float> logits(size_t(B) * V);
  for (auto& x : logits) x = nd(rng);
  // row 3: a small explicit distribution over 5 tokens
  for (int i = 0; i < V; ++i) logits[size_t(3) * V + i] = -1e4f;
  const int toks[5] = {10, 20, 30, 40, 50};
  const float lp[5] = {0.0f, -0.5f, -1.0f, -2.0f, -8.0f};
  for (int i = 0; i < 5; ++i) logits[size_t(3) * V + toks[i]] = lp[i];

  auto run = [&](std::vector<kernels::SamplingParamsDev> params, int step) {
    auto lg = upload(logits);
    auto prm = upload(params);
    std::vector<int> slot = {0, 1, 2, 3}, steps(B, step), hl(B, 0);
    auto slot_dev = upload(slot);
    auto step_dev = upload(steps);
    auto hl_dev = upload(hl);
    DeviceBuffer hist(size_t(B) * 16 * 4), out(B * 4), lpo(B * 4);
    kernels::SamplingArgs a;
    a.logits = lg.as<float>();
    a.ld = V;
    a.vocab = V;
    a.params = prm.as<kernels::SamplingParamsDev>();
    a.row_slot = slot_dev.as<int>();
    a.history = hist.as<int32_t>();
    a.history_len = hl_dev.as<int>();
    a.history_cap = 16;
    a.step = step_dev.as<int>();
    a.tokens_out = out.as<int32_t>();
    a.logprob_out = lpo.as<float>();
    kernels::sample(a, B, nullptr);
    QWEN38_CUDA(cudaDeviceSynchronize());
    return download<int32_t>(out, B);
  };
  // greedy == argmax, top_k=1 == argmax
  std::vector<int> argmax(B);
  for (int b = 0; b < B; ++b)
    argmax[size_t(b)] = int(std::max_element(logits.begin() + ptrdiff_t(b) * V, logits.begin() + ptrdiff_t(b + 1) * V) -
                            (logits.begin() + ptrdiff_t(b) * V));
  {
    auto t = run(std::vector<kernels::SamplingParamsDev>(B, {0.0f, 0, 1.0f, 0.0f, 0.0f, 1.0f, 1}), 0);
    for (int b = 0; b < B; ++b) CHECK(t[size_t(b)] == argmax[size_t(b)]);
    auto t2 = run(std::vector<kernels::SamplingParamsDev>(B, {1.0f, 1, 1.0f, 0.0f, 0.0f, 1.0f, 1}), 0);
    for (int b = 0; b < B; ++b) CHECK(t2[size_t(b)] == argmax[size_t(b)]);
  }
  // row 3 empirical distribution with top_p = 0.9: nucleus {10, 20, 30} (0.49+0.30+0.18 >= 0.9)
  {
    std::map<int, int> counts;
    const int trials = 3000;
    for (int s = 0; s < trials; ++s) {
      std::vector<kernels::SamplingParamsDev> prm(B, {1.0f, 0, 0.9f, 0.0f, 0.0f, 1.0f, 1234});
      counts[run(prm, s)[3]]++;
    }
    CHECK_MSG(counts.count(40) == 0 && counts.count(50) == 0, "top-p leaked tokens outside the nucleus");
    double z = 0;
    for (int i = 0; i < 3; ++i) z += std::exp(lp[i]);
    for (int i = 0; i < 3; ++i) {
      const double expect = std::exp(lp[i]) / z * trials;
      CHECK_MSG(std::fabs(counts[toks[i]] - expect) < 5 * std::sqrt(expect), "token %d: %d samples, expected %.0f",
                toks[i], counts[toks[i]], expect);
    }
  }
  // min_p = 0.3 keeps tokens with p >= 0.3 * p_max: logits >= log(0.3) = -1.2 -> {10, 20, 30}
  {
    for (int s = 0; s < 500; ++s) {
      std::vector<kernels::SamplingParamsDev> prm(B, {1.0f, 0, 1.0f, 0.3f, 0.0f, 1.0f, 99});
      const int t = run(prm, s)[3];
      CHECK(t == 10 || t == 20 || t == 30);
    }
  }
  // determinism: same seed and step -> same tokens
  {
    std::vector<kernels::SamplingParamsDev> prm(B, {0.8f, 50, 0.95f, 0.0f, 0.0f, 1.0f, 777});
    CHECK(run(prm, 5) == run(prm, 5));
  }
}
