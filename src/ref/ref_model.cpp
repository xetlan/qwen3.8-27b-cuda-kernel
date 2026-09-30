#include "ref/ref_model.h"

#include "qwen38/bq.h"
#include "qwen38/common.h"

namespace qwen38::ref {

namespace {

std::vector<float> load_f32(const ModelFile& f, const std::string& name) {
  auto s = f.direct<float>(name);
  return {s.begin(), s.end()};
}

std::vector<float> dequant(const ModelFile& f, const std::string& name) {
  return BQView::from_file(f, name).dequant_all();
}

}  // namespace

RefModel::RefModel(const ModelFile& file, RefOptions options) : opt_(options), cfg_(file.config()) {
  const int64_t params = file.tensor("lm_head").numel() * 2 +
                         int64_t(cfg_.num_layers) * 3 * cfg_.hidden_size * cfg_.intermediate_size;
  QWEN38_CHECK(params < 3'000'000'000LL, "the CPU reference model is meant for small test models");
  embed_ = file.direct<uint16_t>("embed");
  lm_head_ = dequant(file, "lm_head");
  layers_.resize(cfg_.num_layers);
  for (int i = 0; i < cfg_.num_layers; ++i) {
    Layer& L = layers_[i];
    const std::string p = std::to_string(i) + ".";
    L.attention = cfg_.is_attention[i] != 0;
    if (L.attention) {
      L.w_in = dequant(file, p + "attn.qkvg");
      L.in_rows = int(file.tensor(p + "attn.qkvg").shape[0]);
      L.w_out = dequant(file, p + "attn.o");
      L.q_norm = load_f32(file, p + "attn.q_norm");
      L.k_norm = load_f32(file, p + "attn.k_norm");
      L.mixer_width = cfg_.q_dim();
    } else {
      L.w_in = dequant(file, p + "gdn.qkvz");
      L.in_rows = int(file.tensor(p + "gdn.qkvz").shape[0]);
      {
        auto ab = file.direct<uint16_t>(p + "gdn.ab");
        L.w_ab.resize(ab.size());
        for (size_t e = 0; e < ab.size(); ++e) L.w_ab[e] = bf16_to_f32(ab[e]);
        L.ab_rows = int(file.tensor(p + "gdn.ab").shape[0]);
      }
      L.w_out = dequant(file, p + "gdn.out");
      L.conv = load_f32(file, p + "gdn.conv");
      L.A = load_f32(file, p + "gdn.A");
      L.dt_bias = load_f32(file, p + "gdn.dt_bias");
      L.gdn_norm = load_f32(file, p + "gdn.norm");
      L.mixer_width = cfg_.gdn_value_dim();
    }
    L.w_gate_up = dequant(file, p + "mlp.gate_up");
    L.w_down = dequant(file, p + "mlp.down");
  }
}

RefModel::Sequence RefModel::new_sequence() const {
  Sequence s;
  for (int i = 0; i < cfg_.num_layers; ++i) {
    if (cfg_.is_attention[i]) {
      s.kv.emplace_back();
      s.kv.back().resize(cfg_.num_key_value_heads, cfg_.head_dim, opt_.max_positions);
    } else {
      GdnState g;
      g.s.assign(size_t(cfg_.linear_num_value_heads) * cfg_.linear_value_head_dim * cfg_.linear_key_head_dim, 0.0f);
      g.conv.assign(size_t(cfg_.linear_conv_kernel_dim - 1) * cfg_.gdn_conv_dim(), 0.0f);
      s.gdn.push_back(std::move(g));
    }
  }
  return s;
}

std::vector<float> RefModel::forward(Sequence& seq, std::span<const int32_t> tokens, bool all_logits) const {
  const int H = cfg_.hidden_size, V = cfg_.vocab_size;
  const int blocks = cfg_.norm_blocks();
  const bool exact = opt_.exact;
  std::vector<float> logits_out;

  std::vector<float> residual(H), xbf(H), partials(blocks);
  std::vector<float> proj, ab, mixer, act(cfg_.intermediate_size), q(cfg_.q_dim()), logits(V);

  for (size_t t = 0; t < tokens.size(); ++t) {
    const int token = tokens[t];
    QWEN38_CHECK(token >= 0 && token < V, "token id out of range");
    QWEN38_CHECK(seq.pos < opt_.max_positions, "sequence longer than RefOptions::max_positions");
    for (int d = 0; d < H; ++d) residual[d] = bf16_to_f32(embed_[size_t(token) * H + d]);
    xbf = residual;
    norm_partials(residual.data(), H, partials.data());
    GemmOut res_out{nullptr, 0, residual.data(), xbf.data(), partials.data()};

    int attn_i = 0, gdn_i = 0;
    for (int li = 0; li < cfg_.num_layers; ++li) {
      const Layer& L = layers_[li];
      float rs = norm_row_scale(partials.data(), blocks, H, cfg_.rms_norm_eps);
      proj.assign(L.in_rows, 0.0f);
      qgemm(xbf.data(), 1, L.w_in, L.in_rows, H, &rs, Epilogue::StoreBF16, GemmOut{proj.data(), L.in_rows}, exact);
      mixer.assign(L.mixer_width, 0.0f);
      if (L.attention) {
        KvStore& kv = seq.kv[attn_i++];
        qk_norm_rope_append(cfg_, proj.data(), seq.pos, L.q_norm.data(), L.k_norm.data(), q.data(), kv,
                            opt_.kv_dtype, exact);
        attention_token(cfg_, q.data(), proj.data() + cfg_.q_dim(), seq.pos, kv, mixer.data(), exact);
      } else {
        ab.assign(L.ab_rows, 0.0f);
        qgemm(xbf.data(), 1, L.w_ab, L.ab_rows, H, &rs, Epilogue::StoreF32, GemmOut{ab.data(), L.ab_rows}, exact);
        gdn_token(cfg_, proj.data(), ab.data(), L.conv.data(), L.A.data(), L.dt_bias.data(), L.gdn_norm.data(),
                  seq.gdn[gdn_i++], mixer.data(), exact);
      }
      qgemm(mixer.data(), 1, L.w_out, H, L.mixer_width, nullptr, Epilogue::Residual, res_out, exact);

      rs = norm_row_scale(partials.data(), blocks, H, cfg_.rms_norm_eps);
      qgemm(xbf.data(), 1, L.w_gate_up, 2 * cfg_.intermediate_size, H, &rs, Epilogue::SwiGLU,
            GemmOut{act.data(), cfg_.intermediate_size}, exact);
      qgemm(act.data(), 1, L.w_down, H, cfg_.intermediate_size, nullptr, Epilogue::Residual, res_out, exact);
    }
    seq.pos += 1;
    if (all_logits || t + 1 == tokens.size()) {
      const float rs = norm_row_scale(partials.data(), blocks, H, cfg_.rms_norm_eps);
      qgemm(xbf.data(), 1, lm_head_, V, H, &rs, Epilogue::StoreF32, GemmOut{logits.data(), V}, exact);
      logits_out.insert(logits_out.end(), logits.begin(), logits.end());
    }
  }
  return logits_out;
}

}  // namespace qwen38::ref
