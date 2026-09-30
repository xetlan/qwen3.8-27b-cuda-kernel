#include "ref/ops.h"

#include <algorithm>
#include <cmath>

#include "qwen38/common.h"

namespace qwen38::ref {

float norm_row_scale(const float* partials, int blocks, int hidden, float eps) {
  float sum = 0.0f;
  for (int b = 0; b < blocks; ++b) sum += partials[b];
  return 1.0f / std::sqrt(sum / float(hidden) + eps);
}

void norm_partials(const float* row, int hidden, float* partials) {
  for (int b = 0; b < hidden / 128; ++b) {
    double s = 0;
    for (int i = 0; i < 128; ++i) s += double(row[b * 128 + i]) * row[b * 128 + i];
    partials[b] = float(s);
  }
}

void matmul_nt(const float* a, int m, int k, const float* w, int n, float* c) {
  for (int i = 0; i < m; ++i) {
    const float* ar = a + size_t(i) * k;
    for (int j = 0; j < n; ++j) {
      const float* wr = w + size_t(j) * k;
      double acc = 0;
      for (int t = 0; t < k; ++t) acc += double(ar[t]) * wr[t];
      c[size_t(i) * n + j] = float(acc);
    }
  }
}

void qgemm(const float* a, int m, const std::vector<float>& w_dequant, int n, int k, const float* row_scale,
           Epilogue ep, const GemmOut& out, bool exact) {
  std::vector<float> c(size_t(m) * n);
  matmul_nt(a, m, k, w_dequant.data(), n, c.data());
  auto rb = [exact](float x) { return exact ? x : round_bf16(x); };
  for (int i = 0; i < m; ++i) {
    const float rs = row_scale ? row_scale[i] : 1.0f;
    const float* cr = c.data() + size_t(i) * n;
    switch (ep) {
      case Epilogue::StoreBF16:
        for (int j = 0; j < n; ++j) out.out[size_t(i) * out.ldo + j] = rb(cr[j] * rs);
        break;
      case Epilogue::StoreF32:
        for (int j = 0; j < n; ++j) out.out[size_t(i) * out.ldo + j] = cr[j] * rs;
        break;
      case Epilogue::SwiGLU:
        for (int j = 0; j < n / 2; ++j) {
          const float g = cr[2 * j] * rs, u = cr[2 * j + 1] * rs;
          out.out[size_t(i) * out.ldo + j] = rb(silu(g) * u);
        }
        break;
      case Epilogue::Residual: {
        float* res = out.residual + size_t(i) * n;
        float* xb = out.xbf + size_t(i) * n;
        for (int j = 0; j < n; ++j) {
          res[j] += cr[j];
          xb[j] = rb(res[j]);
        }
        norm_partials(res, n, out.partials + size_t(i) * (n / 128));
        break;
      }
    }
  }
}

void KvStore::resize(int heads, int dim, int cap) {
  kv_heads = heads;
  head_dim = dim;
  capacity = cap;
  k.assign(size_t(heads) * cap * dim, 0.0f);
  v.assign(size_t(heads) * cap * dim, 0.0f);
}

void int8_group_roundtrip(float* x, int n) {
  for (int g0 = 0; g0 < n; g0 += 64) {
    float amax = 0.0f;
    for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(x[g0 + i]));
    const float scale = round_f16(amax / 127.0f);
    for (int i = 0; i < 64; ++i) {
      float q = 0.0f;
      if (scale > 0.0f) q = std::clamp(std::nearbyint(x[g0 + i] / scale), -127.0f, 127.0f);
      x[g0 + i] = q * scale;
    }
  }
}

namespace {

void rms_offset_norm(const float* x, int n, const float* weight, float eps, float* y) {
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
  const float r = 1.0f / std::sqrt(float(ss / n) + eps);
  for (int i = 0; i < n; ++i) y[i] = x[i] * r * weight[i];
}

void rope_partial(float* y, int rotary_dim, int pos, double theta) {
  const int half = rotary_dim / 2;
  for (int i = 0; i < half; ++i) {
    const float inv = 1.0f / std::pow(float(theta), float(2 * i) / float(rotary_dim));
    const float angle = float(pos) * inv;
    const float c = std::cos(angle), s = std::sin(angle);
    const float x1 = y[i], x2 = y[i + half];
    y[i] = x1 * c - x2 * s;
    y[i + half] = x2 * c + x1 * s;
  }
}

void store_kv(float* dst, const float* src, int n, KvDtype dtype, bool exact) {
  std::copy(src, src + n, dst);
  if (exact || dtype == KvDtype::F32) return;
  if (dtype == KvDtype::BF16) {
    for (int i = 0; i < n; ++i) dst[i] = round_bf16(dst[i]);
  } else {
    int8_group_roundtrip(dst, n);
  }
}

}  // namespace

void qk_norm_rope_append(const ModelConfig& cfg, const float* proj_row, int pos, const float* q_norm,
                         const float* k_norm, float* q_out, KvStore& kv, KvDtype kv_dtype, bool exact) {
  const int nq = cfg.num_attention_heads, nkv = cfg.num_key_value_heads, hd = cfg.head_dim;
  std::vector<float> y(hd);
  for (int h = 0; h < nq; ++h) {
    rms_offset_norm(proj_row + size_t(h) * hd, hd, q_norm, cfg.rms_norm_eps, y.data());
    rope_partial(y.data(), cfg.rotary_dim, pos, cfg.rope_theta);
    for (int d = 0; d < hd; ++d) q_out[size_t(h) * hd + d] = exact ? y[d] : round_f16(y[d]);
  }
  const float* kbase = proj_row + size_t(2) * nq * hd;
  const float* vbase = kbase + size_t(nkv) * hd;
  for (int h = 0; h < nkv; ++h) {
    rms_offset_norm(kbase + size_t(h) * hd, hd, k_norm, cfg.rms_norm_eps, y.data());
    rope_partial(y.data(), cfg.rotary_dim, pos, cfg.rope_theta);
    store_kv(kv.k_at(h, pos), y.data(), hd, kv_dtype, exact);
    store_kv(kv.v_at(h, pos), vbase + size_t(h) * hd, hd, kv_dtype, exact);
  }
}

void attention_token(const ModelConfig& cfg, const float* q, const float* gate, int pos, const KvStore& kv,
                     float* out, bool exact) {
  const int nq = cfg.num_attention_heads, hd = cfg.head_dim, group = cfg.attn_group();
  const double scale = 1.0 / std::sqrt(double(hd));
  std::vector<double> s(size_t(pos) + 1);
  for (int h = 0; h < nq; ++h) {
    const int kvh = h / group;
    double mx = -1e300;
    for (int j = 0; j <= pos; ++j) {
      const float* kr = kv.k.data() + (size_t(kvh) * kv.capacity + j) * hd;
      double dot = 0;
      for (int d = 0; d < hd; ++d) dot += double(q[size_t(h) * hd + d]) * kr[d];
      s[j] = dot * scale;
      mx = std::max(mx, s[j]);
    }
    double l = 0;
    for (int j = 0; j <= pos; ++j) {
      s[j] = std::exp(s[j] - mx);
      l += s[j];
    }
    for (int d = 0; d < hd; ++d) {
      double o = 0;
      for (int j = 0; j <= pos; ++j) o += s[j] * kv.v[(size_t(kvh) * kv.capacity + j) * hd + d];
      const float val = float(o / l) * sigmoid(gate[size_t(h) * hd + d]);
      out[size_t(h) * hd + d] = exact ? val : round_bf16(val);
    }
  }
}

void gdn_token(const ModelConfig& cfg, const float* proj_row, const float* ab_row, const float* conv_w,
               const float* A, const float* dt_bias, const float* norm_w, GdnState& st, float* out,
               bool exact) {
  const int lk = cfg.linear_num_key_heads, lv = cfg.linear_num_value_heads;
  const int dk = cfg.linear_key_head_dim, dv = cfg.linear_value_head_dim;
  const int conv_dim = cfg.gdn_conv_dim();
  const int taps = cfg.linear_conv_kernel_dim;
  const int ratio = lv / lk;

  // causal depthwise conv over [q | k | v] with the stored history, then SiLU
  std::vector<float> act(conv_dim);
  for (int c = 0; c < conv_dim; ++c) {
    float acc = 0.0f;
    for (int j = 0; j < taps - 1; ++j) acc += conv_w[size_t(c) * taps + j] * st.conv[size_t(j) * conv_dim + c];
    acc += conv_w[size_t(c) * taps + taps - 1] * proj_row[c];
    act[c] = silu(acc);
  }
  for (int j = 0; j < taps - 2; ++j)
    std::copy(st.conv.begin() + size_t(j + 1) * conv_dim, st.conv.begin() + size_t(j + 2) * conv_dim,
              st.conv.begin() + size_t(j) * conv_dim);
  std::copy(proj_row, proj_row + conv_dim, st.conv.begin() + size_t(taps - 2) * conv_dim);

  float* qall = act.data();
  float* kall = act.data() + size_t(lk) * dk;
  const float* vall = act.data() + size_t(2) * lk * dk;
  const float* z = proj_row + conv_dim;
  const float qscale = 1.0f / std::sqrt(float(dk));
  for (int h = 0; h < lk; ++h) {
    for (float* vec : {qall + size_t(h) * dk, kall + size_t(h) * dk}) {
      double ss = 0;
      for (int d = 0; d < dk; ++d) ss += double(vec[d]) * vec[d];
      const float r = 1.0f / std::sqrt(float(ss) + 1e-6f);
      for (int d = 0; d < dk; ++d) vec[d] *= r;
    }
    for (int d = 0; d < dk; ++d) qall[size_t(h) * dk + d] *= qscale;
  }

  std::vector<float> o(dv);
  for (int j = 0; j < lv; ++j) {
    const float* q = qall + size_t(j / ratio) * dk;
    const float* k = kall + size_t(j / ratio) * dk;
    const float* v = vall + size_t(j) * dv;
    const float beta = sigmoid(ab_row[j]);
    const float g = A[j] * softplus(ab_row[lv + j] + dt_bias[j]);
    const float decay = std::exp(g);
    float* S = st.s.data() + size_t(j) * dv * dk;
    for (int a = 0; a < dv; ++a) {
      float* row = S + size_t(a) * dk;
      float kv_mem = 0.0f;
      for (int b = 0; b < dk; ++b) {
        row[b] *= decay;
        kv_mem += row[b] * k[b];
      }
      const float delta = beta * (v[a] - kv_mem);
      float acc = 0.0f;
      for (int b = 0; b < dk; ++b) {
        row[b] += delta * k[b];
        acc += row[b] * q[b];
      }
      o[a] = acc;
    }
    double ss = 0;
    for (int a = 0; a < dv; ++a) ss += double(o[a]) * o[a];
    const float r = 1.0f / std::sqrt(float(ss / dv) + cfg.rms_norm_eps);
    for (int a = 0; a < dv; ++a) {
      const float val = o[a] * r * norm_w[a] * silu(z[size_t(j) * dv + a]);
      out[size_t(j) * dv + a] = exact ? val : round_bf16(val);
    }
  }
}

}  // namespace qwen38::ref
