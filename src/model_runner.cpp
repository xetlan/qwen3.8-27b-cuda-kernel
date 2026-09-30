#include "qwen38/model_runner.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

#include "qwen38/common.h"

namespace qwen38 {

namespace {

constexpr size_t kMiB = 1024 * 1024;
constexpr int kMaxSplits = 64;

size_t align256(size_t x) { return (x + 255) / 256 * 256; }

std::string mib(size_t bytes) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(0);
  os << double(bytes) / double(kMiB) << " MiB";
  return os.str();
}

// Bucketed batch sizes that get their own decode graph.
int bucket_for(int n, int max_batch) {
  static const int kBuckets[] = {1, 2, 4, 8, 12, 16, 24, 32, 48, 64};
  for (int b : kBuckets)
    if (b >= n && b <= max_batch) return b;
  return max_batch;
}

}  // namespace

std::string MemoryPlan::describe() const {
  std::ostringstream os;
  os << "VRAM free at start " << mib(free_at_start) << ": weights " << mib(weights) << ", workspace "
     << mib(workspace) << ", GDN state " << mib(state) << ", KV " << mib(kv) << " (" << kv_tokens
     << " tokens), headroom " << mib(reserve);
  if (embedding_host) os << "; embedding in pinned host memory " << mib(embedding_host);
  return os.str();
}

// Step metadata: one contiguous struct, filled on the host and copied with one H2D memcpy.
struct MetaLayout {
  size_t tokens, tok_pos, tok_seq, seq_start, seq_slot, seq_pos0, row_slot, step, params, bytes;
  MetaLayout(int max_tokens, int max_rows) {
    size_t o = 0;
    auto take = [&](size_t bytes) {
      const size_t at = o;
      o = align256(o + bytes);
      return at;
    };
    tokens = take(size_t(max_tokens) * 4);
    tok_pos = take(size_t(max_tokens) * 4);
    tok_seq = take(size_t(max_tokens) * 4);
    seq_start = take(size_t(max_rows + 1) * 4);
    seq_slot = take(size_t(max_rows) * 4);
    seq_pos0 = take(size_t(max_rows) * 4);
    row_slot = take(size_t(max_rows) * 4);
    step = take(size_t(max_rows) * 4);
    params = take(size_t(max_rows) * sizeof(kernels::SamplingParamsDev));
    bytes = o;
  }
};

struct ModelRunner::Impl {
  const ModelFile& file;
  ModelConfig cfg;
  RunnerOptions opt;
  DeviceInfo dev;
  MemoryPlan plan;
  std::unique_ptr<ModelWeights> weights;
  std::unique_ptr<KvCache> kv;
  std::unique_ptr<GdnStatePool> state;
  Stream stream;
  DeviceArena ws;
  int max_tokens = 0;  // rows of every activation buffer
  int max_rows = 0;    // sequences per step, incl. padding rows
  int dummy_slot = 0;

  // activations
  float* residual = nullptr;
  __nv_bfloat16* xbf = nullptr;
  float* partials = nullptr;
  __nv_bfloat16* proj = nullptr;
  int proj_ld = 0;
  float* ab = nullptr;
  float* qkv = nullptr;
  __nv_bfloat16* mixer = nullptr;
  __nv_bfloat16* act = nullptr;
  __half* q16 = nullptr;
  float* logits = nullptr;
  float* split = nullptr;
  float* logprob_rows = nullptr;
  kernels::GemmWorkspace gemm_ws;
  // sampling state
  int32_t* history = nullptr;
  int* history_len = nullptr;
  int32_t* tokens_out = nullptr;
  float* logprob_out = nullptr;
  int32_t* targets = nullptr;
  // metadata
  MetaLayout meta{1, 1};
  uint8_t* meta_dev = nullptr;
  PinnedBuffer meta_host;
  PinnedBuffer out_host;  // tokens (int32) then logprobs (float), max_rows each
  PinnedBuffer staging;   // generic D2H staging
  std::map<int, Graph> graphs;
  bool graphs_enabled = true;
  // profiling
  bool profiling = false;
  OpProfile prof;
  std::vector<std::pair<std::string, std::unique_ptr<Event>>> marks;

  Impl(const ModelFile& f, const RunnerOptions& o) : file(f), cfg(f.config()), opt(o) {}

  template <typename T>
  T* host_meta(size_t off) const {
    return reinterpret_cast<T*>(static_cast<uint8_t*>(meta_host.get()) + off);
  }
  template <typename T>
  T* dev_meta(size_t off) const {
    return reinterpret_cast<T*>(meta_dev + off);
  }

  kernels::BatchDesc batch_desc(int num_seqs, int num_tokens, int max_per_seq) const {
    kernels::BatchDesc b;
    b.num_seqs = num_seqs;
    b.num_tokens = num_tokens;
    b.max_tokens_per_seq = max_per_seq;
    b.seq_start = dev_meta<int>(meta.seq_start);
    b.seq_slot = dev_meta<int>(meta.seq_slot);
    b.seq_pos0 = dev_meta<int>(meta.seq_pos0);
    b.tok_pos = dev_meta<int>(meta.tok_pos);
    b.tok_seq = dev_meta<int>(meta.tok_seq);
    return b;
  }

  void mark(const char* name) {
    if (!profiling) return;
    auto ev = std::make_unique<Event>(true);
    ev->record(stream.get());
    marks.emplace_back(name, std::move(ev));
  }

  void collect_marks() {
    if (!profiling || marks.size() < 2) {
      marks.clear();
      return;
    }
    marks.back().second->sync();
    for (size_t i = 1; i < marks.size(); ++i) {
      const float ms = marks[i].second->ms_since(*marks[i - 1].second);
      prof.ms[marks[i].first] += ms;
      prof.calls[marks[i].first] += 1;
    }
    marks.clear();
  }

  void setup();
  void forward(int m, const kernels::BatchDesc& batch, bool decode, int lm_row, int lm_rows);
  void run_sampler(int rows);
  void upload_meta() {
    QWEN38_CUDA(cudaMemcpyAsync(meta_dev, meta_host.get(), meta.bytes, cudaMemcpyHostToDevice, stream.get()));
  }
  void download_results(int rows) {
    QWEN38_CUDA(cudaMemcpyAsync(out_host.get(), tokens_out, size_t(rows) * 4, cudaMemcpyDeviceToHost, stream.get()));
    QWEN38_CUDA(cudaMemcpyAsync(static_cast<uint8_t*>(out_host.get()) + size_t(max_rows) * 4, logprob_out,
                              size_t(rows) * 4, cudaMemcpyDeviceToHost, stream.get()));
  }
  StepResult read_results(int rows) const {
    StepResult r;
    const auto* t = out_host.as<int32_t>();
    const auto* lp = reinterpret_cast<const float*>(static_cast<const uint8_t*>(out_host.get()) + size_t(max_rows) * 4);
    r.tokens.assign(t, t + rows);
    r.logprobs.assign(lp, lp + rows);
    return r;
  }
  static kernels::SamplingParamsDev to_dev(const SamplingParams& s) {
    return {s.temperature, s.top_k, s.top_p, s.min_p, s.presence_penalty, s.repetition_penalty, s.seed};
  }
};

void ModelRunner::Impl::setup() {
  dev = query_device(0);
  if (dev.sm_major != 8 || dev.sm_minor != 6) {
    std::cerr << "qwen38: warning: built for sm_86 (RTX 3090), running on " << dev.name << " (sm_" << dev.sm_major
              << dev.sm_minor << ")\n";
  }
  if (opt.embedding == EmbeddingPlacement::Host && !dev.can_map_host) {
    std::cerr << "qwen38: device cannot map host memory, keeping the embedding in VRAM\n";
    opt.embedding = EmbeddingPlacement::Device;
  }
  const int H = cfg.hidden_size, V = cfg.vocab_size;
  max_rows = opt.max_batch + 1;  // + dummy row/slot for graph padding
  dummy_slot = opt.max_batch;
  max_tokens = std::max(opt.prefill_chunk, max_rows);
  QWEN38_CHECK(opt.prefill_chunk >= 64 && opt.prefill_chunk % 64 == 0, "prefill chunk must be a multiple of 64");

  // ---- memory plan -----------------------------------------------------------------------------
  plan.free_at_start = dev.free_mem;
  plan.weights = ModelWeights::required_device_bytes(file, opt.embedding);
  if (opt.embedding == EmbeddingPlacement::Host) plan.embedding_host = size_t(file.tensor("embed").bytes);

  int proj_rows = 0;
  for (int i = 0; i < cfg.num_layers; ++i) {
    const std::string name = std::to_string(i) + (cfg.is_attention[i] ? ".attn.qkvg" : ".gdn.qkvz");
    proj_rows = std::max(proj_rows, int(file.tensor(name).shape[0]));
  }
  proj_ld = proj_rows;
  const size_t T = size_t(max_tokens);
  const size_t mixer_w = size_t(std::max(cfg.q_dim(), cfg.gdn_value_dim()));
  size_t w = 0;
  auto add = [&](size_t bytes) { w += align256(bytes); };
  add(T * H * 4);                                  // residual
  add(T * H * 2);                                  // xbf
  add(T * cfg.norm_blocks() * 4);                  // partials
  add(T * size_t(proj_ld) * 2);                    // proj
  add(T * 2 * size_t(cfg.linear_num_value_heads) * 4);  // ab
  add(T * size_t(cfg.gdn_conv_dim()) * 4);         // qkv
  add(T * mixer_w * 2);                            // mixer
  add(T * size_t(cfg.intermediate_size) * 2);      // act
  add(T * size_t(cfg.q_dim()) * 2);                // q16
  add(size_t(max_rows) * V * 4);                   // logits
  add(size_t(max_rows) * cfg.num_attention_heads * kMaxSplits * (kernels::kHeadDim + 2) * 4);  // split scratch
  add(T * 4);                                      // logprob rows
  const size_t gemm_partials = size_t(2) * 3 * dev.num_sms * 128 * 128;  // floats
  add(gemm_partials * 4);
  add(65536 * 4);                                  // tile counters
  add(size_t(max_rows) * opt.history_cap * 4);     // history
  add(size_t(max_rows) * 4 * 4);                   // history_len, tokens_out, logprob_out, spare
  add(T * 4);                                      // targets
  meta = MetaLayout(max_tokens, max_rows);
  add(meta.bytes);
  plan.workspace = w + 4096;
  plan.state = GdnStatePool::bytes_per_slot(cfg) * size_t(max_rows);
  plan.reserve = opt.reserve_mib * kMiB;

  const size_t page_bytes = KvCache::page_bytes_all_layers(cfg, opt.kv_dtype);
  const size_t fixed = plan.weights + plan.workspace + plan.state + plan.reserve;
  QWEN38_CHECK(plan.free_at_start > fixed + page_bytes * 2, "not enough GPU memory: need ",
             (fixed + page_bytes * 2) / kMiB, " MiB before any KV cache, ", plan.free_at_start / kMiB, " MiB free");
  int64_t pages = int64_t((plan.free_at_start - fixed) / std::max<size_t>(page_bytes, 1));
  if (opt.kv_tokens > 0) pages = std::min<int64_t>(pages, ceil_div(opt.kv_tokens, kernels::kPageTokens) + 1);
  pages = std::min<int64_t>(pages, 1 << 22);
  plan.kv_pages = int(pages);
  plan.kv = size_t(pages) * page_bytes;
  plan.kv_tokens = (pages - 1) * kernels::kPageTokens;  // one page belongs to the dummy sequence

  // ---- allocations ----------------------------------------------------------------------------
  weights = std::make_unique<ModelWeights>(file, opt.embedding);
  const int max_pages_per_seq = int(ceil_div(opt.max_context, kernels::kPageTokens));
  kv = std::make_unique<KvCache>(cfg, opt.kv_dtype, plan.kv_pages, max_rows, max_pages_per_seq);
  QWEN38_CHECK(kv->ensure(dummy_slot, 1), "KV cache too small");
  state = std::make_unique<GdnStatePool>(cfg, max_rows);

  ws = DeviceArena(plan.workspace);
  residual = ws.allocate_array<float>(T * H);
  xbf = ws.allocate_array<__nv_bfloat16>(T * H);
  partials = ws.allocate_array<float>(T * cfg.norm_blocks());
  proj = ws.allocate_array<__nv_bfloat16>(T * size_t(proj_ld));
  ab = ws.allocate_array<float>(T * 2 * size_t(cfg.linear_num_value_heads));
  qkv = ws.allocate_array<float>(T * size_t(cfg.gdn_conv_dim()));
  mixer = ws.allocate_array<__nv_bfloat16>(T * mixer_w);
  act = ws.allocate_array<__nv_bfloat16>(T * size_t(cfg.intermediate_size));
  q16 = ws.allocate_array<__half>(T * size_t(cfg.q_dim()));
  logits = ws.allocate_array<float>(size_t(max_rows) * V);
  split = ws.allocate_array<float>(size_t(max_rows) * cfg.num_attention_heads * kMaxSplits * (kernels::kHeadDim + 2));
  logprob_rows = ws.allocate_array<float>(T);
  gemm_ws.partials = ws.allocate_array<float>(gemm_partials);
  gemm_ws.partial_floats = gemm_partials;
  gemm_ws.counters = ws.allocate_array<int>(65536);
  gemm_ws.num_counters = 65536;
  history = ws.allocate_array<int32_t>(size_t(max_rows) * opt.history_cap);
  history_len = ws.allocate_array<int>(max_rows);
  tokens_out = ws.allocate_array<int32_t>(max_rows);
  logprob_out = ws.allocate_array<float>(max_rows);
  targets = ws.allocate_array<int32_t>(T);
  meta_dev = static_cast<uint8_t*>(ws.allocate(meta.bytes));
  QWEN38_CUDA(cudaMemset(gemm_ws.counters, 0, 65536 * 4));
  QWEN38_CUDA(cudaMemset(history_len, 0, size_t(max_rows) * 4));
  meta_host = PinnedBuffer(meta.bytes, false);
  std::memset(meta_host.get(), 0, meta.bytes);
  out_host = PinnedBuffer(size_t(max_rows) * 8, false);
  graphs_enabled = opt.cuda_graphs;
  if (opt.verbose) std::cerr << "qwen38: " << plan.describe() << "\n";
}

void ModelRunner::Impl::forward(int m, const kernels::BatchDesc& batch, bool decode, int lm_row, int lm_rows) {
  cudaStream_t s = stream.get();
  const int H = cfg.hidden_size;
  const float eps = cfg.rms_norm_eps;
  const kernels::KvCacheDesc kvd = kv->desc();
  const kernels::GdnStateDesc sd = state->desc();
  kernels::GdnDims gd;
  gd.num_k_heads = cfg.linear_num_key_heads;
  gd.num_v_heads = cfg.linear_num_value_heads;
  gd.conv_dim = cfg.gdn_conv_dim();
  gd.ld_proj = proj_ld;

  mark("start");
  kernels::embed(weights->embedding(), dev_meta<int32_t>(meta.tokens), m, H, residual, xbf, partials, s);
  mark("embed");

  kernels::GemmEpilogue ep_res;
  ep_res.kind = kernels::Epilogue::Residual;
  ep_res.residual = residual;
  ep_res.xbf = xbf;
  ep_res.partials_out = partials;

  int attn_i = 0, gdn_i = 0;
  for (const LayerWeights& W : weights->layers()) {
    kernels::GemmEpilogue ep_in;
    ep_in.kind = kernels::Epilogue::StoreBF16;
    ep_in.out = proj;
    ep_in.ldo = proj_ld;
    ep_in.n_store = W.in_proj.rows;
    ep_in.partials_in = partials;
    ep_in.hidden = H;
    ep_in.eps = eps;
    kernels::qgemm(xbf, m, W.in_proj, ep_in, gemm_ws, s);
    mark(W.attention ? "gemm.attn_qkvg" : "gemm.gdn_qkvz");
    if (W.attention) {
      kernels::QkNormRopeParams qp;
      qp.proj = proj;
      qp.ld_proj = proj_ld;
      qp.q_norm = W.q_norm;
      qp.k_norm = W.k_norm;
      qp.eps = eps;
      qp.rope_theta = float(cfg.rope_theta);
      qp.num_q_heads = cfg.num_attention_heads;
      qp.num_kv_heads = cfg.num_key_value_heads;
      qp.q_out = q16;
      qp.attn_layer = attn_i;
      kernels::qk_norm_rope_append(qp, batch, kvd, s);
      mark("attn.qk_norm_rope_kv");
      kernels::AttentionParams ap;
      ap.q = q16;
      ap.gate = proj + cfg.q_dim();
      ap.ld_gate = proj_ld;
      ap.out = mixer;
      ap.num_q_heads = cfg.num_attention_heads;
      ap.num_kv_heads = cfg.num_key_value_heads;
      ap.attn_layer = attn_i;
      ap.split_scratch = split;
      ap.max_splits = kMaxSplits;
      if (decode) {
        kernels::attention_decode(ap, batch, kvd, s);
        mark("attn.decode");
      } else {
        kernels::attention_prefill(ap, batch, kvd, s);
        mark("attn.prefill");
      }
      ++attn_i;
    } else {
      kernels::gdn_ab(xbf, m, H, W.ab, W.ab_rows, partials, eps, ab, s);
      mark("gdn.ab");
      kernels::GdnConvParams cp;
      cp.proj = proj;
      cp.conv_w = W.conv;
      cp.qkv = qkv;
      cp.gdn_layer = gdn_i;
      kernels::gdn_conv(cp, gd, batch, sd, s);
      mark("gdn.conv");
      kernels::GdnRecurrenceParams rp;
      rp.qkv = qkv;
      rp.ab = ab;
      rp.A = W.A;
      rp.dt_bias = W.dt_bias;
      rp.z = proj + cfg.gdn_conv_dim();
      rp.ld_z = proj_ld;
      rp.norm_w = W.gdn_norm;
      rp.eps = eps;
      rp.out = mixer;
      rp.gdn_layer = gdn_i;
      kernels::gdn_recurrence(rp, gd, batch, sd, s);
      mark("gdn.recurrence");
      ++gdn_i;
    }
    kernels::qgemm(mixer, m, W.out_proj, ep_res, gemm_ws, s);
    mark(W.attention ? "gemm.attn_o" : "gemm.gdn_out");

    kernels::GemmEpilogue ep_mlp;
    ep_mlp.kind = kernels::Epilogue::SwiGLU;
    ep_mlp.out = act;
    ep_mlp.ldo = cfg.intermediate_size;
    ep_mlp.partials_in = partials;
    ep_mlp.hidden = H;
    ep_mlp.eps = eps;
    kernels::qgemm(xbf, m, W.gate_up, ep_mlp, gemm_ws, s);
    mark("gemm.mlp_gate_up");
    kernels::qgemm(act, m, W.down, ep_res, gemm_ws, s);
    mark("gemm.mlp_down");
  }

  if (lm_rows > 0) {
    kernels::GemmEpilogue ep_head;
    ep_head.kind = kernels::Epilogue::StoreF32;
    ep_head.out = logits;
    ep_head.ldo = cfg.vocab_size;
    ep_head.partials_in = partials + size_t(lm_row) * cfg.norm_blocks();
    ep_head.hidden = H;
    ep_head.eps = eps;
    kernels::qgemm(xbf + size_t(lm_row) * H, lm_rows, weights->lm_head(), ep_head, gemm_ws, s);
    mark("gemm.lm_head");
  }
}

void ModelRunner::Impl::run_sampler(int rows) {
  kernels::SamplingArgs sa;
  sa.logits = logits;
  sa.ld = cfg.vocab_size;
  sa.vocab = cfg.vocab_size;
  sa.params = dev_meta<kernels::SamplingParamsDev>(meta.params);
  sa.row_slot = dev_meta<int>(meta.row_slot);
  sa.history = history;
  sa.history_len = history_len;
  sa.history_cap = opt.history_cap;
  sa.step = dev_meta<int>(meta.step);
  sa.tokens_out = tokens_out;
  sa.logprob_out = logprob_out;
  kernels::sample(sa, rows, stream.get());
  mark("sample");
}

ModelRunner::ModelRunner(const ModelFile& file, const RunnerOptions& options)
    : impl_(std::make_unique<Impl>(file, options)) {
  impl_->setup();
}

ModelRunner::~ModelRunner() = default;

const ModelConfig& ModelRunner::config() const { return impl_->cfg; }
const RunnerOptions& ModelRunner::options() const { return impl_->opt; }
const MemoryPlan& ModelRunner::memory() const { return impl_->plan; }
const DeviceInfo& ModelRunner::device() const { return impl_->dev; }
int ModelRunner::num_slots() const { return impl_->opt.max_batch; }
KvCache& ModelRunner::kv() { return *impl_->kv; }
const ModelWeights& ModelRunner::weights() const { return *impl_->weights; }
cudaStream_t ModelRunner::stream() const { return impl_->stream.get(); }
const kernels::GemmWorkspace& ModelRunner::gemm_workspace() const { return impl_->gemm_ws; }

void ModelRunner::reset_slot(int slot) {
  QWEN38_CHECK(slot >= 0 && slot < impl_->opt.max_batch, "reset_slot: slot out of range");
  impl_->state->reset(slot, impl_->stream.get());
  QWEN38_CUDA(cudaMemsetAsync(impl_->history_len + slot, 0, sizeof(int), impl_->stream.get()));
}

StepResult ModelRunner::prefill(int slot, int pos0, std::span<const int32_t> tokens, const SamplingParams* sample,
                                int step) {
  Impl& I = *impl_;
  const int n = int(tokens.size());
  QWEN38_CHECK(n > 0 && n <= I.opt.prefill_chunk, "prefill: chunk of ", n, " tokens (max ", I.opt.prefill_chunk, ")");
  QWEN38_CHECK(I.kv->pages_of(slot) * kernels::kPageTokens >= pos0 + n, "prefill: KV pages not reserved");
  std::memcpy(I.host_meta<int32_t>(I.meta.tokens), tokens.data(), size_t(n) * 4);
  for (int i = 0; i < n; ++i) {
    I.host_meta<int>(I.meta.tok_pos)[i] = pos0 + i;
    I.host_meta<int>(I.meta.tok_seq)[i] = 0;
  }
  I.host_meta<int>(I.meta.seq_start)[0] = 0;
  I.host_meta<int>(I.meta.seq_start)[1] = n;
  I.host_meta<int>(I.meta.seq_slot)[0] = slot;
  I.host_meta<int>(I.meta.seq_pos0)[0] = pos0;
  I.host_meta<int>(I.meta.row_slot)[0] = slot;
  I.host_meta<int>(I.meta.step)[0] = step;
  if (sample) I.host_meta<kernels::SamplingParamsDev>(I.meta.params)[0] = Impl::to_dev(*sample);
  I.kv->sync(I.stream.get());
  I.upload_meta();
  I.forward(n, I.batch_desc(1, n, n), /*decode=*/false, n - 1, sample ? 1 : 0);
  StepResult result;
  if (sample) {
    I.run_sampler(1);
    I.download_results(1);
  }
  I.stream.sync();
  I.collect_marks();
  if (sample) result = I.read_results(1);
  return result;
}

// Fill the step metadata for a decode batch padded to `bucket` rows. Padding rows (and, for
// graph warm-up, all rows) run a private dummy sequence at position 0 in the dummy slot.
static void fill_decode_meta(ModelRunner::Impl& I, std::span<const DecodeRow> rows, int bucket, bool all_dummy);

StepResult ModelRunner::decode(std::span<const DecodeRow> rows) {
  Impl& I = *impl_;
  const int n = int(rows.size());
  QWEN38_CHECK(n > 0 && n <= I.opt.max_batch, "decode: batch of ", n, " rows (max ", I.opt.max_batch, ")");
  for (const DecodeRow& r : rows)
    QWEN38_CHECK(I.kv->pages_of(r.slot) * kernels::kPageTokens > r.position, "decode: KV page missing");
  const bool use_graph = I.graphs_enabled && !I.profiling;
  const int bucket = use_graph ? bucket_for(n, I.opt.max_batch) : n;
  I.kv->sync(I.stream.get());
  const auto batch = I.batch_desc(bucket, bucket, 1);
  if (use_graph) {
    auto it = I.graphs.find(bucket);
    if (it == I.graphs.end()) {
      // Warm-up outside capture (kernel attributes, lazy module loading), on the dummy slot only
      // so that no real sequence's GDN state or penalty history advances.
      fill_decode_meta(I, rows, bucket, /*all_dummy=*/true);
      I.upload_meta();
      I.forward(bucket, batch, true, 0, bucket);
      I.run_sampler(bucket);
      I.stream.sync();
      Graph g;
      g.begin_capture(I.stream.get());
      I.upload_meta();  // captured as a memcpy node: copies whatever the pinned buffer holds at launch
      I.forward(bucket, batch, true, 0, bucket);
      I.run_sampler(bucket);
      I.download_results(bucket);
      g.end_capture(I.stream.get());
      it = I.graphs.emplace(bucket, std::move(g)).first;
    }
    fill_decode_meta(I, rows, bucket, false);
    it->second.launch(I.stream.get());
  } else {
    fill_decode_meta(I, rows, bucket, false);
    I.upload_meta();
    I.forward(bucket, batch, true, 0, bucket);
    I.run_sampler(bucket);
    I.download_results(bucket);
  }
  I.stream.sync();
  I.collect_marks();
  StepResult r = I.read_results(bucket);
  r.tokens.resize(n);
  r.logprobs.resize(n);
  return r;
}

static void fill_decode_meta(ModelRunner::Impl& I, std::span<const DecodeRow> rows, int bucket, bool all_dummy) {
  const int n = int(rows.size());
  auto* tok = I.host_meta<int32_t>(I.meta.tokens);
  auto* tok_pos = I.host_meta<int>(I.meta.tok_pos);
  auto* tok_seq = I.host_meta<int>(I.meta.tok_seq);
  auto* seq_start = I.host_meta<int>(I.meta.seq_start);
  auto* seq_slot = I.host_meta<int>(I.meta.seq_slot);
  auto* seq_pos0 = I.host_meta<int>(I.meta.seq_pos0);
  auto* row_slot = I.host_meta<int>(I.meta.row_slot);
  auto* step = I.host_meta<int>(I.meta.step);
  auto* params = I.host_meta<kernels::SamplingParamsDev>(I.meta.params);
  for (int i = 0; i < bucket; ++i) {
    const bool real = i < n && !all_dummy;
    const DecodeRow* r = real ? &rows[i] : nullptr;
    tok[i] = real ? r->token : 0;
    tok_pos[i] = real ? r->position : 0;
    tok_seq[i] = i;
    seq_start[i] = i;
    seq_slot[i] = real ? r->slot : I.dummy_slot;
    seq_pos0[i] = real ? r->position : 0;
    row_slot[i] = seq_slot[i];
    step[i] = real ? r->step : 0;
    params[i] = real ? ModelRunner::Impl::to_dev(r->sampling)
                     : kernels::SamplingParamsDev{0.0f, 0, 1.0f, 0.0f, 0.0f, 1.0f, 0};
  }
  seq_start[bucket] = bucket;
}

std::vector<float> ModelRunner::score(int slot, int pos0, std::span<const int32_t> tokens,
                                      std::span<const int32_t> targets) {
  Impl& I = *impl_;
  QWEN38_CHECK(tokens.size() == targets.size(), "score: tokens/targets size");
  std::vector<float> out;
  out.reserve(tokens.size());
  int pos = pos0;
  for (size_t off = 0; off < tokens.size(); off += size_t(I.opt.prefill_chunk)) {
    const int n = int(std::min<size_t>(size_t(I.opt.prefill_chunk), tokens.size() - off));
    QWEN38_CHECK(I.kv->ensure(slot, pos + n), "score: KV cache full");
    std::memcpy(I.host_meta<int32_t>(I.meta.tokens), tokens.data() + off, size_t(n) * 4);
    for (int i = 0; i < n; ++i) {
      I.host_meta<int>(I.meta.tok_pos)[i] = pos + i;
      I.host_meta<int>(I.meta.tok_seq)[i] = 0;
    }
    I.host_meta<int>(I.meta.seq_start)[0] = 0;
    I.host_meta<int>(I.meta.seq_start)[1] = n;
    I.host_meta<int>(I.meta.seq_slot)[0] = slot;
    I.host_meta<int>(I.meta.seq_pos0)[0] = pos;
    I.kv->sync(I.stream.get());
    I.upload_meta();
    QWEN38_CUDA(cudaMemcpyAsync(I.targets, targets.data() + off, size_t(n) * 4, cudaMemcpyHostToDevice,
                              I.stream.get()));
    I.forward(n, I.batch_desc(1, n, n), false, 0, 0);
    // LM head + log-softmax in row blocks that fit the logits buffer
    for (int r0 = 0; r0 < n; r0 += I.max_rows) {
      const int rows = std::min(I.max_rows, n - r0);
      kernels::GemmEpilogue ep;
      ep.kind = kernels::Epilogue::StoreF32;
      ep.out = I.logits;
      ep.ldo = I.cfg.vocab_size;
      ep.partials_in = I.partials + size_t(r0) * I.cfg.norm_blocks();
      ep.hidden = I.cfg.hidden_size;
      ep.eps = I.cfg.rms_norm_eps;
      kernels::qgemm(I.xbf + size_t(r0) * I.cfg.hidden_size, rows, I.weights->lm_head(), ep, I.gemm_ws,
                     I.stream.get());
      kernels::target_logprobs(I.logits, I.cfg.vocab_size, I.cfg.vocab_size, I.targets + r0, rows,
                               I.logprob_rows + r0, I.stream.get());
    }
    std::vector<float> chunk(static_cast<size_t>(n));
    QWEN38_CUDA(cudaMemcpyAsync(chunk.data(), I.logprob_rows, size_t(n) * 4, cudaMemcpyDeviceToHost, I.stream.get()));
    I.stream.sync();
    out.insert(out.end(), chunk.begin(), chunk.end());
    pos += n;
  }
  return out;
}

std::vector<float> ModelRunner::prefill_logits(int slot, int pos0, std::span<const int32_t> tokens) {
  Impl& I = *impl_;
  const int n = int(tokens.size());
  QWEN38_CHECK(n > 0 && n <= I.opt.prefill_chunk, "prefill_logits: chunk size");
  QWEN38_CHECK(I.kv->ensure(slot, pos0 + n), "prefill_logits: KV cache full");
  std::memcpy(I.host_meta<int32_t>(I.meta.tokens), tokens.data(), size_t(n) * 4);
  for (int i = 0; i < n; ++i) {
    I.host_meta<int>(I.meta.tok_pos)[i] = pos0 + i;
    I.host_meta<int>(I.meta.tok_seq)[i] = 0;
  }
  I.host_meta<int>(I.meta.seq_start)[0] = 0;
  I.host_meta<int>(I.meta.seq_start)[1] = n;
  I.host_meta<int>(I.meta.seq_slot)[0] = slot;
  I.host_meta<int>(I.meta.seq_pos0)[0] = pos0;
  I.kv->sync(I.stream.get());
  I.upload_meta();
  I.forward(n, I.batch_desc(1, n, n), false, 0, 0);
  const int V = I.cfg.vocab_size;
  std::vector<float> out(size_t(n) * V);
  for (int r0 = 0; r0 < n; r0 += I.max_rows) {
    const int rows = std::min(I.max_rows, n - r0);
    kernels::GemmEpilogue ep;
    ep.kind = kernels::Epilogue::StoreF32;
    ep.out = I.logits;
    ep.ldo = V;
    ep.partials_in = I.partials + size_t(r0) * I.cfg.norm_blocks();
    ep.hidden = I.cfg.hidden_size;
    ep.eps = I.cfg.rms_norm_eps;
    kernels::qgemm(I.xbf + size_t(r0) * I.cfg.hidden_size, rows, I.weights->lm_head(), ep, I.gemm_ws, I.stream.get());
    QWEN38_CUDA(cudaMemcpyAsync(out.data() + size_t(r0) * V, I.logits, size_t(rows) * V * 4, cudaMemcpyDeviceToHost,
                              I.stream.get()));
    I.stream.sync();
  }
  return out;
}

void ModelRunner::set_profiling(bool on) { impl_->profiling = on; }
const OpProfile& ModelRunner::profile() const { return impl_->prof; }
void ModelRunner::clear_profile() { impl_->prof = OpProfile{}; }
void ModelRunner::set_graphs_enabled(bool on) { impl_->graphs_enabled = on; }

}  // namespace qwen38
