// Standalone byte-exact equivalence + timing harness for lib/jxl/enc_cluster.cc.
//
// Proves OLD (upstream @ main 00f4d7fc) vs NEW (branch
// perf/enc-cluster-fuse-reindex-jun30-v8n3) produce IDENTICAL clustering
// results — same output histograms (counts + total) and same context map
// (histogram_symbols) — across:
//   - kBest (full priority-queue merge refinement), kFast, kFastest
//   - prev_histograms == 0 and prev_histograms > 0 (incremental) paths
//   - many randomized histogram sets of varying count, alphabet, overlap
// then times the two end-to-end pipelines interleaved (start-rotated to cancel
// thermal drift).
//
// No libjxl build needed: the algorithm is mirrored verbatim from source. The
// SIMD entropy/distance/KL kernels are mirrored with an explicit kRounding-lane
// partial-sum model so the float accumulation order matches the real Highway
// code; OLD and NEW share one MockANSPopulationCost / Entropy so the test
// isolates the *structural* refactor (scratch reuse, union-find, swap-based
// reindex, fused add+entropy, common/tail split) — exactly what must stay
// byte-exact. The mock cost is engineered to yield a realistic mix of
// accepted/rejected merges so the refinement loop is genuinely exercised.
//
// Build: clang++ -std=c++17 -O2 tools/enc_cluster_ab.cc -o enc_cluster_ab

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>
#include <cstdlib>
#include <new>
#include <queue>
#include <tuple>
#include <vector>

using std::size_t;
using ANSHistBin = int32_t;

// Deterministic allocation counter (noise-free proxy for the memory/efficiency
// win: scratch reuse + swap-based reindex remove heap traffic regardless of
// machine timing variance).
static size_t g_allocs = 0;
static bool g_count = false;
void* operator new(std::size_t n) {
  if (g_count) ++g_allocs;
  void* p = std::malloc(n ? n : 1);
  if (!p) throw std::bad_alloc();
  return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

static constexpr size_t kRounding = 8;
static constexpr float kInfinity = std::numeric_limits<float>::infinity();

static size_t DivCeil(size_t a, size_t b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Mock Histogram (mirror of lib/jxl/enc_ans_params.h Histogram essentials).
struct Histogram {
  std::vector<ANSHistBin> counts;
  size_t total_count = 0;
  mutable float entropy = 0.0f;

  void Add(size_t symbol) {
    if (counts.size() <= symbol) {
      counts.resize(DivCeil(symbol + 1, kRounding) * kRounding);
    }
    ++counts[symbol];
    ++total_count;
  }
  void AddHistogram(const Histogram& other) {
    if (other.counts.size() > counts.size()) counts.resize(other.counts.size());
    for (size_t i = 0; i < other.counts.size(); ++i) counts[i] += other.counts[i];
    total_count += other.total_count;
  }
  size_t alphabet_size() const {
    for (int i = static_cast<int>(counts.size()) - 1; i >= 0; --i) {
      if (counts[i] > 0) return i + 1;
    }
    return 0;
  }
  void swap(Histogram& o) {
    counts.swap(o.counts);
    std::swap(total_count, o.total_count);
    std::swap(entropy, o.entropy);
  }
  // Deterministic pure stand-in for ANSPopulationCost: a table term that grows
  // with alphabet size (so merging overlapping supports can save bits) plus a
  // Shannon data term. Identical for OLD and NEW, so byte-exactness of the
  // refactor is what is under test. Returns a float exactly as the real one.
  float ANSPopulationCost() const {
    if (counts.size() > 1u << 16) return std::numeric_limits<float>::max();
    if (total_count == 0) return 0.0f;
    const float inv = 1.0f / static_cast<float>(total_count);
    float data = 0.0f;
    size_t alpha = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
      const ANSHistBin c = counts[i];
      if (c > 0) {
        data += static_cast<float>(c) * -std::log2(static_cast<float>(c) * inv);
        alpha = i + 1;
      }
    }
    const float table = 12.0f * static_cast<float>(alpha) + 2.0f;
    return data + table;
  }
};

// kRounding-lane partial-sum entropy term, mirroring Entropy() under SIMD.
// The real kernel uses Highway FastLog2f, which is finite at 0, so a zero count
// contributes -0*FastLog2f(0) == 0; std::log2(0) is -inf and 0*-inf is NaN, so
// guard count==0 to reproduce the real (NaN-free) zero contribution.
static inline float EntropyTerm(float count, float inv_total, float total) {
  if (count == total) return 0.0f;
  if (count == 0.0f) return 0.0f;
  return 0.0f - count * std::log2(inv_total * count);
}

// ===========================================================================
// OLD — verbatim mirror of upstream enc_cluster.cc @ main.
namespace old_impl {

void HistogramEntropy(const Histogram& a) {
  a.entropy = 0.0f;
  if (a.total_count == 0) return;
  const float inv_tot = 1.0f / static_cast<float>(a.total_count);
  const float total = static_cast<float>(a.total_count);
  float lanes[kRounding] = {0};
  for (size_t i = 0; i < a.counts.size(); i += kRounding)
    for (size_t l = 0; l < kRounding; ++l)
      lanes[l] += EntropyTerm(static_cast<float>(a.counts[i + l]), inv_tot, total);
  float s = 0;
  for (float v : lanes) s += v;
  a.entropy += s;
}

float HistogramDistance(const Histogram& a, const Histogram& b) {
  if (a.total_count == 0 || b.total_count == 0) return 0;
  const float inv_tot = 1.0f / static_cast<float>(a.total_count + b.total_count);
  const float total = static_cast<float>(a.total_count + b.total_count);
  float lanes[kRounding] = {0};
  for (size_t i = 0; i < std::max(a.counts.size(), b.counts.size()); i += kRounding)
    for (size_t l = 0; l < kRounding; ++l) {
      const float av = (a.counts.size() > i + l) ? static_cast<float>(a.counts[i + l]) : 0.0f;
      const float bv = (b.counts.size() > i + l) ? static_cast<float>(b.counts[i + l]) : 0.0f;
      lanes[l] += EntropyTerm(av + bv, inv_tot, total);
    }
  float s = 0;
  for (float v : lanes) s += v;
  return s - a.entropy - b.entropy;
}

float HistogramKLDivergence(const Histogram& actual, const Histogram& coding) {
  if (actual.total_count == 0) return 0;
  if (coding.total_count == 0) return kInfinity;
  const float coding_inv = 1.0f / static_cast<float>(coding.total_count);
  float lanes[kRounding] = {0};
  for (size_t i = 0; i < actual.counts.size(); i += kRounding)
    for (size_t l = 0; l < kRounding; ++l) {
      const float c = static_cast<float>(actual.counts[i + l]);
      const float cc = (coding.counts.size() > i + l) ? static_cast<float>(coding.counts[i + l]) : 0.0f;
      float neg_cost;
      if (c == 0.0f) {
        neg_cost = 0.0f;  // IfThenZeroElse
      } else if (cc == 0.0f) {
        neg_cost = -kInfinity;
      } else {
        neg_cost = std::log2(cc * coding_inv);
      }
      lanes[l] += -(c * neg_cost);  // NegMulAdd accumulates -(c*neg_cost)
    }
  float s = 0;
  for (float v : lanes) s += v;
  return s - actual.entropy;
}

bool FastClusterHistograms(const std::vector<Histogram>& in, size_t max_histograms,
                           std::vector<Histogram>* out,
                           std::vector<uint32_t>* histogram_symbols) {
  const size_t prev_histograms = out->size();
  histogram_symbols->clear();
  histogram_symbols->resize(in.size(), max_histograms);
  std::vector<float> dists(in.size(), std::numeric_limits<float>::max());
  size_t largest_idx = 0;
  for (size_t i = 0; i < in.size(); i++) {
    if (in[i].total_count == 0) {
      (*histogram_symbols)[i] = 0;
      dists[i] = 0.0f;
      continue;
    }
    HistogramEntropy(in[i]);
    if (in[i].total_count > in[largest_idx].total_count) largest_idx = i;
  }
  if (prev_histograms > 0) {
    for (size_t j = 0; j < prev_histograms; ++j) HistogramEntropy((*out)[j]);
    for (size_t i = 0; i < in.size(); i++) {
      if (dists[i] == 0.0f) continue;
      for (size_t j = 0; j < prev_histograms; ++j)
        dists[i] = std::min(HistogramKLDivergence(in[i], (*out)[j]), dists[i]);
    }
    auto max_dist = std::max_element(dists.begin(), dists.end());
    if (*max_dist > 0.0f) largest_idx = max_dist - dists.begin();
  }
  constexpr float kMinDistanceForDistinct = 48.0f;
  while (out->size() < max_histograms) {
    (*histogram_symbols)[largest_idx] = out->size();
    out->push_back(in[largest_idx]);
    dists[largest_idx] = 0.0f;
    largest_idx = 0;
    for (size_t i = 0; i < in.size(); i++) {
      if (dists[i] == 0.0f) continue;
      dists[i] = std::min(HistogramDistance(in[i], out->back()), dists[i]);
      if (dists[i] > dists[largest_idx]) largest_idx = i;
    }
    if (dists[largest_idx] < kMinDistanceForDistinct) break;
  }
  for (size_t i = 0; i < in.size(); i++) {
    if ((*histogram_symbols)[i] != max_histograms) continue;
    size_t best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (size_t j = 0; j < out->size(); j++) {
      float dist = j < prev_histograms ? HistogramKLDivergence(in[i], (*out)[j])
                                       : HistogramDistance(in[i], (*out)[j]);
      if (dist < best_dist) { best = j; best_dist = dist; }
    }
    if (best >= prev_histograms) {
      (*out)[best].AddHistogram(in[i]);
      HistogramEntropy((*out)[best]);
    }
    (*histogram_symbols)[i] = best;
  }
  return true;
}

void HistogramReindex(std::vector<Histogram>* out, size_t prev_histograms,
                      std::vector<uint32_t>* symbols) {
  std::vector<Histogram> tmp(*out);
  std::map<int, int> new_index;
  for (size_t i = 0; i < prev_histograms; ++i) new_index[i] = i;
  int next_index = prev_histograms;
  for (uint32_t symbol : *symbols) {
    if (new_index.find(symbol) == new_index.end()) {
      new_index[symbol] = next_index;
      (*out)[next_index] = tmp[symbol];
      ++next_index;
    }
  }
  out->resize(next_index);
  for (uint32_t& symbol : *symbols) symbol = new_index[symbol];
}

enum class Clustering { kBest, kFast, kFastest };

bool ClusterHistograms(Clustering clustering, const std::vector<Histogram>& in,
                       size_t max_histograms, std::vector<Histogram>* out,
                       std::vector<uint32_t>* histogram_symbols) {
  size_t prev_histograms = out->size();
  max_histograms = std::min(max_histograms, in.size());
  if (clustering == Clustering::kFastest)
    max_histograms = std::min(max_histograms, static_cast<size_t>(4));
  FastClusterHistograms(in, prev_histograms + max_histograms, out, histogram_symbols);
  if (prev_histograms == 0 && clustering == Clustering::kBest) {
    for (auto& histo : *out) histo.entropy = histo.ANSPopulationCost();
    uint32_t next_version = 2;
    std::vector<uint32_t> version(out->size(), 1);
    std::vector<uint32_t> renumbering(out->size());
    std::iota(renumbering.begin(), renumbering.end(), 0);
    struct HistogramPair {
      float cost; uint32_t first; uint32_t second; uint32_t version;
      bool operator<(const HistogramPair& o) const {
        return std::make_tuple(cost, first, second, version) >
               std::make_tuple(o.cost, o.first, o.second, o.version);
      }
    };
    std::priority_queue<HistogramPair> pairs_to_merge;
    for (uint32_t i = 0; i < out->size(); i++)
      for (uint32_t j = i + 1; j < out->size(); j++) {
        Histogram histo;
        histo.AddHistogram((*out)[i]);
        histo.AddHistogram((*out)[j]);
        float cost = histo.ANSPopulationCost();
        cost -= (*out)[i].entropy + (*out)[j].entropy;
        if (cost >= 0) continue;
        pairs_to_merge.push(HistogramPair{cost, i, j, std::max(version[i], version[j])});
      }
    while (!pairs_to_merge.empty()) {
      uint32_t first = pairs_to_merge.top().first;
      uint32_t second = pairs_to_merge.top().second;
      uint32_t ver = pairs_to_merge.top().version;
      pairs_to_merge.pop();
      if (ver != std::max(version[first], version[second]) ||
          version[first] == 0 || version[second] == 0)
        continue;
      (*out)[first].AddHistogram((*out)[second]);
      (*out)[first].entropy = (*out)[first].ANSPopulationCost();
      for (uint32_t& item : renumbering) if (item == second) item = first;
      version[second] = 0;
      version[first] = next_version++;
      for (uint32_t j = 0; j < out->size(); j++) {
        if (j == first) continue;
        if (version[j] == 0) continue;
        Histogram histo;
        histo.AddHistogram((*out)[first]);
        histo.AddHistogram((*out)[j]);
        float merge_cost = histo.ANSPopulationCost();
        merge_cost -= (*out)[first].entropy + (*out)[j].entropy;
        if (merge_cost >= 0) continue;
        pairs_to_merge.push(HistogramPair{merge_cost, std::min(first, j),
                                          std::max(first, j),
                                          std::max(version[first], version[j])});
      }
    }
    std::vector<uint32_t> reverse_renumbering(out->size(), static_cast<uint32_t>(-1));
    size_t num_alive = 0;
    for (size_t i = 0; i < out->size(); i++) {
      if (version[i] == 0) continue;
      (*out)[num_alive++] = (*out)[i];
      reverse_renumbering[i] = num_alive - 1;
    }
    out->resize(num_alive);
    for (uint32_t& item : *histogram_symbols)
      item = reverse_renumbering[renumbering[item]];
  }
  HistogramReindex(out, prev_histograms, histogram_symbols);
  return true;
}

}  // namespace old_impl

// ===========================================================================
// NEW — verbatim mirror of branch enc_cluster.cc.
namespace new_impl {

void HistogramEntropy(const Histogram& a) {
  a.entropy = 0.0f;
  if (a.total_count == 0) return;
  const float inv_tot = 1.0f / static_cast<float>(a.total_count);
  const float total = static_cast<float>(a.total_count);
  float lanes[kRounding] = {0};
  for (size_t i = 0; i < a.counts.size(); i += kRounding)
    for (size_t l = 0; l < kRounding; ++l)
      lanes[l] += EntropyTerm(static_cast<float>(a.counts[i + l]), inv_tot, total);
  float s = 0;
  for (float v : lanes) s += v;
  a.entropy += s;
}

// Fused add + entropy (mirror of HistogramAddAndEntropy).
void HistogramAddAndEntropy(Histogram& dst, const Histogram& src) {
  const size_t dst_size = dst.counts.size();
  const size_t src_size = src.counts.size();
  if (src_size > dst_size) dst.counts.resize(src_size);
  dst.total_count += src.total_count;
  dst.entropy = 0.0f;
  if (dst.total_count == 0) return;
  const float inv_tot = 1.0f / static_cast<float>(dst.total_count);
  const float total = static_cast<float>(dst.total_count);
  float lanes[kRounding] = {0};
  const size_t common = std::min(dst_size, src_size);
  const size_t grown = std::max(dst_size, src_size);
  size_t i = 0;
  for (; i < common; i += kRounding)
    for (size_t l = 0; l < kRounding; ++l) {
      const ANSHistBin v = dst.counts[i + l] + src.counts[i + l];
      dst.counts[i + l] = v;
      lanes[l] += EntropyTerm(static_cast<float>(v), inv_tot, total);
    }
  const Histogram& longer = (src_size > dst_size) ? src : dst;
  for (; i < grown; i += kRounding)
    for (size_t l = 0; l < kRounding; ++l) {
      const ANSHistBin v = longer.counts[i + l];
      if (src_size > dst_size) dst.counts[i + l] = v;
      lanes[l] += EntropyTerm(static_cast<float>(v), inv_tot, total);
    }
  float s = 0;
  for (float v : lanes) s += v;
  dst.entropy = s;
}

float HistogramDistance(const Histogram& a, const Histogram& b) {
  if (a.total_count == 0 || b.total_count == 0) return 0;
  const float inv_tot = 1.0f / static_cast<float>(a.total_count + b.total_count);
  const float total = static_cast<float>(a.total_count + b.total_count);
  float lanes[kRounding] = {0};
  const size_t a_size = a.counts.size();
  const size_t b_size = b.counts.size();
  const size_t common = std::min(a_size, b_size);
  size_t i = 0;
  for (; i < common; i += kRounding)
    for (size_t l = 0; l < kRounding; ++l)
      lanes[l] += EntropyTerm(static_cast<float>(a.counts[i + l] + b.counts[i + l]),
                              inv_tot, total);
  const Histogram& longer = a_size > b_size ? a : b;
  for (; i < longer.counts.size(); i += kRounding)
    for (size_t l = 0; l < kRounding; ++l)
      lanes[l] += EntropyTerm(static_cast<float>(longer.counts[i + l]), inv_tot, total);
  float s = 0;
  for (float v : lanes) s += v;
  return s - a.entropy - b.entropy;
}

float HistogramKLDivergence(const Histogram& actual, const Histogram& coding) {
  if (actual.total_count == 0) return 0;
  if (coding.total_count == 0) return kInfinity;
  const float coding_inv = 1.0f / static_cast<float>(coding.total_count);
  float lanes[kRounding] = {0};
  const size_t common = std::min(actual.counts.size(), coding.counts.size());
  for (size_t i = common; i < actual.counts.size(); i += kRounding)
    for (size_t l = 0; l < kRounding; ++l)
      if (actual.counts[i + l] != 0) return kInfinity;
  for (size_t i = 0; i < common; i += kRounding)
    for (size_t l = 0; l < kRounding; ++l) {
      const float c = static_cast<float>(actual.counts[i + l]);
      const float cc = static_cast<float>(coding.counts[i + l]);
      float neg_cost;
      if (c == 0.0f) neg_cost = 0.0f;
      else if (cc == 0.0f) neg_cost = -kInfinity;
      else neg_cost = std::log2(cc * coding_inv);
      lanes[l] += -(c * neg_cost);
    }
  float s = 0;
  for (float v : lanes) s += v;
  return s - actual.entropy;
}

bool FastClusterHistograms(const std::vector<Histogram>& in, size_t max_histograms,
                           std::vector<Histogram>* out,
                           std::vector<uint32_t>* histogram_symbols) {
  const size_t prev_histograms = out->size();
  histogram_symbols->clear();
  histogram_symbols->resize(in.size(), max_histograms);
  std::vector<float> dists(in.size(), std::numeric_limits<float>::max());
  size_t largest_idx = 0;
  for (size_t i = 0; i < in.size(); i++) {
    if (in[i].total_count == 0) {
      (*histogram_symbols)[i] = 0;
      dists[i] = 0.0f;
      continue;
    }
    HistogramEntropy(in[i]);
    if (in[i].total_count > in[largest_idx].total_count) largest_idx = i;
  }
  if (prev_histograms > 0) {
    for (size_t j = 0; j < prev_histograms; ++j) HistogramEntropy((*out)[j]);
    for (size_t i = 0; i < in.size(); i++) {
      if (dists[i] == 0.0f) continue;
      for (size_t j = 0; j < prev_histograms; ++j)
        dists[i] = std::min(HistogramKLDivergence(in[i], (*out)[j]), dists[i]);
    }
    if (!dists.empty()) {
      auto max_dist = std::max_element(dists.begin(), dists.end());
      if (*max_dist > 0.0f) largest_idx = max_dist - dists.begin();
    }
  }
  constexpr float kMinDistanceForDistinct = 48.0f;
  while (out->size() < max_histograms) {
    (*histogram_symbols)[largest_idx] = out->size();
    out->push_back(in[largest_idx]);
    dists[largest_idx] = 0.0f;
    largest_idx = 0;
    for (size_t i = 0; i < in.size(); i++) {
      if (dists[i] == 0.0f) continue;
      dists[i] = std::min(HistogramDistance(in[i], out->back()), dists[i]);
      if (dists[i] > dists[largest_idx]) largest_idx = i;
    }
    if (dists[largest_idx] < kMinDistanceForDistinct) break;
  }
  for (size_t i = 0; i < in.size(); i++) {
    if ((*histogram_symbols)[i] != max_histograms) continue;
    size_t best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (size_t j = 0; j < out->size(); j++) {
      float dist = j < prev_histograms ? HistogramKLDivergence(in[i], (*out)[j])
                                       : HistogramDistance(in[i], (*out)[j]);
      if (dist < best_dist) { best = j; best_dist = dist; }
    }
    if (best >= prev_histograms) HistogramAddAndEntropy((*out)[best], in[i]);
    (*histogram_symbols)[i] = best;
  }
  return true;
}

constexpr uint32_t kInvalidHistogram = std::numeric_limits<uint32_t>::max();

void HistogramReindex(std::vector<Histogram>* out, size_t prev_histograms,
                      std::vector<uint32_t>* symbols) {
  const size_t num_histograms = out->size();
  std::vector<uint32_t> new_index(num_histograms, kInvalidHistogram);
  for (size_t i = 0; i < prev_histograms; ++i) new_index[i] = static_cast<uint32_t>(i);
  size_t next_index = prev_histograms;
  bool is_identity = true;
  for (uint32_t& symbol : *symbols) {
    uint32_t mapped = new_index[symbol];
    if (mapped == kInvalidHistogram) {
      mapped = static_cast<uint32_t>(next_index);
      new_index[symbol] = mapped;
      is_identity &= (symbol == next_index);
      ++next_index;
    }
    symbol = mapped;
  }
  if (!is_identity) {
    size_t spare = next_index;
    for (size_t i = 0; i < num_histograms; ++i)
      if (new_index[i] == kInvalidHistogram) new_index[i] = static_cast<uint32_t>(spare++);
    for (size_t i = 0; i < num_histograms; ++i)
      while (new_index[i] != i) {
        const size_t target = new_index[i];
        (*out)[i].swap((*out)[target]);
        std::swap(new_index[i], new_index[target]);
      }
  }
  out->resize(next_index);
}

uint32_t FindHistogramRepresentative(std::vector<uint32_t>* parent, uint32_t index) {
  while ((*parent)[index] != index) {
    (*parent)[index] = (*parent)[(*parent)[index]];
    index = (*parent)[index];
  }
  return index;
}

float HistogramMergeCost(const Histogram& first, const Histogram& second, Histogram* merged) {
  const size_t first_size = first.counts.size();
  const size_t second_size = second.counts.size();
  const size_t common = std::min(first_size, second_size);
  merged->counts.resize(std::max(first_size, second_size));
  for (size_t i = 0; i < common; ++i) merged->counts[i] = first.counts[i] + second.counts[i];
  if (first_size > second_size)
    std::copy(first.counts.begin() + common, first.counts.end(), merged->counts.begin() + common);
  else
    std::copy(second.counts.begin() + common, second.counts.end(), merged->counts.begin() + common);
  merged->total_count = first.total_count + second.total_count;
  float cost = merged->ANSPopulationCost();
  return cost - first.entropy - second.entropy;
}

enum class Clustering { kBest, kFast, kFastest };

bool ClusterHistograms(Clustering clustering, const std::vector<Histogram>& in,
                       size_t max_histograms, std::vector<Histogram>* out,
                       std::vector<uint32_t>* histogram_symbols) {
  if (in.empty()) { histogram_symbols->clear(); return true; }
  size_t prev_histograms = out->size();
  max_histograms = std::min(max_histograms, in.size());
  if (clustering == Clustering::kFastest)
    max_histograms = std::min(max_histograms, static_cast<size_t>(4));
  FastClusterHistograms(in, prev_histograms + max_histograms, out, histogram_symbols);
  if (prev_histograms == 0 && clustering == Clustering::kBest) {
    for (auto& histo : *out) histo.entropy = histo.ANSPopulationCost();
    uint32_t next_version = 2;
    std::vector<uint32_t> version(out->size(), 1);
    std::vector<uint32_t> parent(out->size());
    std::iota(parent.begin(), parent.end(), 0);
    Histogram merged;
    struct HistogramPair {
      float cost; uint32_t first; uint32_t second; uint32_t version;
      bool operator<(const HistogramPair& o) const {
        return std::make_tuple(cost, first, second, version) >
               std::make_tuple(o.cost, o.first, o.second, o.version);
      }
    };
    std::priority_queue<HistogramPair> pairs_to_merge;
    for (uint32_t i = 0; i < out->size(); i++)
      for (uint32_t j = i + 1; j < out->size(); j++) {
        float cost = HistogramMergeCost((*out)[i], (*out)[j], &merged);
        if (cost >= 0) continue;
        pairs_to_merge.push(HistogramPair{cost, i, j, std::max(version[i], version[j])});
      }
    while (!pairs_to_merge.empty()) {
      uint32_t first = pairs_to_merge.top().first;
      uint32_t second = pairs_to_merge.top().second;
      uint32_t ver = pairs_to_merge.top().version;
      pairs_to_merge.pop();
      if (ver != std::max(version[first], version[second]) ||
          version[first] == 0 || version[second] == 0)
        continue;
      (*out)[first].AddHistogram((*out)[second]);
      (*out)[first].entropy = (*out)[first].ANSPopulationCost();
      parent[second] = first;
      version[second] = 0;
      version[first] = next_version++;
      for (uint32_t j = 0; j < out->size(); j++) {
        if (j == first) continue;
        if (version[j] == 0) continue;
        float merge_cost = HistogramMergeCost((*out)[first], (*out)[j], &merged);
        if (merge_cost >= 0) continue;
        pairs_to_merge.push(HistogramPair{merge_cost, std::min(first, j),
                                          std::max(first, j),
                                          std::max(version[first], version[j])});
      }
    }
    for (uint32_t& item : *histogram_symbols)
      item = FindHistogramRepresentative(&parent, item);
  }
  HistogramReindex(out, prev_histograms, histogram_symbols);
  return true;
}

}  // namespace new_impl

// ===========================================================================
// Test driver.
static uint32_t XorShift(uint32_t& s) {
  s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}

// Build a random histogram set. Clusters drawn from a few "archetypes" so that
// some histograms genuinely overlap (driving merges) and others do not.
static std::vector<Histogram> MakeSet(uint32_t& s, size_t n, size_t alphabet) {
  const size_t n_arch = 2 + XorShift(s) % 5;
  std::vector<std::vector<int>> arch(n_arch);
  for (auto& a : arch) {
    a.assign(alphabet, 0);
    const size_t support = 1 + XorShift(s) % alphabet;
    for (size_t k = 0; k < support; ++k) a[XorShift(s) % alphabet] = 1 + XorShift(s) % 50;
  }
  std::vector<Histogram> in(n);
  for (size_t i = 0; i < n; ++i) {
    const auto& base = arch[XorShift(s) % n_arch];
    const int reps = 1 + XorShift(s) % 8;
    bool empty = (XorShift(s) % 16) == 0;  // exercise total_count==0 path
    if (!empty) {
      for (size_t sym = 0; sym < alphabet; ++sym)
        for (int r = 0; r < base[sym] * reps; ++r) in[i].Add(sym);
      // a little idiosyncratic noise so archetypes are not identical
      for (int k = 0; k < static_cast<int>(XorShift(s) % 4); ++k)
        in[i].Add(XorShift(s) % alphabet);
    }
  }
  return in;
}

struct Result {
  std::vector<uint32_t> symbols;
  std::vector<Histogram> out;
};

static bool SameHist(const Histogram& a, const Histogram& b) {
  if (a.total_count != b.total_count) return false;
  // Compare meaningful prefix (alphabet); trailing kRounding padding is zero.
  const size_t n = std::max(a.alphabet_size(), b.alphabet_size());
  for (size_t i = 0; i < n; ++i) {
    const ANSHistBin av = i < a.counts.size() ? a.counts[i] : 0;
    const ANSHistBin bv = i < b.counts.size() ? b.counts[i] : 0;
    if (av != bv) return false;
  }
  return true;
}

int main() {
  uint32_t seed = 0xC0FFEEu;
  const char* names[3] = {"kBest", "kFast", "kFastest"};

  // ---- (0) direct kernel fuzz: KL and Distance OLD vs NEW ----
  {
    uint32_t ks = 0xBEEF1u;
    size_t kl_bad = 0, dist_bad = 0, ent_bad = 0, n = 0;
    for (int t = 0; t < 200000; ++t) {
      const size_t alpha = kRounding * (1 + ks % 6);
      Histogram a, b;
      ks = XorShift(ks);
      const size_t sa = ks % alpha, sb = XorShift(ks) % alpha;
      for (size_t k = 0; k < sa; ++k) a.Add(XorShift(ks) % alpha);
      for (size_t k = 0; k < sb; ++k) b.Add(XorShift(ks) % alpha);
      old_impl::HistogramEntropy(a); float ea_o = a.entropy;
      new_impl::HistogramEntropy(a); float ea_n = a.entropy;
      if (ea_o != ea_n && !(std::isnan(ea_o) && std::isnan(ea_n))) ent_bad++;
      old_impl::HistogramEntropy(a); old_impl::HistogramEntropy(b);
      float klo = old_impl::HistogramKLDivergence(a, b);
      float kln = new_impl::HistogramKLDivergence(a, b);
      if (klo != kln && !(std::isinf(klo) && std::isinf(kln))) kl_bad++;
      float dso = old_impl::HistogramDistance(a, b);
      float dsn = new_impl::HistogramDistance(a, b);
      if (dso != dsn) dist_bad++;
      n++;
    }
    printf("kernel fuzz n=%zu: entropy_bad=%zu KL_bad=%zu dist_bad=%zu\n",
           n, ent_bad, kl_bad, dist_bad);
  }

  size_t cases = 0, fails = 0;
  size_t total_merges_exercised = 0;

  for (int mode = 0; mode < 3; ++mode) {
    for (int trial = 0; trial < 2000; ++trial) {
      const size_t n = 1 + XorShift(seed) % 40;
      const size_t alphabet = kRounding * (1 + XorShift(seed) % 8);
      const size_t max_h = 1 + XorShift(seed) % n;
      const bool with_prev = (XorShift(seed) % 3) == 0;

      std::vector<Histogram> in = MakeSet(seed, n, alphabet);

      // Optional pre-existing histograms (incremental path; not for kBest refine,
      // which only runs when prev==0 — matching source).
      std::vector<Histogram> out_init;
      if (with_prev) {
        const size_t p = 1 + XorShift(seed) % 3;
        std::vector<Histogram> pv = MakeSet(seed, p, alphabet);
        for (auto& h : pv) if (h.total_count > 0) out_init.push_back(h);
      }

      old_impl::Clustering om = static_cast<old_impl::Clustering>(mode);
      new_impl::Clustering nm = static_cast<new_impl::Clustering>(mode);

      Result ro, rn;
      ro.out = out_init;
      rn.out = out_init;
      old_impl::ClusterHistograms(om, in, max_h, &ro.out, &ro.symbols);
      new_impl::ClusterHistograms(nm, in, max_h, &rn.out, &rn.symbols);

      ++cases;
      bool ok = (ro.symbols == rn.symbols) && (ro.out.size() == rn.out.size());
      if (ok) {
        for (size_t i = 0; i < ro.out.size(); ++i)
          if (!SameHist(ro.out[i], rn.out[i])) { ok = false; break; }
      }
      if (!ok) {
        if (fails == 0) {
          printf("FAIL mode=%s trial=%d n=%zu alpha=%zu max_h=%zu prev=%zu :"
                 " symbols=%s sizes %zu/%zu\n",
                 names[mode], trial, n, alphabet, max_h, out_init.size(),
                 (ro.symbols == rn.symbols) ? "match" : "DIFFER",
                 ro.out.size(), rn.out.size());
          printf("  OLD sym:");
          for (uint32_t v : ro.symbols) printf(" %u", v);
          printf("\n  NEW sym:");
          for (uint32_t v : rn.symbols) printf(" %u", v);
          printf("\n");
          // Isolate: FastCluster only (pre-reindex).
          std::vector<Histogram> fo = out_init, fn = out_init;
          std::vector<uint32_t> so, sn;
          old_impl::FastClusterHistograms(in, out_init.size() + std::min(max_h, in.size()), &fo, &so);
          new_impl::FastClusterHistograms(in, out_init.size() + std::min(max_h, in.size()), &fn, &sn);
          printf("  FastCluster syms %s; out sizes %zu/%zu\n",
                 (so == sn) ? "match" : "DIFFER", fo.size(), fn.size());
          printf("  FC OLD:");
          for (uint32_t v : so) printf(" %u", v);
          printf("\n  FC NEW:");
          for (uint32_t v : sn) printf(" %u", v);
          printf("\n");
        }
        ++fails;
      }
      if (mode == 0) total_merges_exercised += (n > ro.out.size() ? n - ro.out.size() : 0);
    }
  }
  printf("equivalence: cases=%zu fails=%zu -> %s  (kBest merge activity ~%zu)\n",
         cases, fails, fails == 0 ? "ALL BYTE-EXACT" : "MISMATCH",
         total_merges_exercised);

  // ---- timing: end-to-end kBest pipeline, interleaved + start-rotated ----
  std::vector<std::vector<Histogram>> corpus;
  std::vector<size_t> maxhs;
  uint32_t ts = 0x1234567u;
  for (int i = 0; i < 200; ++i) {
    const size_t n = 8 + XorShift(ts) % 56;
    const size_t alphabet = kRounding * (2 + XorShift(ts) % 10);
    corpus.push_back(MakeSet(ts, n, alphabet));
    maxhs.push_back(1 + XorShift(ts) % n);
  }

  // ---- (D) deterministic allocation count over the whole corpus ----
  auto count_allocs = [&](bool use_new) {
    g_allocs = 0;
    g_count = true;
    for (size_t i = 0; i < corpus.size(); ++i) {
      std::vector<Histogram> out;
      std::vector<uint32_t> syms;
      if (use_new)
        new_impl::ClusterHistograms(new_impl::Clustering::kBest, corpus[i], maxhs[i], &out, &syms);
      else
        old_impl::ClusterHistograms(old_impl::Clustering::kBest, corpus[i], maxhs[i], &out, &syms);
    }
    g_count = false;
    return g_allocs;
  };
  const size_t alloc_old = count_allocs(false);
  const size_t alloc_new = count_allocs(true);
  printf("kBest pipeline heap allocations: OLD=%zu NEW=%zu  (-%.1f%%)\n",
         alloc_old, alloc_new,
         100.0 * (double)(alloc_old - alloc_new) / (double)alloc_old);
  const int kReps = 8, kPasses = 8;
  volatile double sink = 0;
  double t_old = 0, t_new = 0;
  for (int pass = 0; pass < kPasses; ++pass) {
    bool old_first = (pass & 1) == 0;
    for (int side = 0; side < 2; ++side) {
      bool do_old = (side == 0) == old_first;
      auto a0 = std::chrono::high_resolution_clock::now();
      for (int r = 0; r < kReps; ++r)
        for (size_t i = 0; i < corpus.size(); ++i) {
          std::vector<Histogram> out;
          std::vector<uint32_t> syms;
          if (do_old)
            old_impl::ClusterHistograms(old_impl::Clustering::kBest, corpus[i], maxhs[i], &out, &syms);
          else
            new_impl::ClusterHistograms(new_impl::Clustering::kBest, corpus[i], maxhs[i], &out, &syms);
          sink += out.size() + syms.size();
        }
      auto a1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(a1 - a0).count();
      if (do_old) t_old += ms; else t_new += ms;
    }
  }
  printf("kBest pipeline ms: OLD=%.1f NEW=%.1f  speedup=%.3fx  (sink=%.0f)\n",
         t_old, t_new, t_old / t_new, (double)sink);
  return fails == 0 ? 0 : 1;
}
