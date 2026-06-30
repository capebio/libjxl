// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_cluster.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <tuple>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_ans_params.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_cluster.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/fast_math-inl.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::AllTrue;
using hwy::HWY_NAMESPACE::Eq;
using hwy::HWY_NAMESPACE::GetLane;
using hwy::HWY_NAMESPACE::IfThenZeroElse;
using hwy::HWY_NAMESPACE::StoreU;
using hwy::HWY_NAMESPACE::SumOfLanes;
using hwy::HWY_NAMESPACE::Zero;

template <class V>
V Entropy(V count, V inv_total, V total) {
  const HWY_CAPPED(float, Histogram::kRounding) d;
  const auto zero = Set(d, 0.0f);
  // TODO(eustas): why (0 - x) instead of Neg(x)?
  return IfThenZeroElse(
      Eq(count, total),
      Sub(zero, Mul(count, FastLog2f(d, Mul(inv_total, count)))));
}

void HistogramCondition(Histogram& a) {
  const HWY_CAPPED(int32_t, Histogram::kRounding) di;
  const auto kZero = Zero(di);
  auto total = kZero;
  int nz_pos = -static_cast<int>(Lanes(di));
  for (size_t i = 0; i < a.counts.size(); i += Lanes(di)) {
    const auto counts = LoadU(di, &a.counts[i]);
    const bool nz = !AllTrue(di, Eq(counts, kZero));
    total = Add(total, counts);
    if (nz) nz_pos = i;
  }
  a.counts.resize(nz_pos + Lanes(di));
  a.total_count = GetLane(SumOfLanes(di, total));
}

void HistogramEntropy(const Histogram& a) {
  a.entropy = 0.0f;
  if (a.total_count == 0) return;

  const HWY_CAPPED(float, Histogram::kRounding) df;
  const HWY_CAPPED(int32_t, Histogram::kRounding) di;

  const auto inv_tot = Set(df, 1.0f / a.total_count);
  auto entropy_lanes = Zero(df);
  auto total = Set(df, a.total_count);

  for (size_t i = 0; i < a.counts.size(); i += Lanes(di)) {
    const auto counts = LoadU(di, &a.counts[i]);
    entropy_lanes =
        Add(entropy_lanes, Entropy(ConvertTo(df, counts), inv_tot, total));
  }
  a.entropy += GetLane(SumOfLanes(df, entropy_lanes));
}

// Fold `src` into `dst` and refresh dst.entropy in a single pass over the
// count vector. FastClusterHistograms needs dst.entropy immediately after
// growing a cluster, so the previous AddHistogram()+HistogramEntropy() pair
// traversed the same counts twice. Counts are always a multiple of kRounding
// (>= SIMD width), so every loop bound is lane-aligned. Byte-exact with the
// two-call sequence: identical summed counts, identical ascending lane
// accumulation order, identical total_count.
void HistogramAddAndEntropy(Histogram& dst, const Histogram& src) {
  const size_t dst_size = dst.counts.size();
  const size_t src_size = src.counts.size();
  if (src_size > dst_size) {
    dst.counts.resize(src_size);
  }
  dst.total_count += src.total_count;
  dst.entropy = 0.0f;
  if (dst.total_count == 0) return;

  const HWY_CAPPED(float, Histogram::kRounding) df;
  const HWY_CAPPED(int32_t, Histogram::kRounding) di;
  const size_t lanes = Lanes(di);

  const auto inv_tot = Set(df, 1.0f / dst.total_count);
  auto entropy_lanes = Zero(df);
  const auto total = Set(df, dst.total_count);
  const size_t common_size = std::min(dst_size, src_size);

  size_t i = 0;
  for (; i < common_size; i += lanes) {
    const auto counts =
        Add(LoadU(di, &dst.counts[i]), LoadU(di, &src.counts[i]));
    StoreU(counts, di, &dst.counts[i]);
    entropy_lanes =
        Add(entropy_lanes, Entropy(ConvertTo(df, counts), inv_tot, total));
  }
  if (src_size > dst_size) {
    for (; i < src_size; i += lanes) {
      const auto counts = LoadU(di, &src.counts[i]);
      StoreU(counts, di, &dst.counts[i]);
      entropy_lanes =
          Add(entropy_lanes, Entropy(ConvertTo(df, counts), inv_tot, total));
    }
  } else {
    for (; i < dst_size; i += lanes) {
      const auto counts = LoadU(di, &dst.counts[i]);
      entropy_lanes =
          Add(entropy_lanes, Entropy(ConvertTo(df, counts), inv_tot, total));
    }
  }
  dst.entropy = GetLane(SumOfLanes(df, entropy_lanes));
}

float HistogramDistance(const Histogram& a, const Histogram& b) {
  if (a.total_count == 0 || b.total_count == 0) return 0;

  const HWY_CAPPED(float, Histogram::kRounding) df;
  const HWY_CAPPED(int32_t, Histogram::kRounding) di;
  const size_t lanes = Lanes(di);
  const size_t a_size = a.counts.size();
  const size_t b_size = b.counts.size();
  const size_t common_size = std::min(a_size, b_size);

  const auto inv_tot = Set(df, 1.0f / (a.total_count + b.total_count));
  auto distance_lanes = Zero(df);
  const auto total = Set(df, a.total_count + b.total_count);

  // Overlapping alphabet: both operands present, no per-lane size test.
  size_t i = 0;
  for (; i < common_size; i += lanes) {
    const auto counts =
        ConvertTo(df, Add(LoadU(di, &a.counts[i]), LoadU(di, &b.counts[i])));
    distance_lanes = Add(distance_lanes, Entropy(counts, inv_tot, total));
  }
  // Tail of the longer operand: the shorter contributes zero.
  const Histogram& longer = a_size > b_size ? a : b;
  for (; i < longer.counts.size(); i += lanes) {
    const auto counts = ConvertTo(df, LoadU(di, &longer.counts[i]));
    distance_lanes = Add(distance_lanes, Entropy(counts, inv_tot, total));
  }
  const float total_distance = GetLane(SumOfLanes(df, distance_lanes));
  return total_distance - a.entropy - b.entropy;
}

constexpr const float kInfinity = std::numeric_limits<float>::infinity();

float HistogramKLDivergence(const Histogram& actual, const Histogram& coding) {
  if (actual.total_count == 0) return 0;
  if (coding.total_count == 0) return kInfinity;

  const HWY_CAPPED(float, Histogram::kRounding) df;
  const HWY_CAPPED(int32_t, Histogram::kRounding) di;
  const size_t lanes = Lanes(di);
  const auto zero = Zero(di);
  const size_t common_size =
      std::min(actual.counts.size(), coding.counts.size());

  // Any positive actual count beyond coding's alphabet has zero coding
  // probability => infinite divergence. Reject without log work. This matches
  // the original SIMD path, which produced +inf for those lanes; the finite
  // path below is byte-exact because those tail lanes contributed exactly 0.
  for (size_t i = common_size; i < actual.counts.size(); i += lanes) {
    if (!AllTrue(di, Eq(LoadU(di, &actual.counts[i]), zero))) {
      return kInfinity;
    }
  }

  const auto coding_inv = Set(df, 1.0f / coding.total_count);
  auto cost_lanes = Zero(df);
  for (size_t i = 0; i < common_size; i += lanes) {
    const auto counts = LoadU(di, &actual.counts[i]);
    const auto coding_counts = LoadU(di, &coding.counts[i]);
    const auto coding_probs = Mul(ConvertTo(df, coding_counts), coding_inv);
    const auto neg_coding_cost = BitCast(
        df,
        IfThenZeroElse(Eq(counts, zero),
                       IfThenElse(Eq(coding_counts, zero),
                                  BitCast(di, Set(df, -kInfinity)),
                                  BitCast(di, FastLog2f(df, coding_probs)))));
    cost_lanes = NegMulAdd(ConvertTo(df, counts), neg_coding_cost, cost_lanes);
  }
  const float total_cost = GetLane(SumOfLanes(df, cost_lanes));
  return total_cost - actual.entropy;
}

// First step of a k-means clustering with a fancy distance metric.
Status FastClusterHistograms(const std::vector<Histogram>& in,
                             size_t max_histograms, std::vector<Histogram>* out,
                             std::vector<uint32_t>* histogram_symbols) {
  const size_t prev_histograms = out->size();
  out->reserve(max_histograms);
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
    if (in[i].total_count > in[largest_idx].total_count) {
      largest_idx = i;
    }
  }

  if (prev_histograms > 0) {
    for (size_t j = 0; j < prev_histograms; ++j) {
      HistogramEntropy((*out)[j]);
    }
    for (size_t i = 0; i < in.size(); i++) {
      if (dists[i] == 0.0f) continue;
      for (size_t j = 0; j < prev_histograms; ++j) {
        dists[i] = std::min(HistogramKLDivergence(in[i], (*out)[j]), dists[i]);
      }
    }
    auto max_dist = std::max_element(dists.begin(), dists.end());
    if (*max_dist > 0.0f) {
      largest_idx = max_dist - dists.begin();
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
      if (dist < best_dist) {
        best = j;
        best_dist = dist;
      }
    }
    JXL_ENSURE(best_dist < std::numeric_limits<float>::max());
    if (best >= prev_histograms) {
      // Fused: AddHistogram(in[i]) + HistogramEntropy, single counts traversal.
      HistogramAddAndEntropy((*out)[best], in[i]);
    }
    (*histogram_symbols)[i] = best;
  }
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(FastClusterHistograms);  // Local function
HWY_EXPORT(HistogramEntropy);       // Local function
HWY_EXPORT(HistogramCondition);     // Local function

void Histogram::Condition() { HWY_DYNAMIC_DISPATCH(HistogramCondition)(*this); }

float Histogram::ShannonEntropy() const {
  HWY_DYNAMIC_DISPATCH(HistogramEntropy)(*this);
  return entropy;
}

namespace {
// -----------------------------------------------------------------------------
// Histogram refinement

constexpr uint32_t kInvalidHistogram = std::numeric_limits<uint32_t>::max();

// Reorder histograms in *out so that the new symbols in *symbols come in
// increasing order. The permutation is applied in place with swaps: histogram
// payload (count) vectors are never deep-copied merely to canonicalize the
// context map. Unreferenced (dead) histograms are permuted past the live tail
// and dropped by the final resize.
void HistogramReindex(std::vector<Histogram>* out, size_t prev_histograms,
                      std::vector<uint32_t>* symbols) {
  const size_t num_histograms = out->size();
  if (prev_histograms == num_histograms) return;

  // new_index maps old slot -> new slot. Pre-existing histograms keep theirs.
  std::vector<uint32_t> new_index(num_histograms, kInvalidHistogram);
  for (size_t i = 0; i < prev_histograms; ++i) {
    new_index[i] = static_cast<uint32_t>(i);
  }

  size_t next_index = prev_histograms;
  bool is_identity = true;
  for (uint32_t& symbol : *symbols) {
    uint32_t new_symbol = new_index[symbol];
    if (new_symbol == kInvalidHistogram) {
      new_symbol = static_cast<uint32_t>(next_index);
      new_index[symbol] = new_symbol;
      is_identity &= symbol == next_index;
      ++next_index;
    }
    symbol = new_symbol;
  }

  if (next_index == prev_histograms) {
    // No new histograms survived; drop any dead tail.
    out->resize(prev_histograms);
    return;
  }

  // Complete old->new into a full permutation by sending dead slots past the
  // live region (they get truncated below).
  size_t next_unused = next_index;
  for (size_t old_index = 0; old_index < num_histograms; ++old_index) {
    if (new_index[old_index] == kInvalidHistogram) {
      new_index[old_index] = static_cast<uint32_t>(next_unused++);
    }
  }

  if (!is_identity) {
    // Cycle-sort: the element currently at i belongs at new_index[i].
    for (size_t i = 0; i < num_histograms; ++i) {
      while (new_index[i] != i) {
        const size_t target = new_index[i];
        (*out)[i].swap((*out)[target]);
        std::swap(new_index[i], new_index[target]);
      }
    }
  }
  out->resize(next_index);
}

// Union-find with path halving. Returns the surviving cluster that `index` was
// merged into, equivalent to the old per-merge renumbering scan.
uint32_t FindHistogramRepresentative(std::vector<uint32_t>* parent,
                                     uint32_t index) {
  while ((*parent)[index] != index) {
    (*parent)[index] = (*parent)[(*parent)[index]];
    index = (*parent)[index];
  }
  return index;
}

// Exact merge cost of `first` + `second`, reusing a single scratch histogram
// for the merged counts instead of allocating a fresh Histogram per candidate
// pair. Byte-exact with the old `Histogram histo; histo.AddHistogram(first);
// histo.AddHistogram(second); cost - (first.entropy + second.entropy)`: the
// entropy subtraction keeps the original `cost - (a + b)` rounding order.
StatusOr<float> HistogramMergeCost(const Histogram& first,
                                   const Histogram& second, Histogram* merged) {
  const size_t first_size = first.counts.size();
  const size_t second_size = second.counts.size();
  const size_t common_size = std::min(first_size, second_size);
  const size_t merged_size = std::max(first_size, second_size);

  merged->counts.resize(merged_size);
  for (size_t i = 0; i < common_size; ++i) {
    merged->counts[i] = first.counts[i] + second.counts[i];
  }
  if (first_size > second_size) {
    std::copy(first.counts.begin() + common_size, first.counts.end(),
              merged->counts.begin() + common_size);
  } else {
    std::copy(second.counts.begin() + common_size, second.counts.end(),
              merged->counts.begin() + common_size);
  }
  merged->total_count = first.total_count + second.total_count;

  JXL_ASSIGN_OR_RETURN(float cost, merged->ANSPopulationCost());
  return cost - (first.entropy + second.entropy);
}

}  // namespace

// Clusters similar histograms in 'in' together, the selected histograms are
// placed in 'out', and for each index in 'in', *histogram_symbols will
// indicate which of the 'out' histograms is the best approximation.
Status ClusterHistograms(const HistogramParams& params,
                         const std::vector<Histogram>& in,
                         size_t max_histograms, std::vector<Histogram>* out,
                         std::vector<uint32_t>* histogram_symbols) {
  // Guard the empty-input path: with pre-existing histograms present,
  // FastClusterHistograms' seed search would dereference std::max_element over
  // an empty distance vector.
  if (in.empty()) {
    histogram_symbols->clear();
    return true;
  }

  size_t prev_histograms = out->size();
  max_histograms = std::min(max_histograms, params.max_histograms);
  max_histograms = std::min(max_histograms, in.size());
  if (params.clustering == HistogramParams::ClusteringType::kFastest) {
    max_histograms = std::min(max_histograms, static_cast<size_t>(4));
  }

  JXL_RETURN_IF_ERROR(HWY_DYNAMIC_DISPATCH(FastClusterHistograms)(
      in, prev_histograms + max_histograms, out, histogram_symbols));

  if (prev_histograms == 0 &&
      params.clustering == HistogramParams::ClusteringType::kBest) {
    for (auto& histo : *out) {
      JXL_ASSIGN_OR_RETURN(histo.entropy, histo.ANSPopulationCost());
    }
    uint32_t next_version = 2;
    std::vector<uint32_t> version(out->size(), 1);
    // Union-find parent links: parent[c] points at the cluster c merged into.
    std::vector<uint32_t> parent(out->size());
    std::iota(parent.begin(), parent.end(), 0u);

    // Try to pair up clusters if doing so reduces the total cost.

    struct HistogramPair {
      // validity of a pair: p.version == max(version[i], version[j])
      float cost;
      uint32_t first;
      uint32_t second;
      uint32_t version;
      // We use > because priority queues sort in *decreasing* order, but we
      // want lower cost elements to appear first.
      bool operator<(const HistogramPair& other) const {
        return std::make_tuple(cost, first, second, version) >
               std::make_tuple(other.cost, other.first, other.second,
                               other.version);
      }
    };

    // Create list of all pairs by increasing merging cost.
    std::priority_queue<HistogramPair> pairs_to_merge;
    Histogram merged;  // Reused scratch for every merge-cost evaluation.
    for (uint32_t i = 0; i < out->size(); i++) {
      for (uint32_t j = i + 1; j < out->size(); j++) {
        JXL_ASSIGN_OR_RETURN(
            float cost, HistogramMergeCost((*out)[i], (*out)[j], &merged));
        // Avoid enqueueing pairs that are not advantageous to merge.
        if (cost >= 0) continue;
        pairs_to_merge.push(
            HistogramPair{cost, i, j, std::max(version[i], version[j])});
      }
    }

    // Merge the best pair to merge, add new pairs that get formed as a
    // consequence.
    while (!pairs_to_merge.empty()) {
      uint32_t first = pairs_to_merge.top().first;
      uint32_t second = pairs_to_merge.top().second;
      uint32_t ver = pairs_to_merge.top().version;
      pairs_to_merge.pop();
      if (ver != std::max(version[first], version[second]) ||
          version[first] == 0 || version[second] == 0) {
        continue;
      }
      (*out)[first].AddHistogram((*out)[second]);
      JXL_ASSIGN_OR_RETURN((*out)[first].entropy,
                           (*out)[first].ANSPopulationCost());
      parent[second] = first;
      version[second] = 0;
      version[first] = next_version++;
      for (uint32_t j = 0; j < out->size(); j++) {
        if (j == first) continue;
        if (version[j] == 0) continue;
        JXL_ASSIGN_OR_RETURN(
            float merge_cost,
            HistogramMergeCost((*out)[first], (*out)[j], &merged));
        // Avoid enqueueing pairs that are not advantageous to merge.
        if (merge_cost >= 0) continue;
        pairs_to_merge.push(
            HistogramPair{merge_cost, std::min(first, j), std::max(first, j),
                          std::max(version[first], version[j])});
      }
    }

    // Each surviving cluster still occupies its original slot. Resolve every
    // context to its union-find root; HistogramReindex below performs the sole
    // compaction + canonical first-use ordering pass (dead slots dropped).
    for (uint32_t& item : *histogram_symbols) {
      item = FindHistogramRepresentative(&parent, item);
    }
  }

  // Convert the context map to a canonical form.
  HistogramReindex(out, prev_histograms, histogram_symbols);
  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE
