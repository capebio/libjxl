// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// A/B harness for the ANS reverse-map construction in
// ANSEncodingHistogram::ANSBuildInfoTable (lib/jxl/enc_ans.cc).
//
// OLD: probe AliasTable::Lookup for each of the ANS_TAB_SIZE states.
// NEW: walk the alias table entry-by-entry, writing two straight-line store
//      runs per entry (left = own symbol, right = right_value).
//
// Proves the two fills produce byte-identical reverse maps across a range of
// normalized histograms / alphabet sizes / log_alpha_sizes, then times both.
//
// Build (clang-cl, reuse the jxl_enc-obj flags):
//   clang-cl /O2 /Ob2 /DNDEBUG -std:c++17 /EHsc -TP \
//     -IC:\Foo\raw-converter-wasm\external\libjxl-012 \
//     -IC:\Foo\raw-converter-wasm\external\libjxl-012\third_party\highway \
//     -IC:\Foo\bld-libjxl-static\lib\include \
//     tools\ans_reverse_map_ab.cc lib\jxl\ans_common.cc /Fe:ans_ab.exe

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "lib/jxl/ans_common.h"
#include "lib/jxl/ans_params.h"

using jxl::AliasTable;
using jxl::InitAliasTable;

namespace {

struct Info {
  uint16_t* reverse_map_;
  uint16_t freq_;
};

// Shared per-symbol pool partition (identical in OLD and NEW).
void Partition(size_t alphabet_size, const int32_t* counts, Info* info,
               uint16_t* pool) {
  size_t pool_pos = 0;
  for (size_t s = 0; s < std::max(size_t{1}, alphabet_size); ++s) {
    const int32_t freq =
        (s == alphabet_size) ? ANS_TAB_SIZE : counts[s];
    info[s].freq_ = static_cast<uint16_t>(freq);
    info[s].reverse_map_ = pool + pool_pos;
    pool_pos += static_cast<size_t>(freq);
  }
}

// OLD: per-state Lookup (verbatim from the pre-change ANSBuildInfoTable).
void FillOld(const AliasTable::Entry* table, size_t log_alpha_size,
             size_t alphabet_size, const int32_t* counts, Info* info,
             uint16_t* pool) {
  Partition(alphabet_size, counts, info, pool);
  const size_t les = ANS_LOG_TAB_SIZE - log_alpha_size;
  const size_t esm1 = (size_t{1} << les) - 1;
  for (int i = 0; i < ANS_TAB_SIZE; i++) {
    AliasTable::Symbol s = AliasTable::Lookup(table, i, les, esm1);
    info[s.value].reverse_map_[s.offset] = static_cast<uint16_t>(i);
  }
}

// NEW: entry-walk (verbatim from the changed ANSBuildInfoTable).
void FillNew(const AliasTable::Entry* table, size_t log_alpha_size,
             size_t alphabet_size, const int32_t* counts, Info* info,
             uint16_t* pool) {
  Partition(alphabet_size, counts, info, pool);
  const size_t log_entry_size = ANS_LOG_TAB_SIZE - log_alpha_size;
  const size_t entry_size = size_t{1} << log_entry_size;
  const size_t table_size = size_t{1} << log_alpha_size;
  for (size_t i = 0; i < table_size; ++i) {
    const AliasTable::Entry& e = table[i];
    const size_t base = i << log_entry_size;
    const size_t cutoff = e.cutoff;
    if (cutoff != 0) {
      uint16_t* left = info[i].reverse_map_;
      for (size_t pos = 0; pos < cutoff; ++pos) {
        left[pos] = static_cast<uint16_t>(base + pos);
      }
    }
    uint16_t* right = info[e.right_value].reverse_map_;
    const size_t off1 = e.offsets1;
    for (size_t pos = cutoff; pos < entry_size; ++pos) {
      right[off1 + pos] = static_cast<uint16_t>(base + pos);
    }
  }
}

size_t CeilLog2(size_t n) {
  size_t l = 0;
  while ((size_t{1} << l) < n) ++l;
  return l;
}

// One normalized histogram (sum of counts == ANS_TAB_SIZE) with `alpha`
// leading nonzero symbols; a few trailing/embedded zeros optionally injected.
struct Case {
  std::vector<int32_t> dist;      // for InitAliasTable (by value copy)
  std::vector<int32_t> counts;    // same, for the fill partition
  size_t alphabet_size;
  size_t log_alpha_size;
  std::vector<AliasTable::Entry> table;
};

Case MakeCase(std::mt19937& rng, size_t alpha, bool embed_zeros) {
  Case c;
  c.dist.assign(alpha, 0);
  if (alpha == 0) {
    // Empty histogram: InitAliasTable synthesizes a single-symbol table.
    c.alphabet_size = 0;
  } else if (alpha == 1) {
    c.dist[0] = ANS_TAB_SIZE;
    c.alphabet_size = 1;
  } else {
    // Start each symbol at 1, distribute the remainder by random weights.
    std::vector<double> w(alpha);
    double sw = 0;
    std::uniform_real_distribution<double> ud(0.05, 1.0);
    for (size_t k = 0; k < alpha; ++k) {
      w[k] = ud(rng);
      sw += w[k];
    }
    int remaining = ANS_TAB_SIZE - static_cast<int>(alpha);
    int used = 0;
    for (size_t k = 0; k < alpha; ++k) {
      int add = static_cast<int>(w[k] / sw * remaining);
      c.dist[k] = 1 + add;
      used += add;
    }
    c.dist[0] += remaining - used;  // absorb rounding into symbol 0
    if (embed_zeros) {
      // Zero out a couple of interior symbols, donate their mass to symbol 0.
      for (size_t z : {alpha / 3, (2 * alpha) / 3}) {
        if (z > 0 && z < alpha && c.dist[z] > 1) {
          c.dist[0] += c.dist[z];
          c.dist[z] = 0;
        }
      }
    }
    // Trim trailing zeros for alphabet_size (as the normalized histogram does).
    size_t used_len = alpha;
    while (used_len > 0 && c.dist[used_len - 1] == 0) --used_len;
    c.alphabet_size = used_len;
  }
  c.counts = c.dist;
  size_t need = std::max<size_t>(c.alphabet_size, 1);
  c.log_alpha_size = std::max<size_t>(5, CeilLog2(std::max<size_t>(need, 2)));
  if (c.log_alpha_size > 8) c.log_alpha_size = 8;
  c.table.resize(size_t{1} << c.log_alpha_size);
  jxl::Status st = InitAliasTable(c.dist, ANS_LOG_TAB_SIZE, c.log_alpha_size,
                                  c.table.data());
  if (!st) {
    std::fprintf(stderr, "InitAliasTable failed for alpha=%zu\n", alpha);
  }
  return c;
}

}  // namespace

int main() {
  std::mt19937 rng(12345);
  std::vector<Case> cases;
  const size_t alphas[] = {0, 1, 2, 3, 5, 8, 16, 31, 32, 64, 100, 200, 255, 256};
  for (size_t a : alphas) {
    cases.push_back(MakeCase(rng, a, false));
    if (a >= 6) cases.push_back(MakeCase(rng, a, true));
  }
  // Extra random cases for coverage.
  for (int r = 0; r < 40; ++r) {
    size_t a = 2 + (rng() % 255);
    cases.push_back(MakeCase(rng, a, (rng() & 1) != 0));
  }

  std::vector<uint16_t> poolO(ANS_TAB_SIZE), poolN(ANS_TAB_SIZE);
  std::vector<Info> infoO(256), infoN(256);

  // 1) Byte-exact verification.
  size_t mismatches = 0, checked = 0;
  for (const Case& c : cases) {
    std::fill(poolO.begin(), poolO.end(), uint16_t{0xFFFF});
    std::fill(poolN.begin(), poolN.end(), uint16_t{0xFFFF});
    FillOld(c.table.data(), c.log_alpha_size, c.alphabet_size, c.counts.data(),
            infoO.data(), poolO.data());
    FillNew(c.table.data(), c.log_alpha_size, c.alphabet_size, c.counts.data(),
            infoN.data(), poolN.data());
    if (std::memcmp(poolO.data(), poolN.data(),
                    ANS_TAB_SIZE * sizeof(uint16_t)) != 0) {
      ++mismatches;
      std::fprintf(stderr, "MISMATCH alpha=%zu log_alpha=%zu\n",
                   c.alphabet_size, c.log_alpha_size);
    }
    ++checked;
  }
  std::printf("byte-exact: %zu/%zu cases identical (%zu mismatches)\n",
              checked - mismatches, checked, mismatches);

  // 2) Timing (interleaved OLD/NEW to cancel drift).
  const int kRounds = 6000;
  volatile uint64_t sink = 0;
  double t_old = 0, t_new = 0;
  for (int round = 0; round < kRounds; ++round) {
    bool old_first = (round & 1) == 0;
    for (int which = 0; which < 2; ++which) {
      bool do_old = (which == 0) == old_first;
      auto t0 = std::chrono::steady_clock::now();
      uint64_t acc = 0;
      for (const Case& c : cases) {
        if (do_old) {
          FillOld(c.table.data(), c.log_alpha_size, c.alphabet_size,
                  c.counts.data(), infoO.data(), poolO.data());
          acc += poolO[c.alphabet_size % ANS_TAB_SIZE] + poolO[ANS_TAB_SIZE - 1];
        } else {
          FillNew(c.table.data(), c.log_alpha_size, c.alphabet_size,
                  c.counts.data(), infoN.data(), poolN.data());
          acc += poolN[c.alphabet_size % ANS_TAB_SIZE] + poolN[ANS_TAB_SIZE - 1];
        }
      }
      auto t1 = std::chrono::steady_clock::now();
      double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
      if (do_old) t_old += dt; else t_new += dt;
      sink += acc;
    }
  }
  std::printf("timing over %d rounds x %zu cases:\n", kRounds, cases.size());
  std::printf("  OLD (Lookup)     : %.2f ms\n", t_old);
  std::printf("  NEW (entry-walk) : %.2f ms\n", t_new);
  std::printf("  NEW/OLD = %.4f  (%.1f%% %s)\n", t_new / t_old,
              (1.0 - t_new / t_old) * 100.0,
              (t_new < t_old) ? "faster" : "SLOWER");
  (void)sink;
  return mismatches == 0 ? 0 : 1;
}
