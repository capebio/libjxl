// Standalone A/B harness for InitAliasTable (lib/jxl/ans_common.cc).
//
// Self-contained: stubs the few libjxl symbols InitAliasTable needs (Entry
// layout, ANS_* params, JXL_ENSURE) so it compiles with a plain clang++ and
// requires no libjxl/highway build. It embeds the OLD (pre-refactor, by-value
// vector) and NEW (allocation-free, const&) implementations side by side,
// proves they produce byte-identical alias tables for a large corpus of valid
// distributions, then times both interleaved.
//
//   clang++ -O3 -std=c++17 tools/ans_common_ab.cc -o ans_common_ab && ./ans_common_ab
//
// Exit code 0 and "BYTE-EXACT: PASS" => safe to land.

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ---- libjxl stubs -----------------------------------------------------------
#define ANS_LOG_TAB_SIZE 12u
#define ANS_TAB_SIZE (1 << ANS_LOG_TAB_SIZE)
#define ANS_MAX_ALPHABET_SIZE 256
#define JXL_RESTRICT __restrict__
// InitAliasTable's JXL_ENSURE aborts the build on a broken invariant; here we
// turn it into a hard assert (valid corpus never trips it).
#define JXL_ENSURE(cond) \
  do {                   \
    if (!(cond)) {       \
      std::fprintf(stderr, "JXL_ENSURE failed: %s\n", #cond); \
      std::abort();      \
    }                    \
  } while (0)

#pragma pack(push, 1)
struct Entry {
  uint8_t cutoff;
  uint8_t right_value;
  uint16_t freq0;
  uint16_t offsets1;
  uint16_t freq1_xor_freq0;
};
#pragma pack(pop)
static_assert(sizeof(Entry) == 8, "Entry must be 8 bytes");

using Status = bool;

// ---- OLD: verbatim pre-refactor implementation ------------------------------
static Status InitAliasTable_OLD(std::vector<int32_t> distribution,
                                 uint32_t log_range, size_t log_alpha_size,
                                 Entry* JXL_RESTRICT a) {
  const uint32_t range = 1 << log_range;
  const size_t table_size = 1 << log_alpha_size;
  JXL_ENSURE(table_size <= range);
  while (!distribution.empty() && distribution.back() == 0) {
    distribution.pop_back();
  }
  if (distribution.empty()) {
    distribution.emplace_back(range);
  }
  JXL_ENSURE(distribution.size() <= table_size);
  const uint32_t entry_size = range >> log_alpha_size;
  int single_symbol = -1;
  int sum = 0;
  for (size_t sym = 0; sym < distribution.size(); sym++) {
    int32_t v = distribution[sym];
    sum += v;
    if (v == ANS_TAB_SIZE) {
      JXL_ENSURE(single_symbol == -1);
      single_symbol = sym;
    }
  }
  JXL_ENSURE(static_cast<uint32_t>(sum) == range);
  if (single_symbol != -1) {
    uint8_t sym = single_symbol;
    JXL_ENSURE(single_symbol == sym);
    for (size_t i = 0; i < table_size; i++) {
      a[i].right_value = sym;
      a[i].cutoff = 0;
      a[i].offsets1 = entry_size * i;
      a[i].freq0 = 0;
      a[i].freq1_xor_freq0 = ANS_TAB_SIZE;
    }
    return true;
  }
  std::vector<uint32_t> underfull_posn;
  std::vector<uint32_t> overfull_posn;
  std::vector<uint32_t> cutoffs(1 << log_alpha_size);
  for (size_t i = 0; i < distribution.size(); i++) {
    cutoffs[i] = distribution[i];
    if (cutoffs[i] > entry_size) {
      overfull_posn.push_back(i);
    } else if (cutoffs[i] < entry_size) {
      underfull_posn.push_back(i);
    }
  }
  for (uint32_t i = distribution.size(); i < table_size; i++) {
    cutoffs[i] = 0;
    underfull_posn.push_back(i);
  }
  while (!overfull_posn.empty()) {
    uint32_t overfull_i = overfull_posn.back();
    overfull_posn.pop_back();
    JXL_ENSURE(!underfull_posn.empty());
    uint32_t underfull_i = underfull_posn.back();
    underfull_posn.pop_back();
    uint32_t underfull_by = entry_size - cutoffs[underfull_i];
    cutoffs[overfull_i] -= underfull_by;
    a[underfull_i].right_value = overfull_i;
    a[underfull_i].offsets1 = cutoffs[overfull_i];
    if (cutoffs[overfull_i] < entry_size) {
      underfull_posn.push_back(overfull_i);
    } else if (cutoffs[overfull_i] > entry_size) {
      overfull_posn.push_back(overfull_i);
    }
  }
  for (uint32_t i = 0; i < table_size; i++) {
    if (cutoffs[i] == entry_size) {
      a[i].right_value = i;
      a[i].offsets1 = 0;
      a[i].cutoff = 0;
    } else {
      a[i].offsets1 -= cutoffs[i];
      a[i].cutoff = cutoffs[i];
    }
    const size_t freq0 = i < distribution.size() ? distribution[i] : 0;
    const size_t i1 = a[i].right_value;
    const size_t freq1 = i1 < distribution.size() ? distribution[i1] : 0;
    a[i].freq0 = static_cast<uint16_t>(freq0);
    a[i].freq1_xor_freq0 = static_cast<uint16_t>(freq1 ^ freq0);
  }
  return true;
}

// ---- NEW: allocation-free, const& (mirror of refactored ans_common.cc) ------
static Status InitAliasTable_NEW(const std::vector<int32_t>& distribution,
                                 uint32_t log_range, size_t log_alpha_size,
                                 Entry* JXL_RESTRICT a) {
  const uint32_t range = 1u << log_range;
  const size_t table_size = 1u << log_alpha_size;
  JXL_ENSURE(table_size <= range);
  const uint32_t entry_size = range >> log_alpha_size;

  size_t distribution_size = distribution.size();
  while (distribution_size != 0 && distribution[distribution_size - 1] == 0) {
    --distribution_size;
  }

  if (distribution_size == 0) {
    for (size_t i = 0; i < table_size; i++) {
      a[i].right_value = 0;
      a[i].cutoff = 0;
      a[i].offsets1 = static_cast<uint16_t>(entry_size * i);
      a[i].freq0 = 0;
      a[i].freq1_xor_freq0 = ANS_TAB_SIZE;
    }
    return true;
  }

  JXL_ENSURE(distribution_size <= table_size);
  int single_symbol = -1;
  int sum = 0;
  for (size_t sym = 0; sym < distribution_size; sym++) {
    int32_t v = distribution[sym];
    sum += v;
    if (v == ANS_TAB_SIZE) {
      JXL_ENSURE(single_symbol == -1);
      single_symbol = sym;
    }
  }
  JXL_ENSURE(static_cast<uint32_t>(sum) == range);
  if (single_symbol != -1) {
    uint8_t sym = single_symbol;
    JXL_ENSURE(single_symbol == sym);
    for (size_t i = 0; i < table_size; i++) {
      a[i].right_value = sym;
      a[i].cutoff = 0;
      a[i].offsets1 = entry_size * i;
      a[i].freq0 = 0;
      a[i].freq1_xor_freq0 = ANS_TAB_SIZE;
    }
    return true;
  }

  std::array<uint8_t, ANS_MAX_ALPHABET_SIZE> underfull_posn;
  std::array<uint8_t, ANS_MAX_ALPHABET_SIZE> overfull_posn;
  size_t num_underfull = 0;
  size_t num_overfull = 0;

  for (size_t i = 0; i < distribution_size; i++) {
    const uint32_t cutoff = static_cast<uint32_t>(distribution[i]);
    a[i].freq0 = static_cast<uint16_t>(cutoff);
    a[i].freq1_xor_freq0 = static_cast<uint16_t>(cutoff);
    if (cutoff > entry_size) {
      overfull_posn[num_overfull++] = static_cast<uint8_t>(i);
    } else if (cutoff < entry_size) {
      underfull_posn[num_underfull++] = static_cast<uint8_t>(i);
    }
  }
  for (size_t i = distribution_size; i < table_size; i++) {
    a[i].freq0 = 0;
    a[i].freq1_xor_freq0 = 0;
    underfull_posn[num_underfull++] = static_cast<uint8_t>(i);
  }
  while (num_overfull != 0) {
    const uint8_t overfull_i = overfull_posn[--num_overfull];
    JXL_ENSURE(num_underfull != 0);
    const uint8_t underfull_i = underfull_posn[--num_underfull];
    const uint32_t underfull_by = entry_size - a[underfull_i].freq1_xor_freq0;
    a[overfull_i].freq1_xor_freq0 =
        static_cast<uint16_t>(a[overfull_i].freq1_xor_freq0 - underfull_by);
    a[underfull_i].right_value = overfull_i;
    a[underfull_i].offsets1 = a[overfull_i].freq1_xor_freq0;
    if (a[overfull_i].freq1_xor_freq0 < entry_size) {
      underfull_posn[num_underfull++] = overfull_i;
    } else if (a[overfull_i].freq1_xor_freq0 > entry_size) {
      overfull_posn[num_overfull++] = overfull_i;
    }
  }
  for (size_t i = 0; i < table_size; i++) {
    const uint16_t cutoff = a[i].freq1_xor_freq0;
    if (cutoff == entry_size) {
      a[i].right_value = static_cast<uint8_t>(i);
      a[i].offsets1 = 0;
      a[i].cutoff = 0;
    } else {
      a[i].offsets1 = static_cast<uint16_t>(a[i].offsets1 - cutoff);
      a[i].cutoff = static_cast<uint8_t>(cutoff);
    }
    a[i].freq1_xor_freq0 =
        static_cast<uint16_t>(a[a[i].right_value].freq0 ^ a[i].freq0);
  }
  return true;
}

// ---- corpus -----------------------------------------------------------------
struct Case {
  std::vector<int32_t> dist;
  size_t log_alpha_size;
};

// Random valid distribution: n symbols summing to ANS_TAB_SIZE, each in
// [0, ANS_TAB_SIZE], with table_size = 1<<log_alpha_size >= n.
static Case MakeCase(std::mt19937& rng) {
  std::uniform_int_distribution<int> la_pick(5, 8);
  size_t log_alpha_size = la_pick(rng);
  size_t table_size = size_t{1} << log_alpha_size;
  std::uniform_int_distribution<int> n_pick(1, static_cast<int>(table_size));
  size_t n = n_pick(rng);

  std::vector<int32_t> dist(n, 0);
  // Distribute ANS_TAB_SIZE units across n buckets via random weights.
  std::vector<double> w(n);
  std::uniform_real_distribution<double> wd(0.0, 1.0);
  double wsum = 0.0;
  for (size_t i = 0; i < n; i++) { w[i] = wd(rng) + 1e-6; wsum += w[i]; }
  int assigned = 0;
  for (size_t i = 0; i < n; i++) {
    int v = static_cast<int>(w[i] / wsum * ANS_TAB_SIZE);
    dist[i] = v;
    assigned += v;
  }
  // Fix rounding so the sum is exactly ANS_TAB_SIZE.
  int diff = ANS_TAB_SIZE - assigned;
  size_t idx = 0;
  while (diff > 0) { dist[idx % n] += 1; diff--; idx++; }
  while (diff < 0) {
    if (dist[idx % n] > 0) { dist[idx % n] -= 1; diff++; }
    idx++;
  }
  // Occasionally inject trailing zeros to exercise the trim path.
  if ((rng() & 3) == 0 && n < table_size) {
    dist.push_back(0);
    if (n + 1 < table_size && (rng() & 1)) dist.push_back(0);
  }
  return {std::move(dist), log_alpha_size};
}

int main() {
  std::mt19937 rng(0xA5C0DE);
  const int kCases = 200000;

  std::vector<Case> corpus;
  corpus.reserve(kCases);
  // Hand-picked edge cases first.
  corpus.push_back({{ANS_TAB_SIZE / 2, ANS_TAB_SIZE / 2}, 8});
  corpus.push_back({{ANS_TAB_SIZE}, 8});
  corpus.push_back({{0, 0, 0, ANS_TAB_SIZE, 0}, 8});
  corpus.push_back({{}, 8});             // empty -> degenerate
  corpus.push_back({{0, 0, 0}, 8});      // all-zero -> degenerate
  corpus.push_back({{ANS_TAB_SIZE}, 5}); // single symbol, 7-bit-ish entry
  for (int i = 0; i < kCases; i++) corpus.push_back(MakeCase(rng));

  // ---- byte-exactness ----
  size_t fails = 0;
  Entry old_t[ANS_MAX_ALPHABET_SIZE];
  Entry new_t[ANS_MAX_ALPHABET_SIZE];
  for (const Case& c : corpus) {
    std::memset(old_t, 0, sizeof(old_t));
    std::memset(new_t, 0, sizeof(new_t));
    size_t table_size = size_t{1} << c.log_alpha_size;
    Status o = InitAliasTable_OLD(c.dist, ANS_LOG_TAB_SIZE, c.log_alpha_size, old_t);
    Status n = InitAliasTable_NEW(c.dist, ANS_LOG_TAB_SIZE, c.log_alpha_size, new_t);
    if (o != n || std::memcmp(old_t, new_t, table_size * sizeof(Entry)) != 0) {
      if (fails < 5) {
        std::fprintf(stderr, "MISMATCH: n=%zu log_alpha=%zu\n",
                     c.dist.size(), c.log_alpha_size);
      }
      fails++;
    }
  }
  std::printf("checked %zu cases, %zu byte-diffs\n", corpus.size(), fails);
  std::printf("BYTE-EXACT: %s\n", fails == 0 ? "PASS" : "FAIL");

  // ---- timing (interleaved, checksum to prevent DCE) ----
  const int kReps = 40;
  uint64_t csum_old = 0, csum_new = 0;
  double t_old = 0, t_new = 0;
  for (int rep = 0; rep < kReps; rep++) {
    {
      auto s = std::chrono::high_resolution_clock::now();
      for (const Case& c : corpus) {
        InitAliasTable_OLD(c.dist, ANS_LOG_TAB_SIZE, c.log_alpha_size, old_t);
        csum_old += old_t[0].offsets1 + old_t[(1u << c.log_alpha_size) - 1].cutoff;
      }
      auto e = std::chrono::high_resolution_clock::now();
      t_old += std::chrono::duration<double, std::milli>(e - s).count();
    }
    {
      auto s = std::chrono::high_resolution_clock::now();
      for (const Case& c : corpus) {
        InitAliasTable_NEW(c.dist, ANS_LOG_TAB_SIZE, c.log_alpha_size, new_t);
        csum_new += new_t[0].offsets1 + new_t[(1u << c.log_alpha_size) - 1].cutoff;
      }
      auto e = std::chrono::high_resolution_clock::now();
      t_new += std::chrono::duration<double, std::milli>(e - s).count();
    }
  }
  size_t total_builds = corpus.size() * kReps;
  std::printf("OLD: %.2f ms total, %.1f ns/build\n", t_old, t_old * 1e6 / total_builds);
  std::printf("NEW: %.2f ms total, %.1f ns/build\n", t_new, t_new * 1e6 / total_builds);
  std::printf("speedup: %.3fx  (checksums %llu / %llu)\n", t_old / t_new,
              (unsigned long long)csum_old, (unsigned long long)csum_new);
  return fails == 0 ? 0 : 1;
}
