// A/B equivalence harness for enc_cluster.cc reorder/renumber rewrite.
//
// Proves that the NEW non-SIMD data-structure changes produce byte-identical
// `out` ordering and `symbols` arrays vs the ORIGINAL libjxl logic, over many
// randomized merge sequences and symbol arrays. Does NOT need libjxl/Highway:
// the merge *decisions* are byte-exact by construction (HistogramMergeCost
// reproduces identical counts -> identical ANSPopulationCost); what is tested
// here is purely the renumbering (scan vs union-find) and the canonicalizing
// reindex (std::map + deep-copy vs in-place cycle-sort).
//
//   OLD: per-merge renumbering scan + reverse_renumbering compaction +
//        map-based HistogramReindex.
//   NEW: union-find FindRep + in-place cycle-sort HistogramReindex.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <numeric>
#include <random>
#include <vector>

// Mock Histogram: identity is its integer `id` (stands in for the count
// payload). swap() swaps payload, matching jxl::Histogram::swap semantics for
// reordering purposes.
struct Histo {
  int id = -1;
  void swap(Histo& o) { std::swap(id, o.id); }
};

// ---- ORIGINAL HistogramReindex (map + deep copy) ----------------------------
static void ReindexOld(std::vector<Histo>* out, size_t prev_histograms,
                       std::vector<uint32_t>* symbols) {
  std::vector<Histo> tmp(*out);
  std::map<int, int> new_index;
  for (size_t i = 0; i < prev_histograms; ++i) new_index[i] = i;
  int next_index = static_cast<int>(prev_histograms);
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

// ---- NEW HistogramReindex (in-place cycle-sort) -----------------------------
static const uint32_t kInvalid = UINT32_MAX;
static void ReindexNew(std::vector<Histo>* out, size_t prev_histograms,
                       std::vector<uint32_t>* symbols) {
  const size_t num = out->size();
  if (prev_histograms == num) return;
  std::vector<uint32_t> ni(num, kInvalid);
  for (size_t i = 0; i < prev_histograms; ++i) ni[i] = (uint32_t)i;
  size_t next_index = prev_histograms;
  bool is_identity = true;
  for (uint32_t& symbol : *symbols) {
    uint32_t ns = ni[symbol];
    if (ns == kInvalid) {
      ns = (uint32_t)next_index;
      ni[symbol] = ns;
      is_identity &= symbol == next_index;
      ++next_index;
    }
    symbol = ns;
  }
  if (next_index == prev_histograms) {
    out->resize(prev_histograms);
    return;
  }
  size_t next_unused = next_index;
  for (size_t oi = 0; oi < num; ++oi)
    if (ni[oi] == kInvalid) ni[oi] = (uint32_t)next_unused++;
  if (!is_identity) {
    for (size_t i = 0; i < num; ++i) {
      while (ni[i] != i) {
        const size_t t = ni[i];
        (*out)[i].swap((*out)[t]);
        std::swap(ni[i], ni[t]);
      }
    }
  }
  out->resize(next_index);
}

static uint32_t FindRep(std::vector<uint32_t>* parent, uint32_t i) {
  while ((*parent)[i] != i) {
    (*parent)[i] = (*parent)[(*parent)[i]];
    i = (*parent)[i];
  }
  return i;
}

// Run one randomized kBest-style scenario through OLD and NEW full pipelines
// and compare. Returns true on match.
static bool RunCase(std::mt19937& rng, int& fail_detail) {
  std::uniform_int_distribution<int> nd(1, 24);
  const int N = nd(rng);

  // out[i].id == i initially (distinct payloads).
  std::vector<Histo> out0(N);
  for (int i = 0; i < N; ++i) out0[i].id = i;

  // Simulate a random valid merge sequence, maintaining BOTH structures.
  std::vector<uint32_t> version(N, 1);
  std::vector<uint32_t> renum(N);          // OLD
  std::iota(renum.begin(), renum.end(), 0u);
  std::vector<uint32_t> parent(N);         // NEW
  std::iota(parent.begin(), parent.end(), 0u);

  std::uniform_int_distribution<int> merges_d(0, N);
  int num_merges = merges_d(rng);
  for (int m = 0; m < num_merges; ++m) {
    // collect alive
    std::vector<int> alive;
    for (int i = 0; i < N; ++i)
      if (version[i] != 0) alive.push_back(i);
    if (alive.size() < 2) break;
    std::uniform_int_distribution<int> ad(0, (int)alive.size() - 1);
    int a = alive[ad(rng)], b = alive[ad(rng)];
    while (b == a) b = alive[ad(rng)];
    uint32_t first = std::min(a, b), second = std::max(a, b);
    // Mirror the merge-loop's choices is not required for the renumbering test;
    // we just need first to survive, second to die, consistently in both.
    // OLD scan:
    for (uint32_t& it : renum)
      if (it == second) it = first;
    // NEW union-find:
    parent[second] = first;
    version[second] = 0;
  }

  // Build a symbols array referencing every original cluster at least once
  // (matches the real invariant: every alive root is referenced), plus extras.
  std::vector<uint32_t> base_syms(N);
  std::iota(base_syms.begin(), base_syms.end(), 0u);
  std::uniform_int_distribution<int> extra_d(0, 3 * N);
  int extra = extra_d(rng);
  std::vector<uint32_t> syms = base_syms;
  std::uniform_int_distribution<int> sd(0, N - 1);
  for (int e = 0; e < extra; ++e) syms.push_back(sd(rng));
  std::shuffle(syms.begin(), syms.end(), rng);

  // ---- OLD full pipeline ----
  std::vector<Histo> outO = out0;
  std::vector<uint32_t> symsO = syms;
  {
    std::vector<uint32_t> rev(N, kInvalid);
    size_t num_alive = 0;
    for (int i = 0; i < N; ++i) {
      if (version[i] == 0) continue;
      outO[num_alive++] = outO[i];
      rev[i] = (uint32_t)(num_alive - 1);
    }
    outO.resize(num_alive);
    for (uint32_t& it : symsO) it = rev[renum[it]];
    ReindexOld(&outO, 0, &symsO);
  }

  // ---- NEW full pipeline ----
  std::vector<Histo> outN = out0;
  std::vector<uint32_t> symsN = syms;
  {
    for (uint32_t& it : symsN) it = FindRep(&parent, it);
    ReindexNew(&outN, 0, &symsN);
  }

  // Compare.
  if (outO.size() != outN.size()) { fail_detail = 1; return false; }
  for (size_t i = 0; i < outO.size(); ++i)
    if (outO[i].id != outN[i].id) { fail_detail = 2; return false; }
  if (symsO != symsN) { fail_detail = 3; return false; }
  return true;
}

// Separately exercise the prev_histograms>0 (fast-path) reindex: OLD vs NEW.
static bool RunPrevCase(std::mt19937& rng) {
  std::uniform_int_distribution<int> pd(0, 6);
  std::uniform_int_distribution<int> nd(0, 18);
  const size_t prev = pd(rng);
  const size_t extra = nd(rng);
  const size_t N = prev + extra;
  if (N == 0) return true;
  std::vector<Histo> out0(N);
  for (size_t i = 0; i < N; ++i) out0[i].id = (int)i;

  // symbols: each in [0, N). Pre-existing [0,prev) may appear; new ones >=prev
  // appear in arbitrary order. Mirror FastCluster output: any symbol in [0,N).
  std::uniform_int_distribution<int> ld(1, 40);
  size_t L = ld(rng);
  std::vector<uint32_t> syms(L);
  std::uniform_int_distribution<int> sd(0, (int)N - 1);
  for (auto& s : syms) s = sd(rng);

  std::vector<Histo> outO = out0, outN = out0;
  std::vector<uint32_t> symsO = syms, symsN = syms;
  ReindexOld(&outO, prev, &symsO);
  ReindexNew(&outN, prev, &symsN);

  if (outO.size() != outN.size()) return false;
  for (size_t i = 0; i < outO.size(); ++i)
    if (outO[i].id != outN[i].id) return false;
  return symsO == symsN;
}

int main() {
  std::mt19937 rng(0xC0FFEE);
  const int kKBest = 2000000;
  const int kPrev = 2000000;
  long kbest_ok = 0, prev_ok = 0;
  int fd = 0;
  for (int t = 0; t < kKBest; ++t) {
    if (!RunCase(rng, fd)) {
      printf("KBEST MISMATCH at case %d (detail=%d)\n", t, fd);
      return 1;
    }
    ++kbest_ok;
  }
  for (int t = 0; t < kPrev; ++t) {
    if (!RunPrevCase(rng)) {
      printf("PREV MISMATCH at case %d\n", t);
      return 1;
    }
    ++prev_ok;
  }
  printf("ALL PASS: kBest=%ld  prev-path=%ld  byte-identical out+symbols\n",
         kbest_ok, prev_ok);
  return 0;
}
