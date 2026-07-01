// Standalone A/B for the enc_lz77.cc special-distance lookup change.
//
// OLD: std::unordered_map<int,int> built by counting down (smallest code wins
//      on duplicate distance), queried with find() in the inner match loop.
// NEW: fixed 256-slot open-addressed table (key 0 == empty; special distances
//      are always >= 1), same descending overwrite build, array lookup.
//
// Proves: (1) byte-exact — the returned special code (and the full dist_symbol
// the caller derives from it) is identical for every distance and multiplier;
// (2) faster — the array probe beats node-chasing in unordered_map.
//
// Build:  clang++ -std=c++17 -O2 tools/enc_lz77_special_dist_ab.cc -o adist
// Run:    ./adist

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <vector>

// ---- copied verbatim from lib/jxl/dec_ans.h --------------------------------
static constexpr size_t kNumSpecialDistances = 120;
static constexpr int8_t kSpecialDistances[kNumSpecialDistances][2] = {
    {0, 1},  {1, 0},  {1, 1},  {-1, 1}, {0, 2},  {2, 0},  {1, 2},  {-1, 2},
    {2, 1},  {-2, 1}, {2, 2},  {-2, 2}, {0, 3},  {3, 0},  {1, 3},  {-1, 3},
    {3, 1},  {-3, 1}, {2, 3},  {-2, 3}, {3, 2},  {-3, 2}, {0, 4},  {4, 0},
    {1, 4},  {-1, 4}, {4, 1},  {-4, 1}, {3, 3},  {-3, 3}, {2, 4},  {-2, 4},
    {4, 2},  {-4, 2}, {0, 5},  {3, 4},  {-3, 4}, {4, 3},  {-4, 3}, {5, 0},
    {1, 5},  {-1, 5}, {5, 1},  {-5, 1}, {2, 5},  {-2, 5}, {5, 2},  {-5, 2},
    {4, 4},  {-4, 4}, {3, 5},  {-3, 5}, {5, 3},  {-5, 3}, {0, 6},  {6, 0},
    {1, 6},  {-1, 6}, {6, 1},  {-6, 1}, {2, 6},  {-2, 6}, {6, 2},  {-6, 2},
    {4, 5},  {-4, 5}, {5, 4},  {-5, 4}, {3, 6},  {-3, 6}, {6, 3},  {-6, 3},
    {0, 7},  {7, 0},  {1, 7},  {-1, 7}, {5, 5},  {-5, 5}, {7, 1},  {-7, 1},
    {4, 6},  {-4, 6}, {6, 4},  {-6, 4}, {2, 7},  {-2, 7}, {7, 2},  {-7, 2},
    {3, 7},  {-3, 7}, {7, 3},  {-7, 3}, {5, 6},  {-5, 6}, {6, 5},  {-6, 5},
    {8, 0},  {4, 7},  {-4, 7}, {7, 4},  {-7, 4}, {8, 1},  {8, 2},  {6, 6},
    {-6, 6}, {8, 3},  {5, 7},  {-5, 7}, {7, 5},  {-7, 5}, {8, 4},  {6, 7},
    {-6, 7}, {7, 6},  {-7, 6}, {8, 5},  {7, 7},  {-7, 7}, {8, 6},  {8, 7}};
static inline int SpecialDistance(size_t index, int multiplier) {
  int dist = kSpecialDistances[index][0] +
             static_cast<int>(multiplier) * kSpecialDistances[index][1];
  return (dist > 1) ? dist : 1;
}

// ---- OLD ------------------------------------------------------------------
struct OldTable {
  std::unordered_map<int, int> t;
  void Build(int mult) {
    if (!mult) return;
    for (int i = kNumSpecialDistances - 1; i >= 0; --i)
      t[SpecialDistance(i, mult)] = i;
  }
  int Lookup(int key) const {
    auto it = t.find(key);
    return it == t.end() ? -1 : it->second;
  }
};

// ---- NEW (identical to the patched HashChain fixed table) ------------------
struct NewTable {
  static constexpr size_t kSlots = 256;
  static constexpr uint32_t kMask = kSlots - 1;
  int keys[kSlots] = {};
  int16_t codes[kSlots] = {};
  static uint32_t Hash(int key) {
    return (static_cast<uint32_t>(key) * 2654435761u) >> 24;
  }
  void Insert(int key, int code) {
    uint32_t s = Hash(key);
    for (;;) {
      if (keys[s] == 0 || keys[s] == key) {
        keys[s] = key;
        codes[s] = static_cast<int16_t>(code);
        return;
      }
      s = (s + 1) & kMask;
    }
  }
  void Build(int mult) {
    if (!mult) return;
    for (int i = kNumSpecialDistances - 1; i >= 0; --i)
      Insert(SpecialDistance(i, mult), i);
  }
  int Lookup(int key) const {
    uint32_t s = Hash(key);
    for (;;) {
      int k = keys[s];
      if (k == key) return codes[s];
      if (k == 0) return -1;
      s = (s + 1) & kMask;
    }
  }
};

int main() {
  const int mults[] = {0, 1, 2, 3, 7, 16, 64, 100, 640, 1920, 4096, 8192};
  long long checks = 0, mism = 0;
  // 1) Byte-exact: every distance & multiplier, hits and misses.
  for (int mult : mults) {
    OldTable o;
    NewTable n;
    o.Build(mult);
    n.Build(mult);
    size_t num_special = mult == 0 ? 0 : kNumSpecialDistances;
    int maxd = mult == 0 ? 40000 : 8 * mult + 32;
    for (int dist = 1; dist <= maxd; dist++) {
      int co = o.Lookup(dist), cn = n.Lookup(dist);
      // full dist_symbol as the caller computes it
      long long so = (co < 0) ? (long long)(num_special + dist - 1) : co;
      long long sn = (cn < 0) ? (long long)(num_special + dist - 1) : cn;
      checks++;
      if (co != cn || so != sn) {
        if (mism < 10)
          printf("MISMATCH mult=%d dist=%d: old code=%d sym=%lld | new code=%d sym=%lld\n",
                 mult, dist, co, so, cn, sn);
        mism++;
      }
    }
  }
  printf("byte-exact: %lld checks, %lld mismatches\n", checks, mism);

  // 2) Timing: mixed hit/miss workload, interleaved OLD/NEW rounds.
  const int mult = 1920;  // typical HD width
  OldTable o;
  NewTable n;
  o.Build(mult);
  n.Build(mult);
  std::vector<int> probe;
  probe.reserve(1 << 16);
  for (int k = 0; k < (1 << 16); k++) {
    // interleave special keys (hits) with arbitrary distances (misses)
    if (k & 1)
      probe.push_back(SpecialDistance(k % kNumSpecialDistances, mult));
    else
      probe.push_back((k * 2654435761u) % 40000 + 1);
  }
  volatile long long sink = 0;
  double old_ns = 1e300, new_ns = 1e300;
  for (int round = 0; round < 7; round++) {
    // OLD
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      long long acc = 0;
      for (int rep = 0; rep < 64; rep++)
        for (int d : probe) acc += o.Lookup(d);
      auto t1 = std::chrono::high_resolution_clock::now();
      sink += acc;
      double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() /
                  (64.0 * probe.size());
      if (ns < old_ns) old_ns = ns;
    }
    // NEW
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      long long acc = 0;
      for (int rep = 0; rep < 64; rep++)
        for (int d : probe) acc += n.Lookup(d);
      auto t1 = std::chrono::high_resolution_clock::now();
      sink += acc;
      double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() /
                  (64.0 * probe.size());
      if (ns < new_ns) new_ns = ns;
    }
  }
  printf("lookup ns/op (min of 7): OLD=%.3f  NEW=%.3f  speedup=%.2fx\n",
         old_ns, new_ns, old_ns / new_ns);
  printf("sink=%lld\n", (long long)sink);
  return mism == 0 ? 0 : 1;
}
