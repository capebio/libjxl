// Self-contained A/B flip-flop for the FuzzyErosion optimization in
// lib/jxl/enc_adaptive_quantization.cc.
//
// FuzzyErosion is pure scalar C++ (no Highway), so we can lift the OLD and NEW
// bodies verbatim into one translation unit and prove the rewrite is bit-exact
// while measuring the speedup. Both arms share StoreMin4 and are compiled with
// identical flags, so any FP contraction applies equally — a byte difference
// can only come from the index-mapping / accumulation-order change itself.
//
// Build (no deps):
//   clang++ -O2 -std=c++17 tools/enc_aq_fuzzy_ab.cc -o enc_aq_fuzzy_ab
// Run:
//   ./enc_aq_fuzzy_ab
//
// Exits non-zero if any configuration is not byte-identical.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

// ---- minimal ImageF / Rect mocks matching libjxl addressing ----------------
struct Img {
  int xs = 0, ys = 0;
  std::vector<float> d;
  void alloc(int x, int y) {
    xs = x;
    ys = y;
    d.assign(static_cast<size_t>(x) * y, 0.f);
  }
  float* Row(int y) { return d.data() + static_cast<size_t>(y) * xs; }
  const float* Row(int y) const { return d.data() + static_cast<size_t>(y) * xs; }
  int xsize() const { return xs; }
  int ysize() const { return ys; }
};
struct Rect {
  int x0_, y0_, xs_, ys_;
  int x0() const { return x0_; }
  int y0() const { return y0_; }
  int xsize() const { return xs_; }
  int ysize() const { return ys_; }
  // libjxl: to_rect.Row(to, r) == to->Row(y0()+r) + x0()
  float* Row(Img& to, int r) const { return to.Row(y0_ + r) + x0_; }
};

// ---- shared helper (identical to lib/jxl) ----------------------------------
static inline void StoreMin4(const float v, float& min0, float& min1,
                             float& min2, float& min3) {
  if (v < min3) {
    if (v < min0) {
      min3 = min2;
      min2 = min1;
      min1 = min0;
      min0 = v;
    } else if (v < min1) {
      min3 = min2;
      min2 = min1;
      min1 = v;
    } else if (v < min2) {
      min3 = min2;
      min2 = v;
    } else {
      min3 = v;
    }
  }
}

static inline float FuzzyErosionRank4(const float center, const float left,
                                      const float right, const float top_left,
                                      const float top, const float top_right,
                                      const float bottom_left,
                                      const float bottom,
                                      const float bottom_right, const float k0,
                                      const float k1, const float k2,
                                      const float k3) {
  float min0 = center;
  float min1 = left;
  float min2 = right;
  float min3 = top_left;
  if (min0 > min1) std::swap(min0, min1);
  if (min0 > min2) std::swap(min0, min2);
  if (min0 > min3) std::swap(min0, min3);
  if (min1 > min2) std::swap(min1, min2);
  if (min1 > min3) std::swap(min1, min3);
  if (min2 > min3) std::swap(min2, min3);
  StoreMin4(top, min0, min1, min2, min3);
  StoreMin4(top_right, min0, min1, min2, min3);
  StoreMin4(bottom_left, min0, min1, min2, min3);
  StoreMin4(bottom, min0, min1, min2, min3);
  StoreMin4(bottom_right, min0, min1, min2, min3);
  return k0 * min0 + k1 * min1 + k2 * min2 + k3 * min3;
}

static void ComputeKMul(float bt, float kMul[4]) {
  static const float kMulBase[4] = {0.125, 0.1, 0.09, 0.06};
  static const float kMulAdd[4] = {0.0, -0.1, -0.09, -0.06};
  float mul = 0.0;
  if (bt < 2.0f) mul = (2.0f - bt) * (1.0f / 2.0f);
  float norm_sum = 0.0;
  for (int ii = 0; ii < 4; ++ii) {
    kMul[ii] = kMulBase[ii] + mul * kMulAdd[ii];
    norm_sum += kMul[ii];
  }
  static const float kTotal = 0.29959705784054957;
  for (int ii = 0; ii < 4; ++ii) kMul[ii] *= kTotal / norm_sum;
}

// ---- OLD: verbatim from libjxl @ 2169106a ----------------------------------
static void FuzzyErosion_OLD(float bt, const Rect& from_rect, const Img& from,
                             const Rect& to_rect, Img* to) {
  const int xsize = from.xsize();
  const int ysize = from.ysize();
  constexpr int kStep = 1;
  float kMul[4];
  ComputeKMul(bt, kMul);
  for (int fy = 0; fy < from_rect.ysize(); ++fy) {
    int y = fy + from_rect.y0();
    int ym1 = y >= kStep ? y - kStep : y;
    int yp1 = y + kStep < ysize ? y + kStep : y;
    const float* rowt = from.Row(ym1);
    const float* row = from.Row(y);
    const float* rowb = from.Row(yp1);
    float* row_out = to_rect.Row(*to, fy / 2);
    for (int fx = 0; fx < from_rect.xsize(); ++fx) {
      int x = fx + from_rect.x0();
      int xm1 = x >= kStep ? x - kStep : x;
      int xp1 = x + kStep < xsize ? x + kStep : x;
      float mn[4] = {row[x], row[xm1], row[xp1], rowt[xm1]};
      if (mn[0] > mn[1]) std::swap(mn[0], mn[1]);
      if (mn[0] > mn[2]) std::swap(mn[0], mn[2]);
      if (mn[0] > mn[3]) std::swap(mn[0], mn[3]);
      if (mn[1] > mn[2]) std::swap(mn[1], mn[2]);
      if (mn[1] > mn[3]) std::swap(mn[1], mn[3]);
      if (mn[2] > mn[3]) std::swap(mn[2], mn[3]);
      StoreMin4(rowt[x], mn[0], mn[1], mn[2], mn[3]);
      StoreMin4(rowt[xp1], mn[0], mn[1], mn[2], mn[3]);
      StoreMin4(rowb[xm1], mn[0], mn[1], mn[2], mn[3]);
      StoreMin4(rowb[x], mn[0], mn[1], mn[2], mn[3]);
      StoreMin4(rowb[xp1], mn[0], mn[1], mn[2], mn[3]);
      float v = kMul[0] * mn[0] + kMul[1] * mn[1] + kMul[2] * mn[2] +
                kMul[3] * mn[3];
      if (fx % 2 == 0 && fy % 2 == 0) {
        row_out[fx / 2] = v;
      } else {
        row_out[fx / 2] += v;
      }
    }
  }
}

// ---- NEW: verbatim from the optimized libjxl --------------------------------
static void FuzzyErosion_NEW(float bt, const Rect& from_rect, const Img& from,
                             const Rect& to_rect, Img* to) {
  const int xsize = from.xsize();
  const int ysize = from.ysize();
  constexpr int kStep = 1;
  float kMul[4];
  ComputeKMul(bt, kMul);
  for (int oy = 0; oy < to_rect.ysize(); ++oy) {
    const int ya = from_rect.y0() + 2 * oy;
    const int ym1 = ya >= kStep ? ya - kStep : ya;
    const int y1 = ya + kStep < ysize ? ya + kStep : ya;
    const int y2 = y1 + kStep < ysize ? y1 + kStep : y1;
    const float* row_m1 = from.Row(ym1);
    const float* row0 = from.Row(ya);
    const float* row1 = from.Row(y1);
    const float* row2 = from.Row(y2);
    float* row_out = to_rect.Row(*to, oy);
    for (int ox = 0; ox < to_rect.xsize(); ++ox) {
      const int xa = from_rect.x0() + 2 * ox;
      const int xm1 = xa >= kStep ? xa - kStep : xa;
      const int x1 = xa + kStep < xsize ? xa + kStep : xa;
      const int x2 = x1 + kStep < xsize ? x1 + kStep : x1;
      const float va = FuzzyErosionRank4(
          row0[xa], row0[xm1], row0[x1], row_m1[xm1], row_m1[xa], row_m1[x1],
          row1[xm1], row1[xa], row1[x1], kMul[0], kMul[1], kMul[2], kMul[3]);
      const float vb = FuzzyErosionRank4(
          row0[x1], row0[xa], row0[x2], row_m1[xa], row_m1[x1], row_m1[x2],
          row1[xa], row1[x1], row1[x2], kMul[0], kMul[1], kMul[2], kMul[3]);
      const float vc = FuzzyErosionRank4(
          row1[xa], row1[xm1], row1[x1], row0[xm1], row0[xa], row0[x1],
          row2[xm1], row2[xa], row2[x1], kMul[0], kMul[1], kMul[2], kMul[3]);
      const float vd = FuzzyErosionRank4(
          row1[x1], row1[xa], row1[x2], row0[xa], row0[x1], row0[x2],
          row2[xa], row2[x1], row2[x2], kMul[0], kMul[1], kMul[2], kMul[3]);
      row_out[ox] = ((va + vb) + vc) + vd;
    }
  }
}

// ---- harness ----------------------------------------------------------------
static uint64_t fnv(const Img& m) {
  uint64_t h = 1469598103934665603ull;
  const auto* p = reinterpret_cast<const uint8_t*>(m.d.data());
  size_t n = m.d.size() * sizeof(float);
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

// deterministic LCG
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint32_t next() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<uint32_t>(s >> 33);
  }
  float unit() { return (next() & 0xffffff) / static_cast<float>(0xffffff); }
};

struct Cfg {
  int to_x, to_y;       // output (block) dims
  int fx0, fy0;         // from_rect offset (0 or 1)
  bool tight_edge;      // make `from` just big enough so right/bottom clamp
  int to_off_x, to_off_y;  // nonzero to_rect offset into `to`
  float bt;             // butteraugli_target
};

int main() {
  const Cfg cfgs[] = {
      {2, 2, 0, 0, true, 0, 0, 1.0f},
      {3, 3, 1, 1, true, 1, 2, 0.5f},
      {4, 4, 0, 1, false, 2, 0, 2.5f},
      {16, 16, 1, 0, true, 0, 3, 1.7f},
      {32, 24, 1, 1, false, 5, 5, 0.9f},
      {64, 64, 0, 0, true, 0, 0, 3.0f},
      {128, 96, 1, 1, true, 3, 1, 1.3f},
      {256, 256, 1, 0, false, 0, 0, 0.2f},
      {512, 384, 0, 1, true, 2, 2, 1.0f},
  };
  const int nc = sizeof(cfgs) / sizeof(cfgs[0]);

  bool all_exact = true;
  // Per-config large timing accumulation (interleaved A/B, start-rotated).
  double t_old_total = 0, t_new_total = 0;

  for (int ci = 0; ci < nc; ++ci) {
    const Cfg& c = cfgs[ci];
    const int from_xs = c.to_x * 2, from_ys = c.to_y * 2;
    // `from` image: from_rect occupies [fx0, fx0+from_xs) x [fy0, fy0+from_ys).
    // tight_edge => from is exactly that big (last col/row clamps);
    // else pad by +3 so interior never clamps at the far edge.
    const int pad = c.tight_edge ? 0 : 3;
    Img from;
    from.alloc(c.fx0 + from_xs + pad, c.fy0 + from_ys + pad);
    Rng rng(0x9e3779b97f4a7c15ull ^ (uint64_t)(ci + 1) * 0x123456789ull);
    for (auto& v : from.d) {
      float r = rng.unit() * 0.30f;
      // Inject ties: snap ~1/4 of values to a small quantized grid so the
      // StoreMin4 strict-less-than comparisons see equal neighbours.
      if ((rng.next() & 3) == 0) r = std::floor(r * 8.f) / 8.f;
      v = r;
    }

    // Output images (offset region inside a larger canvas).
    Img to_old, to_new;
    to_old.alloc(c.to_off_x + c.to_x + 2, c.to_off_y + c.to_y + 2);
    to_new.alloc(c.to_off_x + c.to_x + 2, c.to_off_y + c.to_y + 2);
    // Prefill with a sentinel to catch any cell the NEW path forgets to write.
    for (auto& v : to_old.d) v = -123.5f;
    for (auto& v : to_new.d) v = -123.5f;

    Rect from_rect{c.fx0, c.fy0, from_xs, from_ys};
    Rect to_rect{c.to_off_x, c.to_off_y, c.to_x, c.to_y};

    FuzzyErosion_OLD(c.bt, from_rect, from, to_rect, &to_old);
    FuzzyErosion_NEW(c.bt, from_rect, from, to_rect, &to_new);

    const uint64_t h_old = fnv(to_old), h_new = fnv(to_new);
    const bool exact = (h_old == h_new);
    all_exact &= exact;

    // timing: interleave, start-rotated, take median of warm rounds.
    const int rounds = c.to_x >= 128 ? 9 : 200;
    std::vector<double> dt_old, dt_new;
    volatile float sink = 0;
    for (int r = 0; r < rounds; ++r) {
      if (r & 1) {  // NEW first on odd rounds
        auto t0 = std::chrono::high_resolution_clock::now();
        FuzzyErosion_NEW(c.bt, from_rect, from, to_rect, &to_new);
        auto t1 = std::chrono::high_resolution_clock::now();
        FuzzyErosion_OLD(c.bt, from_rect, from, to_rect, &to_old);
        auto t2 = std::chrono::high_resolution_clock::now();
        dt_new.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        dt_old.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
      } else {
        auto t0 = std::chrono::high_resolution_clock::now();
        FuzzyErosion_OLD(c.bt, from_rect, from, to_rect, &to_old);
        auto t1 = std::chrono::high_resolution_clock::now();
        FuzzyErosion_NEW(c.bt, from_rect, from, to_rect, &to_new);
        auto t2 = std::chrono::high_resolution_clock::now();
        dt_old.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        dt_new.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
      }
      sink += to_new.d[0] + to_old.d[0];
    }
    (void)sink;
    std::sort(dt_old.begin(), dt_old.end());
    std::sort(dt_new.begin(), dt_new.end());
    const double mo = dt_old[dt_old.size() / 2];
    const double mn = dt_new[dt_new.size() / 2];
    t_old_total += mo;
    t_new_total += mn;
    const double saved = 100.0 * (mo - mn) / mo;

    printf("cfg %d  to=%dx%d fx0=%d fy0=%d %s tooff=%d,%d bt=%.2f  "
           "%s  old=%.2fus new=%.2fus  saved=%+.1f%%\n",
           ci, c.to_x, c.to_y, c.fx0, c.fy0,
           c.tight_edge ? "tight" : "pad  ", c.to_off_x, c.to_off_y, c.bt,
           exact ? "BYTE-EXACT" : "*** MISMATCH ***", mo, mn, saved);
  }

  printf("\n%s   aggregate median sum: old=%.1fus new=%.1fus  saved=%+.1f%%\n",
         all_exact ? "ALL BYTE-EXACT" : "*** BYTE MISMATCH ***", t_old_total,
         t_new_total, 100.0 * (t_old_total - t_new_total) / t_old_total);
  return all_exact ? 0 : 1;
}
