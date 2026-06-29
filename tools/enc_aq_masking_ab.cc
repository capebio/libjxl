// Isolated A/B for the AQ masking-difference precompute loop in
// AdaptiveQuantizationImpl::ComputeTile (lib/jxl/enc_adaptive_quantization.cc).
//
// OLD = per-row scalar/SIMD diff written to diff_buffer with a read-modify-write
//       accumulation across 4 rows, then a 4-column reduce into pre_erosion.
// NEW = the four rows summed in registers and stored once (no diff_buffer RMW).
//
// Both share the exact RatioOfDerivatives / MaskingSqrt helpers and keep every
// pixel on the same scalar/SIMD path, so any difference is purely the row-
// accumulation order — which is byte-exact by FP-add commutativity. This bench
// asserts pre_erosion is bit-identical and times both arms interleaved.
//
// Single static Highway target (compile with -mavx2 -mfma). The byte-exactness
// argument is lane-width independent, so an AVX2 result also covers wasm/SSE.
//
// Build:
//   clang++ -O2 -mavx2 -mfma -std=c++17 -I third_party/highway \
//       tools/enc_aq_masking_ab.cc -o enc_aq_masking_ab
// Run: ./enc_aq_masking_ab   (exit !=0 on any byte mismatch)

#include <hwy/highway.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace hn = hwy::HWY_NAMESPACE;
using namespace hwy::HWY_NAMESPACE;  // NOLINT — match libjxl call style

// --- constants (from enc_adaptive_quantization.cc) --------------------------
static constexpr float kInvLog2e = 0.6931471805599453f;  // ln(2)
static const float kSGmul = 226.77216153508914f;
static const float kSGmul2 = 1.0f / 73.377132366608819f;
static const float kSGRetMul = kSGmul2 * 18.6580932135f * kInvLog2e;
static const float kSGVOffset = 7.7825991679894591f;
static const float match_gamma_offset = 0.019f;
static const float limit = 0.2f;

template <bool invert, typename D, typename V>
static V RatioOfDeriv(const D d, V v) {
  float kEpsilon = 1e-2;
  v = ZeroIfNegative(v);
  const auto kNumMul = Set(d, kSGRetMul * 3 * kSGmul);
  const auto kVOffset = Set(d, kSGVOffset * kInvLog2e + kEpsilon);
  const auto kDenMul = Set(d, kInvLog2e * kSGmul);
  const auto v2 = Mul(v, v);
  const auto num = MulAdd(kNumMul, v2, Set(d, kEpsilon));
  const auto den = MulAdd(Mul(kDenMul, v), v2, kVOffset);
  return invert ? Div(num, den) : Div(den, num);
}
template <bool invert = false>
static float RatioOfDeriv(float v) {
  hn::CappedTag<float, 1> d1;
  auto vs = Load(d1, &v);
  return GetLane(RatioOfDeriv<invert>(d1, vs));
}

template <typename D, typename V>
static V MaskingSqrt(const D d, V v) {
  static const float kLogOffset = 27.505837037000106f;
  static const float kMul = 211.66567973503678f;
  static const float kSqrtMul = std::sqrt(static_cast<float>(kMul * 1e8));
  return Mul(Set(d, 0.25f), Sqrt(MulAdd(v, Set(d, kSqrtMul), Set(d, kLogOffset))));
}
static float MaskingSqrt(const float v) {
  hn::CappedTag<float, 1> d1;
  auto vs = Load(d1, &v);
  return GetLane(MaskingSqrt(d1, vs));
}

// --- a single-plane float image --------------------------------------------
struct Plane {
  int xs = 0, ys = 0;
  std::vector<float> d;
  void alloc(int x, int y) { xs = x; ys = y; d.assign((size_t)x * y, 0.f); }
  const float* Row(int y) const { return d.data() + (size_t)y * xs; }
  float* Row(int y) { return d.data() + (size_t)y * xs; }
};

// --- OLD: per-row diff_buffer RMW -------------------------------------------
static void OldLoop(const Plane& xyb, int x_start, int x_end, int y_start,
                    int y_end, float* diff_buffer, Plane* pre_erosion) {
  const int xsize = xyb.xs, ysize = xyb.ys;
  const D df;
  for (int y = y_start; y < y_end; ++y) {
    int y2 = y + 1 < ysize ? y + 1 : y;
    int y1 = y > 0 ? y - 1 : y;
    const float* row_in = xyb.Row(y);
    const float* row_in1 = xyb.Row(y1);
    const float* row_in2 = xyb.Row(y2);
    float* row_out = diff_buffer;
    auto scalar_pixel = [&](int x) {
      const int x2 = x + 1 < xsize ? x + 1 : x;
      const int x1 = x > 0 ? x - 1 : x;
      const float base = 0.25f * (row_in2[x] + row_in1[x] + row_in[x1] + row_in[x2]);
      const float gammac = RatioOfDeriv(row_in[x] + match_gamma_offset);
      float diff = gammac * (row_in[x] - base);
      diff *= diff;
      if (diff >= limit) diff = limit;
      diff = MaskingSqrt(diff);
      if ((y % 4) != 0) row_out[x - x_start] += diff;
      else row_out[x - x_start] = diff;
    };
    int x = x_start;
    if (x_start == 0) { scalar_pixel(x_start); ++x; }
    const auto mgo = Set(df, match_gamma_offset);
    const auto quarter = Set(df, 0.25f);
    for (; x + 1 + (int)Lanes(df) < x_end; x += Lanes(df)) {
      const auto in = LoadU(df, row_in + x);
      const auto in_r = LoadU(df, row_in + x + 1);
      const auto in_l = LoadU(df, row_in + x - 1);
      const auto in_t = LoadU(df, row_in2 + x);
      const auto in_b = LoadU(df, row_in1 + x);
      auto base = Mul(quarter, Add(Add(in_r, in_l), Add(in_t, in_b)));
      auto gv = RatioOfDeriv<false>(df, Add(in, mgo));
      auto diff = Mul(gv, Sub(in, base));
      diff = Mul(diff, diff);
      diff = Min(diff, Set(df, limit));
      diff = MaskingSqrt(df, diff);
      if ((y & 3) != 0) diff = Add(diff, LoadU(df, row_out + x - x_start));
      StoreU(diff, df, row_out + x - x_start);
    }
    for (; x < x_end; ++x) scalar_pixel(x);
    if (y % 4 == 3) {
      float* row_d_out = pre_erosion->Row((y - y_start) / 4);
      for (int qx = 0; qx < (x_end - x_start) / 4; qx++)
        row_d_out[qx] = (row_out[qx*4] + row_out[qx*4+1] + row_out[qx*4+2] +
                         row_out[qx*4+3]) * 0.25f;
    }
  }
}

// --- NEW: four rows summed in registers, stored once ------------------------
static void NewLoop(const Plane& xyb, int x_start, int x_end, int y_start,
                    int y_end, float* diff_buffer, Plane* pre_erosion) {
  const int xsize = xyb.xs, ysize = xyb.ys;
  const D df;
  const auto mgo = Set(df, match_gamma_offset);
  const auto quarter = Set(df, 0.25f);
  const auto limit_v = Set(df, limit);
  auto scalar_diff = [&](const float* r, const float* r1, const float* r2,
                         int x) -> float {
    const int x2 = x + 1 < xsize ? x + 1 : x;
    const int x1 = x > 0 ? x - 1 : x;
    const float base = 0.25f * (r2[x] + r1[x] + r[x1] + r[x2]);
    const float gammac = RatioOfDeriv(r[x] + match_gamma_offset);
    float diff = gammac * (r[x] - base);
    diff *= diff;
    if (diff >= limit) diff = limit;
    return MaskingSqrt(diff);
  };
  auto simd_diff = [&](const float* r, const float* r1, const float* r2, int x) {
    const auto in = LoadU(df, r + x);
    const auto in_r = LoadU(df, r + x + 1);
    const auto in_l = LoadU(df, r + x - 1);
    const auto in_t = LoadU(df, r2 + x);
    const auto in_b = LoadU(df, r1 + x);
    auto base = Mul(quarter, Add(Add(in_r, in_l), Add(in_t, in_b)));
    auto gv = RatioOfDeriv<false>(df, Add(in, mgo));
    auto diff = Mul(gv, Sub(in, base));
    diff = Mul(diff, diff);
    diff = Min(diff, limit_v);
    return MaskingSqrt(df, diff);
  };
  for (int yg = y_start; yg < y_end; yg += 4) {
    float* row_out = diff_buffer;
    const float* r[4]; const float* r1[4]; const float* r2[4];
    for (int k = 0; k < 4; ++k) {
      int yy = yg + k;
      int yy2 = yy + 1 < ysize ? yy + 1 : yy;
      int yy1 = yy > 0 ? yy - 1 : yy;
      r[k] = xyb.Row(yy); r1[k] = xyb.Row(yy1); r2[k] = xyb.Row(yy2);
    }
    int x = x_start;
    if (x_start == 0) {
      float s = scalar_diff(r[0], r1[0], r2[0], x);
      s += scalar_diff(r[1], r1[1], r2[1], x);
      s += scalar_diff(r[2], r1[2], r2[2], x);
      s += scalar_diff(r[3], r1[3], r2[3], x);
      row_out[x - x_start] = s; ++x;
    }
    for (; x + 1 + (int)Lanes(df) < x_end; x += Lanes(df)) {
      auto s = simd_diff(r[0], r1[0], r2[0], x);
      s = Add(s, simd_diff(r[1], r1[1], r2[1], x));
      s = Add(s, simd_diff(r[2], r1[2], r2[2], x));
      s = Add(s, simd_diff(r[3], r1[3], r2[3], x));
      StoreU(s, df, row_out + x - x_start);
    }
    for (; x < x_end; ++x) {
      float s = scalar_diff(r[0], r1[0], r2[0], x);
      s += scalar_diff(r[1], r1[1], r2[1], x);
      s += scalar_diff(r[2], r1[2], r2[2], x);
      s += scalar_diff(r[3], r1[3], r2[3], x);
      row_out[x - x_start] = s;
    }
    float* row_d_out = pre_erosion->Row((yg - y_start) / 4);
    for (int qx = 0; qx < (x_end - x_start) / 4; qx++)
      row_d_out[qx] = (row_out[qx*4] + row_out[qx*4+1] + row_out[qx*4+2] +
                       row_out[qx*4+3]) * 0.25f;
  }
}

static uint64_t fnv(const Plane& p) {
  uint64_t h = 1469598103934665603ull;
  const auto* b = reinterpret_cast<const uint8_t*>(p.d.data());
  for (size_t i = 0, n = p.d.size() * sizeof(float); i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}
struct Rng { uint64_t s; explicit Rng(uint64_t x):s(x){}
  uint32_t n(){ s=s*6364136223846793005ull+1442695040888963407ull; return (uint32_t)(s>>33);}
  float u(){ return (n()&0xffffff)/(float)0xffffff; } };

int main() {
  // whole-image tiles (x_start=0,x_end=W,y_start=0,y_end=H): exercises all edge
  // clamps + first-pixel-scalar + scalar-tail. W,H multiples of 8.
  const int sizes[][2] = {{64,64},{256,256},{512,384},{1024,768},{2048,1536}};
  bool all_exact = true; double tot_old = 0, tot_new = 0;
  for (auto& wh : sizes) {
    const int W = wh[0], H = wh[1];
    // The SIMD guard (x + 1 + Lanes < x_end) keeps every load within [0, W), so
    // a tight W*H image is safe; ties injected to exercise the selection.
    Plane img; img.alloc(W, H);
    Rng rng(0x1234567ull ^ (uint64_t)W * 2654435761ull);
    for (auto& v : img.d) { float r = rng.u() * 0.3f; if ((rng.n() & 3) == 0) r = std::floor(r*8)/8; v = r; }

    const int xs0 = 0, xs1 = W, ys0 = 0, ys1 = H;
    std::vector<float> dbuf_o(W + 16, 0.f), dbuf_n(W + 16, 0.f);
    Plane pe_o, pe_n; pe_o.alloc((xs1-xs0)/4, (ys1-ys0)/4); pe_n.alloc((xs1-xs0)/4, (ys1-ys0)/4);
    for (auto& v : pe_o.d) v = -7.f; for (auto& v : pe_n.d) v = -7.f;

    OldLoop(img, xs0, xs1, ys0, ys1, dbuf_o.data(), &pe_o);
    NewLoop(img, xs0, xs1, ys0, ys1, dbuf_n.data(), &pe_n);
    bool exact = fnv(pe_o) == fnv(pe_n);
    all_exact &= exact;

    const int rounds = W >= 1024 ? 25 : 400;
    std::vector<double> to, tn; volatile float sink = 0;
    for (int rr = 0; rr < rounds; ++rr) {
      if (rr & 1) {
        auto a = std::chrono::high_resolution_clock::now();
        NewLoop(img, xs0, xs1, ys0, ys1, dbuf_n.data(), &pe_n);
        auto b = std::chrono::high_resolution_clock::now();
        OldLoop(img, xs0, xs1, ys0, ys1, dbuf_o.data(), &pe_o);
        auto c = std::chrono::high_resolution_clock::now();
        tn.push_back(std::chrono::duration<double,std::micro>(b-a).count());
        to.push_back(std::chrono::duration<double,std::micro>(c-b).count());
      } else {
        auto a = std::chrono::high_resolution_clock::now();
        OldLoop(img, xs0, xs1, ys0, ys1, dbuf_o.data(), &pe_o);
        auto b = std::chrono::high_resolution_clock::now();
        NewLoop(img, xs0, xs1, ys0, ys1, dbuf_n.data(), &pe_n);
        auto c = std::chrono::high_resolution_clock::now();
        to.push_back(std::chrono::duration<double,std::micro>(b-a).count());
        tn.push_back(std::chrono::duration<double,std::micro>(c-b).count());
      }
      sink += pe_o.d[0] + pe_n.d[0];
    }
    (void)sink;
    std::sort(to.begin(), to.end()); std::sort(tn.begin(), tn.end());
    double mo = to[to.size()/2], mn = tn[tn.size()/2];
    tot_old += mo; tot_new += mn;
    printf("%4dx%-4d  %-11s  old=%9.2fus new=%9.2fus  saved=%+.1f%%\n",
           W, H, exact ? "BYTE-EXACT" : "*MISMATCH*", mo, mn, 100.0*(mo-mn)/mo);
  }
  printf("\n%s  aggregate median sum old=%.1fus new=%.1fus saved=%+.1f%%\n",
         all_exact ? "ALL BYTE-EXACT" : "*** MISMATCH ***", tot_old, tot_new,
         100.0*(tot_old-tot_new)/tot_old);
  return all_exact ? 0 : 1;
}
