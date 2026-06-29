// Standalone A/B harness for the enc_convolve_separable5 BORDER-ROW dedup.
// OLD = border ConvolveRow computing five independent horizontal convolutions.
// NEW = same, but reusing the convolution whenever two source rows alias after
// mirrored-boundary remapping (HorzFiveDedup). Both arms share identical row-
// pointer setup (incl. the tiny-height LUT) so the only difference is the reuse.
//
// Verifies byte-exactness (FNV hash of the border output rows, OLD == NEW) over
// a width/height sweep that hits every kSizeModN and every border topology
// (top/bottom single borders + tiny-height double reflections), then times the
// border pass OLD vs NEW interleaved on wide images where border work dominates.
//
// Single Highway target (static dispatch): native -march=native -> AVX2/512,
// emcc -msimd128 -> WASM SIMD128 (4 lanes).
//
// Build native:  clang++ -O3 -march=native -std=c++17 -I<hwy> conv5_border_ab.cc -o conv5_border_ab
// Build wasm:    emcc -O3 -msimd128 -std=c++17 -I<hwy> conv5_border_ab.cc -o conv5_border_ab.js
//                  -s ENVIRONMENT=node -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=268435456

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <hwy/highway.h>
#if HWY_TARGET <= (1 << HWY_HIGHEST_TARGET_BIT_X86)
#include <xmmintrin.h>
#endif

#ifndef JXL_RESTRICT
#define JXL_RESTRICT __restrict__
#endif

namespace hn = hwy::HWY_NAMESPACE;

using D = hn::CappedTag<float, 16>;
using DI32 = hn::CappedTag<int32_t, 16>;
using V = hn::Vec<D>;
using VI32 = hn::Vec<DI32>;
using I = decltype(hn::SetTableIndices(D(), static_cast<int32_t*>(nullptr)));

static constexpr int64_t kRadius = 2;

static inline int64_t Mirror(int64_t x, const int64_t xsize) {
  while (x < 0 || x >= xsize) {
    if (x < 0) {
      x = -x - 1;
    } else {
      x = 2 * xsize - 1 - x;
    }
  }
  return x;
}

struct Neighbors {
  static V FirstL1(const V c) {
#if HWY_CAP_GE256
    const D d;
    HWY_ALIGN constexpr int32_t lanes[16] = {0, 0, 1, 2,  3,  4,  5,  6,
                                             7, 8, 9, 10, 11, 12, 13, 14};
    return hn::TableLookupLanes(c, hn::SetTableIndices(d, lanes));
#elif HWY_TARGET == HWY_SCALAR
    return c;
#else
#if HWY_TARGET <= (1 << HWY_HIGHEST_TARGET_BIT_X86)
    return V{_mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(2, 1, 0, 0))};
#else
    const D d;
    HWY_ALIGN constexpr int32_t lanes[4] = {0, 0, 1, 2};
    return hn::TableLookupLanes(c, hn::SetTableIndices(d, lanes));
#endif
#endif
  }
  static V FirstL2(const V c) {
#if HWY_CAP_GE256
    const D d;
    HWY_ALIGN constexpr int32_t lanes[16] = {1, 0, 0, 1, 2,  3,  4,  5,
                                             6, 7, 8, 9, 10, 11, 12, 13};
    return hn::TableLookupLanes(c, hn::SetTableIndices(d, lanes));
#elif HWY_TARGET == HWY_SCALAR
    const D d;
    return hn::Zero(d);
#else
#if HWY_TARGET <= (1 << HWY_HIGHEST_TARGET_BIT_X86)
    return V{_mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(1, 0, 0, 1))};
#else
    const D d;
    HWY_ALIGN constexpr int32_t lanes[4] = {1, 0, 0, 1};
    return hn::TableLookupLanes(c, hn::SetTableIndices(d, lanes));
#endif
#endif
  }
};

template <int M>
static I MirrorLanes() {
  D d;
  DI32 di32;
  const VI32 up = hn::Min(hn::Iota(di32, M), hn::Set(di32, hn::Lanes(d) - 1));
  const VI32 down =
      hn::Max(hn::Iota(di32, M - static_cast<int>(hn::Lanes(d))), hn::Zero(di32));
  return hn::IndicesFromVec(d, hn::Sub(up, down));
}

struct Weights {
  float horz[20];
  float vert[20];
};

static V HorzConvolveFirst(const float* JXL_RESTRICT row, const int64_t x,
                           const int64_t xsize, const V wh0, const V wh1,
                           const V wh2) {
  const D d;
  const V c = hn::LoadU(d, row + x);
  const V mul0 = hn::Mul(c, wh0);
#if HWY_TARGET == HWY_SCALAR
  const V l1 = hn::LoadU(d, row + Mirror(x - 1, xsize));
  const V l2 = hn::LoadU(d, row + Mirror(x - 2, xsize));
#else
  (void)xsize;
  const V l1 = Neighbors::FirstL1(c);
  const V l2 = Neighbors::FirstL2(c);
#endif
  const V r1 = hn::LoadU(d, row + x + 1);
  const V r2 = hn::LoadU(d, row + x + 2);
  const V mul1 = hn::MulAdd(hn::Add(l1, r1), wh1, mul0);
  const V mul2 = hn::MulAdd(hn::Add(l2, r2), wh2, mul1);
  return mul2;
}

template <size_t kSizeModN>
static V HorzConvolveLast(const float* JXL_RESTRICT row, const int64_t x,
                          const int64_t xsize, const V wh0, const V wh1,
                          const V wh2, const I ml1, const I ml2) {
  const D d;
  const V c = hn::LoadU(d, row + x);
  const V mul0 = hn::Mul(c, wh0);
  const V l1 = hn::LoadU(d, row + x - 1);
  const V l2 = hn::LoadU(d, row + x - 2);
  V r1, r2;
#if HWY_TARGET == HWY_SCALAR
  r1 = hn::LoadU(d, row + Mirror(x + 1, xsize));
  r2 = hn::LoadU(d, row + Mirror(x + 2, xsize));
  (void)ml1;
  (void)ml2;
#else
  const size_t N = hn::Lanes(d);
  if (kSizeModN == 0) {
    r2 = hn::TableLookupLanes(c, ml2);
    r1 = hn::TableLookupLanes(c, ml1);
  } else {
    const auto last = hn::LoadU(d, row + xsize - N);
    r2 = hn::TableLookupLanes(last, ml1);
    r1 = last;
  }
#endif
  const V mul1 = hn::MulAdd(hn::Add(l1, r1), wh1, mul0);
  const V mul2 = hn::MulAdd(hn::Add(l2, r2), wh2, mul1);
  return mul2;
}

static V HorzConvolve(const float* JXL_RESTRICT pos, const V wh0, const V wh1,
                      const V wh2) {
  const D d;
  const V c = hn::LoadU(d, pos);
  const V mul0 = hn::Mul(c, wh0);
  const V l1 = hn::LoadU(d, pos - 1);
  const V r1 = hn::LoadU(d, pos + 1);
  const V l2 = hn::LoadU(d, pos - 2);
  const V r2 = hn::LoadU(d, pos + 2);
  const V mul1 = hn::MulAdd(hn::Add(l1, r1), wh1, mul0);
  const V mul2 = hn::MulAdd(hn::Add(l2, r2), wh2, mul1);
  return mul2;
}

template <size_t kSizeModN, int kRegion>
static V HorzPick(const float* JXL_RESTRICT row, const int64_t x,
                  const int64_t xsize, const V wh0, const V wh1, const V wh2,
                  const I ml1, const I ml2) {
  if (kRegion == 0) return HorzConvolveFirst(row, x, xsize, wh0, wh1, wh2);
  if (kRegion == 2)
    return HorzConvolveLast<kSizeModN>(row, x, xsize, wh0, wh1, wh2, ml1, ml2);
  (void)ml1;
  (void)ml2;
  (void)xsize;
  return HorzConvolve(row + x, wh0, wh1, wh2);
}

// Mirror of the production HorzFiveDedup.
template <size_t kSizeModN, int kRegion>
static void HorzFiveDedup(const float* r_t2, const float* r_t1,
                          const float* r_m, const float* r_b1,
                          const float* r_b2, const int64_t x,
                          const int64_t xsize, const V wh0, const V wh1,
                          const V wh2, const I ml1, const I ml2, V* h_t2,
                          V* h_t1, V* h_m, V* h_b1, V* h_b2) {
  *h_m = HorzPick<kSizeModN, kRegion>(r_m, x, xsize, wh0, wh1, wh2, ml1, ml2);
  *h_t1 = (r_t1 == r_m)
              ? *h_m
              : HorzPick<kSizeModN, kRegion>(r_t1, x, xsize, wh0, wh1, wh2, ml1,
                                             ml2);
  *h_b1 = (r_b1 == r_m)    ? *h_m
          : (r_b1 == r_t1) ? *h_t1
                           : HorzPick<kSizeModN, kRegion>(r_b1, x, xsize, wh0,
                                                          wh1, wh2, ml1, ml2);
  *h_t2 = (r_t2 == r_m)    ? *h_m
          : (r_t2 == r_t1) ? *h_t1
          : (r_t2 == r_b1) ? *h_b1
                           : HorzPick<kSizeModN, kRegion>(r_t2, x, xsize, wh0,
                                                          wh1, wh2, ml1, ml2);
  *h_b2 = (r_b2 == r_m)    ? *h_m
          : (r_b2 == r_t1) ? *h_t1
          : (r_b2 == r_b1) ? *h_b1
          : (r_b2 == r_t2) ? *h_t2
                           : HorzPick<kSizeModN, kRegion>(r_b2, x, xsize, wh0,
                                                          wh1, wh2, ml1, ml2);
}

struct Plane {
  std::vector<float> buf;
  int64_t xsize, ysize, stride;
  // kRadius guard rows above and below so the convolution's row_m +/- 2*stride
  // pointers are always formed in-bounds (production ImageF is laid out
  // differently; this just avoids UB the optimizer could exploit at -O3).
  Plane(int64_t xs, int64_t ys) : xsize(xs), ysize(ys) {
    const int64_t N = hn::Lanes(D());
    stride = ((xs + N - 1) / N) * N + 2 * N;
    buf.assign(stride * (ys + 2 * kRadius), 0.0f);
  }
  float* Row(int64_t y) { return buf.data() + (y + kRadius) * stride; }
  const float* ConstRow(int64_t y) const {
    return buf.data() + (y + kRadius) * stride;
  }
};

// One border row, shared pointer setup; kDedup selects 5-independent vs reuse.
template <size_t kSizeModN, bool kDedup>
static void BorderRow(const Plane& in, Plane& out, const Weights& w,
                      const int64_t y) {
  const D d;
  const int64_t N = hn::Lanes(d);
  const int64_t stride = in.stride;
  const int64_t neg_stride = -stride;
  const int64_t xsize = in.xsize;
  const int64_t ysize = in.ysize;
  const float* row_m = in.ConstRow(y);
  float* JXL_RESTRICT row_out = out.Row(y);
  const float* row_t2 = row_m + 2 * neg_stride;
  const float* row_t1 = row_m + 1 * neg_stride;
  const float* row_b1 = row_m + 1 * stride;
  const float* row_b2 = row_m + 2 * stride;

  const int64_t img_y = y;
  if (ysize <= 2 * kRadius) {  // tiny-height double reflections
    static const int kBorderLut[4 * 8] = {
        0, 0, 0, 0, 0, -1, -1, -1,  // 1 row
        1, 0, 0, 1, 1, 0,  -1, -1,  // 2 rows
        1, 0, 0, 1, 2, 2,  1,  -1,  // 3 rows
        1, 0, 0, 1, 2, 3,  3,  2,   // 4 rows
    };
    const int64_t o = ysize * 8 - 6 + img_y;
    row_t2 = in.ConstRow(kBorderLut[o - 2]);
    row_t1 = in.ConstRow(kBorderLut[o - 1]);
    row_b1 = in.ConstRow(kBorderLut[o + 1]);
    row_b2 = in.ConstRow(kBorderLut[o + 2]);
  } else if (img_y < kRadius) {
    if (img_y == 0) {
      row_t1 = row_m;
      row_t2 = row_b1;
    } else {
      row_t2 = row_t1;
    }
  } else {
    if (img_y + 1 == ysize) {
      row_b1 = row_m;
      row_b2 = row_t1;
    } else {
      row_b2 = row_b1;
    }
  }

  const V wh0 = hn::LoadDup128(d, w.horz + 0 * 4);
  const V wh1 = hn::LoadDup128(d, w.horz + 1 * 4);
  const V wh2 = hn::LoadDup128(d, w.horz + 2 * 4);
  const V wv0 = hn::LoadDup128(d, w.vert + 0 * 4);
  const V wv1 = hn::LoadDup128(d, w.vert + 1 * 4);
  const V wv2 = hn::LoadDup128(d, w.vert + 2 * 4);
  const I ml1 = MirrorLanes<1>();
  const I ml2 = MirrorLanes<2>();

  int64_t x = 0;

#define EMIT(REGION)                                                        \
  do {                                                                      \
    V ht2, ht1, hm, hb1, hb2;                                               \
    if constexpr (kDedup) {                                                 \
      HorzFiveDedup<kSizeModN, REGION>(row_t2, row_t1, row_m, row_b1,       \
                                       row_b2, x, xsize, wh0, wh1, wh2, ml1, \
                                       ml2, &ht2, &ht1, &hm, &hb1, &hb2);    \
    } else {                                                                \
      hm = HorzPick<kSizeModN, REGION>(row_m, x, xsize, wh0, wh1, wh2, ml1, \
                                       ml2);                                \
      ht1 = HorzPick<kSizeModN, REGION>(row_t1, x, xsize, wh0, wh1, wh2,    \
                                        ml1, ml2);                          \
      hb1 = HorzPick<kSizeModN, REGION>(row_b1, x, xsize, wh0, wh1, wh2,    \
                                        ml1, ml2);                          \
      ht2 = HorzPick<kSizeModN, REGION>(row_t2, x, xsize, wh0, wh1, wh2,    \
                                        ml1, ml2);                          \
      hb2 = HorzPick<kSizeModN, REGION>(row_b2, x, xsize, wh0, wh1, wh2,    \
                                        ml1, ml2);                          \
    }                                                                       \
    const V c0 = hn::Mul(hm, wv0);                                          \
    const V c1 = hn::MulAdd(hn::Add(ht1, hb1), wv1, c0);                    \
    const V c2 = hn::MulAdd(hn::Add(ht2, hb2), wv2, c1);                    \
    hn::Store(c2, d, row_out + x);                                          \
  } while (0)

  for (; x < kRadius; x += N) EMIT(0);
  for (; x + N + kRadius <= xsize; x += N) EMIT(1);
  if (kSizeModN < kRadius) {
    EMIT(2);
    x += N;
  }
#undef EMIT

  if (kSizeModN != 0) {
    const float* rows[5] = {row_t2, row_t1, row_m, row_b1, row_b2};
    for (; x < xsize; ++x) {
      float mul = 0.0f;
      for (int64_t dy = -kRadius; dy <= kRadius; ++dy) {
        const float wy = w.vert[std::abs(dy) * 4];
        const float* clamped_row = rows[dy + 2];
        for (int64_t dx = -kRadius; dx <= kRadius; ++dx) {
          const float wx = w.horz[std::abs(dx) * 4];
          const int64_t clamped_x = Mirror(x + dx, xsize);
          mul += clamped_row[clamped_x] * wx * wy;
        }
      }
      row_out[x] = mul;
    }
  }
}

// Computes the [0,ybegin) and [yend,ysize) border-row ranges like RunRows.
static void BorderRange(int64_t ysize, int64_t* ybegin, int64_t* yend) {
  int64_t b = 0, e = ysize;
  while (b < e && b < kRadius) b++;
  while (b < e && e + kRadius > ysize) e--;
  *ybegin = b;
  *yend = e;
}

template <bool kDedup>
static void RunBorders(const Plane& in, Plane& out, const Weights& w) {
  const int64_t N = hn::Lanes(D());
  int64_t yb, ye;
  BorderRange(in.ysize, &yb, &ye);
  auto row = [&](int64_t y) {
    switch (in.xsize % N) {
      case 0: BorderRow<0, kDedup>(in, out, w, y); break;
      case 1: BorderRow<1, kDedup>(in, out, w, y); break;
      default: BorderRow<2, kDedup>(in, out, w, y); break;
    }
  };
  for (int64_t y = 0; y < yb; ++y) row(y);
  for (int64_t y = ye; y < in.ysize; ++y) row(y);
}

static uint64_t FnvBorders(const Plane& p) {
  int64_t yb, ye;
  BorderRange(p.ysize, &yb, &ye);
  uint64_t h = 1469598103934665603ull;
  auto hashRow = [&](int64_t y) {
    const float* row = p.ConstRow(y);
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(row);
    for (int64_t i = 0; i < p.xsize * (int64_t)sizeof(float); ++i) {
      h ^= bytes[i];
      h *= 1099511628211ull;
    }
  };
  for (int64_t y = 0; y < yb; ++y) hashRow(y);
  for (int64_t y = ye; y < p.ysize; ++y) hashRow(y);
  return h;
}

static void FillInput(Plane& p, uint32_t seed) {
  for (int64_t y = 0; y < p.ysize; ++y) {
    float* row = p.Row(y);
    for (int64_t x = 0; x < p.xsize; ++x) {
      uint32_t h = seed ^ (uint32_t)(x * 374761393u) ^ (uint32_t)(y * 668265263u);
      h = (h ^ (h >> 13)) * 1274126177u;
      float n = (h & 0xFFFF) / 65535.0f;
      row[x] = 0.5f + 0.4f * std::sin(x * 0.013f + y * 0.017f) + 0.1f * (n - 0.5f);
    }
  }
}

static double NowMs() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch())
      .count();
}

int main() {
  setbuf(stdout, nullptr);
  Weights w;
  const float hw[3] = {0.4f, 0.25f, 0.05f};
  const float vw[3] = {0.4f, 0.25f, 0.05f};
  for (int g = 0; g < 5; ++g)
    for (int j = 0; j < 4; ++j) {
      w.horz[g * 4 + j] = (g < 3) ? hw[g] : 0.0f;
      w.vert[g * 4 + j] = (g < 3) ? vw[g] : 0.0f;
    }

  const int64_t N = hn::Lanes(D());
  printf("target=%s lanes=%lld\n", hwy::TargetName(HWY_TARGET), (long long)N);

  // ---- Byte-exact sweep: every kSizeModN x every border topology ----
  std::vector<int64_t> widths = {66,  67,  68,  69,  128, 129, 130,
                                 131, 512, 513, 514, 515, 1024};
  std::vector<int64_t> heights = {1, 2, 3, 4, 5, 6, 7, 8, 16, 33, 64};
  int fails = 0, checks = 0;
  for (int64_t xs : widths) {
    if (xs < N + kRadius) continue;
    for (int64_t ys : heights) {
      Plane in(xs, ys);
      FillInput(in, 12345u);
      Plane oOld(xs, ys), oNew(xs, ys);
      RunBorders<false>(in, oOld, w);
      RunBorders<true>(in, oNew, w);
      checks++;
      if (FnvBorders(oOld) != FnvBorders(oNew)) {
        printf("  MISMATCH xs=%lld ys=%lld\n", (long long)xs, (long long)ys);
        fails++;
      }
    }
  }
  printf("byte-exact: %s (%d mismatches / %d configs)\n",
         fails == 0 ? "PASS" : "FAIL", fails, checks);

  // ---- Timing: border-dominated images, interleaved (flipflop discipline) ----
  struct TCfg {
    int64_t xs, ys;
    int reps;
    const char* note;
  };
  std::vector<TCfg> tcfgs = {
      {16384, 1, 4000, "LUT all-alias"},   {16384, 2, 4000, "LUT 2-row"},
      {16384, 3, 3000, "LUT 3-row"},       {16384, 4, 3000, "LUT 4-row"},
      {16384, 8, 2000, "normal 4 border"}, {8192, 64, 2000, "small img 4 border"},
  };
  for (auto& t : tcfgs) {
    if (t.xs < N + kRadius) continue;
    Plane in(t.xs, t.ys);
    FillInput(in, 999u);
    Plane out(t.xs, t.ys);
    RunBorders<false>(in, out, w);  // warm
    RunBorders<true>(in, out, w);
    std::vector<double> sOld, sNew;
    for (int r = 0; r < t.reps; ++r) {
      if (r & 1) {
        { double a = NowMs(); RunBorders<false>(in, out, w); sOld.push_back(NowMs() - a); }
        { double a = NowMs(); RunBorders<true>(in, out, w); sNew.push_back(NowMs() - a); }
      } else {
        { double a = NowMs(); RunBorders<true>(in, out, w); sNew.push_back(NowMs() - a); }
        { double a = NowMs(); RunBorders<false>(in, out, w); sOld.push_back(NowMs() - a); }
      }
    }
    std::sort(sOld.begin(), sOld.end());
    std::sort(sNew.begin(), sNew.end());
    double mo = sOld[sOld.size() / 2], mn = sNew[sNew.size() / 2];
    printf("border %lldx%-3lld (%-18s): OLD %.4f ms  NEW %.4f ms  saved %+.1f%%\n",
           (long long)t.xs, (long long)t.ys, t.note, mo, mn,
           100.0 * (mo - mn) / mo);
  }
  return fails == 0 ? 0 : 1;
}
