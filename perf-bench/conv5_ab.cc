// Standalone A/B harness for enc_convolve_separable5 optimization.
// OLD = original row-by-row ConvolveRow. NEW = vertical-band rolling ring.
// Single Highway target (static dispatch): native -march=native gives AVX2/512,
// emcc -msimd128 gives WASM SIMD128 (4 lanes). Verifies byte-exactness (FNV
// hash of full output plane) and times OLD vs NEW interleaved.
//
// Build native:  clang++ -O3 -march=native -std=c++17 -I<hwy> conv5_ab.cc -o conv5_ab
// Build wasm:    emcc -O3 -msimd128 -std=c++17 -I<hwy> conv5_ab.cc -o conv5_ab.js
//                  -s ENVIRONMENT=node -s ALLOW_MEMORY_GROWTH=1

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
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

// ---- mirror (copy of jxl::Mirror) ----
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

// ---- Neighbors (copy of jxl convolve-inl.h) ----
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

// ---- shared horizontal helpers (identical in OLD/NEW) ----
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

struct Plane {
  std::vector<float> buf;
  int64_t xsize, ysize, stride;
  Plane(int64_t xs, int64_t ys) : xsize(xs), ysize(ys) {
    const int64_t N = hn::Lanes(D());
    stride = ((xs + N - 1) / N) * N + 2 * N;  // padded, with slack
    buf.assign(stride * ys, 0.0f);
  }
  float* Row(int64_t y) { return buf.data() + y * stride; }
  const float* ConstRow(int64_t y) const { return buf.data() + y * stride; }
};

// ============ OLD: original row-by-row ============
template <size_t kSizeModN, bool kBorder>
static void OldConvolveRow(const Plane& in, Plane& out, const Weights& w,
                           const int64_t y) {
  const D d;
  const int64_t stride = in.stride;
  const int64_t neg_stride = -stride;
  const int64_t xsize = in.xsize;
  const float* JXL_RESTRICT row_m = in.ConstRow(y);
  float* JXL_RESTRICT row_out = out.Row(y);
  const float* JXL_RESTRICT row_t2 = row_m + 2 * neg_stride;
  const float* JXL_RESTRICT row_t1 = row_m + 1 * neg_stride;
  const float* JXL_RESTRICT row_b1 = row_m + 1 * stride;
  const float* JXL_RESTRICT row_b2 = row_m + 2 * stride;
  if (kBorder) {
    const int64_t img_y = y;
    if (img_y < kRadius) {
      if (img_y == 0) {
        row_t1 = row_m;
        row_t2 = row_b1;
      } else {
        row_t2 = row_t1;
      }
    } else {
      if (img_y + 1 == in.ysize) {
        row_b1 = row_m;
        row_b2 = row_t1;
      } else {
        row_b2 = row_b1;
      }
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
  const int64_t N = hn::Lanes(d);
  for (; x < kRadius; x += N) {
    const V conv0 = hn::Mul(HorzConvolveFirst(row_m, x, xsize, wh0, wh1, wh2), wv0);
    const V conv1t = HorzConvolveFirst(row_t1, x, xsize, wh0, wh1, wh2);
    const V conv1b = HorzConvolveFirst(row_b1, x, xsize, wh0, wh1, wh2);
    const V conv1 = hn::MulAdd(hn::Add(conv1t, conv1b), wv1, conv0);
    const V conv2t = HorzConvolveFirst(row_t2, x, xsize, wh0, wh1, wh2);
    const V conv2b = HorzConvolveFirst(row_b2, x, xsize, wh0, wh1, wh2);
    const V conv2 = hn::MulAdd(hn::Add(conv2t, conv2b), wv2, conv1);
    hn::Store(conv2, d, row_out + x);
  }
  for (; x + N + kRadius <= xsize; x += N) {
    const V conv0 = hn::Mul(HorzConvolve(row_m + x, wh0, wh1, wh2), wv0);
    const V conv1t = HorzConvolve(row_t1 + x, wh0, wh1, wh2);
    const V conv1b = HorzConvolve(row_b1 + x, wh0, wh1, wh2);
    const V conv1 = hn::MulAdd(hn::Add(conv1t, conv1b), wv1, conv0);
    const V conv2t = HorzConvolve(row_t2 + x, wh0, wh1, wh2);
    const V conv2b = HorzConvolve(row_b2 + x, wh0, wh1, wh2);
    const V conv2 = hn::MulAdd(hn::Add(conv2t, conv2b), wv2, conv1);
    hn::Store(conv2, d, row_out + x);
  }
  if (kSizeModN < kRadius) {
    const V conv0 = hn::Mul(
        HorzConvolveLast<kSizeModN>(row_m, x, xsize, wh0, wh1, wh2, ml1, ml2), wv0);
    const V conv1t = HorzConvolveLast<kSizeModN>(row_t1, x, xsize, wh0, wh1, wh2, ml1, ml2);
    const V conv1b = HorzConvolveLast<kSizeModN>(row_b1, x, xsize, wh0, wh1, wh2, ml1, ml2);
    const V conv1 = hn::MulAdd(hn::Add(conv1t, conv1b), wv1, conv0);
    const V conv2t = HorzConvolveLast<kSizeModN>(row_t2, x, xsize, wh0, wh1, wh2, ml1, ml2);
    const V conv2b = HorzConvolveLast<kSizeModN>(row_b2, x, xsize, wh0, wh1, wh2, ml1, ml2);
    const V conv2 = hn::MulAdd(hn::Add(conv2t, conv2b), wv2, conv1);
    hn::Store(conv2, d, row_out + x);
    x += N;
  }
  if (kSizeModN != 0) {
    const float* JXL_RESTRICT rows[5] = {row_t2, row_t1, row_m, row_b1, row_b2};
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

template <size_t kSizeModN>
static void OldRun(const Plane& in, Plane& out, const Weights& w) {
  int64_t ybegin = 0, yend = in.ysize;
  // interior bounds
  int64_t ib = 0, ie = in.ysize;
  while (ib < ie && ib < kRadius) ib++;
  while (ib < ie && ie + kRadius > in.ysize) ie--;
  for (int64_t y = ybegin; y < ib; ++y) OldConvolveRow<kSizeModN, true>(in, out, w, y);
  for (int64_t y = ib; y < ie; ++y) OldConvolveRow<kSizeModN, false>(in, out, w, y);
  for (int64_t y = ie; y < yend; ++y) OldConvolveRow<kSizeModN, true>(in, out, w, y);
}

// ============ NEW: vertical-band rolling ring ============
template <size_t kSizeModN, int kRegion>
static V HorzPick(const float* JXL_RESTRICT row, const int64_t x,
                  const int64_t xsize, const V wh0, const V wh1, const V wh2,
                  const I ml1, const I ml2) {
  if (kRegion == 0) return HorzConvolveFirst(row, x, xsize, wh0, wh1, wh2);
  if (kRegion == 2) return HorzConvolveLast<kSizeModN>(row, x, xsize, wh0, wh1, wh2, ml1, ml2);
  (void)ml1; (void)ml2; (void)xsize;
  return HorzConvolve(row + x, wh0, wh1, wh2);
}

template <size_t kSizeModN, int kRegion>
static void RingColumn(const Plane& in, Plane& out, const int64_t y0,
                       const int64_t y1, const int64_t x, const V wh0,
                       const V wh1, const V wh2, const V wv0, const V wv1,
                       const V wv2, const I ml1, const I ml2) {
  const D d;
  const int64_t stride = in.stride;
  const int64_t xsize = in.xsize;
  const float* JXL_RESTRICT base = in.ConstRow(y0);
  const float* JXL_RESTRICT r_in = base + 2 * stride;
  V h0 = HorzPick<kSizeModN, kRegion>(base - 2 * stride, x, xsize, wh0, wh1, wh2, ml1, ml2);
  V h1 = HorzPick<kSizeModN, kRegion>(base - 1 * stride, x, xsize, wh0, wh1, wh2, ml1, ml2);
  V h2 = HorzPick<kSizeModN, kRegion>(base, x, xsize, wh0, wh1, wh2, ml1, ml2);
  V h3 = HorzPick<kSizeModN, kRegion>(base + 1 * stride, x, xsize, wh0, wh1, wh2, ml1, ml2);
  V h4 = HorzPick<kSizeModN, kRegion>(r_in, x, xsize, wh0, wh1, wh2, ml1, ml2);
  for (int64_t y = y0;; ++y) {
    const V conv0 = hn::Mul(h2, wv0);
    const V conv1 = hn::MulAdd(hn::Add(h1, h3), wv1, conv0);
    const V conv2 = hn::MulAdd(hn::Add(h0, h4), wv2, conv1);
    hn::Store(conv2, d, out.Row(y) + x);
    if (y + 1 == y1) break;
    r_in += stride;
    h0 = h1; h1 = h2; h2 = h3; h3 = h4;
    h4 = HorzPick<kSizeModN, kRegion>(r_in, x, xsize, wh0, wh1, wh2, ml1, ml2);
  }
}

template <size_t kSizeModN>
static void ConvolveInteriorBand(const Plane& in, Plane& out, const Weights& w,
                                 const int64_t y0, const int64_t y1) {
  const D d;
  const int64_t N = hn::Lanes(d);
  const int64_t xsize = in.xsize;
  const V wh0 = hn::LoadDup128(d, w.horz + 0 * 4);
  const V wh1 = hn::LoadDup128(d, w.horz + 1 * 4);
  const V wh2 = hn::LoadDup128(d, w.horz + 2 * 4);
  const V wv0 = hn::LoadDup128(d, w.vert + 0 * 4);
  const V wv1 = hn::LoadDup128(d, w.vert + 1 * 4);
  const V wv2 = hn::LoadDup128(d, w.vert + 2 * 4);
  const I ml1 = MirrorLanes<1>();
  const I ml2 = MirrorLanes<2>();
  int64_t x = 0;
  for (; x < kRadius; x += N)
    RingColumn<kSizeModN, 0>(in, out, y0, y1, x, wh0, wh1, wh2, wv0, wv1, wv2, ml1, ml2);
  for (; x + N + kRadius <= xsize; x += N)
    RingColumn<kSizeModN, 1>(in, out, y0, y1, x, wh0, wh1, wh2, wv0, wv1, wv2, ml1, ml2);
  if (kSizeModN < kRadius) {
    RingColumn<kSizeModN, 2>(in, out, y0, y1, x, wh0, wh1, wh2, wv0, wv1, wv2, ml1, ml2);
    x += N;
  }
  if (kSizeModN != 0) {
    const int64_t stride = in.stride;
    for (int64_t y = y0; y < y1; ++y) {
      const float* JXL_RESTRICT row_m = in.ConstRow(y);
      const float* JXL_RESTRICT rows[5] = {row_m - 2 * stride, row_m - 1 * stride,
                                           row_m, row_m + 1 * stride, row_m + 2 * stride};
      float* JXL_RESTRICT row_out = out.Row(y);
      for (int64_t xx = x; xx < xsize; ++xx) {
        float mul = 0.0f;
        for (int64_t dy = -kRadius; dy <= kRadius; ++dy) {
          const float wy = w.vert[std::abs(dy) * 4];
          const float* clamped_row = rows[dy + 2];
          for (int64_t dx = -kRadius; dx <= kRadius; ++dx) {
            const float wx = w.horz[std::abs(dx) * 4];
            const int64_t clamped_x = Mirror(xx + dx, xsize);
            mul += clamped_row[clamped_x] * wx * wy;
          }
        }
        row_out[xx] = mul;
      }
    }
  }
}

template <size_t kSizeModN>
static void NewRun(const Plane& in, Plane& out, const Weights& w, int64_t band) {
  int64_t ib = 0, ie = in.ysize;
  while (ib < ie && ib < kRadius) ib++;
  while (ib < ie && ie + kRadius > in.ysize) ie--;
  for (int64_t y = 0; y < ib; ++y) OldConvolveRow<kSizeModN, true>(in, out, w, y);
  // bands over [ib, ie)
  for (int64_t b0 = ib; b0 < ie; b0 += band) {
    const int64_t b1 = std::min(b0 + band, ie);
    ConvolveInteriorBand<kSizeModN>(in, out, w, b0, b1);
  }
  for (int64_t y = ie; y < in.ysize; ++y) OldConvolveRow<kSizeModN, true>(in, out, w, y);
}

// ============ XTILE: tall band + narrow x-tile + scratch ring ============
// Bulk interior columns (region 1) processed as ONE tall band over the full
// interior height, tiled in x. A 5-row scratch ring of horizontal convolutions
// is slid down the band, so each interior horizontal convolution is computed
// exactly once (4-row halo per tile, vs the register ring's `band+4` per
// `band`-row band => 1.5x at band=8). Resident set = 5 * tileW floats (L1),
// independent of band height, so it does not thrash like a tall register-ring
// band. Edge columns (region 0 left-mirror, region 2 right-mirror) and the
// scalar tail keep the proven register ring / 2D loop. Vertical FMA order is
// identical to RingColumn, horizontals via the same HorzConvolve => byte-exact.

static void HorzFillTileInterior(const float* JXL_RESTRICT srow, int64_t tx0,
                                 int64_t tx1, int64_t N, const V wh0, const V wh1,
                                 const V wh2, float* JXL_RESTRICT dst) {
  const D d;
  int64_t j = 0;
  for (int64_t x = tx0; x < tx1; x += N, j += N) {
    // scratch buffer alignment is not guaranteed to the vector width -> StoreU.
    hn::StoreU(HorzConvolve(srow + x, wh0, wh1, wh2), d, dst + j);
  }
}

template <size_t kSizeModN>
static void ConvolveInteriorXTile(const Plane& in, Plane& out, const Weights& w,
                                  int64_t ib, int64_t ie, int64_t tile_vecs) {
  const D d;
  const int64_t N = hn::Lanes(d);
  const int64_t xsize = in.xsize;
  const int64_t stride = in.stride;
  const V wh0 = hn::LoadDup128(d, w.horz + 0 * 4);
  const V wh1 = hn::LoadDup128(d, w.horz + 1 * 4);
  const V wh2 = hn::LoadDup128(d, w.horz + 2 * 4);
  const V wv0 = hn::LoadDup128(d, w.vert + 0 * 4);
  const V wv1 = hn::LoadDup128(d, w.vert + 1 * 4);
  const V wv2 = hn::LoadDup128(d, w.vert + 2 * 4);
  const I ml1 = MirrorLanes<1>();
  const I ml2 = MirrorLanes<2>();

  // Region 0 (left mirror): x in [0, kRadius). Register ring, full height.
  int64_t x = 0;
  for (; x < kRadius; x += N)
    RingColumn<kSizeModN, 0>(in, out, ib, ie, x, wh0, wh1, wh2, wv0, wv1, wv2,
                             ml1, ml2);

  // Region 1 (interior bulk): x in [x, x1e), x-tiled scratch ring.
  int64_t x1e = x;
  while (x1e + N + kRadius <= xsize) x1e += N;
  int64_t tileW = tile_vecs * N;
  static std::vector<float> scratch;  // single-threaded harness; per-thread real
  for (int64_t tx0 = x; tx0 < x1e; tx0 += tileW) {
    const int64_t tx1 = std::min(tx0 + tileW, x1e);
    const int64_t W = tx1 - tx0;
    if ((int64_t)scratch.size() < 5 * W) scratch.assign(5 * W, 0.0f);
    float* slot[5];
    for (int k = 0; k < 5; ++k) slot[k] = scratch.data() + k * W;
    // Initial ring: H(ib-2 .. ib+2).
    for (int k = 0; k < 5; ++k)
      HorzFillTileInterior(in.ConstRow(ib - 2 + k), tx0, tx1, N, wh0, wh1, wh2,
                           slot[k]);
    int64_t next_row = ib + 3;
    for (int64_t y = ib;; ++y) {
      float* JXL_RESTRICT orow = out.Row(y);
      const float* JXL_RESTRICT h0 = slot[0];
      const float* JXL_RESTRICT h1 = slot[1];
      const float* JXL_RESTRICT h2 = slot[2];
      const float* JXL_RESTRICT h3 = slot[3];
      const float* JXL_RESTRICT h4 = slot[4];
      for (int64_t j = 0; j < W; j += N) {
        const V c0 = hn::Mul(hn::LoadU(d, h2 + j), wv0);
        const V c1 = hn::MulAdd(hn::Add(hn::LoadU(d, h1 + j), hn::LoadU(d, h3 + j)),
                                wv1, c0);
        const V c2 = hn::MulAdd(hn::Add(hn::LoadU(d, h0 + j), hn::LoadU(d, h4 + j)),
                                wv2, c1);
        hn::Store(c2, d, orow + tx0 + j);
      }
      if (y + 1 == ie) break;
      float* t = slot[0];
      slot[0] = slot[1];
      slot[1] = slot[2];
      slot[2] = slot[3];
      slot[3] = slot[4];
      slot[4] = t;
      HorzFillTileInterior(in.ConstRow(next_row), tx0, tx1, N, wh0, wh1, wh2,
                           slot[4]);
      ++next_row;
    }
  }
  x = x1e;

  // Region 2 (right mirror): last full vector. Register ring, full height.
  if (kSizeModN < kRadius) {
    RingColumn<kSizeModN, 2>(in, out, ib, ie, x, wh0, wh1, wh2, wv0, wv1, wv2,
                             ml1, ml2);
    x += N;
  }

  // Scalar tail: identical 2D accumulation order to RingColumn path.
  if (kSizeModN != 0) {
    for (int64_t y = ib; y < ie; ++y) {
      const float* JXL_RESTRICT row_m = in.ConstRow(y);
      const float* JXL_RESTRICT rows[5] = {row_m - 2 * stride, row_m - 1 * stride,
                                           row_m, row_m + 1 * stride,
                                           row_m + 2 * stride};
      float* JXL_RESTRICT row_out = out.Row(y);
      for (int64_t xx = x; xx < xsize; ++xx) {
        float mul = 0.0f;
        for (int64_t dy = -kRadius; dy <= kRadius; ++dy) {
          const float wy = w.vert[std::abs(dy) * 4];
          const float* clamped_row = rows[dy + 2];
          for (int64_t dx = -kRadius; dx <= kRadius; ++dx) {
            const float wx = w.horz[std::abs(dx) * 4];
            const int64_t clamped_x = Mirror(xx + dx, xsize);
            mul += clamped_row[clamped_x] * wx * wy;
          }
        }
        row_out[xx] = mul;
      }
    }
  }
}

template <size_t kSizeModN>
static void NewRunXTile(const Plane& in, Plane& out, const Weights& w,
                        int64_t tile_vecs) {
  int64_t ib = 0, ie = in.ysize;
  while (ib < ie && ib < kRadius) ib++;
  while (ib < ie && ie + kRadius > in.ysize) ie--;
  for (int64_t y = 0; y < ib; ++y) OldConvolveRow<kSizeModN, true>(in, out, w, y);
  if (ie > ib) ConvolveInteriorXTile<kSizeModN>(in, out, w, ib, ie, tile_vecs);
  for (int64_t y = ie; y < in.ysize; ++y)
    OldConvolveRow<kSizeModN, true>(in, out, w, y);
}

static void DispatchXTile(const Plane& in, Plane& out, const Weights& w,
                          int64_t tv) {
  const int64_t N = hn::Lanes(D());
  switch (in.xsize % N) {
    case 0: NewRunXTile<0>(in, out, w, tv); break;
    case 1: NewRunXTile<1>(in, out, w, tv); break;
    default: NewRunXTile<2>(in, out, w, tv); break;
  }
}

static void DispatchOld(const Plane& in, Plane& out, const Weights& w) {
  const int64_t N = hn::Lanes(D());
  switch (in.xsize % N) {
    case 0: OldRun<0>(in, out, w); break;
    case 1: OldRun<1>(in, out, w); break;
    default: OldRun<2>(in, out, w); break;
  }
}
static void DispatchNew(const Plane& in, Plane& out, const Weights& w, int64_t band) {
  const int64_t N = hn::Lanes(D());
  switch (in.xsize % N) {
    case 0: NewRun<0>(in, out, w, band); break;
    case 1: NewRun<1>(in, out, w, band); break;
    default: NewRun<2>(in, out, w, band); break;
  }
}

// ---- helpers ----
static uint64_t FnvPlane(const Plane& p) {
  uint64_t h = 1469598103934665603ull;
  for (int64_t y = 0; y < p.ysize; ++y) {
    const float* row = p.ConstRow(y);
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(row);
    for (int64_t i = 0; i < p.xsize * (int64_t)sizeof(float); ++i) {
      h ^= bytes[i];
      h *= 1099511628211ull;
    }
  }
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
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  Weights w;
  const float hw[3] = {0.4f, 0.25f, 0.05f};  // center, dist1, dist2
  const float vw[3] = {0.4f, 0.25f, 0.05f};
  for (int g = 0; g < 5; ++g)
    for (int j = 0; j < 4; ++j) {
      w.horz[g * 4 + j] = (g < 3) ? hw[g] : 0.0f;
      w.vert[g * 4 + j] = (g < 3) ? vw[g] : 0.0f;
    }

  const int64_t N = hn::Lanes(D());
  printf("target=%s lanes=%lld\n", hwy::TargetName(HWY_TARGET), (long long)N);

  // Correctness: sweep widths to hit every kSizeModN, plus heights.
  struct Cfg { int64_t xs, ys; };
  std::vector<Cfg> cfgs = {
      {64, 48}, {128, 96}, {127, 65}, {130, 70}, {131, 71}, {256, 200},
      {512, 384}, {257, 257}, {258, 130}, {259, 131}, {1024, 768}, {2048, 1536}};
  int fails = 0;
  for (auto& c : cfgs) {
    if (c.xs < N + kRadius) continue;
    Plane in(c.xs, c.ys);
    FillInput(in, 12345u);
    Plane outOld(c.xs, c.ys), outNew(c.xs, c.ys);
    DispatchOld(in, outOld, w);
    uint64_t hOld = FnvPlane(outOld);
    for (int64_t band : {1, 2, 4, 8, 16}) {
      Plane on(c.xs, c.ys);
      DispatchNew(in, on, w, band);
      uint64_t hNew = FnvPlane(on);
      if (hNew != hOld) {
        printf("  MISMATCH ring xs=%lld ys=%lld band=%lld  old=%016llx new=%016llx\n",
               (long long)c.xs, (long long)c.ys, (long long)band,
               (unsigned long long)hOld, (unsigned long long)hNew);
        fails++;
      }
    }
    for (int64_t tv : {1, 2, 4, 8, 16, 32}) {
      Plane on(c.xs, c.ys);
      DispatchXTile(in, on, w, tv);
      uint64_t hNew = FnvPlane(on);
      if (hNew != hOld) {
        printf("  MISMATCH xtile xs=%lld ys=%lld tv=%lld  old=%016llx new=%016llx\n",
               (long long)c.xs, (long long)c.ys, (long long)tv,
               (unsigned long long)hOld, (unsigned long long)hNew);
        fails++;
      }
    }
  }
  printf("byte-exact check: %s (%d mismatches over %zu cfgs x [5 bands + 6 tiles])\n",
         fails == 0 ? "PASS" : "FAIL", fails, cfgs.size());

  // Headline: interleaved A/B (OLD vs NEW band=8) with start rotation so
  // shared thermal/system drift hits both arms equally (flipflop discipline).
  struct TCfg { int64_t xs, ys; int reps; };
  std::vector<TCfg> tcfgs = {{512, 512, 200}, {1024, 1024, 80}, {2048, 2048, 30}};
  const int64_t kBand = 8;
  for (auto& t : tcfgs) {
    Plane in(t.xs, t.ys);
    FillInput(in, 999u);
    Plane out(t.xs, t.ys);
    DispatchOld(in, out, w);              // warm
    DispatchNew(in, out, w, kBand);
    std::vector<double> sOld, sNew;
    for (int r = 0; r < t.reps; ++r) {
      if (r & 1) {
        { double a = NowMs(); DispatchOld(in, out, w); sOld.push_back(NowMs() - a); }
        { double a = NowMs(); DispatchNew(in, out, w, kBand); sNew.push_back(NowMs() - a); }
      } else {
        { double a = NowMs(); DispatchNew(in, out, w, kBand); sNew.push_back(NowMs() - a); }
        { double a = NowMs(); DispatchOld(in, out, w); sOld.push_back(NowMs() - a); }
      }
    }
    std::sort(sOld.begin(), sOld.end());
    std::sort(sNew.begin(), sNew.end());
    double mo = sOld[sOld.size() / 2], mn = sNew[sNew.size() / 2];
    printf("size %lldx%lld [interleaved]: OLD %.3f ms  NEW(band=8) %.3f ms  saved %+.1f%%\n",
           (long long)t.xs, (long long)t.ys, mo, mn, 100.0 * (mo - mn) / mo);
  }

  // Band-size scan (sequential medians) to confirm the chosen default.
  std::vector<int64_t> bands = {2, 4, 8, 16, 32};
  for (auto& t : tcfgs) {
    Plane in(t.xs, t.ys);
    FillInput(in, 999u);
    Plane out(t.xs, t.ys);
    DispatchOld(in, out, w);
    std::vector<double> so;
    for (int r = 0; r < t.reps; ++r) { double a = NowMs(); DispatchOld(in, out, w); so.push_back(NowMs() - a); }
    std::sort(so.begin(), so.end());
    double mo = so[so.size() / 2];
    printf("size %lldx%lld scan: OLD %.3f ms\n", (long long)t.xs, (long long)t.ys, mo);
    for (int64_t band : bands) {
      std::vector<double> s;
      for (int r = 0; r < t.reps; ++r) { double a = NowMs(); DispatchNew(in, out, w, band); s.push_back(NowMs() - a); }
      std::sort(s.begin(), s.end());
      double med = s[s.size() / 2];
      printf("    NEW band=%-3lld %.3f ms  saved %+.1f%%\n", (long long)band, med, 100.0 * (mo - med) / mo);
    }
  }

  // Headline #2: RING(band=8, current shipped) vs XTILE, interleaved with start
  // rotation. This is the decision: does the scratch x-tile beat the register
  // ring? Reports XTILE saving over RING (the live baseline).
  const int64_t kXTileVecs = 16;
  for (auto& t : tcfgs) {
    Plane in(t.xs, t.ys);
    FillInput(in, 999u);
    Plane out(t.xs, t.ys);
    DispatchNew(in, out, w, kBand);          // warm
    DispatchXTile(in, out, w, kXTileVecs);
    std::vector<double> sRing, sTile;
    for (int r = 0; r < t.reps; ++r) {
      if (r & 1) {
        { double a = NowMs(); DispatchNew(in, out, w, kBand); sRing.push_back(NowMs() - a); }
        { double a = NowMs(); DispatchXTile(in, out, w, kXTileVecs); sTile.push_back(NowMs() - a); }
      } else {
        { double a = NowMs(); DispatchXTile(in, out, w, kXTileVecs); sTile.push_back(NowMs() - a); }
        { double a = NowMs(); DispatchNew(in, out, w, kBand); sRing.push_back(NowMs() - a); }
      }
    }
    std::sort(sRing.begin(), sRing.end());
    std::sort(sTile.begin(), sTile.end());
    double mr = sRing[sRing.size() / 2], mt = sTile[sTile.size() / 2];
    printf("size %lldx%lld [interleaved]: RING(band=8) %.3f ms  XTILE(tv=16) %.3f ms  saved %+.1f%%\n",
           (long long)t.xs, (long long)t.ys, mr, mt, 100.0 * (mr - mt) / mr);
  }

  // XTILE tile-width scan (sequential medians) vs RING(band=8) baseline.
  std::vector<int64_t> tvs = {4, 8, 16, 32, 64};
  for (auto& t : tcfgs) {
    Plane in(t.xs, t.ys);
    FillInput(in, 999u);
    Plane out(t.xs, t.ys);
    DispatchNew(in, out, w, kBand);
    std::vector<double> sr;
    for (int r = 0; r < t.reps; ++r) { double a = NowMs(); DispatchNew(in, out, w, kBand); sr.push_back(NowMs() - a); }
    std::sort(sr.begin(), sr.end());
    double mr = sr[sr.size() / 2];
    printf("size %lldx%lld scan: RING(band=8) %.3f ms\n", (long long)t.xs, (long long)t.ys, mr);
    for (int64_t tv : tvs) {
      std::vector<double> s;
      for (int r = 0; r < t.reps; ++r) { double a = NowMs(); DispatchXTile(in, out, w, tv); s.push_back(NowMs() - a); }
      std::sort(s.begin(), s.end());
      double med = s[s.size() / 2];
      printf("    XTILE tv=%-3lld %.3f ms  saved %+.1f%%\n", (long long)tv, med, 100.0 * (mr - med) / mr);
    }
  }
  return fails == 0 ? 0 : 1;
}
