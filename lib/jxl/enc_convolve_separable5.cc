// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/convolve.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_convolve_separable5.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/convolve-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::IndicesFromVec;
using hwy::HWY_NAMESPACE::Iota;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Min;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Sub;
using hwy::HWY_NAMESPACE::Vec;

using D = HWY_CAPPED(float, 16);
using DI32 = HWY_CAPPED(int32_t, 16);
using V = Vec<D>;
using VI32 = Vec<DI32>;
using I = decltype(SetTableIndices(D(), static_cast<int32_t*>(nullptr)));

// 5x5 convolution by separable kernel with a single scan through the input.
// This is more cache-efficient than separate horizontal/vertical passes, and
// possibly faster (given enough registers) than tiling and/or transposing.
//
// Overview: imagine a 5x5 window around a central pixel. First convolve the
// rows by multiplying the pixels with the corresponding weights from
// WeightsSeparable5.horz[abs(x_offset) * 4]. Then multiply each of these
// intermediate results by the corresponding vertical weight, i.e.
// vert[abs(y_offset) * 4]. Finally, store the sum of these values as the
// convolution result at the position of the central pixel in the output.
//
// Each of these operations uses SIMD vectors. The central pixel and most
// importantly the output are aligned, so neighnoring pixels (e.g. x_offset=1)
// require unaligned loads. Because weights are supplied in identical groups of
// 4, we can use LoadDup128 to load them (slightly faster).
//
// Uses mirrored boundary handling. Until x >= kRadius, the horizontal
// convolution uses Neighbors class to shuffle vectors as if each of its lanes
// had been loaded from the mirrored offset. Similarly, the last full vector to
// write uses mirroring. In the case of scalar vectors, Neighbors is not usable
// and the value is loaded directly. Otherwise, the number of valid pixels
// modulo the vector size enables a small optimization: for smaller offsets,
// a non-mirrored load is sufficient.
class Separable5Impl {
 public:
  using Simd = HWY_CAPPED(float, 16);
  static constexpr int64_t kRadius = 2;

  Separable5Impl(const ImageF* in, const Rect& rect,
                 const WeightsSeparable5* weights, ThreadPool* pool,
                 ImageF* out)
      : in(in), rect(rect), weights(weights), pool(pool), out(out) {}

  Status Run() {
#if HWY_TARGET == HWY_SCALAR
    // First/Last use mirrored loads of up to +/- kRadius.
    size_t min_width = 2 * kRadius;
#else
    size_t min_width = Lanes(Simd()) + kRadius;
#endif

    if (rect.xsize() >= min_width) {
      JXL_ENSURE(SameSize(rect, *out));

      if (in->ysize() <= 2 * kRadius) {  // tiny height: cross-row-reuse kernel
        switch (rect.xsize() % Lanes(Simd())) {
          case 0:
            return RunTinyHeight<0>();
          case 1:
            return RunTinyHeight<1>();
          default:
            return RunTinyHeight<2>();
        }
      }

      switch (rect.xsize() % Lanes(Simd())) {
        case 0:
          return RunRows<0>();
        case 1:
          return RunRows<1>();
        default:
          // kSizeModN >= kRadius all behave identically: the last-vector path is
          // guarded by `kSizeModN < kRadius`, so only the scalar tail differs and
          // it is driven by the runtime `xsize`, not the template value. Collapse
          // them to a single RunRows<2> to avoid a redundant specialization.
          return RunRows<2>();
      }
    } else {
      return SlowSeparable5(*in, rect, *weights, pool, out, Rect(*out));
    }
  }

  template <size_t kSizeModN, bool kBorder>
  JXL_NOINLINE void ConvolveRow(const uint32_t y) {
    const D d;
    const size_t xsize = rect.xsize();
    float* const JXL_RESTRICT row_out = out->Row(y);
    // ConvolveRow only handles border rows; reflection is unconditional.
    const float* JXL_RESTRICT rows[5];
    ComputeRowPointers(y, rows);
    (void)kBorder;

    const V wh0 = LoadDup128(d, weights->horz + 0 * 4);
    const V wh1 = LoadDup128(d, weights->horz + 1 * 4);
    const V wh2 = LoadDup128(d, weights->horz + 2 * 4);
    const V wv0 = LoadDup128(d, weights->vert + 0 * 4);
    const V wv1 = LoadDup128(d, weights->vert + 1 * 4);
    const V wv2 = LoadDup128(d, weights->vert + 2 * 4);
    const I ml1 = MirrorLanes<1>();
    const I ml2 = MirrorLanes<2>();

    size_t x = 0;

    // First vector(s): mirrored left border (more than one for scalars).
    for (; x < kRadius; x += Lanes(d)) {
      Store(BorderColumn<kSizeModN, 0>(rows, x, xsize, wh0, wh1, wh2, wv0, wv1,
                                       wv2, ml1, ml2),
            d, row_out + x);
    }

    // Main loop: load inputs without padding.
    for (; x + Lanes(d) + kRadius <= xsize; x += Lanes(d)) {
      Store(BorderColumn<kSizeModN, 1>(rows, x, xsize, wh0, wh1, wh2, wv0, wv1,
                                       wv2, ml1, ml2),
            d, row_out + x);
    }

    // Last full vector to write (the above loop handled mod >= kRadius).
#if HWY_TARGET == HWY_SCALAR
    while (x < xsize) {
#else
    if (kSizeModN < kRadius) {
#endif
      Store(BorderColumn<kSizeModN, 2>(rows, x, xsize, wh0, wh1, wh2, wv0, wv1,
                                       wv2, ml1, ml2),
            d, row_out + x);
      x += Lanes(d);
    }

    // If mod = 0, the above vector was the last.
    if (kSizeModN != 0) {
      // Tail starts at x >= Lanes(d) > kRadius, so x-kRadius >= 0 always; only
      // the final kRadius columns can read past xsize on the right.
      const int64_t safe_end = static_cast<int64_t>(xsize) - kRadius;
      for (; static_cast<int64_t>(x) < safe_end; ++x) {
        row_out[x] = ScalarPixel<false>(rows, x, xsize, weights);
      }
      for (; x < xsize; ++x) {
        row_out[x] = ScalarPixel<true>(rows, x, xsize, weights);
      }
    }
  }

 private:
  template <size_t kSizeModN>
  JXL_INLINE Status RunRows() {
    // NB: borders are image-bound, not rect-bound.
    size_t ybegin = rect.y0();
    size_t yend = rect.y1();
    while (ybegin < yend && ybegin < kRadius) {
      ybegin++;
    }
    while (ybegin < yend && yend + kRadius > in->ysize()) {
      yend--;
    }
    if (ybegin > rect.y0()) {
      JXL_RETURN_IF_ERROR(RunBorderRows<kSizeModN>(0, ybegin - rect.y0()));
    }
    if (yend > ybegin) {
      JXL_RETURN_IF_ERROR(
          RunInteriorRows<kSizeModN>(ybegin - rect.y0(), yend - rect.y0()));
    }
    if (yend < rect.y1()) {
      JXL_RETURN_IF_ERROR(
          RunBorderRows<kSizeModN>(yend - rect.y0(), rect.ysize()));
    }
    return true;
  }

  template <size_t kSizeModN>
  JXL_INLINE Status RunBorderRows(const size_t ybegin, const size_t yend) {
    for (size_t y = ybegin; y < yend; ++y) {
      ConvolveRow<kSizeModN, true>(y);
    }
    return true;
  }

  // Number of output rows a single pool task convolves as one vertical band.
  // A band reuses horizontal convolutions across its rows via a rolling ring
  // (see RingColumn), so it computes `band + 4` horizontal convolutions per
  // column instead of `5 * band`. Larger bands amortize the 4-row halo better
  // but expose fewer parallel tasks; 8 balances reuse against parallelism.
  static constexpr size_t kRowsPerBand = 8;

  // Picks the horizontal convolution variant for a column region at compile
  // time. kRegion: 0 = first (left border), 1 = interior, 2 = last full vector.
  // The branches fold away because kRegion is a template constant.
  template <size_t kSizeModN, int kRegion>
  static JXL_MAYBE_INLINE V HorzPick(const float* const JXL_RESTRICT row,
                                     const int64_t x, const int64_t xsize,
                                     const V wh0, const V wh1, const V wh2,
                                     const I ml1, const I ml2) {
    if (kRegion == 0) {
      return HorzConvolveFirst(row, x, xsize, wh0, wh1, wh2);
    }
    if (kRegion == 2) {
      return HorzConvolveLast<kSizeModN>(row, x, xsize, wh0, wh1, wh2, ml1, ml2);
    }
    (void)ml1;
    (void)ml2;
    (void)xsize;
    return HorzConvolve(row + x, wh0, wh1, wh2);
  }

  // Fills rows[] = {t2, t1, m, b1, b2} for output row y, applying image-bound
  // vertical mirroring (incl. the tiny-height <=4 double-reflection LUT). For
  // interior rows the defaults (+/- stride) are kept.
  JXL_INLINE void ComputeRowPointers(const size_t y,
                                     const float* JXL_RESTRICT rows[5]) const {
    const int64_t stride = in->PixelsPerRow();
    const float* const JXL_RESTRICT row_m = rect.ConstRow(*in, y);
    const float* row_t2 = row_m - 2 * stride;
    const float* row_t1 = row_m - 1 * stride;
    const float* row_b1 = row_m + 1 * stride;
    const float* row_b2 = row_m + 2 * stride;
    const size_t img_y = rect.y0() + y;
    if (in->ysize() <= 2 * kRadius) {  // Very special: double reflections
      static constexpr size_t kBorderLut[4 * 8] = {
          0, 0, 0, 0, 0, 0xBAD, 0xBAD, 0xBAD,  // 1 row
          1, 0, 0, 1, 1, 0,     0xBAD, 0xBAD,  // 2 rows
          1, 0, 0, 1, 2, 2,     1,     0xBAD,  // 3 rows
          1, 0, 0, 1, 2, 3,     3,     2,      // 4 rows
      };
      JXL_DASSERT(in->ysize() <= 4);
      const size_t o = in->ysize() * 8 - 6 + img_y;
      row_t2 = in->ConstRow(kBorderLut[o - 2]) + rect.x0();
      row_t1 = in->ConstRow(kBorderLut[o - 1]) + rect.x0();
      row_b1 = in->ConstRow(kBorderLut[o + 1]) + rect.x0();
      row_b2 = in->ConstRow(kBorderLut[o + 2]) + rect.x0();
    } else if (img_y < kRadius) {
      if (img_y == 0) {
        row_t1 = row_m;
        row_t2 = row_b1;
      } else {
        JXL_DASSERT(img_y == 1);
        row_t2 = row_t1;
      }
    } else if (img_y + kRadius >= in->ysize()) {
      if (img_y + 1 == in->ysize()) {
        row_b1 = row_m;
        row_b2 = row_t1;
      } else {
        JXL_DASSERT(img_y + 2 == in->ysize());
        row_b2 = row_b1;
      }
    }
    rows[0] = row_t2;
    rows[1] = row_t1;
    rows[2] = row_m;
    rows[3] = row_b1;
    rows[4] = row_b2;
  }

  // One output SIMD column with horizontal-convolution dedup: reflected source
  // rows frequently alias (e.g. row_t1 == row_m at the top edge), so each
  // distinct row is convolved once and reused. The vertical combine is the
  // identical FMA sequence ConvolveRow used, so output is byte-exact.
  template <size_t kSizeModN, int kRegion>
  JXL_MAYBE_INLINE V BorderColumn(const float* const JXL_RESTRICT rows[5],
                                  const int64_t x, const int64_t xsize,
                                  const V wh0, const V wh1, const V wh2,
                                  const V wv0, const V wv1, const V wv2,
                                  const I ml1, const I ml2) const {
    const V h2 = HorzPick<kSizeModN, kRegion>(rows[2], x, xsize, wh0, wh1, wh2,
                                              ml1, ml2);
    const V h1 = (rows[1] == rows[2])
                     ? h2
                     : HorzPick<kSizeModN, kRegion>(rows[1], x, xsize, wh0, wh1,
                                                    wh2, ml1, ml2);
    const V h3 = (rows[3] == rows[2])   ? h2
                 : (rows[3] == rows[1]) ? h1
                                        : HorzPick<kSizeModN, kRegion>(
                                              rows[3], x, xsize, wh0, wh1, wh2,
                                              ml1, ml2);
    const V h0 = (rows[0] == rows[2])   ? h2
                 : (rows[0] == rows[1]) ? h1
                 : (rows[0] == rows[3]) ? h3
                                        : HorzPick<kSizeModN, kRegion>(
                                              rows[0], x, xsize, wh0, wh1, wh2,
                                              ml1, ml2);
    const V h4 = (rows[4] == rows[2])   ? h2
                 : (rows[4] == rows[1]) ? h1
                 : (rows[4] == rows[3]) ? h3
                 : (rows[4] == rows[0]) ? h0
                                        : HorzPick<kSizeModN, kRegion>(
                                              rows[4], x, xsize, wh0, wh1, wh2,
                                              ml1, ml2);
    const V conv0 = Mul(h2, wv0);
    const V conv1 = MulAdd(Add(h1, h3), wv1, conv0);
    const V conv2 = MulAdd(Add(h0, h4), wv2, conv1);
    return conv2;
  }

  // Convolves a single SIMD column for every output row in [y0, y1) using a
  // rolling ring of the five horizontal convolutions. Each step rotates the ring
  // and computes exactly one new horizontal convolution (the incoming bottom
  // row), reusing the other four. The per-pixel vertical accumulation is the
  // identical FMA sequence used by ConvolveRow, so the output is byte-exact.
  template <size_t kSizeModN, int kRegion>
  JXL_MAYBE_INLINE void RingColumn(const size_t y0, const size_t y1,
                                   const int64_t x, const V wh0, const V wh1,
                                   const V wh2, const V wv0, const V wv1,
                                   const V wv2, const I ml1, const I ml2) const {
    const D d;
    const int64_t stride = in->PixelsPerRow();
    const int64_t xsize = rect.xsize();
    const float* const JXL_RESTRICT base = rect.ConstRow(*in, y0);
    const float* JXL_RESTRICT r_in = base + 2 * stride;

    V h0 = HorzPick<kSizeModN, kRegion>(base - 2 * stride, x, xsize, wh0, wh1,
                                        wh2, ml1, ml2);
    V h1 = HorzPick<kSizeModN, kRegion>(base - 1 * stride, x, xsize, wh0, wh1,
                                        wh2, ml1, ml2);
    V h2 = HorzPick<kSizeModN, kRegion>(base, x, xsize, wh0, wh1, wh2, ml1, ml2);
    V h3 = HorzPick<kSizeModN, kRegion>(base + 1 * stride, x, xsize, wh0, wh1,
                                        wh2, ml1, ml2);
    V h4 = HorzPick<kSizeModN, kRegion>(r_in, x, xsize, wh0, wh1, wh2, ml1, ml2);

    for (size_t y = y0;; ++y) {
      const V conv0 = Mul(h2, wv0);
      const V conv1 = MulAdd(Add(h1, h3), wv1, conv0);
      const V conv2 = MulAdd(Add(h0, h4), wv2, conv1);
      Store(conv2, d, out->Row(y) + x);
      if (y + 1 == y1) break;
      r_in += stride;
      h0 = h1;
      h1 = h2;
      h2 = h3;
      h3 = h4;
      h4 = HorzPick<kSizeModN, kRegion>(r_in, x, xsize, wh0, wh1, wh2, ml1, ml2);
    }
  }

  // Convolves a contiguous band of interior output rows [y0, y1). Walks columns
  // left-to-right (first / interior / last vector, then scalar tail), running a
  // vertical rolling ring down the band for each SIMD column.
  template <size_t kSizeModN>
  JXL_INLINE void ConvolveInteriorBand(const size_t y0, const size_t y1) {
    const D d;
    const int64_t N = Lanes(d);
    const int64_t xsize = rect.xsize();

    const V wh0 = LoadDup128(d, weights->horz + 0 * 4);
    const V wh1 = LoadDup128(d, weights->horz + 1 * 4);
    const V wh2 = LoadDup128(d, weights->horz + 2 * 4);
    const V wv0 = LoadDup128(d, weights->vert + 0 * 4);
    const V wv1 = LoadDup128(d, weights->vert + 1 * 4);
    const V wv2 = LoadDup128(d, weights->vert + 2 * 4);
    const I ml1 = MirrorLanes<1>();
    const I ml2 = MirrorLanes<2>();

    int64_t x = 0;

    // First vector(s): mirrored left border.
    for (; x < kRadius; x += N) {
      RingColumn<kSizeModN, 0>(y0, y1, x, wh0, wh1, wh2, wv0, wv1, wv2, ml1,
                               ml2);
    }

    // Interior vectors: no padding needed.
    for (; x + N + kRadius <= xsize; x += N) {
      RingColumn<kSizeModN, 1>(y0, y1, x, wh0, wh1, wh2, wv0, wv1, wv2, ml1,
                               ml2);
    }

    // Last full vector (mirrored right border), if it is not already covered.
    if (kSizeModN < kRadius) {
      RingColumn<kSizeModN, 2>(y0, y1, x, wh0, wh1, wh2, wv0, wv1, wv2, ml1,
                               ml2);
      x += N;
    }

    // Scalar remainder: identical 25-term accumulation order to ConvolveRow.
    if (kSizeModN != 0) {
      const int64_t stride = in->PixelsPerRow();
      const int64_t safe_end = xsize - kRadius;
      for (size_t y = y0; y < y1; ++y) {
        const float* const JXL_RESTRICT row_m = rect.ConstRow(*in, y);
        const float* const JXL_RESTRICT rows[5] = {
            row_m - 2 * stride, row_m - 1 * stride, row_m, row_m + 1 * stride,
            row_m + 2 * stride};
        float* const JXL_RESTRICT row_out = out->Row(y);
        int64_t xx = x;
        for (; xx < safe_end; ++xx)
          row_out[xx] = ScalarPixel<false>(rows, xx, xsize, weights);
        for (; xx < xsize; ++xx)
          row_out[xx] = ScalarPixel<true>(rows, xx, xsize, weights);
      }
    }
  }

  template <size_t kSizeModN>
  JXL_INLINE Status RunInteriorRows(const size_t ybegin, const size_t yend) {
    const size_t count = yend - ybegin;
    const size_t num_bands = (count + kRowsPerBand - 1) / kRowsPerBand;
    const auto process_band = [&](const uint32_t band,
                                  size_t /*thread*/) HWY_ATTR {
      const size_t b0 = ybegin + static_cast<size_t>(band) * kRowsPerBand;
      const size_t b1 = std::min(b0 + kRowsPerBand, yend);
      ConvolveInteriorBand<kSizeModN>(b0, b1);
      return true;
    };
    return RunOnPool(pool, 0, static_cast<uint32_t>(num_bands),
                     ThreadPool::NoInit, process_band, "ConvolveBands");
  }

  // Convolves one SIMD column for every output row when in->ysize() <= 4. Each
  // of the (<=4) distinct source rows is convolved once into hrow[], then every
  // output row is formed from the kBorderLut index mapping + the identical
  // vertical FMA combine. Full cross-row reuse vs ConvolveRow's per-row recompute.
  template <size_t kSizeModN, int kRegion>
  JXL_MAYBE_INLINE void TinyColumn(const size_t ysz, const int64_t x,
                                   const int64_t xsize, const size_t* lut,
                                   const V wh0, const V wh1, const V wh2,
                                   const V wv0, const V wv1, const V wv2,
                                   const I ml1, const I ml2) const {
    const D d;
    V hrow[2 * kRadius];  // ysz <= 2*kRadius == 4
    for (size_t j = 0; j < ysz; ++j) {
      hrow[j] = HorzPick<kSizeModN, kRegion>(in->ConstRow(j) + rect.x0(), x,
                                             xsize, wh0, wh1, wh2, ml1, ml2);
    }
    for (size_t y = 0; y < rect.ysize(); ++y) {
      const size_t img_y = rect.y0() + y;
      const size_t o = ysz * 8 - 6 + img_y;
      const V h0 = hrow[lut[o - 2]];
      const V h1 = hrow[lut[o - 1]];
      const V h2 = hrow[img_y];
      const V h3 = hrow[lut[o + 1]];
      const V h4 = hrow[lut[o + 2]];
      const V conv0 = Mul(h2, wv0);
      const V conv1 = MulAdd(Add(h1, h3), wv1, conv0);
      const V conv2 = MulAdd(Add(h0, h4), wv2, conv1);
      Store(conv2, d, out->Row(y) + x);
    }
  }

  // Tiny-height fast path (in->ysize() <= 2*kRadius). Byte-exact vs ConvolveRow.
  template <size_t kSizeModN>
  JXL_INLINE Status RunTinyHeight() {
    const D d;
    const int64_t xsize = rect.xsize();
    const size_t ysz = in->ysize();
    static constexpr size_t kBorderLut[4 * 8] = {
        0, 0, 0, 0, 0, 0xBAD, 0xBAD, 0xBAD,  // 1 row
        1, 0, 0, 1, 1, 0,     0xBAD, 0xBAD,  // 2 rows
        1, 0, 0, 1, 2, 2,     1,     0xBAD,  // 3 rows
        1, 0, 0, 1, 2, 3,     3,     2,      // 4 rows
    };
    JXL_DASSERT(ysz <= 4);
    const V wh0 = LoadDup128(d, weights->horz + 0 * 4);
    const V wh1 = LoadDup128(d, weights->horz + 1 * 4);
    const V wh2 = LoadDup128(d, weights->horz + 2 * 4);
    const V wv0 = LoadDup128(d, weights->vert + 0 * 4);
    const V wv1 = LoadDup128(d, weights->vert + 1 * 4);
    const V wv2 = LoadDup128(d, weights->vert + 2 * 4);
    const I ml1 = MirrorLanes<1>();
    const I ml2 = MirrorLanes<2>();

    int64_t x = 0;
    for (; x < kRadius; x += Lanes(d)) {
      TinyColumn<kSizeModN, 0>(ysz, x, xsize, kBorderLut, wh0, wh1, wh2, wv0,
                               wv1, wv2, ml1, ml2);
    }
    for (; x + Lanes(d) + kRadius <= xsize; x += Lanes(d)) {
      TinyColumn<kSizeModN, 1>(ysz, x, xsize, kBorderLut, wh0, wh1, wh2, wv0,
                               wv1, wv2, ml1, ml2);
    }
#if HWY_TARGET == HWY_SCALAR
    while (x < xsize) {
#else
    if (kSizeModN < kRadius) {
#endif
      TinyColumn<kSizeModN, 2>(ysz, x, xsize, kBorderLut, wh0, wh1, wh2, wv0,
                               wv1, wv2, ml1, ml2);
      x += Lanes(d);
    }

    if (kSizeModN != 0) {
      const int64_t safe_end = xsize - kRadius;
      for (size_t y = 0; y < rect.ysize(); ++y) {
        const float* JXL_RESTRICT rows[5];
        ComputeRowPointers(y, rows);
        float* const JXL_RESTRICT row_out = out->Row(y);
        int64_t xx = x;
        for (; xx < safe_end; ++xx)
          row_out[xx] = ScalarPixel<false>(rows, xx, xsize, weights);
        for (; xx < xsize; ++xx)
          row_out[xx] = ScalarPixel<true>(rows, xx, xsize, weights);
      }
    }
    return true;
  }

  // Returns IndicesFromVec(d, indices) such that TableLookupLanes on the
  // rightmost unaligned vector (rightmost sample in its most-significant lane)
  // returns the mirrored values, with the mirror outside the last valid sample.
  template <int M>
  static JXL_INLINE I MirrorLanes() {
    static_assert(M >= 1 && M <= 2, "Only M in range {1..2} is supported");
    D d;
    DI32 di32;
    const VI32 up = Min(Iota(di32, M), Set(di32, Lanes(d) - 1));
    const VI32 down =
        Max(Iota(di32, M - static_cast<int>(Lanes(d))), Zero(di32));
    return IndicesFromVec(d, Sub(up, down));
  }

  // Same as HorzConvolve for the first/last vector in a row.
  static JXL_MAYBE_INLINE V HorzConvolveFirst(
      const float* const JXL_RESTRICT row, const int64_t x, const int64_t xsize,
      const V wh0, const V wh1, const V wh2) {
    const D d;
    const V c = LoadU(d, row + x);
    const V mul0 = Mul(c, wh0);

#if HWY_TARGET == HWY_SCALAR
    const V l1 = LoadU(d, row + Mirror(x - 1, xsize));
    const V l2 = LoadU(d, row + Mirror(x - 2, xsize));
#else
    (void)xsize;
    const V l1 = Neighbors::FirstL1(c);
    const V l2 = Neighbors::FirstL2(c);
#endif

    const V r1 = LoadU(d, row + x + 1);
    const V r2 = LoadU(d, row + x + 2);

    const V mul1 = MulAdd(Add(l1, r1), wh1, mul0);
    const V mul2 = MulAdd(Add(l2, r2), wh2, mul1);
    return mul2;
  }

  template <size_t kSizeModN>
  static JXL_MAYBE_INLINE V HorzConvolveLast(
      const float* const JXL_RESTRICT row, const int64_t x, const int64_t xsize,
      const V wh0, const V wh1, const V wh2, const I ml1, const I ml2) {
    const D d;
    const V c = LoadU(d, row + x);
    const V mul0 = Mul(c, wh0);

    const V l1 = LoadU(d, row + x - 1);
    const V l2 = LoadU(d, row + x - 2);

    V r1;
    V r2;
#if HWY_TARGET == HWY_SCALAR
    r1 = LoadU(d, row + Mirror(x + 1, xsize));
    r2 = LoadU(d, row + Mirror(x + 2, xsize));
    (void)ml1;
    (void)ml2;
#else
    const size_t N = Lanes(d);
    if (kSizeModN == 0) {
      r2 = TableLookupLanes(c, ml2);
      r1 = TableLookupLanes(c, ml1);
    } else {  // == 1
      const auto last = LoadU(d, row + xsize - N);
      r2 = TableLookupLanes(last, ml1);
      r1 = last;
    }
#endif

    // Sum of pixels with Manhattan distance i, multiplied by weights[i].
    const V sum1 = Add(l1, r1);
    const V mul1 = MulAdd(sum1, wh1, mul0);
    const V sum2 = Add(l2, r2);
    const V mul2 = MulAdd(sum2, wh2, mul1);
    return mul2;
  }

  // Requires kRadius valid pixels before/after pos.
  static JXL_MAYBE_INLINE V HorzConvolve(const float* const JXL_RESTRICT pos,
                                         const V wh0, const V wh1,
                                         const V wh2) {
    const D d;
    const V c = LoadU(d, pos);
    const V mul0 = Mul(c, wh0);

    // Loading anew is faster than combining vectors.
    const V l1 = LoadU(d, pos - 1);
    const V r1 = LoadU(d, pos + 1);
    const V l2 = LoadU(d, pos - 2);
    const V r2 = LoadU(d, pos + 2);
    // Sum of pixels with Manhattan distance i, multiplied by weights[i].
    const V sum1 = Add(l1, r1);
    const V mul1 = MulAdd(sum1, wh1, mul0);
    const V sum2 = Add(l2, r2);
    const V mul2 = MulAdd(sum2, wh2, mul1);
    return mul2;
  }

  // One output pixel via the reference 25-tap accumulation. Identical loop order
  // to the original scalar tail (dy outer, dx inner, mul += row[x']*wx*wy), so
  // output is byte-exact. kMirror=false drops the Mirror() call for interior
  // columns where x+dx is already in [0, xsize): the index — hence the value —
  // is identical.
  template <bool kMirror>
  static JXL_INLINE float ScalarPixel(const float* const JXL_RESTRICT rows[5],
                                      const int64_t x, const int64_t xsize,
                                      const WeightsSeparable5* weights) {
    float mul = 0.0f;
    for (int64_t dy = -kRadius; dy <= kRadius; ++dy) {
      const float wy = weights->vert[std::abs(dy) * 4];
      const float* JXL_RESTRICT clamped_row = rows[dy + 2];
      for (int64_t dx = -kRadius; dx <= kRadius; ++dx) {
        const float wx = weights->horz[std::abs(dx) * 4];
        const int64_t cx = kMirror ? Mirror(x + dx, xsize) : (x + dx);
        mul += clamped_row[cx] * wx * wy;
      }
    }
    return mul;
  }

  const ImageF* in;
  const Rect rect;
  const WeightsSeparable5* weights;
  ThreadPool* pool;
  ImageF* out;
};

Status Separable5(const ImageF& in, const Rect& rect,
                  const WeightsSeparable5& weights, ThreadPool* pool,
                  ImageF* out) {
  Separable5Impl impl(&in, rect, &weights, pool, out);
  return impl.Run();
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(Separable5);
Status Separable5(const ImageF& in, const Rect& rect,
                  const WeightsSeparable5& weights, ThreadPool* pool,
                  ImageF* out) {
  return HWY_DYNAMIC_DISPATCH(Separable5)(in, rect, weights, pool, out);
}

}  // namespace jxl
#endif  // HWY_ONCE
