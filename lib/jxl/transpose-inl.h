// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Block transpose for DCT/IDCT

#include "lib/jxl/base/compiler_specific.h"

#if defined(LIB_JXL_TRANSPOSE_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_TRANSPOSE_INL_H_
#undef LIB_JXL_TRANSPOSE_INL_H_
#else
#define LIB_JXL_TRANSPOSE_INL_H_
#endif

#include <stddef.h>
#include <type_traits>

#include <hwy/highway.h>

#include "lib/jxl/base/status.h"
#include "lib/jxl/dct_block-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

#ifndef JXL_INLINE_TRANSPOSE
// Workaround for issue #42 - (excessive?) inlining causes invalid codegen.
#if defined(__arm__)
#define JXL_INLINE_TRANSPOSE HWY_NOINLINE
#else
#define JXL_INLINE_TRANSPOSE HWY_INLINE
#endif
#endif  // JXL_INLINE_TRANSPOSE

// Simple wrapper that ensures that a function will not be inlined.
// Still used by dct-inl.h (DCT1DCapped large-block paths).
template <typename T, typename... Args>
JXL_NOINLINE void NoInlineWrapper(const T& f, const Args&... args) {
  return f(args...);
}

// Dispatch tag: N_LANES = 0 (scalar), 2, 4, or 8.
template <int N_LANES>
struct TransposeSimdTag {};

// Returns the best SIMD lane count for a ROWS×COLS block.
// On HWY_CAP_GE256 (AVX2/AVX-512): prefer 8-lane tiles, fall back to 4-lane,
// then 2-lane, then scalar.  Previously only 8-lane was selected, leaving
// 4×4, 4×8, and 8×4 blocks as scalar on AVX2 — this fixes that gap.
#if HWY_CAP_GE256
constexpr int TransposeLaneCount(size_t ROWS, size_t COLS) {
  return (ROWS % 8 == 0 && COLS % 8 == 0) ? 8
       : (ROWS % 4 == 0 && COLS % 4 == 0) ? 4
       : (ROWS % 2 == 0 && COLS % 2 == 0) ? 2
       : 0;
}
#elif HWY_TARGET != HWY_SCALAR
constexpr int TransposeLaneCount(size_t ROWS, size_t COLS) {
  return (ROWS % 4 == 0 && COLS % 4 == 0) ? 4
       : (ROWS % 2 == 0 && COLS % 2 == 0) ? 2
       : 0;
}
#else
constexpr int TransposeLaneCount(size_t ROWS, size_t COLS) {
  (void)ROWS;
  (void)COLS;
  return 0;
}
#endif

// ── Scalar kernel ────────────────────────────────────────────────────────────
template <size_t ROWS_or_0, size_t COLS_or_0, class From, class To>
JXL_INLINE_TRANSPOSE void GenericTransposeBlock(
    TransposeSimdTag<0> /* tag */, const From& from, const To& to,
    size_t ROWSp, size_t COLSp) {
  const size_t ROWS = ROWS_or_0 == 0 ? ROWSp : ROWS_or_0;
  const size_t COLS = COLS_or_0 == 0 ? COLSp : COLS_or_0;
  for (size_t n = 0; n < ROWS; ++n) {
    for (size_t m = 0; m < COLS; ++m) {
      to.Write(from.Read(n, m), m, n);
    }
  }
}

// ── 2-lane SIMD kernel ───────────────────────────────────────────────────────
// Covers 2×N / N×2 blocks (e.g. DCT2-style narrow transforms).
#if HWY_TARGET != HWY_SCALAR
template <size_t ROWS_or_0, size_t COLS_or_0, class From, class To>
JXL_INLINE_TRANSPOSE void GenericTransposeBlock(
    TransposeSimdTag<2> /* tag */, const From& from, const To& to,
    size_t ROWSp, size_t COLSp) {
  const size_t ROWS = ROWS_or_0 == 0 ? ROWSp : ROWS_or_0;
  const size_t COLS = COLS_or_0 == 0 ? COLSp : COLS_or_0;
  static_assert(ROWS_or_0 % 2 == 0, "Invalid number of rows");
  static_assert(COLS_or_0 % 2 == 0, "Invalid number of columns");
  const BlockDesc<2> d;
  for (size_t n = 0; n < ROWS; n += 2) {
    for (size_t m = 0; m < COLS; m += 2) {
      const auto p0 = from.LoadPart(d, n + 0, m);
      const auto p1 = from.LoadPart(d, n + 1, m);
      to.StorePart(d, InterleaveLower(d, p0, p1), m + 0, n);
      to.StorePart(d, InterleaveUpper(d, p0, p1), m + 1, n);
    }
  }
}
#endif  // HWY_TARGET != HWY_SCALAR

// ── 4-lane SIMD kernel ───────────────────────────────────────────────────────
// Available on all non-scalar targets, including HWY_CAP_GE256.
// On GE256 this now covers 4×4, 4×8, and 8×4 — previously scalar there.
#if HWY_TARGET != HWY_SCALAR
template <size_t ROWS_or_0, size_t COLS_or_0, class From, class To>
JXL_INLINE_TRANSPOSE void GenericTransposeBlock(
    TransposeSimdTag<4> /* tag */, const From& from, const To& to,
    size_t ROWSp, size_t COLSp) {
  const size_t ROWS = ROWS_or_0 == 0 ? ROWSp : ROWS_or_0;
  const size_t COLS = COLS_or_0 == 0 ? COLSp : COLS_or_0;
  JXL_DASSERT(Lanes(BlockDesc<4>()) == 4);
  static_assert(ROWS_or_0 % 4 == 0, "Invalid number of rows");
  static_assert(COLS_or_0 % 4 == 0, "Invalid number of columns");
  const BlockDesc<4> d;
  for (size_t n = 0; n < ROWS; n += 4) {
    for (size_t m = 0; m < COLS; m += 4) {
      const auto p0 = from.LoadPart(d, n + 0, m);
      const auto p1 = from.LoadPart(d, n + 1, m);
      const auto p2 = from.LoadPart(d, n + 2, m);
      const auto p3 = from.LoadPart(d, n + 3, m);

      const auto q0 = InterleaveLower(d, p0, p2);
      const auto q1 = InterleaveLower(d, p1, p3);
      const auto q2 = InterleaveUpper(d, p0, p2);
      const auto q3 = InterleaveUpper(d, p1, p3);

      to.StorePart(d, InterleaveLower(d, q0, q1), m + 0, n);
      to.StorePart(d, InterleaveUpper(d, q0, q1), m + 1, n);
      to.StorePart(d, InterleaveLower(d, q2, q3), m + 2, n);
      to.StorePart(d, InterleaveUpper(d, q2, q3), m + 3, n);
    }
  }
}
#endif  // HWY_TARGET != HWY_SCALAR

// ── 8-lane SIMD kernel (HWY_CAP_GE256: AVX2 / AVX-512 class) ────────────────
// Surprisingly, this straightforward 3-stage permutation network (24 shuffles
// on port5) is faster than load128+insert and LoadDup128+ConcatUpperLower+blend.
// Final ConcatLowerLower/ConcatUpperUpper results stored directly — avoids
// extending live ranges of the r* intermediates through named i* variables.
#if HWY_CAP_GE256
template <size_t ROWS_or_0, size_t COLS_or_0, class From, class To>
JXL_INLINE_TRANSPOSE void GenericTransposeBlock(
    TransposeSimdTag<8> /* tag */, const From& from, const To& to,
    size_t ROWSp, size_t COLSp) {
  const size_t ROWS = ROWS_or_0 == 0 ? ROWSp : ROWS_or_0;
  const size_t COLS = COLS_or_0 == 0 ? COLSp : COLS_or_0;
  JXL_DASSERT(Lanes(BlockDesc<8>()) == 8);
  static_assert(ROWS_or_0 % 8 == 0, "Invalid number of rows");
  static_assert(COLS_or_0 % 8 == 0, "Invalid number of columns");
  for (size_t n = 0; n < ROWS; n += 8) {
    for (size_t m = 0; m < COLS; m += 8) {
      const BlockDesc<8> d;
      const auto i0 = from.LoadPart(d, n + 0, m);
      const auto i1 = from.LoadPart(d, n + 1, m);
      const auto i2 = from.LoadPart(d, n + 2, m);
      const auto i3 = from.LoadPart(d, n + 3, m);
      const auto i4 = from.LoadPart(d, n + 4, m);
      const auto i5 = from.LoadPart(d, n + 5, m);
      const auto i6 = from.LoadPart(d, n + 6, m);
      const auto i7 = from.LoadPart(d, n + 7, m);

      const auto q0 = InterleaveLower(d, i0, i2);
      const auto q1 = InterleaveLower(d, i1, i3);
      const auto q2 = InterleaveUpper(d, i0, i2);
      const auto q3 = InterleaveUpper(d, i1, i3);
      const auto q4 = InterleaveLower(d, i4, i6);
      const auto q5 = InterleaveLower(d, i5, i7);
      const auto q6 = InterleaveUpper(d, i4, i6);
      const auto q7 = InterleaveUpper(d, i5, i7);

      const auto r0 = InterleaveLower(d, q0, q1);
      const auto r1 = InterleaveUpper(d, q0, q1);
      const auto r2 = InterleaveLower(d, q2, q3);
      const auto r3 = InterleaveUpper(d, q2, q3);
      const auto r4 = InterleaveLower(d, q4, q5);
      const auto r5 = InterleaveUpper(d, q4, q5);
      const auto r6 = InterleaveLower(d, q6, q7);
      const auto r7 = InterleaveUpper(d, q6, q7);

      to.StorePart(d, ConcatLowerLower(d, r4, r0), m + 0, n);
      to.StorePart(d, ConcatLowerLower(d, r5, r1), m + 1, n);
      to.StorePart(d, ConcatLowerLower(d, r6, r2), m + 2, n);
      to.StorePart(d, ConcatLowerLower(d, r7, r3), m + 3, n);
      to.StorePart(d, ConcatUpperUpper(d, r4, r0), m + 4, n);
      to.StorePart(d, ConcatUpperUpper(d, r5, r1), m + 5, n);
      to.StorePart(d, ConcatUpperUpper(d, r6, r2), m + 6, n);
      to.StorePart(d, ConcatUpperUpper(d, r7, r3), m + 7, n);
    }
  }
}
#endif  // HWY_CAP_GE256

// ── Large-block no-inline helper ─────────────────────────────────────────────
// Replaces the old NoInlineWrapper + function-pointer indirection.
// rows/cols stay runtime to prevent the compiler from unrolling large shapes.
template <int kLanes, class From, class To>
JXL_NOINLINE void TransposeLargeNoInline(const From& from, const To& to,
                                         size_t rows, size_t cols) {
  GenericTransposeBlock<0, 0>(TransposeSimdTag<kLanes>{}, from, to, rows, cols);
}

template <size_t N, size_t M, typename = void>
struct Transpose {
  template <typename From, typename To>
  static void Run(const From& from, const To& to) {
    // This does not guarantee anything, just saves from the most stupid
    // mistakes.
    JXL_DASSERT(from.Address(0, 0) != to.Address(0, 0));
    GenericTransposeBlock<N, M>(TransposeSimdTag<TransposeLaneCount(N, M)>{},
                                from, to, N, M);
  }
};

// Avoid inlining and unrolling transposes for large blocks.
template <size_t N, size_t M>
struct Transpose<
    N, M, typename std::enable_if<(N >= 8 && M >= 8 && N * M >= 512)>::type> {
  template <typename From, typename To>
  static void Run(const From& from, const To& to) {
    // This does not guarantee anything, just saves from the most stupid
    // mistakes.
    JXL_DASSERT(from.Address(0, 0) != to.Address(0, 0));
    TransposeLargeNoInline<TransposeLaneCount(N, M)>(from, to, N, M);
  }
};

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // LIB_JXL_TRANSPOSE_INL_H_
