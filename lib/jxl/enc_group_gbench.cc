// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Flipflop benchmarks for enc_group optimizations.
// Registered in jxl_gbench via jxl_lists.cmake.
//
// Pairs (run with --benchmark_filter=BM_):
//
//   BM_QuantizeAC_DCT8_Old  vs  BM_QuantizeAC_DCT8_New
//     Stack-allocated xsize==1 mask vs static const mask.
//
//   BM_RoundtripY_Old  vs  BM_RoundtripY_New
//     Two-loop (quantize then reload+reconstruct) vs fused single-loop.
//
//   BM_CflBoth_Old  vs  BM_CflBoth_New
//     Always-fused both-channel loop vs four-way dispatch (both active).
//
//   BM_CflNone_Old  vs  BM_CflNone_New
//     Both-channel loop (x=b=0) vs four-way dispatch (cfl_mask==0, skip).
//
//   BM_AdjustDCT16_Old  vs  BM_AdjustDCT16_New
//     Per-coeff DC-region branch vs loop-bound fix.
//
// VERDICTS (2026-07-01, native AVX2, clang 22, interleaved standalone A/B;
// all pairs verified byte-exact):
//   QuantizeAC static mask  new/old 0.99-1.00  -> LANDED (general path; the
//       win is on narrow-SIMD/WASM where the wide-vector fast path is skipped).
//   RoundtripY fused        new/old 1.24-1.26  -> REJECTED (~25% regression;
//       fusing breaks the two tight vectorized loops / adds register pressure).
//   Cfl four-way dispatch   both-active 1.00, skip-one 0.67, skip-both 0.22
//       -> LANDED (byte-exact: a zero ratio leaves its channel unchanged;
//       base_correlation_x_==0 makes the X ratio exactly zero on neutral tiles).
//   AdjustDCT16 loop-bound  new/old 1.05-1.08  -> REJECTED (~6% regression;
//       the removed branch is trivially predicted and the variable loop start
//       hurts codegen). See docs/1 rejected optimizations.md.

#include "benchmark/benchmark.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/frame_dimensions.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_group_gbench.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

// quantizer-inl.h uses HWY; must follow foreach_target.h so it's re-included
// for each ISA target.
#include "lib/jxl/quantizer-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::MaskFromVec;
using hwy::HWY_NAMESPACE::Round;

// Prevent the compiler from constant-folding a benchmark's CfL ratios (which
// would let it delete a cfl_mask==0 branch, or prove NegMulAdd(0,y,x)==x, at
// compile time). In production the ratios come from the cmap at runtime, so
// launder them through DoNotOptimize to measure the real dispatch cost.
static float Launder(float v) {
  benchmark::DoNotOptimize(v);
  return v;
}

// ---------------------------------------------------------------------------
// BM_QuantizeAC_DCT8: stack-allocated mask (old) vs static const (new).
// ---------------------------------------------------------------------------

HWY_NOINLINE void QuantizeAC_DCT8_Old_Impl(benchmark::State& state) {
  HWY_ALIGN float block_in[kDCTBlockSize];
  HWY_ALIGN int32_t block_out[kDCTBlockSize] = {};
  for (size_t i = 0; i < kDCTBlockSize; i++)
    block_in[i] = static_cast<float>(i % 64) / 64.f + 0.1f;

  HWY_ALIGN float qm[kDCTBlockSize];
  for (size_t i = 0; i < kDCTBlockSize; i++) qm[i] = 1.0f / 64.f;
  float thresholds[4] = {0.58f, 0.64f, 0.64f, 0.64f};

  HWY_CAPPED(float, kBlockDim) df;
  HWY_CAPPED(int32_t, kBlockDim) di;
  HWY_CAPPED(uint32_t, kBlockDim) du;
  const auto quantv = hwy::HWY_NAMESPACE::Set(df, 1.0f);

  for (auto _ : state) {
    (void)_;
    for (size_t y = 0; y < kBlockDim; y++) {
      const size_t yfix = static_cast<size_t>(y >= kBlockDim / 2) * 2;
      const size_t off = y * kBlockDim;
      for (size_t x = 0; x < kBlockDim; x += Lanes(df)) {
        // OLD: stack-allocated every inner iteration.
        HWY_ALIGN uint32_t kMask[kBlockDim] = {0, 0, 0, 0, ~0u, ~0u, ~0u, ~0u};
        const auto mask =
            MaskFromVec(hwy::HWY_NAMESPACE::BitCast(
                df, hwy::HWY_NAMESPACE::Load(du, kMask + x)));
        const auto threshold = IfThenElse(
            mask,
            hwy::HWY_NAMESPACE::Set(df, thresholds[yfix + 1]),
            hwy::HWY_NAMESPACE::Set(df, thresholds[yfix]));
        const auto q = hwy::HWY_NAMESPACE::Mul(
            hwy::HWY_NAMESPACE::Load(df, qm + off + x), quantv);
        const auto val = hwy::HWY_NAMESPACE::Mul(
            q, hwy::HWY_NAMESPACE::Load(df, block_in + off + x));
        const auto v =
            ConvertTo(di, IfThenElseZero(Ge(Abs(val), threshold), Round(val)));
        hwy::HWY_NAMESPACE::Store(v, di, block_out + off + x);
      }
    }
    benchmark::DoNotOptimize(block_out);
  }
}

HWY_NOINLINE void QuantizeAC_DCT8_New_Impl(benchmark::State& state) {
  HWY_ALIGN float block_in[kDCTBlockSize];
  HWY_ALIGN int32_t block_out[kDCTBlockSize] = {};
  for (size_t i = 0; i < kDCTBlockSize; i++)
    block_in[i] = static_cast<float>(i % 64) / 64.f + 0.1f;

  HWY_ALIGN float qm[kDCTBlockSize];
  for (size_t i = 0; i < kDCTBlockSize; i++) qm[i] = 1.0f / 64.f;
  float thresholds[4] = {0.58f, 0.64f, 0.64f, 0.64f};

  HWY_CAPPED(float, kBlockDim) df;
  HWY_CAPPED(int32_t, kBlockDim) di;
  HWY_CAPPED(uint32_t, kBlockDim) du;
  const auto quantv = hwy::HWY_NAMESPACE::Set(df, 1.0f);

  for (auto _ : state) {
    (void)_;
    for (size_t y = 0; y < kBlockDim; y++) {
      const size_t yfix = static_cast<size_t>(y >= kBlockDim / 2) * 2;
      const size_t off = y * kBlockDim;
      for (size_t x = 0; x < kBlockDim; x += Lanes(df)) {
        // NEW: static const — initialized once.
        HWY_ALIGN static const uint32_t kMask[kBlockDim] = {
            0, 0, 0, 0, ~0u, ~0u, ~0u, ~0u};
        const auto mask =
            MaskFromVec(hwy::HWY_NAMESPACE::BitCast(
                df, hwy::HWY_NAMESPACE::Load(du, kMask + x)));
        const auto threshold = IfThenElse(
            mask,
            hwy::HWY_NAMESPACE::Set(df, thresholds[yfix + 1]),
            hwy::HWY_NAMESPACE::Set(df, thresholds[yfix]));
        const auto q = hwy::HWY_NAMESPACE::Mul(
            hwy::HWY_NAMESPACE::Load(df, qm + off + x), quantv);
        const auto val = hwy::HWY_NAMESPACE::Mul(
            q, hwy::HWY_NAMESPACE::Load(df, block_in + off + x));
        const auto v =
            ConvertTo(di, IfThenElseZero(Ge(Abs(val), threshold), Round(val)));
        hwy::HWY_NAMESPACE::Store(v, di, block_out + off + x);
      }
    }
    benchmark::DoNotOptimize(block_out);
  }
}

// ---------------------------------------------------------------------------
// BM_RoundtripY: two-loop (old) vs fused quantize+reconstruct (new).
// ---------------------------------------------------------------------------

HWY_NOINLINE void RoundtripY_Old_Impl(benchmark::State& state) {
  HWY_ALIGN float coeff_in[kDCTBlockSize];
  HWY_ALIGN int32_t quantized[kDCTBlockSize] = {};
  HWY_ALIGN float coeff_recon[kDCTBlockSize] = {};
  for (size_t i = 0; i < kDCTBlockSize; i++)
    coeff_in[i] = static_cast<float>(i % 64) / 64.f + 0.1f;

  HWY_ALIGN float qm[kDCTBlockSize], dqm[kDCTBlockSize];
  for (size_t i = 0; i < kDCTBlockSize; i++) {
    qm[i] = 1.0f / 64.f;
    dqm[i] = 64.0f;
  }
  float thresholds[4] = {0.58f, 0.64f, 0.64f, 0.64f};
  static const float kBiases[4] = {0.f, 0.f, 0.f, 0.f};

  HWY_CAPPED(float, kBlockDim) df;
  HWY_CAPPED(int32_t, kBlockDim) di;
  HWY_CAPPED(uint32_t, kBlockDim) du;
  HWY_CAPPED(float, kDCTBlockSize) dfw;
  HWY_CAPPED(int32_t, kDCTBlockSize) diw;
  const auto quantv = hwy::HWY_NAMESPACE::Set(df, 1.0f);
  const auto inv_qacv = hwy::HWY_NAMESPACE::Set(dfw, 1.0f);

  for (auto _ : state) {
    (void)_;
    // Pass 1: quantize Y.
    for (size_t y = 0; y < kBlockDim; y++) {
      const size_t yfix = static_cast<size_t>(y >= kBlockDim / 2) * 2;
      const size_t off = y * kBlockDim;
      for (size_t x = 0; x < kBlockDim; x += Lanes(df)) {
        HWY_ALIGN static const uint32_t kMask[kBlockDim] = {
            0, 0, 0, 0, ~0u, ~0u, ~0u, ~0u};
        const auto mask =
            MaskFromVec(hwy::HWY_NAMESPACE::BitCast(
                df, hwy::HWY_NAMESPACE::Load(du, kMask + x)));
        const auto threshold = IfThenElse(
            mask,
            hwy::HWY_NAMESPACE::Set(df, thresholds[yfix + 1]),
            hwy::HWY_NAMESPACE::Set(df, thresholds[yfix]));
        const auto q = hwy::HWY_NAMESPACE::Mul(
            hwy::HWY_NAMESPACE::Load(df, qm + off + x), quantv);
        const auto val = hwy::HWY_NAMESPACE::Mul(
            q, hwy::HWY_NAMESPACE::Load(df, coeff_in + off + x));
        const auto v =
            ConvertTo(di, IfThenElseZero(Ge(Abs(val), threshold), Round(val)));
        hwy::HWY_NAMESPACE::Store(v, di, quantized + off + x);
      }
    }
    // Pass 2: reload quantized, reconstruct.
    for (size_t k = 0; k < kDCTBlockSize; k += Lanes(dfw)) {
      const auto q = hwy::HWY_NAMESPACE::Load(diw, quantized + k);
      const auto adj = AdjustQuantBias(diw, 1, q, kBiases);
      const auto d = hwy::HWY_NAMESPACE::Load(dfw, dqm + k);
      hwy::HWY_NAMESPACE::Store(
          hwy::HWY_NAMESPACE::Mul(hwy::HWY_NAMESPACE::Mul(adj, d), inv_qacv),
          dfw, coeff_recon + k);
    }
    benchmark::DoNotOptimize(coeff_recon);
  }
}

HWY_NOINLINE void RoundtripY_New_Impl(benchmark::State& state) {
  HWY_ALIGN float coeff_in[kDCTBlockSize];
  HWY_ALIGN int32_t quantized[kDCTBlockSize] = {};
  HWY_ALIGN float coeff_recon[kDCTBlockSize] = {};
  for (size_t i = 0; i < kDCTBlockSize; i++)
    coeff_in[i] = static_cast<float>(i % 64) / 64.f + 0.1f;

  HWY_ALIGN float qm[kDCTBlockSize], dqm[kDCTBlockSize];
  for (size_t i = 0; i < kDCTBlockSize; i++) {
    qm[i] = 1.0f / 64.f;
    dqm[i] = 64.0f;
  }
  float thresholds[4] = {0.58f, 0.64f, 0.64f, 0.64f};
  static const float kBiases[4] = {0.f, 0.f, 0.f, 0.f};

  HWY_CAPPED(float, kBlockDim) df;
  HWY_CAPPED(int32_t, kBlockDim) di;
  HWY_CAPPED(uint32_t, kBlockDim) du;
  const auto quantv = hwy::HWY_NAMESPACE::Set(df, 1.0f);
  const auto inv_qacv = hwy::HWY_NAMESPACE::Set(df, 1.0f);

  for (auto _ : state) {
    (void)_;
    // Fused: quantize + immediately reconstruct from live register.
    for (size_t y = 0; y < kBlockDim; y++) {
      const size_t yfix = static_cast<size_t>(y >= kBlockDim / 2) * 2;
      const size_t off = y * kBlockDim;
      for (size_t x = 0; x < kBlockDim; x += Lanes(df)) {
        HWY_ALIGN static const uint32_t kMask[kBlockDim] = {
            0, 0, 0, 0, ~0u, ~0u, ~0u, ~0u};
        const auto mask =
            MaskFromVec(hwy::HWY_NAMESPACE::BitCast(
                df, hwy::HWY_NAMESPACE::Load(du, kMask + x)));
        const auto threshold = IfThenElse(
            mask,
            hwy::HWY_NAMESPACE::Set(df, thresholds[yfix + 1]),
            hwy::HWY_NAMESPACE::Set(df, thresholds[yfix]));
        const auto q = hwy::HWY_NAMESPACE::Mul(
            hwy::HWY_NAMESPACE::Load(df, qm + off + x), quantv);
        const auto val = hwy::HWY_NAMESPACE::Mul(
            q, hwy::HWY_NAMESPACE::Load(df, coeff_in + off + x));
        const auto v =
            ConvertTo(di, IfThenElseZero(Ge(Abs(val), threshold), Round(val)));
        hwy::HWY_NAMESPACE::Store(v, di, quantized + off + x);
        // Fused: no store→reload, v is still in registers.
        const auto adj = AdjustQuantBias(di, 1, v, kBiases);
        hwy::HWY_NAMESPACE::Store(
            hwy::HWY_NAMESPACE::Mul(
                hwy::HWY_NAMESPACE::Mul(
                    adj, hwy::HWY_NAMESPACE::Load(df, dqm + off + x)),
                inv_qacv),
            df, coeff_recon + off + x);
      }
    }
    benchmark::DoNotOptimize(coeff_recon);
  }
}

// ---------------------------------------------------------------------------
// BM_CflBoth / BM_CflNone: four-way CfL dispatch.
// ---------------------------------------------------------------------------

HWY_NOINLINE void CflBoth_Old_Impl(benchmark::State& state) {
  HWY_ALIGN float coeffs[3 * kDCTBlockSize];
  for (size_t i = 0; i < 3 * kDCTBlockSize; i++)
    coeffs[i] = static_cast<float>(i % 64) / 64.f + 0.1f;

  constexpr HWY_CAPPED(float, kDCTBlockSize) d;
  const auto xf = hwy::HWY_NAMESPACE::Set(d, Launder(0.3f));
  const auto bf = hwy::HWY_NAMESPACE::Set(d, Launder(-0.1f));
  const size_t sz = kDCTBlockSize;

  for (auto _ : state) {
    (void)_;
    // OLD: always both channels.
    for (size_t k = 0; k < sz; k += Lanes(d)) {
      const auto in_y = hwy::HWY_NAMESPACE::Load(d, coeffs + sz + k);
      hwy::HWY_NAMESPACE::Store(
          hwy::HWY_NAMESPACE::NegMulAdd(
              xf, in_y, hwy::HWY_NAMESPACE::Load(d, coeffs + k)),
          d, coeffs + k);
      hwy::HWY_NAMESPACE::Store(
          hwy::HWY_NAMESPACE::NegMulAdd(
              bf, in_y, hwy::HWY_NAMESPACE::Load(d, coeffs + 2 * sz + k)),
          d, coeffs + 2 * sz + k);
    }
    benchmark::DoNotOptimize(coeffs);
  }
}

HWY_NOINLINE void CflBoth_New_Impl(benchmark::State& state) {
  HWY_ALIGN float coeffs[3 * kDCTBlockSize];
  for (size_t i = 0; i < 3 * kDCTBlockSize; i++)
    coeffs[i] = static_cast<float>(i % 64) / 64.f + 0.1f;

  constexpr HWY_CAPPED(float, kDCTBlockSize) d;
  const float x_ratio = Launder(0.3f), b_ratio = Launder(-0.1f);
  const uint8_t cfl_mask = static_cast<uint8_t>(
      (x_ratio != 0.f ? 1u : 0u) | (b_ratio != 0.f ? 2u : 0u));
  const auto xf = hwy::HWY_NAMESPACE::Set(d, x_ratio);
  const auto bf = hwy::HWY_NAMESPACE::Set(d, b_ratio);
  const size_t sz = kDCTBlockSize;

  for (auto _ : state) {
    (void)_;
    // NEW: four-way dispatch — both active here.
    if (cfl_mask == 3) {
      for (size_t k = 0; k < sz; k += Lanes(d)) {
        const auto in_y = hwy::HWY_NAMESPACE::Load(d, coeffs + sz + k);
        hwy::HWY_NAMESPACE::Store(
            hwy::HWY_NAMESPACE::NegMulAdd(
                xf, in_y, hwy::HWY_NAMESPACE::Load(d, coeffs + k)),
            d, coeffs + k);
        hwy::HWY_NAMESPACE::Store(
            hwy::HWY_NAMESPACE::NegMulAdd(
                bf, in_y,
                hwy::HWY_NAMESPACE::Load(d, coeffs + 2 * sz + k)),
            d, coeffs + 2 * sz + k);
      }
    }
    benchmark::DoNotOptimize(coeffs);
  }
}

HWY_NOINLINE void CflNone_Old_Impl(benchmark::State& state) {
  HWY_ALIGN float coeffs[3 * kDCTBlockSize];
  for (size_t i = 0; i < 3 * kDCTBlockSize; i++)
    coeffs[i] = static_cast<float>(i % 64) / 64.f + 0.1f;

  constexpr HWY_CAPPED(float, kDCTBlockSize) d;
  // OLD: both channels always, factors happen to be zero.
  const auto xf = hwy::HWY_NAMESPACE::Set(d, Launder(0.0f));
  const auto bf = hwy::HWY_NAMESPACE::Set(d, Launder(0.0f));
  const size_t sz = kDCTBlockSize;

  for (auto _ : state) {
    (void)_;
    for (size_t k = 0; k < sz; k += Lanes(d)) {
      const auto in_y = hwy::HWY_NAMESPACE::Load(d, coeffs + sz + k);
      hwy::HWY_NAMESPACE::Store(
          hwy::HWY_NAMESPACE::NegMulAdd(
              xf, in_y, hwy::HWY_NAMESPACE::Load(d, coeffs + k)),
          d, coeffs + k);
      hwy::HWY_NAMESPACE::Store(
          hwy::HWY_NAMESPACE::NegMulAdd(
              bf, in_y, hwy::HWY_NAMESPACE::Load(d, coeffs + 2 * sz + k)),
          d, coeffs + 2 * sz + k);
    }
    benchmark::DoNotOptimize(coeffs);
  }
}

HWY_NOINLINE void CflNone_New_Impl(benchmark::State& state) {
  HWY_ALIGN float coeffs[3 * kDCTBlockSize];
  for (size_t i = 0; i < 3 * kDCTBlockSize; i++)
    coeffs[i] = static_cast<float>(i % 64) / 64.f + 0.1f;

  // NEW: cfl_mask==0 → skip loop body entirely.
  const float x_ratio = Launder(0.0f), b_ratio = Launder(0.0f);
  const uint8_t cfl_mask = static_cast<uint8_t>(
      (x_ratio != 0.f ? 1u : 0u) | (b_ratio != 0.f ? 2u : 0u));

  for (auto _ : state) {
    (void)_;
    if (cfl_mask != 0) {
      // (never entered for x=b=0)
      constexpr HWY_CAPPED(float, kDCTBlockSize) d;
      const auto xf = hwy::HWY_NAMESPACE::Set(d, x_ratio);
      const auto bf = hwy::HWY_NAMESPACE::Set(d, b_ratio);
      const size_t sz = kDCTBlockSize;
      for (size_t k = 0; k < sz; k += Lanes(d)) {
        const auto in_y = hwy::HWY_NAMESPACE::Load(d, coeffs + sz + k);
        hwy::HWY_NAMESPACE::Store(
            hwy::HWY_NAMESPACE::NegMulAdd(
                xf, in_y, hwy::HWY_NAMESPACE::Load(d, coeffs + k)),
            d, coeffs + k);
        hwy::HWY_NAMESPACE::Store(
            hwy::HWY_NAMESPACE::NegMulAdd(
                bf, in_y,
                hwy::HWY_NAMESPACE::Load(d, coeffs + 2 * sz + k)),
            d, coeffs + 2 * sz + k);
      }
    }
    benchmark::DoNotOptimize(coeffs);
  }
}

// ---------------------------------------------------------------------------
// BM_AdjustDCT16: per-coeff branch (old) vs loop-bound fix (new).
// Scalar — no HWY ops needed, but in HWY_NAMESPACE for dispatch machinery.
// ---------------------------------------------------------------------------

HWY_NOINLINE void AdjustDCT16_Old_Impl(benchmark::State& state) {
  constexpr size_t xsize = 2, ysize = 2;
  constexpr size_t W = xsize * kBlockDim;
  constexpr size_t H = ysize * kBlockDim;
  HWY_ALIGN float block_in[W * H];
  float qm[W * H];
  for (size_t i = 0; i < W * H; i++) {
    block_in[i] = static_cast<float>(i % 64) / 128.f;
    qm[i] = 1.0f / 64.f;
  }
  float thresholds[4] = {0.58f, 0.64f, 0.64f, 0.64f};
  const float qac = 1.0f;

  for (auto _ : state) {
    (void)_;
    float sum_of_error = 0, sum_of_vals = 0;
    for (size_t y = 0; y < H; y++) {
      for (size_t x = 0; x < W; x++) {
        // OLD: per-coefficient branch to skip DC region.
        if (x < xsize && y < ysize) continue;
        const size_t pos = y * W + x;
        const size_t hfix =
            (static_cast<size_t>(y >= H / 2) * 2 +
             static_cast<size_t>(x >= W / 2));
        const float val = block_in[pos] * (qm[pos] * qac);
        const float v = (std::abs(val) < thresholds[hfix]) ? 0.f : rintf(val);
        sum_of_error += std::abs(val - v);
        sum_of_vals += std::abs(v);
      }
    }
    benchmark::DoNotOptimize(sum_of_error);
    benchmark::DoNotOptimize(sum_of_vals);
  }
}

HWY_NOINLINE void AdjustDCT16_New_Impl(benchmark::State& state) {
  constexpr size_t xsize = 2, ysize = 2;
  constexpr size_t W = xsize * kBlockDim;
  constexpr size_t H = ysize * kBlockDim;
  HWY_ALIGN float block_in[W * H];
  float qm[W * H];
  for (size_t i = 0; i < W * H; i++) {
    block_in[i] = static_cast<float>(i % 64) / 128.f;
    qm[i] = 1.0f / 64.f;
  }
  float thresholds[4] = {0.58f, 0.64f, 0.64f, 0.64f};
  const float qac = 1.0f;

  for (auto _ : state) {
    (void)_;
    float sum_of_error = 0, sum_of_vals = 0;
    for (size_t y = 0; y < H; y++) {
      // NEW: skip DC region via loop bound, no per-coeff branch.
      const size_t x_begin = (y < ysize) ? xsize : 0;
      for (size_t x = x_begin; x < W; x++) {
        const size_t pos = y * W + x;
        const size_t hfix =
            (static_cast<size_t>(y >= H / 2) * 2 +
             static_cast<size_t>(x >= W / 2));
        const float val = block_in[pos] * (qm[pos] * qac);
        const float v = (std::abs(val) < thresholds[hfix]) ? 0.f : rintf(val);
        sum_of_error += std::abs(val - v);
        sum_of_vals += std::abs(v);
      }
    }
    benchmark::DoNotOptimize(sum_of_error);
    benchmark::DoNotOptimize(sum_of_vals);
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace jxl {
namespace {

HWY_EXPORT(QuantizeAC_DCT8_Old_Impl);
HWY_EXPORT(QuantizeAC_DCT8_New_Impl);
HWY_EXPORT(RoundtripY_Old_Impl);
HWY_EXPORT(RoundtripY_New_Impl);
HWY_EXPORT(CflBoth_Old_Impl);
HWY_EXPORT(CflBoth_New_Impl);
HWY_EXPORT(CflNone_Old_Impl);
HWY_EXPORT(CflNone_New_Impl);
HWY_EXPORT(AdjustDCT16_Old_Impl);
HWY_EXPORT(AdjustDCT16_New_Impl);

static void BM_QuantizeAC_DCT8_Old(benchmark::State& state) {
  HWY_DYNAMIC_DISPATCH(QuantizeAC_DCT8_Old_Impl)(state);
}
BENCHMARK(BM_QuantizeAC_DCT8_Old);

static void BM_QuantizeAC_DCT8_New(benchmark::State& state) {
  HWY_DYNAMIC_DISPATCH(QuantizeAC_DCT8_New_Impl)(state);
}
BENCHMARK(BM_QuantizeAC_DCT8_New);

static void BM_RoundtripY_Old(benchmark::State& state) {
  HWY_DYNAMIC_DISPATCH(RoundtripY_Old_Impl)(state);
}
BENCHMARK(BM_RoundtripY_Old);

static void BM_RoundtripY_New(benchmark::State& state) {
  HWY_DYNAMIC_DISPATCH(RoundtripY_New_Impl)(state);
}
BENCHMARK(BM_RoundtripY_New);

static void BM_CflBoth_Old(benchmark::State& state) {
  HWY_DYNAMIC_DISPATCH(CflBoth_Old_Impl)(state);
}
BENCHMARK(BM_CflBoth_Old);

static void BM_CflBoth_New(benchmark::State& state) {
  HWY_DYNAMIC_DISPATCH(CflBoth_New_Impl)(state);
}
BENCHMARK(BM_CflBoth_New);

static void BM_CflNone_Old(benchmark::State& state) {
  HWY_DYNAMIC_DISPATCH(CflNone_Old_Impl)(state);
}
BENCHMARK(BM_CflNone_Old);

static void BM_CflNone_New(benchmark::State& state) {
  HWY_DYNAMIC_DISPATCH(CflNone_New_Impl)(state);
}
BENCHMARK(BM_CflNone_New);

static void BM_AdjustDCT16_Old(benchmark::State& state) {
  HWY_DYNAMIC_DISPATCH(AdjustDCT16_Old_Impl)(state);
}
BENCHMARK(BM_AdjustDCT16_Old);

static void BM_AdjustDCT16_New(benchmark::State& state) {
  HWY_DYNAMIC_DISPATCH(AdjustDCT16_New_Impl)(state);
}
BENCHMARK(BM_AdjustDCT16_New);

}  // namespace
}  // namespace jxl

#endif  // HWY_ONCE
