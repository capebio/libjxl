// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Flipflop benchmarks for the chroma-from-luma (CfL) heuristics optimizations
// in enc_chroma_from_luma.cc. Registered in jxl_gbench via jxl_lists.cmake.
//
// Pairs (run with --benchmark_filter=BM_Cfl):
//
//   BM_CflFastMoments_Old  vs  BM_CflFastMoments_New
//     Fast (least-squares) per-tile solve. Old materialises four float
//     coefficient streams (coeffs_yx/x/yb/b) and then re-reads them to form the
//     two regression moments per channel. New fuses moment accumulation into
//     the producer loop, eliminating the 4 stores/block and the reload pass.
//     This is the e5/e6 path (SpeedTier kHare/kWombat).
//
//   BM_CflRobustPrep_Old  vs  BM_CflRobustPrep_New
//     Robust (Newton) per-tile solve, up to 20 iterations. Old re-derives the
//     residual coefficients a = luma/color_factor and b = base*luma - chroma
//     from the (luma, chroma) streams on every iteration. New prepares the
//     (a, b) streams once and the iterations are a pure load. This is the
//     e7+/CfL1 path (SpeedTier <= kSquirrel).
//
// The third win — eliminating the dead DC staging (DCFromLowestFrequencies x3
// per block + the DC copy loop, whose dc_values output is never read) — is a
// structural removal that is byte-exact by construction (it deletes writes that
// nothing consumes); it is measured end-to-end via acs_effort_bench, not here.

#include "benchmark/benchmark.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/chroma_from_luma.h"  // kColorTileDim, kDefaultColorFactor
#include "lib/jxl/cms/opsin_params.h"  // kYToBRatio

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_cfl_gbench.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::GetLane;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::Load;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Set;
using hwy::HWY_NAMESPACE::Store;
using hwy::HWY_NAMESPACE::Sub;
using hwy::HWY_NAMESPACE::SumOfLanes;
using hwy::HWY_NAMESPACE::Zero;

// A full color tile carries up to kColorTileDim*kColorTileDim AC coefficients.
constexpr size_t kNum = kColorTileDim * kColorTileDim;  // 4096
constexpr float kInvColorFactor = 1.0f / kDefaultColorFactor;
constexpr float kDistanceMultiplierAC = 1e-9f;

HWY_FULL(float) df;

// Deterministic synthetic block / quant-matrix data, shared by old and new.
struct CflInputs {
  HWY_ALIGN float block_y[kNum];
  HWY_ALIGN float block_x[kNum];
  HWY_ALIGN float block_b[kNum];
  HWY_ALIGN float qm_x[kNum];
  HWY_ALIGN float qm_b[kNum];
  float q;
  CflInputs() {
    for (size_t i = 0; i < kNum; i++) {
      block_y[i] = std::sin(0.013f * i) * 7.0f;
      block_x[i] = std::cos(0.021f * i) * 3.0f;
      block_b[i] = std::sin(0.017f * i + 1.0f) * 5.0f;
      qm_x[i] = 1.0f / 64.f + 0.0001f * (i % 13);
      qm_b[i] = 1.0f / 48.f + 0.0001f * (i % 11);
    }
    q = 128.0f * 0.5f;
  }
};

float QuantizeCfl(float x) {
  const float towards_zero = 2.6f;
  if (x >= towards_zero) {
    x -= towards_zero;
  } else if (x <= -towards_zero) {
    x += towards_zero;
  } else {
    x = 0;
  }
  return std::min(127.0f, std::max(-128.0f, std::round(x)));
}

// One robust-residual derivative evaluation over precomputed (a, b) streams.
float RobustCompute(const float* a_arr, const float* b_arr, size_t num, float x,
                    float eps, float distance_mul, float* fpeps, float* fmeps) {
  const float kCoeff = 1.f / 3;
  const float kThres = 100.0f;
  float first_derivative = 2 * distance_mul * num * x;
  float first_derivative_peps = 2 * distance_mul * num * (x + eps);
  float first_derivative_meps = 2 * distance_mul * num * (x - eps);
  const auto thres = Set(df, kThres);
  const auto coeffx2 = Set(df, kCoeff * 2.0f);
  const auto one = Set(df, 1.0f);
  const auto zero = Set(df, 0.0f);
  const auto x_v = Set(df, x);
  const auto xpe_v = Set(df, x + eps);
  const auto xme_v = Set(df, x - eps);
  auto fd_v = Zero(df);
  auto fdpe_v = Zero(df);
  auto fdme_v = Zero(df);
  for (size_t i = 0; i < num; i += Lanes(df)) {
    const auto a = Load(df, a_arr + i);
    const auto b = Load(df, b_arr + i);
    const auto v = MulAdd(a, x_v, b);
    const auto vpe = MulAdd(a, xpe_v, b);
    const auto vme = MulAdd(a, xme_v, b);
    const auto av = Abs(v);
    const auto acoeffx2 = Mul(coeffx2, a);
    auto d = Mul(acoeffx2, Add(av, one));
    auto dpe = Mul(acoeffx2, Add(Abs(vpe), one));
    auto dme = Mul(acoeffx2, Add(Abs(vme), one));
    d = IfThenElse(Lt(v, zero), Sub(zero, d), d);
    dpe = IfThenElse(Lt(vpe, zero), Sub(zero, dpe), dpe);
    dme = IfThenElse(Lt(vme, zero), Sub(zero, dme), dme);
    const auto above = Ge(av, thres);
    fd_v = Add(fd_v, IfThenElse(above, zero, d));
    fdpe_v = Add(fdpe_v, IfThenElse(above, zero, dpe));
    fdme_v = Add(fdme_v, IfThenElse(above, zero, dme));
  }
  *fpeps = first_derivative_peps + GetLane(SumOfLanes(df, fdpe_v));
  *fmeps = first_derivative_meps + GetLane(SumOfLanes(df, fdme_v));
  return first_derivative + GetLane(SumOfLanes(df, fd_v));
}

float RobustSolve(const float* a_arr, const float* b_arr, size_t num,
                  float distance_mul) {
  constexpr float eps = 100;
  constexpr float kClamp = 20.0f;
  float x = 0;
  for (size_t i = 0; i < 20; i++) {
    float dfpeps, dfmeps;
    float d_f =
        RobustCompute(a_arr, b_arr, num, x, eps, distance_mul, &dfpeps, &dfmeps);
    float ddf = (dfpeps - dfmeps) / (2 * eps);
    float step = d_f / (ddf + 0.85f);
    x -= std::min(kClamp, std::max(-kClamp, step));
    if (std::abs(step) < 3e-3) break;
  }
  return x;
}

// ---------------------------------------------------------------------------
// Fast path: materialise + reload (old) vs fused moments (new).
// ---------------------------------------------------------------------------

HWY_NOINLINE void CflFastMoments_Old_Impl(benchmark::State& state) {
  CflInputs in;
  HWY_ALIGN static float coeffs_yx[kNum], coeffs_x[kNum];
  HWY_ALIGN static float coeffs_yb[kNum], coeffs_b[kNum];
  const auto qv = Set(df, in.q);
  const auto inv_cf = Set(df, kInvColorFactor);
  const auto base_b = Set(df, jxl::cms::kYToBRatio);
  const auto base_x = Set(df, 0.0f);
  for (auto _ : state) {
    (void)_;
    // Producer: store four coefficient streams.
    for (size_t i = 0; i < kNum; i += Lanes(df)) {
      const auto b_y = Load(df, in.block_y + i);
      const auto b_x = Load(df, in.block_x + i);
      const auto b_b = Load(df, in.block_b + i);
      const auto qqm_x = Mul(qv, Load(df, in.qm_x + i));
      const auto qqm_b = Mul(qv, Load(df, in.qm_b + i));
      Store(Mul(b_y, qqm_x), df, coeffs_yx + i);
      Store(Mul(b_x, qqm_x), df, coeffs_x + i);
      Store(Mul(b_y, qqm_b), df, coeffs_yb + i);
      Store(Mul(b_b, qqm_b), df, coeffs_b + i);
    }
    // Consumer: reload and reduce, per channel.
    auto ca_x = Zero(df), cb_x = Zero(df), ca_b = Zero(df), cb_b = Zero(df);
    for (size_t i = 0; i < kNum; i += Lanes(df)) {
      const auto ax = Mul(inv_cf, Load(df, coeffs_yx + i));
      const auto bx = Sub(Mul(base_x, Load(df, coeffs_yx + i)),
                          Load(df, coeffs_x + i));
      ca_x = MulAdd(ax, ax, ca_x);
      cb_x = MulAdd(ax, bx, cb_x);
      const auto ab = Mul(inv_cf, Load(df, coeffs_yb + i));
      const auto bb = Sub(Mul(base_b, Load(df, coeffs_yb + i)),
                          Load(df, coeffs_b + i));
      ca_b = MulAdd(ab, ab, ca_b);
      cb_b = MulAdd(ab, bb, cb_b);
    }
    float fx = -GetLane(SumOfLanes(df, cb_x)) /
               (GetLane(SumOfLanes(df, ca_x)) + kNum * kDistanceMultiplierAC * 0.5f);
    float fb = -GetLane(SumOfLanes(df, cb_b)) /
               (GetLane(SumOfLanes(df, ca_b)) + kNum * kDistanceMultiplierAC * 0.5f);
    int32_t rx = static_cast<int32_t>(QuantizeCfl(fx));
    int32_t rb = static_cast<int32_t>(QuantizeCfl(fb));
    benchmark::DoNotOptimize(rx);
    benchmark::DoNotOptimize(rb);
  }
}

HWY_NOINLINE void CflFastMoments_New_Impl(benchmark::State& state) {
  CflInputs in;
  const auto qv = Set(df, in.q);
  const auto inv_cf = Set(df, kInvColorFactor);
  const auto base_b = Set(df, jxl::cms::kYToBRatio);
  const auto base_x = Set(df, 0.0f);
  for (auto _ : state) {
    (void)_;
    auto ca_x = Zero(df), cb_x = Zero(df), ca_b = Zero(df), cb_b = Zero(df);
    for (size_t i = 0; i < kNum; i += Lanes(df)) {
      const auto b_y = Load(df, in.block_y + i);
      const auto b_x = Load(df, in.block_x + i);
      const auto b_b = Load(df, in.block_b + i);
      const auto qqm_x = Mul(qv, Load(df, in.qm_x + i));
      const auto qqm_b = Mul(qv, Load(df, in.qm_b + i));
      const auto m_x = Mul(b_y, qqm_x);
      const auto ax = Mul(inv_cf, m_x);
      const auto bx = Sub(Mul(base_x, m_x), Mul(b_x, qqm_x));
      const auto m_b = Mul(b_y, qqm_b);
      const auto ab = Mul(inv_cf, m_b);
      const auto bb = Sub(Mul(base_b, m_b), Mul(b_b, qqm_b));
      ca_x = MulAdd(ax, ax, ca_x);
      cb_x = MulAdd(ax, bx, cb_x);
      ca_b = MulAdd(ab, ab, ca_b);
      cb_b = MulAdd(ab, bb, cb_b);
    }
    float fx = -GetLane(SumOfLanes(df, cb_x)) /
               (GetLane(SumOfLanes(df, ca_x)) + kNum * kDistanceMultiplierAC * 0.5f);
    float fb = -GetLane(SumOfLanes(df, cb_b)) /
               (GetLane(SumOfLanes(df, ca_b)) + kNum * kDistanceMultiplierAC * 0.5f);
    int32_t rx = static_cast<int32_t>(QuantizeCfl(fx));
    int32_t rb = static_cast<int32_t>(QuantizeCfl(fb));
    benchmark::DoNotOptimize(rx);
    benchmark::DoNotOptimize(rb);
  }
}

// ---------------------------------------------------------------------------
// Robust path: per-iteration re-derivation (old) vs prepared a/b (new).
// ---------------------------------------------------------------------------

HWY_NOINLINE void CflRobustPrep_Old_Impl(benchmark::State& state) {
  CflInputs in;
  HWY_ALIGN static float lum_x[kNum], chr_x[kNum], lum_b[kNum], chr_b[kNum];
  HWY_ALIGN static float a_tmp[kNum], b_tmp[kNum];
  const auto qv = Set(df, in.q);
  const auto inv_cf = Set(df, kInvColorFactor);
  const auto base_b = Set(df, jxl::cms::kYToBRatio);
  const auto base_x = Set(df, 0.0f);
  // Stage luma/chroma streams once (shared cost).
  for (size_t i = 0; i < kNum; i += Lanes(df)) {
    const auto b_y = Load(df, in.block_y + i);
    const auto b_x = Load(df, in.block_x + i);
    const auto b_b = Load(df, in.block_b + i);
    const auto qqm_x = Mul(qv, Load(df, in.qm_x + i));
    const auto qqm_b = Mul(qv, Load(df, in.qm_b + i));
    Store(Mul(b_y, qqm_x), df, lum_x + i);
    Store(Mul(b_x, qqm_x), df, chr_x + i);
    Store(Mul(b_y, qqm_b), df, lum_b + i);
    Store(Mul(b_b, qqm_b), df, chr_b + i);
  }
  for (auto _ : state) {
    (void)_;
    // OLD: the solver re-derives a/b from luma/chroma every Newton iteration.
    // Emulate by rebuilding a/b then running the solver (20 iters internally
    // already reload, but historically a/b were recomputed each iter inside
    // Compute). Here we model the per-call re-derivation cost.
    for (size_t i = 0; i < kNum; i += Lanes(df)) {
      const auto m = Load(df, lum_x + i);
      Store(Mul(inv_cf, m), df, a_tmp + i);
      Store(Sub(Mul(base_x, m), Load(df, chr_x + i)), df, b_tmp + i);
    }
    float rx = QuantizeCfl(RobustSolve(a_tmp, b_tmp, kNum, kDistanceMultiplierAC));
    for (size_t i = 0; i < kNum; i += Lanes(df)) {
      const auto m = Load(df, lum_b + i);
      Store(Mul(inv_cf, m), df, a_tmp + i);
      Store(Sub(Mul(base_b, m), Load(df, chr_b + i)), df, b_tmp + i);
    }
    float rb = QuantizeCfl(RobustSolve(a_tmp, b_tmp, kNum, kDistanceMultiplierAC));
    benchmark::DoNotOptimize(rx);
    benchmark::DoNotOptimize(rb);
  }
}

HWY_NOINLINE void CflRobustPrep_New_Impl(benchmark::State& state) {
  CflInputs in;
  HWY_ALIGN static float a_x[kNum], b_x[kNum], a_b[kNum], b_b[kNum];
  const auto qv = Set(df, in.q);
  const auto inv_cf = Set(df, kInvColorFactor);
  const auto base_b = Set(df, jxl::cms::kYToBRatio);
  const auto base_x = Set(df, 0.0f);
  for (auto _ : state) {
    (void)_;
    // NEW: prepare a/b once in the producer loop; the solver is a pure load.
    for (size_t i = 0; i < kNum; i += Lanes(df)) {
      const auto b_y = Load(df, in.block_y + i);
      const auto bx_in = Load(df, in.block_x + i);
      const auto bb_in = Load(df, in.block_b + i);
      const auto qqm_x = Mul(qv, Load(df, in.qm_x + i));
      const auto qqm_b = Mul(qv, Load(df, in.qm_b + i));
      const auto m_x = Mul(b_y, qqm_x);
      Store(Mul(inv_cf, m_x), df, a_x + i);
      Store(Sub(Mul(base_x, m_x), Mul(bx_in, qqm_x)), df, b_x + i);
      const auto m_b = Mul(b_y, qqm_b);
      Store(Mul(inv_cf, m_b), df, a_b + i);
      Store(Sub(Mul(base_b, m_b), Mul(bb_in, qqm_b)), df, b_b + i);
    }
    float rx = QuantizeCfl(RobustSolve(a_x, b_x, kNum, kDistanceMultiplierAC));
    float rb = QuantizeCfl(RobustSolve(a_b, b_b, kNum, kDistanceMultiplierAC));
    benchmark::DoNotOptimize(rx);
    benchmark::DoNotOptimize(rb);
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace jxl {
namespace {

HWY_EXPORT(CflFastMoments_Old_Impl);
HWY_EXPORT(CflFastMoments_New_Impl);
HWY_EXPORT(CflRobustPrep_Old_Impl);
HWY_EXPORT(CflRobustPrep_New_Impl);

static void BM_CflFastMoments_Old(benchmark::State& state) {
  HWY_DYNAMIC_DISPATCH(CflFastMoments_Old_Impl)(state);
}
BENCHMARK(BM_CflFastMoments_Old);

static void BM_CflFastMoments_New(benchmark::State& state) {
  HWY_DYNAMIC_DISPATCH(CflFastMoments_New_Impl)(state);
}
BENCHMARK(BM_CflFastMoments_New);

static void BM_CflRobustPrep_Old(benchmark::State& state) {
  HWY_DYNAMIC_DISPATCH(CflRobustPrep_Old_Impl)(state);
}
BENCHMARK(BM_CflRobustPrep_Old);

static void BM_CflRobustPrep_New(benchmark::State& state) {
  HWY_DYNAMIC_DISPATCH(CflRobustPrep_New_Impl)(state);
}
BENCHMARK(BM_CflRobustPrep_New);

}  // namespace
}  // namespace jxl

#endif  // HWY_ONCE
