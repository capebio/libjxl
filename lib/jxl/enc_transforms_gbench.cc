// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Flipflop benchmarks for enc_transforms optimizations.
// Registered in jxl_gbench via jxl_lists.cmake.
//
// Pairs (run with --benchmark_filter=BM_DCT2X2 / BM_AFV / BM_DCFrom):
//
//   BM_DCT2X2_ThreePass  vs  BM_DCT2X2_Fused
//     Three materialisation passes vs single-pass fused scalar bypass.
//
//   BM_AFV_Original  vs  BM_AFV_Strided
//     block-copy + AFVDCT4x4 vs AFVDCT4x4Strided (no intermediate copy,
//     dead-store loop halved to ix<4).
//
//   BM_DCFromLowest_Direct  /  BM_DCFromLowest_FastPath  /  BM_DCFromLowest_Dispatch
//     Raw dc[0]=block[0] baseline, pre-dispatch switch (trivial cases),
//     and full Highway dispatch on DCT16X16 (shows dispatch cost).

#include "benchmark/benchmark.h"
#include "lib/jxl/enc_transforms.h"

#include <cstddef>
#include <cstdint>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/dct_scales.h"
#include "lib/jxl/frame_dimensions.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_transforms_gbench.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/dct-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// -----------------------------------------------------------------------
// Reference (old) implementations inlined for comparison.
// -----------------------------------------------------------------------

template <size_t S>
void DCT2TopBlock_Ref(const float* block, size_t stride, float* out) {
  static_assert(kBlockDim % S == 0, "");
  static_assert(S % 2 == 0, "");
  float temp[kDCTBlockSize];
  constexpr size_t num_2x2 = S / 2;
  for (size_t y = 0; y < num_2x2; y++) {
    for (size_t x = 0; x < num_2x2; x++) {
      float c00 = block[y * 2 * stride + x * 2];
      float c01 = block[y * 2 * stride + x * 2 + 1];
      float c10 = block[(y * 2 + 1) * stride + x * 2];
      float c11 = block[(y * 2 + 1) * stride + x * 2 + 1];
      float r00 = (c00 + c01 + c10 + c11) * 0.25f;
      float r01 = (c00 + c01 - c10 - c11) * 0.25f;
      float r10 = (c00 - c01 + c10 - c11) * 0.25f;
      float r11 = (c00 - c01 - c10 + c11) * 0.25f;
      temp[y * kBlockDim + x] = r00;
      temp[y * kBlockDim + num_2x2 + x] = r01;
      temp[(y + num_2x2) * kBlockDim + x] = r10;
      temp[(y + num_2x2) * kBlockDim + num_2x2 + x] = r11;
    }
  }
  for (size_t y = 0; y < S; y++) {
    for (size_t x = 0; x < S; x++) {
      out[y * kBlockDim + x] = temp[y * kBlockDim + x];
    }
  }
}

// Old AFVDCT4x4 (k4x4AFVBasisTranspose was function-local before refactor).
HWY_NOINLINE void AFVDCT4x4_Ref(const float* JXL_RESTRICT pixels,
                                 float* JXL_RESTRICT coeffs) {
  HWY_ALIGN static constexpr float kBasis[16][16] = {
      {0.2500000000000000f, 0.8769029297991420f, 0.f, 0.f, 0.f,
       -0.4105377591765233f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
       0.f},
      {0.2500000000000000f, 0.2206518106944235f, 0.f, 0.f,
       -0.7071067811865474f, 0.6235485373547691f, 0.f, 0.f, 0.f, 0.f, 0.f,
       0.f, 0.f, 0.f, 0.f, 0.f},
      {0.2500000000000000f, -0.1014005039375376f, 0.4067007583026075f,
       -0.2125574805828875f, 0.f, -0.0643507165794627f, -0.4517556589999482f,
       -0.3046847507248690f, 0.3017929516615495f, 0.4082482904638627f,
       0.1747866975480809f, -0.2110560104933578f, -0.1426608480880726f,
       -0.1381354035075859f, -0.1743760259965107f, 0.1135498731499434f},
      {0.2500000000000000f, -0.1014005039375375f, 0.4444481661973445f,
       0.3085497062849767f, 0.f, -0.0643507165794627f, 0.1585450355184006f,
       0.5112616136591823f, 0.2579236279634118f, 0.f, 0.0812611176717539f,
       0.1856718091610980f, -0.3416446842253372f, 0.3302282550303788f,
       0.0702790691196284f, -0.0741750459581035f},
      {0.2500000000000000f, 0.2206518106944236f, 0.f, 0.f,
       0.7071067811865476f, 0.6235485373547694f, 0.f, 0.f, 0.f, 0.f, 0.f,
       0.f, 0.f, 0.f, 0.f, 0.f},
      {0.2500000000000000f, -0.1014005039375378f, 0.f, 0.4706702258572536f,
       0.f, -0.0643507165794628f, -0.0403851516082220f, 0.f,
       0.1627234014286620f, 0.f, 0.f, 0.f, 0.7367497537172237f,
       0.0875511500058708f, -0.2921026642334881f, 0.1940289303259434f},
      {0.2500000000000000f, -0.1014005039375377f, 0.1957439937204294f,
       -0.1621205195722993f, 0.f, -0.0643507165794628f, 0.0074182263792424f,
       -0.2904801297289980f, 0.0952002265347504f, 0.f, -0.3675398009862027f,
       0.4921585901373873f, 0.2462710772207515f, -0.0794670660590957f,
       0.3623817333531167f, -0.4351904965232280f},
      {0.2500000000000000f, -0.1014005039375376f, 0.2929100136981264f, 0.f,
       0.f, -0.0643507165794627f, 0.3935103426921017f, -0.0657870154914280f,
       0.f, -0.4082482904638628f, -0.3078822139579090f, -0.3852501370925192f,
       -0.0857401903551931f, -0.4613374887461511f, 0.f, 0.2191868483885747f},
      {0.2500000000000000f, -0.1014005039375376f, -0.4067007583026072f,
       -0.2125574805828705f, 0.f, -0.0643507165794627f, -0.4517556589999464f,
       0.3046847507248840f, 0.3017929516615503f, -0.4082482904638635f,
       -0.1747866975480813f, 0.2110560104933581f, -0.1426608480880734f,
       -0.1381354035075829f, -0.1743760259965108f, 0.1135498731499426f},
      {0.2500000000000000f, -0.1014005039375377f, -0.1957439937204287f,
       -0.1621205195722833f, 0.f, -0.0643507165794628f, 0.0074182263792444f,
       0.2904801297290076f, 0.0952002265347505f, 0.f, 0.3675398009862011f,
       -0.4921585901373891f, 0.2462710772207514f, -0.0794670660591026f,
       0.3623817333531165f, -0.4351904965232251f},
      {0.2500000000000000f, -0.1014005039375375f, 0.f, -0.4706702258572528f,
       0.f, -0.0643507165794627f, 0.1107416575309343f, 0.f,
       -0.1627234014286617f, 0.f, 0.f, 0.f, 0.1488339922711357f,
       0.4972464710953509f, 0.2921026642334879f, 0.5550443808910661f},
      {0.2500000000000000f, -0.1014005039375377f, 0.1137907446044809f,
       -0.1464291867126764f, 0.f, -0.0643507165794628f, 0.0829816309488205f,
       -0.2388977352334460f, -0.3531238544981630f, -0.4082482904638630f,
       0.4826689115059883f, 0.1741941265991622f, -0.0476868035022925f,
       0.1253805944856366f, -0.4326608024727445f, -0.2546827712406646f},
      {0.2500000000000000f, -0.1014005039375377f, -0.4444481661973438f,
       0.3085497062849487f, 0.f, -0.0643507165794628f, 0.1585450355183970f,
       -0.5112616136592012f, 0.2579236279634129f, 0.f, -0.0812611176717504f,
       -0.1856718091610990f, -0.3416446842253373f, 0.3302282550303805f,
       0.0702790691196282f, -0.0741750459581023f},
      {0.2500000000000000f, -0.1014005039375376f, -0.2929100136981264f, 0.f,
       0.f, -0.0643507165794627f, 0.3935103426921022f, 0.0657870154914254f,
       0.f, 0.4082482904638634f, 0.3078822139579031f, 0.3852501370925211f,
       -0.0857401903551927f, -0.4613374887461554f, 0.f, 0.2191868483885728f},
      {0.2500000000000000f, -0.1014005039375376f, -0.1137907446044814f,
       -0.1464291867126654f, 0.f, -0.0643507165794627f, 0.0829816309488214f,
       0.2388977352334547f, -0.3531238544981624f, 0.4082482904638630f,
       -0.4826689115059858f, -0.1741941265991621f, -0.0476868035022928f,
       0.1253805944856431f, -0.4326608024727457f, -0.2546827712406641f},
      {0.2500000000000000f, -0.1014005039375374f, 0.f, 0.4251149611657548f,
       0.f, -0.0643507165794626f, -0.4517556589999480f, 0.f,
       -0.6035859033230976f, 0.f, 0.f, 0.f, -0.1426608480880724f,
       -0.1381354035075845f, 0.3487520519930227f, 0.1135498731499429f},
  };
  const HWY_CAPPED(float, 16) d;
  for (size_t i = 0; i < 16; i += Lanes(d)) {
    auto scalar = Zero(d);
    for (size_t j = 0; j < 16; j++) {
      auto px = Set(d, pixels[j]);
      auto basis = Load(d, kBasis[j] + i);
      scalar = MulAdd(px, basis, scalar);
    }
    Store(scalar, d, coeffs + i);
  }
}

// -----------------------------------------------------------------------
// Benchmark implementations (take benchmark::State& directly so they
// can be called via HWY_DYNAMIC_DISPATCH with the right target).
// -----------------------------------------------------------------------

HWY_NOINLINE void BM_DCT2X2_ThreePass_Impl(benchmark::State& state,
                                            float* pixels, float* out_a,
                                            float* out_b) {
  float* outs[2] = {out_a, out_b};
  size_t i = 0;
  for (auto _ : state) {
    (void)_;
    float tmp[kDCTBlockSize];
    DCT2TopBlock_Ref<8>(pixels, kBlockDim, tmp);
    DCT2TopBlock_Ref<4>(tmp, kBlockDim, tmp);
    DCT2TopBlock_Ref<2>(tmp, kBlockDim, outs[i & 1]);
    ++i;
  }
}

HWY_NOINLINE void BM_AFV_Original_Impl(benchmark::State& state,
                                       float* pixels, float* out_a,
                                       float* out_b, float* scratch) {
  float* outs[2] = {out_a, out_b};
  size_t i = 0;
  for (auto _ : state) {
    (void)_;
    // afv_kind==0: afv_x=0, afv_y=0, no flips.
    HWY_ALIGN float block[4 * 8] = {};
    for (size_t iy = 0; iy < 4; iy++) {
      for (size_t ix = 0; ix < 4; ix++) {
        block[iy * 4 + ix] = pixels[iy * kBlockDim + ix];
      }
    }
    HWY_ALIGN float coeff[4 * 4];
    AFVDCT4x4_Ref(block, coeff);
    float* dst = outs[i & 1];
    for (size_t iy = 0; iy < 4; iy++) {
      for (size_t ix = 0; ix < 4; ix++) {
        dst[iy * 2 * 8 + ix * 2] = coeff[iy * 4 + ix];
      }
    }
    ComputeScaledDCT<4, 4>()(DCTFrom(pixels + 4, kBlockDim), block, scratch);
    // Old: ix < 8 (16 dead stores).
    for (size_t iy = 0; iy < 4; iy++) {
      for (size_t ix = 0; ix < 8; ix++) {
        dst[iy * 2 * 8 + ix * 2 + 1] = block[iy * 4 + ix];
      }
    }
    ComputeScaledDCT<4, 8>()(
        DCTFrom(pixels + 4 * kBlockDim, kBlockDim), block, scratch);
    for (size_t iy = 0; iy < 4; iy++) {
      for (size_t ix = 0; ix < 8; ix++) {
        dst[(1 + iy * 2) * 8 + ix] = block[iy * 8 + ix];
      }
    }
    float b00 = dst[0] * 0.25f;
    float b01 = dst[1];
    float b10 = dst[8];
    dst[0] = (b00 + b01 + 2 * b10) * 0.25f;
    dst[1] = (b00 - b01) * 0.5f;
    dst[8] = (b00 + b01 - 2 * b10) * 0.25f;
    ++i;
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace jxl {
namespace {

HWY_EXPORT(BM_DCT2X2_ThreePass_Impl);
HWY_EXPORT(BM_AFV_Original_Impl);

// -----------------------------------------------------------------------
// Benchmark registrations.
// -----------------------------------------------------------------------

static void BM_DCT2X2_ThreePass(benchmark::State& state) {
  HWY_ALIGN_MAX float pixels[kDCTBlockSize];
  HWY_ALIGN_MAX float out_a[kDCTBlockSize] = {};
  HWY_ALIGN_MAX float out_b[kDCTBlockSize] = {};
  for (size_t i = 0; i < kDCTBlockSize; i++) {
    pixels[i] = static_cast<float>(i % 64) / 64.f + 0.1f;
  }
  HWY_DYNAMIC_DISPATCH(BM_DCT2X2_ThreePass_Impl)(state, pixels, out_a, out_b);
  benchmark::DoNotOptimize(out_a);
  benchmark::DoNotOptimize(out_b);
}
BENCHMARK(BM_DCT2X2_ThreePass);

static void BM_DCT2X2_Fused(benchmark::State& state) {
  HWY_ALIGN_MAX float pixels[kDCTBlockSize];
  HWY_ALIGN_MAX float out_a[kDCTBlockSize] = {};
  HWY_ALIGN_MAX float out_b[kDCTBlockSize] = {};
  for (size_t i = 0; i < kDCTBlockSize; i++) {
    pixels[i] = static_cast<float>(i % 64) / 64.f + 0.1f;
  }
  float* outs[2] = {out_a, out_b};
  size_t k = 0;
  for (auto _ : state) {
    (void)_;
    // Hits scalar bypass in TransformFromPixels — no Highway dispatch.
    TransformFromPixels(AcStrategyType::DCT2X2, pixels, kBlockDim, outs[k & 1],
                        /*scratch_space=*/nullptr);
    ++k;
  }
  benchmark::DoNotOptimize(out_a);
  benchmark::DoNotOptimize(out_b);
}
BENCHMARK(BM_DCT2X2_Fused);

static void BM_AFV_Original(benchmark::State& state) {
  HWY_ALIGN_MAX float pixels[kDCTBlockSize];
  HWY_ALIGN_MAX float out_a[kDCTBlockSize] = {};
  HWY_ALIGN_MAX float out_b[kDCTBlockSize] = {};
  HWY_ALIGN_MAX float scratch[4 * 8 * 5] = {};
  for (size_t i = 0; i < kDCTBlockSize; i++) {
    pixels[i] = static_cast<float>(i % 64) / 64.f + 0.1f;
  }
  HWY_DYNAMIC_DISPATCH(BM_AFV_Original_Impl)
  (state, pixels, out_a, out_b, scratch);
  benchmark::DoNotOptimize(out_a);
  benchmark::DoNotOptimize(out_b);
}
BENCHMARK(BM_AFV_Original);

static void BM_AFV_Strided(benchmark::State& state) {
  HWY_ALIGN_MAX float pixels[kDCTBlockSize];
  HWY_ALIGN_MAX float out_a[kDCTBlockSize] = {};
  HWY_ALIGN_MAX float out_b[kDCTBlockSize] = {};
  HWY_ALIGN_MAX float scratch[4 * 8 * 5] = {};
  for (size_t i = 0; i < kDCTBlockSize; i++) {
    pixels[i] = static_cast<float>(i % 64) / 64.f + 0.1f;
  }
  float* outs[2] = {out_a, out_b};
  size_t k = 0;
  for (auto _ : state) {
    (void)_;
    TransformFromPixels(AcStrategyType::AFV0, pixels, kBlockDim, outs[k & 1],
                        scratch);
    ++k;
  }
  benchmark::DoNotOptimize(out_a);
  benchmark::DoNotOptimize(out_b);
}
BENCHMARK(BM_AFV_Strided);

// dc[0]=block[0] absolute baseline — no function call.
static void BM_DCFromLowest_Direct(benchmark::State& state) {
  HWY_ALIGN_MAX float block[kDCTBlockSize] = {};
  volatile float dc = 0.f;
  block[0] = 1.0f;
  for (auto _ : state) {
    (void)_;
    dc = block[0];
  }
  benchmark::DoNotOptimize(dc);
}
BENCHMARK(BM_DCFromLowest_Direct);

// Pre-dispatch switch catches trivial cases (DCT → dc[0]=block[0]).
static void BM_DCFromLowest_FastPath(benchmark::State& state) {
  HWY_ALIGN_MAX float block[kDCTBlockSize] = {};
  HWY_ALIGN_MAX float scratch[4 * 32 * 32] = {};
  block[0] = 1.0f;
  float dc_storage[4] = {};
  size_t k = 0;
  for (auto _ : state) {
    (void)_;
    DCFromLowestFrequencies(AcStrategyType::DCT, block, &dc_storage[k & 3], 1,
                            scratch);
    ++k;
  }
  benchmark::DoNotOptimize(dc_storage);
}
BENCHMARK(BM_DCFromLowest_FastPath);

// DCT16X16 still dispatches through Highway — shows per-call dispatch cost.
static void BM_DCFromLowest_Dispatch(benchmark::State& state) {
  HWY_ALIGN_MAX float block[4 * kDCTBlockSize] = {};
  HWY_ALIGN_MAX float scratch[4 * 32 * 32] = {};
  block[0] = 1.0f;
  float dc_storage[2 * 2 * 2] = {};
  size_t k = 0;
  for (auto _ : state) {
    (void)_;
    DCFromLowestFrequencies(AcStrategyType::DCT16X16, block,
                            dc_storage + (k & 1) * 4, 2, scratch);
    ++k;
  }
  benchmark::DoNotOptimize(dc_storage);
}
BENCHMARK(BM_DCFromLowest_Dispatch);

}  // namespace
}  // namespace jxl

#endif  // HWY_ONCE
