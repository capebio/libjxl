// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/frame_dimensions.h"

#if defined(LIB_JXL_ENC_TRANSFORMS_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_ENC_TRANSFORMS_INL_H_
#undef LIB_JXL_ENC_TRANSFORMS_INL_H_
#else
#define LIB_JXL_ENC_TRANSFORMS_INL_H_
#endif

#include <cstddef>
#include <cstdint>
#include <hwy/highway.h>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/dct-inl.h"
#include "lib/jxl/dct_scales.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {

enum class AcStrategyType : uint32_t;

namespace HWY_NAMESPACE {
namespace {

using hwy::HWY_NAMESPACE::Vec;

constexpr size_t kMaxBlocks = 32;

// ToBlock adaptor that writes DCT output row `i` directly to coefficient row
// `(base_row + i * row_step)` of the 8-wide coefficient block. Passed as the
// `final_to` sink in the 4-arg ComputeScaledDCT overload to eliminate the
// intermediate block[] copy and the caller scatter loop for DCT8X4/DCT4X8.
class CoeffRowSinkTo {
 public:
  CoeffRowSinkTo(float* JXL_RESTRICT coeffs, size_t base_row, size_t row_step)
      : coeffs_(coeffs), base_row_(base_row), row_step_(row_step) {}

  template <typename D>
  HWY_INLINE void StorePart(D d, const Vec<D>& v, size_t row, size_t off) const {
    StoreU(v, d, Address(row, off));
  }

  HWY_INLINE void Write(float v, size_t row, size_t off) const {
    *Address(row, off) = v;
  }

  HWY_INLINE float* Address(size_t row, size_t off) const {
    return coeffs_ + (base_row_ + row * row_step_) * kBlockDim + off;
  }

  size_t Stride() const { return kBlockDim; }

 private:
  float* JXL_RESTRICT coeffs_;
  size_t base_row_;
  size_t row_step_;
};

// Inverse of ReinterpretingDCT.
template <size_t DCT_ROWS, size_t DCT_COLS, size_t LF_ROWS, size_t LF_COLS,
          size_t ROWS, size_t COLS>
HWY_INLINE void ReinterpretingIDCT(const float* input,
                                   const size_t input_stride, float* output,
                                   const size_t output_stride, float* scratch) {
  static_assert(ROWS <= kMaxBlocks, "Unsupported block size");
  static_assert(COLS <= kMaxBlocks, "Unsupported block size");
  float* block = scratch;
  if (ROWS < COLS) {
    for (size_t y = 0; y < LF_ROWS; y++) {
      for (size_t x = 0; x < LF_COLS; x++) {
        block[y * COLS + x] = input[y * input_stride + x] *
                              DCTTotalResampleScale<DCT_ROWS, ROWS>(y) *
                              DCTTotalResampleScale<DCT_COLS, COLS>(x);
      }
    }
  } else {
    for (size_t y = 0; y < LF_COLS; y++) {
      for (size_t x = 0; x < LF_ROWS; x++) {
        block[y * ROWS + x] = input[y * input_stride + x] *
                              DCTTotalResampleScale<DCT_COLS, COLS>(y) *
                              DCTTotalResampleScale<DCT_ROWS, ROWS>(x);
      }
    }
  }

  float* scratch_space = scratch + kMaxBlocks * kMaxBlocks;
  ComputeScaledIDCT<ROWS, COLS>()(block, DCTTo(output, output_stride),
                                  scratch_space);
}

struct DCT2x2Result {
  float r00;
  float r01;
  float r10;
  float r11;
};

// Keep these expressions left-associative. Do not rewrite using pairwise partial
// sums: that changes float rounding and breaks byte-exact output.
HWY_INLINE DCT2x2Result ComputeDCT2x2(const float c00, const float c01,
                                       const float c10, const float c11) {
  DCT2x2Result result;
  result.r00 = (c00 + c01 + c10 + c11) * 0.25f;
  result.r01 = (c00 + c01 - c10 - c11) * 0.25f;
  result.r10 = (c00 - c01 + c10 - c11) * 0.25f;
  result.r11 = (c00 - c01 - c10 + c11) * 0.25f;
  return result;
}

// Fused 8x8 DCT2 multilevel kernel. Writes high-frequency bands directly and
// retains only the four values needed for each subsequent level, eliminating
// the three-pass materialisation of 64+16+4 temporary coefficients.
HWY_INLINE void DCT2Transform8x8(const float* input, const size_t stride,
                                  float* output) {
  static_assert(kBlockDim == 8, "This is the fixed 8x8 DCT2 layout");
  constexpr size_t kHalf = kBlockDim / 2;    // 4
  constexpr size_t kQuarter = kHalf / 2;     // 2

  float level2[kQuarter * kQuarter];

  for (size_t y2 = 0; y2 < kQuarter; ++y2) {
    for (size_t x2 = 0; x2 < kQuarter; ++x2) {
      float level1[4];

      for (size_t dy = 0; dy < 2; ++dy) {
        for (size_t dx = 0; dx < 2; ++dx) {
          const size_t block_y = 2 * y2 + dy;
          const size_t block_x = 2 * x2 + dx;
          const size_t src = 2 * block_y * stride + 2 * block_x;

          const DCT2x2Result stage1 =
              ComputeDCT2x2(input[src], input[src + 1],
                            input[src + stride], input[src + stride + 1]);

          level1[dy * 2 + dx] = stage1.r00;

          // Stage-1 LH / HL / HH written directly to output.
          output[block_y * kBlockDim + kHalf + block_x] = stage1.r01;
          output[(kHalf + block_y) * kBlockDim + block_x] = stage1.r10;
          output[(kHalf + block_y) * kBlockDim + kHalf + block_x] = stage1.r11;
        }
      }

      const DCT2x2Result stage2 =
          ComputeDCT2x2(level1[0], level1[1], level1[2], level1[3]);

      level2[y2 * kQuarter + x2] = stage2.r00;

      // Stage-2 LH / HL / HH inside the top-left 4x4 region.
      output[y2 * kBlockDim + kQuarter + x2] = stage2.r01;
      output[(kQuarter + y2) * kBlockDim + x2] = stage2.r10;
      output[(kQuarter + y2) * kBlockDim + kQuarter + x2] = stage2.r11;
    }
  }

  const DCT2x2Result stage3 =
      ComputeDCT2x2(level2[0], level2[1], level2[2], level2[3]);

  output[0] = stage3.r00;
  output[1] = stage3.r01;
  output[kBlockDim] = stage3.r10;
  output[kBlockDim + 1] = stage3.r11;
}

HWY_ALIGN static constexpr float k4x4AFVBasisTranspose[16][16] = {
      {
          0.2500000000000000,
          0.8769029297991420f,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          -0.4105377591765233f,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
      },
      {
          0.2500000000000000,
          0.2206518106944235f,
          0.0000000000000000,
          0.0000000000000000,
          -0.7071067811865474f,
          0.6235485373547691f,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
      },
      {
          0.2500000000000000,
          -0.1014005039375376f,
          0.4067007583026075f,
          -0.2125574805828875f,
          0.0000000000000000,
          -0.0643507165794627f,
          -0.4517556589999482f,
          -0.3046847507248690f,
          0.3017929516615495f,
          0.4082482904638627f,
          0.1747866975480809f,
          -0.2110560104933578f,
          -0.1426608480880726f,
          -0.1381354035075859f,
          -0.1743760259965107f,
          0.1135498731499434f,
      },
      {
          0.2500000000000000,
          -0.1014005039375375f,
          0.4444481661973445f,
          0.3085497062849767f,
          0.0000000000000000f,
          -0.0643507165794627f,
          0.1585450355184006f,
          0.5112616136591823f,
          0.2579236279634118f,
          0.0000000000000000,
          0.0812611176717539f,
          0.1856718091610980f,
          -0.3416446842253372f,
          0.3302282550303788f,
          0.0702790691196284f,
          -0.0741750459581035f,
      },
      {
          0.2500000000000000,
          0.2206518106944236f,
          0.0000000000000000,
          0.0000000000000000,
          0.7071067811865476f,
          0.6235485373547694f,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
      },
      {
          0.2500000000000000,
          -0.1014005039375378f,
          0.0000000000000000,
          0.4706702258572536f,
          0.0000000000000000,
          -0.0643507165794628f,
          -0.0403851516082220f,
          0.0000000000000000,
          0.1627234014286620f,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.7367497537172237f,
          0.0875511500058708f,
          -0.2921026642334881f,
          0.1940289303259434f,
      },
      {
          0.2500000000000000,
          -0.1014005039375377f,
          0.1957439937204294f,
          -0.1621205195722993f,
          0.0000000000000000,
          -0.0643507165794628f,
          0.0074182263792424f,
          -0.2904801297289980f,
          0.0952002265347504f,
          0.0000000000000000,
          -0.3675398009862027f,
          0.4921585901373873f,
          0.2462710772207515f,
          -0.0794670660590957f,
          0.3623817333531167f,
          -0.4351904965232280f,
      },
      {
          0.2500000000000000,
          -0.1014005039375376f,
          0.2929100136981264f,
          0.0000000000000000,
          0.0000000000000000,
          -0.0643507165794627f,
          0.3935103426921017f,
          -0.0657870154914280f,
          0.0000000000000000,
          -0.4082482904638628f,
          -0.3078822139579090f,
          -0.3852501370925192f,
          -0.0857401903551931f,
          -0.4613374887461511f,
          0.0000000000000000,
          0.2191868483885747f,
      },
      {
          0.2500000000000000,
          -0.1014005039375376f,
          -0.4067007583026072f,
          -0.2125574805828705f,
          0.0000000000000000,
          -0.0643507165794627f,
          -0.4517556589999464f,
          0.3046847507248840f,
          0.3017929516615503f,
          -0.4082482904638635f,
          -0.1747866975480813f,
          0.2110560104933581f,
          -0.1426608480880734f,
          -0.1381354035075829f,
          -0.1743760259965108f,
          0.1135498731499426f,
      },
      {
          0.2500000000000000,
          -0.1014005039375377f,
          -0.1957439937204287f,
          -0.1621205195722833f,
          0.0000000000000000,
          -0.0643507165794628f,
          0.0074182263792444f,
          0.2904801297290076f,
          0.0952002265347505f,
          0.0000000000000000,
          0.3675398009862011f,
          -0.4921585901373891f,
          0.2462710772207514f,
          -0.0794670660591026f,
          0.3623817333531165f,
          -0.4351904965232251f,
      },
      {
          0.2500000000000000,
          -0.1014005039375375f,
          0.0000000000000000,
          -0.4706702258572528f,
          0.0000000000000000,
          -0.0643507165794627f,
          0.1107416575309343f,
          0.0000000000000000,
          -0.1627234014286617f,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.1488339922711357f,
          0.4972464710953509f,
          0.2921026642334879f,
          0.5550443808910661f,
      },
      {
          0.2500000000000000,
          -0.1014005039375377f,
          0.1137907446044809f,
          -0.1464291867126764f,
          0.0000000000000000,
          -0.0643507165794628f,
          0.0829816309488205f,
          -0.2388977352334460f,
          -0.3531238544981630f,
          -0.4082482904638630f,
          0.4826689115059883f,
          0.1741941265991622f,
          -0.0476868035022925f,
          0.1253805944856366f,
          -0.4326608024727445f,
          -0.2546827712406646f,
      },
      {
          0.2500000000000000,
          -0.1014005039375377f,
          -0.4444481661973438f,
          0.3085497062849487f,
          0.0000000000000000,
          -0.0643507165794628f,
          0.1585450355183970f,
          -0.5112616136592012f,
          0.2579236279634129f,
          0.0000000000000000,
          -0.0812611176717504f,
          -0.1856718091610990f,
          -0.3416446842253373f,
          0.3302282550303805f,
          0.0702790691196282f,
          -0.0741750459581023f,
      },
      {
          0.2500000000000000,
          -0.1014005039375376f,
          -0.2929100136981264f,
          0.0000000000000000,
          0.0000000000000000,
          -0.0643507165794627f,
          0.3935103426921022f,
          0.0657870154914254f,
          0.0000000000000000,
          0.4082482904638634f,
          0.3078822139579031f,
          0.3852501370925211f,
          -0.0857401903551927f,
          -0.4613374887461554f,
          0.0000000000000000,
          0.2191868483885728f,
      },
      {
          0.2500000000000000,
          -0.1014005039375376f,
          -0.1137907446044814f,
          -0.1464291867126654f,
          0.0000000000000000,
          -0.0643507165794627f,
          0.0829816309488214f,
          0.2388977352334547f,
          -0.3531238544981624f,
          0.4082482904638630f,
          -0.4826689115059858f,
          -0.1741941265991621f,
          -0.0476868035022928f,
          0.1253805944856431f,
          -0.4326608024727457f,
          -0.2546827712406641f,
      },
      {
          0.2500000000000000,
          -0.1014005039375374f,
          0.0000000000000000,
          0.4251149611657548f,
          0.0000000000000000,
          -0.0643507165794626f,
          -0.4517556589999480f,
          0.0000000000000000,
          -0.6035859033230976f,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          -0.1426608480880724f,
          -0.1381354035075845f,
          0.3487520519930227f,
          0.1135498731499429f,
      },
  };

void AFVDCT4x4(const float* JXL_RESTRICT pixels, float* JXL_RESTRICT coeffs) {
  const HWY_CAPPED(float, 16) d;
  for (size_t i = 0; i < 16; i += Lanes(d)) {
    auto scalar = Zero(d);
    for (size_t j = 0; j < 16; j++) {
      auto px = Set(d, pixels[j]);
      auto basis = Load(d, k4x4AFVBasisTranspose[j] + i);
      scalar = MulAdd(px, basis, scalar);
    }
    Store(scalar, d, coeffs + i);
  }
}

// Variant of AFVDCT4x4 that reads the oriented 4x4 quadrant directly from the
// strided 8x8 pixel block, eliminating the intermediate block[] copy.
// afv_kind encodes which quadrant (bit 0 = flip-x, bit 1 = flip-y).
template <size_t afv_kind>
HWY_INLINE void AFVDCT4x4Strided(const float* JXL_RESTRICT pixels,
                                  const size_t pixels_stride,
                                  float* JXL_RESTRICT coeffs) {
  static_assert(afv_kind < 4, "Invalid AFV kind");

  constexpr bool kFlipX = (afv_kind & 1) != 0;
  constexpr bool kFlipY = (afv_kind & 2) != 0;

  const ptrdiff_t row_step =
      kFlipY ? -static_cast<ptrdiff_t>(pixels_stride)
             : static_cast<ptrdiff_t>(pixels_stride);
  const ptrdiff_t col_step = kFlipX ? -1 : 1;

  // When kFlipX, afv_x==1, so the quadrant starts at column 4 and is reversed:
  // first sampled column = 4+3=7. When !kFlipX, afv_x==0, first column = 0.
  // Same logic for rows via kFlipY.
  const float* const first =
      pixels + (kFlipY ? 7 * pixels_stride : 0) + (kFlipX ? 7 : 0);

  const HWY_CAPPED(float, 16) d;
  for (size_t i = 0; i < 16; i += Lanes(d)) {
    auto sum = Zero(d);
    size_t j = 0;
    for (size_t y = 0; y < 4; ++y) {
      const float* const row = first + static_cast<ptrdiff_t>(y) * row_step;
      for (size_t x = 0; x < 4; ++x, ++j) {
        const auto px = Set(d, row[static_cast<ptrdiff_t>(x) * col_step]);
        const auto basis = Load(d, k4x4AFVBasisTranspose[j] + i);
        sum = MulAdd(px, basis, sum);
      }
    }
    Store(sum, d, coeffs + i);
  }
}

// Coefficient layout:
//  - (even, even) positions hold AFV coefficients
//  - (odd, even) positions hold DCT4x4 coefficients
//  - (any, odd) positions hold DCT4x8 coefficients
template <size_t afv_kind>
void AFVTransformFromPixels(const float* JXL_RESTRICT pixels,
                            size_t pixels_stride,
                            float* JXL_RESTRICT coefficients) {
  HWY_ALIGN float scratch_space[4 * 8 * 5];
  size_t afv_x = afv_kind & 1;
  size_t afv_y = afv_kind / 2;
  // Read directly from the strided source: no intermediate block[] copy needed.
  HWY_ALIGN float coeff[4 * 4];
  AFVDCT4x4Strided<afv_kind>(pixels, pixels_stride, coeff);
  // AFV coefficients in (even, even) positions.
  for (size_t iy = 0; iy < 4; iy++) {
    for (size_t ix = 0; ix < 4; ix++) {
      coefficients[iy * 2 * 8 + ix * 2] = coeff[iy * 4 + ix];
    }
  }
  // 4x4 DCT of the block with same y and different x.
  HWY_ALIGN float block[4 * 8];
  ComputeScaledDCT<4, 4>()(
      DCTFrom(pixels + afv_y * 4 * pixels_stride + (afv_x == 1 ? 0 : 4),
              pixels_stride),
      block, scratch_space);
  // ... in (odd, even) positions. ix >= 4 writes positions that the 4x8 DCT
  // below always overwrites before any caller can observe them.
  for (size_t iy = 0; iy < 4; iy++) {
    for (size_t ix = 0; ix < 4; ix++) {
      coefficients[iy * 2 * 8 + ix * 2 + 1] = block[iy * 4 + ix];
    }
  }
  // 4x8 DCT of the other half of the block.
  ComputeScaledDCT<4, 8>()(
      DCTFrom(pixels + (afv_y == 1 ? 0 : 4) * pixels_stride, pixels_stride),
      block, scratch_space);
  for (size_t iy = 0; iy < 4; iy++) {
    for (size_t ix = 0; ix < 8; ix++) {
      coefficients[(1 + iy * 2) * 8 + ix] = block[iy * 8 + ix];
    }
  }
  float block00 = coefficients[0] * 0.25f;
  float block01 = coefficients[1];
  float block10 = coefficients[8];
  coefficients[0] = (block00 + block01 + 2 * block10) * 0.25f;
  coefficients[1] = (block00 - block01) * 0.5f;
  coefficients[8] = (block00 + block01 - 2 * block10) * 0.25f;
}

HWY_MAYBE_UNUSED void TransformFromPixels(const AcStrategyType strategy,
                                          const float* JXL_RESTRICT pixels,
                                          size_t pixels_stride,
                                          float* JXL_RESTRICT coefficients,
                                          float* JXL_RESTRICT scratch_space) {
  using Type = AcStrategyType;
  switch (strategy) {
    case Type::IDENTITY: {
      for (size_t y = 0; y < 2; y++) {
        for (size_t x = 0; x < 2; x++) {
          float block_dc = 0;
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              block_dc += pixels[(y * 4 + iy) * pixels_stride + x * 4 + ix];
            }
          }
          block_dc *= 1.0f / 16;
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              if (ix == 1 && iy == 1) continue;
              coefficients[(y + iy * 2) * 8 + x + ix * 2] =
                  pixels[(y * 4 + iy) * pixels_stride + x * 4 + ix] -
                  pixels[(y * 4 + 1) * pixels_stride + x * 4 + 1];
            }
          }
          coefficients[(y + 2) * 8 + x + 2] = coefficients[y * 8 + x];
          coefficients[y * 8 + x] = block_dc;
        }
      }
      float block00 = coefficients[0];
      float block01 = coefficients[1];
      float block10 = coefficients[8];
      float block11 = coefficients[9];
      coefficients[0] = (block00 + block01 + block10 + block11) * 0.25f;
      coefficients[1] = (block00 + block01 - block10 - block11) * 0.25f;
      coefficients[8] = (block00 - block01 + block10 - block11) * 0.25f;
      coefficients[9] = (block00 - block01 - block10 + block11) * 0.25f;
      break;
    }
    case Type::DCT8X4: {
      for (size_t x = 0; x < 2; x++) {
        HWY_ALIGN float block[4 * 8];
        ComputeScaledDCT<8, 4>()(DCTFrom(pixels + x * 4, pixels_stride), block,
                                 CoeffRowSinkTo(coefficients, x, 2),
                                 scratch_space);
      }
      float block0 = coefficients[0];
      float block1 = coefficients[8];
      coefficients[0] = (block0 + block1) * 0.5f;
      coefficients[8] = (block0 - block1) * 0.5f;
      break;
    }
    case Type::DCT4X8: {
      for (size_t y = 0; y < 2; y++) {
        HWY_ALIGN float block[4 * 8];
        ComputeScaledDCT<4, 8>()(
            DCTFrom(pixels + y * 4 * pixels_stride, pixels_stride), block,
            CoeffRowSinkTo(coefficients, y, 2), scratch_space);
      }
      float block0 = coefficients[0];
      float block1 = coefficients[8];
      coefficients[0] = (block0 + block1) * 0.5f;
      coefficients[8] = (block0 - block1) * 0.5f;
      break;
    }
    case Type::DCT4X4: {
      for (size_t y = 0; y < 2; y++) {
        for (size_t x = 0; x < 2; x++) {
          HWY_ALIGN float block[4 * 4];
          ComputeScaledDCT<4, 4>()(
              DCTFrom(pixels + y * 4 * pixels_stride + x * 4, pixels_stride),
              block, scratch_space);
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              coefficients[(y + iy * 2) * 8 + x + ix * 2] = block[iy * 4 + ix];
            }
          }
        }
      }
      float block00 = coefficients[0];
      float block01 = coefficients[1];
      float block10 = coefficients[8];
      float block11 = coefficients[9];
      coefficients[0] = (block00 + block01 + block10 + block11) * 0.25f;
      coefficients[1] = (block00 + block01 - block10 - block11) * 0.25f;
      coefficients[8] = (block00 - block01 + block10 - block11) * 0.25f;
      coefficients[9] = (block00 - block01 - block10 + block11) * 0.25f;
      break;
    }
    case Type::DCT2X2: {
      DCT2Transform8x8(pixels, pixels_stride, coefficients);
      break;
    }
    case Type::DCT16X16: {
      ComputeScaledDCT<16, 16>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT16X8: {
      ComputeScaledDCT<16, 8>()(DCTFrom(pixels, pixels_stride), coefficients,
                                scratch_space);
      break;
    }
    case Type::DCT8X16: {
      ComputeScaledDCT<8, 16>()(DCTFrom(pixels, pixels_stride), coefficients,
                                scratch_space);
      break;
    }
    case Type::DCT32X8: {
      ComputeScaledDCT<32, 8>()(DCTFrom(pixels, pixels_stride), coefficients,
                                scratch_space);
      break;
    }
    case Type::DCT8X32: {
      ComputeScaledDCT<8, 32>()(DCTFrom(pixels, pixels_stride), coefficients,
                                scratch_space);
      break;
    }
    case Type::DCT32X16: {
      ComputeScaledDCT<32, 16>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT16X32: {
      ComputeScaledDCT<16, 32>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT32X32: {
      ComputeScaledDCT<32, 32>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT: {
      ComputeScaledDCT<8, 8>()(DCTFrom(pixels, pixels_stride), coefficients,
                               scratch_space);
      break;
    }
    case Type::AFV0: {
      AFVTransformFromPixels<0>(pixels, pixels_stride, coefficients);
      break;
    }
    case Type::AFV1: {
      AFVTransformFromPixels<1>(pixels, pixels_stride, coefficients);
      break;
    }
    case Type::AFV2: {
      AFVTransformFromPixels<2>(pixels, pixels_stride, coefficients);
      break;
    }
    case Type::AFV3: {
      AFVTransformFromPixels<3>(pixels, pixels_stride, coefficients);
      break;
    }
    case Type::DCT64X64: {
      ComputeScaledDCT<64, 64>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT64X32: {
      ComputeScaledDCT<64, 32>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT32X64: {
      ComputeScaledDCT<32, 64>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT128X128: {
      ComputeScaledDCT<128, 128>()(DCTFrom(pixels, pixels_stride), coefficients,
                                   scratch_space);
      break;
    }
    case Type::DCT128X64: {
      ComputeScaledDCT<128, 64>()(DCTFrom(pixels, pixels_stride), coefficients,
                                  scratch_space);
      break;
    }
    case Type::DCT64X128: {
      ComputeScaledDCT<64, 128>()(DCTFrom(pixels, pixels_stride), coefficients,
                                  scratch_space);
      break;
    }
    case Type::DCT256X256: {
      ComputeScaledDCT<256, 256>()(DCTFrom(pixels, pixels_stride), coefficients,
                                   scratch_space);
      break;
    }
    case Type::DCT256X128: {
      ComputeScaledDCT<256, 128>()(DCTFrom(pixels, pixels_stride), coefficients,
                                   scratch_space);
      break;
    }
    case Type::DCT128X256: {
      ComputeScaledDCT<128, 256>()(DCTFrom(pixels, pixels_stride), coefficients,
                                   scratch_space);
      break;
    }
  }
}

// `scratch_space` should be at least 4 * kMaxBlocks * kMaxBlocks elements.
HWY_MAYBE_UNUSED void DCFromLowestFrequencies(const AcStrategyType strategy,
                                              const float* block, float* dc,
                                              size_t dc_stride,
                                              float* scratch_space) {
  using Type = AcStrategyType;
  switch (strategy) {
    case Type::DCT16X8: {
      ReinterpretingIDCT</*DCT_ROWS=*/2 * kBlockDim, /*DCT_COLS=*/kBlockDim,
                         /*LF_ROWS=*/2, /*LF_COLS=*/1, /*ROWS=*/2, /*COLS=*/1>(
          block, 2 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT8X16: {
      ReinterpretingIDCT</*DCT_ROWS=*/kBlockDim, /*DCT_COLS=*/2 * kBlockDim,
                         /*LF_ROWS=*/1, /*LF_COLS=*/2, /*ROWS=*/1, /*COLS=*/2>(
          block, 2 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT16X16: {
      ReinterpretingIDCT</*DCT_ROWS=*/2 * kBlockDim, /*DCT_COLS=*/2 * kBlockDim,
                         /*LF_ROWS=*/2, /*LF_COLS=*/2, /*ROWS=*/2, /*COLS=*/2>(
          block, 2 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT32X8: {
      ReinterpretingIDCT</*DCT_ROWS=*/4 * kBlockDim, /*DCT_COLS=*/kBlockDim,
                         /*LF_ROWS=*/4, /*LF_COLS=*/1, /*ROWS=*/4, /*COLS=*/1>(
          block, 4 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT8X32: {
      ReinterpretingIDCT</*DCT_ROWS=*/kBlockDim, /*DCT_COLS=*/4 * kBlockDim,
                         /*LF_ROWS=*/1, /*LF_COLS=*/4, /*ROWS=*/1, /*COLS=*/4>(
          block, 4 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT32X16: {
      ReinterpretingIDCT</*DCT_ROWS=*/4 * kBlockDim, /*DCT_COLS=*/2 * kBlockDim,
                         /*LF_ROWS=*/4, /*LF_COLS=*/2, /*ROWS=*/4, /*COLS=*/2>(
          block, 4 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT16X32: {
      ReinterpretingIDCT</*DCT_ROWS=*/2 * kBlockDim, /*DCT_COLS=*/4 * kBlockDim,
                         /*LF_ROWS=*/2, /*LF_COLS=*/4, /*ROWS=*/2, /*COLS=*/4>(
          block, 4 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT32X32: {
      ReinterpretingIDCT</*DCT_ROWS=*/4 * kBlockDim, /*DCT_COLS=*/4 * kBlockDim,
                         /*LF_ROWS=*/4, /*LF_COLS=*/4, /*ROWS=*/4, /*COLS=*/4>(
          block, 4 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT64X32: {
      ReinterpretingIDCT</*DCT_ROWS=*/8 * kBlockDim, /*DCT_COLS=*/4 * kBlockDim,
                         /*LF_ROWS=*/8, /*LF_COLS=*/4, /*ROWS=*/8, /*COLS=*/4>(
          block, 8 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT32X64: {
      ReinterpretingIDCT</*DCT_ROWS=*/4 * kBlockDim, /*DCT_COLS=*/8 * kBlockDim,
                         /*LF_ROWS=*/4, /*LF_COLS=*/8, /*ROWS=*/4, /*COLS=*/8>(
          block, 8 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT64X64: {
      ReinterpretingIDCT</*DCT_ROWS=*/8 * kBlockDim, /*DCT_COLS=*/8 * kBlockDim,
                         /*LF_ROWS=*/8, /*LF_COLS=*/8, /*ROWS=*/8, /*COLS=*/8>(
          block, 8 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT128X64: {
      ReinterpretingIDCT<
          /*DCT_ROWS=*/16 * kBlockDim, /*DCT_COLS=*/8 * kBlockDim,
          /*LF_ROWS=*/16, /*LF_COLS=*/8, /*ROWS=*/16, /*COLS=*/8>(
          block, 16 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT64X128: {
      ReinterpretingIDCT<
          /*DCT_ROWS=*/8 * kBlockDim, /*DCT_COLS=*/16 * kBlockDim,
          /*LF_ROWS=*/8, /*LF_COLS=*/16, /*ROWS=*/8, /*COLS=*/16>(
          block, 16 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT128X128: {
      ReinterpretingIDCT<
          /*DCT_ROWS=*/16 * kBlockDim, /*DCT_COLS=*/16 * kBlockDim,
          /*LF_ROWS=*/16, /*LF_COLS=*/16, /*ROWS=*/16, /*COLS=*/16>(
          block, 16 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT256X128: {
      ReinterpretingIDCT<
          /*DCT_ROWS=*/32 * kBlockDim, /*DCT_COLS=*/16 * kBlockDim,
          /*LF_ROWS=*/32, /*LF_COLS=*/16, /*ROWS=*/32, /*COLS=*/16>(
          block, 32 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT128X256: {
      ReinterpretingIDCT<
          /*DCT_ROWS=*/16 * kBlockDim, /*DCT_COLS=*/32 * kBlockDim,
          /*LF_ROWS=*/16, /*LF_COLS=*/32, /*ROWS=*/16, /*COLS=*/32>(
          block, 32 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT256X256: {
      ReinterpretingIDCT<
          /*DCT_ROWS=*/32 * kBlockDim, /*DCT_COLS=*/32 * kBlockDim,
          /*LF_ROWS=*/32, /*LF_COLS=*/32, /*ROWS=*/32, /*COLS=*/32>(
          block, 32 * kBlockDim, dc, dc_stride, scratch_space);
      break;
    }
    case Type::DCT:
    case Type::DCT2X2:
    case Type::DCT4X4:
    case Type::DCT4X8:
    case Type::DCT8X4:
    case Type::AFV0:
    case Type::AFV1:
    case Type::AFV2:
    case Type::AFV3:
    case Type::IDENTITY:
      dc[0] = block[0];
      break;
  }
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // LIB_JXL_ENC_TRANSFORMS_INL_H_
