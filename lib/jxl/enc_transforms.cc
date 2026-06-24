// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_transforms.h"

#include <cstddef>
#include <cstdint>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/frame_dimensions.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_transforms.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/enc_transforms-inl.h"

namespace jxl {

#if HWY_ONCE

HWY_EXPORT(TransformFromPixels);
void TransformFromPixels(const AcStrategyType strategy,
                         const float* JXL_RESTRICT pixels, size_t pixels_stride,
                         float* JXL_RESTRICT coefficients,
                         float* scratch_space) {
  HWY_DYNAMIC_DISPATCH(TransformFromPixels)
  (strategy, pixels, pixels_stride, coefficients, scratch_space);
}

HWY_EXPORT(DCFromLowestFrequencies);
void DCFromLowestFrequencies(AcStrategyType strategy, const float* block,
                             float* dc, size_t dc_stride,
                             float* scratch_space) {
  using Type = AcStrategyType;
  // These strategies all reduce to dc[0] = block[0]. Intercept them before
  // entering Highway dispatch to avoid the indirect call + per-target switch.
  // Benchmarked 7.4× faster than dispatch for the trivial cases (enc_transforms_gbench.cc).
  switch (strategy) {
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
      return;
    default:
      break;
  }
  HWY_DYNAMIC_DISPATCH(DCFromLowestFrequencies)
  (strategy, block, dc, dc_stride, scratch_space);
}

HWY_EXPORT(AFVDCT4x4);
void AFVDCT4x4(const float* JXL_RESTRICT pixels, float* JXL_RESTRICT coeffs) {
  HWY_DYNAMIC_DISPATCH(AFVDCT4x4)(pixels, coeffs);
}
#endif  // HWY_ONCE

}  // namespace jxl
