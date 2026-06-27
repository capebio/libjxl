// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_gaborish.h"

#include <jxl/memory_manager.h>

#include <hwy/base.h>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/convolve.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"

namespace jxl {

Status GaborishInverse(Image3F* in_out, const Rect& rect, const float mul[3],
                       ThreadPool* pool) {
  JxlMemoryManager* memory_manager = in_out->memory_manager();
  WeightsSymmetric5 weights[3];
  // Only an approximation. One or even two 3x3, and rank-1 (separable) 5x5
  // are insufficient. The numbers here have been obtained by butteraugli
  // based optimizing the whole system and the errors produced are likely
  // more favorable for good rate-distortion compromises rather than
  // just using mathematical optimization to find the inverse.
  static const float kGaborish[5] = {
      -0.09495815671340026, -0.041031725066768575,  0.013710004822696948,
      0.006510206083837737, -0.0014789063378272242,
  };
  for (int i = 0; i < 3; ++i) {
    double sum = 1.0 + mul[i] * 4 *
                           (kGaborish[0] + kGaborish[1] + kGaborish[2] +
                            kGaborish[4] + 2 * kGaborish[3]);
    if (sum < 1e-5) {
      sum = 1e-5;
    }
    const float normalize = static_cast<float>(1.0 / sum);
    const float normalize_mul = mul[i] * normalize;
    weights[i] = WeightsSymmetric5{{HWY_REP4(normalize)},
                                   {HWY_REP4(normalize_mul * kGaborish[0])},
                                   {HWY_REP4(normalize_mul * kGaborish[2])},
                                   {HWY_REP4(normalize_mul * kGaborish[1])},
                                   {HWY_REP4(normalize_mul * kGaborish[4])},
                                   {HWY_REP4(normalize_mul * kGaborish[3])}};
  }
  // Reduce memory footprint by only allocating a single scratch plane and
  // rotating it into the output Image3F. We cannot *allocate* a full plane in
  // the Image3F (doing so might give planes different strides), so we filter
  // into a separate temporary image and reuse the existing planes in place.
  //
  // Only the convolution rectangle xrect is touched, so the scratch image is
  // sized to xrect rather than the whole plane: for a partial/streaming rect
  // this drops the scratch allocation and the back-copy from a full plane to a
  // tile, while a full-frame rect retains effectively the same cost. The three
  // Symmetric5 evaluations, their source coordinates, weights and output pixels
  // are byte-for-byte identical to the full-plane-scratch version, including
  // pixels outside xrect after the plane rotation.
  const Rect xrect = rect.Extend(3, Rect(*in_out));
  ImageF temp;
  JXL_ASSIGN_OR_RETURN(
      temp, ImageF::Create(memory_manager, xrect.xsize(), xrect.ysize()));
  const Rect temp_rect(temp);
  // Filter channel 0 into the scratch tile first, preserving it before plane 0
  // is overwritten by channel 1's result.
  JXL_RETURN_IF_ERROR(Symmetric5(in_out->Plane(0), xrect, weights[0], pool,
                                 &temp, temp_rect));
  JXL_RETURN_IF_ERROR(Symmetric5(in_out->Plane(1), xrect, weights[1], pool,
                                 &in_out->Plane(0), xrect));
  JXL_RETURN_IF_ERROR(Symmetric5(in_out->Plane(2), xrect, weights[2], pool,
                                 &in_out->Plane(1), xrect));
  // Copy filtered channel 0 from the tile back into plane 2.
  JXL_RETURN_IF_ERROR(CopyImageTo(temp_rect, temp, xrect, &in_out->Plane(2)));
  // Planes now hold channels 1, 2, 0; rotate them back to 0, 1, 2.
  in_out->Plane(0).Swap(in_out->Plane(2));
  in_out->Plane(1).Swap(in_out->Plane(2));
  return true;
}

}  // namespace jxl
