// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_GROUP_H_
#define LIB_JXL_ENC_GROUP_H_

#include <cstddef>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/image.h"
#include "lib/jxl/memory_manager_internal.h"

namespace jxl {

struct AuxOut;
struct PassesEncoderState;

// Per-worker reusable scratch for ComputeCoefficients. Both buffers are sized
// by transform-independent constants (functions of AcStrategy::kMaxCoeffArea
// and the SIMD width only), so a single allocation per worker thread is reused
// across every group and every frame instead of two aligned allocations per
// group. Lazily allocated on first use; a default-constructed (empty) instance
// is valid input.
struct ACGroupScratch {
  AlignedMemory imem;  // 3 * kMaxCoeffArea int32 (quantized planes)
  AlignedMemory fmem;  // (5 * kMaxCoeffArea + dct_scratch) float (coeffs+scratch)
};

// Fills DC
Status ComputeCoefficients(size_t group_idx, PassesEncoderState* enc_state,
                           const Image3F& opsin, const Rect& rect, Image3F* dc,
                           ACGroupScratch* scratch);

Status EncodeGroupTokenizedCoefficients(size_t group_idx, size_t pass_idx,
                                        size_t histogram_idx,
                                        const PassesEncoderState& enc_state,
                                        BitWriter* writer, AuxOut* aux_out);

}  // namespace jxl

#endif  // LIB_JXL_ENC_GROUP_H_
