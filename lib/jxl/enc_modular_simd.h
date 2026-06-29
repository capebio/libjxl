// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_MODULAR_SIMD_H_
#define LIB_JXL_ENC_MODULAR_SIMD_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/modular_image.h"

namespace jxl {

// Reusable scratch for EstimateCost. A per-group RCT search calls EstimateCost
// up to ~19 times on the same image dimensions; without reuse each call freshly
// allocates the aligned SIMD row buffer and the per-context histogram backings.
// Threading one workspace through that loop keeps the scratch alive. Byte-exact:
// the row buffer is fully overwritten before each read and the histograms are
// cleared before use, so a reused workspace yields identical costs to fresh ones.
struct EstimateCostWorkspace {
  AlignedMemory buffer;          // SIMD row scratch (max_diff_row | token_row)
  size_t buffer_bytes = 0;       // byte capacity currently backing `buffer`
  std::vector<Histogram> histo;  // per-context histograms (grow-and-retain)
};

StatusOr<float> EstimateCost(const Image& img);
StatusOr<float> EstimateCost(const Image& img, EstimateCostWorkspace& workspace);

namespace estimate_cost_detail {
constexpr size_t kLastThreshold = 501;
constexpr size_t kLastCtx = 16;
const std::array<uint8_t, kLastThreshold>& ContextMap();
}  // namespace estimate_cost_detail

}  // namespace jxl

#endif  // LIB_JXL_ENC_MODULAR_SIMD_H_
