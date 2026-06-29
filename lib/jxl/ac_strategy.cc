// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/ac_strategy.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"

namespace jxl {

// Tries to generalize zig-zag order to non-square blocks. Surprisingly, in
// square block frequency along the (i + j == const) diagonals is roughly the
// same. For historical reasons, consecutive diagonals are traversed
// in alternating directions - so called "zig-zag" (or "snake") order.
template <bool is_lut>
static void CoeffOrderAndLut(AcStrategy acs, coeff_order_t* out) {
  size_t cx = acs.covered_blocks_x();
  size_t cy = acs.covered_blocks_y();
  CoefficientLayout(&cy, &cx);

  // CoefficientLayout ensures cx >= cy, with xs = cx / cy a power of two.
  // The zig-zag is conceptually computed for a cx x cx block, keeping only the
  // diagonal positions whose (post-swap) y is a multiple of xs. Instead of
  // visiting every position of the enclosing square and discarding (xs-1)/xs of
  // them, we step the inner index directly by xs from the correct residue, so
  // only retained positions are visited. This emits the IDENTICAL permutation
  // (byte-exact) while cutting the inner loop body by a factor of xs (2x for
  // 2:1 layouts, 4x for 4:1). n = cx * kBlockDim is a multiple of xs, which is
  // used to derive the second-half residues.
  const size_t xs = cx / cy;
  const size_t xsm = xs - 1;
  const size_t xss = CeilLog2Nonzero(xs);
  const size_t n = cx * kBlockDim;
  size_t cur = cx * cy;
  // First half of the block.
  for (size_t i = 0; i < n; i++) {
    if (i & 1) {
      // After swap: x = i - j, y = j; y % xs == 0  =>  j stepped by xs from 0.
      for (size_t j = 0; j <= i; j += xs) {
        size_t x = i - j;
        size_t y = j >> xss;
        size_t val = (x < cx && y < cy) ? y * cx + x : cur++;
        if (is_lut) {
          out[y * n + x] = val;
        } else {
          out[val] = y * n + x;
        }
      }
    } else {
      // No swap: x = j, y = i - j; (i - j) % xs == 0  =>  j == i (mod xs).
      for (size_t j = i & xsm; j <= i; j += xs) {
        size_t x = j;
        size_t y = (i - j) >> xss;
        size_t val = (x < cx && y < cy) ? y * cx + x : cur++;
        if (is_lut) {
          out[y * n + x] = val;
        } else {
          out[val] = y * n + x;
        }
      }
    }
  }
  // Second half. n % xs == 0, so n - 1 == xsm (mod xs).
  for (size_t ip = n - 1; ip > 0; ip--) {
    size_t i = ip - 1;
    if (i & 1) {
      // After swap: x = n-1-j, y = n-1-i+j; y % xs == 0  =>  j == i+1 (mod xs).
      for (size_t j = (i + 1) & xsm; j <= i; j += xs) {
        size_t x = n - 1 - j;
        size_t y = (n - 1 - i + j) >> xss;
        size_t val = cur++;
        if (is_lut) {
          out[y * n + x] = val;
        } else {
          out[val] = y * n + x;
        }
      }
    } else {
      // No swap: x = n-1-i+j, y = n-1-j; y % xs == 0  =>  j == xsm (mod xs).
      for (size_t j = xsm; j <= i; j += xs) {
        size_t x = n - 1 - i + j;
        size_t y = (n - 1 - j) >> xss;
        size_t val = cur++;
        if (is_lut) {
          out[y * n + x] = val;
        } else {
          out[val] = y * n + x;
        }
      }
    }
  }
}

void AcStrategy::ComputeNaturalCoeffOrder(coeff_order_t* order) const {
  CoeffOrderAndLut</*is_lut=*/false>(*this, order);
}
void AcStrategy::ComputeNaturalCoeffOrderLut(coeff_order_t* lut) const {
  CoeffOrderAndLut</*is_lut=*/true>(*this, lut);
}

#if JXL_CXX_LANG < JXL_CXX_17
constexpr size_t AcStrategy::kMaxCoeffBlocks;
constexpr size_t AcStrategy::kMaxBlockDim;
constexpr size_t AcStrategy::kMaxCoeffArea;
#endif

StatusOr<AcStrategyImage> AcStrategyImage::Create(
    JxlMemoryManager* memory_manager, size_t xsize, size_t ysize) {
  AcStrategyImage img;
  JXL_ASSIGN_OR_RETURN(img.layers_,
                       ImageB::Create(memory_manager, xsize, ysize));
  img.row_ = img.layers_.Row(0);
  img.stride_ = img.layers_.PixelsPerRow();
  return img;
}

size_t AcStrategyImage::CountBlocks(AcStrategyType type) const {
  size_t ret = 0;
  for (size_t y = 0; y < layers_.ysize(); y++) {
    const uint8_t* JXL_RESTRICT row = layers_.ConstRow(y);
    for (size_t x = 0; x < layers_.xsize(); x++) {
      if (row[x] == ((static_cast<uint8_t>(type) << 1) | 1)) ret++;
    }
  }
  return ret;
}

}  // namespace jxl
