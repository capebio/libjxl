// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_group_border.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/frame_dimensions.h"

namespace jxl {

void GroupBorderAssigner::Init(const FrameDimensions& frame_dim) {
  frame_dim_ = frame_dim;
  size_t x_size = frame_dim_.xsize_groups;
  size_t y_size = frame_dim_.ysize_groups;
  // Flat 1D allocation: row y, word w → counters_[y * counters_stride_ + w].
  // Eliminates one pointer dereference per GroupDone atomic op vs nested vectors.
  counters_stride_ = (x_size + 1 + 7) / 8;
  const size_t n = (y_size + 1) * counters_stride_;
  counters_ = std::make_unique<std::atomic<uint32_t>[]>(n);
  for (size_t i = 0; i < n; ++i) {
    counters_[i] = 0;
  }
  // Counters at image borders don't have anything on the other side, we
  // pre-fill their value to have more uniform handling afterwards.
  auto set = [this](size_t x, size_t y, uint32_t corners) {
    counters_[y * counters_stride_ + x / 8] |= corners << (4 * (x & 7u));
  };
  for (size_t x = 0; x < x_size + 1; x++) {
    set(x, 0, kTopLeft | kTopRight);
    set(x, y_size, kBottomLeft | kBottomRight);
  }
  for (size_t y = 0; y < y_size + 1; y++) {
    set(0, y, kTopLeft | kBottomLeft);
    set(x_size, y, kTopRight | kBottomRight);
  }
}

void GroupBorderAssigner::ClearDone(size_t group_id) {
  // Clear the two same-row corners (columns x and x+1) in one atomic RMW when
  // they share a counter word — true for 7 of every 8 column positions. The
  // corners occupy disjoint nibbles, so masking both bits in a single fetch_and
  // is byte-identical to two sequential fetch_and ops, while halving the locked
  // memory operations. On a word boundary (x & 7 == 7) fall back to two ops.
  auto clear_pair = [this](size_t x, size_t y, uint32_t bit_left,
                           uint32_t bit_right) {
    const size_t idx_l = y * counters_stride_ + x / 8;
    const size_t idx_r = y * counters_stride_ + (x + 1) / 8;
    const size_t shift_l = 4 * (x & 7u);
    const size_t shift_r = 4 * ((x + 1) & 7u);
    if (JXL_LIKELY(idx_l == idx_r)) {
      counters_[idx_l].fetch_and(
          ~((bit_left << shift_l) | (bit_right << shift_r)),
          std::memory_order_release);
    } else {
      counters_[idx_l].fetch_and(~(bit_left << shift_l),
                                 std::memory_order_release);
      counters_[idx_r].fetch_and(~(bit_right << shift_r),
                                 std::memory_order_release);
    }
  };
  size_t x = group_id % frame_dim_.xsize_groups;
  size_t y = group_id / frame_dim_.xsize_groups;
  clear_pair(x, y, kBottomRight, kBottomLeft);
  clear_pair(x, y + 1, kTopRight, kTopLeft);
}

// Looking at each corner between groups, we can guarantee that the four
// involved groups will agree between each other regarding the order in which
// each of the four groups terminated. Thus, the last of the four groups
// gets the responsibility of handling the corner. For borders, every border
// is assigned to its top corner (for vertical borders) or to its left corner
// (for horizontal borders): the order as seen on those corners will decide who
// handles that border.

void GroupBorderAssigner::GroupDone(size_t group_id, size_t padx, size_t pady,
                                    Rect* rects_to_finalize,
                                    size_t* num_to_finalize) {
  size_t x = group_id % frame_dim_.xsize_groups;
  size_t y = group_id / frame_dim_.xsize_groups;
  Rect block_rect(x * frame_dim_.group_dim / kBlockDim,
                  y * frame_dim_.group_dim / kBlockDim,
                  frame_dim_.group_dim / kBlockDim,
                  frame_dim_.group_dim / kBlockDim, frame_dim_.xsize_blocks,
                  frame_dim_.ysize_blocks);

  // Publish two same-row corners (columns x and x+1) with a single atomic RMW
  // when they share a counter word — true for 7 of every 8 column positions.
  // The corners occupy disjoint nibbles, so OR-ing both bits in one fetch_or
  // returns the same per-corner old value (hence the same status and the same
  // "last group handles the corner" decision) as two sequential fetch_or ops,
  // while halving the locked memory operations on this per-group hot path. The
  // acq-rel semantics (needed so the group's pixel writes are visible before
  // the corner is observed done) are preserved on the combined op.
  auto fetch_status_pair = [this](size_t x, size_t y, uint32_t bit_left,
                                  uint32_t bit_right, size_t* out_left,
                                  size_t* out_right) {
    const size_t idx_l = y * counters_stride_ + x / 8;
    const size_t idx_r = y * counters_stride_ + (x + 1) / 8;
    const size_t shift_l = 4 * (x & 7u);
    const size_t shift_r = 4 * ((x + 1) & 7u);
    size_t old_l;
    size_t old_r;
    if (JXL_LIKELY(idx_l == idx_r)) {
      const size_t old = counters_[idx_l].fetch_or(
          (bit_left << shift_l) | (bit_right << shift_r),
          std::memory_order_acq_rel);
      old_l = old;
      old_r = old;
    } else {
      // Column boundary (x & 7 == 7): corners live in different words.
      old_l = counters_[idx_l].fetch_or(bit_left << shift_l,
                                        std::memory_order_acq_rel);
      old_r = counters_[idx_r].fetch_or(bit_right << shift_r,
                                        std::memory_order_acq_rel);
    }
    const size_t status_l = (old_l >> shift_l) & 0xF;
    const size_t status_r = (old_r >> shift_r) & 0xF;
    JXL_DASSERT((bit_left & status_l) == 0);
    JXL_DASSERT((bit_right & status_r) == 0);
    *out_left = (bit_left | status_l) & 0xF;
    *out_right = (bit_right | status_r) & 0xF;
  };

  size_t top_left_status;
  size_t top_right_status;
  size_t bottom_left_status;
  size_t bottom_right_status;
  fetch_status_pair(x, y, kBottomRight, kBottomLeft, &top_left_status,
                    &top_right_status);
  fetch_status_pair(x, y + 1, kTopRight, kTopLeft, &bottom_left_status,
                    &bottom_right_status);

  size_t x1 = block_rect.x0() + block_rect.xsize();
  size_t y1 = block_rect.y0() + block_rect.ysize();

  bool is_last_group_x = frame_dim_.xsize_groups == x + 1;
  bool is_last_group_y = frame_dim_.ysize_groups == y + 1;

  // Start of border of neighbouring group, end of border of this group, start
  // of border of this group (on the other side), end of border of next group.
  size_t xpos[4] = {
      block_rect.x0() == 0 ? 0 : block_rect.x0() * kBlockDim - padx,
      block_rect.x0() == 0
          ? 0
          : std::min(frame_dim_.xsize, block_rect.x0() * kBlockDim + padx),
      is_last_group_x ? frame_dim_.xsize : x1 * kBlockDim - padx,
      std::min(frame_dim_.xsize, x1 * kBlockDim + padx)};
  size_t ypos[4] = {
      block_rect.y0() == 0 ? 0 : block_rect.y0() * kBlockDim - pady,
      block_rect.y0() == 0
          ? 0
          : std::min(frame_dim_.ysize, block_rect.y0() * kBlockDim + pady),
      is_last_group_y ? frame_dim_.ysize : y1 * kBlockDim - pady,
      std::min(frame_dim_.ysize, y1 * kBlockDim + pady)};

  *num_to_finalize = 0;
  auto append_rect = [&](size_t x0, size_t x1, size_t y0, size_t y1) {
    Rect rect(xpos[x0], ypos[y0], xpos[x1] - xpos[x0], ypos[y1] - ypos[y0]);
    if (rect.xsize() == 0 || rect.ysize() == 0) return;
    JXL_DASSERT(*num_to_finalize < kMaxToFinalize);
    rects_to_finalize[(*num_to_finalize)++] = rect;
  };

  // Because of how group borders are assigned, it is impossible that we need to
  // process the left and right side of some area but not the center area. Thus,
  // we compute the first/last part to process in every horizontal strip and
  // merge them together. We first collect a mask of what parts should be
  // processed.
  // We do this horizontally rather than vertically because horizontal borders
  // are larger.
  bool available_parts_mask[3][3] = {};  // [x][y]
  // Center
  available_parts_mask[1][1] = true;
  // Corners
  if (top_left_status == 0xF) available_parts_mask[0][0] = true;
  if (top_right_status == 0xF) available_parts_mask[2][0] = true;
  if (bottom_right_status == 0xF) available_parts_mask[2][2] = true;
  if (bottom_left_status == 0xF) available_parts_mask[0][2] = true;
  // Other borders
  if (top_left_status & kTopRight) available_parts_mask[1][0] = true;
  if (top_left_status & kBottomLeft) available_parts_mask[0][1] = true;
  if (top_right_status & kBottomRight) available_parts_mask[2][1] = true;
  if (bottom_left_status & kBottomRight) available_parts_mask[1][2] = true;

  // Collect horizontal ranges.
  constexpr size_t kNoSegment = 3;
  std::pair<size_t, size_t> horizontal_segments[3] = {{kNoSegment, kNoSegment},
                                                      {kNoSegment, kNoSegment},
                                                      {kNoSegment, kNoSegment}};
  for (size_t py = 0; py < 3; py++) {
    for (size_t px = 0; px < 3; px++) {
      if (!available_parts_mask[px][py]) continue;
      JXL_DASSERT(horizontal_segments[py].second == kNoSegment ||
                  horizontal_segments[py].second == px);
      JXL_DASSERT((horizontal_segments[py].first == kNoSegment) ==
                  (horizontal_segments[py].second == kNoSegment));
      if (horizontal_segments[py].first == kNoSegment) {
        horizontal_segments[py].first = px;
      }
      horizontal_segments[py].second = px + 1;
    }
  }
  if (horizontal_segments[0] == horizontal_segments[1] &&
      horizontal_segments[0] == horizontal_segments[2]) {
    append_rect(horizontal_segments[0].first, horizontal_segments[0].second, 0,
                3);
  } else if (horizontal_segments[0] == horizontal_segments[1]) {
    append_rect(horizontal_segments[0].first, horizontal_segments[0].second, 0,
                2);
    append_rect(horizontal_segments[2].first, horizontal_segments[2].second, 2,
                3);
  } else if (horizontal_segments[1] == horizontal_segments[2]) {
    append_rect(horizontal_segments[0].first, horizontal_segments[0].second, 0,
                1);
    append_rect(horizontal_segments[1].first, horizontal_segments[1].second, 1,
                3);
  } else {
    append_rect(horizontal_segments[0].first, horizontal_segments[0].second, 0,
                1);
    append_rect(horizontal_segments[1].first, horizontal_segments[1].second, 1,
                2);
    append_rect(horizontal_segments[2].first, horizontal_segments[2].second, 2,
                3);
  }
}

}  // namespace jxl
