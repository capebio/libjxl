// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Standalone A/B + equivalence harness for GroupBorderAssigner.
//
// Proves that the atomic-coalescing change in dec_group_border.cc is
// behaviour-identical to the original (each corner/border is still finalized by
// the same group with the same Rect) and reports the reduction in atomic
// read-modify-write operations.
//
// Build (from the libjxl-012 submodule root):
//   clang++ -std=c++17 -O2 -I. tools/dec_group_border_ab.cc \
//       lib/jxl/dec_group_border.cc -o /tmp/dgb_ab
//
// - RefOld  : verbatim copy of the ORIGINAL GroupDone/ClearDone (4 separate
//             fetch_or / fetch_and) with an atomic-op counter.
// - RefNew  : the coalesced algorithm with an atomic-op counter.
// - Real    : the shipped jxl::GroupBorderAssigner (linked from the real .cc).
//
// Test: for every frame geometry and completion order, RefOld, RefNew and Real
// must assign the identical set of Rects to each group. Any mismatch = FAIL.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <utility>
#include <vector>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/dec_group_border.h"

namespace jxl {

// ---------------------------------------------------------------------------
// Reference implementations operating on a plain (non-atomic) counter array.
// Single-threaded, so plain uint32_t OR/AND models fetch_or/fetch_and exactly.
// ---------------------------------------------------------------------------

static constexpr uint8_t kTopLeft = 0x01;
static constexpr uint8_t kTopRight = 0x02;
static constexpr uint8_t kBottomRight = 0x04;
static constexpr uint8_t kBottomLeft = 0x08;
static constexpr size_t kMaxToFinalize = 3;

struct RefBase {
  FrameDimensions frame_dim_;
  std::vector<uint32_t> counters_;
  size_t counters_stride_ = 0;
  uint64_t ops = 0;  // atomic RMW op counter

  void Init(const FrameDimensions& frame_dim) {
    frame_dim_ = frame_dim;
    size_t x_size = frame_dim_.xsize_groups;
    size_t y_size = frame_dim_.ysize_groups;
    counters_stride_ = (x_size + 1 + 7) / 8;
    const size_t n = (y_size + 1) * counters_stride_;
    counters_.assign(n, 0);
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

  // Shared Rect-assembly (identical to the real code) given the 4 statuses.
  void Assemble(size_t x, size_t y, size_t padx, size_t pady,
                size_t top_left_status, size_t top_right_status,
                size_t bottom_right_status, size_t bottom_left_status,
                Rect* rects_to_finalize, size_t* num_to_finalize) {
    Rect block_rect(x * frame_dim_.group_dim / kBlockDim,
                    y * frame_dim_.group_dim / kBlockDim,
                    frame_dim_.group_dim / kBlockDim,
                    frame_dim_.group_dim / kBlockDim, frame_dim_.xsize_blocks,
                    frame_dim_.ysize_blocks);
    size_t x1 = block_rect.x0() + block_rect.xsize();
    size_t y1 = block_rect.y0() + block_rect.ysize();
    bool is_last_group_x = frame_dim_.xsize_groups == x + 1;
    bool is_last_group_y = frame_dim_.ysize_groups == y + 1;
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
    auto append_rect = [&](size_t ax0, size_t ax1, size_t ay0, size_t ay1) {
      Rect rect(xpos[ax0], ypos[ay0], xpos[ax1] - xpos[ax0],
                ypos[ay1] - ypos[ay0]);
      if (rect.xsize() == 0 || rect.ysize() == 0) return;
      rects_to_finalize[(*num_to_finalize)++] = rect;
    };

    bool available_parts_mask[3][3] = {};
    available_parts_mask[1][1] = true;
    if (top_left_status == 0xF) available_parts_mask[0][0] = true;
    if (top_right_status == 0xF) available_parts_mask[2][0] = true;
    if (bottom_right_status == 0xF) available_parts_mask[2][2] = true;
    if (bottom_left_status == 0xF) available_parts_mask[0][2] = true;
    if (top_left_status & kTopRight) available_parts_mask[1][0] = true;
    if (top_left_status & kBottomLeft) available_parts_mask[0][1] = true;
    if (top_right_status & kBottomRight) available_parts_mask[2][1] = true;
    if (bottom_left_status & kBottomRight) available_parts_mask[1][2] = true;

    constexpr size_t kNoSegment = 3;
    std::pair<size_t, size_t> horizontal_segments[3] = {
        {kNoSegment, kNoSegment},
        {kNoSegment, kNoSegment},
        {kNoSegment, kNoSegment}};
    for (size_t py = 0; py < 3; py++) {
      for (size_t px = 0; px < 3; px++) {
        if (!available_parts_mask[px][py]) continue;
        if (horizontal_segments[py].first == kNoSegment) {
          horizontal_segments[py].first = px;
        }
        horizontal_segments[py].second = px + 1;
      }
    }
    if (horizontal_segments[0] == horizontal_segments[1] &&
        horizontal_segments[0] == horizontal_segments[2]) {
      append_rect(horizontal_segments[0].first, horizontal_segments[0].second,
                  0, 3);
    } else if (horizontal_segments[0] == horizontal_segments[1]) {
      append_rect(horizontal_segments[0].first, horizontal_segments[0].second,
                  0, 2);
      append_rect(horizontal_segments[2].first, horizontal_segments[2].second,
                  2, 3);
    } else if (horizontal_segments[1] == horizontal_segments[2]) {
      append_rect(horizontal_segments[0].first, horizontal_segments[0].second,
                  0, 1);
      append_rect(horizontal_segments[1].first, horizontal_segments[1].second,
                  1, 3);
    } else {
      append_rect(horizontal_segments[0].first, horizontal_segments[0].second,
                  0, 1);
      append_rect(horizontal_segments[1].first, horizontal_segments[1].second,
                  1, 2);
      append_rect(horizontal_segments[2].first, horizontal_segments[2].second,
                  2, 3);
    }
  }
};

// Original: 4 separate fetch_or / fetch_and.
struct RefOld : RefBase {
  size_t fetch_status(size_t x, size_t y, uint32_t bit) {
    size_t shift = 4 * (x & 7u);
    uint32_t& w = counters_[y * counters_stride_ + x / 8];
    size_t status = w;
    w |= (bit << shift);
    ops++;
    status >>= shift;
    return (bit | status) & 0xF;
  }
  void GroupDone(size_t group_id, size_t padx, size_t pady,
                 Rect* rects_to_finalize, size_t* num_to_finalize) {
    size_t x = group_id % frame_dim_.xsize_groups;
    size_t y = group_id / frame_dim_.xsize_groups;
    size_t tl = fetch_status(x, y, kBottomRight);
    size_t tr = fetch_status(x + 1, y, kBottomLeft);
    size_t br = fetch_status(x + 1, y + 1, kTopLeft);
    size_t bl = fetch_status(x, y + 1, kTopRight);
    Assemble(x, y, padx, pady, tl, tr, br, bl, rects_to_finalize,
             num_to_finalize);
  }
  void ClearDone(size_t group_id) {
    auto clear = [this](size_t x, size_t y, uint32_t corners) {
      counters_[y * counters_stride_ + x / 8] &= ~(corners << (4 * (x & 7u)));
      ops++;
    };
    size_t x = group_id % frame_dim_.xsize_groups;
    size_t y = group_id / frame_dim_.xsize_groups;
    clear(x, y, kBottomRight);
    clear(x + 1, y, kBottomLeft);
    clear(x, y + 1, kTopRight);
    clear(x + 1, y + 1, kTopLeft);
  }
};

// New: coalesced same-row pair.
struct RefNew : RefBase {
  void fetch_status_pair(size_t x, size_t y, uint32_t bit_left,
                         uint32_t bit_right, size_t* out_left,
                         size_t* out_right) {
    const size_t idx_l = y * counters_stride_ + x / 8;
    const size_t idx_r = y * counters_stride_ + (x + 1) / 8;
    const size_t shift_l = 4 * (x & 7u);
    const size_t shift_r = 4 * ((x + 1) & 7u);
    size_t old_l, old_r;
    if (idx_l == idx_r) {
      uint32_t& w = counters_[idx_l];
      old_l = w;
      old_r = w;
      w |= (bit_left << shift_l) | (bit_right << shift_r);
      ops++;
    } else {
      uint32_t& wl = counters_[idx_l];
      old_l = wl;
      wl |= (bit_left << shift_l);
      ops++;
      uint32_t& wr = counters_[idx_r];
      old_r = wr;
      wr |= (bit_right << shift_r);
      ops++;
    }
    *out_left = (bit_left | ((old_l >> shift_l) & 0xF)) & 0xF;
    *out_right = (bit_right | ((old_r >> shift_r) & 0xF)) & 0xF;
  }
  void GroupDone(size_t group_id, size_t padx, size_t pady,
                 Rect* rects_to_finalize, size_t* num_to_finalize) {
    size_t x = group_id % frame_dim_.xsize_groups;
    size_t y = group_id / frame_dim_.xsize_groups;
    size_t tl, tr, bl, br;
    fetch_status_pair(x, y, kBottomRight, kBottomLeft, &tl, &tr);
    fetch_status_pair(x, y + 1, kTopRight, kTopLeft, &bl, &br);
    Assemble(x, y, padx, pady, tl, tr, br, bl, rects_to_finalize,
             num_to_finalize);
  }
  void ClearDone(size_t group_id) {
    auto clear_pair = [this](size_t x, size_t y, uint32_t bl, uint32_t br) {
      const size_t idx_l = y * counters_stride_ + x / 8;
      const size_t idx_r = y * counters_stride_ + (x + 1) / 8;
      const size_t shift_l = 4 * (x & 7u);
      const size_t shift_r = 4 * ((x + 1) & 7u);
      if (idx_l == idx_r) {
        counters_[idx_l] &= ~((bl << shift_l) | (br << shift_r));
        ops++;
      } else {
        counters_[idx_l] &= ~(bl << shift_l);
        counters_[idx_r] &= ~(br << shift_r);
        ops += 2;
      }
    };
    size_t x = group_id % frame_dim_.xsize_groups;
    size_t y = group_id / frame_dim_.xsize_groups;
    clear_pair(x, y, kBottomRight, kBottomLeft);
    clear_pair(x, y + 1, kTopRight, kTopLeft);
  }
};

}  // namespace jxl

// ---------------------------------------------------------------------------

using jxl::FrameDimensions;
using jxl::GroupBorderAssigner;
using jxl::Rect;
using jxl::RefNew;
using jxl::RefOld;

static bool RectEq(const Rect& a, const Rect& b) {
  return a.x0() == b.x0() && a.y0() == b.y0() && a.xsize() == b.xsize() &&
         a.ysize() == b.ysize();
}

// Deterministic LCG so runs are reproducible (no wall-clock/random seeding).
struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed) {}
  uint32_t next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
};

int main() {
  const std::vector<std::pair<size_t, size_t>> sizes = {
      // width_px, height_px -- chosen to sweep xsize_groups across the 8-corner
      // word boundary (group_dim = 128 with group_size_shift = 0).
      {128, 128},   {256, 128},   {896, 128},   {1024, 128}, {1152, 128},
      {1920, 128},  {2048, 128},  {1152, 384},  {2048, 1152}, {1152, 1152},
      {900, 700},   {1000, 640},  {1930, 1090}, {130, 900},  {3000, 200},
  };
  const std::vector<std::pair<size_t, size_t>> pads = {{2, 2}, {0, 0}, {3, 1}};

  size_t configs = 0, order_runs = 0;
  uint64_t total_groups = 0;
  uint64_t old_ops = 0, new_ops = 0;
  bool ok = true;

  for (const auto& sz : sizes) {
    FrameDimensions fd;
    // group_size_shift=0 -> group_dim=128; no subsampling; VarDCT; no upsampling
    fd.Set(sz.first, sz.second, /*group_size_shift=*/0, /*max_hshift=*/0,
           /*max_vshift=*/0, /*modular_mode=*/false, /*upsampling=*/1);
    const size_t ng = fd.num_groups;
    for (const auto& pad : pads) {
      // Try several completion orders: forward, reverse, and pseudo-random.
      std::vector<std::vector<size_t>> orders;
      {
        std::vector<size_t> fwd(ng), rev(ng);
        for (size_t i = 0; i < ng; i++) fwd[i] = i;
        for (size_t i = 0; i < ng; i++) rev[i] = ng - 1 - i;
        orders.push_back(fwd);
        orders.push_back(rev);
      }
      for (uint64_t seed = 1; seed <= 6; seed++) {
        std::vector<size_t> perm(ng);
        for (size_t i = 0; i < ng; i++) perm[i] = i;
        Lcg rng(seed * 2654435761ULL + sz.first * 131 + sz.second);
        for (size_t i = ng; i > 1; i--) {
          size_t j = rng.next() % i;
          std::swap(perm[i - 1], perm[j]);
        }
        orders.push_back(perm);
      }

      configs++;
      for (const auto& order : orders) {
        order_runs++;
        RefOld ro;
        RefNew rn;
        GroupBorderAssigner real;
        ro.Init(fd);
        rn.Init(fd);
        real.Init(fd);
        for (size_t gid : order) {
          Rect ro_r[3], rn_r[3], real_r[3];
          size_t ro_n = 0, rn_n = 0, real_n = 0;
          ro.GroupDone(gid, pad.first, pad.second, ro_r, &ro_n);
          rn.GroupDone(gid, pad.first, pad.second, rn_r, &rn_n);
          real.GroupDone(gid, pad.first, pad.second, real_r, &real_n);
          total_groups++;
          // Compare counts.
          if (ro_n != rn_n || ro_n != real_n) {
            fprintf(stderr,
                    "MISMATCH count px=%zux%zu pad=%zu,%zu gid=%zu: "
                    "old=%zu new=%zu real=%zu\n",
                    sz.first, sz.second, pad.first, pad.second, gid, ro_n, rn_n,
                    real_n);
            ok = false;
            continue;
          }
          // Compare each rect (order within the group is deterministic and
          // identical across impls, so positionwise comparison is valid).
          for (size_t k = 0; k < ro_n; k++) {
            if (!RectEq(ro_r[k], rn_r[k]) || !RectEq(ro_r[k], real_r[k])) {
              fprintf(stderr,
                      "MISMATCH rect px=%zux%zu pad=%zu,%zu gid=%zu k=%zu: "
                      "old=(%zu,%zu,%zu,%zu) new=(%zu,%zu,%zu,%zu) "
                      "real=(%zu,%zu,%zu,%zu)\n",
                      sz.first, sz.second, pad.first, pad.second, gid, k,
                      (size_t)ro_r[k].x0(), (size_t)ro_r[k].y0(),
                      ro_r[k].xsize(), ro_r[k].ysize(), (size_t)rn_r[k].x0(),
                      (size_t)rn_r[k].y0(), rn_r[k].xsize(), rn_r[k].ysize(),
                      (size_t)real_r[k].x0(), (size_t)real_r[k].y0(),
                      real_r[k].xsize(), real_r[k].ysize());
              ok = false;
            }
          }
        }
        // Also exercise ClearDone equivalence: clear all, both must return to
        // the same counter state (checked via a fresh full sweep afterwards).
        for (size_t gid : order) {
          ro.ClearDone(gid);
          rn.ClearDone(gid);
        }
        if (ro.counters_ != rn.counters_) {
          fprintf(stderr, "MISMATCH ClearDone counter state px=%zux%zu\n",
                  sz.first, sz.second);
          ok = false;
        }
      }
    }
  }

  // Dedicated atomic-op-count pass: one forward-order GroupDone sweep per size
  // (pad = 2,2), summing the RMW ops the border hot path performs.
  for (const auto& sz : sizes) {
    FrameDimensions fd;
    fd.Set(sz.first, sz.second, 0, 0, 0, false, 1);
    const size_t ng = fd.num_groups;
    RefOld ro;
    RefNew rn;
    ro.Init(fd);
    rn.Init(fd);
    Rect r[3];
    size_t n = 0;
    for (size_t gid = 0; gid < ng; gid++) {
      ro.GroupDone(gid, 2, 2, r, &n);
      rn.GroupDone(gid, 2, 2, r, &n);
    }
    old_ops += ro.ops;
    new_ops += rn.ops;
  }

  printf("configs=%zu order_runs=%zu group_calls=%llu\n", configs, order_runs,
         (unsigned long long)total_groups);
  printf("GroupDone atomic RMW ops (all sizes): OLD=%llu NEW=%llu  (%.1f%% of OLD)\n",
         (unsigned long long)old_ops, (unsigned long long)new_ops,
         old_ops ? 100.0 * static_cast<double>(new_ops) / old_ops : 0.0);
  printf(ok ? "EQUIVALENCE: PASS\n" : "EQUIVALENCE: FAIL\n");
  return ok ? 0 : 1;
}
