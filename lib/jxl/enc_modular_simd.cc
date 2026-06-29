// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_modular_simd.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/modular_image.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_modular_simd.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#if HWY_TARGET == HWY_SCALAR
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/pack_signed.h"
#endif

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::And;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::GetLane;
using hwy::HWY_NAMESPACE::Gt;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::Iota;
using hwy::HWY_NAMESPACE::Load;
using hwy::HWY_NAMESPACE::LoadU;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Min;
using hwy::HWY_NAMESPACE::Not;
using hwy::HWY_NAMESPACE::Set;
using hwy::HWY_NAMESPACE::ShiftLeft;
using hwy::HWY_NAMESPACE::ShiftRight;
using hwy::HWY_NAMESPACE::Store;
using hwy::HWY_NAMESPACE::StoreU;
using hwy::HWY_NAMESPACE::Sub;
using hwy::HWY_NAMESPACE::Xor;
using hwy::HWY_NAMESPACE::Zero;

StatusOr<float> EstimateCost(const Image& img) {
  size_t histo_cost = 0;
  float histo_cost_frac = 0.0f;
  size_t extra_bits = 0;

#if HWY_TARGET == HWY_SCALAR
  HybridUintConfig config;
  uint32_t cutoffs[] = {0,  1,  3,  5,   7,   11,  15,  23, 31,
                        47, 63, 95, 127, 191, 255, 392, 500};
  // One context per cutoff (17). Matches estimate_cost_detail::kLastCtx + 1 and
  // the SIMD ContextMap; the old "+ 1" produced an unused 18th histogram.
  constexpr size_t nc = sizeof(cutoffs) / sizeof(*cutoffs);
  Histogram histo[nc] = {};
  for (const Channel& ch : img.channel) {
    const ptrdiff_t onerow = ch.plane.PixelsPerRow();
    for (size_t y = 0; y < ch.h; y++) {
      const pixel_type* JXL_RESTRICT r = ch.Row(y);
      for (size_t x = 0; x < ch.w; x++) {
        pixel_type_w left = (x ? r[x - 1] : y ? *(r + x - onerow) : 0);
        pixel_type_w top = (y ? *(r + x - onerow) : left);
        pixel_type_w topleft = (x && y ? *(r + x - 1 - onerow) : left);
        size_t max_diff =
            std::max({left, top, topleft}) - std::min({left, top, topleft});
        // ctx = highest cutoff index whose cutoff <= max_diff (small diffs ->
        // low ctx), matching the SIMD ContextMap ordering. The scalar path
        // previously used (max_diff < c), which reversed the numbering.
        size_t ctx = 0;
        for (size_t i = 1; i < nc; i++) {
          ctx += (max_diff >= cutoffs[i]) ? 1 : 0;
        }
        pixel_type res = r[x] - ClampedGradient(top, left, topleft);
        uint32_t token;
        uint32_t nbits;
        uint32_t bits;
        config.Encode(PackSigned(res), &token, &nbits, &bits);
        histo[ctx].Add(token);
        extra_bits += nbits;
      }
    }
    for (auto& h : histo) {
      float f_cost = h.ShannonEntropy();
      size_t i_cost = f_cost;
      histo_cost += i_cost;
      histo_cost_frac += f_cost - i_cost;
      h.Clear();
    }
  }
#else
  JxlMemoryManager* memory_manager = img.memory_manager();
  const auto& ctx_map = estimate_cost_detail::ContextMap();
  const HWY_FULL(int32_t) di;
  const HWY_FULL(uint32_t) du;
  const HWY_FULL(float) df;
  const auto kOne = Set(du, 1);
  const auto kSplit = Set(du, 16);
  const auto kExpOffset2 = Set(du, 129);  // 127 + 2
  const auto kTokenBias = Set(du, 8);
  // Token field multiplier is 4 == 1<<kTokenShift (msb_in_token=2). Use an
  // explicit ShiftLeft rather than Mul by 4: a vector Mul by a power-of-two
  // constant is mis-lowered by the x86 backend to a vpsllq + shuffle + blend
  // sequence, whereas ShiftLeft emits a single vpslld. (No-op on wasm, where
  // the backend already strength-reduces the multiply.)
  constexpr int kTokenShift = 2;
  const auto kMsbMask = Set(du, 3);
  const auto kMaxDiffCap = Set(du, estimate_cost_detail::kLastThreshold - 1);
  const auto kIota = Iota(du, 0);
  const auto kLargeThreshold = Set(du, (1 << 22) - 1);
  constexpr size_t kLargeShiftVal = 10;
  const auto kLargeShift = Set(du, kLargeShiftVal);

  size_t max_w = 0;
  for (const Channel& ch : img.channel) {
    if (ch.h == 0) continue;
    max_w = std::max(max_w, ch.w);
  }
  if (max_w == 0) return 0.0f;
  max_w = RoundUpTo(max_w, Lanes(du));
  max_w = std::max(max_w, 2 * Lanes(du));

  JXL_ASSIGN_OR_RETURN(
      AlignedMemory buffer,
      AlignedMemory::Create(memory_manager, max_w * 2 * sizeof(uint32_t)));
  uint32_t* max_diff_row = buffer.address<uint32_t>();
  uint32_t* token_row = max_diff_row + max_w;
  int32_t* primer = buffer.address<int32_t>();
  int32_t* top_primer = primer + max_w;

  Histogram histo[estimate_cost_detail::kLastCtx + 1] = {};
  // extra_bits_lanes is a uint32 vector; each lane accumulates per-vector
  // extra-bit counts. On large/high-entropy inputs a lane can wrap before the
  // final SumOfLanes, undercounting by whole multiples of 2^32. Flush into the
  // size_t accumulator periodically so the running per-lane sum stays small.
  auto extra_bits_lanes = Zero(du);
  uint32_t vectors_since_flush = 0;
  // Each lane adds at most ~32 extra bits per vector (token nbits <= 31), and the
  // final SumOfLanes folds all `Lanes(du)` lanes into one uint32 accumulator.
  // Flush before `Lanes * period * kMaxExtraBitsPerLane` can wrap 2^32 — derived
  // from the actual lane count instead of the old fixed 4096, which forced a
  // SumOfLanes reduction every 4096 vectors on large/high-entropy rows. Byte-exact:
  // the running total is identical, only the flush cadence changes.
  constexpr uint32_t kMaxExtraBitsPerLane = 32;
  const uint32_t kExtraBitsFlushPeriod = static_cast<uint32_t>(
      std::numeric_limits<uint32_t>::max() /
      (kMaxExtraBitsPerLane * Lanes(du)));
  const auto FlushExtraBits = [&] {
    extra_bits += static_cast<size_t>(GetLane(SumOfLanes(du, extra_bits_lanes)));
    extra_bits_lanes = Zero(du);
    vectors_since_flush = 0;
  };
  for (const Channel& ch : img.channel) {
    if (ch.h == 0 || ch.w == 0) continue;
    for (auto& h : histo) {
      h.EnsureCapacity(32 * 4);
    }
    const pixel_type* JXL_RESTRICT r = ch.Row(0);
    const pixel_type* JXL_RESTRICT last = primer;
    primer[0] = 0;
    StoreU(Load(di, r), di, primer + 1);
    for (size_t x = 0; x < ch.w; x += Lanes(di)) {
      const auto left = LoadU(di, last);
      const auto central = Load(di, r + x);
      const auto ures = BitCast(du, Sub(central, left));
      const auto packed =
          Xor(ShiftLeft<1>(ures), Sub(ShiftRight<31>(Not(ures)), kOne));
      const auto is_large = Gt(packed, kLargeThreshold);
      const auto packed_shifted = ShiftRight<kLargeShiftVal>(packed);
      const auto not_literal = Ge(packed, kSplit);
      const auto packed_fixed = IfThenElse(is_large, packed_shifted, packed);
      const auto v = BitCast(du, ConvertTo(df, packed_fixed));
      const auto eb_raw = Sub(ShiftRight<23>(v), kExpOffset2);
      const auto eb = IfThenElse(is_large, Add(eb_raw, kLargeShift), eb_raw);
      const auto token = Add(Add(kTokenBias, ShiftLeft<kTokenShift>(eb)),
                             And(ShiftRight<21>(v), kMsbMask));
      const auto eb_fixed = IfThenElseZero(not_literal, eb);
      const auto token_fixed = IfThenElse(not_literal, token, packed);
      if (x + Lanes(di) <= ch.w) {
        extra_bits_lanes = Add(extra_bits_lanes, eb_fixed);
      } else {
        const auto tail_mask =
            Lt(kIota, Set(du, static_cast<uint32_t>(ch.w - x)));
        extra_bits_lanes =
            Add(extra_bits_lanes, IfThenElseZero(tail_mask, eb_fixed));
      }
      Store(token_fixed, du, token_row + x);
      last = r + x + Lanes(di) - 1;
      if (++vectors_since_flush == kExtraBitsFlushPeriod) FlushExtraBits();
    }
    {
      // Coalesce runs of identical tokens before updating the histogram:
      // repeated FastAdd to the same counts slot serializes on store-to-load
      // forwarding. Row 0 has no max_diff context, so every token is ctx 0.
      uint32_t run = 0;
      uint32_t last_tok = token_row[0];
      for (size_t x = 0; x < ch.w; x++) {
        const uint32_t tok = token_row[x];
        if (tok == last_tok) {
          run++;
        } else {
          histo[0].FastAddN(last_tok, run);
          last_tok = tok;
          run = 1;
        }
      }
      histo[0].FastAddN(last_tok, run);
    }
    for (size_t y = 1; y < ch.h; y++) {
      r = ch.Row(y);
      const pixel_type* JXL_RESTRICT t = ch.Row(y - 1);
      last = primer;
      primer[0] = t[0];
      StoreU(Load(di, r), di, primer + 1);
      top_primer[0] = t[0];
      StoreU(Load(di, t), di, top_primer + 1);
      const pixel_type* JXL_RESTRICT top_last = top_primer;
      for (size_t x = 0; x < ch.w; x += Lanes(di)) {
        const auto left = LoadU(di, last);
        const auto central = Load(di, r + x);
        const auto topleft = LoadU(di, top_last);
        const auto top = Load(di, t + x);
        const auto m = Min(left, top);
        const auto M = Max(left, top);
        const auto maxx = Max(topleft, M);
        const auto minn = Min(topleft, m);
        const auto max_diff = BitCast(du, Sub(maxx, minn));
        Store(Min(max_diff, kMaxDiffCap), du, max_diff_row + x);
        const auto overshoot = Lt(topleft, m);
        const auto undershoot = Gt(topleft, M);
        const auto grad =
            BitCast(di, Sub(Add(BitCast(du, top), BitCast(du, left)),
                            BitCast(du, topleft)));
        const auto prediction =
            IfThenElse(undershoot, m, IfThenElse(overshoot, M, grad));
        const auto ures = BitCast(du, Sub(central, prediction));
        const auto packed =
            Xor(ShiftLeft<1>(ures), Sub(ShiftRight<31>(Not(ures)), kOne));
        const auto is_large = Gt(packed, kLargeThreshold);
        const auto packed_shifted = ShiftRight<kLargeShiftVal>(packed);
        const auto not_literal = Ge(packed, kSplit);
        const auto packed_fixed = IfThenElse(is_large, packed_shifted, packed);
        const auto v = BitCast(du, ConvertTo(df, packed_fixed));
        const auto eb_raw = Sub(ShiftRight<23>(v), kExpOffset2);
        const auto eb = IfThenElse(is_large, Add(eb_raw, kLargeShift), eb_raw);
        const auto token = Add(Add(kTokenBias, ShiftLeft<kTokenShift>(eb)),
                               And(ShiftRight<21>(v), kMsbMask));
        const auto eb_fixed = IfThenElseZero(not_literal, eb);
        const auto token_fixed = IfThenElse(not_literal, token, packed);
        if (x + Lanes(di) <= ch.w) {
          extra_bits_lanes = Add(extra_bits_lanes, eb_fixed);
        } else {
          const auto tail_mask =
              Lt(kIota, Set(du, static_cast<uint32_t>(ch.w - x)));
          extra_bits_lanes =
              Add(extra_bits_lanes, IfThenElseZero(tail_mask, eb_fixed));
        }
        Store(token_fixed, du, token_row + x);
        last = r + x + Lanes(di) - 1;
        top_last = t + x + Lanes(di) - 1;
        if (++vectors_since_flush == kExtraBitsFlushPeriod) FlushExtraBits();
      }
      {
        // Coalesce runs of identical (ctx, token) keys (see row 0). The token
        // fits in 8 bits (default hybrid-uint max token is 127), so pack
        // ctx in the high bits and token in the low byte to compare in one op.
        uint32_t run = 0;
        uint32_t last_key =
            (static_cast<uint32_t>(ctx_map[max_diff_row[0]]) << 8) | token_row[0];
        for (size_t x = 0; x < ch.w; x++) {
          const uint32_t key =
              (static_cast<uint32_t>(ctx_map[max_diff_row[x]]) << 8) |
              token_row[x];
          if (key == last_key) {
            run++;
          } else {
            histo[last_key >> 8].FastAddN(last_key & 0xff, run);
            last_key = key;
            run = 1;
          }
        }
        histo[last_key >> 8].FastAddN(last_key & 0xff, run);
      }
    }
    for (auto& h : histo) {
      h.Condition();
      float f_cost = h.ShannonEntropy();
      size_t i_cost = f_cost;
      histo_cost += i_cost;
      histo_cost_frac += f_cost - i_cost;
      h.Clear();
    }
  }
  FlushExtraBits();
#endif
  size_t total_cost =
      extra_bits + histo_cost + static_cast<size_t>(histo_cost_frac);
  return total_cost;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(EstimateCost);

StatusOr<float> EstimateCost(const Image& img) {
  return HWY_DYNAMIC_DISPATCH(EstimateCost)(img);
}

namespace estimate_cost_detail {
/*
cutoffs = [0, 1, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500]
ctx_map = [max(c for c, cutoff in enumerate(cutoffs) if cutoff <= i) for i in range(501)]
*/
const std::array<uint8_t, kLastThreshold>& ContextMap() {
  static const std::array<uint8_t, kLastThreshold> kCtxMap = {
      0,  1,  1,  2,  2,  3,  3,  4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,
      6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,
      8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,
      9,  9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 10,
      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
      10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
      11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
      11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
      12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
      12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
      12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13,
      13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
      13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
      13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
      13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
      14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
      14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
      14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
      14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
      14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
      14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
      14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15,
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16};
  return kCtxMap;
}
}  // namespace estimate_cost_detail

}  // namespace jxl
#endif
