// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_group.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>
#ifdef JXL_DEC_TRANSFORM_STATS
#include <atomic>
#endif

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/jpeg/jpeg_data.h"
#include "lib/jxl/render_pipeline/render_pipeline.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/dec_group.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/common.h"  // kMaxNumPasses
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_transforms-inl.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/quantizer-inl.h"
#include "lib/jxl/quantizer.h"

#ifndef LIB_JXL_DEC_GROUP_CC
#define LIB_JXL_DEC_GROUP_CC
namespace jxl {

struct AuxOut;

// Interface for reading groups for DecodeGroupImpl.
class GetBlock {
 public:
  virtual void StartRow(size_t by) = 0;
  virtual Status LoadBlock(size_t bx, size_t by, const AcStrategy& acs,
                           size_t size, size_t log2_covered_blocks,
                           ACPtr block[3], ACType ac_type) = 0;
  virtual ~GetBlock() {}
};

// Controls whether DecodeGroupImpl renders to pixels or not.
enum DrawMode {
  // Render to pixels.
  kDraw = 0,
  // Don't render to pixels.
  kDontDraw = 1,
};

// Pre-computed per-frame JPEG reconstruction parameters.
// Computed once in DecodeGroup / DecodeGroupForRoundtrip and passed to
// DecodeGroupImpl, avoiding repeated validation and integer divisions per group.
struct JpegGroupParams {
  alignas(64) int32_t scaled_qtable[3 * kDCTBlockSize];
  std::array<int, 3> jpeg_c_map;
  std::array<int, 3> dc_offset;
  bool is_gray = false;
};

#ifdef JXL_DEC_TRANSFORM_STATS
// Compile-gated decode-path counters.  Enable with -DJXL_DEC_TRANSFORM_STATS.
// Dumped to stderr at program exit via the static destructor.
struct DecTransformStats {
  static constexpr size_t kMaxStrats = 32;
  static constexpr size_t kNzBuckets = 7;  // 0 | 1-2 | 3-4 | 5-8 | 9-16 | 17-32 | >32

  static size_t NzBucket(size_t nz) {
    if (nz == 0) return 0;
    if (nz <= 2) return 1;
    if (nz <= 4) return 2;
    if (nz <= 8) return 3;
    if (nz <= 16) return 4;
    if (nz <= 32) return 5;
    return 6;
  }

  std::atomic<uint64_t> block_count{0};
  // strategy histogram
  std::atomic<uint64_t> strat_blocks[kMaxStrats]{};
  std::atomic<uint64_t> strat_coeffs[kMaxStrats]{};
  // active component mask: bit0=X_active bit1=B_active (Y always active)
  // slot 0=Y-only  1=X+Y  2=Y+B  3=X+Y+B
  std::atomic<uint64_t> comp_mask[4]{};
  // AC nzeros histogram per channel (0=X 1=Y 2=B)
  std::atomic<uint64_t> nz_hist[3][kNzBuckets]{};
  // CfL mode per block: bit0=x_cfl_active bit1=b_cfl_active
  std::atomic<uint64_t> cfl_mode[4]{};
  // coefficient pass mode
  std::atomic<uint64_t> accumulate_blocks{0};
  std::atomic<uint64_t> single_pass_blocks{0};
  // dequant work ledger (coefficient slots = blocks * 64 per channel)
  std::atomic<uint64_t> slots_total{0};    // 3 * size per block (always 3 channels)
  std::atomic<uint64_t> slots_rendered{0}; // channels that pass the activity check
  std::atomic<uint64_t> slots_inactive{0}; // channels that are skipped by IDCT/render

  static DecTransformStats& Get() {
    static DecTransformStats s;
    return s;
  }

  ~DecTransformStats() { Print(); }

  void Print() const {
    const uint64_t total = block_count.load(std::memory_order_relaxed);
    fprintf(stderr, "\n=== JXL_DEC_TRANSFORM_STATS: %llu blocks ===\n",
            (unsigned long long)total);
    if (total == 0) return;

    fprintf(stderr, "--- Strategy (count | coeff_vol | %%blocks) ---\n");
    for (size_t i = 0; i < kMaxStrats; i++) {
      uint64_t bc = strat_blocks[i].load(std::memory_order_relaxed);
      if (bc == 0) continue;
      uint64_t cv = strat_coeffs[i].load(std::memory_order_relaxed);
      fprintf(stderr, "  [%2zu] %8llu blk  %12llu coeff  %5.1f%%\n",
              i, (unsigned long long)bc, (unsigned long long)cv,
              100.0 * bc / total);
    }

    fprintf(stderr, "--- Active component mask ---\n");
    static const char* const kMaskName[] = {"Y-only", "X+Y", "Y+B", "X+Y+B"};
    for (int i = 0; i < 4; i++) {
      uint64_t c = comp_mask[i].load(std::memory_order_relaxed);
      if (c == 0) continue;
      fprintf(stderr, "  %-8s %8llu  %5.1f%%\n", kMaskName[i],
              (unsigned long long)c, 100.0 * c / total);
    }

    static const char* const kChName[] = {"X", "Y", "B"};
    static const char* const kNzName[] = {"0","1-2","3-4","5-8","9-16","17-32",">32"};
    fprintf(stderr, "--- AC nzeros histogram ---\n");
    for (int c = 0; c < 3; c++) {
      uint64_t ch_total = 0;
      for (size_t b = 0; b < kNzBuckets; b++)
        ch_total += nz_hist[c][b].load(std::memory_order_relaxed);
      if (ch_total == 0) continue;
      fprintf(stderr, "  %s:", kChName[c]);
      for (size_t b = 0; b < kNzBuckets; b++) {
        uint64_t cnt = nz_hist[c][b].load(std::memory_order_relaxed);
        if (cnt == 0) continue;
        fprintf(stderr, "  %s=%.1f%%", kNzName[b], 100.0 * cnt / ch_total);
      }
      fprintf(stderr, "\n");
    }

    fprintf(stderr, "--- CfL mode ---\n");
    static const char* const kCflName[] = {"none","X-only","B-only","X+B"};
    for (int i = 0; i < 4; i++) {
      uint64_t c = cfl_mode[i].load(std::memory_order_relaxed);
      if (c == 0) continue;
      fprintf(stderr, "  %-8s %8llu  %5.1f%%\n", kCflName[i],
              (unsigned long long)c, 100.0 * c / total);
    }

    uint64_t acc = accumulate_blocks.load(std::memory_order_relaxed);
    uint64_t sp  = single_pass_blocks.load(std::memory_order_relaxed);
    fprintf(stderr, "--- Pass mode ---\n");
    fprintf(stderr, "  accumulate %llu  single-pass %llu\n",
            (unsigned long long)acc, (unsigned long long)sp);

    uint64_t sd = slots_total.load(std::memory_order_relaxed);
    uint64_t sr = slots_rendered.load(std::memory_order_relaxed);
    uint64_t si = slots_inactive.load(std::memory_order_relaxed);
    fprintf(stderr, "--- Dequant work ledger ---\n");
    fprintf(stderr, "  total=%llu  rendered=%llu (%.1f%%)  inactive=%llu (%.1f%%)\n",
            (unsigned long long)sd,
            (unsigned long long)sr, sd ? 100.0 * sr / sd : 0.0,
            (unsigned long long)si, sd ? 100.0 * si / sd : 0.0);
  }
};
#endif  // JXL_DEC_TRANSFORM_STATS

}  // namespace jxl
#endif  // LIB_JXL_DEC_GROUP_CC

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::AllFalse;
using hwy::HWY_NAMESPACE::Gt;
using hwy::HWY_NAMESPACE::Le;
using hwy::HWY_NAMESPACE::MaskFromVec;
using hwy::HWY_NAMESPACE::Or;
using hwy::HWY_NAMESPACE::Rebind;
using hwy::HWY_NAMESPACE::ShiftRight;

using D = HWY_FULL(float);
using DU = HWY_FULL(uint32_t);
using DI = HWY_FULL(int32_t);
using DI16 = Rebind<int16_t, DI>;
using DI16_FULL = HWY_CAPPED(int16_t, kDCTBlockSize);
constexpr D d;
constexpr DI di;
constexpr DI16 di16;
constexpr DI16_FULL di16_full;

// TODO(veluca): consider SIMDfying.
void Transpose8x8InPlace(int32_t* JXL_RESTRICT block) {
  for (size_t x = 0; x < 8; x++) {
    for (size_t y = x + 1; y < 8; y++) {
      std::swap(block[y * 8 + x], block[x * 8 + y]);
    }
  }
}

template <ACType ac_type>
void DequantLane(Vec<D> scaled_dequant_x, Vec<D> scaled_dequant_y,
                 Vec<D> scaled_dequant_b,
                 const float* JXL_RESTRICT dequant_matrices, size_t size,
                 size_t k, Vec<D> x_cc_mul, Vec<D> b_cc_mul,
                 const float* JXL_RESTRICT biases, ACPtr qblock[3],
                 float* JXL_RESTRICT block) {
  const auto x_mul = Mul(Load(d, dequant_matrices + k), scaled_dequant_x);
  const auto y_mul =
      Mul(Load(d, dequant_matrices + size + k), scaled_dequant_y);
  const auto b_mul =
      Mul(Load(d, dequant_matrices + 2 * size + k), scaled_dequant_b);

  Vec<DI> quantized_x_int;
  Vec<DI> quantized_y_int;
  Vec<DI> quantized_b_int;
  if (ac_type == ACType::k16) {
    quantized_x_int = PromoteTo(di, Load(di16, qblock[0].ptr16 + k));
    quantized_y_int = PromoteTo(di, Load(di16, qblock[1].ptr16 + k));
    quantized_b_int = PromoteTo(di, Load(di16, qblock[2].ptr16 + k));
  } else {
    quantized_x_int = Load(di, qblock[0].ptr32 + k);
    quantized_y_int = Load(di, qblock[1].ptr32 + k);
    quantized_b_int = Load(di, qblock[2].ptr32 + k);
  }

  const auto dequant_x_cc =
      Mul(AdjustQuantBias(di, 0, quantized_x_int, biases), x_mul);
  const auto dequant_y =
      Mul(AdjustQuantBias(di, 1, quantized_y_int, biases), y_mul);
  const auto dequant_b_cc =
      Mul(AdjustQuantBias(di, 2, quantized_b_int, biases), b_mul);

  const auto dequant_x = MulAdd(x_cc_mul, dequant_y, dequant_x_cc);
  const auto dequant_b = MulAdd(b_cc_mul, dequant_y, dequant_b_cc);
  Store(dequant_x, d, block + k);
  Store(dequant_y, d, block + size + k);
  Store(dequant_b, d, block + 2 * size + k);
}

template <ACType ac_type>
void DequantBlock(float inv_global_scale, int quant, float x_dm_multiplier,
                  float b_dm_multiplier, Vec<D> x_cc_mul, Vec<D> b_cc_mul,
                  AcStrategyType kind, size_t size, const Quantizer& quantizer,
                  size_t covered_blocks, const size_t* sbx,
                  const float* JXL_RESTRICT* JXL_RESTRICT dc_row,
                  size_t dc_stride, const float* JXL_RESTRICT biases,
                  ACPtr qblock[3], float* JXL_RESTRICT block,
                  float* JXL_RESTRICT scratch) {
  const auto scaled_dequant_s = inv_global_scale / quant;

  const auto scaled_dequant_x = Set(d, scaled_dequant_s * x_dm_multiplier);
  const auto scaled_dequant_y = Set(d, scaled_dequant_s);
  const auto scaled_dequant_b = Set(d, scaled_dequant_s * b_dm_multiplier);

  const float* dequant_matrices = quantizer.DequantMatrix(kind, 0);

  for (size_t k = 0; k < covered_blocks * kDCTBlockSize; k += Lanes(d)) {
    DequantLane<ac_type>(scaled_dequant_x, scaled_dequant_y, scaled_dequant_b,
                         dequant_matrices, size, k, x_cc_mul, b_cc_mul, biases,
                         qblock, block);
  }
  if (JXL_LIKELY(covered_blocks == 1)) {
    block[0]        = dc_row[0][sbx[0]];
    block[size]     = dc_row[1][sbx[1]];
    block[2 * size] = dc_row[2][sbx[2]];
  } else {
    for (size_t c = 0; c < 3; c++) {
      LowestFrequenciesFromDC(kind, dc_row[c] + sbx[c], dc_stride,
                              block + c * size, scratch);
    }
  }
}

// DecodeGroupImpl renders one group to pixels. kReadCoefficients is false
// for progressive redraws that render the persistent coefficient store: that
// path must not construct or touch the entropy reader. kDontDraw groups are
// handled by DecodeGroupNoDraw (below, in HWY_ONCE).
template <bool kReadCoefficients>
Status DecodeGroupImpl(const FrameHeader& frame_header,
                       GetBlock* JXL_RESTRICT get_block,
                       GroupDecCache* JXL_RESTRICT group_dec_cache,
                       PassesDecoderState* JXL_RESTRICT dec_state,
                       size_t thread, size_t group_idx,
                       RenderPipelineInput& render_pipeline_input,
                       jpeg::JPEGData* jpeg_data,
                       const JpegGroupParams* JXL_RESTRICT jpeg_params) {
  // TODO(veluca): investigate cache usage in this function.
  const Rect block_rect =
      dec_state->shared->frame_dim.BlockGroupRect(group_idx);
  const AcStrategyImage& ac_strategy = dec_state->shared->ac_strategy;

  const size_t xsize_blocks = block_rect.xsize();
  const size_t ysize_blocks = block_rect.ysize();

  const size_t dc_stride = dec_state->shared->dc->PixelsPerRow();

  const float inv_global_scale = dec_state->shared->quantizer.InvGlobalScale();

  const YCbCrChromaSubsampling& cs = frame_header.chroma_subsampling;

  const auto kJpegDctMin = Set(di16_full, -4095);
  const auto kJpegDctMax = Set(di16_full, 4095);

  size_t idct_stride[3];
  for (size_t c = 0; c < 3; c++) {
    idct_stride[c] = render_pipeline_input.GetBuffer(c).first->PixelsPerRow();
  }

  ACType ac_type = dec_state->coefficients->Type();
  auto dequant_block = ac_type == ACType::k16 ? DequantBlock<ACType::k16>
                                              : DequantBlock<ACType::k32>;
  // Whether or not coefficients should be stored for future usage, and/or read
  // from past usage.
  bool accumulate = !dec_state->coefficients->IsEmpty();
  // Offset of the current block in the group.
  size_t offset = 0;

  // jpeg_params is non-null iff jpeg_data is non-null; params were computed
  // once per frame in DecodeGroup / DecodeGroupForRoundtrip.
  const ColorCorrelation& color_correlation = dec_state->shared->cmap.base();

  size_t hshift[3] = {cs.HShift(0), cs.HShift(1), cs.HShift(2)};
  size_t vshift[3] = {cs.VShift(0), cs.VShift(1), cs.VShift(2)};
  Rect r[3];
  for (size_t i = 0; i < 3; i++) {
    r[i] =
        Rect(block_rect.x0() >> hshift[i], block_rect.y0() >> vshift[i],
             block_rect.xsize() >> hshift[i], block_rect.ysize() >> vshift[i]);
    if (!r[i].IsInside({0, 0, dec_state->shared->dc->Plane(i).xsize(),
                        dec_state->shared->dc->Plane(i).ysize()})) {
      return JXL_FAILURE("Frame dimensions are too big for the image.");
    }
  }

  for (size_t by = 0; by < ysize_blocks; ++by) {
    if constexpr (kReadCoefficients) {
      get_block->StartRow(by);
    }
    size_t sby[3] = {by >> vshift[0], by >> vshift[1], by >> vshift[2]};

    const int32_t* JXL_RESTRICT row_quant =
        block_rect.ConstRow(dec_state->shared->raw_quant_field, by);

    const float* JXL_RESTRICT dc_rows[3] = {
        r[0].ConstPlaneRow(*dec_state->shared->dc, 0, sby[0]),
        r[1].ConstPlaneRow(*dec_state->shared->dc, 1, sby[1]),
        r[2].ConstPlaneRow(*dec_state->shared->dc, 2, sby[2]),
    };

    const size_t ty = (block_rect.y0() + by) / kColorTileDimInBlocks;
    AcStrategyRow acs_row = ac_strategy.ConstRow(block_rect, by);

    const int8_t* JXL_RESTRICT row_cmap[3] = {
        dec_state->shared->cmap.ytox_map.ConstRow(ty),
        nullptr,
        dec_state->shared->cmap.ytob_map.ConstRow(ty),
    };

    float* JXL_RESTRICT idct_row[3];
    int16_t* JXL_RESTRICT jpeg_row[3];
    for (size_t c = 0; c < 3; c++) {
      const auto& buffer = render_pipeline_input.GetBuffer(c);
      idct_row[c] = buffer.second.Row(buffer.first, sby[c] * kBlockDim);
      if (jpeg_data) {
        auto& component = jpeg_data->components[jpeg_params->jpeg_c_map[c]];
        jpeg_row[c] =
            component.coeffs.data() +
            (component.width_in_blocks * (r[c].y0() + sby[c]) + r[c].x0()) *
                kDCTBlockSize;
      }
    }

    size_t bx = 0;
    for (size_t tx = 0; tx < DivCeil(xsize_blocks, kColorTileDimInBlocks);
         tx++) {
      size_t abs_tx = tx + block_rect.x0() / kColorTileDimInBlocks;
      const float x_cc_mul_s = color_correlation.YtoXRatio(row_cmap[0][abs_tx]);
      const float b_cc_mul_s = color_correlation.YtoBRatio(row_cmap[2][abs_tx]);
      auto x_cc_mul = Set(d, x_cc_mul_s);
      auto b_cc_mul = Set(d, b_cc_mul_s);
      const bool no_cfl = (x_cc_mul_s == 0.0f && b_cc_mul_s == 0.0f);
      // Increment bx by llf_x because those iterations would otherwise
      // immediately continue (!IsFirstBlock). Reduces mispredictions.
      for (; bx < xsize_blocks && bx < (tx + 1) * kColorTileDimInBlocks;) {
        size_t sbx[3] = {bx >> hshift[0], bx >> hshift[1], bx >> hshift[2]};
        AcStrategy acs = acs_row[bx];
        const size_t llf_x = acs.covered_blocks_x();

        // Can only happen in the second or lower rows of a varblock.
        if (JXL_UNLIKELY(!acs.IsFirstBlock())) {
          bx += llf_x;
          continue;
        }
        const size_t log2_covered_blocks = acs.log2_covered_blocks();

        const size_t covered_blocks = 1 << log2_covered_blocks;
        const size_t size = covered_blocks * kDCTBlockSize;

        ACPtr qblock[3];
        if (accumulate) {
          for (size_t c = 0; c < 3; c++) {
            qblock[c] = dec_state->coefficients->PlaneRow(c, group_idx, offset);
          }
        } else {
          if (ac_type == ACType::k16) {
            memset(group_dec_cache->dec_group_qblock16, 0,
                   size * 3 * sizeof(int16_t));
            for (size_t c = 0; c < 3; c++) {
              qblock[c].ptr16 = group_dec_cache->dec_group_qblock16 + c * size;
            }
          } else {
            memset(group_dec_cache->dec_group_qblock, 0,
                   size * 3 * sizeof(int32_t));
            for (size_t c = 0; c < 3; c++) {
              qblock[c].ptr32 = group_dec_cache->dec_group_qblock + c * size;
            }
          }
        }
        if constexpr (kReadCoefficients) {
          JXL_RETURN_IF_ERROR(get_block->LoadBlock(
              bx, by, acs, size, log2_covered_blocks, qblock, ac_type));
        }
        offset += size;

        // DC-only fast path: detect channels where all AC coefficients are zero.
        // Fires on ~89% X, ~86% B, and ~2-49% Y blocks in real photos.
        bool dc_only[3] = {false, false, false};
        if (JXL_LIKELY(!jpeg_data) && JXL_LIKELY(covered_blocks == 1)) {
          for (size_t c = 0; c < 3; c++) {
            if (c != 1 && !no_cfl) continue;
            bool all_zero = true;
            if (ac_type == ACType::k16) {
              const int16_t* JXL_RESTRICT p = qblock[c].ptr16;
              for (size_t k = 1; k < size; k++) {
                if (p[k]) { all_zero = false; break; }
              }
            } else {
              const int32_t* JXL_RESTRICT p = qblock[c].ptr32;
              for (size_t k = 1; k < size; k++) {
                if (p[k]) { all_zero = false; break; }
              }
            }
            dc_only[c] = all_zero;
          }
        }

        if (JXL_UNLIKELY(jpeg_data)) {
          if (acs.Strategy() != AcStrategyType::DCT) {
            return JXL_FAILURE(
                "Can only decode to JPEG if only DCT-8 is used.");
          }

          HWY_ALIGN int32_t transposed_dct_y[64];
          for (size_t c : {1, 0, 2}) {
            // Propagate only Y for grayscale.
            if (jpeg_params->is_gray && c != 1) {
              continue;
            }
            if ((sbx[c] << hshift[c] != bx) || (sby[c] << vshift[c] != by)) {
              continue;
            }
            int16_t* JXL_RESTRICT jpeg_pos =
                jpeg_row[c] + sbx[c] * kDCTBlockSize;
            // JPEG XL is transposed, JPEG is not.
            auto* transposed_dct = qblock[c].ptr32;
            Transpose8x8InPlace(transposed_dct);
            // No CfL - no need to store the y block converted to integers.
            if (!cs.Is444() ||
                (row_cmap[0][abs_tx] == 0 && row_cmap[2][abs_tx] == 0)) {
              for (size_t i = 0; i < 64; i += Lanes(d)) {
                const auto ini = Load(di, transposed_dct + i);
                const auto ini16 = DemoteTo(di16, ini);
                StoreU(ini16, di16, jpeg_pos + i);
              }
            } else if (c == 1) {
              // Y channel: save for restoring X/B, but nothing else to do.
              for (size_t i = 0; i < 64; i += Lanes(d)) {
                const auto ini = Load(di, transposed_dct + i);
                Store(ini, di, transposed_dct_y + i);
                const auto ini16 = DemoteTo(di16, ini);
                StoreU(ini16, di16, jpeg_pos + i);
              }
            } else {
              // transposed_dct_y contains the y channel block, transposed.
              const auto scale =
                  Set(di, ColorCorrelation::RatioJPEG(row_cmap[c][abs_tx]));
              const auto round = Set(di, 1 << (kCFLFixedPointPrecision - 1));
              for (int i = 0; i < 64; i += Lanes(d)) {
                auto in = Load(di, transposed_dct + i);
                auto in_y = Load(di, transposed_dct_y + i);
                auto qt = Load(di, jpeg_params->scaled_qtable + c * size + i);
                auto coeff_scale = ShiftRight<kCFLFixedPointPrecision>(
                    Add(Mul(qt, scale), round));
                auto cfl_factor = ShiftRight<kCFLFixedPointPrecision>(
                    Add(Mul(in_y, coeff_scale), round));
                StoreU(DemoteTo(di16, Add(in, cfl_factor)), di16, jpeg_pos + i);
              }
            }
            jpeg_pos[0] =
                Clamp1<float>(dc_rows[c][sbx[c]] - jpeg_params->dc_offset[c],
                              -2047, 2047);
            auto overflow = MaskFromVec(Set(di16_full, 0));
            auto underflow = MaskFromVec(Set(di16_full, 0));
            for (int i = 0; i < 64; i += Lanes(di16_full)) {
              auto in = LoadU(di16_full, jpeg_pos + i);
              overflow = Or(overflow, Gt(in, kJpegDctMax));
              underflow = Or(underflow, Lt(in, kJpegDctMin));
            }
            if (!AllFalse(di16_full, Or(overflow, underflow))) {
              return JXL_FAILURE("JPEG DCT coefficients out of range");
            }
          }
        } else {
          HWY_ALIGN float* const block = group_dec_cache->dec_group_block;
          // Skip dequant entirely when all three channels are DC-only.
          if (JXL_LIKELY(!(dc_only[0] && dc_only[1] && dc_only[2]))) {
            dequant_block(
                inv_global_scale, row_quant[bx], dec_state->x_dm_multiplier,
                dec_state->b_dm_multiplier, x_cc_mul, b_cc_mul, acs.Strategy(),
                size, dec_state->shared->quantizer,
                acs.covered_blocks_y() * acs.covered_blocks_x(), sbx, dc_rows,
                dc_stride,
                dec_state->output_encoding_info.opsin_params.quant_biases, qblock,
                block, group_dec_cache->scratch_space);
          }

          for (size_t c : {1, 0, 2}) {
            if ((sbx[c] << hshift[c] != bx) || (sby[c] << vshift[c] != by)) {
              continue;
            }
            float* JXL_RESTRICT idct_pos = idct_row[c] + sbx[c] * kBlockDim;
            if (dc_only[c]) {
              // IDCT([DC,0,...]) = DC everywhere: fill directly, skip IDCT.
              const float dc_val = dc_rows[c][sbx[c]];
              for (size_t y = 0; y < kBlockDim; y++) {
                float* JXL_RESTRICT r = idct_pos + y * idct_stride[c];
                for (size_t x = 0; x < kBlockDim; x++) r[x] = dc_val;
              }
            } else {
              TransformToPixels(acs.Strategy(), block + c * size, idct_pos,
                                idct_stride[c], group_dec_cache->scratch_space);
            }
          }

#ifdef JXL_DEC_TRANSFORM_STATS
          {
            auto& st = jxl::DecTransformStats::Get();
            st.block_count.fetch_add(1, std::memory_order_relaxed);

            // Strategy histogram.
            const size_t strat_idx =
                static_cast<size_t>(acs.Strategy()) % DecTransformStats::kMaxStrats;
            st.strat_blocks[strat_idx].fetch_add(1, std::memory_order_relaxed);
            st.strat_coeffs[strat_idx].fetch_add(size, std::memory_order_relaxed);

            // Active component mask (Y always active; track X and B).
            int cmask = 0;
            if ((sbx[0] << hshift[0] == bx) && (sby[0] << vshift[0] == by)) cmask |= 1;
            if ((sbx[2] << hshift[2] == bx) && (sby[2] << vshift[2] == by)) cmask |= 2;
            st.comp_mask[cmask].fetch_add(1, std::memory_order_relaxed);

            // AC nzeros per channel (skip LLF slots [0..covered_blocks)).
            for (size_t c = 0; c < 3; c++) {
              size_t nz = 0;
              if (ac_type == ACType::k16) {
                for (size_t k = covered_blocks; k < size; k++)
                  if (qblock[c].ptr16[k]) nz++;
              } else {
                for (size_t k = covered_blocks; k < size; k++)
                  if (qblock[c].ptr32[k]) nz++;
              }
              st.nz_hist[c][DecTransformStats::NzBucket(nz)]
                  .fetch_add(1, std::memory_order_relaxed);
            }

            // CfL mode (nonzero cmap entry = active).
            int cfl_mask = (row_cmap[0][abs_tx] != 0 ? 1 : 0) |
                           (row_cmap[2][abs_tx] != 0 ? 2 : 0);
            st.cfl_mode[cfl_mask].fetch_add(1, std::memory_order_relaxed);

            // Pass mode.
            if (accumulate) {
              st.accumulate_blocks.fetch_add(1, std::memory_order_relaxed);
            } else {
              st.single_pass_blocks.fetch_add(1, std::memory_order_relaxed);
            }

            // Dequant work ledger: always dequant 3 channels × size coeffs.
            st.slots_total.fetch_add(3 * size, std::memory_order_relaxed);
            for (size_t c = 0; c < 3; c++) {
              if ((sbx[c] << hshift[c] == bx) && (sby[c] << vshift[c] == by)) {
                st.slots_rendered.fetch_add(size, std::memory_order_relaxed);
              } else {
                st.slots_inactive.fetch_add(size, std::memory_order_relaxed);
              }
            }
          }
#endif  // JXL_DEC_TRANSFORM_STATS
        }
        bx += llf_x;
      }
    }
  }
  return true;
}

// Keep the bitstream and stored-coefficient paths as separate Highway
// dispatch targets. The latter has no GetBlock argument so it cannot
// accidentally enter the entropy path during a progressive redraw.
Status DecodeGroupFromBitstream(
    const FrameHeader& frame_header, GetBlock* JXL_RESTRICT get_block,
    GroupDecCache* JXL_RESTRICT group_dec_cache,
    PassesDecoderState* JXL_RESTRICT dec_state, size_t thread,
    size_t group_idx, RenderPipelineInput& render_pipeline_input,
    jpeg::JPEGData* jpeg_data,
    const JpegGroupParams* JXL_RESTRICT jpeg_params) {
  return DecodeGroupImpl<true>(
      frame_header, get_block, group_dec_cache, dec_state, thread, group_idx,
      render_pipeline_input, jpeg_data, jpeg_params);
}

Status DecodeGroupFromStoredCoefficients(
    const FrameHeader& frame_header,
    GroupDecCache* JXL_RESTRICT group_dec_cache,
    PassesDecoderState* JXL_RESTRICT dec_state, size_t thread,
    size_t group_idx, RenderPipelineInput& render_pipeline_input,
    jpeg::JPEGData* jpeg_data,
    const JpegGroupParams* JXL_RESTRICT jpeg_params) {
  JXL_ENSURE(!dec_state->coefficients->IsEmpty());
  return DecodeGroupImpl<false>(
      frame_header, nullptr, group_dec_cache, dec_state, thread, group_idx,
      render_pipeline_input, jpeg_data, jpeg_params);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
namespace {
// Decode quantized AC coefficients of DCT blocks.
// LLF components in the output block will not be modified.
template <ACType ac_type, bool uses_lz77>
Status DecodeACVarBlock(size_t ctx_offset, size_t log2_covered_blocks,
                        uint8_t* JXL_RESTRICT row_nzeros,
                        const uint8_t* JXL_RESTRICT row_nzeros_top,
                        size_t nzeros_stride, size_t c, size_t bx, size_t by,
                        size_t lbx, AcStrategy acs,
                        const coeff_order_t* JXL_RESTRICT coeff_order,
                        BitReader* JXL_RESTRICT br,
                        ANSSymbolReader* JXL_RESTRICT decoder,
                        const std::vector<uint8_t>& context_map,
                        const uint8_t* qdc_row, const int32_t* qf_row,
                        const BlockCtxMap& block_ctx_map, ACPtr block,
                        size_t shift = 0) {
  // Equal to number of LLF coefficients.
  const size_t covered_blocks = 1 << log2_covered_blocks;
  const size_t size = covered_blocks * kDCTBlockSize;
  int32_t predicted_nzeros =
      PredictFromTopAndLeft(row_nzeros_top, row_nzeros, bx, 32);

  size_t ord = kStrategyOrder[acs.RawStrategy()];
  const coeff_order_t* JXL_RESTRICT order =
      &coeff_order[CoeffOrderOffset(ord, c)];

  size_t block_ctx = block_ctx_map.Context(qdc_row[lbx], qf_row[bx], ord, c);
  const int32_t nzero_ctx =
      block_ctx_map.NonZeroContext(predicted_nzeros, block_ctx) + ctx_offset;

  size_t nzeros =
      decoder->ReadHybridUintInlined<uses_lz77>(nzero_ctx, br, context_map);
  if (nzeros > size - covered_blocks) {
    return JXL_FAILURE("Invalid AC: nzeros %" PRIuS " too large for %" PRIuS
                       " 8x8 blocks",
                       nzeros, covered_blocks);
  }
  // (nzeros + covered_blocks - 1) >> log2_covered_blocks averages the AC nonzero
  // count over the covered blocks; nzeros <= size - covered_blocks bounds it to
  // [0, 63], so it always fits in the uint8_t plane.
  const uint8_t avg_nzeros =
      static_cast<uint8_t>((nzeros + covered_blocks - 1) >> log2_covered_blocks);
  JXL_DASSERT(((nzeros + covered_blocks - 1) >> log2_covered_blocks) <= 63);
  for (size_t y = 0; y < acs.covered_blocks_y(); y++) {
    for (size_t x = 0; x < acs.covered_blocks_x(); x++) {
      row_nzeros[bx + x + y * nzeros_stride] = avg_nzeros;
    }
  }

  const size_t histo_offset =
      ctx_offset + block_ctx_map.ZeroDensityContextsOffset(block_ctx);

  size_t prev = (nzeros > size / 16 ? 0 : 1);
  if (JXL_LIKELY(nzeros != 0)) {
    for (size_t k = covered_blocks; k < size && nzeros != 0; ++k) {
      const size_t ctx =
          histo_offset + ZeroDensityContext(nzeros, k, covered_blocks,
                                            log2_covered_blocks, prev);
      const size_t u_coeff =
          decoder->ReadHybridUintInlined<uses_lz77>(ctx, br, context_map);
      // Hand-rolled version of UnpackSigned, shifting before the conversion to
      // signed integer to avoid undefined behavior of shifting negative numbers.
      const size_t magnitude = u_coeff >> 1;
      const size_t neg_sign = (~u_coeff) & 1;
      const ptrdiff_t coeff =
          static_cast<ptrdiff_t>((magnitude ^ (neg_sign - 1)) << shift);
      if (ac_type == ACType::k16) {
        block.ptr16[order[k]] += coeff;
      } else {
        block.ptr32[order[k]] += coeff;
      }
      prev = static_cast<size_t>(u_coeff != 0);
      nzeros -= prev;
    }
  }
  if (JXL_UNLIKELY(nzeros != 0)) {
    return JXL_FAILURE("Invalid AC: nzeros at end of block is %" PRIuS
                       ", should be 0. Block (%" PRIuS ", %" PRIuS
                       "), channel %" PRIuS,
                       nzeros, bx, by, c);
  }

  return true;
}

// Structs used by DecodeGroupImpl to get a quantized block.
// GetBlockFromBitstream uses ANS decoding (and thus keeps track of row
// pointers in row_nzeros), GetBlockFromEncoder simply reads the coefficient
// image provided by the encoder.

struct GetBlockFromBitstream : public GetBlock {
  void StartRow(size_t by) override {
    qf_row = rect.ConstRow(*qf, by);
    quant_dc_row = quant_dc->ConstRow(rect.y0() + by) + rect.x0();
    for (size_t c = 0; c < 3; ++c) {
      const size_t sby = by >> vshift[c];
      for (size_t i = 0; i < num_passes; ++i) {
        row_nzeros[i][c] = group_dec_cache->num_nzeroes[i].PlaneRow(c, sby);
        row_nzeros_top[i][c] =
            sby == 0
                ? nullptr
                : group_dec_cache->num_nzeroes[i].ConstPlaneRow(c, sby - 1);
      }
    }
  }

  Status LoadBlock(size_t bx, size_t by, const AcStrategy& acs, size_t size,
                   size_t log2_covered_blocks, ACPtr block[3],
                   ACType ac_type) override {
    for (size_t c : {1, 0, 2}) {
      size_t sbx = bx >> hshift[c];
      size_t sby = by >> vshift[c];
      if (JXL_UNLIKELY((sbx << hshift[c] != bx) || (sby << vshift[c] != by))) {
        continue;
      }

      for (size_t pass = 0; pass < num_passes; pass++) {
        auto decode_ac_varblock =
            decoders[pass].UsesLZ77()
                ? (ac_type == ACType::k16 ? DecodeACVarBlock<ACType::k16, 1>
                                          : DecodeACVarBlock<ACType::k32, 1>)
                : (ac_type == ACType::k16 ? DecodeACVarBlock<ACType::k16, 0>
                                          : DecodeACVarBlock<ACType::k32, 0>);
        JXL_RETURN_IF_ERROR(decode_ac_varblock(
            ctx_offset[pass], log2_covered_blocks, row_nzeros[pass][c],
            row_nzeros_top[pass][c], nzeros_stride, c, sbx, sby, bx, acs,
            &coeff_orders[pass * coeff_order_size], readers[pass],
            &decoders[pass], context_map[pass], quant_dc_row, qf_row,
            *block_ctx_map, block[c], shift_for_pass[pass]));
      }
    }
    return true;
  }

  Status Init(const FrameHeader& frame_header,
              BitReader* JXL_RESTRICT* JXL_RESTRICT readers_,
              size_t num_passes_, size_t group_idx, size_t histo_selector_bits,
              const Rect& rect_, GroupDecCache* JXL_RESTRICT group_dec_cache_,
              PassesDecoderState* dec_state, size_t first_pass) {
    for (size_t i = 0; i < 3; i++) {
      hshift[i] = frame_header.chroma_subsampling.HShift(i);
      vshift[i] = frame_header.chroma_subsampling.VShift(i);
    }
    coeff_order_size = dec_state->shared->coeff_order_size;
    coeff_orders =
        dec_state->shared->coeff_orders.data() + first_pass * coeff_order_size;
    context_map = dec_state->context_map.data() + first_pass;
    readers = readers_;
    num_passes = num_passes_;
    shift_for_pass = frame_header.passes.shift + first_pass;
    group_dec_cache = group_dec_cache_;
    rect = rect_;
    block_ctx_map = &dec_state->shared->block_ctx_map;
    qf = &dec_state->shared->raw_quant_field;
    quant_dc = &dec_state->shared->quant_dc;

    for (size_t pass = 0; pass < num_passes; pass++) {
      // Select which histogram set to use among those of the current pass.
      size_t cur_histogram = 0;
      if (histo_selector_bits != 0) {
        cur_histogram = readers[pass]->ReadBits(histo_selector_bits);
      }
      if (cur_histogram >= dec_state->shared->num_histograms) {
        return JXL_FAILURE("Invalid histogram selector");
      }
      ctx_offset[pass] = cur_histogram * block_ctx_map->NumACContexts();

      JXL_ASSIGN_OR_RETURN(
          decoders[pass],
          ANSSymbolReader::Create(&dec_state->code[pass + first_pass],
                                  readers[pass]));
    }
    nzeros_stride = group_dec_cache->num_nzeroes[0].PixelsPerRow();
    for (size_t i = 0; i < num_passes; i++) {
      JXL_ENSURE(
          nzeros_stride ==
          static_cast<size_t>(group_dec_cache->num_nzeroes[i].PixelsPerRow()));
    }
    return true;
  }

  const uint32_t* shift_for_pass = nullptr;  // not owned
  const coeff_order_t* JXL_RESTRICT coeff_orders;
  size_t coeff_order_size;
  const std::vector<uint8_t>* JXL_RESTRICT context_map;
  ANSSymbolReader decoders[kMaxNumPasses];
  BitReader* JXL_RESTRICT* JXL_RESTRICT readers;
  size_t num_passes;
  size_t ctx_offset[kMaxNumPasses];
  size_t nzeros_stride;
  uint8_t* JXL_RESTRICT row_nzeros[kMaxNumPasses][3];
  const uint8_t* JXL_RESTRICT row_nzeros_top[kMaxNumPasses][3];
  GroupDecCache* JXL_RESTRICT group_dec_cache;
  const BlockCtxMap* block_ctx_map;
  const ImageI* qf;
  const ImageB* quant_dc;
  const int32_t* qf_row;
  const uint8_t* quant_dc_row;
  Rect rect;
  size_t hshift[3], vshift[3];
};

struct GetBlockFromEncoder : public GetBlock {
  void StartRow(size_t by) override {}

  Status LoadBlock(size_t bx, size_t by, const AcStrategy& acs, size_t size,
                   size_t log2_covered_blocks, ACPtr block[3],
                   ACType ac_type) override {
    JXL_ENSURE(ac_type == ACType::k32);
    for (size_t c = 0; c < 3; ++c) {
      for (size_t i = 0; i < quantized_ac->size(); ++i) {
        int32_t* JXL_RESTRICT dst = block[c].ptr32;
        const int32_t* JXL_RESTRICT src = rows[i][c] + offset;
        const uint32_t shift = shift_for_pass[i];
        if (shift == 0) {
          // TODO(veluca): SIMD.
          for (size_t k = 0; k < size; ++k) dst[k] += src[k];
        } else {
          const int32_t scale = int32_t{1} << shift;
          for (size_t k = 0; k < size; ++k) dst[k] += src[k] * scale;
        }
      }
    }
    offset += size;
    return true;
  }

  static StatusOr<GetBlockFromEncoder> Create(
      const std::vector<std::unique_ptr<ACImage>>& ac, size_t group_idx,
      const uint32_t* shift_for_pass) {
    GetBlockFromEncoder result(ac, group_idx, shift_for_pass);
    // TODO(veluca): not supported with chroma subsampling.
    for (size_t i = 0; i < ac.size(); i++) {
      JXL_ENSURE(ac[i]->Type() == ACType::k32);
      for (size_t c = 0; c < 3; c++) {
        result.rows[i][c] = ac[i]->PlaneRow(c, group_idx, 0).ptr32;
      }
    }
    return result;
  }

  const std::vector<std::unique_ptr<ACImage>>* JXL_RESTRICT quantized_ac;
  size_t offset = 0;
  const int32_t* JXL_RESTRICT rows[kMaxNumPasses][3];
  const uint32_t* shift_for_pass = nullptr;  // not owned

 private:
  GetBlockFromEncoder(const std::vector<std::unique_ptr<ACImage>>& ac,
                      size_t group_idx, const uint32_t* shift_for_pass)
      : quantized_ac(&ac), shift_for_pass(shift_for_pass) {}
};

HWY_EXPORT(DecodeGroupFromBitstream);
HWY_EXPORT(DecodeGroupFromStoredCoefficients);

}  // namespace

// Validate and pre-compute per-frame JPEG reconstruction parameters.
// Called once per frame (not per group) from DecodeGroup /
// DecodeGroupForRoundtrip.
StatusOr<JpegGroupParams> PrepareJpegGroupParams(
    const FrameHeader& frame_header, const PassesDecoderState& dec_state,
    const jpeg::JPEGData& jpeg_data) {
  JpegGroupParams params;
  const ColorCorrelation& cmap = dec_state.shared->cmap.base();
  if (!cmap.IsJPEGCompatible()) {
    return JXL_FAILURE("The CfL map is not JPEG-compatible");
  }
  params.is_gray = (jpeg_data.components.size() == 1);
  JXL_ENSURE(frame_header.color_transform != ColorTransform::kXYB);
  params.jpeg_c_map = JpegOrder(frame_header.color_transform, params.is_gray);
  const std::vector<QuantEncoding>& qe =
      dec_state.shared->matrices.encodings();
  if (qe.empty() || qe[0].mode != QuantEncoding::Mode::kQuantModeRAW ||
      std::abs(qe[0].qraw.qtable_den - 1.f / (8 * 255)) > 1e-8f) {
    return JXL_FAILURE("Quantization table is not a JPEG quantization table.");
  }
  JXL_ENSURE(qe[0].qraw.qtable->size() == 3 * 8 * 8);
  const int* qtable = qe[0].qraw.qtable->data();
  for (size_t c = 0; c < 3; ++c) {
    if (frame_header.color_transform == ColorTransform::kNone) {
      params.dc_offset[c] = 1024 / qtable[64 * c];
    }
    for (size_t i = 0; i < 64; ++i) {
      // Transpose the matrix, as it will be used on the transposed block.
      const int num = qtable[64 + i];
      const int den = qtable[64 * c + i];
      if (num <= 0 || den <= 0 || num >= 65536 || den >= 65536) {
        return JXL_FAILURE("Invalid JPEG quantization table");
      }
      params.scaled_qtable[64 * c + (i % 8) * 8 + (i / 8)] =
          (1 << kCFLFixedPointPrecision) * num / den;
    }
  }
  return params;
}

// Coefficient-only decode for kDontDraw passes: skips all render-path setup
// (DC rows, chroma tables, IDCT buffers, JPEG params) and goes straight to
// entropy-decoding into the coefficient store.
Status DecodeGroupNoDraw(GetBlock* JXL_RESTRICT get_block, size_t group_idx,
                         PassesDecoderState* JXL_RESTRICT dec_state) {
  JXL_ENSURE(!dec_state->coefficients->IsEmpty());
  const Rect block_rect =
      dec_state->shared->frame_dim.BlockGroupRect(group_idx);
  const AcStrategyImage& ac_strategy = dec_state->shared->ac_strategy;
  const size_t xsize_blocks = block_rect.xsize();
  const size_t ysize_blocks = block_rect.ysize();
  const ACType ac_type = dec_state->coefficients->Type();
  size_t offset = 0;

  for (size_t by = 0; by < ysize_blocks; ++by) {
    get_block->StartRow(by);
    const AcStrategyRow acs_row = ac_strategy.ConstRow(block_rect, by);
    for (size_t bx = 0; bx < xsize_blocks;) {
      const AcStrategy acs = acs_row[bx];
      const size_t llf_x = acs.covered_blocks_x();
      if (JXL_UNLIKELY(!acs.IsFirstBlock())) {
        bx += llf_x;
        continue;
      }
      const size_t log2_covered_blocks = acs.log2_covered_blocks();
      const size_t size = (size_t{1} << log2_covered_blocks) * kDCTBlockSize;
      ACPtr qblock[3];
      for (size_t c = 0; c < 3; ++c) {
        qblock[c] = dec_state->coefficients->PlaneRow(c, group_idx, offset);
      }
      JXL_RETURN_IF_ERROR(get_block->LoadBlock(
          bx, by, acs, size, log2_covered_blocks, qblock, ac_type));
      offset += size;
      bx += llf_x;
    }
  }
  return true;
}

Status DecodeGroup(const FrameHeader& frame_header,
                   BitReader* JXL_RESTRICT* JXL_RESTRICT readers,
                   size_t num_passes, size_t group_idx,
                   PassesDecoderState* JXL_RESTRICT dec_state,
                   GroupDecCache* JXL_RESTRICT group_dec_cache, size_t thread,
                   RenderPipelineInput& render_pipeline_input,
                   jpeg::JPEGData* JXL_RESTRICT jpeg_data, size_t first_pass,
                   bool force_draw, bool dc_only, bool* should_run_pipeline) {
  JxlMemoryManager* memory_manager = dec_state->memory_manager();
  const DrawMode draw =
      (num_passes + first_pass == frame_header.passes.num_passes) || force_draw
          ? kDraw
          : kDontDraw;

  if (should_run_pipeline) {
    *should_run_pipeline = draw != kDontDraw;
  }

  if (draw == kDraw && num_passes == 0 && first_pass == 0) {
    JXL_RETURN_IF_ERROR(group_dec_cache->InitDCBufferOnce(memory_manager));
    const YCbCrChromaSubsampling& cs = frame_header.chroma_subsampling;
    // Allocate RowInfo once for all channels; sizes are channel-invariant.
    RenderPipelineStage::RowInfo input_rows(1, std::vector<float*>(5));
    RenderPipelineStage::RowInfo output_rows(1, std::vector<float*>(8));
    for (size_t c : {0, 1, 2}) {
      const size_t hs = cs.HShift(c);
      const size_t vs = cs.VShift(c);
      // We reuse filter_input_storage here as it is not currently in use.
      const Rect src_rect_precs =
          dec_state->shared->frame_dim.BlockGroupRect(group_idx);
      const Rect src_rect =
          Rect(src_rect_precs.x0() >> hs, src_rect_precs.y0() >> vs,
               src_rect_precs.xsize() >> hs, src_rect_precs.ysize() >> vs);
      const Rect copy_rect(kRenderPipelineXOffset, 2, src_rect.xsize(),
                           src_rect.ysize());
      JXL_RETURN_IF_ERROR(
          CopyImageToWithPadding(src_rect, dec_state->shared->dc->Plane(c), 2,
                                 copy_rect, &group_dec_cache->dc_buffer));
      // Mirrorpad. Interleaving left and right padding ensures that padding
      // works out correctly even for images with DC size of 1.
      for (size_t y = 0; y < src_rect.ysize() + 4; ++y) {
        const size_t xend = kRenderPipelineXOffset +
                            (dec_state->shared->dc->Plane(c).xsize() >> hs) -
                            src_rect.x0();
        for (size_t ix = 0; ix < 2; ++ix) {
          if (src_rect.x0() == 0) {
            group_dec_cache->dc_buffer.Row(y)[kRenderPipelineXOffset - ix - 1] =
                group_dec_cache->dc_buffer.Row(y)[kRenderPipelineXOffset + ix];
          }
          if (src_rect.x0() + src_rect.xsize() + 2 >=
              (dec_state->shared->dc->xsize() >> hs)) {
            group_dec_cache->dc_buffer.Row(y)[xend + ix] =
                group_dec_cache->dc_buffer.Row(y)[xend - ix - 1];
          }
        }
      }
      const auto& buffer = render_pipeline_input.GetBuffer(c);
      const Rect dst_rect = buffer.second;
      ImageF* upsampling_dst = buffer.first;
      JXL_ENSURE(dst_rect.IsInside(*upsampling_dst));

      for (size_t y = src_rect.y0(); y < src_rect.y0() + src_rect.ysize();
           ++y) {
        for (ptrdiff_t iy = 0; iy < 5; ++iy) {
          input_rows[0][iy] = group_dec_cache->dc_buffer.Row(
              Mirror(static_cast<ptrdiff_t>(y) + iy - 2,
                     dec_state->shared->dc->Plane(c).ysize() >> vs) +
              2 - src_rect.y0());
        }
        for (size_t iy = 0; iy < 8; ++iy) {
          output_rows[0][iy] =
              dst_rect.Row(upsampling_dst, ((y - src_rect.y0()) << 3) + iy) -
              kRenderPipelineXOffset;
        }
        // Arguments set to 0/nullptr are not used.
        JXL_RETURN_IF_ERROR(dec_state->upsampler8x->ProcessRow(
            input_rows, output_rows, /*xextra_left=*/0, /*xextra_right=*/0,
            src_rect.xsize(), 0, 0, thread));
      }
    }
    return true;
  }

  size_t histo_selector_bits = 0;
  if (dc_only) {
    JXL_ENSURE(num_passes == 0);
  } else {
    JXL_ENSURE(dec_state->shared->num_histograms > 0);
    histo_selector_bits = CeilLog2Nonzero(dec_state->shared->num_histograms);
  }

  // A progressive redraw has no new entropy-coded passes. Coefficients were
  // retained by earlier no-draw or partial-draw calls, so bypass the entire
  // reader/histogram setup and render directly from that store.
  const bool render_from_stored_coefficients =
      draw == kDraw && num_passes == 0 && !dec_state->coefficients->IsEmpty();
  if (render_from_stored_coefficients) {
    const JpegGroupParams* jpeg_params_ptr = nullptr;
    JpegGroupParams jpeg_params;
    if (jpeg_data) {
      JXL_ASSIGN_OR_RETURN(
          jpeg_params,
          PrepareJpegGroupParams(frame_header, *dec_state, *jpeg_data));
      jpeg_params_ptr = &jpeg_params;
    }
    return HWY_DYNAMIC_DISPATCH(DecodeGroupFromStoredCoefficients)(
        frame_header, group_dec_cache, dec_state, thread, group_idx,
        render_pipeline_input, jpeg_data, jpeg_params_ptr);
  }

  // Stack-allocate: GetBlockFromBitstream has no ownership requirements and
  // outlives only this call frame.
  GetBlockFromBitstream get_block;
  JXL_RETURN_IF_ERROR(get_block.Init(
      frame_header, readers, num_passes, group_idx, histo_selector_bits,
      dec_state->shared->frame_dim.BlockGroupRect(group_idx), group_dec_cache,
      dec_state, first_pass));

  if (draw == kDontDraw) {
    JXL_RETURN_IF_ERROR(DecodeGroupNoDraw(&get_block, group_idx, dec_state));
  } else {
    // Compute JPEG params once per frame (not per group); null when no JPEG.
    const JpegGroupParams* jpeg_params_ptr = nullptr;
    JpegGroupParams jpeg_params;
    if (jpeg_data) {
      JXL_ASSIGN_OR_RETURN(jpeg_params,
                           PrepareJpegGroupParams(frame_header, *dec_state,
                                                  *jpeg_data));
      jpeg_params_ptr = &jpeg_params;
    }
    JXL_RETURN_IF_ERROR(HWY_DYNAMIC_DISPATCH(DecodeGroupFromBitstream)(
        frame_header, &get_block, group_dec_cache, dec_state, thread, group_idx,
        render_pipeline_input, jpeg_data, jpeg_params_ptr));
  }

  for (size_t pass = 0; pass < num_passes; ++pass) {
    if (!get_block.decoders[pass].CheckANSFinalState()) {
      return JXL_FAILURE("ANS checksum failure.");
    }
  }
  return true;
}

Status DecodeGroupForRoundtrip(const FrameHeader& frame_header,
                               const std::vector<std::unique_ptr<ACImage>>& ac,
                               size_t group_idx,
                               PassesDecoderState* JXL_RESTRICT dec_state,
                               GroupDecCache* JXL_RESTRICT group_dec_cache,
                               size_t thread,
                               RenderPipelineInput& render_pipeline_input,
                               jpeg::JPEGData* JXL_RESTRICT jpeg_data,
                               AuxOut* aux_out) {
  JxlMemoryManager* memory_manager = dec_state->memory_manager();
  JXL_ASSIGN_OR_RETURN(
      GetBlockFromEncoder get_block,
      GetBlockFromEncoder::Create(ac, group_idx, frame_header.passes.shift));
  // Encoder-recon path may use any strategy; pass the known maximum area
  // directly (AcStrategy::kMaxCoeffArea = DCT256 = 65536 coefficients) to
  // avoid a full 27-strategy scan inside InitOnce.
  JXL_RETURN_IF_ERROR(group_dec_cache->InitOnce(
      memory_manager,
      /*num_passes=*/0,
      /*max_block_area=*/AcStrategy::kMaxCoeffArea));

  const JpegGroupParams* jpeg_params_ptr = nullptr;
  JpegGroupParams jpeg_params;
  if (jpeg_data) {
    JXL_ASSIGN_OR_RETURN(
        jpeg_params,
        PrepareJpegGroupParams(frame_header, *dec_state, *jpeg_data));
    jpeg_params_ptr = &jpeg_params;
  }
  return HWY_DYNAMIC_DISPATCH(DecodeGroupFromBitstream)(
      frame_header, &get_block, group_dec_cache, dec_state, thread, group_idx,
      render_pipeline_input, jpeg_data, jpeg_params_ptr);
}

}  // namespace jxl
#endif  // HWY_ONCE
