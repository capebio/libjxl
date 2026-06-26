// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_chroma_from_luma.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <hwy/base.h>  // HWY_ALIGN_MAX
#include <limits>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"
#include "lib/jxl/quant_weights.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_chroma_from_luma.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/cms/opsin_params.h"
#include "lib/jxl/dec_transforms-inl.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_transforms-inl.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/simd_util.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::GetLane;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::Lt;

static HWY_FULL(float) df;

struct CFLFunction {
  static constexpr float kCoeff = 1.f / 3;
  static constexpr float kThres = 100.0f;
  // values_a/values_b are the precomputed residual coefficients: the color
  // residual is values_a[i] * x + values_b[i], with
  //   values_a = luma / color_factor,  values_b = base * luma - chroma.
  // Folding a/b into the producer loop keeps the 20 Newton iterations a pure
  // load (no per-iteration Mul/Sub re-derivation) and is bit-identical to the
  // previous in-solver recomputation.
  CFLFunction(const float* values_a, const float* values_b, size_t num,
              float distance_mul)
      : values_a(values_a),
        values_b(values_b),
        num(num),
        distance_mul(distance_mul) {
    JXL_DASSERT(num % Lanes(df) == 0);
  }

  // Returns f'(x), where f is 1/3 * sum ((|color residual| + 1)^2-1) +
  // distance_mul * x^2 * num.
  float Compute(float x, float eps, float* fpeps, float* fmeps) const {
    float first_derivative = 2 * distance_mul * num * x;
    float first_derivative_peps = 2 * distance_mul * num * (x + eps);
    float first_derivative_meps = 2 * distance_mul * num * (x - eps);

    const auto thres = Set(df, kThres);
    const auto coeffx2 = Set(df, kCoeff * 2.0f);
    const auto one = Set(df, 1.0f);
    const auto zero = Set(df, 0.0f);
    const auto x_v = Set(df, x);
    const auto xpe_v = Set(df, x + eps);
    const auto xme_v = Set(df, x - eps);
    auto fd_v = Zero(df);
    auto fdpe_v = Zero(df);
    auto fdme_v = Zero(df);

    for (size_t i = 0; i < num; i += Lanes(df)) {
      // color residual = ax + b
      const auto a = Load(df, values_a + i);
      const auto b = Load(df, values_b + i);
      const auto v = MulAdd(a, x_v, b);
      const auto vpe = MulAdd(a, xpe_v, b);
      const auto vme = MulAdd(a, xme_v, b);
      const auto av = Abs(v);
      const auto avpe = Abs(vpe);
      const auto avme = Abs(vme);
      const auto acoeffx2 = Mul(coeffx2, a);
      auto d = Mul(acoeffx2, Add(av, one));
      auto dpe = Mul(acoeffx2, Add(avpe, one));
      auto dme = Mul(acoeffx2, Add(avme, one));
      d = IfThenElse(Lt(v, zero), Sub(zero, d), d);
      dpe = IfThenElse(Lt(vpe, zero), Sub(zero, dpe), dpe);
      dme = IfThenElse(Lt(vme, zero), Sub(zero, dme), dme);
      const auto above = Ge(av, thres);
      // TODO(eustas): use IfThenElseZero
      fd_v = Add(fd_v, IfThenElse(above, zero, d));
      fdpe_v = Add(fdpe_v, IfThenElse(above, zero, dpe));
      fdme_v = Add(fdme_v, IfThenElse(above, zero, dme));
    }

    *fpeps = first_derivative_peps + GetLane(SumOfLanes(df, fdpe_v));
    *fmeps = first_derivative_meps + GetLane(SumOfLanes(df, fdme_v));
    return first_derivative + GetLane(SumOfLanes(df, fd_v));
  }

  const float* JXL_RESTRICT values_a;
  const float* JXL_RESTRICT values_b;
  size_t num;
  float distance_mul;
};

// Shrink the raw multiplier towards zero and clamp/round to the signed-byte
// range. Shared epilogue for both the fast and robust solvers.
int32_t QuantizeCflMultiplier(float x) {
  // CFL seems to be tricky for larger transforms for HF components
  // close to zero. This heuristic brings the solutions closer to zero
  // and reduces red-green oscillations. A better approach would
  // look into variance of the multiplier within separate (e.g. 8x8)
  // areas and only apply this heuristic where there is a high variance.
  // This would give about 1 % more compression density.
  float towards_zero = 2.6;
  if (x >= towards_zero) {
    x -= towards_zero;
  } else if (x <= -towards_zero) {
    x += towards_zero;
  } else {
    x = 0;
  }
  return jxl::Clamp1(std::round(x), -128.0f, 127.0f);
}

// Fast least-squares multiplier from the accumulated residual moments
// sum_a2 = sum(a^2) and sum_ab = sum(a*b). The moments are accumulated by the
// producer loop (ComputeTile), so the four per-tile coefficient streams never
// need to be materialised and re-read in the fast path.
int32_t FinishMultiplierFast(float sum_a2, float sum_ab, size_t num,
                             float distance_mul) {
  if (num == 0) {
    return 0;
  }
  // + distance_mul * x^2 * num
  float x = -sum_ab / (sum_a2 + num * distance_mul * 0.5f);
  return QuantizeCflMultiplier(x);
}

// Robust multiplier via up to 20 Newton iterations over the precomputed
// residual coefficients (values_a = luma / color_factor,
// values_b = base * luma - chroma).
int32_t FindBestMultiplierRobust(const float* values_a, const float* values_b,
                                 size_t num, float distance_mul) {
  if (num == 0) {
    return 0;
  }
  constexpr float eps = 100;
  constexpr float kClamp = 20.0f;
  CFLFunction fn(values_a, values_b, num, distance_mul);
  float x = 0;
  // Up to 20 Newton iterations, with approximate derivatives.
  // Derivatives are approximate due to the high amount of noise in the exact
  // derivatives.
  for (size_t i = 0; i < 20; i++) {
    float dfpeps;
    float dfmeps;
    float d_f = fn.Compute(x, eps, &dfpeps, &dfmeps);
    float ddf = (dfpeps - dfmeps) / (2 * eps);
    float kExperimentalInsignificantStabilizer = 0.85;
    float step = d_f / (ddf + kExperimentalInsignificantStabilizer);
    x -= std::min(kClamp, std::max(-kClamp, step));
    if (std::abs(step) < 3e-3) break;
  }
  return QuantizeCflMultiplier(x);
}

Status ComputeTile(const Image3F& opsin, const Rect& opsin_rect,
                   const DequantMatrices& dequant,
                   const AcStrategyImage* ac_strategy,
                   const ImageI* raw_quant_field, const Quantizer* quantizer,
                   const Rect& rect, bool fast, bool use_dct8, ImageSB* map_x,
                   ImageSB* map_b, Span<float> mem) {
  static_assert(kEncTileDimInBlocks == kColorTileDimInBlocks,
                "Invalid color tile dim");
  constexpr float kDistanceMultiplierAC = 1e-9f;
  const size_t dct_scratch_size =
      3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;

  const size_t y0 = rect.y0();
  const size_t x0 = rect.x0();
  const size_t x1 = rect.x0() + rect.xsize();
  const size_t y1 = rect.y0() + rect.ysize();
  const size_t stride = opsin.PixelsPerRow();

  int ty = y0 / kColorTileDimInBlocks;
  int tx = x0 / kColorTileDimInBlocks;

  int8_t* JXL_RESTRICT row_out_x = map_x->Row(ty);
  int8_t* JXL_RESTRICT row_out_b = map_b->Row(ty);

  // All are aligned.
  float* HWY_RESTRICT block_y = mem.begin();
  float* HWY_RESTRICT block_x = block_y + AcStrategy::kMaxCoeffArea;
  float* HWY_RESTRICT block_b = block_x + AcStrategy::kMaxCoeffArea;
  JXL_ENSURE(mem.remove_prefix(3 * AcStrategy::kMaxCoeffArea));
  // Per-tile residual coefficients in solver-native (a, b) form, where the
  // color residual is a*x + b. Only materialised on the robust path, which
  // re-reads them across up to 20 Newton iterations; the fast path accumulates
  // its moments directly and never touches these.
  float* HWY_RESTRICT coeffs_ax = mem.begin();
  float* HWY_RESTRICT coeffs_bx = coeffs_ax + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT coeffs_ab = coeffs_bx + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT coeffs_bb = coeffs_ab + kColorTileDim * kColorTileDim;
  JXL_ENSURE(mem.remove_prefix(4 * kColorTileDim * kColorTileDim));
  float* HWY_RESTRICT scratch_space = mem.begin();
  JXL_ENSURE(mem.size() == 2 * AcStrategy::kMaxCoeffArea + dct_scratch_size);

  constexpr float kInvColorFactor = 1.0f / kDefaultColorFactor;
  const auto inv_color_factor = Set(df, kInvColorFactor);
  const auto base_x = Set(df, 0.0f);
  const auto base_b = Set(df, jxl::cms::kYToBRatio);

  // Fast-path moment accumulators: sum(a^2) and sum(a*b) per channel.
  auto sum_a2_x = Zero(df);
  auto sum_ab_x = Zero(df);
  auto sum_a2_b = Zero(df);
  auto sum_ab_b = Zero(df);

  size_t num_ac = 0;

  for (size_t y = y0; y < y1; ++y) {
    const float* JXL_RESTRICT row_y =
        opsin_rect.ConstPlaneRow(opsin, 1, y * kBlockDim);
    const float* JXL_RESTRICT row_x =
        opsin_rect.ConstPlaneRow(opsin, 0, y * kBlockDim);
    const float* JXL_RESTRICT row_b =
        opsin_rect.ConstPlaneRow(opsin, 2, y * kBlockDim);

    for (size_t x = x0; x < x1; x++) {
      AcStrategy acs = use_dct8
                           ? AcStrategy::FromRawStrategy(AcStrategyType::DCT)
                           : ac_strategy->ConstRow(y)[x];
      if (!acs.IsFirstBlock()) continue;
      const auto strategy = acs.Strategy();
      const size_t cbx = acs.covered_blocks_x();
      const size_t cby = acs.covered_blocks_y();
      // The CfL map only consumes the AC coefficients. The DC values that used
      // to be staged here fed a DC-correlation pass that no longer exists, so
      // the transforms now drive the AC residual computation directly.
      TransformFromPixels(strategy, row_y + x * kBlockDim, stride, block_y,
                          scratch_space);
      TransformFromPixels(strategy, row_x + x * kBlockDim, stride, block_x,
                          scratch_space);
      TransformFromPixels(strategy, row_b + x * kBlockDim, stride, block_b,
                          scratch_space);
      const float* const JXL_RESTRICT qm_x = dequant.InvMatrix(strategy, 0);
      const float* const JXL_RESTRICT qm_b = dequant.InvMatrix(strategy, 2);

      // Do not use this block for computing AC CfL.
      if (cbx + x0 > x1 || cby + y0 > y1) {
        continue;
      }

      // Lay out the AC coefficients contiguously. The order in which
      // coefficients get stored does not matter.
      size_t cx = cbx;
      size_t cy = cby;
      CoefficientLayout(&cy, &cx);
      // Zero out LFs. This introduces terms in the optimization loop that
      // don't affect the result, as they are all 0, but allow for simpler
      // SIMDfication.
      for (size_t iy = 0; iy < cy; iy++) {
        for (size_t ix = 0; ix < cx; ix++) {
          block_y[cx * kBlockDim * iy + ix] = 0;
          block_x[cx * kBlockDim * iy + ix] = 0;
          block_b[cx * kBlockDim * iy + ix] = 0;
        }
      }
      // Unclear why this is like it is. (This works slightly better
      // than the previous approach which was also a hack.)
      const float qq =
          (raw_quant_field == nullptr) ? 1.0f : raw_quant_field->Row(y)[x];
      // Experimentally values 128-130 seem best -- I don't know why we
      // need this multiplier.
      const float kStrangeMultiplier = 128;
      float q = use_dct8 ? 1 : quantizer->Scale() * kStrangeMultiplier * qq;
      const auto qv = Set(df, q);
      for (size_t i = 0; i < cx * cy * 64; i += Lanes(df)) {
        const auto b_y = Load(df, block_y + i);
        const auto b_x = Load(df, block_x + i);
        const auto b_b = Load(df, block_b + i);
        const auto qqm_x = Mul(qv, Load(df, qm_x + i));
        const auto qqm_b = Mul(qv, Load(df, qm_b + i));
        // Residual coefficients: color residual = a * x + b.
        const auto m_x = Mul(b_y, qqm_x);
        const auto a_x = Mul(inv_color_factor, m_x);
        const auto bx = Sub(Mul(base_x, m_x), Mul(b_x, qqm_x));
        const auto m_b = Mul(b_y, qqm_b);
        const auto a_b = Mul(inv_color_factor, m_b);
        const auto bb = Sub(Mul(base_b, m_b), Mul(b_b, qqm_b));
        if (fast) {
          sum_a2_x = MulAdd(a_x, a_x, sum_a2_x);
          sum_ab_x = MulAdd(a_x, bx, sum_ab_x);
          sum_a2_b = MulAdd(a_b, a_b, sum_a2_b);
          sum_ab_b = MulAdd(a_b, bb, sum_ab_b);
        } else {
          Store(a_x, df, coeffs_ax + num_ac);
          Store(bx, df, coeffs_bx + num_ac);
          Store(a_b, df, coeffs_ab + num_ac);
          Store(bb, df, coeffs_bb + num_ac);
        }
        num_ac += Lanes(df);
      }
    }
  }
  JXL_ENSURE(num_ac % Lanes(df) == 0);
  if (fast) {
    row_out_x[tx] = FinishMultiplierFast(GetLane(SumOfLanes(df, sum_a2_x)),
                                         GetLane(SumOfLanes(df, sum_ab_x)),
                                         num_ac, kDistanceMultiplierAC);
    row_out_b[tx] = FinishMultiplierFast(GetLane(SumOfLanes(df, sum_a2_b)),
                                         GetLane(SumOfLanes(df, sum_ab_b)),
                                         num_ac, kDistanceMultiplierAC);
  } else {
    row_out_x[tx] = FindBestMultiplierRobust(coeffs_ax, coeffs_bx, num_ac,
                                             kDistanceMultiplierAC);
    row_out_b[tx] = FindBestMultiplierRobust(coeffs_ab, coeffs_bb, num_ac,
                                             kDistanceMultiplierAC);
  }
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(ComputeTile);

Status CfLHeuristics::ComputeTile(const Rect& r, const Image3F& opsin,
                                  const Rect& opsin_rect,
                                  const DequantMatrices& dequant,
                                  const AcStrategyImage* ac_strategy,
                                  const ImageI* raw_quant_field,
                                  const Quantizer* quantizer, bool fast,
                                  size_t thread, ColorCorrelationMap* cmap) {
  bool use_dct8 = ac_strategy == nullptr;
  Span<float> scratch(mem.address<float>() + thread * ItemsPerThread(),
                      ItemsPerThread());
  return HWY_DYNAMIC_DISPATCH(ComputeTile)(
      opsin, opsin_rect, dequant, ac_strategy, raw_quant_field, quantizer, r,
      fast, use_dct8, &cmap->ytox_map, &cmap->ytob_map, scratch);
}

Status ColorCorrelationEncodeDC(const ColorCorrelation& color_correlation,
                                BitWriter* writer, LayerType layer,
                                AuxOut* aux_out) {
  float color_factor = color_correlation.GetColorFactor();
  float base_correlation_x = color_correlation.GetBaseCorrelationX();
  float base_correlation_b = color_correlation.GetBaseCorrelationB();
  int32_t ytox_dc = color_correlation.GetYToXDC();
  int32_t ytob_dc = color_correlation.GetYToBDC();

  return writer->WithMaxBits(
      1 + 2 * kBitsPerByte + 12 + 32, layer, aux_out, [&]() -> Status {
        if (ytox_dc == 0 && ytob_dc == 0 &&
            color_factor == kDefaultColorFactor && base_correlation_x == 0.0f &&
            base_correlation_b == jxl::cms::kYToBRatio) {
          writer->Write(1, 1);
          return true;
        }
        writer->Write(1, 0);
        JXL_RETURN_IF_ERROR(
            U32Coder::Write(kColorFactorDist, color_factor, writer));
        JXL_RETURN_IF_ERROR(F16Coder::Write(base_correlation_x, writer));
        JXL_RETURN_IF_ERROR(F16Coder::Write(base_correlation_b, writer));
        writer->Write(kBitsPerByte,
                      ytox_dc - std::numeric_limits<int8_t>::min());
        writer->Write(kBitsPerByte,
                      ytob_dc - std::numeric_limits<int8_t>::min());
        return true;
      });
}

}  // namespace jxl
#endif  // HWY_ONCE
