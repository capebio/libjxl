// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/quant_weights.h"

namespace jxl {

static const int32_t kDefaultQuant = 64;

#if JXL_CXX_LANG < JXL_CXX_17
constexpr int32_t Quantizer::kQuantMax;
#endif

Quantizer::Quantizer(const DequantMatrices& dequant)
    : Quantizer(dequant, kDefaultQuant, kGlobalScaleDenom / kDefaultQuant) {}

Quantizer::Quantizer(const DequantMatrices& dequant, int quant_dc,
                     int global_scale)
    : global_scale_(global_scale), quant_dc_(quant_dc), dequant_(&dequant) {
  RecomputeFromGlobalScale();
  // inv_quant_dc_ is already set by RecomputeFromGlobalScale() above using the
  // identical expression; zero_bias_ field removed (write-only dead).
}

void Quantizer::ComputeGlobalScaleAndQuant(float quant_dc, float quant_median,
                                           float quant_median_absd) {
  // Target value for the median value in the quant field.
  const float kQuantFieldTarget = 5;
  // We reduce the median of the quant field by the median absolute deviation:
  // higher resolution on highly varying quant fields.
  float scale = kGlobalScaleDenom * (quant_median - quant_median_absd) /
                kQuantFieldTarget;
  // Ensure that new_global_scale is positive and no more than 1<<15.
  if (scale < 1) scale = 1;
  if (scale > (1 << 15)) scale = 1 << 15;
  int new_global_scale = static_cast<int>(scale);
  // Ensure that quant_dc_ will always be at least
  // 0.625 * kGlobalScaleDenom/kGlobalScaleNumerator = 10.
  const int scaled_quant_dc =
      static_cast<int>(quant_dc * kGlobalScaleNumerator * 1.6);
  if (new_global_scale > scaled_quant_dc) {
    new_global_scale = scaled_quant_dc;
    if (new_global_scale <= 0) new_global_scale = 1;
  }
  // The intermediate RecomputeFromGlobalScale() only needs inv_global_scale_
  // here; every other derived field it would compute uses the still-stale
  // quant_dc_ and is overwritten by the final recompute below. Derive the
  // reciprocal locally (same double->float path as the member) so the result
  // is byte-identical, then rebuild all derived state exactly once.
  const float inv_global_scale =
      static_cast<float>(1.0 * kGlobalScaleDenom / new_global_scale);
  float fval = quant_dc * inv_global_scale + 0.5f;
  fval = std::min<float>(1 << 16, fval);

  global_scale_ = new_global_scale;
  quant_dc_ = static_cast<int>(fval);
  RecomputeFromGlobalScale();
}

void Quantizer::SetQuantFieldRect(const ImageF& qf, const Rect& rect,
                                  ImageI* JXL_RESTRICT raw_quant_field) const {
  for (size_t y = 0; y < rect.ysize(); ++y) {
    const float* JXL_RESTRICT row_qf = rect.ConstRow(qf, y);
    int32_t* JXL_RESTRICT row_qi = rect.Row(raw_quant_field, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      int val = ClampVal(row_qf[x] * inv_global_scale_ + 0.5f);
      row_qi[x] = val;
    }
  }
}

Status Quantizer::SetQuantField(const float quant_dc, const ImageF& qf,
                                ImageI* JXL_RESTRICT raw_quant_field) {
  const size_t xsize = qf.xsize();
  std::vector<float> data(xsize * qf.ysize());
  for (size_t y = 0; y < qf.ysize(); ++y) {
    const float* JXL_RESTRICT row_qf = qf.Row(y);
    memcpy(data.data() + y * xsize, row_qf, xsize * sizeof(*row_qf));
  }
  // The median selection already destroys the row order, so reuse the same
  // scratch buffer for the absolute deviations instead of allocating a second
  // full quant-field-sized vector. The deviation multiset is independent of the
  // order in which `data` is left by nth_element, so both medians are exact.
  const auto middle = data.begin() + data.size() / 2;
  std::nth_element(data.begin(), middle, data.end());
  const float quant_median = *middle;
  for (float& v : data) {
    v = std::abs(v - quant_median);
  }
  std::nth_element(data.begin(), middle, data.end());
  const float quant_median_absd = *middle;
  ComputeGlobalScaleAndQuant(quant_dc, quant_median, quant_median_absd);
  if (raw_quant_field) {
    JXL_ENSURE(SameSize(*raw_quant_field, qf));
    SetQuantFieldRect(qf, Rect(qf), raw_quant_field);
  }
  return true;
}

void Quantizer::SetQuant(float quant_dc, float quant_ac,
                         ImageI* JXL_RESTRICT raw_quant_field) {
  ComputeGlobalScaleAndQuant(quant_dc, quant_ac, 0);
  int32_t val = ClampVal(quant_ac * inv_global_scale_ + 0.5f);
  FillImage(val, raw_quant_field);
}

Status QuantizerParams::VisitFields(Visitor* JXL_RESTRICT visitor) {
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(
      BitsOffset(11, 1), BitsOffset(11, 2049), BitsOffset(12, 4097),
      BitsOffset(16, 8193), 1, &global_scale));
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Val(16), BitsOffset(5, 1),
                                         BitsOffset(8, 1), BitsOffset(16, 1), 1,
                                         &quant_dc));
  return true;
}

QuantizerParams Quantizer::GetParams() const {
  QuantizerParams params;
  params.global_scale = global_scale_;
  params.quant_dc = quant_dc_;
  return params;
}

Status Quantizer::Decode(BitReader* reader) {
  QuantizerParams params;
  JXL_RETURN_IF_ERROR(Bundle::Read(reader, &params));
  global_scale_ = static_cast<int>(params.global_scale);
  quant_dc_ = static_cast<int>(params.quant_dc);
  RecomputeFromGlobalScale();
  return true;
}

void Quantizer::DumpQuantizationMap(const ImageI& raw_quant_field) const {
  printf("Global scale: %d (%.7f)\nDC quant: %d\n", global_scale_,
         global_scale_ * 1.0 / kGlobalScaleDenom, quant_dc_);
  printf("AC quantization Map:\n");
  for (size_t y = 0; y < raw_quant_field.ysize(); ++y) {
    for (size_t x = 0; x < raw_quant_field.xsize(); ++x) {
      printf(" %3d", raw_quant_field.Row(y)[x]);
    }
    printf("\n");
  }
}

}  // namespace jxl
