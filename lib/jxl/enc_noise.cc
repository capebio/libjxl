// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_noise.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <utility>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_optimize.h"
#include "lib/jxl/image.h"
#include "lib/jxl/noise.h"

namespace jxl {
namespace {

using OptimizeArray = optimize::Array<double, NoiseParams::kNumNoisePoints>;

constexpr int kNoiseBlockSize = 8;

using NoiseTile = std::array<std::array<float, kNoiseBlockSize>,
                             kNoiseBlockSize>;

// Loads the estimator's XY channel once per 8x8 tile. The expression is kept
// as 0.5f * (Y + X), matching the original source-level floating-point order.
NoiseTile LoadNoiseTileXY(const Image3F& opsin, const size_t x,
                          const size_t y) {
  NoiseTile xy;
  for (int y_bl = 0; y_bl < kNoiseBlockSize; ++y_bl) {
    const float* const x_row = opsin.PlaneRow(0, y + y_bl);
    const float* const y_row = opsin.PlaneRow(1, y + y_bl);
    for (int x_bl = 0; x_bl < kNoiseBlockSize; ++x_bl) {
      xy[y_bl][x_bl] = 0.5f * (y_row[x + x_bl] + x_row[x + x_bl]);
    }
  }
  return xy;
}

float GetScoreSumsOfAbsoluteDifferences(const Image3F& opsin, const int x,
                                        const int y, const int block_size) {
  constexpr int kSmallBlockSizeX = 3;
  constexpr int kSmallBlockSizeY = 4;
  constexpr int kNumSAD =
      (kNoiseBlockSize - kSmallBlockSizeX) *
      (kNoiseBlockSize - kSmallBlockSizeY);
  constexpr int kOffset = 2;
  JXL_DASSERT(block_size == kNoiseBlockSize);

  const NoiseTile xy = LoadNoiseTileXY(opsin, x, y);

  int counter = 0;
  std::array<float, kNumSAD> sad;
  for (int y_bl = 0; y_bl + kSmallBlockSizeY < block_size; ++y_bl) {
    for (int x_bl = 0; x_bl + kSmallBlockSizeX < block_size; ++x_bl) {
      float sad_sum = 0;
      // Size of the center patch: compare all patches inside the window with
      // the center one.
      for (int cy = 0; cy < kSmallBlockSizeY; ++cy) {
        for (int cx = 0; cx < kSmallBlockSizeX; ++cx) {
          const float wnd = xy[y_bl + cy][x_bl + cx];
          const float center = xy[kOffset + cy][kOffset + cx];
          sad_sum += std::abs(center - wnd);
        }
      }
      sad[counter++] = sad_sum;
    }
  }
  JXL_DASSERT(counter == kNumSAD);

  constexpr int kSamples = kNumSAD / 2;
  // As with ROAD (rank order absolute distance), we keep the smallest half of
  // the values in SAD (we use here the more robust patch SAD instead of
  // absolute single-pixel differences).
  std::sort(sad.begin(), sad.end());
  const float total_sad_sum =
      std::accumulate(sad.begin(), sad.begin() + kSamples, 0.0f);
  return total_sad_sum / kSamples;
}

class NoiseHistogram {
 public:
  static constexpr int kBins = 256;

  NoiseHistogram() { std::fill(bins, bins + kBins, 0); }

  void Increment(const float x) { bins[Index(x)] += 1; }
  int Get(const float x) const { return bins[Index(x)]; }
  int Bin(const size_t bin) const { return bins[bin]; }

  int Mode() const {
    size_t max_idx = 0;
    for (size_t i = 0; i < kBins; i++) {
      if (bins[i] > bins[max_idx]) max_idx = i;
    }
    return max_idx;
  }

  double Quantile(double q01) const {
    const int64_t total = std::accumulate(bins, bins + kBins, int64_t{1});
    const int64_t target = static_cast<int64_t>(q01 * total);
    // Until sum >= target:
    int64_t sum = 0;
    size_t i = 0;
    for (; i < kBins; ++i) {
      sum += bins[i];
      // Exact match: assume middle of bin i
      if (sum == target) {
        return i + 0.5;
      }
      if (sum > target) break;
    }

    // Next non-empty bin (in case histogram is sparsely filled)
    size_t next = i + 1;
    while (next < kBins && bins[next] == 0) {
      ++next;
    }

    // Linear interpolation according to how far into next we went
    const double excess = target - sum;
    const double weight_next = bins[Index(next)] / excess;
    return ClampX(next * weight_next + i * (1.0 - weight_next));
  }

  // Inter-quartile range
  double IQR() const { return Quantile(0.75) - Quantile(0.25); }

 private:
  template <typename T>
  T ClampX(const T x) const {
    return jxl::Clamp1<T>(x, 0, kBins - 1);
  }
  size_t Index(const float x) const { return ClampX(static_cast<int>(x)); }

  uint32_t bins[kBins];
};

std::vector<float> GetSADScoresForPatches(const Image3F& opsin,
                                          const size_t block_s,
                                          const size_t num_bin,
                                          NoiseHistogram* sad_histogram) {
  const size_t num_blocks =
      (opsin.ysize() / block_s) * (opsin.xsize() / block_s);
  std::vector<float> sad_scores;
  sad_scores.reserve(num_blocks);

  for (size_t y = 0; y + block_s <= opsin.ysize(); y += block_s) {
    for (size_t x = 0; x + block_s <= opsin.xsize(); x += block_s) {
      const float sad_sc =
          GetScoreSumsOfAbsoluteDifferences(opsin, x, y, block_s);
      sad_scores.push_back(sad_sc);
      sad_histogram->Increment(sad_sc * num_bin);
    }
  }
  return sad_scores;
}

float GetSADThreshold(const NoiseHistogram& histogram, const int num_bin) {
  // Here we assume that the most patches with similar SAD value is a "flat"
  // patches. However, some images might contain regular texture part and
  // generate second strong peak at the histogram
  // TODO(user) handle bimodal and heavy-tailed case
  const int mode = histogram.Mode();
  return static_cast<float>(mode) / NoiseHistogram::kBins;
}

// loss = sum asym * (F(x) - nl)^2 + kReg * num_points * sum (w[i] - w[i+1])^2
// where asym = 1 if F(x) < nl, kAsym if F(x) > nl.
struct LossFunction {
  explicit LossFunction(const std::vector<NoiseLevel>& nl0) : nl(nl0) {}

  double Compute(const OptimizeArray& w, OptimizeArray* df,
                 bool skip_regularization = false) const {
    constexpr double kReg = 0.005;
    constexpr double kAsym = 1.1;
    double loss_function = 0;
    for (size_t i = 0; i < w.size(); i++) {
      (*df)[i] = 0;
    }
    for (const NoiseLevel& ind : nl) {
      std::pair<int, float> pos = IndexAndFrac(ind.intensity);
      JXL_DASSERT(pos.first >= 0 && static_cast<size_t>(pos.first) <
                                        NoiseParams::kNumNoisePoints - 1);
      double low = w[pos.first];
      double hi = w[pos.first + 1];
      double val = low * (1.0f - pos.second) + hi * pos.second;
      double dist = val - ind.noise_level;
      if (dist > 0) {
        loss_function += kAsym * dist * dist;
        (*df)[pos.first] -= kAsym * (1.0f - pos.second) * dist;
        (*df)[pos.first + 1] -= kAsym * pos.second * dist;
      } else {
        loss_function += dist * dist;
        (*df)[pos.first] -= (1.0f - pos.second) * dist;
        (*df)[pos.first + 1] -= pos.second * dist;
      }
    }
    if (skip_regularization) return loss_function;
    for (size_t i = 0; i + 1 < w.size(); i++) {
      double diff = w[i] - w[i + 1];
      loss_function += kReg * nl.size() * diff * diff;
      (*df)[i] -= kReg * diff * nl.size();
      (*df)[i + 1] += kReg * diff * nl.size();
    }
    return loss_function;
  }

  const std::vector<NoiseLevel>& nl;
};

void OptimizeNoiseParameters(const std::vector<NoiseLevel>& noise_level,
                             NoiseParams* noise_params, float mul) {
  constexpr double kMaxError = 1e-3;
  static const double kPrecision = 1e-8;
  static const int kMaxIter = 40;

  if (noise_level.empty()) {
    noise_params->Clear();
    return;
  }

  float avg = 0;
  for (const NoiseLevel& nl : noise_level) {
    avg += nl.noise_level;
  }
  avg /= noise_level.size();

  LossFunction loss_function(noise_level);
  OptimizeArray parameter_vector;
  for (size_t i = 0; i < parameter_vector.size(); i++) {
    parameter_vector[i] = avg;
  }

  parameter_vector = optimize::OptimizeWithScaledConjugateGradientMethod(
      loss_function, parameter_vector, kPrecision, kMaxIter);

  // Clamp here to account codestream limits.
  for (size_t i = 0; i < parameter_vector.size(); i++) {
    parameter_vector[i] =
        jxl::Clamp1<float>(parameter_vector[i] * mul, 0.0f, kNoiseLutMax);
  }

  OptimizeArray unused;
  float loss = loss_function.Compute(parameter_vector, &unused,
                                     /*skip_regularization=*/true) /
               noise_level.size();

  // Approximation went too badly: escape with no noise at all.
  if (loss > kMaxError) {
    noise_params->Clear();
    return;
  }

  for (size_t i = 0; i < parameter_vector.size(); i++) {
    noise_params->lut[i] = parameter_vector[i];
  }
}

std::vector<NoiseLevel> GetNoiseLevel(
    const Image3F& opsin, const std::vector<float>& texture_strength,
    const float threshold, const size_t block_s) {
  JXL_DASSERT(block_s == static_cast<size_t>(kNoiseBlockSize));
  std::vector<NoiseLevel> noise_level_per_intensity;
  noise_level_per_intensity.reserve(texture_strength.size());

  constexpr int kFiltSize = 1;
  constexpr int kPaddedBlockSize = kNoiseBlockSize + 2 * kFiltSize;
  static const float kLaplFilter[kFiltSize * 2 + 1][kFiltSize * 2 + 1] = {
      {-0.25f, -1.0f, -0.25f},
      {-1.0f, 5.0f, -1.0f},
      {-0.25f, -1.0f, -0.25f},
  };

  // The noise model is built based on channel 0.5 * (X+Y) as we notice that it
  // is similar to the model 0.5 * (Y-X)
  size_t patch_index = 0;

  for (size_t y = 0; y + block_s <= opsin.ysize(); y += block_s) {
    for (size_t x = 0; x + block_s <= opsin.xsize(); x += block_s) {
      if (texture_strength[patch_index] <= threshold) {
        const NoiseTile xy = LoadNoiseTileXY(opsin, x, y);

        // Fill a one-pixel mirror-without-edge-repeat border once per selected
        // tile. This is exactly the y_bl - y_f / x_bl - x_f reflection used by
        // the old boundary branches.
        std::array<std::array<float, kPaddedBlockSize>, kPaddedBlockSize>
            padded_xy;
        for (int y_pad = 0; y_pad < kPaddedBlockSize; ++y_pad) {
          const int y_bl = y_pad - kFiltSize;
          const int source_y =
              y_bl < 0 ? -y_bl
                       : (y_bl >= kNoiseBlockSize
                              ? 2 * kNoiseBlockSize - 2 - y_bl
                              : y_bl);
          for (int x_pad = 0; x_pad < kPaddedBlockSize; ++x_pad) {
            const int x_bl = x_pad - kFiltSize;
            const int source_x =
                x_bl < 0 ? -x_bl
                         : (x_bl >= kNoiseBlockSize
                                ? 2 * kNoiseBlockSize - 2 - x_bl
                                : x_bl);
            padded_xy[y_pad][x_pad] = xy[source_y][source_x];
          }
        }

        // Calculate mean value in the original row-major order.
        float mean_int = 0;
        for (int y_bl = 0; y_bl < kNoiseBlockSize; ++y_bl) {
          for (int x_bl = 0; x_bl < kNoiseBlockSize; ++x_bl) {
            mean_int += xy[y_bl][x_bl];
          }
        }
        mean_int /= block_s * block_s;

        // Calculate the Laplacian response. The coefficient and accumulation
        // order match the old y_f=-1..1, x_f=-1..1 loops exactly.
        float noise_level = 0;
        size_t count = 0;
        for (int y_bl = 0; y_bl < kNoiseBlockSize; ++y_bl) {
          for (int x_bl = 0; x_bl < kNoiseBlockSize; ++x_bl) {
            float filtered_value = 0;
            for (int y_f = 0; y_f < 2 * kFiltSize + 1; ++y_f) {
              for (int x_f = 0; x_f < 2 * kFiltSize + 1; ++x_f) {
                filtered_value += padded_xy[y_bl + y_f][x_bl + x_f] *
                                  kLaplFilter[y_f][x_f];
              }
            }
            noise_level += std::abs(filtered_value);
            ++count;
          }
        }
        noise_level /= count;
        NoiseLevel nl;
        nl.intensity = mean_int;
        nl.noise_level = noise_level;
        noise_level_per_intensity.push_back(nl);
      }
      ++patch_index;
    }
  }
  return noise_level_per_intensity;
}

Status EncodeFloatParam(float val, float precision, BitWriter* writer) {
  JXL_ENSURE(val >= 0);
  const int absval_quant = static_cast<int>(std::lround(val * precision));
  JXL_ENSURE(absval_quant < (1 << 10));
  writer->Write(10, absval_quant);
  return true;
}

}  // namespace

Status GetNoiseParameter(const Image3F& opsin, NoiseParams* noise_params,
                         float quality_coef) {
  // A non-positive multiplier cannot yield a valid emitted model. Avoid the
  // full image analysis and make the output state explicit.
  if (quality_coef <= 0.0f) {
    noise_params->Clear();
    return false;
  }

  // The size of a patch in decoder might be different from encoder's patch
  // size.
  // For encoder: the patch size should be big enough to estimate
  //              noise level, but, at the same time, it should be not too big
  //              to be able to estimate intensity value of the patch
  const size_t block_s = 8;
  const size_t kNumBin = 256;
  NoiseHistogram sad_histogram;
  std::vector<float> sad_scores =
      GetSADScoresForPatches(opsin, block_s, kNumBin, &sad_histogram);
  float sad_threshold = GetSADThreshold(sad_histogram, kNumBin);
  // If threshold is too large, the image has a strong pattern. This pattern
  // fools our model and it will add too much noise. Therefore, we do not add
  // noise for such images
  if (sad_threshold > 0.15f || sad_threshold <= 0.0f) {
    noise_params->Clear();
    return false;
  }
  std::vector<NoiseLevel> nl =
      GetNoiseLevel(opsin, sad_scores, sad_threshold, block_s);

  OptimizeNoiseParameters(nl, noise_params, quality_coef * 1.4f);
  return noise_params->HasAny();
}

Status EncodeNoise(const NoiseParams& noise_params, BitWriter* writer,
                   LayerType layer, AuxOut* aux_out) {
  JXL_ENSURE(noise_params.HasAny());

  return writer->WithMaxBits(
      NoiseParams::kNumNoisePoints * 16, layer, aux_out, [&]() -> Status {
        for (float i : noise_params.lut) {
          JXL_RETURN_IF_ERROR(EncodeFloatParam(i, kNoisePrecision, writer));
        }
        return true;
      });
}

}  // namespace jxl
