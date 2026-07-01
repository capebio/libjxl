// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// A/B harness for the UpsamplingStage inner loop (render_pipeline/
// stage_upsampling.cc ProcessRowImpl).
//
// OLD: reload the 25 input taps (and clamp bounds) inside the per-subpixel loop
//      — the real code cannot hoist them because input[] and the output rows are
//      opaque pointers the compiler must assume may alias.
// NEW: load the 25 taps + clamp bounds once per x, reuse across all N*N subpixels.
//
// FillOld/FillNew are HWY_NOINLINE and take opaque `const float* const*` /
// `float*` params, reproducing that aliasing barrier so OLD really does reload.
// Proves byte-identical output and times both. Build (clang-cl):
//   clang-cl /O2 /Ob2 /DNDEBUG -std:c++17 /EHsc /arch:AVX2 -TP ^
//     -IC:\Foo\raw-converter-wasm\external\libjxl-012\third_party\highway ^
//     tools\upsampling_loadhoist_ab.cc /Fe:ups_ab.exe

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

using DF = hn::ScalableTag<float>;

template <int N>
HWY_NOINLINE void FillOld(const float* const* input, const float* kernel,
                          const float* mins, const float* maxs, size_t len,
                          float* out, size_t stride) {
  const DF df;
  const size_t L = hn::Lanes(df);
  const float* in[25];
  for (int i = 0; i < 25; ++i) in[i] = input[i];
  for (size_t x = 0; x < len; x += L) {
    for (int oy = 0; oy < N; ++oy) {
      for (int ox = 0; ox < N; ++ox) {
        const int k = N * oy + ox;
        auto acc0 = hn::Mul(hn::LoadU(df, in[0]), hn::Set(df, kernel[k * 25 + 0]));
        auto acc1 = hn::Mul(hn::LoadU(df, in[1]), hn::Set(df, kernel[k * 25 + 1]));
        auto acc2 = hn::Mul(hn::LoadU(df, in[2]), hn::Set(df, kernel[k * 25 + 2]));
        for (int i = 3; i < 24; i += 3) {
          acc0 = hn::MulAdd(hn::LoadU(df, in[i]), hn::Set(df, kernel[k * 25 + i]), acc0);
          acc1 = hn::MulAdd(hn::LoadU(df, in[i + 1]), hn::Set(df, kernel[k * 25 + i + 1]), acc1);
          acc2 = hn::MulAdd(hn::LoadU(df, in[i + 2]), hn::Set(df, kernel[k * 25 + i + 2]), acc2);
        }
        acc0 = hn::MulAdd(hn::LoadU(df, in[24]), hn::Set(df, kernel[k * 25 + 24]), acc0);
        auto r = hn::Add(hn::Add(acc1, acc2), acc0);
        r = hn::Clamp(r, hn::LoadU(df, mins + x), hn::LoadU(df, maxs + x));
        hn::StoreU(r, df, out + static_cast<size_t>(N * oy + ox) * stride + x);
      }
    }
    for (int i = 0; i < 25; ++i) in[i] += L;
  }
}

template <int N>
HWY_NOINLINE void FillNew(const float* const* input, const float* kernel,
                          const float* mins, const float* maxs, size_t len,
                          float* out, size_t stride) {
  const DF df;
  using V = hn::Vec<DF>;
  const size_t L = hn::Lanes(df);
  const float* in[25];
  for (int i = 0; i < 25; ++i) in[i] = input[i];
  for (size_t x = 0; x < len; x += L) {
    V vin[25];
    for (int i = 0; i < 25; ++i) vin[i] = hn::LoadU(df, in[i]);
    const V vmin = hn::LoadU(df, mins + x);
    const V vmax = hn::LoadU(df, maxs + x);
    for (int oy = 0; oy < N; ++oy) {
      for (int ox = 0; ox < N; ++ox) {
        const int k = N * oy + ox;
        auto acc0 = hn::Mul(vin[0], hn::Set(df, kernel[k * 25 + 0]));
        auto acc1 = hn::Mul(vin[1], hn::Set(df, kernel[k * 25 + 1]));
        auto acc2 = hn::Mul(vin[2], hn::Set(df, kernel[k * 25 + 2]));
        for (int i = 3; i < 24; i += 3) {
          acc0 = hn::MulAdd(vin[i], hn::Set(df, kernel[k * 25 + i]), acc0);
          acc1 = hn::MulAdd(vin[i + 1], hn::Set(df, kernel[k * 25 + i + 1]), acc1);
          acc2 = hn::MulAdd(vin[i + 2], hn::Set(df, kernel[k * 25 + i + 2]), acc2);
        }
        acc0 = hn::MulAdd(vin[24], hn::Set(df, kernel[k * 25 + 24]), acc0);
        auto r = hn::Add(hn::Add(acc1, acc2), acc0);
        r = hn::Clamp(r, vmin, vmax);
        hn::StoreU(r, df, out + static_cast<size_t>(N * oy + ox) * stride + x);
      }
    }
    for (int i = 0; i < 25; ++i) in[i] += L;
  }
}

template <int N>
double RunN(const char* label) {
  const size_t W = 1024;          // input pixels per row (len)
  const size_t pad = 8;
  const size_t rowlen = W + 4 + 2 * pad;
  std::mt19937 rng(99 + N);
  std::uniform_real_distribution<float> ud(0.0f, 1.0f);
  std::vector<std::vector<float>> rows(5, std::vector<float>(rowlen));
  for (auto& r : rows) for (auto& v : r) v = ud(rng);
  std::vector<float> kernel(64 * 25);
  for (auto& v : kernel) v = ud(rng) * 0.1f;
  std::vector<float> mins(W + pad), maxs(W + pad);
  for (size_t i = 0; i < mins.size(); ++i) { mins[i] = 0.0f; maxs[i] = 1.0f; }
  // input[5*(iy+2)+(ix+2)] = row[iy] + (base + ix); base = pad+2 so ix=-2 valid.
  const float* input[25];
  for (int iy = -2; iy <= 2; ++iy)
    for (int ix = -2; ix <= 2; ++ix)
      input[5 * (iy + 2) + (ix + 2)] = rows[iy + 2].data() + (pad + 2) + ix;
  const size_t stride = W + pad;
  std::vector<float> out_old(static_cast<size_t>(N) * N * stride, 0.0f);
  std::vector<float> out_new(static_cast<size_t>(N) * N * stride, 0.0f);

  FillOld<N>(input, kernel.data(), mins.data(), maxs.data(), W, out_old.data(), stride);
  FillNew<N>(input, kernel.data(), mins.data(), maxs.data(), W, out_new.data(), stride);
  bool eq = std::memcmp(out_old.data(), out_new.data(),
                        out_old.size() * sizeof(float)) == 0;

  const int iters = 4000;
  volatile double sink = 0;
  double t_old = 0, t_new = 0;
  for (int r = 0; r < iters; ++r) {
    bool old_first = (r & 1) == 0;
    for (int w = 0; w < 2; ++w) {
      bool do_old = (w == 0) == old_first;
      auto t0 = std::chrono::steady_clock::now();
      if (do_old)
        FillOld<N>(input, kernel.data(), mins.data(), maxs.data(), W, out_old.data(), stride);
      else
        FillNew<N>(input, kernel.data(), mins.data(), maxs.data(), W, out_new.data(), stride);
      auto t1 = std::chrono::steady_clock::now();
      double dt = std::chrono::duration<double, std::micro>(t1 - t0).count();
      if (do_old) t_old += dt; else t_new += dt;
    }
    sink += out_old[0] + out_new[0];
  }
  std::printf("N=%d (%s): byte-exact=%s  OLD %.1f us  NEW %.1f us  NEW/OLD %.4f (%.1f%% %s)\n",
              N, label, eq ? "IDENTICAL" : "DIFFER", t_old / iters, t_new / iters,
              t_new / t_old, (1.0 - t_new / t_old) * 100.0,
              (t_new < t_old) ? "faster" : "SLOWER");
  (void)sink;
  return eq ? 0 : 1;
}

int main() {
  int bad = 0;
  bad += static_cast<int>(RunN<2>("2x"));
  bad += static_cast<int>(RunN<4>("4x"));
  bad += static_cast<int>(RunN<8>("8x"));
  std::printf("Lanes(float)=%zu\n", hn::Lanes(DF()));
  return bad;
}
