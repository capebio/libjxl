// Standalone byte-exact equivalence + timing harness for quant_weights opts.
// Proves OLD vs NEW produce IDENTICAL output and times the change for:
//   (1) GetQuantWeights channel-fused radial geometry + num_bands==1 hoist
//   (2) ComputeQuantTable direct-write into inv_table (drop temp vector + copy)
// No libjxl build needed: the structure under change is mirrored in scalar C++.
// OLD and NEW share the SAME per-element primitives (mult/interp), so an
// identical hash proves the loop-restructure maps every output element to the
// same value (the only thing the patch changes); per-lane SIMD mechanics are
// untouched between OLD and NEW in the real code.
//
// Build: clang++ -std=c++17 -O2 tools/quant_weights_equiv_ab.cc -o qw_equiv
// Source of truth: external/libjxl-012/lib/jxl/quant_weights.cc @00f4d7fc
//   (GetQuantWeights, ComputeQuantTable reciprocal loop)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

static constexpr size_t kMaxDistanceBands = 17;  // 1 + (1 << 4)
static const float kSqrt2 = std::sqrt(2.0f);
static constexpr float kAlmostZero = 1e-8f;

// ---- shared primitives (identical for OLD and NEW) ----
static inline float Mult(float v) {
  if (v > 0.0f) return 1.0f + v;
  return 1.0f / (1.0f - v);
}
// Scalar mirror of InterpolateVec for one position (ConvertTo truncates >=0).
static inline float Interp(float scaled_pos, const float* array) {
  int idx = static_cast<int>(scaled_pos);
  float frac = scaled_pos - static_cast<float>(idx);
  float a = array[idx];
  float b = array[idx + 1];
  return a * std::pow(b / a, frac);
}

// ---- (1) GetQuantWeights ----
// OLD: per-channel recompute of bands/scale/geometry (verbatim structure).
static bool GetQuantWeights_OLD(size_t ROWS, size_t COLS,
                                const float bands_in[3][kMaxDistanceBands],
                                size_t num_bands, float* out) {
  for (size_t c = 0; c < 3; c++) {
    float bands[kMaxDistanceBands] = {bands_in[c][0]};
    if (bands[0] < kAlmostZero) return false;
    for (size_t i = 1; i < num_bands; i++) {
      bands[i] = bands[i - 1] * Mult(bands_in[c][i]);
      if (bands[i] < kAlmostZero) return false;
    }
    float scale = (num_bands - 1) / (kSqrt2 + 1e-6f);
    float rcpcol = scale / (COLS - 1);
    float rcprow = scale / (ROWS - 1);
    for (uint32_t y = 0; y < ROWS; y++) {
      float dy = y * rcprow;
      float dy2 = dy * dy;
      for (uint32_t x = 0; x < COLS; x++) {  // scalar over each lane position
        float dx = x * rcpcol;
        float scaled_distance = std::sqrt(dx * dx + dy2);
        float weight = num_bands == 1 ? bands[0] : Interp(scaled_distance, bands);
        out[c * COLS * ROWS + y * COLS + x] = weight;
      }
    }
  }
  return true;
}

// NEW: bands computed once per channel; geometry once; num_bands==1 hoisted.
static bool GetQuantWeights_NEW(size_t ROWS, size_t COLS,
                                const float bands_in[3][kMaxDistanceBands],
                                size_t num_bands, float* out) {
  float bands[3][kMaxDistanceBands];
  for (size_t c = 0; c < 3; c++) {
    bands[c][0] = bands_in[c][0];
    if (bands[c][0] < kAlmostZero) return false;
    for (size_t i = 1; i < num_bands; i++) {
      bands[c][i] = bands[c][i - 1] * Mult(bands_in[c][i]);
      if (bands[c][i] < kAlmostZero) return false;
    }
  }
  const size_t plane = COLS * ROWS;
  if (num_bands == 1) {
    for (size_t c = 0; c < 3; c++) std::fill_n(out + c * plane, plane, bands[c][0]);
    return true;
  }
  const float scale = (num_bands - 1) / (kSqrt2 + 1e-6f);
  const float rcpcol = scale / (COLS - 1);
  const float rcprow = scale / (ROWS - 1);
  for (uint32_t y = 0; y < ROWS; y++) {
    float dy = y * rcprow;
    float dy2 = dy * dy;
    for (uint32_t x = 0; x < COLS; x++) {
      float dx = x * rcpcol;
      float scaled_distance = std::sqrt(dx * dx + dy2);
      out[0 * plane + y * COLS + x] = Interp(scaled_distance, bands[0]);
      out[1 * plane + y * COLS + x] = Interp(scaled_distance, bands[1]);
      out[2 * plane + y * COLS + x] = Interp(scaled_distance, bands[2]);
    }
  }
  return true;
}

// ---- (2) ComputeQuantTable reciprocal step ----
// OLD: temp vector holds weights, then table = 1/w AND inv = w (redundant copy).
static void Recip_OLD(const float* gen, size_t n, float* table, float* inv) {
  std::vector<float> weights(gen, gen + n);  // temp alloc + fill (the copy cost)
  for (size_t i = 0; i < n; i++) {
    float w = weights[i];
    table[i] = 1.0f / w;
    inv[i] = w;
  }
}
// NEW: weights already live in inv; derive table in place, no temp, no copy.
static void Recip_NEW(const float* gen, size_t n, float* table, float* inv) {
  for (size_t i = 0; i < n; i++) inv[i] = gen[i];  // models GetQuantWeights write
  for (size_t i = 0; i < n; i++) {
    table[i] = 1.0f / inv[i];
    // inv[i] already correct.
  }
}

static uint64_t FNV(const float* p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  const uint8_t* b = reinterpret_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n * sizeof(float); i++) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}

int main() {
  // Deterministic PRNG (no <random> to keep it trivial/portable).
  uint64_t s = 0x9e3779b97f4a7c15ull;
  auto rnd = [&]() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return (s >> 11) * (1.0 / 9007199254740992.0);
  };

  const size_t dims[] = {4, 8, 16, 32, 64, 128, 256};
  size_t configs = 0, fails = 0;
  size_t maxnum = 256 * 256;
  std::vector<float> a(3 * maxnum), b(3 * maxnum);

  for (size_t ri = 0; ri < 7; ri++) {
    for (size_t ci = 0; ci < 7; ci++) {
      size_t ROWS = dims[ri], COLS = dims[ci];
      if (COLS % 4 != 0) continue;  // GetQuantWeights requires COLS multiple of 4
      for (size_t nb = 1; nb <= kMaxDistanceBands; nb++) {
        // Generate valid-ish bands: positive base, bounded multipliers.
        float bands[3][kMaxDistanceBands];
        for (size_t c = 0; c < 3; c++) {
          bands[c][0] = 0.05f + 50.0f * (float)rnd();
          for (size_t i = 1; i < nb; i++) bands[c][i] = -0.4f + 2.0f * (float)rnd();
        }
        bool ok1 = GetQuantWeights_OLD(ROWS, COLS, bands, nb, a.data());
        bool ok2 = GetQuantWeights_NEW(ROWS, COLS, bands, nb, b.data());
        configs++;
        if (ok1 != ok2) { fails++; continue; }
        if (!ok1) continue;
        if (FNV(a.data(), 3 * ROWS * COLS) != FNV(b.data(), 3 * ROWS * COLS)) {
          fails++;
          printf("MISMATCH GetQuantWeights ROWS=%zu COLS=%zu nb=%zu\n", ROWS, COLS, nb);
        }
      }
    }
  }
  printf("(1) GetQuantWeights fusion: %zu configs, %zu fails\n", configs, fails);

  // (2) reciprocal direct-write byte-exact (256x256 => 3*65536).
  {
    size_t n = 3 * 256 * 256;
    std::vector<float> gen(n), t1(n), i1(n), t2(n), i2(n);
    for (size_t i = 0; i < n; i++) gen[i] = 0.01f + 100.0f * (float)rnd();
    Recip_OLD(gen.data(), n, t1.data(), i1.data());
    Recip_NEW(gen.data(), n, t2.data(), i2.data());
    bool eq = FNV(t1.data(), n) == FNV(t2.data(), n) &&
              FNV(i1.data(), n) == FNV(i2.data(), n);
    printf("(2) ComputeQuantTable direct-write: %s\n", eq ? "BYTE-EXACT" : "MISMATCH");
    if (!eq) fails++;
  }

  // ---- microbench (interleaved OLD/NEW, start-rotated) ----
  auto now = [] { return std::chrono::high_resolution_clock::now(); };
  auto us = [](auto d) {
    return std::chrono::duration<double, std::micro>(d).count();
  };

  // Bench A: GetQuantWeights 256x256, nb=8 (the slow case).
  {
    float bands[3][kMaxDistanceBands];
    for (size_t c = 0; c < 3; c++) {
      bands[c][0] = 1.0f;
      for (size_t i = 1; i < 8; i++) bands[c][i] = 0.3f;
    }
    const int reps = 200;
    double told = 0, tnew = 0;
    volatile float sink = 0;
    for (int r = 0; r < reps; r++) {
      if (r & 1) {
        auto t = now(); GetQuantWeights_OLD(256, 256, bands, 8, a.data()); told += us(now() - t); sink += a[0];
        t = now(); GetQuantWeights_NEW(256, 256, bands, 8, b.data()); tnew += us(now() - t); sink += b[0];
      } else {
        auto t = now(); GetQuantWeights_NEW(256, 256, bands, 8, b.data()); tnew += us(now() - t); sink += b[0];
        t = now(); GetQuantWeights_OLD(256, 256, bands, 8, a.data()); told += us(now() - t); sink += a[0];
      }
    }
    printf("BenchA GetQuantWeights 256x256 nb=8: OLD %.1f us  NEW %.1f us  speedup %.3fx\n",
           told / reps, tnew / reps, told / tnew);
  }

  // Bench B: reciprocal+alloc 256x256.
  {
    size_t n = 3 * 256 * 256;
    std::vector<float> gen(n), t1(n), i1(n);
    for (size_t i = 0; i < n; i++) gen[i] = 0.5f + (float)rnd();
    const int reps = 200;
    double told = 0, tnew = 0;
    volatile float sink = 0;
    for (int r = 0; r < reps; r++) {
      if (r & 1) {
        auto t = now(); Recip_OLD(gen.data(), n, t1.data(), i1.data()); told += us(now() - t); sink += t1[0];
        t = now(); Recip_NEW(gen.data(), n, t1.data(), i1.data()); tnew += us(now() - t); sink += t1[0];
      } else {
        auto t = now(); Recip_NEW(gen.data(), n, t1.data(), i1.data()); tnew += us(now() - t); sink += t1[0];
        t = now(); Recip_OLD(gen.data(), n, t1.data(), i1.data()); told += us(now() - t); sink += t1[0];
      }
    }
    printf("BenchB recip+alloc 3*256x256: OLD %.1f us  NEW %.1f us  speedup %.3fx\n",
           told / reps, tnew / reps, told / tnew);
  }

  printf("\nTOTAL fails: %zu\n", fails);
  return fails == 0 ? 0 : 1;
}
