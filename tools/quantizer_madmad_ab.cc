// A/B harness for Quantizer::SetQuantField median + MAD computation.
//
// OLD: two full-image scratch vectors (data + deviations).
// NEW: one scratch vector reused in place for the absolute deviations.
//
// Verifies (1) the two paths produce byte-identical (quant_median,
// quant_median_absd) on random data, and (2) the throughput / allocation
// difference. Self-contained: no libjxl dependency.
//
// Build (clang):
//   clang++ -O3 -std=c++17 tools/quantizer_madmad_ab.cc -o madmad_ab
// Run:
//   ./madmad_ab

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>

namespace {

// ---- OLD: two-buffer ----
void OldMadMad(const float* src, size_t n, float* out_median,
               float* out_absd) {
  std::vector<float> data(n);
  std::memcpy(data.data(), src, n * sizeof(float));  // same input multiset
  std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
  const float quant_median = data[data.size() / 2];
  std::vector<float> deviations(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    deviations[i] = std::abs(data[i] - quant_median);
  }
  std::nth_element(deviations.begin(),
                   deviations.begin() + deviations.size() / 2,
                   deviations.end());
  *out_median = quant_median;
  *out_absd = deviations[deviations.size() / 2];
}

// ---- NEW: one-buffer ----
void NewMadMad(const float* src, size_t n, float* out_median,
               float* out_absd) {
  std::vector<float> data(n);
  std::memcpy(data.data(), src, n * sizeof(float));
  const auto middle = data.begin() + data.size() / 2;
  std::nth_element(data.begin(), middle, data.end());
  const float quant_median = *middle;
  for (float& v : data) {
    v = std::abs(v - quant_median);
  }
  std::nth_element(data.begin(), middle, data.end());
  *out_median = quant_median;
  *out_absd = *middle;
}

double TimeMs(void (*fn)(const float*, size_t, float*, float*),
              const float* src, size_t n, int iters, float* sink) {
  using clk = std::chrono::steady_clock;
  auto t0 = clk::now();
  float m = 0, d = 0;
  for (int i = 0; i < iters; i++) {
    fn(src, n, &m, &d);
    sink[0] += m + d;  // defeat DCE
  }
  auto t1 = clk::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace

int main() {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(0.f, 256.f);

  // Correctness: many random fields of assorted sizes (incl. duplicates,
  // negatives, exact ties at the median).
  int fails = 0, checks = 0;
  for (int t = 0; t < 5000; t++) {
    size_t n = 1 + (rng() % 4096);
    std::vector<float> src(n);
    for (auto& v : src) {
      float x = dist(rng);
      // inject ties + negatives to stress the partition order-dependence
      if (rng() % 7 == 0) x = std::floor(x);
      if (rng() % 11 == 0) x = -x;
      v = x;
    }
    float mo, ao, mn, an;
    OldMadMad(src.data(), n, &mo, &ao);
    NewMadMad(src.data(), n, &mn, &an);
    checks++;
    if (std::memcmp(&mo, &mn, sizeof(float)) != 0 ||
        std::memcmp(&ao, &an, sizeof(float)) != 0) {
      if (fails < 5) {
        printf("MISMATCH n=%zu  median %.9g/%.9g  absd %.9g/%.9g\n", n, mo, mn,
               ao, an);
      }
      fails++;
    }
  }
  printf("correctness: %d checks, %d byte-exact mismatches\n", checks, fails);

  // Timing across the 5 corpus sizes.
  const size_t sizes[] = {64ul * 64, 256ul * 256, 512ul * 512, 1024ul * 1024,
                          2048ul * 2048};
  float sink[1] = {0};
  printf("\n%-12s %10s %10s %8s\n", "elems", "old_ms", "new_ms", "ratio");
  for (size_t n : sizes) {
    std::vector<float> src(n);
    for (auto& v : src) v = dist(rng);
    int iters = n >= (1ul << 20) ? 50 : 400;
    // warm
    TimeMs(OldMadMad, src.data(), n, 3, sink);
    TimeMs(NewMadMad, src.data(), n, 3, sink);
    double told = TimeMs(OldMadMad, src.data(), n, iters, sink);
    double tnew = TimeMs(NewMadMad, src.data(), n, iters, sink);
    printf("%-12zu %10.3f %10.3f %8.3f\n", n, told, tnew, tnew / told);
  }
  printf("(sink=%g)\n", sink[0]);
  return fails == 0 ? 0 : 1;
}
