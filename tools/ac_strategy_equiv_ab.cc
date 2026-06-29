// Standalone byte-exact equivalence harness for ac_strategy / ac_context opts.
// Proves OLD vs NEW produce IDENTICAL output for:
//   (1) CoeffOrderAndLut exact rectangular zig-zag traversal (order + lut)
//   (2) ZeroDensityContext covered_blocks==1 specialization (ZeroDensityContext1)
//   (3) AcStrategyImage::SetNoBoundsCheck unchecked writer split
// No libjxl build needed: pure functions are mirrored verbatim from source.
// Build: clang++ -std=c++17 -O2 tools/ac_strategy_equiv_ab.cc -o ac_equiv
//
// Source of truth: external/libjxl-012/lib/jxl/{ac_strategy.cc,ac_strategy.h,
//                  ac_context.h,coeff_order_fwd.h,base/bits.h} @10783f7e

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

using coeff_order_t = uint32_t;
static constexpr size_t kBlockDim = 8;
static constexpr int kNumValidStrategies = 27;

// ---- mirrored helpers (verbatim) ----
static size_t covered_x(int s) {
  static const uint8_t kLut[] = {1, 1, 1, 1,  2, 4,  1,  2,  1,
                                 4, 2, 4, 1,  1, 1,  1,  1,  1,
                                 8, 4, 8, 16, 8, 16, 32, 16, 32};
  return kLut[s];
}
static size_t covered_y(int s) {
  static const uint8_t kLut[] = {1, 1, 1, 1,  2,  4, 2,  1,  4,
                                 1, 4, 2, 1,  1,  1, 1,  1,  1,
                                 8, 8, 4, 16, 16, 8, 32, 32, 16};
  return kLut[s];
}
static inline size_t CoefficientRows(size_t r, size_t c) { return r < c ? r : c; }
static inline size_t CoefficientColumns(size_t r, size_t c) { return r < c ? c : r; }
static inline void CoefficientLayout(size_t* rows, size_t* cols) {
  size_t r = *rows, c = *cols;
  *rows = CoefficientRows(r, c);
  *cols = CoefficientColumns(r, c);
}
static inline size_t FloorLog2Nonzero(size_t x) {
  size_t n = 0;
  while (x > 1) { x >>= 1; ++n; }
  return n;
}
static inline size_t CeilLog2Nonzero(size_t x) {
  size_t f = FloorLog2Nonzero(x);
  if ((x & (x - 1)) == 0) return f;
  return f + 1;
}

// ---- (1a) OLD CoeffOrderAndLut (verbatim from ac_strategy.cc) ----
template <bool is_lut>
static void CoeffOrder_OLD(int s, coeff_order_t* out) {
  size_t cx = covered_x(s);
  size_t cy = covered_y(s);
  CoefficientLayout(&cy, &cx);
  size_t xs = cx / cy;
  size_t xsm = xs - 1;
  size_t xss = CeilLog2Nonzero(xs);
  size_t cur = cx * cy;
  for (size_t i = 0; i < cx * kBlockDim; i++) {
    for (size_t j = 0; j <= i; j++) {
      size_t x = j;
      size_t y = i - j;
      if (i % 2) { size_t t = x; x = y; y = t; }
      if ((y & xsm) != 0) continue;
      y >>= xss;
      size_t val = 0;
      if (x < cx && y < cy) val = y * cx + x; else val = cur++;
      if (is_lut) out[y * cx * kBlockDim + x] = val;
      else out[val] = y * cx * kBlockDim + x;
    }
  }
  for (size_t ip = cx * kBlockDim - 1; ip > 0; ip--) {
    size_t i = ip - 1;
    for (size_t j = 0; j <= i; j++) {
      size_t x = cx * kBlockDim - 1 - (i - j);
      size_t y = cx * kBlockDim - 1 - j;
      if (i % 2) { size_t t = x; x = y; y = t; }
      if ((y & xsm) != 0) continue;
      y >>= xss;
      size_t val = cur++;
      if (is_lut) out[y * cx * kBlockDim + x] = val;
      else out[val] = y * cx * kBlockDim + x;
    }
  }
}

// ---- (1b) NEW CoeffOrderAndLut: exact residue-stepping, no discard ----
template <bool is_lut>
static void CoeffOrder_NEW(int s, coeff_order_t* out) {
  size_t cx = covered_x(s);
  size_t cy = covered_y(s);
  CoefficientLayout(&cy, &cx);
  const size_t xs = cx / cy;
  const size_t xsm = xs - 1;
  const size_t xss = CeilLog2Nonzero(xs);
  const size_t n = cx * kBlockDim;
  size_t cur = cx * cy;
  // First half.
  for (size_t i = 0; i < n; i++) {
    if (i & 1) {
      // After swap: x = i - j, y = j; require j divisible by xs.
      for (size_t j = 0; j <= i; j += xs) {
        size_t x = i - j;
        size_t y = j >> xss;
        size_t val = (x < cx && y < cy) ? y * cx + x : cur++;
        if (is_lut) out[y * n + x] = val; else out[val] = y * n + x;
      }
    } else {
      // No swap: x = j, y = i - j; require j == i (mod xs).
      for (size_t j = i & xsm; j <= i; j += xs) {
        size_t x = j;
        size_t y = (i - j) >> xss;
        size_t val = (x < cx && y < cy) ? y * cx + x : cur++;
        if (is_lut) out[y * n + x] = val; else out[val] = y * n + x;
      }
    }
  }
  // Second half. n is divisible by xs.
  for (size_t ip = n - 1; ip > 0; ip--) {
    size_t i = ip - 1;
    if (i & 1) {
      // After swap: x = n-1-j, y = n-1-i+j; require j == i+1 (mod xs).
      for (size_t j = (i + 1) & xsm; j <= i; j += xs) {
        size_t x = n - 1 - j;
        size_t y = (n - 1 - i + j) >> xss;
        size_t val = cur++;
        if (is_lut) out[y * n + x] = val; else out[val] = y * n + x;
      }
    } else {
      // No swap: x = n-1-i+j, y = n-1-j; require j == xsm (mod xs).
      for (size_t j = xsm; j <= i; j += xs) {
        size_t x = n - 1 - i + j;
        size_t y = (n - 1 - j) >> xss;
        size_t val = cur++;
        if (is_lut) out[y * n + x] = val; else out[val] = y * n + x;
      }
    }
  }
}

// ---- (2) ZeroDensityContext ----
static const uint16_t kCoeffFreqContext[64] = {
    0xBAD, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    15,    15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22,
    23,    23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26,
    27,    27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
};
static const uint16_t kCoeffNumNonzeroContext[64] = {
    0xBAD, 0,   31,  62,  62,  93,  93,  93,  93,  123, 123, 123, 123,
    152,   152, 152, 152, 152, 152, 152, 152, 180, 180, 180, 180, 180,
    180,   180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
    206,   206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206,   206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
};
static inline size_t ZDC_OLD(size_t nzl, size_t k, size_t cb, size_t l2, size_t prev) {
  nzl = (nzl + cb - 1) >> l2;
  k >>= l2;
  return (kCoeffNumNonzeroContext[nzl] + kCoeffFreqContext[k]) * 2 + prev;
}
static inline size_t ZDC1_NEW(size_t nzl, size_t k, size_t prev) {
  return (kCoeffNumNonzeroContext[nzl] + kCoeffFreqContext[k]) * 2 + prev;
}

// ---- (3) SetNoBoundsCheck writers ----
static void Set_OLD(uint8_t* row, size_t stride, size_t x, size_t y, int type) {
  size_t bx = covered_x(type), by = covered_y(type);
  for (size_t iy = 0; iy < by; iy++)
    for (size_t ix = 0; ix < bx; ix++)
      row[(y + iy) * stride + x + ix] =
          (uint8_t)((type << 1) | (((iy | ix) == 0) ? 1 : 0));
}
static void Set_NEW(uint8_t* row, size_t stride, size_t x, size_t y, int type) {
  size_t bx = covered_x(type), by = covered_y(type);
  uint8_t interior = (uint8_t)(type << 1);
  uint8_t* dst = row + y * stride + x;
  for (size_t iy = 0; iy < by; iy++) {
    uint8_t* r = dst + iy * stride;
    for (size_t ix = 0; ix < bx; ix++) r[ix] = interior;
  }
  dst[0] |= 1;
}

int main() {
  int fails = 0;
  size_t checks = 0;

  // (1) Coeff order + lut, all strategies, both modes.
  for (int s = 0; s < kNumValidStrategies; s++) {
    size_t cx = covered_x(s), cy = covered_y(s);
    size_t cl_r = cy, cl_c = cx; CoefficientLayout(&cl_r, &cl_c);
    size_t area = cl_c * kBlockDim * cl_c * kBlockDim;
    std::vector<coeff_order_t> ord_o(area, 0xFFFFFFFF), ord_n(area, 0xFFFFFFFF);
    std::vector<coeff_order_t> lut_o(area, 0xFFFFFFFF), lut_n(area, 0xFFFFFFFF);
    CoeffOrder_OLD<false>(s, ord_o.data());
    CoeffOrder_NEW<false>(s, ord_n.data());
    CoeffOrder_OLD<true>(s, lut_o.data());
    CoeffOrder_NEW<true>(s, lut_n.data());
    bool ok = (ord_o == ord_n) && (lut_o == lut_n);
    checks += 2 * area;
    if (!ok) { printf("FAIL CoeffOrder strat=%d (cx=%zu cy=%zu)\n", s, cx, cy); fails++; }
  }

  // (2) ZeroDensityContext cb==1 vs ZDC1, all nzeros x k.
  for (size_t nzl = 1; nzl < 64; nzl++) {
    for (size_t k = 1; k < 64; k++) {
      for (size_t prev = 0; prev < 2; prev++) {
        size_t a = ZDC_OLD(nzl, k, /*cb=*/1, /*l2=*/0, prev);
        size_t b = ZDC1_NEW(nzl, k, prev);
        checks++;
        if (a != b) { printf("FAIL ZDC1 nzl=%zu k=%zu prev=%zu (%zu!=%zu)\n",
                             nzl, k, prev, a, b); fails++; }
      }
    }
  }

  // (3) Set writer, all strategies, into a padded grid.
  for (int s = 0; s < kNumValidStrategies; s++) {
    size_t bx = covered_x(s), by = covered_y(s);
    size_t stride = bx + 4, h = by + 4;
    std::vector<uint8_t> go(stride * h, 0xFF), gn(stride * h, 0xFF);
    Set_OLD(go.data(), stride, 1, 1, s);
    Set_NEW(gn.data(), stride, 1, 1, s);
    checks += stride * h;
    if (go != gn) { printf("FAIL Set strat=%d\n", s); fails++; }
  }

  printf("checks=%zu fails=%d -> %s\n", checks, fails,
         fails == 0 ? "ALL BYTE-EXACT" : "MISMATCH");

  // ---- timing microbench: interleaved A/B order generation ----
  // Largest strategies dominate the loop cost; cover the full set per pass.
  const int kReps = 4000;
  std::vector<coeff_order_t> buf(256 * 256, 0);
  volatile uint64_t sink = 0;
  double t_old = 0, t_new = 0;
  for (int pass = 0; pass < 8; pass++) {
    // OLD
    auto a0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < kReps; r++)
      for (int s = 0; s < kNumValidStrategies; s++) {
        CoeffOrder_OLD<false>(s, buf.data());
        sink += buf[0] + buf[buf.size() - 1];
      }
    auto a1 = std::chrono::high_resolution_clock::now();
    // NEW
    for (int r = 0; r < kReps; r++)
      for (int s = 0; s < kNumValidStrategies; s++) {
        CoeffOrder_NEW<false>(s, buf.data());
        sink += buf[0] + buf[buf.size() - 1];
      }
    auto a2 = std::chrono::high_resolution_clock::now();
    t_old += std::chrono::duration<double, std::milli>(a1 - a0).count();
    t_new += std::chrono::duration<double, std::milli>(a2 - a1).count();
  }
  printf("order-gen ms: OLD=%.2f NEW=%.2f  speedup=%.3fx  (sink=%llu)\n",
         t_old, t_new, t_old / t_new, (unsigned long long)sink);
  return fails == 0 ? 0 : 1;
}
