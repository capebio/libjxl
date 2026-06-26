// Standalone flipflop microbench for XybToRgb generic-Highway specializations.
//
// NOT part of the cmake build (no libjxl link). Header-only Highway, single
// static target (build with -mavx2 / -msse4 / etc.). Measures the two
// x86/wasm-benchable byte-exact wins proposed for dec_xyb-inl.h:
//
//   1. EqualBias: when opsin_biases[0..2] and opsin_biases_cbrt[0..2] are
//      bit-equal (the default opsin transform), materialise one cbrt bias and
//      one neg bias instead of three each -> fewer live vector constants,
//      lower AVX2 register pressure. Byte-exact by construction.
//
//   2. Gray: when the 3x3 inverse matrix has three bit-identical rows (the
//      grayscale OUTPUT-encoding case from SetColorEncoding's srgb_to_luma),
//      compute the matrix product once (3 muls) instead of three times (9
//      muls), store the single result to all three planes. Byte-exact when
//      rows are bit-identical. NB: this is genuine grayscale-image decode, NOT
//      a user grayscale view-filter on a colour image (that lives app-side).
//
// Interleaved A/B with start-rotation to cancel thermal drift; correctness gate
// compares New output vs Old output bit-for-bit.
//
// Build (native AVX2):
//   clang++ -O2 -mavx2 -std=c++17 -DNDEBUG \
//     -I <primary>/external/libjxl-012/third_party/highway \
//     lib/jxl/dec_xyb_standalone_bench.cc -o dec_xyb_bench
//
// CAVEAT (recorded lesson): native AVX2 is a PROXY. wasm SIMD128 codegen can
// diverge; a confirmed AVX2 win must be re-measured on the real wasm build
// before landing into the live XybToRgb / stage_xyb hot path.

#include <hwy/highway.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

namespace hn = hwy::HWY_NAMESPACE;

struct OpsinParams {
  float inverse_opsin_matrix[9 * 4];
  float opsin_biases[4];
  float opsin_biases_cbrt[4];
};

// Default libjxl opsin inverse matrix (intensity_target 255), broadcast x4.
// Values are representative; exact magnitudes don't affect timing, only the
// byte-exact comparison between Old and New (which use the SAME params).
static void InitDefault(OpsinParams* p, bool gray) {
  const float m[9] = {
      11.031566901960783f, -9.866943921568629f, -0.16462299647058826f,
      -3.254147380392157f,  4.418770392156863f, -0.16462299647058826f,
      -3.6588512862745097f, 2.7129230470588235f, 1.9459282392156863f};
  for (int k = 0; k < 9; k++) {
    float v = m[k];
    if (gray) {
      // Three identical rows = luma row (srgb_to_luma * inverse), modelled as
      // the sRGB luminance combination of the matrix columns.
      int col = k % 3;
      const float lum[3] = {0.2126f, 0.7152f, 0.0722f};
      v = lum[0] * m[0 + col] + lum[1] * m[3 + col] + lum[2] * m[6 + col];
    }
    for (int j = 0; j < 4; j++) p->inverse_opsin_matrix[k * 4 + j] = v;
  }
  const float nb = -0.0037930732552754493f;
  const float cb = std::cbrt(nb);
  for (int c = 0; c < 3; c++) {
    p->opsin_biases[c] = nb;
    p->opsin_biases_cbrt[c] = cb;
  }
  p->opsin_biases[3] = p->opsin_biases_cbrt[3] = 1.0f;
}

template <class D, class V>
static HWY_INLINE void XybToRgb_Generic(D d, V ox, V oy, V ob,
                                        const OpsinParams& pp, V* lr, V* lg,
                                        V* lb) {
  const auto neg_bias_rgb = hn::LoadDup128(d, pp.opsin_biases);
  const auto nbr = hn::Broadcast<0>(neg_bias_rgb);
  const auto nbg = hn::Broadcast<1>(neg_bias_rgb);
  const auto nbb = hn::Broadcast<2>(neg_bias_rgb);
  auto gr = hn::Add(oy, ox);
  auto gg = hn::Sub(oy, ox);
  auto gb = ob;
  gr = hn::Sub(gr, hn::Set(d, pp.opsin_biases_cbrt[0]));
  gg = hn::Sub(gg, hn::Set(d, pp.opsin_biases_cbrt[1]));
  gb = hn::Sub(gb, hn::Set(d, pp.opsin_biases_cbrt[2]));
  const auto gr2 = hn::Mul(gr, gr);
  const auto gg2 = hn::Mul(gg, gg);
  const auto gb2 = hn::Mul(gb, gb);
  const auto mr = hn::MulAdd(gr2, gr, nbr);
  const auto mg = hn::MulAdd(gg2, gg, nbg);
  const auto mb = hn::MulAdd(gb2, gb, nbb);
  const float* im = pp.inverse_opsin_matrix;
  *lr = hn::Mul(hn::LoadDup128(d, &im[0 * 4]), mr);
  *lg = hn::Mul(hn::LoadDup128(d, &im[3 * 4]), mr);
  *lb = hn::Mul(hn::LoadDup128(d, &im[6 * 4]), mr);
  *lr = hn::MulAdd(hn::LoadDup128(d, &im[1 * 4]), mg, *lr);
  *lg = hn::MulAdd(hn::LoadDup128(d, &im[4 * 4]), mg, *lg);
  *lb = hn::MulAdd(hn::LoadDup128(d, &im[7 * 4]), mg, *lb);
  *lr = hn::MulAdd(hn::LoadDup128(d, &im[2 * 4]), mb, *lr);
  *lg = hn::MulAdd(hn::LoadDup128(d, &im[5 * 4]), mb, *lg);
  *lb = hn::MulAdd(hn::LoadDup128(d, &im[8 * 4]), mb, *lb);
}

// EqualBias: one cbrt bias + one neg bias (bit-equal across channels).
template <class D, class V>
static HWY_INLINE void XybToRgb_EqualBias(D d, V ox, V oy, V ob,
                                          const OpsinParams& pp, V* lr, V* lg,
                                          V* lb) {
  const auto neg_bias = hn::Set(d, pp.opsin_biases[0]);
  const auto cbrt_bias = hn::Set(d, pp.opsin_biases_cbrt[0]);
  auto gr = hn::Sub(hn::Add(oy, ox), cbrt_bias);
  auto gg = hn::Sub(hn::Sub(oy, ox), cbrt_bias);
  auto gb = hn::Sub(ob, cbrt_bias);
  const auto mr = hn::MulAdd(hn::Mul(gr, gr), gr, neg_bias);
  const auto mg = hn::MulAdd(hn::Mul(gg, gg), gg, neg_bias);
  const auto mb = hn::MulAdd(hn::Mul(gb, gb), gb, neg_bias);
  const float* im = pp.inverse_opsin_matrix;
  *lr = hn::Mul(hn::LoadDup128(d, &im[0 * 4]), mr);
  *lg = hn::Mul(hn::LoadDup128(d, &im[3 * 4]), mr);
  *lb = hn::Mul(hn::LoadDup128(d, &im[6 * 4]), mr);
  *lr = hn::MulAdd(hn::LoadDup128(d, &im[1 * 4]), mg, *lr);
  *lg = hn::MulAdd(hn::LoadDup128(d, &im[4 * 4]), mg, *lg);
  *lb = hn::MulAdd(hn::LoadDup128(d, &im[7 * 4]), mg, *lb);
  *lr = hn::MulAdd(hn::LoadDup128(d, &im[2 * 4]), mb, *lr);
  *lg = hn::MulAdd(hn::LoadDup128(d, &im[5 * 4]), mb, *lg);
  *lb = hn::MulAdd(hn::LoadDup128(d, &im[8 * 4]), mb, *lb);
}

// Gray specialized: three identical matrix rows -> one matrix product.
template <class D, class V>
static HWY_INLINE void XybToRgb_Gray(D d, V ox, V oy, V ob,
                                     const OpsinParams& pp, V* lr, V* lg,
                                     V* lb) {
  const auto neg_bias = hn::Set(d, pp.opsin_biases[0]);
  const auto cbrt_bias = hn::Set(d, pp.opsin_biases_cbrt[0]);
  auto gr = hn::Sub(hn::Add(oy, ox), cbrt_bias);
  auto gg = hn::Sub(hn::Sub(oy, ox), cbrt_bias);
  auto gb = hn::Sub(ob, cbrt_bias);
  const auto mr = hn::MulAdd(hn::Mul(gr, gr), gr, neg_bias);
  const auto mg = hn::MulAdd(hn::Mul(gg, gg), gg, neg_bias);
  const auto mb = hn::MulAdd(hn::Mul(gb, gb), gb, neg_bias);
  const float* im = pp.inverse_opsin_matrix;
  auto lin = hn::Mul(hn::LoadDup128(d, &im[0 * 4]), mr);
  lin = hn::MulAdd(hn::LoadDup128(d, &im[1 * 4]), mg, lin);
  lin = hn::MulAdd(hn::LoadDup128(d, &im[2 * 4]), mb, lin);
  *lr = lin;
  *lg = lin;
  *lb = lin;
}

enum Kernel { kGeneric, kEqualBias, kGray };

static double RunKernel(Kernel kern, const OpsinParams& pp, const float* ix,
                        const float* iy, const float* ib, float* or_, float* og,
                        float* ob_, size_t n, int reps) {
  const hn::ScalableTag<float> d;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < reps; r++) {
    for (size_t x = 0; x < n; x += hn::Lanes(d)) {
      const auto vx = hn::Load(d, ix + x);
      const auto vy = hn::Load(d, iy + x);
      const auto vb = hn::Load(d, ib + x);
      auto lr = hn::Zero(d), lg = hn::Zero(d), lb = hn::Zero(d);
      if (kern == kGeneric)
        XybToRgb_Generic(d, vx, vy, vb, pp, &lr, &lg, &lb);
      else if (kern == kEqualBias)
        XybToRgb_EqualBias(d, vx, vy, vb, pp, &lr, &lg, &lb);
      else
        XybToRgb_Gray(d, vx, vy, vb, pp, &lr, &lg, &lb);
      hn::Store(lr, d, or_ + x);
      hn::Store(lg, d, og + x);
      hn::Store(lb, d, ob_ + x);
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static bool BitEqual(const float* a, const float* b, size_t n) {
  return memcmp(a, b, n * sizeof(float)) == 0;
}

int main() {
  const size_t n = size_t(1) << 20;  // 1M px/channel, vector-aligned
  const int reps = 20;
  const int rounds = 11;

  std::vector<float> ix(n), iy(n), ib(n);
  for (size_t i = 0; i < n; i++) {
    ix[i] = std::sin(0.013 * i) * 0.02f;       // x: ~[-0.02,0.02]
    iy[i] = 0.4f + std::cos(0.021 * i) * 0.4f;  // y: ~[0,0.8]
    ib[i] = 0.4f + std::sin(0.017 * i) * 0.4f;  // b: ~[0,0.8]
  }
  std::vector<float> ra(n), ga(n), ba(n), rb(n), gb(n), bb(n);

  printf("Highway static target: %s, Lanes=%zu\n", hwy::TargetName(HWY_STATIC_TARGET),
         (size_t)hn::Lanes(hn::ScalableTag<float>()));

  // ---- Pair 1: Generic vs EqualBias (default colour transform) ----
  {
    OpsinParams pp;
    InitDefault(&pp, /*gray=*/false);
    RunKernel(kGeneric, pp, ix.data(), iy.data(), ib.data(), ra.data(),
              ga.data(), ba.data(), n, 1);
    RunKernel(kEqualBias, pp, ix.data(), iy.data(), ib.data(), rb.data(),
              gb.data(), bb.data(), n, 1);
    bool ok = BitEqual(ra.data(), rb.data(), n) &&
              BitEqual(ga.data(), gb.data(), n) &&
              BitEqual(ba.data(), bb.data(), n);
    printf("\n[Pair1 Generic vs EqualBias] byte-exact: %s\n", ok ? "YES" : "NO");
    std::vector<double> tg, te;
    for (int r = 0; r < rounds; r++) {
      // rotate start order each round
      if (r & 1) {
        te.push_back(RunKernel(kEqualBias, pp, ix.data(), iy.data(), ib.data(),
                               rb.data(), gb.data(), bb.data(), n, reps));
        tg.push_back(RunKernel(kGeneric, pp, ix.data(), iy.data(), ib.data(),
                               ra.data(), ga.data(), ba.data(), n, reps));
      } else {
        tg.push_back(RunKernel(kGeneric, pp, ix.data(), iy.data(), ib.data(),
                               ra.data(), ga.data(), ba.data(), n, reps));
        te.push_back(RunKernel(kEqualBias, pp, ix.data(), iy.data(), ib.data(),
                               rb.data(), gb.data(), bb.data(), n, reps));
      }
    }
    std::sort(tg.begin(), tg.end());
    std::sort(te.begin(), te.end());
    double mg = tg[rounds / 2], me = te[rounds / 2];
    printf("  Generic   median: %.3f ms\n", mg);
    printf("  EqualBias median: %.3f ms\n", me);
    printf("  delta: %+.2f%% (neg = New faster)\n", (me - mg) / mg * 100.0);
  }

  // ---- Pair 2: Gray-generic vs Gray-specialized (grayscale output) ----
  {
    OpsinParams pp;
    InitDefault(&pp, /*gray=*/true);
    RunKernel(kGeneric, pp, ix.data(), iy.data(), ib.data(), ra.data(),
              ga.data(), ba.data(), n, 1);
    RunKernel(kGray, pp, ix.data(), iy.data(), ib.data(), rb.data(), gb.data(),
              bb.data(), n, 1);
    bool ok = BitEqual(ra.data(), rb.data(), n) &&
              BitEqual(ga.data(), gb.data(), n) &&
              BitEqual(ba.data(), bb.data(), n);
    printf("\n[Pair2 Gray-generic vs Gray-spec] byte-exact: %s\n",
           ok ? "YES" : "NO");
    std::vector<double> tg, te;
    for (int r = 0; r < rounds; r++) {
      if (r & 1) {
        te.push_back(RunKernel(kGray, pp, ix.data(), iy.data(), ib.data(),
                               rb.data(), gb.data(), bb.data(), n, reps));
        tg.push_back(RunKernel(kGeneric, pp, ix.data(), iy.data(), ib.data(),
                               ra.data(), ga.data(), ba.data(), n, reps));
      } else {
        tg.push_back(RunKernel(kGeneric, pp, ix.data(), iy.data(), ib.data(),
                               ra.data(), ga.data(), ba.data(), n, reps));
        te.push_back(RunKernel(kGray, pp, ix.data(), iy.data(), ib.data(),
                               rb.data(), gb.data(), bb.data(), n, reps));
      }
    }
    std::sort(tg.begin(), tg.end());
    std::sort(te.begin(), te.end());
    double mg = tg[rounds / 2], me = te[rounds / 2];
    printf("  Gray-generic median: %.3f ms\n", mg);
    printf("  Gray-spec    median: %.3f ms\n", me);
    printf("  delta: %+.2f%% (neg = New faster)\n", (me - mg) / mg * 100.0);
  }
  return 0;
}
