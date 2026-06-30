// A/B harness for the X-CfL-free dequant specialization in dec_group.cc.
//
// Models the AVX2 codegen of DequantSingleBlock's inner lane (AdjustQuantBias
// + the X/Y/B dequant combine). Two variants:
//   BASE : X = fmadd(x_cc_mul, dequant_y, dequant_x_cc)   (current code)
//   NEW  : X = dequant_x_cc                               (kXCfL == false)
//
// Proves: with x_cc_mul == 0 (the XYB-default X-channel case the
// specialization targets), BASE and NEW produce bit-identical output blocks.
// Control: with x_cc_mul != 0 the two differ, so the harness really exercises
// the X combine. Then times both interleaved with start rotation.
//
// On WASM (simd128, no native FMA) MulAdd lowers to mul+add, so the no-X path
// drops two ops per X lane instead of one; the win is larger there. This
// native bench establishes byte-exactness and direction; WASM decode timing is
// the integrator gate.
//
// Build (from anywhere):
//   clang++ -O3 -mavx2 -mfma -std=c++17 dec_xzero_ab.cc -o xzero_ab

#include <immintrin.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>

namespace {

constexpr int kBlock = 64;       // floats per channel block (8x8 DCT)
constexpr int kLanes = 8;        // AVX2 float lanes

// Faithful port of jxl::AdjustQuantBias (quantizer-inl.h), AVX2.
inline __m256 AdjustQuantBias(const __m256i quant_i, float bias_c,
                              float bias3) {
  const __m256 quant = _mm256_cvtepi32_ps(quant_i);
  const __m256 kSign = _mm256_castsi256_ps(_mm256_set1_epi32(INT32_MIN));
  const __m256 sign = _mm256_and_ps(quant, kSign);
  const __m256 abs_quant = _mm256_andnot_ps(kSign, quant);
  const __m256 is_01 = _mm256_cmp_ps(abs_quant, _mm256_set1_ps(1.125f), _CMP_LT_OQ);
  const __m256 not_0 = _mm256_cmp_ps(abs_quant, _mm256_setzero_ps(), _CMP_GT_OQ);
  const __m256 one_bias =
      _mm256_and_ps(not_0, _mm256_xor_ps(_mm256_set1_ps(bias_c), sign));
  // NegMulAdd(bias3, rcp(quant), quant) = quant - bias3*rcp(quant)
  const __m256 rcp = _mm256_rcp_ps(quant);
  const __m256 bias = _mm256_fnmadd_ps(_mm256_set1_ps(bias3), rcp, quant);
  return _mm256_blendv_ps(bias, one_bias, is_01);
}

template <bool kXCfL>
void DequantBlock(const int32_t* qx, const int32_t* qy, const int32_t* qb,
                  const float* dm, float sdx, float sdy, float sdb,
                  float x_cc, float b_cc, const float* biases, float* out) {
  const __m256 vx_cc = _mm256_set1_ps(x_cc);
  const __m256 vb_cc = _mm256_set1_ps(b_cc);
  const __m256 vsdx = _mm256_set1_ps(sdx);
  const __m256 vsdy = _mm256_set1_ps(sdy);
  const __m256 vsdb = _mm256_set1_ps(sdb);
  for (int k = 0; k < kBlock; k += kLanes) {
    const __m256 x_mul = _mm256_mul_ps(_mm256_loadu_ps(dm + k), vsdx);
    const __m256 y_mul = _mm256_mul_ps(_mm256_loadu_ps(dm + kBlock + k), vsdy);
    const __m256 b_mul = _mm256_mul_ps(_mm256_loadu_ps(dm + 2 * kBlock + k), vsdb);
    const __m256i qxi = _mm256_loadu_si256((const __m256i*)(qx + k));
    const __m256i qyi = _mm256_loadu_si256((const __m256i*)(qy + k));
    const __m256i qbi = _mm256_loadu_si256((const __m256i*)(qb + k));
    const __m256 dqx_cc = _mm256_mul_ps(AdjustQuantBias(qxi, biases[0], biases[3]), x_mul);
    const __m256 dqy = _mm256_mul_ps(AdjustQuantBias(qyi, biases[1], biases[3]), y_mul);
    const __m256 dqb_cc = _mm256_mul_ps(AdjustQuantBias(qbi, biases[2], biases[3]), b_mul);
    __m256 dqx;
    if (kXCfL) {
      dqx = _mm256_fmadd_ps(vx_cc, dqy, dqx_cc);
    } else {
      dqx = dqx_cc;
    }
    const __m256 dqb = _mm256_fmadd_ps(vb_cc, dqy, dqb_cc);
    _mm256_storeu_ps(out + k, dqx);
    _mm256_storeu_ps(out + kBlock + k, dqy);
    _mm256_storeu_ps(out + 2 * kBlock + k, dqb);
  }
}

struct Tile {
  std::vector<int32_t> qx, qy, qb;
  std::vector<float> dm;
};

double now_ns() {
  using namespace std::chrono;
  return duration<double, std::nano>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main() {
  const int kTiles = 4096;  // color tiles worth of single blocks
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> qd(-64, 64);          // quantized coeffs
  std::uniform_real_distribution<float> md(0.01f, 4.0f);   // dequant matrix
  float biases[4] = {0.0625f, 0.0625f, 0.0625f, 0.145f};   // ~kZeroBias-ish

  std::vector<Tile> tiles(kTiles);
  for (auto& t : tiles) {
    t.qx.resize(kBlock); t.qy.resize(kBlock); t.qb.resize(kBlock);
    t.dm.resize(3 * kBlock);
    for (int i = 0; i < kBlock; i++) {
      t.qx[i] = qd(rng); t.qy[i] = qd(rng); t.qb[i] = qd(rng);
    }
    for (int i = 0; i < 3 * kBlock; i++) t.dm[i] = md(rng);
  }

  const float sdx = 0.013f, sdy = 0.011f, sdb = 0.017f;
  const float b_cc = 0.42f;  // nonzero B factor (XYB default-ish)

  // ---- Byte-exact check: x_cc == 0 ----
  std::vector<float> ob(3 * kBlock), on(3 * kBlock);
  int mism = 0;
  for (auto& t : tiles) {
    DequantBlock<true>(t.qx.data(), t.qy.data(), t.qb.data(), t.dm.data(),
                       sdx, sdy, sdb, /*x_cc=*/0.0f, b_cc, biases, ob.data());
    DequantBlock<false>(t.qx.data(), t.qy.data(), t.qb.data(), t.dm.data(),
                        sdx, sdy, sdb, /*x_cc=*/0.0f, b_cc, biases, on.data());
    if (memcmp(ob.data(), on.data(), 3 * kBlock * sizeof(float)) != 0) mism++;
  }
  printf("byte-exact (x_cc==0): %s  (%d/%d tiles mismatched)\n",
         mism == 0 ? "PASS" : "FAIL", mism, kTiles);

  // ---- Control: x_cc != 0 must differ ----
  int differ = 0;
  for (auto& t : tiles) {
    DequantBlock<true>(t.qx.data(), t.qy.data(), t.qb.data(), t.dm.data(),
                       sdx, sdy, sdb, /*x_cc=*/0.31f, b_cc, biases, ob.data());
    DequantBlock<false>(t.qx.data(), t.qy.data(), t.qb.data(), t.dm.data(),
                        sdx, sdy, sdb, /*x_cc=*/0.31f, b_cc, biases, on.data());
    if (memcmp(ob.data(), on.data(), 3 * kBlock * sizeof(float)) != 0) differ++;
  }
  printf("control (x_cc!=0): %s  (%d/%d tiles differ as expected)\n",
         differ == kTiles ? "PASS" : "FAIL", differ, kTiles);

  // ---- Interleaved timing (start-rotated, sum to defeat DCE) ----
  const int kReps = 2000;
  double t_base = 0, t_new = 0;
  volatile float sink = 0;
  for (int rep = 0; rep < kReps; rep++) {
    const bool base_first = (rep & 1) == 0;
    for (int side = 0; side < 2; side++) {
      const bool do_base = (side == 0) == base_first;
      double s = now_ns();
      float acc = 0;
      for (auto& t : tiles) {
        if (do_base) {
          DequantBlock<true>(t.qx.data(), t.qy.data(), t.qb.data(), t.dm.data(),
                             sdx, sdy, sdb, 0.0f, b_cc, biases, ob.data());
        } else {
          DequantBlock<false>(t.qx.data(), t.qy.data(), t.qb.data(), t.dm.data(),
                              sdx, sdy, sdb, 0.0f, b_cc, biases, ob.data());
        }
        acc += ob[0] + ob[kBlock] + ob[2 * kBlock];
      }
      double e = now_ns() - s;
      if (do_base) t_base += e; else t_new += e;
      sink += acc;
    }
  }
  double per_base = t_base / (kReps * (double)kTiles);
  double per_new = t_new / (kReps * (double)kTiles);
  printf("timing/block: BASE(fmadd) %.3f ns   NEW(no-X) %.3f ns   delta %+.2f%% %s\n",
         per_base, per_new, (per_new - per_base) / per_base * 100.0,
         per_new < per_base ? "(NEW faster)" : "(NEW not faster)");
  printf("sink=%f\n", (float)sink);
  return 0;
}
