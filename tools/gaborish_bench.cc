// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Standalone OLD-vs-NEW parity + timing driver for GaborishInverse.
//
// The shipped GaborishInverse (lib/jxl/enc_gaborish.cc) was changed to filter
// into an xrect-sized scratch tile instead of copying a full plane. This driver
// links the new implementation and re-implements the original full-plane-scratch
// version locally as GaborishInverseOld, so we can prove byte-for-byte identical
// output (including pixels outside xrect after the plane rotation) and measure
// the partial-rect allocation/copy win.
//
// Usage:
//   gaborish_bench verify  [size]           -> exit 0 iff OLD == NEW byte-exact
//                                              (checks full + partial rect)
//   gaborish_bench old|new full|partial size reps
//                                           -> run impl `reps` times over rect;
//                                              process wall-clock is the measure.

#include <jxl/memory_manager.h>

#include <hwy/base.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/convolve.h"
#include "lib/jxl/enc_gaborish.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/memory_manager_internal.h"

namespace jxl {

// Verbatim copy of the original (pre-optimization) GaborishInverse: allocates a
// full-plane scratch image and copies all of plane 2 into it.
static Status GaborishInverseOld(Image3F* in_out, const Rect& rect,
                                 const float mul[3], ThreadPool* pool) {
  JxlMemoryManager* memory_manager = in_out->memory_manager();
  WeightsSymmetric5 weights[3];
  static const float kGaborish[5] = {
      -0.09495815671340026, -0.041031725066768575,  0.013710004822696948,
      0.006510206083837737, -0.0014789063378272242,
  };
  for (int i = 0; i < 3; ++i) {
    double sum = 1.0 + mul[i] * 4 *
                           (kGaborish[0] + kGaborish[1] + kGaborish[2] +
                            kGaborish[4] + 2 * kGaborish[3]);
    if (sum < 1e-5) {
      sum = 1e-5;
    }
    const float normalize = static_cast<float>(1.0 / sum);
    const float normalize_mul = mul[i] * normalize;
    weights[i] = WeightsSymmetric5{{HWY_REP4(normalize)},
                                   {HWY_REP4(normalize_mul * kGaborish[0])},
                                   {HWY_REP4(normalize_mul * kGaborish[2])},
                                   {HWY_REP4(normalize_mul * kGaborish[1])},
                                   {HWY_REP4(normalize_mul * kGaborish[4])},
                                   {HWY_REP4(normalize_mul * kGaborish[3])}};
  }
  ImageF temp;
  JXL_ASSIGN_OR_RETURN(temp,
                       ImageF::Create(memory_manager, in_out->Plane(2).xsize(),
                                      in_out->Plane(2).ysize()));
  JXL_RETURN_IF_ERROR(CopyImageTo(in_out->Plane(2), &temp));
  Rect xrect = rect.Extend(3, Rect(*in_out));
  JXL_RETURN_IF_ERROR(Symmetric5(in_out->Plane(0), xrect, weights[0], pool,
                                 &in_out->Plane(2), xrect));
  JXL_RETURN_IF_ERROR(Symmetric5(in_out->Plane(1), xrect, weights[1], pool,
                                 &in_out->Plane(0), xrect));
  JXL_RETURN_IF_ERROR(
      Symmetric5(temp, xrect, weights[2], pool, &in_out->Plane(1), xrect));
  in_out->Plane(0).Swap(in_out->Plane(1));
  in_out->Plane(0).Swap(in_out->Plane(2));
  return true;
}

namespace {

JxlMemoryManager* Mm() {
  static JxlMemoryManager mm;
  static const bool init = MemoryManagerInit(&mm, nullptr);
  (void)init;
  return &mm;
}

void CopyBase(const Image3F& from, Image3F* to) {
  for (size_t c = 0; c < 3; ++c) {
    (void)CopyImageTo(from.Plane(c), &to->Plane(c));
  }
}

Image3F Create3(size_t S) {
  StatusOr<Image3F> r = Image3F::Create(Mm(), S, S);
  if (!r.ok()) {
    std::fprintf(stderr, "Image3F::Create failed for S=%zu\n", S);
    std::abort();
  }
  return std::move(r).value_();
}

// Deterministic pseudo-random base image in [0,1).
Status MakeBase(size_t S, Image3F* out) {
  *out = Create3(S);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < S; ++y) {
      float* row = out->PlaneRow(c, y);
      for (size_t x = 0; x < S; ++x) row[x] = dist(rng);
    }
  }
  return true;
}

bool PlanesEqual(const Image3F& a, const Image3F& b) {
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < a.ysize(); ++y) {
      const float* ra = a.PlaneRow(c, y);
      const float* rb = b.PlaneRow(c, y);
      if (std::memcmp(ra, rb, a.xsize() * sizeof(float)) != 0) {
        // Pinpoint first differing pixel for diagnostics.
        for (size_t x = 0; x < a.xsize(); ++x) {
          if (ra[x] != rb[x]) {
            std::fprintf(stderr, "MISMATCH c=%zu y=%zu x=%zu old=%a new=%a\n", c,
                         y, x, ra[x], rb[x]);
            return false;
          }
        }
        return false;
      }
    }
  }
  return true;
}

Rect MakeRect(bool full, size_t S) {
  if (full) return Rect(0, 0, S, S);
  // Block-aligned interior rect, mirroring the streaming/group case.
  const size_t q = (S / 4) & ~size_t{7};
  const size_t h = (S / 2) & ~size_t{7};
  return Rect(q, q, h, h);
}

int RunVerify(size_t S) {
  Image3F base;
  if (!MakeBase(S, &base)) return 2;
  const float mul[3] = {1.0f, 1.0f, 1.0f};
  for (bool full : {true, false}) {
    Rect rect = MakeRect(full, S);
    Image3F a = Create3(S);
    Image3F b = Create3(S);
    CopyBase(base, &a);
    CopyBase(base, &b);
    if (!GaborishInverseOld(&a, rect, mul, nullptr)) return 2;
    if (!GaborishInverse(&b, rect, mul, nullptr)) return 2;
    if (!PlanesEqual(a, b)) {
      std::fprintf(stderr, "PARITY FAIL rect=%s S=%zu\n",
                   full ? "full" : "partial", S);
      return 1;
    }
    std::fprintf(stderr, "parity ok rect=%s S=%zu\n", full ? "full" : "partial",
                 S);
  }
  std::printf("VERIFY OK\n");
  return 0;
}

int RunTiming(bool use_new, bool full, size_t S, int reps) {
  Image3F base;
  if (!MakeBase(S, &base)) return 2;
  const float mul[3] = {1.0f, 1.0f, 1.0f};
  Rect rect = MakeRect(full, S);
  Image3F work = Create3(S);
  double sum = 0.0;
  for (int r = 0; r < reps; ++r) {
    CopyBase(base, &work);  // restore input (gaborish is in-place)
    Status ok = use_new ? GaborishInverse(&work, rect, mul, nullptr)
                        : GaborishInverseOld(&work, rect, mul, nullptr);
    if (!ok) return 2;
    sum += work.PlaneRow(1, S / 2)[S / 2];
  }
  std::fprintf(stderr, "sum=%.6f\n", sum);
  return 0;
}

}  // namespace
}  // namespace jxl

int main(int argc, char** argv) {
  using namespace jxl;
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s verify [size] | %s old|new full|partial size reps\n",
                 argv[0], argv[0]);
    return 64;
  }
  std::string mode = argv[1];
  if (mode == "verify") {
    size_t S = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 256;
    return RunVerify(S);
  }
  if (argc < 5) {
    std::fprintf(stderr, "timing needs: old|new full|partial size reps\n");
    return 64;
  }
  bool use_new = (mode == "new");
  bool full = (std::string(argv[2]) == "full");
  size_t S = std::strtoull(argv[3], nullptr, 10);
  int reps = std::atoi(argv[4]);
  return RunTiming(use_new, full, S, reps);
}
