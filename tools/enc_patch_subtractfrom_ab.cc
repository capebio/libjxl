// Standalone A/B harness for PatchDictionaryEncoder::SubtractFrom loop reorder.
//
// Models the exact inner logic of the OLD (per-pixel blend dispatch) vs NEW
// (mode-hoisted, plane-major) implementations on synthetic float planes.
// Verifies byte-exact equality of the resulting opsin buffers and times both
// interleaved with start-rotation to cancel thermal drift.
//
// Build (clang-cl, no libjxl needed):
//   clang-cl /O2 /std:c++17 /EHsc tools/enc_patch_subtractfrom_ab.cc /Fe:ab.exe
// Run:
//   ab.exe

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

enum class Mode { kNone = 0, kReplace = 1, kAdd = 2 };

struct Patch {
  uint32_t bx, by;     // dest top-left in opsin
  uint32_t x0, y0;     // src top-left in reference
  uint32_t xsize, ysize;
  Mode mode;
};

struct Planes {
  std::vector<float> p[3];
  size_t stride;
  size_t h;
};

static Planes MakePlanes(size_t w, size_t h, uint32_t seed) {
  Planes pl;
  pl.stride = w;
  pl.h = h;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  for (int c = 0; c < 3; c++) {
    pl.p[c].resize(w * h);
    for (auto& v : pl.p[c]) v = d(rng);
  }
  return pl;
}

// Real code resolves active patches per row via an interval structure, not a
// linear scan; precompute the per-row lists once so timing isolates the
// subtract loops (the only thing this change touches).
using RowLists = std::vector<std::vector<size_t>>;
static RowLists BuildRowLists(const std::vector<Patch>& patches, size_t h) {
  RowLists rl(h);
  for (size_t i = 0; i < patches.size(); i++) {
    const Patch& p = patches[i];
    for (size_t y = p.by; y < p.by + p.ysize && y < h; y++) rl[y].push_back(i);
  }
  return rl;
}

// ---- OLD: per-pixel blend dispatch, channel-inner ----
static void SubtractOld(const std::vector<Patch>& patches, const RowLists& rl,
                        const Planes& ref, Planes* opsin) {
  for (size_t y = 0; y < opsin->h; y++) {
    float* rows[3] = {opsin->p[0].data() + y * opsin->stride,
                      opsin->p[1].data() + y * opsin->stride,
                      opsin->p[2].data() + y * opsin->stride};
    for (size_t pos_idx : rl[y]) {
      const Patch& pos = patches[pos_idx];
      Mode mode = pos.mode;
      size_t by = pos.by, bx = pos.bx, xsize = pos.xsize;
      size_t iy = y - by;
      const float* ref_rows[3] = {
          ref.p[0].data() + (pos.y0 + iy) * ref.stride + pos.x0,
          ref.p[1].data() + (pos.y0 + iy) * ref.stride + pos.x0,
          ref.p[2].data() + (pos.y0 + iy) * ref.stride + pos.x0};
      for (size_t ix = 0; ix < xsize; ix++) {
        for (size_t c = 0; c < 3; c++) {
          if (mode == Mode::kAdd) {
            rows[c][bx + ix] -= ref_rows[c][ix];
          } else if (mode == Mode::kReplace) {
            rows[c][bx + ix] = 0;
          } else if (mode == Mode::kNone) {
            // nothing
          }
        }
      }
    }
  }
}

// ---- NEW: mode-hoisted, plane-major ----
static void SubtractNew(const std::vector<Patch>& patches, const RowLists& rl,
                        const Planes& ref, Planes* opsin) {
  for (size_t y = 0; y < opsin->h; y++) {
    float* rows[3] = {opsin->p[0].data() + y * opsin->stride,
                      opsin->p[1].data() + y * opsin->stride,
                      opsin->p[2].data() + y * opsin->stride};
    for (size_t pos_idx : rl[y]) {
      const Patch& pos = patches[pos_idx];
      Mode mode = pos.mode;
      size_t by = pos.by, bx = pos.bx, xsize = pos.xsize;
      if (mode == Mode::kNone) continue;
      if (mode == Mode::kReplace) {
        for (size_t c = 0; c < 3; c++) std::fill_n(rows[c] + bx, xsize, 0.0f);
        continue;
      }
      // kAdd
      size_t iy = y - by;
      for (size_t c = 0; c < 3; c++) {
        float* dst = rows[c] + bx;
        const float* src = ref.p[c].data() + (pos.y0 + iy) * ref.stride + pos.x0;
        for (size_t ix = 0; ix < xsize; ix++) dst[ix] -= src[ix];
      }
    }
  }
}

int main() {
  const size_t W = 2048, H = 1536;
  // Reference is the packed patch atlas: small and cache-resident in reality,
  // NOT a full-frame random plane. Model it as a compact atlas.
  const size_t AW = 512, AH = 512;
  Planes base = MakePlanes(W, H, 1);
  Planes ref = MakePlanes(AW, AH, 2);

  // Build a text-like scatter of patches. Mostly kAdd (the only mode this
  // encoder emits on color channels); a few kNone/kReplace to exercise branches.
  std::vector<Patch> patches;
  std::mt19937 rng(7);
  std::uniform_int_distribution<uint32_t> px(0, (uint32_t)W - 40);
  std::uniform_int_distribution<uint32_t> py(0, (uint32_t)H - 40);
  std::uniform_int_distribution<uint32_t> sz(8, 48);
  std::uniform_int_distribution<int> modepick(0, 99);
  for (int i = 0; i < 3000; i++) {
    Patch p;
    p.bx = px(rng);
    p.by = py(rng);
    p.xsize = sz(rng);
    p.ysize = sz(rng);
    p.x0 = px(rng) % (uint32_t)(AW - p.xsize);
    p.y0 = py(rng) % (uint32_t)(AH - p.ysize);
    int m = modepick(rng);
    p.mode = (m < 90) ? Mode::kAdd : (m < 95 ? Mode::kReplace : Mode::kNone);
    patches.push_back(p);
  }

  RowLists rl = BuildRowLists(patches, H);

  // --- Byte-exact check ---
  Planes a = base, b = base;
  SubtractOld(patches, rl, ref, &a);
  SubtractNew(patches, rl, ref, &b);
  size_t mism = 0;
  for (int c = 0; c < 3; c++) {
    if (std::memcmp(a.p[c].data(), b.p[c].data(), a.p[c].size() * sizeof(float)))  {
      for (size_t i = 0; i < a.p[c].size(); i++)
        if (a.p[c][i] != b.p[c][i]) mism++;
    }
  }
  printf("byte-exact check: %zu mismatches (bitwise memcmp over 3 planes)\n",
         mism);

  // --- Interleaved timing (rotate start each round) ---
  const int ROUNDS = 200;
  double min_old = 1e30, min_new = 1e30;
  volatile double sink = 0;
  for (int r = 0; r < ROUNDS; r++) {
    if (r & 1) {
      Planes t = base;
      auto t0 = std::chrono::high_resolution_clock::now();
      SubtractOld(patches, rl, ref, &t);
      auto t1 = std::chrono::high_resolution_clock::now();
      min_old = std::min(min_old, std::chrono::duration<double, std::micro>(t1 - t0).count());
      sink += t.p[0][0];
      Planes u = base;
      auto u0 = std::chrono::high_resolution_clock::now();
      SubtractNew(patches, rl, ref, &u);
      auto u1 = std::chrono::high_resolution_clock::now();
      min_new = std::min(min_new, std::chrono::duration<double, std::micro>(u1 - u0).count());
      sink += u.p[0][0];
    } else {
      Planes u = base;
      auto u0 = std::chrono::high_resolution_clock::now();
      SubtractNew(patches, rl, ref, &u);
      auto u1 = std::chrono::high_resolution_clock::now();
      min_new = std::min(min_new, std::chrono::duration<double, std::micro>(u1 - u0).count());
      sink += u.p[0][0];
      Planes t = base;
      auto t0 = std::chrono::high_resolution_clock::now();
      SubtractOld(patches, rl, ref, &t);
      auto t1 = std::chrono::high_resolution_clock::now();
      min_old = std::min(min_old, std::chrono::duration<double, std::micro>(t1 - t0).count());
      sink += t.p[0][0];
    }
  }
  printf("min OLD = %.1f us   min NEW = %.1f us   NEW/OLD = %.3f  (sink=%g)\n",
         min_old, min_new, min_new / min_old, (double)sink);
  return mism == 0 ? 0 : 1;
}
