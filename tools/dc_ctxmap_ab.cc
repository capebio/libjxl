// Standalone byte-exact equivalence + timing harness for the DequantDC
// DC-context-map builder (lib/jxl/compressed_dc.cc, num_dc_ctxs>1 branch).
//
// Proves OLD (generic per-pixel) vs NEW (#5 4:4:4 specialized + #6 subsampled
// native-resolution chroma reuse) produce IDENTICAL quant_dc bytes across:
//   - 4:4:4, 4:2:2, 4:2:0, 4:4:0 (mixed) subsampling
//   - 0..3 thresholds per channel, several threshold sets
//   - several rect sizes incl. odd widths/heights
//   - deterministic pseudo-random + patterned DC integer planes
// then times OLD vs NEW interleaved (start-rotated per pass to cancel drift).
//
// No libjxl build needed: the inner loops are mirrored verbatim from source.
// Build: clang++ -std=c++17 -O2 tools/dc_ctxmap_ab.cc -o dc_ctxmap_ab
//
// Source of truth: lib/jxl/compressed_dc.cc DequantDC @ branch
//                  perf/dec-compressed-dc-ctxmap-jun29-q7x (off main 0ba69efd).

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

using std::size_t;

// A native-resolution integer plane (row-major).
struct Plane {
  size_t w = 0, h = 0;
  std::vector<int32_t> v;
  const int32_t* Row(size_t y) const { return v.data() + y * w; }
};

struct Subsampling {
  size_t hsx, hsy, hsb, vsx, vsy, vsb;
  bool Is444() const {
    return hsx == 0 && hsy == 0 && hsb == 0 && vsx == 0 && vsy == 0 && vsb == 0;
  }
};

struct Config {
  const char* name;
  size_t xsize, ysize;
  Subsampling ss;
  std::vector<int> tx, ty, tb;  // dc_thresholds[0..2]
  Plane px, py, pb;             // channel[1]=X, channel[0]=Y, channel[2]=B
};

// Deterministic PRNG (xorshift) — reproducible across machines.
static uint32_t XorShift(uint32_t& s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

static Plane MakePlane(size_t w, size_t h, uint32_t seed, int lo, int hi,
                       bool patterned) {
  Plane p;
  p.w = w;
  p.h = h;
  p.v.resize(w * h);
  uint32_t s = seed ? seed : 0x9e3779b9u;
  const int span = hi - lo + 1;
  for (size_t y = 0; y < h; y++) {
    for (size_t x = 0; x < w; x++) {
      int val;
      if (patterned) {
        // Smooth-ish gradient + occasional spikes to exercise all buckets.
        val = lo + static_cast<int>((x * 3 + y * 7) % span);
        if (((x ^ y) & 7) == 0) val = hi;
      } else {
        val = lo + static_cast<int>(XorShift(s) % static_cast<uint32_t>(span));
      }
      p.v[y * w + x] = val;
    }
  }
  return p;
}

// ---- OLD: generic per-pixel builder (verbatim from source) ----
static void BuildOLD(const Config& c, uint8_t* qdc) {
  const Subsampling& ss = c.ss;
  for (size_t y = 0; y < c.ysize; y++) {
    uint8_t* qdc_row_val = qdc + y * c.xsize;
    const int32_t* quant_row_x = c.px.Row(y >> ss.vsx);
    const int32_t* quant_row_y = c.py.Row(y >> ss.vsy);
    const int32_t* quant_row_b = c.pb.Row(y >> ss.vsb);
    for (size_t x = 0; x < c.xsize; x++) {
      int bucket_x = 0;
      int bucket_y = 0;
      int bucket_b = 0;
      for (int t : c.tx) {
        if (quant_row_x[x >> ss.hsx] > t) bucket_x++;
      }
      for (int t : c.ty) {
        if (quant_row_y[x >> ss.hsy] > t) bucket_y++;
      }
      for (int t : c.tb) {
        if (quant_row_b[x >> ss.hsb] > t) bucket_b++;
      }
      int bucket = bucket_x;
      bucket *= c.tb.size() + 1;
      bucket += bucket_b;
      bucket *= c.ty.size() + 1;
      bucket += bucket_y;
      qdc_row_val[x] = bucket;
    }
  }
}

// ---- NEW: #5 (4:4:4) + #6 (subsampled native-resolution reuse) ----
static void BuildNEW(const Config& c, uint8_t* qdc) {
  const Subsampling& ss = c.ss;
  const int* tx = c.tx.data();
  const int* ty = c.ty.data();
  const int* tb = c.tb.data();
  const size_t nx = c.tx.size();
  const size_t ny = c.ty.size();
  const size_t nb = c.tb.size();
  const size_t base_b = nb + 1;
  const size_t base_y = ny + 1;
  if (ss.Is444()) {
    for (size_t y = 0; y < c.ysize; y++) {
      uint8_t* qdc_row_val = qdc + y * c.xsize;
      const int32_t* quant_row_x = c.px.Row(y);
      const int32_t* quant_row_y = c.py.Row(y);
      const int32_t* quant_row_b = c.pb.Row(y);
      for (size_t x = 0; x < c.xsize; x++) {
        const int32_t vx = quant_row_x[x];
        const int32_t vy = quant_row_y[x];
        const int32_t vb = quant_row_b[x];
        int bucket_x = 0;
        int bucket_y = 0;
        int bucket_b = 0;
        for (size_t i = 0; i < nx; i++) bucket_x += vx > tx[i];
        for (size_t i = 0; i < ny; i++) bucket_y += vy > ty[i];
        for (size_t i = 0; i < nb; i++) bucket_b += vb > tb[i];
        qdc_row_val[x] = static_cast<uint8_t>(
            (bucket_x * base_b + bucket_b) * base_y + bucket_y);
      }
    }
  } else {
    for (size_t y = 0; y < c.ysize; y++) {
      uint8_t* qdc_row_val = qdc + y * c.xsize;
      const int32_t* quant_row_x = c.px.Row(y >> ss.vsx);
      const int32_t* quant_row_y = c.py.Row(y >> ss.vsy);
      const int32_t* quant_row_b = c.pb.Row(y >> ss.vsb);
      size_t cached_xc = ~static_cast<size_t>(0);
      size_t cached_bc = ~static_cast<size_t>(0);
      int bucket_x = 0;
      int bucket_b = 0;
      for (size_t x = 0; x < c.xsize; x++) {
        const size_t xc = x >> ss.hsx;
        if (xc != cached_xc) {
          const int32_t vx = quant_row_x[xc];
          bucket_x = 0;
          for (size_t i = 0; i < nx; i++) bucket_x += vx > tx[i];
          cached_xc = xc;
        }
        const size_t bc = x >> ss.hsb;
        if (bc != cached_bc) {
          const int32_t vb = quant_row_b[bc];
          bucket_b = 0;
          for (size_t i = 0; i < nb; i++) bucket_b += vb > tb[i];
          cached_bc = bc;
        }
        const int32_t vy = quant_row_y[x >> ss.hsy];
        int bucket_y = 0;
        for (size_t i = 0; i < ny; i++) bucket_y += vy > ty[i];
        qdc_row_val[x] = static_cast<uint8_t>(
            (bucket_x * base_b + bucket_b) * base_y + bucket_y);
      }
    }
  }
}

static Config MakeConfig(const char* name, size_t xs, size_t ys, Subsampling ss,
                         std::vector<int> tx, std::vector<int> ty,
                         std::vector<int> tb, uint32_t seed, bool patterned) {
  Config c;
  c.name = name;
  c.xsize = xs;
  c.ysize = ys;
  c.ss = ss;
  c.tx = std::move(tx);
  c.ty = std::move(ty);
  c.tb = std::move(tb);
  // Native plane dims: cover max index reached by (size-1)>>shift.
  auto nw = [](size_t s, size_t sh) { return ((s - 1) >> sh) + 1; };
  c.px = MakePlane(nw(xs, ss.hsx), nw(ys, ss.vsx), seed + 1, -40, 40, patterned);
  c.py = MakePlane(nw(xs, ss.hsy), nw(ys, ss.vsy), seed + 2, -40, 40, patterned);
  c.pb = MakePlane(nw(xs, ss.hsb), nw(ys, ss.vsb), seed + 3, -40, 40, patterned);
  return c;
}

int main() {
  const Subsampling s444{0, 0, 0, 0, 0, 0};
  const Subsampling s422{1, 0, 1, 0, 0, 0};
  const Subsampling s420{1, 0, 1, 1, 0, 1};
  const Subsampling s440{0, 0, 0, 1, 0, 1};  // mixed: vertical-only chroma

  std::vector<std::vector<int>> tsets = {
      {}, {0}, {-10, 10}, {-20, 0, 20}, {-30, -5, 5, 30}};

  std::vector<Config> configs;
  struct SS { const char* tag; Subsampling ss; };
  std::vector<SS> sss = {{"444", s444}, {"422", s422}, {"420", s420}, {"440", s440}};
  std::vector<std::pair<size_t, size_t>> sizes = {
      {64, 64}, {256, 256}, {257, 129}, {130, 258}, {512, 384}};

  uint32_t seed = 12345;
  for (auto& sentry : sss) {
    for (size_t ti = 0; ti < tsets.size(); ti++) {
      for (auto& sz : sizes) {
        // rotate threshold sets per channel for variety
        const auto& tx = tsets[ti];
        const auto& ty = tsets[(ti + 1) % tsets.size()];
        const auto& tb = tsets[(ti + 2) % tsets.size()];
        bool patterned = ((ti + sz.first) & 1) == 0;
        configs.push_back(MakeConfig(sentry.tag, sz.first, sz.second, sentry.ss,
                                     tx, ty, tb, seed, patterned));
        seed += 100;
      }
    }
  }

  // ---- (A) byte-exact equivalence ----
  size_t checks = 0;
  int fails = 0;
  for (auto& c : configs) {
    std::vector<uint8_t> old_out(c.xsize * c.ysize, 0xCC);
    std::vector<uint8_t> new_out(c.xsize * c.ysize, 0x33);
    BuildOLD(c, old_out.data());
    BuildNEW(c, new_out.data());
    checks += old_out.size();
    if (old_out != new_out) {
      // locate first mismatch
      size_t idx = 0;
      while (idx < old_out.size() && old_out[idx] == new_out[idx]) idx++;
      printf("FAIL %s %zux%zu  first diff @%zu old=%u new=%u\n", c.name,
             c.xsize, c.ysize, idx, old_out[idx], new_out[idx]);
      fails++;
    }
  }
  printf("checks=%zu configs=%zu fails=%d -> %s\n", checks, configs.size(),
         fails, fails == 0 ? "ALL BYTE-EXACT" : "MISMATCH");

  // ---- (B) interleaved timing ----
  std::vector<std::vector<uint8_t>> bufs(configs.size());
  for (size_t i = 0; i < configs.size(); i++)
    bufs[i].assign(configs[i].xsize * configs[i].ysize, 0);

  const int kReps = 60;
  const int kPasses = 10;
  volatile uint64_t sink = 0;
  double t_old = 0, t_new = 0;
  for (int pass = 0; pass < kPasses; pass++) {
    bool old_first = (pass & 1) == 0;  // start-rotate
    for (int side = 0; side < 2; side++) {
      bool do_old = (side == 0) == old_first;
      auto a0 = std::chrono::high_resolution_clock::now();
      for (int r = 0; r < kReps; r++) {
        for (size_t i = 0; i < configs.size(); i++) {
          if (do_old)
            BuildOLD(configs[i], bufs[i].data());
          else
            BuildNEW(configs[i], bufs[i].data());
          sink += bufs[i][0] + bufs[i][bufs[i].size() - 1];
        }
      }
      auto a1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(a1 - a0).count();
      if (do_old) t_old += ms; else t_new += ms;
    }
  }
  printf("ctxmap-build ms (all configs): OLD=%.2f NEW=%.2f  speedup=%.3fx  "
         "(sink=%llu)\n",
         t_old, t_new, t_old / t_new, (unsigned long long)sink);

  // ---- (C) subsampled-only timing (where #6 reuse should pay) ----
  std::vector<size_t> sub_idx;
  for (size_t i = 0; i < configs.size(); i++)
    if (!configs[i].ss.Is444()) sub_idx.push_back(i);
  double so = 0, sn = 0;
  for (int pass = 0; pass < kPasses; pass++) {
    bool old_first = (pass & 1) == 0;
    for (int side = 0; side < 2; side++) {
      bool do_old = (side == 0) == old_first;
      auto a0 = std::chrono::high_resolution_clock::now();
      for (int r = 0; r < kReps; r++)
        for (size_t i : sub_idx) {
          if (do_old) BuildOLD(configs[i], bufs[i].data());
          else BuildNEW(configs[i], bufs[i].data());
          sink += bufs[i][0];
        }
      auto a1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(a1 - a0).count();
      if (do_old) so += ms; else sn += ms;
    }
  }
  printf("ctxmap-build ms (subsampled only): OLD=%.2f NEW=%.2f  speedup=%.3fx\n",
         so, sn, so / sn);

  return fails == 0 ? 0 : 1;
}
