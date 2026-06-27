// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Standalone, dependency-light flipflop harness for the render-pipeline
// per-thread descriptor-reuse optimization. Avoids the codec/PNG/gtest deps of
// render_pipeline_test.cc so it can be built against jxl-internal only.
//
// The optimization: GetInputBuffers() used to allocate a fresh
// std::vector<std::pair<ImageF*, Rect>> on every call (once per decoded group,
// per progressive pass). It now leases a reusable per-thread slot owned by the
// pipeline, so steady-state descriptor allocations are zero. This harness
// measures that allocation traffic directly via a global operator new counter.

#include <jxl/memory_manager.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <new>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/render_pipeline/render_pipeline.h"
#include "lib/jxl/render_pipeline/test_render_pipeline_stages.h"

// Global allocation counter so we can observe std::vector (descriptor) heap
// traffic, which does NOT go through JxlMemoryManager.
static std::atomic<size_t> g_global_allocs{0};

namespace jxl {
namespace {

struct CountingMemoryManager {
  JxlMemoryManager mm;
  size_t alloc_count = 0;
  size_t free_count = 0;
  CountingMemoryManager() {
    mm.opaque = this;
    mm.alloc = &Alloc;
    mm.free = &Free;
  }
  static void* Alloc(void* opaque, size_t size) {
    static_cast<CountingMemoryManager*>(opaque)->alloc_count++;
    return malloc(size);
  }
  static void Free(void* opaque, void* address) {
    if (address) static_cast<CountingMemoryManager*>(opaque)->free_count++;
    free(address);
  }
};

StatusOr<std::unique_ptr<RenderPipeline>> BuildPipeline(JxlMemoryManager* mm,
                                                        FrameDimensions* fd) {
  RenderPipeline::Builder builder(mm, /*num_c=*/1);
  JXL_RETURN_IF_ERROR(builder.AddStage(jxl::make_unique<UpsampleXSlowStage>()));
  JXL_RETURN_IF_ERROR(builder.AddStage(jxl::make_unique<UpsampleYSlowStage>()));
  JXL_RETURN_IF_ERROR(builder.AddStage(jxl::make_unique<Check0FinalStage>()));
  fd->Set(/*xsize_px=*/1024, /*ysize_px=*/1024, /*group_size_shift=*/0,
          /*max_hshift=*/0, /*max_vshift=*/0, /*modular_mode=*/false,
          /*upsampling=*/1);
  return std::move(builder).Finalize(*fd);
}

// Final stage that captures the rendered output into a caller-owned buffer,
// so two pipeline implementations can be compared bit-for-bit.
class CaptureFinalStage : public RenderPipelineStage {
 public:
  CaptureFinalStage(std::vector<float>* out, size_t out_xsize, size_t out_ysize)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        out_(out),
        out_xsize_(out_xsize),
        out_ysize_(out_ysize) {}

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    if (ypos >= out_ysize_) return true;
    const float* JXL_RESTRICT row = GetInputRow(input_rows, 0, 0);
    for (size_t x = 0; x < xsize && xpos + x < out_xsize_; x++) {
      (*out_)[ypos * out_xsize_ + (xpos + x)] = row[x];
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return RenderPipelineChannelMode::kInput;
  }
  const char* GetName() const override { return "TEST::CaptureFinalStage"; }

 private:
  std::vector<float>* out_;
  size_t out_xsize_;
  size_t out_ysize_;
};

// Builds a single-group (no inter-group boundaries) upsample pipeline whose
// output is captured into `*out`. `simple` selects the reference (simple) vs
// the optimized (low-memory) implementation.
StatusOr<std::unique_ptr<RenderPipeline>> BuildCapturePipeline(
    JxlMemoryManager* mm, FrameDimensions* fd, std::vector<float>* out,
    bool simple) {
  RenderPipeline::Builder builder(mm, /*num_c=*/1);
  JXL_RETURN_IF_ERROR(builder.AddStage(jxl::make_unique<UpsampleXSlowStage>()));
  JXL_RETURN_IF_ERROR(builder.AddStage(jxl::make_unique<UpsampleYSlowStage>()));
  // 100x100 px (< the 128 group dim) => exactly one group, so a buffer's
  // local coordinates equal absolute coordinates in both implementations.
  fd->Set(/*xsize_px=*/100, /*ysize_px=*/100, /*group_size_shift=*/0,
          /*max_hshift=*/0, /*max_vshift=*/0, /*modular_mode=*/false,
          /*upsampling=*/1);
  out->assign(static_cast<size_t>(fd->xsize_upsampled) * fd->ysize_upsampled,
              0.0f);
  JXL_RETURN_IF_ERROR(builder.AddStage(jxl::make_unique<CaptureFinalStage>(
      out, fd->xsize_upsampled, fd->ysize_upsampled)));
  if (simple) builder.UseSimpleImplementation();
  return std::move(builder).Finalize(*fd);
}

// 64-bit FNV-1a over the raw bytes of the captured floats.
uint64_t HashFloats(const std::vector<float>& v) {
  uint64_t h = 0xcbf29ce484222325ULL;
  const auto* p = reinterpret_cast<const uint8_t*>(v.data());
  for (size_t i = 0; i < v.size() * sizeof(float); i++) {
    h = (h ^ p[i]) * 0x100000001b3ULL;
  }
  return h;
}

Status RunAllGroupsOnce(RenderPipeline* p, const FrameDimensions& fd) {
  for (size_t i = 0; i < fd.num_groups; i++) {
    auto in = p->GetInputBuffers(i, 0);
    const auto& b = in.GetBuffer(0);
    FillPlane(0.0f, b.first, b.second);
    JXL_RETURN_IF_ERROR(in.Done());
  }
  return true;
}

Status Run() {
  // --- Test 1: low-memory single pass renders correctly. ---
  {
    CountingMemoryManager c;
    FrameDimensions fd;
    JXL_ASSIGN_OR_RETURN(auto p, BuildPipeline(&c.mm, &fd));
    JXL_RETURN_IF_ERROR(p->PrepareForThreads(1, false));
    JXL_RETURN_IF_ERROR(RunAllGroupsOnce(p.get(), fd));
    JXL_ENSURE(p->PassesWithAllInput() == 1u);
    printf("[ok] CallAllGroupsLowMemory (num_groups=%zu)\n", fd.num_groups);
  }

  // --- Test 2: two-pass ClearDone invalidation, shuffled order. ---
  // Exercises descriptor-slot reuse across passes and out-of-order groups, and
  // confirms correctness is preserved when slots are recycled.
  {
    CountingMemoryManager c;
    FrameDimensions fd;
    JXL_ASSIGN_OR_RETURN(auto p, BuildPipeline(&c.mm, &fd));
    JXL_RETURN_IF_ERROR(p->PrepareForThreads(1, false));
    std::vector<size_t> groups(fd.num_groups);
    for (size_t i = 0; i < groups.size(); i++) groups[i] = i;
    std::mt19937 rng(123);
    for (size_t pass = 0; pass < 2; pass++) {
      std::shuffle(groups.begin(), groups.end(), rng);
      for (size_t g : groups) {
        if (pass != 0) p->ClearDone(g);
        auto in = p->GetInputBuffers(g, 0);
        const auto& b = in.GetBuffer(0);
        FillPlane(0.0f, b.first, b.second);
        JXL_RETURN_IF_ERROR(in.Done());
      }
      JXL_ENSURE(p->PassesWithAllInput() == pass + 1);
    }
    printf("[ok] CallAllGroupsLowMemoryInvalidate (2 passes)\n");
  }

  // --- Test 3 / FLIPFLOP: descriptor reuse across GetInputBuffers. ---
  // Each GetInputBuffers used to allocate a fresh descriptor vector. With the
  // per-thread reusable slot, only the first sweep warms capacity; subsequent
  // sweeps allocate nothing for descriptors. Measure GetInputBuffers in
  // isolation (no Done() -> no render-side allocations to confound the count).
  {
    CountingMemoryManager c;
    FrameDimensions fd;
    JXL_ASSIGN_OR_RETURN(auto p, BuildPipeline(&c.mm, &fd));
    JXL_RETURN_IF_ERROR(p->PrepareForThreads(1, false));

    auto sweep = [&]() -> size_t {
      size_t before = g_global_allocs.load(std::memory_order_relaxed);
      for (size_t g = 0; g < fd.num_groups; g++) {
        auto in = p->GetInputBuffers(g, 0);
        (void)in.GetBuffer(0);  // lease released when `in` leaves scope
      }
      return g_global_allocs.load(std::memory_order_relaxed) - before;
    };

    size_t warm = sweep();    // warm-up sweep (slot capacity grows)
    size_t steady = sweep();  // steady-state sweep
    printf(
        "[flipflop] GetInputBuffers x%zu (descriptors only): warm sweep "
        "allocs=%zu  steady sweep allocs=%zu\n",
        fd.num_groups, warm, steady);
    JXL_ENSURE(steady == 0);  // steady-state: zero descriptor allocations
    printf("[ok] DescriptorReuseNoRealloc\n");
  }

  // --- Test 4 / FLIPFLOP: multi-thread descriptor reuse. ---
  // Warm-up descriptor allocations are bounded by the number of active thread
  // slots, NOT by the number of groups; steady-state is zero.
  {
    const size_t kThreads = 4;
    CountingMemoryManager c;
    FrameDimensions fd;
    JXL_ASSIGN_OR_RETURN(auto p, BuildPipeline(&c.mm, &fd));
    JXL_RETURN_IF_ERROR(p->PrepareForThreads(kThreads, false));

    auto sweep = [&]() -> size_t {
      size_t before = g_global_allocs.load(std::memory_order_relaxed);
      for (size_t g = 0; g < fd.num_groups; g++) {
        auto in = p->GetInputBuffers(g, g % kThreads);
        (void)in.GetBuffer(0);
      }
      return g_global_allocs.load(std::memory_order_relaxed) - before;
    };

    size_t warm = sweep();
    size_t steady = sweep();
    printf(
        "[flipflop] GetInputBuffers x%zu over %zu threads: warm sweep "
        "allocs=%zu  steady sweep allocs=%zu\n",
        fd.num_groups, kThreads, warm, steady);
    JXL_ENSURE(warm <= kThreads);  // bounded by active slots, not groups
    JXL_ENSURE(steady == 0);
    printf("[ok] MultiThreadDescriptorReuse\n");
  }

  // --- Test 5 / BYTE-EXACT: optimized (low-memory) vs reference (simple). ---
  // Render a deterministic non-trivial pattern through both pipeline
  // implementations and require bit-identical output. The simple pipeline is
  // the "obviously correct" reference; equality proves the descriptor-reuse
  // path delivers the right buffers. Single group => no inter-group boundary
  // ambiguity, so the two implementations must match exactly.
  {
    auto render = [](bool simple, uint64_t* hash) -> Status {
      CountingMemoryManager c;
      FrameDimensions fd;
      std::vector<float> out;
      JXL_ASSIGN_OR_RETURN(auto p,
                           BuildCapturePipeline(&c.mm, &fd, &out, simple));
      JXL_RETURN_IF_ERROR(p->PrepareForThreads(1, false));
      JXL_ENSURE(fd.num_groups == 1);
      auto in = p->GetInputBuffers(0, 0);
      const auto& b = in.GetBuffer(0);
      const Rect& r = b.second;
      ImageF* img = b.first;
      // Deterministic pattern keyed on buffer-local (== absolute) coordinates.
      for (size_t y = 0; y < r.ysize(); y++) {
        float* row = r.Row(img, y);
        for (size_t x = 0; x < r.xsize(); x++) {
          row[x] = static_cast<float>(((x * 131u + y * 977u) & 255u)) *
                   (1.0f / 255.0f);
        }
      }
      JXL_RETURN_IF_ERROR(in.Done());
      *hash = HashFloats(out);
      return true;
    };
    uint64_t h_lowmem = 0, h_simple = 0;
    JXL_RETURN_IF_ERROR(render(/*simple=*/false, &h_lowmem));
    JXL_RETURN_IF_ERROR(render(/*simple=*/true, &h_simple));
    printf(
        "[byte-exact] low-memory(opt) output FNV=%016llx  "
        "simple(ref) FNV=%016llx\n",
        static_cast<unsigned long long>(h_lowmem),
        static_cast<unsigned long long>(h_simple));
    JXL_ENSURE(h_lowmem == h_simple);
    printf("[ok] OptimizedPipelineMatchesReference\n");
  }

  printf("ALL PASS\n");
  return true;
}

}  // namespace
}  // namespace jxl

void* operator new(std::size_t n) {
  g_global_allocs.fetch_add(1, std::memory_order_relaxed);
  void* p = std::malloc(n ? n : 1);
  if (!p) throw std::bad_alloc();
  return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void* operator new[](std::size_t n) {
  g_global_allocs.fetch_add(1, std::memory_order_relaxed);
  void* p = std::malloc(n ? n : 1);
  if (!p) throw std::bad_alloc();
  return p;
}
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

int main() { return jxl::Run() ? 0 : 1; }
