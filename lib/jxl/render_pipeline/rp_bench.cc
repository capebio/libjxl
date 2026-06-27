// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Standalone, dependency-light harness for the render-pipeline scratch-reuse
// work. Avoids the codec/PNG/gtest deps of render_pipeline_test.cc so it can be
// built against jxl-internal only. Acts as the allocation-count flipflop:
// repeated PrepareForThreads must not reallocate same-size stage buffers.

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

  // --- Test 3 / FLIPFLOP: thread-count oscillation must not reallocate. ---
  {
    CountingMemoryManager c;
    FrameDimensions fd;
    JXL_ASSIGN_OR_RETURN(auto p, BuildPipeline(&c.mm, &fd));

    auto t0 = std::chrono::steady_clock::now();
    JXL_RETURN_IF_ERROR(p->PrepareForThreads(8, false));
    size_t a1 = c.alloc_count;
    JXL_RETURN_IF_ERROR(p->PrepareForThreads(1, false));
    size_t a2 = c.alloc_count;
    JXL_RETURN_IF_ERROR(p->PrepareForThreads(8, false));
    size_t a3 = c.alloc_count;
    auto t1 = std::chrono::steady_clock::now();

    printf(
        "[flipflop] allocs: prepare(8)=%zu  prepare(1)=%zu  "
        "prepare(8 again)=%zu  (delta after first = %zu)\n",
        a1, a2, a3, a3 - a1);
    printf("[flipflop] 3x PrepareForThreads wall = %.3f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    JXL_ENSURE(a1 > 0);
    JXL_ENSURE(a2 == a1);  // smaller count: no shrink/realloc
    JXL_ENSURE(a3 == a1);  // larger again: same-size reuse, no realloc
    JXL_RETURN_IF_ERROR(RunAllGroupsOnce(p.get(), fd));
    JXL_ENSURE(p->PassesWithAllInput() == 1u);
    printf("[ok] PrepareForThreadsReuseNoRealloc\n");
  }

  // --- Test 4 / FLIPFLOP: descriptor reuse across GetInputBuffers. ---
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

  // --- Test 5 / FLIPFLOP: multi-thread descriptor reuse. ---
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
