// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DEC_CACHE_H_
#define LIB_JXL_DEC_CACHE_H_

#include <jxl/decode.h>
#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <hwy/base.h>  // HWY_ALIGN_MAX
#include <memory>
#include <vector>

#include "lib/jxl/base/common.h"  // kMaxNumPasses
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/render_pipeline/render_pipeline.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"
#include "lib/jxl/render_pipeline/stage_upsampling.h"

namespace jxl {

constexpr size_t kSigmaBorder = 1;
constexpr size_t kSigmaPadding = 2;

struct PixelCallback {
  PixelCallback() = default;
  PixelCallback(JxlImageOutInitCallback init, JxlImageOutRunCallback run,
                JxlImageOutDestroyCallback destroy, void* init_opaque)
      : init(init), run(run), destroy(destroy), init_opaque(init_opaque) {
#if (JXL_IS_DEBUG_BUILD)
    const bool has_init = (init != nullptr);
    const bool has_run = (run != nullptr);
    const bool has_destroy = (destroy != nullptr);
    const bool healthy = (has_init == has_run) && (has_run == has_destroy);
    JXL_DASSERT(healthy);
#endif
  }

  bool IsPresent() const { return run != nullptr; }

  void* Init(size_t num_threads, size_t num_pixels) const {
    return init(init_opaque, num_threads, num_pixels);
  }

  JxlImageOutInitCallback init = nullptr;
  JxlImageOutRunCallback run = nullptr;
  JxlImageOutDestroyCallback destroy = nullptr;
  void* init_opaque = nullptr;
};

struct ImageOutput {
  // Pixel format of the output pixels, used for buffer and callback output.
  JxlPixelFormat format;
  // Output bit depth for unsigned data types, used for float to int conversion.
  size_t bits_per_sample;
  // Callback for line-by-line output.
  PixelCallback callback;
  // Pixel buffer for image output.
  void* buffer;
  size_t buffer_size;
  // Length of a row of image_buffer in bytes (based on oriented width).
  size_t stride;
};

// Temp images required for decoding a single group. Reduces memory allocations
// for large images because we only initialize min(#threads, #groups) instances.
struct HWY_ALIGN_MAX GroupDecCache {
  // Allocates entropy predictor planes (num_nzeroes) for the given pass count.
  // No-op for passes whose plane is already allocated.
  Status EnsureEntropyPredictors(JxlMemoryManager* memory_manager,
                                  size_t num_passes);

  // Allocates the dequant/scratch arena sized for max_block_area coefficients.
  // max_block_area: precomputed from the frame's used_acs bitmap (see
  // PassesDecoderState::max_ac_block_area). Passing it directly avoids
  // re-scanning all 27 strategies on every group call.
  // No-op if the arena is already large enough.
  Status EnsureRenderWorkspace(JxlMemoryManager* memory_manager,
                               size_t max_block_area);

  // Convenience: calls both. Existing call sites need no changes.
  Status InitOnce(JxlMemoryManager* memory_manager, size_t num_passes,
                  size_t max_block_area) {
    JXL_RETURN_IF_ERROR(EnsureEntropyPredictors(memory_manager, num_passes));
    JXL_RETURN_IF_ERROR(EnsureRenderWorkspace(memory_manager, max_block_area));
    return true;
  }

  Status InitDCBufferOnce(JxlMemoryManager* memory_manager) {
    if (dc_buffer.xsize() == 0) {
      JXL_ASSIGN_OR_RETURN(
          dc_buffer,
          ImageF::Create(memory_manager,
                         kGroupDimInBlocks + kRenderPipelineXOffset * 2,
                         kGroupDimInBlocks + 4));
    }
    return true;
  }

  // Scratch space used by DecGroupImpl().
  float* dec_group_block;
  int32_t* dec_group_qblock;
  int16_t* dec_group_qblock16;

  // For TransformToPixels.
  float* scratch_space;
  // scratch_space is never used at the same time as dec_group_qblock, and only
  // one of dec_group_qblock / dec_group_qblock16 is ever used. We exploit this:
  // all three (scratch_space, dec_group_qblock, dec_group_qblock16) are aliased
  // onto the trailing 4 transform-scratch float blocks of the single arena
  // below. qblock is fully consumed by dequant_block before scratch_space is
  // touched (see DequantLane loop in dec_group.cc), so the lifetimes do not
  // overlap. See EnsureRenderWorkspace() for the layout proof.

  // AC decoding. Stored value is the AC nonzero count averaged over the covered
  // blocks, i.e. (nzeros + covered - 1) >> log2_covered, always in [0, 63], so a
  // single byte per plane suffices (was Image3I = 4 bytes).
  Image3<uint8_t> num_nzeroes[kMaxNumPasses];

  // Buffer for DC upsampling.
  ImageF dc_buffer;

 private:
  // One grow-only arena: 3 float blocks for dequantized coefficients + 4 float
  // blocks of transform scratch. The quantized-coefficient buffers (3x int32 or
  // 3x int16, both shorter-lived and mutually exclusive with scratch) alias the
  // scratch region. 7 float blocks vs the old 7f + 3*int32 + 3*int16 = ~39%
  // less per-worker scratch at the largest transform shape.
  AlignedMemory float_memory_;
  size_t max_block_area_ = 0;
};

// Per-frame decoder state. All the images here should be accessed through a
// group rect (either with block units or pixel units).
struct PassesDecoderState {
  explicit PassesDecoderState(JxlMemoryManager* memory_manager)
      : shared_storage(memory_manager),
        frame_storage_for_referencing(memory_manager) {}

  PassesSharedState shared_storage;
  // Allows avoiding copies for encoder loop.
  const PassesSharedState* JXL_RESTRICT shared = &shared_storage;

  // 8x upsampling stage for DC.
  std::unique_ptr<RenderPipelineStage> upsampler8x;

  // For ANS decoding.
  std::vector<ANSCode> code;
  std::vector<std::vector<uint8_t>> context_map;

  // Multiplier to be applied to the quant matrices of the x channel.
  float x_dm_multiplier;
  float b_dm_multiplier;

  // Sigma values for EPF.
  ImageF sigma;

  // Image dimensions before applying undo_orientation.
  size_t width;
  size_t height;
  ImageOutput main_output;
  std::vector<ImageOutput> extra_output;

  // Whether to use int16 float-XYB-to-uint8-srgb conversion.
  bool fast_xyb_srgb8_conversion;

  // If true, the RGBA output will be unpremultiplied before writing to the
  // output.
  bool unpremul_alpha;

  // The render pipeline will apply this orientation to bring the image to the
  // intended display orientation.
  Orientation undo_orientation;

  // Used for seeding noise.
  size_t visible_frame_index = 0;
  size_t nonvisible_frame_index = 0;

  // Keep track of the transform types used.
  std::atomic<uint32_t> used_acs{0};

  // Maximum block area (in coefficients) required by any strategy in used_acs.
  // Computed once in InitForAC; passed to GroupDecCache::InitOnce so each
  // group skips the 27-strategy scan.
  size_t max_ac_block_area = 0;

  // Storage for coefficients if in "accumulate" mode.
  std::unique_ptr<ACImage> coefficients = make_unique<ACImageT<int32_t>>();

  // Per-block AC-occupancy bitmask for the coefficient store. bit0=X, bit1=Y,
  // bit2=B nonzero AC. OR'd across all accumulated passes. Valid only when
  // !coefficients->IsEmpty(). Sized lazily to num_groups * max_blocks_per_group.
  std::vector<uint8_t> ac_occupancy;

  // Rendering pipeline.
  std::unique_ptr<RenderPipeline> render_pipeline;

  // Storage for the current frame if it can be referenced by future frames.
  ImageBundle frame_storage_for_referencing;

  // Per-thread group decoding scratch. Owned here (rather than in FrameDecoder)
  // so the grow-only buffers survive across frames: a fresh FrameDecoder is
  // constructed per frame, so keeping these here avoids re-allocating the
  // num_nzeroes planes, the dequant/scratch arena and the DC buffer on every
  // animation frame / decoder reuse. Pure scratch, no cross-frame state.
  std::vector<GroupDecCache> group_dec_caches;

  struct PipelineOptions {
    bool use_slow_render_pipeline;
    bool coalescing;
    bool render_spotcolors;
    bool render_noise;
  };

  JxlMemoryManager* memory_manager() const { return shared->memory_manager; }

  Status PreparePipeline(const FrameHeader& frame_header,
                         const ImageMetadata* metadata, ImageBundle* decoded,
                         PipelineOptions options);

  // Information for colour conversions.
  OutputEncodingInfo output_encoding_info;

  // Initializes decoder-specific structures using information from *shared.
  Status Init(const FrameHeader& frame_header) {
    JxlMemoryManager* memory_manager = this->memory_manager();
    x_dm_multiplier = std::pow(1 / (1.25f), frame_header.x_qm_scale - 2.0f);
    b_dm_multiplier = std::pow(1 / (1.25f), frame_header.b_qm_scale - 2.0f);

    main_output.callback = PixelCallback();
    main_output.buffer = nullptr;
    extra_output.clear();

    fast_xyb_srgb8_conversion = false;
    unpremul_alpha = false;
    undo_orientation = Orientation::kIdentity;

    used_acs = 0;

    upsampler8x = GetUpsamplingStage(memory_manager,
                                     shared->metadata->transform_data, 0, 3);
    if (frame_header.loop_filter.epf_iters > 0) {
      const size_t sigma_x = shared->frame_dim.xsize_blocks + 2 * kSigmaPadding;
      const size_t sigma_y = shared->frame_dim.ysize_blocks + 2 * kSigmaPadding;
      // Reuse the existing sigma plane across frames when the dimensions are
      // unchanged (e.g. animation): the EPF stages fully overwrite every sigma
      // value they consume, so a fresh allocation is unnecessary.
      if (sigma.xsize() != sigma_x || sigma.ysize() != sigma_y) {
        JXL_ASSIGN_OR_RETURN(
            sigma, ImageF::Create(memory_manager, sigma_x, sigma_y));
      }
    }
    return true;
  }

  // Initialize the decoder state after all of DC is decoded.
  Status InitForAC(size_t num_passes, ThreadPool* pool);
};

}  // namespace jxl

#endif  // LIB_JXL_DEC_CACHE_H_
