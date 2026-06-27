// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/render_pipeline.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"
#include "lib/jxl/render_pipeline/low_memory_render_pipeline.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"
#include "lib/jxl/render_pipeline/simple_render_pipeline.h"

namespace jxl {

Status RenderPipeline::Builder::AddStage(
    std::unique_ptr<RenderPipelineStage> stage) {
  if (!stage) return JXL_FAILURE("internal: no stage to add");
  stages_.push_back(std::move(stage));
  return true;
}

StatusOr<std::unique_ptr<RenderPipeline>> RenderPipeline::Builder::Finalize(
    FrameDimensions frame_dimensions) && {
  // Check that the last stage is not a kInOut stage for any channel, and that
  // there is at least one stage.
  JXL_ENSURE(!stages_.empty());
  for (size_t c = 0; c < num_c_; c++) {
    JXL_ENSURE(stages_.back()->GetChannelMode(c) !=
               RenderPipelineChannelMode::kInOut);
  }

  std::unique_ptr<RenderPipeline> res;
  if (use_simple_implementation_) {
    res = jxl::make_unique<SimpleRenderPipeline>(memory_manager_);
  } else {
    res = jxl::make_unique<LowMemoryRenderPipeline>(memory_manager_);
  }

  // Cumulative padding and channel shifts both obey the same reverse
  // recurrence over stages, so derive them in a single backward pass. For a
  // kInOut stage, padding[i] = ceil(padding[i+1] / 2^shift) + border and
  // shift[i] = shift[i+1] + stage_shift; otherwise both are copied from the
  // next stage. The final stage contributes neither (its rows stay zero).
  const size_t num_stages = stages_.size();
  res->padding_.resize(num_stages);
  res->channel_shifts_.resize(num_stages);
  for (size_t i = 0; i < num_stages; i++) {
    res->padding_[i].resize(num_c_);
    res->channel_shifts_[i].resize(num_c_);
  }
  for (size_t i = num_stages - 1; i-- > 0;) {
    const auto& stage = stages_[i];
    const auto& settings = stage->settings_;
    for (size_t c = 0; c < num_c_; c++) {
      if (stage->GetChannelMode(c) == RenderPipelineChannelMode::kInOut) {
        res->padding_[i][c].first =
            DivCeil(res->padding_[i + 1][c].first, 1 << settings.shift_x) +
            settings.border_x;
        res->padding_[i][c].second =
            DivCeil(res->padding_[i + 1][c].second, 1 << settings.shift_y) +
            settings.border_y;
        res->channel_shifts_[i][c].first =
            res->channel_shifts_[i + 1][c].first + settings.shift_x;
        res->channel_shifts_[i][c].second =
            res->channel_shifts_[i + 1][c].second + settings.shift_y;
      } else {
        res->padding_[i][c] = res->padding_[i + 1][c];
        res->channel_shifts_[i][c] = res->channel_shifts_[i + 1][c];
      }
    }
  }

  res->frame_dimensions_ = frame_dimensions;
  res->group_completed_passes_.resize(frame_dimensions.num_groups);
  res->stages_ = std::move(stages_);
  JXL_RETURN_IF_ERROR(res->Init());
  return res;
}

RenderPipelineInput RenderPipeline::GetInputBuffers(size_t group_id,
                                                    size_t thread_id) {
  RenderPipelineInput ret;
  JXL_DASSERT(group_id < group_completed_passes_.size());
  JXL_DASSERT(thread_id < input_buffers_.size());
  ret.group_id_ = group_id;
  ret.thread_id_ = thread_id;
  ret.pipeline_ = this;
  // Reuse this thread's descriptor slot instead of allocating a fresh vector.
  std::vector<std::pair<ImageF*, Rect>>* slot = &input_buffers_[thread_id];
  PrepareBuffers(group_id, thread_id, slot);
  ret.buffers_ = slot;
  return ret;
}

Status RenderPipeline::InputReady(
    size_t group_id, size_t thread_id,
    const std::vector<std::pair<ImageF*, Rect>>& buffers) {
  JXL_ENSURE(group_id < group_completed_passes_.size());
  group_completed_passes_[group_id]++;
  for (size_t i = 0; i < buffers.size(); ++i) {
    (void)i;
    JXL_CHECK_PLANE_INITIALIZED(*buffers[i].first, buffers[i].second, i);
  }

  JXL_RETURN_IF_ERROR(ProcessBuffers(group_id, thread_id));
  return true;
}

Status RenderPipeline::PrepareForThreads(size_t num, bool use_group_ids) {
  for (const auto& stage : stages_) {
    JXL_RETURN_IF_ERROR(stage->PrepareForThreads(num));
  }
  // Grow-only: one reusable descriptor slot per thread.
  if (input_buffers_.size() < num) input_buffers_.resize(num);
  JXL_RETURN_IF_ERROR(PrepareForThreadsInternal(num, use_group_ids));
  return true;
}

Status RenderPipelineInput::Done() {
  JXL_ENSURE(pipeline_);
  // Done() is one-shot: InputReady increments the group's completed-pass
  // counter, so detach the pipeline first to prevent a second call (or a
  // retry after error) from double-counting the group.
  RenderPipeline* pipeline = pipeline_;
  pipeline_ = nullptr;
  JXL_DASSERT(buffers_ != nullptr);
  JXL_RETURN_IF_ERROR(pipeline->InputReady(group_id_, thread_id_, *buffers_));
  return true;
}

}  // namespace jxl
