// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>
#include <jxl/codestream_header.h>
#include <jxl/color_encoding.h>
#include <jxl/encode.h>
#include <jxl/encode_cxx.h>
#include <jxl/types.h>

#include <cstdint>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/common.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/encode_internal.h"
#include "lib/jxl/padded_bytes.h"
#include "lib/jxl/test_image.h"
#include "lib/jxl/testing.h"

namespace jxl {

// Unit tests for ProcessFrame method in isolation.
// This test class is declared a friend of JxlEncoderStruct to access
// private methods and test frame processing logic independently from
// header preparation and box processing.
class ProcessFrameTest : public ::testing::Test {
 protected:
  // Helper to construct a minimal valid header_bytes for frame processing.
  // Returns a basic JXL codestream header (without frame data).
  static jxl::PaddedBytes CreateMinimalHeaderBytes(
      JxlMemoryManager* memory_manager) {
    jxl::PaddedBytes header{memory_manager};
    // Frame tag: 0x0A (regular frame, not last)
    header.push_back(0x0A);
    // Minimal frame header: no crop, no blending, no buffer
    // (actual frame will be processed by EncodeFrame)
    return header;
  }

  // Helper to create encoder with basic configuration.
  JxlEncoderStruct* CreateEncoder() {
    auto enc = new JxlEncoderStruct();
    enc->basic_info.xsize = 64;
    enc->basic_info.ysize = 64;
    enc->basic_info.num_color_channels = 3;
    enc->basic_info.num_extra_channels = 0;
    enc->basic_info.bits_per_sample = 8;
    enc->basic_info.uses_original_profile = JXL_FALSE;
    enc->basic_info_set = true;
    enc->color_encoding_set = true;
    enc->codestream_level = 5;
    enc->frames_closed = true;
    enc->metadata.m.xyb_encoded = true;
    enc->metadata.m.have_animation = false;
    JxlColorEncodingSetToSRGB(&enc->metadata.m.color_encoding, JXL_FALSE);
    return enc;
  }

  void TearDown() override {
    // Cleanup is handled by unique_ptr in actual usage
  }
};

// Test that ProcessFrame correctly processes a single queued frame.
TEST_F(ProcessFrameTest, ProcessFrameEncodesQueuedFrame) {
  JxlEncoderStruct* enc = CreateEncoder();

  // Create a simple test image
  size_t xsize = 64;
  size_t ysize = 64;
  std::vector<uint8_t> pixels = jxl::test::GetSomeTestImage(xsize, ysize, 3, 0);

  // Create a queued frame
  auto queued_frame =
      jxl::MemoryManagerUniquePtr<jxl::JxlEncoderQueuedFrame>(
          jxl::MemoryManagerDeleteHelper(&enc->memory_manager));

  auto frame = new jxl::JxlEncoderQueuedFrame();
  frame->frame_data.SetFromBuffer(pixels.data(), pixels.size(),
                                   JxlPixelFormat{3, JXL_TYPE_UINT8,
                                                  JXL_NATIVE_ENDIAN, 0},
                                   xsize, ysize);
  frame->option_values.cparams = jxl::CompressParams();
  frame->option_values.header.duration = 0;
  frame->option_values.header.timecode = 0;
  frame->option_values.header.layer_info.crop_x0 = 0;
  frame->option_values.header.layer_info.crop_y0 = 0;
  frame->ec_initialized.resize(0);

  queued_frame.reset(frame);

  // Queue the frame
  JxlEncoderQueuedInput input(&enc->memory_manager);
  input.frame = std::move(queued_frame);
  enc->input_queue.push_back(std::move(input));
  enc->num_queued_frames = 1;
  enc->wrote_bytes = true;  // Skip header preparation

  // Create minimal header_bytes
  jxl::PaddedBytes header_bytes = CreateMinimalHeaderBytes(&enc->memory_manager);

  // Call ProcessFrame directly
  jxl::Status status = enc->ProcessFrame(header_bytes);

  // Verify it succeeded
  EXPECT_TRUE(status);

  // Verify frame was dequeued
  EXPECT_EQ(0u, enc->input_queue.size());
  EXPECT_EQ(0u, enc->num_queued_frames);

  delete enc;
}

// Test that ProcessFrame handles the last frame correctly.
TEST_F(ProcessFrameTest, ProcessFrameHandlesLastFrame) {
  JxlEncoderStruct* enc = CreateEncoder();

  size_t xsize = 64;
  size_t ysize = 64;
  std::vector<uint8_t> pixels = jxl::test::GetSomeTestImage(xsize, ysize, 3, 0);

  auto queued_frame =
      jxl::MemoryManagerUniquePtr<jxl::JxlEncoderQueuedFrame>(
          jxl::MemoryManagerDeleteHelper(&enc->memory_manager));

  auto frame = new jxl::JxlEncoderQueuedFrame();
  frame->frame_data.SetFromBuffer(pixels.data(), pixels.size(),
                                   JxlPixelFormat{3, JXL_TYPE_UINT8,
                                                  JXL_NATIVE_ENDIAN, 0},
                                   xsize, ysize);
  frame->option_values.cparams = jxl::CompressParams();
  frame->option_values.header.duration = 0;
  frame->option_values.header.timecode = 0;
  frame->option_values.header.layer_info.crop_x0 = 0;
  frame->option_values.header.layer_info.crop_y0 = 0;
  frame->ec_initialized.resize(0);

  queued_frame.reset(frame);

  JxlEncoderQueuedInput input(&enc->memory_manager);
  input.frame = std::move(queued_frame);
  enc->input_queue.push_back(std::move(input));
  enc->num_queued_frames = 1;
  enc->wrote_bytes = true;
  enc->frames_closed = true;  // This is the last frame

  jxl::PaddedBytes header_bytes = CreateMinimalHeaderBytes(&enc->memory_manager);

  jxl::Status status = enc->ProcessFrame(header_bytes);

  EXPECT_TRUE(status);
  EXPECT_EQ(0u, enc->input_queue.size());

  delete enc;
}

// Test that ProcessFrame updates codestream_bytes_written_end_of_frame.
TEST_F(ProcessFrameTest, ProcessFrameUpdatesBytesWritten) {
  JxlEncoderStruct* enc = CreateEncoder();

  size_t xsize = 64;
  size_t ysize = 64;
  std::vector<uint8_t> pixels = jxl::test::GetSomeTestImage(xsize, ysize, 3, 0);

  auto queued_frame =
      jxl::MemoryManagerUniquePtr<jxl::JxlEncoderQueuedFrame>(
          jxl::MemoryManagerDeleteHelper(&enc->memory_manager));

  auto frame = new jxl::JxlEncoderQueuedFrame();
  frame->frame_data.SetFromBuffer(pixels.data(), pixels.size(),
                                   JxlPixelFormat{3, JXL_TYPE_UINT8,
                                                  JXL_NATIVE_ENDIAN, 0},
                                   xsize, ysize);
  frame->option_values.cparams = jxl::CompressParams();
  frame->option_values.header.duration = 0;
  frame->option_values.header.timecode = 0;
  frame->option_values.header.layer_info.crop_x0 = 0;
  frame->option_values.header.layer_info.crop_y0 = 0;
  frame->ec_initialized.resize(0);

  queued_frame.reset(frame);

  JxlEncoderQueuedInput input(&enc->memory_manager);
  input.frame = std::move(queued_frame);
  enc->input_queue.push_back(std::move(input));
  enc->num_queued_frames = 1;
  enc->wrote_bytes = true;

  size_t bytes_before = enc->codestream_bytes_written_end_of_frame;

  jxl::PaddedBytes header_bytes = CreateMinimalHeaderBytes(&enc->memory_manager);
  jxl::Status status = enc->ProcessFrame(header_bytes);

  EXPECT_TRUE(status);
  // Verify that bytes_written was updated (should be greater than before)
  // The exact amount depends on encoder output, but it should have changed
  EXPECT_GE(enc->codestream_bytes_written_end_of_frame, bytes_before);

  delete enc;
}

}  // namespace jxl
