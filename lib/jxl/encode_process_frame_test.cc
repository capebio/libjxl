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
#include "lib/jxl/memory_manager_internal.h"
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
  // Uses JxlEncoderCreate(nullptr) so memory_manager is properly initialized.
  JxlEncoderStruct* CreateEncoder() {
    JxlEncoderStruct* enc = JxlEncoderCreate(nullptr);
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

  // Helper to build a queued frame from a pixel buffer.
  static jxl::MemoryManagerUniquePtr<jxl::JxlEncoderQueuedFrame> MakeFrame(
      JxlEncoderStruct* enc, size_t xsize, size_t ysize,
      const std::vector<uint8_t>& pixels) {
    jxl::JxlEncoderChunkedFrameAdapter frame_data(xsize, ysize, 0);
    const bool ok = frame_data.SetFromBuffer(
        0, pixels.data(), pixels.size(),
        JxlPixelFormat{3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0});
    if (!ok) return {nullptr, jxl::MemoryManagerDeleteHelper(&enc->memory_manager)};
    return jxl::MemoryManagerMakeUnique<jxl::JxlEncoderQueuedFrame>(
        &enc->memory_manager,
        jxl::JxlEncoderQueuedFrame{
            jxl::JxlEncoderFrameSettingsValues{}, std::move(frame_data), {}});
  }

  void TearDown() override {}
};

// Test that ProcessFrame correctly processes a single queued frame.
TEST_F(ProcessFrameTest, ProcessFrameEncodesQueuedFrame) {
  JxlEncoderStruct* enc = CreateEncoder();

  size_t xsize = 64;
  size_t ysize = 64;
  std::vector<uint8_t> pixels = jxl::test::GetSomeTestImage(xsize, ysize, 3, 0);

  auto queued_frame = MakeFrame(enc, xsize, ysize, pixels);
  ASSERT_NE(nullptr, queued_frame);

  JxlEncoderQueuedInput input(&enc->memory_manager);
  input.frame = std::move(queued_frame);
  enc->input_queue.push_back(std::move(input));
  enc->num_queued_frames = 1;
  enc->wrote_bytes = true;  // Skip header preparation

  jxl::PaddedBytes header_bytes = CreateMinimalHeaderBytes(&enc->memory_manager);

  jxl::Status status = enc->ProcessFrame(header_bytes);

  EXPECT_TRUE(status);
  EXPECT_EQ(0u, enc->input_queue.size());
  EXPECT_EQ(0u, enc->num_queued_frames);

  JxlEncoderDestroy(enc);
}

// Test that ProcessFrame handles the last frame correctly.
TEST_F(ProcessFrameTest, ProcessFrameHandlesLastFrame) {
  JxlEncoderStruct* enc = CreateEncoder();

  size_t xsize = 64;
  size_t ysize = 64;
  std::vector<uint8_t> pixels = jxl::test::GetSomeTestImage(xsize, ysize, 3, 0);

  auto queued_frame = MakeFrame(enc, xsize, ysize, pixels);
  ASSERT_NE(nullptr, queued_frame);

  JxlEncoderQueuedInput input(&enc->memory_manager);
  input.frame = std::move(queued_frame);
  enc->input_queue.push_back(std::move(input));
  enc->num_queued_frames = 1;
  enc->wrote_bytes = true;
  // enc->frames_closed is already true from CreateEncoder()

  jxl::PaddedBytes header_bytes = CreateMinimalHeaderBytes(&enc->memory_manager);

  jxl::Status status = enc->ProcessFrame(header_bytes);

  EXPECT_TRUE(status);
  EXPECT_EQ(0u, enc->input_queue.size());

  JxlEncoderDestroy(enc);
}

// Test that ProcessFrame updates codestream_bytes_written_end_of_frame.
TEST_F(ProcessFrameTest, ProcessFrameUpdatesBytesWritten) {
  JxlEncoderStruct* enc = CreateEncoder();

  size_t xsize = 64;
  size_t ysize = 64;
  std::vector<uint8_t> pixels = jxl::test::GetSomeTestImage(xsize, ysize, 3, 0);

  auto queued_frame = MakeFrame(enc, xsize, ysize, pixels);
  ASSERT_NE(nullptr, queued_frame);

  JxlEncoderQueuedInput input(&enc->memory_manager);
  input.frame = std::move(queued_frame);
  enc->input_queue.push_back(std::move(input));
  enc->num_queued_frames = 1;
  enc->wrote_bytes = true;

  size_t bytes_before = enc->codestream_bytes_written_end_of_frame;

  jxl::PaddedBytes header_bytes = CreateMinimalHeaderBytes(&enc->memory_manager);
  jxl::Status status = enc->ProcessFrame(header_bytes);

  EXPECT_TRUE(status);
  EXPECT_GE(enc->codestream_bytes_written_end_of_frame, bytes_before);

  JxlEncoderDestroy(enc);
}

}  // namespace jxl
