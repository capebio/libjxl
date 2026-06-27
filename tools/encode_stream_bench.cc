// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Native streaming-encode driver: encodes a large synthetic RGB image through
// the PUBLIC JxlEncoder API with buffering=2, which routes through
// EncodeFrameStreaming (per-DC-group). Used to profile the streaming path
// (set JXL_STAGE_TIMERS=1 for per-stage fprintf) and as a flipflop timing
// vehicle (process wall-clock).
//
// Usage: encode_stream_bench W H effort reps

#include <jxl/encode.h>
#include <jxl/color_encoding.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

static std::vector<uint8_t> MakeImage(uint32_t W, uint32_t H) {
  // Deterministic mildly-compressible content (smooth gradient + cheap noise),
  // so the encoder does real VarDCT work rather than trivial flat blocks.
  std::vector<uint8_t> px(static_cast<size_t>(W) * H * 3);
  uint32_t s = 0x12345678u;
  for (uint32_t y = 0; y < H; ++y) {
    for (uint32_t x = 0; x < W; ++x) {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      size_t i = (static_cast<size_t>(y) * W + x) * 3;
      px[i + 0] = static_cast<uint8_t>((x * 255u / W) ^ (s & 7));
      px[i + 1] = static_cast<uint8_t>((y * 255u / H) ^ ((s >> 3) & 7));
      px[i + 2] = static_cast<uint8_t>(((x + y) * 255u / (W + H)) ^ ((s >> 6) & 7));
    }
  }
  return px;
}

static bool EncodeOnce(const uint8_t* px, size_t px_size, uint32_t W, uint32_t H,
                       int effort, size_t* out_size) {
  JxlEncoder* enc = JxlEncoderCreate(nullptr);
  if (!enc) return false;
  bool ok = true;

  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = W;
  info.ysize = H;
  info.bits_per_sample = 8;
  info.num_color_channels = 3;
  info.alpha_bits = 0;
  info.uses_original_profile = JXL_FALSE;
  if (JxlEncoderSetBasicInfo(enc, &info) != JXL_ENC_SUCCESS) ok = false;

  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, /*is_gray=*/JXL_FALSE);
  if (ok && JxlEncoderSetColorEncoding(enc, &color) != JXL_ENC_SUCCESS) ok = false;

  JxlEncoderFrameSettings* fs = JxlEncoderFrameSettingsCreate(enc, nullptr);
  JxlEncoderSetFrameDistance(fs, 1.0f);
  JxlEncoderFrameSettingsSetOption(fs, JXL_ENC_FRAME_SETTING_EFFORT, effort);
  // buffering=2 -> streaming path (EncodeFrameStreaming) for large lossy XYB.
  JxlEncoderFrameSettingsSetOption(fs, JXL_ENC_FRAME_SETTING_BUFFERING, 2);

  JxlPixelFormat fmt = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  if (ok && JxlEncoderAddImageFrame(fs, &fmt, px, px_size) != JXL_ENC_SUCCESS)
    ok = false;
  JxlEncoderCloseInput(enc);

  std::vector<uint8_t> out(1 << 20);
  uint8_t* next = out.data();
  size_t avail = out.size();
  JxlEncoderStatus st = JXL_ENC_NEED_MORE_OUTPUT;
  while (ok && st == JXL_ENC_NEED_MORE_OUTPUT) {
    st = JxlEncoderProcessOutput(enc, &next, &avail);
    if (st == JXL_ENC_NEED_MORE_OUTPUT) {
      size_t used = out.size() - avail;
      out.resize(out.size() * 2);
      next = out.data() + used;
      avail = out.size() - used;
    }
  }
  if (st != JXL_ENC_SUCCESS) ok = false;
  if (out_size) *out_size = out.size() - avail;
  JxlEncoderDestroy(enc);
  return ok;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s W H effort reps\n", argv[0]);
    return 64;
  }
  uint32_t W = std::strtoul(argv[1], nullptr, 10);
  uint32_t H = std::strtoul(argv[2], nullptr, 10);
  int effort = std::atoi(argv[3]);
  int reps = std::atoi(argv[4]);

  std::vector<uint8_t> px = MakeImage(W, H);
  size_t last = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) {
    if (!EncodeOnce(px.data(), px.size(), W, H, effort, &last)) {
      std::fprintf(stderr, "encode failed\n");
      return 1;
    }
  }
  auto ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr, "reps=%d out_bytes=%zu total_ms=%.1f per_ms=%.1f\n", reps,
               last, ms, ms / reps);
  return 0;
}
