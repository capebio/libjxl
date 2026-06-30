// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_bit_writer.h"

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

struct BitPatch {
  size_t len;
  uint64_t bits;
};

TEST(BitWriterTest, RandomSequence) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();

  auto mt = jxl::make_unique<std::mt19937>(42);
  std::uniform_int_distribution<> num_bits_dist(1, BitWriter::kMaxBitsPerCall);
  constexpr size_t kNumSequences = 1024 * 1024;
  std::vector<BitPatch> content;
  content.reserve(kNumSequences);
  size_t total_bits = 0;
  for (size_t i = 0; i < kNumSequences; ++i) {
    size_t len = num_bits_dist(*mt);
    uint64_t mask = (static_cast<uint64_t>(1) << len) - 1;
    uint64_t bits = (*mt)() & mask;
    content.emplace_back(BitPatch{len, bits});
    total_bits += len;
  }

  BitWriter writer{memory_manager};
  auto write_content = [&content, &writer]() -> Status {
    for (auto& patch : content) {
      writer.Write(patch.len, patch.bits);
    }
    writer.ZeroPadToByte();
    return true;
  };
  EXPECT_TRUE(writer.WithMaxBits(RoundUpBitsToByteMultiple(total_bits),
                                 LayerType::Header, nullptr, write_content));

  size_t num_mismatches = 0;
  BitReader reader(writer.GetSpan());
  for (auto& patch : content) {
    uint64_t bits = reader.ReadBits(patch.len);
    uint64_t expected_bits = patch.bits;
    if (bits != expected_bits) num_mismatches++;
  }
  EXPECT_TRUE(reader.JumpToByteBoundary());
  EXPECT_TRUE(reader.Close());
  EXPECT_EQ(num_mismatches, 0u);
}

// Exercises AppendUnaligned: the byte-aligned bulk-copy fast path (prefix == 0),
// the unaligned 56-bit block path (prefix != 0 and back-to-back appends), the
// partial-bit tail, and preservation of the zero-tail invariant via a trailing
// Write after the append.
TEST(BitWriterTest, AppendUnaligned) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  // 192 bits = 24 bytes exactly (no partial byte, several 7-byte blocks);
  // 197 bits = 24 bytes + 5 partial bits.
  constexpr size_t kSourceBits[] = {192, 197};

  for (size_t source_bits : kSourceBits) {
    // Build a deterministic source BitWriter holding exactly `source_bits` bits.
    auto mt = jxl::make_unique<std::mt19937>(static_cast<uint32_t>(source_bits));
    std::uniform_int_distribution<> len_dist(1, BitWriter::kMaxBitsPerCall);
    std::vector<BitPatch> patches;
    BitWriter source{memory_manager};
    auto write_source = [&]() -> Status {
      size_t written = 0;
      while (written < source_bits) {
        size_t len = len_dist(*mt);
        if (len > source_bits - written) len = source_bits - written;
        uint64_t mask = (static_cast<uint64_t>(1) << len) - 1;
        uint64_t bits = ((static_cast<uint64_t>((*mt)()) << 32) | (*mt)()) & mask;
        source.Write(len, bits);
        patches.emplace_back(BitPatch{len, bits});
        written += len;
      }
      return true;
    };
    ASSERT_TRUE(source.WithMaxBits(source_bits, LayerType::Header, nullptr,
                                   write_source));
    ASSERT_EQ(source.BitsWritten(), source_bits);

    // Every destination bit offset 0..7 (offset 0 hits the aligned fast path).
    for (size_t prefix_bits = 0; prefix_bits < kBitsPerByte; ++prefix_bits) {
      const uint64_t prefix =
          prefix_bits == 0 ? 0u
                           : ((static_cast<uint64_t>(1) << prefix_bits) - 1);
      BitWriter dst{memory_manager};
      auto build_dst = [&]() -> Status {
        if (prefix_bits != 0) dst.Write(prefix_bits, prefix);
        // First append: destination aligned iff prefix_bits == 0.
        JXL_RETURN_IF_ERROR(dst.AppendUnaligned(source));
        // Second append: destination now unaligned iff source_bits % 8 != 0,
        // exercising the unaligned block path back-to-back (the enc_ans usage).
        JXL_RETURN_IF_ERROR(dst.AppendUnaligned(source));
        // Trailing write proves the fast path preserved the zero tail.
        dst.Write(1, 1);
        dst.ZeroPadToByte();
        return true;
      };
      ASSERT_TRUE(dst.WithMaxBits(prefix_bits + 2 * source_bits + 1 + 7,
                                  LayerType::Header, nullptr, build_dst));

      BitReader reader(dst.GetSpan());
      if (prefix_bits != 0) EXPECT_EQ(reader.ReadBits(prefix_bits), prefix);
      for (int rep = 0; rep < 2; ++rep) {
        for (const BitPatch& patch : patches) {
          EXPECT_EQ(reader.ReadBits(patch.len), patch.bits);
        }
      }
      EXPECT_EQ(reader.ReadBits(1), 1u);
      EXPECT_TRUE(reader.JumpToByteBoundary());
      EXPECT_TRUE(reader.Close());
    }
  }
}

}  // namespace
}  // namespace jxl
