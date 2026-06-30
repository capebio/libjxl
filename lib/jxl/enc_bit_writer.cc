// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_bit_writer.h"

#include <cstdint>
#include <cstring>  // memcpy
#include <memory>
#include <vector>

#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_aux_out.h"

namespace jxl {

namespace {
// Low `kMaxBitsPerCall` (56) bits set; the high 8 bits of a 64-bit source word
// are discarded when transferring seven bytes per Write.
constexpr uint64_t kLow56BitsMask =
    (uint64_t{1} << BitWriter::kMaxBitsPerCall) - 1;
}  // namespace

// WithMaxBits is defined in the header as a template (binds the callable
// directly instead of through std::function).

BitWriter::Allotment::Allotment(size_t max_bits) : max_bits_(max_bits) {}

Status BitWriter::Allotment::Init(BitWriter* JXL_RESTRICT writer) {
  prev_bits_written_ = writer->BitsWritten();
  const size_t prev_bytes = writer->storage_.size();
  const size_t next_bytes = DivCeil(max_bits_, kBitsPerByte);
  if (!writer->storage_.resize(prev_bytes + next_bytes)) {
    called_ = true;
    return false;
  }
  parent_ = writer->current_allotment_;
  writer->current_allotment_ = this;
  return true;
}

BitWriter::Allotment::~Allotment() {
  if (!called_) {
    // Not calling is a bug - unused storage will not be reclaimed.
    JXL_DEBUG_ABORT("Did not call Allotment::ReclaimUnused");
  }
}

Status BitWriter::Allotment::FinishedHistogram(BitWriter* JXL_RESTRICT writer) {
  if (writer == nullptr) return true;
  JXL_ENSURE(!called_);              // Call before ReclaimUnused
  JXL_ENSURE(histogram_bits_ == 0);  // Do not call twice
  JXL_ENSURE(writer->BitsWritten() >= prev_bits_written_);
  if (writer->BitsWritten() >= prev_bits_written_) {
    histogram_bits_ = writer->BitsWritten() - prev_bits_written_;
  }
  return true;
}

Status BitWriter::Allotment::ReclaimAndCharge(BitWriter* JXL_RESTRICT writer,
                                              LayerType layer,
                                              AuxOut* JXL_RESTRICT aux_out) {
  size_t used_bits = 0;
  size_t unused_bits = 0;
  JXL_RETURN_IF_ERROR(PrivateReclaim(writer, &used_bits, &unused_bits));

  // This may be a nested call with aux_out == null. Whenever we know that
  // aux_out is null, we can call ReclaimUnused directly.
  if (aux_out != nullptr) {
    aux_out->layer(layer).total_bits += used_bits;
    aux_out->layer(layer).histogram_bits += HistogramBits();
  }
  return true;
}

Status BitWriter::Allotment::PrivateReclaim(BitWriter* JXL_RESTRICT writer,
                                            size_t* JXL_RESTRICT used_bits,
                                            size_t* JXL_RESTRICT unused_bits) {
  JXL_DASSERT(!called_);  // Do not call twice
  called_ = true;
  if (writer == nullptr) return true;

  JXL_DASSERT(writer->BitsWritten() >= prev_bits_written_);
  *used_bits = writer->BitsWritten() - prev_bits_written_;
  JXL_DASSERT(*used_bits <= max_bits_);
  *unused_bits = max_bits_ - *used_bits;

  // Reclaim unused whole bytes from writer's allotment.
  const size_t unused_bytes = *unused_bits / kBitsPerByte;  // truncate
  if (unused_bytes != 0) {
    JXL_ENSURE(writer->storage_.size() >= unused_bytes);
    JXL_RETURN_IF_ERROR(
        writer->storage_.resize(writer->storage_.size() - unused_bytes));
  }
  writer->current_allotment_ = parent_;
  // Ensure we don't also charge the parent for these bits.
  auto* parent = parent_;
  while (parent != nullptr) {
    parent->prev_bits_written_ += *used_bits;
    parent = parent->parent_;
  }
  return true;
}

Status BitWriter::AppendByteAligned(const Span<const uint8_t>& span) {
  if (span.empty()) return true;
  JXL_ENSURE(BitsWritten() % kBitsPerByte == 0);

  // Grow only to the actual write endpoint (payload + one zero-padding byte).
  // Using BitsWritten()/8 as the base, rather than storage_.size(), avoids
  // leaking one stray byte of accounted size per call on repeated appends and
  // preserves any storage reserved by an enclosing allotment.
  size_t pos = BitsWritten() / kBitsPerByte;
  const size_t required = pos + span.size() + 1;  // +1: extra zero padding
  if (storage_.size() < required) {
    JXL_RETURN_IF_ERROR(storage_.resize(required));
  }

  // Concatenate by copying bytes because both source and destination are bytes.
  memcpy(storage_.data() + pos, span.data(), span.size());
  pos += span.size();
  JXL_ENSURE(pos < storage_.size());
  storage_[pos++] = 0;  // for next Write
  bits_written_ += span.size() * kBitsPerByte;
  return true;
}

Status BitWriter::AppendUnaligned(const BitWriter& other) {
  const size_t other_bits = other.BitsWritten();
  if (other_bits == 0) return true;

  return WithMaxBits(other_bits, LayerType::Header, nullptr, [&] {
    size_t full_bytes = other_bits / kBitsPerByte;
    const size_t remaining_bits = other_bits % kBitsPerByte;
    const uint8_t* src = other.storage_.data();

    if (bits_written_ % kBitsPerByte == 0) {
      // Byte-aligned destination: bulk-copy all full bytes in one memcpy, then
      // restore the single zero byte that Write requires after the copied run.
      uint8_t* dst = storage_.data() + bits_written_ / kBitsPerByte;
      memcpy(dst, src, full_bytes);
      dst[full_bytes] = 0;  // for next Write / zero tail
      bits_written_ += full_bytes * kBitsPerByte;
      if (remaining_bits != 0) {
        Write(remaining_bits,
              src[full_bytes] & ((uint64_t{1} << remaining_bits) - 1));
      }
      return true;
    }

    // Unaligned destination: transfer seven source bytes per Write(56) instead
    // of one byte per Write(8), cutting the per-byte read-modify-write store
    // count by ~7x.
#if JXL_BYTE_ORDER_LITTLE
    while (full_bytes >= 7) {
      uint64_t chunk;
      // Loads 8 bytes; the 8th is always within PaddedBytes' padding and is
      // masked off below.
      memcpy(&chunk, src, sizeof(chunk));
      Write(kMaxBitsPerCall, chunk & kLow56BitsMask);
      src += 7;
      full_bytes -= 7;
    }
#else
    while (full_bytes >= 7) {
      uint64_t chunk = 0;
      for (size_t i = 0; i < 7; ++i) {
        chunk |= static_cast<uint64_t>(src[i]) << (i * kBitsPerByte);
      }
      Write(kMaxBitsPerCall, chunk);
      src += 7;
      full_bytes -= 7;
    }
#endif
    while (full_bytes != 0) {
      Write(kBitsPerByte, *src++);
      --full_bytes;
    }
    if (remaining_bits != 0) {
      Write(remaining_bits, *src & ((uint64_t{1} << remaining_bits) - 1));
    }
    return true;
  });
}

// TODO(lode): avoid code duplication
Status BitWriter::AppendByteAligned(
    const std::vector<std::unique_ptr<BitWriter>>& others) {
  // Total size to add so we can preallocate
  size_t other_bytes = 0;
  for (const auto& writer : others) {
    JXL_ENSURE(writer->BitsWritten() % kBitsPerByte == 0);
    other_bytes += writer->BitsWritten() / kBitsPerByte;  // aligned: exact
  }
  if (other_bytes == 0) {
    // No bytes to append: this happens for example when creating per-group
    // storage for groups, but not writing anything in them for e.g. lossless
    // images with no alpha. Do nothing.
    return true;
  }
  JXL_ENSURE(BitsWritten() % kBitsPerByte == 0);

  // Grow only to the actual write endpoint (see AppendByteAligned(span)).
  size_t pos = BitsWritten() / kBitsPerByte;
  const size_t required = pos + other_bytes + 1;  // +1: extra zero padding
  if (storage_.size() < required) {
    JXL_RETURN_IF_ERROR(storage_.resize(required));
  }

  // Concatenate by copying bytes because both source and destination are bytes.
  for (const auto& writer : others) {
    const Span<const uint8_t> span = writer->GetSpan();
    memcpy(storage_.data() + pos, span.data(), span.size());
    pos += span.size();
  }
  JXL_ENSURE(pos < storage_.size());
  storage_[pos++] = 0;  // for next Write
  bits_written_ += other_bytes * kBitsPerByte;
  return true;
}

// Example: let's assume that 3 bits (Rs below) have been written already:
// BYTE+0       BYTE+1       BYTE+2
// 0000 0RRR    ???? ????    ???? ????
//
// Now, we could write up to 5 bits by just shifting them left by 3 bits and
// OR'ing to BYTE-0.
//
// For n > 5 bits, we write the lowest 5 bits as above, then write the next
// lowest bits into BYTE+1 starting from its lower bits and so on.
void BitWriter::Write(size_t n_bits, uint64_t bits) {
  JXL_DASSERT((bits >> n_bits) == 0);
  JXL_DASSERT(n_bits <= kMaxBitsPerCall);
  size_t bytes_written = bits_written_ / kBitsPerByte;
  uint8_t* p = &storage_[bytes_written];
  const size_t bits_in_first_byte = bits_written_ % kBitsPerByte;
  bits <<= bits_in_first_byte;
#if JXL_BYTE_ORDER_LITTLE
  uint64_t v = *p;
  // Last (partial) or next byte to write must be zero-initialized!
  // PaddedBytes initializes the first, and Write/Append maintain this.
  JXL_DASSERT(v >> bits_in_first_byte == 0);
  v |= bits;
  memcpy(p, &v, sizeof(v));  // Write bytes: possibly more than n_bits/8
#else
  *p++ |= static_cast<uint8_t>(bits & 0xFF);
  for (size_t bits_left_to_write = n_bits + bits_in_first_byte;
       bits_left_to_write >= 9; bits_left_to_write -= 8) {
    bits >>= 8;
    *p++ = static_cast<uint8_t>(bits & 0xFF);
  }
  *p = 0;
#endif
  bits_written_ += n_bits;
}
}  // namespace jxl
