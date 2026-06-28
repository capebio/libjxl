// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_HUFFMAN_H_
#define LIB_JXL_ENC_HUFFMAN_H_

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_bit_writer.h"

namespace jxl {

// Builds a Huffman tree for the given histogram, and encodes it into writer
// in a format that can be read by HuffmanDecodingData::ReadFromBitstream.
// An allotment for `writer` must already have been created by the caller.
// packed_scratch: caller-owned buffer of at least `length` bytes used as
// scratch space for the RLE token stream. If null, a temporary is allocated.
Status BuildAndStoreHuffmanTree(const uint32_t* histogram, size_t length,
                                uint8_t* depth, uint16_t* bits,
                                BitWriter* writer,
                                uint8_t* packed_scratch = nullptr);

}  // namespace jxl

#endif  // LIB_JXL_ENC_HUFFMAN_H_
