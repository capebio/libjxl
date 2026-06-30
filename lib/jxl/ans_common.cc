// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/ans_common.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

namespace jxl {

// First, all trailing non-occurring symbols are removed from the distribution;
// if this leaves the distribution empty, a placeholder symbol with max weight
// is  added. This ensures that the resulting distribution sums to total table
// size. Then, `entry_size` is chosen to be the largest power of two so that
// `table_size` = ANS_TAB_SIZE/`entry_size` is at least as big as the
// distribution size.
// Note that each entry will only ever contain two different symbols, and
// consecutive ranges of offsets, which allows us to use a compact
// representation.
// Each entry is initialized with only the (symbol=i, offset) pairs; then
// positions for which the entry overflows (i.e. distribution[i] > entry_size)
// or is not full are computed, and put into a stack in increasing order.
// Missing symbols in the distribution are padded with 0 (because `table_size`
// >= number of symbols). The `cutoff` value for each entry is initialized to
// the number of occupied slots in that entry (i.e. `distributions[i]`). While
// the overflowing-symbol stack is not empty (which implies that the
// underflowing-symbol stack also is not), the top overfull and underfull
// positions are popped from the stack; the empty slots in the underfull entry
// are then filled with as many slots as needed from the overfull entry; such
// slots are placed after the slots in the overfull entry, and `offsets[1]` is
// computed accordingly. The formerly underfull entry is thus now neither
// underfull nor overfull, and represents exactly two symbols. The overfull
// entry might be either overfull or underfull, and is pushed into the
// corresponding stack.
//
// Allocation-free construction notes: the three former heap `std::vector`s
// (`underfull_posn`, `overfull_posn`, `cutoffs`) are gone. Position stacks are
// fixed `std::array<uint8_t, ANS_MAX_ALPHABET_SIZE>` (table_size never exceeds
// ANS_MAX_ALPHABET_SIZE, so uint8 indices suffice). The per-entry `cutoff`
// scratch is borrowed in `a[i].freq1_xor_freq0` until the finalization loop
// overwrites it with its real value; the construction loops never read
// `freq1_xor_freq0` for any purpose other than this scratch, and `freq0` is
// written to its final value up front, so the produced table is byte-identical
// to the previous implementation for all valid distributions. The distribution
// is taken by const reference and the trailing-zero trim is done with a length
// counter instead of `pop_back`, so no copy of the caller's vector is made.
Status InitAliasTable(const std::vector<int32_t>& distribution,
                      uint32_t log_range, size_t log_alpha_size,
                      AliasTable::Entry* JXL_RESTRICT a) {
  const uint32_t range = 1u << log_range;
  const size_t table_size = 1u << log_alpha_size;
  JXL_ENSURE(table_size <= range);
  const uint32_t entry_size = range >> log_alpha_size;  // this is exact

  // Drop trailing non-occurring symbols (formerly distribution.pop_back()).
  size_t distribution_size = distribution.size();
  while (distribution_size != 0 && distribution[distribution_size - 1] == 0) {
    --distribution_size;
  }

  // Ensure that a valid table is always returned, even for an empty
  // alphabet. Otherwise, a specially-crafted stream might crash the
  // decoder. The old code emplaced a single symbol 0 with weight `range`;
  // that yields the single-symbol table below with sym == 0.
  if (distribution_size == 0) {
    for (size_t i = 0; i < table_size; i++) {
      a[i].right_value = 0;
      a[i].cutoff = 0;
      a[i].offsets1 = static_cast<uint16_t>(entry_size * i);
      a[i].freq0 = 0;
      a[i].freq1_xor_freq0 = ANS_TAB_SIZE;
    }
    return true;
  }

  JXL_ENSURE(distribution_size <= table_size);
  int single_symbol = -1;
  int sum = 0;
  // Special case for single-symbol distributions, that ensures that the state
  // does not change when decoding from such a distribution. Note that, since we
  // hardcode offset0 == 0, it is not straightforward (if at all possible) to
  // fix the general case to produce this result.
  for (size_t sym = 0; sym < distribution_size; sym++) {
    int32_t v = distribution[sym];
    sum += v;
    if (v == ANS_TAB_SIZE) {
      JXL_ENSURE(single_symbol == -1);
      single_symbol = sym;
    }
  }
  JXL_ENSURE(static_cast<uint32_t>(sum) == range);
  if (single_symbol != -1) {
    uint8_t sym = single_symbol;
    JXL_ENSURE(single_symbol == sym);
    for (size_t i = 0; i < table_size; i++) {
      a[i].right_value = sym;
      a[i].cutoff = 0;
      a[i].offsets1 = entry_size * i;
      a[i].freq0 = 0;
      a[i].freq1_xor_freq0 = ANS_TAB_SIZE;
    }
    return true;
  }

  // Position stacks (formerly std::vector<uint32_t>). table_size <=
  // ANS_MAX_ALPHABET_SIZE, so 8-bit indices are sufficient.
  std::array<uint8_t, ANS_MAX_ALPHABET_SIZE> underfull_posn;
  std::array<uint8_t, ANS_MAX_ALPHABET_SIZE> overfull_posn;
  size_t num_underfull = 0;
  size_t num_overfull = 0;

  // Initialize entries. freq0 is written to its final value here; the per-entry
  // cutoff (formerly cutoffs[i]) is stored transiently in freq1_xor_freq0.
  for (size_t i = 0; i < distribution_size; i++) {
    const uint32_t cutoff = static_cast<uint32_t>(distribution[i]);
    a[i].freq0 = static_cast<uint16_t>(cutoff);
    a[i].freq1_xor_freq0 = static_cast<uint16_t>(cutoff);
    if (cutoff > entry_size) {
      overfull_posn[num_overfull++] = static_cast<uint8_t>(i);
    } else if (cutoff < entry_size) {
      underfull_posn[num_underfull++] = static_cast<uint8_t>(i);
    }
  }
  for (size_t i = distribution_size; i < table_size; i++) {
    a[i].freq0 = 0;
    a[i].freq1_xor_freq0 = 0;  // cutoff scratch == 0
    underfull_posn[num_underfull++] = static_cast<uint8_t>(i);
  }
  // Reassign overflow/underflow values.
  while (num_overfull != 0) {
    const uint8_t overfull_i = overfull_posn[--num_overfull];
    JXL_ENSURE(num_underfull != 0);
    const uint8_t underfull_i = underfull_posn[--num_underfull];
    const uint32_t underfull_by = entry_size - a[underfull_i].freq1_xor_freq0;
    a[overfull_i].freq1_xor_freq0 =
        static_cast<uint16_t>(a[overfull_i].freq1_xor_freq0 - underfull_by);
    // overfull positions have their original symbols
    a[underfull_i].right_value = overfull_i;
    a[underfull_i].offsets1 = a[overfull_i].freq1_xor_freq0;
    // Slots in the right part of entry underfull_i were taken from the end
    // of the symbols in entry overfull_i.
    if (a[overfull_i].freq1_xor_freq0 < entry_size) {
      underfull_posn[num_underfull++] = overfull_i;
    } else if (a[overfull_i].freq1_xor_freq0 > entry_size) {
      overfull_posn[num_overfull++] = overfull_i;
    }
  }
  for (size_t i = 0; i < table_size; i++) {
    const uint16_t cutoff = a[i].freq1_xor_freq0;
    if (cutoff == entry_size) {
      a[i].right_value = static_cast<uint8_t>(i);
      a[i].offsets1 = 0;
      a[i].cutoff = 0;
    } else {
      // Note that, if cutoff is not equal to entry_size,
      // a[i].offsets1 was initialized with (overfull cutoff) -
      // (entry_size - a[i].cutoff). Thus, subtracting
      // a[i].cutoff cannot make it negative.
      a[i].offsets1 = static_cast<uint16_t>(a[i].offsets1 - cutoff);
      a[i].cutoff = static_cast<uint8_t>(cutoff);
    }
    // a[k].freq0 holds the final freq for every entry k, so a[right_value].freq0
    // is freq1.
    a[i].freq1_xor_freq0 =
        static_cast<uint16_t>(a[a[i].right_value].freq0 ^ a[i].freq0);
  }
  return true;
}

}  // namespace jxl
