# enc_aq masking 4-row batch (deferred experiment #3) — result

Branch off submodule main @10783f7e (which already contains the landed
byte-exact AQ pass A-J). This adds ONE further byte-exact change.

Change: AdaptiveQuantizationImpl::ComputeTile's masking-difference precompute
summed four source rows into diff_buffer via a per-row read-modify-write; it now
sums the four rows in registers and stores once. Every pixel stays on the exact
same scalar/SIMD path (original SIMD guard kept), so the only difference is the
row-accumulation order, byte-exact by FP-add commutativity:
original d3+(d2+(d1+d0)) == ((d0+d1)+d2)+d3.

Verification — tools/enc_aq_masking_ab.cc, isolated Highway A/B (AVX2 static):
  size       verdict      saved
  64x64      BYTE-EXACT   +11.3%
  256x256    BYTE-EXACT   +10.9%
  512x384    BYTE-EXACT   +10.3%
  1024x768   BYTE-EXACT   +11.4%
  2048x1536  BYTE-EXACT   +10.8%
  aggregate  BYTE-EXACT   +10.9%
pre_erosion bit-identical OLD vs NEW at every size; consistent ~+11% (the win is
read-modify-write elimination, not cache, hence size-independent).

Build: clang++ -O2 -mavx2 -mfma -std=c++17 -I <012-highway-include> \
  tools/enc_aq_masking_ab.cc -o enc_aq_masking_ab

CAVEAT: measured on AVX2 (8-lane). Byte-exactness is lane-width independent so it
holds on wasm/SSE, but SPEED on wasm SIMD128 (4-lane, fewer vector registers,
four simd_diff live at once) is UNMEASURED and could differ (cf. earlier
DCT-fuse SSE-looked-good / wasm-regressed lesson). Build the wasm tier before
landing for the browser.

Rejected sibling #4 (widen SIMD tail guard x+1+Lanes<x_end -> x+Lanes<x_end):
NOT byte-exact. Scalar base sum is ((r2[x]+r1[x])+r[x-1])+r[x+1]; SIMD base sum
is (r[x+1]+r[x-1])+(r2[x]+r1[x]) — different associativity. Widening the guard
moves pixels between the two orderings and changes bits (verified by reasoning;
would fail the end-to-end byte-exact A/B).
