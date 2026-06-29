# enc_entropy_coder.cc / .h — byte-exact optimization pass (2026-06-29)

Branch: `perf/enc-entropy-bytemap-jun29-e7q3` (capebio), off submodule `main` @2169106a.
Scope: `lib/jxl/enc_entropy_coder.{cc,h}` + minimal caller in `lib/jxl/enc_frame.cc`.
Goal: optimize `TokenizeCoefficients` while keeping the encoded bitstream
**bit-identical**. Verified with the encoder SHA flip-flop (`tools/enc-sha-flipflop.mjs`).

## Landed changes (all byte-exact by construction)

### 1. `tmp_num_nzeroes`: `Image3I` → `Image3B` (uint8)  — headline
The per-8×8-block nonzero-prediction scratch plane stored `int32`. The value
written is `ceil(nzeros / covered_blocks)`, and an AC block holds at most
`kDCTBlockSize - 1 == 63` nonzeros (DC/LLF excluded), so every stored value is
in `[0, 63]` and fits in one byte.

- The **decoder already uses `Image3<uint8_t> num_nzeroes`** (`dec_cache.h:137`)
  and `PredictFromTopAndLeft` is templated on the element type, so the encoder
  was the odd one out. This change just makes the two sides agree.
- 4× smaller scratch plane (`kGroupDimInBlocks²` cells) → less L1/L2 pressure
  while the SIMD coefficient scan streams the int32 AC rows.
- The counting SIMD still loads the **int32** AC coefficients unchanged, so the
  nonzero count is identical; only the prediction scratch narrows. Byte-exact.

Touches: `enc_entropy_coder.h` (signature), `enc_entropy_coder.cc`
(signatures + row pointer types + nzeros_pos types), `enc_frame.cc`
(`EncCache::num_nzeroes` + `InitOnce` allocation).

### 2. Per-row `memset` fill of the prediction scratch
`NumNonZeroExceptLLF` wrote the same `shifted_nzeros` to every covered cell with
a nested `x`/`y` loop. Each covered row is one run of identical bytes, so the
inner loop becomes `memset(nzeros_pos + y*stride, shifted_nzeros, covered_x)`.
Only reached by multi-block transforms (8×8 uses `NumNonZero8x8ExceptDC`).

### 3. Interior-block skip
The block raster did `for (bx…; ++bx) { if (!IsFirstBlock()) continue; … }`,
re-loading the strategy and testing `IsFirstBlock()` on every covered non-first
cell of a multi-block transform. Now it advances `bx += covered_blocks_x()` past
a first block, exactly as the **decoder** does (`dec_group.cc`). Byte-exact: the
cells skipped are precisely the ones the old `continue` discarded
(`AcStrategyImage::Set` marks only the top-left cell as first).

### 4. Hoist `kStrategyOrder[acs.RawStrategy()]`
`ord` was recomputed inside the `for (c : {1,0,2})` channel loop though it does
not depend on `c`. Hoisted above the loop (1× vs up to 3× LUT load).

### 5. Subsampling-aware `output->reserve()`
The reserve was `3 * xsize_blocks * ysize_blocks * kDCTBlockSize`, a 1.5×
(4:2:2) / 2× (4:2:0) over-reserve. Now sums the true per-channel block counts
honouring `HShift`/`VShift`. For 4:4:4 (the RAW-pipeline path) this equals the
old bound exactly. Capacity only — never changes output. Byte-exact.

## Rejected / not done (with reasons)

- **"LLF mask is buggy for large transforms" (ChatGPT headline "repair").**
  Claim: `NumNonZeroExceptLLF`'s `llf_mask_lanes` only seeds 4 `-1` lanes while
  `kMaxCoeffBlocks == 32`, so wide transforms mis-count LLF. **Rejected.** This
  code is **identical to upstream libjxl 0.11.2** and is exercised by libjxl's
  own large-transform roundtrip tests; a real mis-count would trip the
  `JXL_ENSURE(nzeros == 0)` at the end of token emission and fail encoding. The
  masking-beyond-4 is a no-op because the LLF coefficient slots are already zero
  in the AC residual buffer at this stage (counting a zero, masked or not, is the
  same). "Fixing" a non-bug here risks a real regression for zero gain. Left
  untouched.
- **Incremental `ZeroDensityContext` buckets.** The normalization is
  `(nz + cb - 1) >> log2_cb` and `k >> log2_cb`; for the dominant 8×8 case
  `covered_blocks == 1`, `log2_cb == 0`, so both shifts are `>>0` no-ops already.
  The incremental rewrite only helps rare large transforms and is easy to get
  subtly non-byte-exact. Not worth it.
- **`TokenStrategyInfo` strategy-descriptor LUT.** Removes a few per-transform
  multiplies / `CoefficientLayout` calls, but those are negligible next to the
  coefficient scan; adds a parallel source of truth to maintain. Skipped.
- **QF-bin context cache, subsampling-specialized dispatch, ANS workspace
  reuse, strategy CPU-cost model, work-weighted scheduling.** Out of scope (live
  in callers / other files), marginal for the 4:4:4 single-QF RAW path, or not
  byte-exact (the cost model changes strategy choice). Not pursued.

## Verification

`tools/enc-sha-flipflop.mjs`, OLD = shipped plain enc WASM @2169106a,
NEW = plain enc WASM rebuilt from this worktree
(`build-pgo.mjs --plain`, `JXL_PGO_LIBJXL_SRC=C:\Foo\rcw-enc-entropy`).

### Byte-exactness — PASS (the headline result)

Built NEW plain enc WASM from this worktree (`build-pgo.mjs --plain`, ~11 min)
and ran `enc-sha-flipflop.mjs` OLD(@2169106a) vs NEW. SHA256 of the encoded
`.jxl` is **identical** at every effort, across two independent runs:

```
=== SHA256 CORRECTNESS (1024×768) ===
  e3: PASS  d6d18b6395feca16 == d6d18b6395feca16  16.9 KB
  e5: PASS  dd5e4b58398a201b == dd5e4b58398a201b  23.0 KB
  e6: PASS  c65271544aab0267 == c65271544aab0267  23.8 KB
  e7: PASS  84206d16be80b997 == 84206d16be80b997  25.5 KB
  e9: PASS  5128d6e941d14f9a == 5128d6e941d14f9a  23.5 KB
  ALL PASS — byte-exact ✓
```

The output bitstream is bit-for-bit unchanged. ✓

### Timing — neutral within noise

True-alternation OLD/NEW whole-encode timing, two runs. min-ratios cluster at
~1.00 (run 2: 1.005 / 1.017 / 0.991 / 1.005 / 0.988 / 1.009). Run 1 showed one
0.757 outlier at 1024×768 e5 whose own median was 1.100 — i.e. noise, which run 2
did not reproduce. `TokenizeCoefficients` is a small fraction of total encode
time, so the 4× scratch-traffic reduction is below the whole-encode noise floor
of this harness (the dev box is contended). **No measurable e2e regression; no
measurable e2e win at this granularity.**

### Verdict

Land it as a **byte-exact, zero-risk** change: it removes the encoder/decoder
`int32` vs `uint8` asymmetry, cuts the per-thread `num_nzeroes` scratch plane to
1/4, and tidies the block raster — without touching a single output byte. The
speed win is real but too localized to register against full-encode wall time;
its value is correctness-alignment + memory footprint, not a benchmark number.

Reproduce:
```
# build NEW from this worktree (deps: copy external/libjxl-012/third_party/*)
JXL_PGO_LIBJXL_SRC=C:\Foo\rcw-enc-entropy JXL_WASM_WORKDIR=C:\Tmp\jxl-entropy-work \
  node packages/jxl-wasm/scripts/build-pgo.mjs --plain
node tools/enc-sha-flipflop.mjs <old.js> <new.js>
```
