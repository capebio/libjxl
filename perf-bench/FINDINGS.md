# enc_convolve_separable5 — y-reuse rolling ring

Branch: `perf/enc-convolve-sep5-yring-2026jun29-q7` (capebio)

## What changed
Interior rows of the separable 5×5 convolution are now convolved in **vertical
bands** (`kRowsPerBand = 8`). Within a band each SIMD column is walked top→bottom
with a **rolling ring** of the five horizontal convolutions: every output row
reuses four of the five horizontal results from the previous row and computes
only the one incoming bottom row. Horizontal-convolution work drops from `5·B` to
`B+4` per column for a band of `B` rows (B=8 → 12 vs 40, a 70% cut).

The per-pixel vertical accumulation keeps the **identical FMA sequence** of the
original `ConvolveRow`, so output is **byte-exact by construction**. Border rows
and the scalar remainder are unchanged. The redundant `RunRows<3>` template
specialization is also collapsed into `RunRows<2>` (identical behavior — the
last-vector path is guarded by `kSizeModN < kRadius`, the scalar tail is driven
by runtime `xsize`).

## Verification (perf-bench/conv5_ab.cc)
Self-contained A/B harness, single Highway static target. OLD = original
row-by-row kernel, NEW = band/ring. Verifies FNV hash of the full output plane
(OLD == NEW) across 12 image configs × 5 band sizes, then times OLD vs NEW
interleaved (start-rotated) plus a band-size scan.

Build + run:
```
HWY=<libjxl>/third_party/highway
# native
clang++ -O3 -march=native -std=c++17 -I "$HWY" conv5_ab.cc -o conv5_ab.exe && ./conv5_ab.exe
# wasm SIMD128 (emsdk env first)
emcc -O3 -msimd128 -std=c++17 -I "$HWY" conv5_ab.cc -o conv5_ab.js \
     -sENVIRONMENT=node -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=268435456 && node conv5_ab.js
```

## Results

**Byte-exact: PASS** — 0 mismatches over 12 configs × 5 bands on BOTH
AVX2 (8-lane) and WASM SIMD128 (4-lane). Widths chosen to hit every
`xsize % lanes` ∈ {0,1,2,3}.

Interleaved A/B, NEW band=8, single-thread:

| size   | AVX2 OLD | AVX2 NEW | saved | WASM OLD | WASM NEW | saved |
|--------|---------:|---------:|------:|---------:|---------:|------:|
| 512²   | 0.631 ms | 0.434 ms | +31%  | 2.016 ms | 1.225 ms | +39%  |
| 1024²  | 2.925 ms | 1.789 ms | +39%  | 7.860 ms | 4.320 ms | +45%  |
| 2048²  | 12.32 ms | 8.253 ms | +33%  | 36.80 ms | 18.85 ms | +49%  |

**WASM — the actual deployment target after the 0.12 engine swap — gains the
most (+39…49%).** This optimization ports to wasm (unlike lane-shuffle tricks
such as DCT transpose-fusion) because it removes redundant horizontal-convolve
*work* (loads + FMAs), not anything lane-width-sensitive.

### Band-size scan
`band=8` is the robust sweet spot on both targets. `band=2/4` leave gains on the
table; `band=16/32` regress on ≥1024² images (the ring touches `band+4` input
rows + `band` output rows at stride spacing per column → too many concurrent
streams thrash cache at large band). Hence the default of 8.

## Notes
- Byte-exactness holds per Highway target by construction (same op DAG); the
  harness confirms it empirically on AVX2 and WASM.
- Multi-threaded path: `RunInteriorRows` distributes bands over `RunOnPool`;
  bands write disjoint output rows → no races, output identical to single-thread.
- Real TU `lib/jxl/enc_convolve_separable5.cc` passes `clang++ -fsyntax-only`
  for all enabled Highway targets (foreach_target).
- Callers: `butteraugli.cc` (Blur) and `enc_detect_dots.cc`.

---

# §x-tile — REJECTED follow-up (tall band + narrow x-tile scratch ring)

Branch: `perf/enc-conv5-xtile-jun29-v7k` (off submodule main `10783f7e`).

## Hypothesis
The register ring recomputes a 4-row halo per `band` (`(B+4)/B = 1.5×`
horizontal work at B=8) because taller register bands thrash cache
(`(H+4)×xsize` resident). A **scratch ring** — full-interior-height band,
processed in narrow x-tiles, holding 5 rows of horizontal convolutions in an
L1 scratch buffer and sliding down — would cut halo recompute to ~1.0× while
keeping only `5×tileW` resident (L1), decoupling halo amortization from cache
pressure. Predicted ~15–20% on top of the ring.

## Implementation
3rd harness variant (`DispatchXTile`): region-0 (left mirror) and region-2
(right mirror) edge columns keep the proven register `RingColumn`; the bulk
interior columns (region 1) use the scratch x-tile; scalar tail unchanged.
Vertical FMA order and `HorzConvolve` are identical to the ring → **byte-exact
by construction** (float scratch store/load is bit-preserving).

## Result — byte-exact PASS, but a clear SPEED REGRESSION

Byte-exact: **PASS**, 0 mismatches over 12 cfgs × 6 tile widths, AVX2 + WASM
(also re-verifies the shipped ring vs OLD on both targets).

RING(band=8) vs XTILE, interleaved median:

| size  | AVX2 | WASM |
|-------|-----:|-----:|
| 512²  | −7%  | −11% |
| 1024² | −97% | −67% |
| 2048² | −84% | −85% |

Tile-width sweep: XTILE only beats the ring at tiny 512² with very wide tiles
(`tv=64`: +15% AVX2, +26% WASM, i.e. whole image in cache). At the real
≥1024² sizes it is **2–3× slower at every tile width**.

## Why it loses
The scratch ring replaces the register ring's zero-traffic vertical reuse with
**1 store + 5 loads of horizontals per output vector**. The work it saves (the
4-row halo recompute) lives in registers/L1 and is cheap; the scratch
round-trips it adds are not. The register ring does not spill on AVX2 or WASM
(it already delivers +39–49% over OLD), so it is the ceiling — exactly the case
where scratch tiling, the "register-pressure fallback", has nothing to recover.

## Verdict
**No production change.** The shipped register rolling ring (`kRowsPerBand=8`)
is optimal for these callers. This branch carries only the extended harness and
this note as the negative-result record; `enc_convolve_separable5.cc` is
untouched (== submodule main). Logged in superproject
`docs/1 rejected optimizations.md` §CONV5-2.
