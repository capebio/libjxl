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
