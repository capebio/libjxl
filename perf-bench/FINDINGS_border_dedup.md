# enc_convolve_separable5 — border-row horizontal dedup

Branch: `perf/enc-conv5-border-dedup-jun29-r3x` (capebio, submodule), off
submodule `main` @ `10783f7e` (which already carries the y-reuse rolling ring).

## What changed
`ConvolveRow` (the border-row kernel) used to evaluate **five** horizontal
convolutions per output vector — one each for `row_t2,row_t1,row_m,row_b1,row_b2`.
On border rows the mirrored-boundary remap makes several of those pointers
**alias** the same physical input row:

| topology | distinct horizontal convolutions (was 5) |
|----------|------------------------------------------|
| top row `img_y==0` (`t1==m`, `t2==b1`) | 3 |
| 2nd row `img_y==1` (`t2==t1`) | 4 |
| tiny height `ysize==1` (LUT: all 5 alias) | 1 |
| tiny height `ysize==2/3/4` (LUT) | 2 / 3 / 4 |

New `HorzFiveDedup<kSizeModN,kRegion>` computes each **distinct** horizontal
convolution once and reuses the result for aliased rows (pointer compares are
loop-invariant and perfectly predictable). The vertical FMA sequence is the
**identical** `conv0/conv1/conv2` accumulation as before, and a reused `V` is
bit-identical to a recomputed one, so output is **byte-exact by construction**.
Interior rows (all five pointers distinct) fall through to five fresh
convolutions exactly as before — no change there. The y-reuse ring handles all
interior rows; this only touches the top/bottom border + tiny-height rows.

## Verification (perf-bench/conv5_border_ab.cc)
Self-contained single-Highway-target A/B harness. Both arms share identical
row-pointer setup (incl. the tiny-height LUT); the only difference is dedup vs.
five independent convolutions. FNV hash of the border output rows, OLD == NEW,
over a 143-config sweep: widths `{66..69,128..131,512..515,1024}` × heights
`{1,2,3,4,5,6,7,8,16,33,64}` — hits every `kSizeModN` and every border topology.

**Byte-exact: PASS — 0 mismatches / 143 configs on ALL of:**
- WASM SIMD128 (4-lane, emcc `-O3 -msimd128`) — the shipping target
- native AVX2 (8-lane, `-march=native`, under ASan `-O1` and `-O3`)
- native 4-lane (SSSE3 static, plain `-O3`)

## Results — border pass timing, interleaved (flipflop discipline)
Median over thousands of reps of the border-only pass on border-dominated images.
Savings scale with aliasing degree (highest for tiny heights):

**WASM SIMD128 (shipping target):**
```
16384x1  (all 5 alias)   OLD 0.0919  NEW 0.0548 ms   +40.4%
16384x2                  OLD 0.1890  NEW 0.1372 ms   +27.4%
16384x3                  OLD 0.3003  NEW 0.2632 ms   +12.4%
16384x4                  OLD 0.4237  NEW 0.4083 ms    +3.6%
16384x8  (4 border rows) OLD 0.4354  NEW 0.4178 ms    +4.0%
 8192x64 (4 border rows) OLD 0.2048  NEW 0.1912 ms    +6.6%
```
Native 4-lane SSSE3 (plain -O3): +10..46% over the same configs (noisier, sub-ms).

## Honest scope
Border rows are `O(4)` per image (top/bottom `kRadius` each); a large full-res
plane is interior-dominated, so this is **neutral-to-tiny on full images** and a
**real win on tiny / coarse images** — i.e. the coarse levels of butteraugli's
Gaussian pyramid (`butteraugli.cc` Blur) and small `enc_detect_dots.cc` inputs.
It is never a regression: aliased rows do strictly less work, distinct rows are
unchanged, and the added pointer compares only run on border rows. Kept per the
"theoretically-better + non-regression" rule.

## Build
```
HWY=<libjxl>/third_party/highway   # or any populated hwy include tree
# wasm (shipping target):
emcc -O3 -msimd128 -std=c++17 -I "$HWY" conv5_border_ab.cc -o conv5_border_ab.js \
     -sENVIRONMENT=node -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=268435456 && node conv5_border_ab.js
# native: build with ASan for the 8-lane AVX2 path on clang-22 (`-march=native`
# plain -O2/-O3 hits an unrelated codegen fault on this box; ASan/-O1/4-lane are
# clean and confirm byte-exactness + the win):
clang++ -O3 -g -fsanitize=address -march=native -std=c++17 -I "$HWY" conv5_border_ab.cc -o ab && ./ab
```

## Notes
- Production `lib/jxl/enc_convolve_separable5.cc` passes `clang++ -fsyntax-only`
  for all enabled Highway targets (foreach_target).
- The dedup also covers `ConvolveRow<kSizeModN,false>` (interior, all distinct)
  — currently unreferenced (the ring owns interior rows) and behaviorally
  identical there.
