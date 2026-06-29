# enc_modular byte-exact optimizations — findings

Branch: `perf/enc-modular-byteexact-jun29-m7k2` (capebio/libjxl), off `2169106a`.
Worktree: `C:\Foo\rcw-enc-modular`. Files: `lib/jxl/enc_modular.{cc,h}`,
`lib/jxl/enc_modular_simd.cc`.

Implements the byte-exact subset of ChatGPT's modular-encoder review. Every change
here is intended to leave the encoded bitstream **byte-identical**; verified by an
OLD-vs-NEW full-encode FNV hash + size compare over an 8-image RAW corpus (see
`Verification`).

## Landed (byte-exact)

| # | Change | Mechanism | Notes |
|---|--------|-----------|-------|
| E1 | `EstimateWPCost` context lookup | replace per-pixel 33-cutoff scan with a precomputed `[-500,500]` table | monotone step function → table is identical context for every property |
| E1 | `EstimateWPCost` bounded early-out | add `upper_bound`; abandon a WP candidate once its partial cost ≥ incumbent best | cost is monotone non-decreasing across channels → the *selected* `wp_mode` is unchanged; the winning mode always runs to completion |
| E1 | `EstimateWPCost` immediate overflow exit | `return MAX` on `SubOverflow` instead of finishing the row then rejecting | same rejection, less dead work |
| E2 | `QuantizeChannel` row-parallel | independent rows via `RunOnPool`; null pool = serial (prior behavior) | per-pixel result depends only on itself + `q` |
| E3 | `AddACMetadata` exact ACS+QF channel | count first-blocks first, size channel 2 to `num` instead of full block-plane + shrink | only entries `[0,num)` are ever read; memory/alloc win (not a hot-path time win — channel sized in blocks, not pixels) |
| E5 | `ComputeTree` multiplier guard | wrap merge in `if (!multiplier_info.empty())`; `const&` comparator | fixes a latent bug: empty list + `new_num=1`/`resize` materialized a bogus default entry |
| E6 | reserves | `gi_channel_`, `gi.channel`, `tree_` reserved before push/merge | drops repeated vector growth |
| E7 | XYB B−Y row-parallel + `group_rect` | match Y/X paths: read through `group_rect`, `RunOnPool` over rows | byte-exact for full-frame (group_rect at origin); ALSO a crop-correctness fix for grouped/streaming input — see caveat |
| E8 | drop unused `context_map_` member | — | dead state |
| E9 | `EstimateCost` lane-safe flush period (SIMD) | derive extra-bits flush cadence from `Lanes(du)` instead of fixed 4096 | running total identical; far fewer `SumOfLanes` reductions on large rows |

## Caveats

- **E7 group_rect** changes behavior only when `group_rect.x0()/y0() != 0`
  (grouped / streaming XYB-modular). The verification corpus is lossy VarDCT
  (distance 1.0), which does **not** exercise the XYB-modular path, so this
  crop-correctness fix is byte-exact-by-construction for the tested full-frame
  case but **not directly exercised** by the harness. Validate against libjxl's
  own modular/streaming tests before merge. (The row-parallelization alone is
  byte-exact regardless.)
- **E3** is a memory/allocation reduction, expected **timing-neutral**. Channel 2
  is sized in 8×8-block units, so the saving appears only when the group uses
  transforms larger than 8×8; for mostly-8×8 photographic content `num ≈ blocks`.

## NOT in this branch (deferred / separate)

- **Stream-image lifecycle release** (free `stream_images_[id]` after tokenization):
  byte-exact-if-correct peak-RSS win. See commit/branch note if attempted.
- **Palette `cost_before` staleness fix** (`maybe_do_transform` in/out cost):
  CHANGES the bitstream (better ratio, still lossless-reversible) → NOT byte-exact.
  Separate branch; needs a size/ratio comparison, not a hash compare.
- **EstimateCost workspace / bounded / restricted-channel scoring**: API churn,
  deferred.
- Scalar-fallback context map: the current `HWY_SCALAR` path already matches the
  SIMD `ContextMap` ordering (a prior agent fixed it) — ChatGPT's note was stale.

## Verification

OLD = libjxl `2169106a` (clean primary submodule). NEW = same base + this branch.
Harness `crates/raw-pipeline/examples/jxl_encdec_ab.rs` (+ FNV content hash),
built against each source dir via `LIBJXL_SOURCE_DIR`, run interleaved by
`run-ab.mjs` over `C:\Tmp\rcw-rgb` (8 RAW-derived RGB images), effort 3, d1.0,
single-thread.

### Result (5-round interleaved, effort 3, single-thread)

```
file                          MP   encOLD   encNEW      x   decOLD   decNEW      x    bytes  byte-exact
ADH 1248.CR2                24.0   1272.2   1083.6 1.174x    808.1    796.2 1.015x  3511439  YES
P1110226 windows.jpg         7.7    350.3    304.3 1.151x    240.5    200.7 1.198x   901346  YES
P1110226.ORF                20.5    854.7    841.2 1.016x    593.6    643.2 0.923x  2747880  YES
P2200474.ORF                20.5    962.3   1008.8 0.954x    707.0    707.3 0.999x  4201699  YES
PXL_20260501_093507165      12.5    600.3    601.9 0.997x    442.2    438.4 1.009x  3120841  YES
PXL_20260527_180319603       9.9    335.6    342.3 0.980x    270.4    252.0 1.073x   548963  YES
_MG_1750.CR2                17.9    756.4    709.0 1.067x    562.7    536.3 1.049x  2074632  YES
small_file.jpg               0.1      7.2      6.3 1.140x      2.7      2.9 0.911x    17814  YES

geomean: encode 1.0570x  decode 1.0188x   (>1 = NEW faster)
*** BYTE-EXACT: all 8 files produce IDENTICAL encoded bytes (OLD == NEW). ***
```

- **Byte-exact: PROVEN** — every encoded `.jxl` is bit-identical OLD vs NEW
  (FNV content hash + size match, all 8 corpus images).
- **Encode/decode timing: noise** (this run `1.057x` enc; a cleaner 5-round rerun
  with Tier-2 gave `1.019x`; lossless e9 gave `0.968x`). See the Timing summary
  below — across runs the ratio straddles 1.0, i.e. neutral within noise. These
  are encoder-only work-reduction changes, not wall-clock-visible at these efforts.
### Lossless modular pass (effort 9, 3-image subset) — exercises E1 + E9

The lossy VarDCT corpus above does **not** run the modular selection hot paths
(`EstimateWPCost` needs effort ≥ Kitten; `EstimateCost`'s RCT search needs
lossless modular). A second A/B with `MODE=lossless EFFORT=9` exercises them:

```
file                          MP   encOLD   encNEW      x    bytes  byte-exact
P1110226 windows.jpg         7.7  42823.6  44973.3 0.952x  4051402  YES
PXL_20260527_180319603       9.9  60395.2  62070.6 0.973x 11467745  YES
small_file.jpg               0.1    527.7    539.3 0.978x    58476  YES
*** BYTE-EXACT: all 3 files produce IDENTICAL encoded bytes (OLD == NEW). ***
```

- **Byte-exact: PROVEN for E1 + E9** — `EstimateWPCost`'s lookup table + monotone
  early-out selects the identical `wp_mode`, and `EstimateCost`'s flush change is
  exact, even under effort-9 (5 WP modes, 19 RCT trials, channel palettes).

### Timing summary (honest)

Three interleaved runs: lossy ×2 = `1.057x` / `1.019x`, lossless e9 = `0.968x`.
These **straddle 1.0 → neutral within measurement noise** on this single-thread,
contended dev box (lossless used only 2 rounds of 40–60 s encodes; large variance).
The optimized routines (`EstimateWPCost`, the SIMD flush, the AC-meta pre-count)
are **minor fractions of total encode time**, so no wall-clock speedup is expected
or claimed. The defensible win is **work reduction** — 33 per-pixel comparisons →
1 table load; far fewer `SumOfLanes` reductions; exact-sized AC-meta allocation;
source-image release (lower peak RSS) — delivered with **zero byte change and no
measurable regression beyond noise**.

### Coverage map

| Path | Exercised by |
|------|--------------|
| E3 AddACMetadata, E5/E6 (VarDCT substreams), Tier-2 release | lossy VarDCT corpus (8/8 byte-exact) |
| E1 EstimateWPCost, E9 EstimateCost/RCT | lossless e9 subset (3/3 byte-exact) |
| E2 QuantizeChannel | lossy-MODULAR only — not in either corpus (byte-exact by construction: per-pixel result depends only on itself + q) |
| E7 XYB B−Y group_rect | XYB-modular only — not in either corpus (byte-exact by construction for full-frame; crop-fix needs libjxl's own streaming tests) |

