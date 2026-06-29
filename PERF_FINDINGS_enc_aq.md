# enc_adaptive_quantization.cc — byte-exact optimization pass

Branch: `perf/enc-aq-byteexact-2026jun29-k4x9` (off submodule `main` @ 2169106a)
Source: consolidation of two independent ChatGPT analyses of this file
(union of the byte-exact-safe suggestions; the behaviour-changing "experiments"
were deliberately deferred — see bottom).

All landed changes are **byte-exact** (the encoded codestream is unchanged):
each is either byte-exact by construction (operation order preserved) or, for
the one non-trivial rewrite (FuzzyErosion), proven empirically — see Verification.

## Path map

`InitialQuantField` → `AdaptiveQuantizationMap` → `ComputeTile` →
`FuzzyErosion` / `PerBlockModulations` runs on **every VarDCT encode** (hot).
`FindBestQuantization` (Kitten/Tortoise butteraugli loop) and
`FindBestQuantizationMaxError` (max_error mode) are **slow-path only**.

## Landed optimizations

### Hot path (every encode)

| # | Change | Why byte-exact |
|---|--------|----------------|
| A | **FuzzyErosion → 2×2 output cells.** The old code scanned the source at full resolution, recomputing a 3×3 rank-4 erosion per source pixel and accumulating four of them into each output via read-modify-write with `%2` branches. The new code walks output cells, stages the shared 4×4 source window once (36 loads → 16), and derives the four overlapping rank-4 erosions. | Selection network, per-edge clamping and the `((a+b)+c)+d` accumulation order are preserved exactly. Proven bit-identical (see Verification). |
| B | **Hoist the Highway descriptor** out of the `PerBlockModulations` row loop. | Descriptor is stateless. |
| C | **MaskingSqrt: precompute the loop-invariant inner `Sqrt(kMul·1e8)`** once as a `static const float`. | `std::sqrt` and Highway `Sqrt` are both IEEE correctly-rounded for `float`, so the constant equals the per-call result bit-for-bit. |
| D | **`reserve()` `pre_erosion`** before the emplace loop. | Allocation only. |

### Slow path (Kitten/Tortoise + max_error)

| # | Change | Why byte-exact |
|---|--------|----------------|
| E | **FindBestQuantization: skip the discarded terminal Butteraugli compare + TileDistMap** on the final iteration in release builds. `RoundtripImage` (which materialises the final encoder coefficients) still runs; only the comparison/tile-map — whose results feed the skipped quant-field update and debug heatmaps — are dropped. | Final score/diffmap/tile_distmap have no release-mode consumer; `num_butteraugli_iters` is still incremented. |
| F | **AdjustQuantField:** skip the 1×1-strategy no-op (`covered==1`), and compute `mean` only when `covered>=4` (the only case it is consumed). | 1×1 path wrote the value back onto itself; `mean` is unused for covered 2–3. |
| G | **TileDistMap: zero-margin fast path** (the sole caller passes `margin==0`) — drops the per-pixel border/corner weighting branches and the `pixels += xmul` accumulation. | With `margin==0`, `xmul==ymul==1`, so `dist_norm` is a plain sum and `pixels` is an exact integer count. |
| H | **MaxError:** clip the block scan to the decoded image once (`x_lo..x_hi`, `y_lo..y_hi`) instead of a per-pixel `if (>= size) continue`; skip the `qf_mul==1.0f` no-op multiply. | Same pixels visited in the same order for the `std::max` reduction; ×1.0 is identity. |
| I | **MaxError fixed-point early exit:** if no block changed this iteration (all `qf_mul==1.0`), the next roundtrip would be identical → break. | The post-loop `SetQuantField` uses the unchanged field; deterministic fixed point. |
| J | **Hoist `quantizer.InvGlobalScale()` / `Scale()`** out of the per-pixel update loops in FindBestQuantization. | Loop-invariant within an iteration (quantizer is only re-set at the next iteration's top). |

## Verification

### FuzzyErosion (A) — empirical byte-exact + timing
`tools/enc_aq_fuzzy_ab.cc` lifts the OLD and NEW bodies into one TU (FuzzyErosion
is pure scalar C++, no Highway) and runs an interleaved A/B over 9 configurations:
output sizes 2×2 … 512×384, `from_rect` offsets ∈ {0,1}, tight (clamping) and
padded edges, non-zero output-rect offsets, varied `butteraugli_target`, and
tie-injected inputs (to exercise the strict-`<` selection network).

```
clang++ -O2 -std=c++17 tools/enc_aq_fuzzy_ab.cc -o enc_aq_fuzzy_ab && ./enc_aq_fuzzy_ab
```

Result: **ALL 9 configs BYTE-EXACT** (FNV of the output image identical OLD==NEW).
Timing (median, interleaved): **+6 – 11 % faster in the realistic per-tile size
regime (16²–128²)** — FuzzyErosion is called once per encoder tile (`rect_out`
≤ `kEncTileDimInBlocks²`), so that regime is what production hits. Roughly neutral
(−1 … +3 %) at huge single-shot sizes where it becomes memory-bandwidth bound.

### Whole file — compilation
Compiled clean as part of `jxl-ffi` via the project's MSVC + clang-cl build
(`cargo build -p jxl-ffi`) with `LIBJXL_SOURCE_DIR` repointed at this worktree,
isolated `CARGO_TARGET_DIR`, third_party junctioned from the primary checkout.
**4m22s, exit 0** — covers B, C, E–J and the Highway multi-target passes.

### End-to-end OLD-vs-NEW codestream A/B (real full-res RAW)
Built `examples/jxl_encdec_ab` against OLD (`external/libjxl-012` @ 2169106a)
and NEW (this worktree); encoded the project's real RAW-derived corpus
(`C:\Tmp\rcw-rgb`: CR2/ORF/DNG/JPG, 0.07–24 MP, 8 files) at **effort 3**
(distance 1.0). Compared encoded byte count **and** decoded-image Butteraugli
(both shift if any quant-field bit differs):

```
file                                         old_bytes  new_bytes  old_bt  new_bt
ADH 1248.CR2 (24MP)                            3511439    3511439   0.4328  0.4328
P1110226 windows.jpg (7.68MP)                  901346     901346    0.0348  0.0348
P1110226.ORF (20.5MP)                          2747880    2747880   0.1005  0.1005
P2200474.ORF (20.5MP)                          4201699    4201699   0.0935  0.0935
PXL_...093507.dng (12.5MP)                     3120841    3120841   0.1261  0.1261
PXL_...180319.dng (9.9MP)                      548963     548963    0.0581  0.0581
_MG_1750.CR2 (17.9MP)                          2074632    2074632   0.1384  0.1384
small_file.jpg (0.07MP)                        17814      17814     0.0538  0.0538
=> 8/8 byte-identical size AND identical Butteraugli
```

Effort 3 exercises the **hot path** (InitialQuantField → ComputeTile →
FuzzyErosion / PerBlockModulations / MaskingSqrt = changes A–D), i.e. the path
that runs on every encode.

**Slow path (effort 9 / Tortoise).** Re-ran the A/B at effort 9, which drives the
`FindBestQuantization` Butteraugli loop (≈44 s/encode on the 7.68 MP file —
confirming the loop is actually entered, so changes E, F, G, J execute):

```
file (e9)               old_bytes  new_bytes  old_bt  new_bt
P1110226 windows.jpg    846727     846727     0.0473  0.0473
small_file.jpg          15519      15519      0.0743  0.0743
=> byte-identical + identical Butteraugli
```

(max_error mode, changes H/I, is not reachable through this harness but is
byte-exact by construction and compiled clean.)

[Timing in the A/B itself is noise — OLD and NEW ran concurrently under CPU
contention and across debug/release profiles; perf evidence is the FuzzyErosion
microbench above.]

The remaining changes are byte-exact by construction (operation order and
visited-element sets preserved, as argued above) and the codestream A/B confirms
it in situ.

## Deferred (NOT landed) — experiments requiring rate-distortion / perf evidence

These appeared in the analyses but are **not byte-exact** or carry real risk;
they need corpus-level size/Butteraugli sweeps, not a unit byte-compare, so they
are out of scope for a byte-exact pass:

- **Shared gamma-Laplacian staging** between the scalar `mask1x1` pass and the
  SIMD pre-erosion pass — changes the numeric route (scalar `log1p` vs SIMD).
- **Fused Gamma/Blue/HF block modulation scan** — byte-exact only if every
  accumulator's lane-reduction order is preserved; real WASM register-pressure
  risk. Needs measurement.
- **`ComputeTile` 4-row masking batching** (remove `diff_buffer` reloads) —
  plausibly byte-exact if the `((d0+d1)+d2)+d3` order is kept, but a substantial
  rewrite; deferred to keep this pass safe.
- **SIMD tail-guard `x + 1 + Lanes < x_end` → `x + Lanes < x_end`** — tiny gain,
  real out-of-bounds-read risk at the row tail; deferred.
- **Strategy-granular TileDistMap → quant-field update fusion** — relies on the
  AdjustQuantField uniformity invariant; risky.
- **Fast `log1p` / fast `pow`** approximations — explicitly not byte-exact.
- **Persistent `AQRoundtripWorkspace`** (reuse decoder/render-pipeline state across
  AQ iterations) — large, needs a trustworthy per-iteration reset contract.
- **Removing the unused `rescale` param** of `FindBestQuantizer` — public-API
  change, out of scope.
