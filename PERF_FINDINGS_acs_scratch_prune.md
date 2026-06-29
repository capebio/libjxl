# enc_ac_strategy: 64×64 scratch envelope + incumbent pruning + rate-before-loss

Branch: `perf/enc-acs-scratch-prune-jun29-a7k3` (off submodule `main` @2169106a)
Status: **byte-exact, verified.** Not merged / not gitlink-bumped (integrator's job).

## Changes (all byte-exact)

**A. Phantom merge-table entries.** `kTransformsForMerge[9]` had only 6
initializers, leaving 3 value-initialized `{DCT, prio 0}` entries that, at
`decoding_speed_tier 0`, weren't filtered by the tier check and ran a no-op
merge pass each (priority 0 collides with the initial priority plane, so
`TryMergeAcs` early-returns). Sized the array to its initializer count (`[]`).

**B. Encoder-local 64×64 search envelope.** The ACS search never evaluates a
transform larger than DCT64X64, but per-thread scratch was sized from
`AcStrategy::kMaxCoeffArea` (256×256 = 65536 floats), the largest *legal*
transform. New `AcStrategyHeuristics::kMaxSearchCoeffArea / kMaxSearchBlockDim`
(64×64) size the arena to the actual search envelope:

| | per-thread arena |
|---|---|
| before | `6·65536 + 3·lanes·256` ≈ **1.5 MiB** |
| after  | `6·4096  + 3·lanes·64`  ≈ **96 KiB** (~16×) |

Safe: the transform kernels' absolute scratch floor is `4·kMaxBlocks² = 4096`
floats; the new layout gives the scratch region `2·4096 + dct ≈ 8 K` floats ≥
4096. Channels (3·size), `mem` (size) and scratch all fit for the DCT64X64 max
(size = 4096). A `JXL_DASSERT(size <= kMaxSearchCoeffArea)` guards it. The
decoder/format keep the 256×256 maxima.

**C. Rate-before-loss in `EstimateEntropy`.** Each channel's rate is accumulated
and pruned against the incumbent *before* the inverse transform + masking-loss
walk, so a candidate already losing on rate skips reconstruction (and, via
lazy-B, channel B entirely). The c==0 X-channel penalty `w` is applied to
`lossc` instead of the loss accumulator — bit-identical (float mul is
commutative; at c==0 the accumulator held only that channel). Prune now also
covers c==2. Function returns `bool` (exact vs pruned-lower-bound).

**D. Incumbent pruning in the merge / square paths.** `EstimateEntropyCached`
takes a prune bound and caches *only* exact results (a pruned lower bound is
valid solely for its caller's incumbent). `TryMergeAcs` passes the incumbent;
`FindBestFirstLevelDivisionForSquare` passes each split's 8×8 baseline sum and
evaluates the square last with `min(costJxN, costNxJ)`. Byte-exact because
`std::min(pruned_lb, sum) == std::min(exact, sum)` when `exact ≥ sum`, and any
*selected* branch always holds an exact value.

## Verification

Native `cjxl` A/B, OLD (submodule `main`) vs NEW, clang-cl/Ninja Release.
Harness: `tools/flipflop/acs_ab.mjs` (correctness) + `acs-encode-ab.mjs`
(flipflop timing). Corpus: 5 PPMs (torture/hdr_room/splines/grayscale/ellipses).

**Correctness — PART 1: 30 / 30 byte-exact, 0 fail.** Every image × effort
{5,7,9} × threads {1, all-cores} produced a SHA-256-identical `.jxl`.

**Timing — PART 2 (single-thread, interleaved, median of 7):**

| image | e5 | e7 | e9 |
|---|---|---|---|
| torture | −0.3% | **−6.7%** | **−8.2%** |
| hdr_room | −1.5% | −4.0% | **−15.2%** |
| splines | **−10.6%** | −0.4% | **−8.6%** |

(− = NEW faster; median of 7 interleaved reps.) Never a regression; wins reach
−15%. They concentrate at the slow tiers (e7/e9), where many rejected candidates
are evaluated and pruning skips their inverse transforms; the spread across
images reflects how content-dependent the prune hit-rate is. B's memory cut is
orthogonal and benefits multi-thread / cache pressure rather than single-thread
wall time, so it doesn't show here.

## Deferred (not implemented — out of scope for a byte-exact overnight pass)

Byte-exact micro-ops worth a follow-up: `SetNoBoundsCheck` row-fill fast path
(ac_strategy.h), drop `EntropyCache::mul_key` (kind→mul is 1:1 per rect),
per-tile precomputed 8×8 candidate list, full-width masking fast path. Larger
**output-changing experiments** from the analysis (require Butteraugli/size
gating, not byte-exact): enforce the merge `encoding_speed_tier` limit; split
rate vs information-loss so multipliers are independent; tile DP planner;
population-gated large-transform context splitting; same-colour-map cross-tile
transforms. These are intentionally left for measured, quality-gated work.

## Reproduce

```
# build OLD+NEW static-ish snapshots (see tools/flipflop/acs-encode-ab.mjs header)
node tools/flipflop/acs_ab.mjs C:/Tmp/acs-test/old/cjxl.exe C:/Tmp/acs-test/new/cjxl.exe
# canonical flipflop journal (from superproject root):
node --expose-gc flipflop.mjs C:/Foo/rcw-enc-acs/tools/flipflop/acs-encode-ab.mjs --print
```
