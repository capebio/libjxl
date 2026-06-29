# enc_modular optimization — handoff

**Agent:** overnight run, 2026-06-29. One-writer discipline: own worktrees, unique
branches, pushed; **main untouched, no gitlink bump** (integrator's job).

## Branches pushed (hand off to integrator)

| Repo | Branch | Base | Contents | Status |
|------|--------|------|----------|--------|
| submodule (capebio/libjxl) | `perf/enc-modular-byteexact-jun29-m7k2` | `2169106a` | 9 byte-exact opts + Tier-2 lifecycle + findings | **VERIFIED byte-exact, pushed** |
| superproject (origin) | `bench/enc-modular-verify-jun29-m7k2` | `6cda6c21` | verify harness (FNV hash + MODE) + runner | pushed (test scaffold; NOT for merge into product) |

Worktrees (leave for review, remove after merge):
`C:\Foo\rcw-enc-modular` (submodule branch), `C:\Foo\rcw-verify` (superproject branch).

## What landed (all byte-exact — see PERF_FINDINGS_enc_modular.md)

E1 WP-cost lookup+early-out+overflow, E2 QuantizeChannel row-parallel,
E3 AddACMetadata exact ACS+QF channel, E5 multiplier_info empty guard,
E6 reserves, E7 XYB B−Y parallel+group_rect, E8 dead `context_map_`,
E9 SIMD extra-bits lane-safe flush, **Tier-2** source-image release after
tokenization (peak RSS).

Verification: OLD(`2169106a`) vs NEW built against each via `LIBJXL_SOURCE_DIR`;
`jxl_encdec_ab` (+FNV content hash) over the RAW corpus, interleaved. **Encoded
bytes bit-identical OLD==NEW** on every file, both lossy (VarDCT) and lossless
(modular) configs.

## Deferred — NOT landed (need deliberate work, documented here)

These change the bitstream or are heuristic/API churn → per CLAUDE.md they need
benchmark data, not an unverified land. ChatGPT flagged most as "experiment first".

### D1. Palette `cost_before` staleness fix (changes bitstream — ratio)
`maybe_do_transform` computes `cost_after` but discards it, so the per-channel
palette loop compares each later candidate against the stale pre-palette cost.
Make the cost an in/out accumulator:

```cpp
// signature: float cost_before  ->  float* JXL_RESTRICT cost_io
const float cost_before = *cost_io;
...
if (cost_after > cost_before) { /* invert + pop, did_it=false */ }
else { *cost_io = cost_after; }   // update running baseline
```
Callers in `try_palettes` pass `&cost_before` (3 sites). **Effect:** changes
transform-selection (and the `nb_colors` heuristic that reads `cost_before`), so
output differs — still lossless-reversible. Direction (helps/hurts ratio) is
uncertain; needs a multi-corpus **size** benchmark (lossless modular, effort ≥5),
not a hash compare. Branch off `2169106a`, measure `bytes` delta with the harness
(`MODE=lossless`).

### D2. EstimateCost workspace reuse (byte-exact, API churn)
Per-group RCT search runs up to 19 `FwdRct + EstimateCost`; each `EstimateCost`
allocates its aligned row scratch + 17 histogram backings afresh. Add an
`EstimateCostWorkspace` (retain row scratch capacity + rounded max width +
histogram capacity) threaded through `PrepareStreamParams`'s RCT loop. Byte-exact
(no value change), removes up to 19 allocator round-trips per group. Deferred:
touches the internal SIMD cost API + plumbing; bench the alloc win first.

### D3. EstimateCost bounded / restricted-channel scoring (risk: ties)
`EstimateCost(img, first_channel, num_channels, upper_bound)` to (a) score only
the 3 transformed colour channels during RCT trials and (b) early-reject losing
candidates. ChatGPT's own caveat: the estimator's fractional-entropy accumulation
means an exact tie could move by one rounded unit → may change RCT choice. Treat
as experiment; verify byte-exact per-corpus before trusting.

### D4. Not pursued (correctly)
FMA in VarDCT DC (changes rounding), manual WP vectorization, SIMD packed-key row
(needs AVX2 flipflop), scalar-fallback context map (already matches ContextMap —
ChatGPT's note was stale on the current source).

## Build / verify recipe (reproduce)

```powershell
# Worktree third_party (only highway/brotli/skcms needed) copied from primary.
# Build harness against a libjxl source dir, MSVC+LLVM toolchain:
C:\Foo\rcw-verify\build-ab.ps1 <LIBJXL_SOURCE_DIR> <out_exe>
#   OLD = C:/Foo/raw-converter-wasm/external/libjxl-012   (2169106a, clean)
#   NEW = C:/Foo/rcw-enc-modular                          (this branch)
# Run interleaved A/B (byte-exact hash + enc/dec timing):
$env:MODE="lossy";    $env:RGB_DIR="C:\Tmp\rcw-rgb";       node C:\Foo\rcw-verify\run-ab.mjs 5 3
$env:MODE="lossless"; $env:RGB_DIR="C:\Tmp\rcw-rgb-small"; node C:\Foo\rcw-verify\run-ab.mjs 2 9
```

## Integrator notes
- Merge order: land `perf/enc-modular-byteexact-jun29-m7k2` on submodule, then bump
  superproject gitlink. The `bench/...` superproject branch is test scaffolding
  (harness `MODE`+hash, runner) — cherry-pick the harness change if wanted, do not
  merge wholesale.
- E7's `group_rect` change is a crop-correctness fix for grouped/streaming
  XYB-modular, NOT exercised by the corpus (lossy VarDCT + lossless RGB). Run
  libjxl's own modular/streaming roundtrip tests before relying on it.
