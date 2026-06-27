# GaborishInverse tile-scratch — measured findings

Branch `perf/gaborish-tile-scratch`. Native clang-cl/Ninja build, `tools/gaborish_bench.cc`
OLD (full-plane scratch) vs NEW (xrect tile scratch), flipflop cmd-mode.

## Correctness
Byte-exact OLD == NEW over all rect modes (`full`, `partial`, `stream`),
pow2 + non-pow2, group size 2064. The symmetric5 coeff hoist is byte-exact by
construction.

## Performance — the win does NOT materialize on any real path

| rect mode | geometry | result (median, trust:high) |
|-----------|----------|------------------------------|
| `full`    | rect = whole plane (one-shot encode path) | −7.5% (within stdev → neutral) |
| `stream`  | plane = DC group + 8px (kBlockDim) border ring; rect = the group (~99% of plane) — **what EncodeFrameStreaming actually passes** | −5.4% (within stdev → neutral/slightly slower) |
| `partial` | rect = interior quarter-plane (rect ≪ plane) | +18.8% faster |

The +18.8% only appears when `rect` is much smaller than the plane. **No real
caller produces that geometry:**

- `enc_modular.cc:707` and the one-shot `EncodeFrameOneShot` path pass the full
  frame rect → plane == rect.
- `EncodeFrameStreaming` (enc_frame.cc) processes one DC group at a time into a
  per-group `color` buffer sized to the group **plus only an 8px border ring**
  (`max_border = kBlockDim`, enc_frame.cc:1553). So `group_rect` ≈ 99% of the
  plane and the tile-scratch only avoids copying/allocating the border ring
  (~1%), which is lost in noise.

Streaming itself is gated to `options.chunked` non-progressive lossy XYB
(facade.ts:530 → buffering=2; `CanDoStreamingEncoding`). The default gallery
encode is buffering=0 → one-shot.

## Conclusion
The change is correct and byte-exact but **perf-neutral on every real encode
path**. It marginally lowers transient per-group scratch allocation in streaming
(border ring only). Keep as a harmless cleanup or drop; do not expect a speedup.
The earlier "18.8% partial-rect win" was a synthetic-geometry artifact.

## Streaming-path profile (encode_stream_bench, buffering=2, 4096² e5)

Profiled the real streaming encode via the public API (`tools/encode_stream_bench.cc`,
JXL_STAGE_TIMERS). 4096² = 4 DC groups. Steady-state per DC group (2nd encode):

| stage | share of group |
|-------|----------------|
| ComputeVarDCTEncodingData (XYB + adaptive-quant + ACS + chroma-from-luma + DCT) | **~91%** |
| EncodeGroups (histogram build + rANS) | ~5% |
| TokenizeAllCoefficients | ~2% |
| setup + alloc + gaborish + AR heuristics (remainder) | ~6% (overlaps) |

**No streaming-specific fat.** The per-group cost is the same VarDCT heuristic
block as one-shot encoding; ACS search dominates (consistent with prior
profiling: ACS = 76% of e5). Per-group buffer re-allocation is negligible in
steady state — the apparent 2× slowdown on the very first groups of a fresh
process is allocator/code cold-start (gone by the 2nd encode), not a per-encode
streaming inefficiency, so libjxl-level buffer reuse buys nothing (the CRT
already recycles same-size blocks).

Conclusion: the streaming path has no cheap, isolated win. The lever is the
shared ACS/DCT/adaptive-quant hot path already targeted by other branches
(ACS-prune, DCT-fusion, enc_group, CfL).
