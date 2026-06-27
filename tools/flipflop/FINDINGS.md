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
