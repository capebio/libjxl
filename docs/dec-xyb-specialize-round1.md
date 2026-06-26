# dec_xyb XybToRgb specialization — Round 1 (2026-06-26)

Branch `perf/dec-xyb-specialize` (submodule worktree). Standalone flipflop
microbench `lib/jxl/dec_xyb_standalone_bench.cc` (header-only Highway, no libjxl
link). Interleaved A/B, start-rotation, median of 11 rounds × 20 reps over 1 Mpx.

## Scope reality
The NEON wins from the source discussion (SQRSHRUN narrowing fusion, bulk/tail
split, opaque-alpha shortcut, RDM unmix, dark-block sRGB shortcut) all live in
`FastXYBTosRGB8`, which is `HWY_TARGET == HWY_NEON` only and returns
`JXL_UNREACHABLE` on x86/wasm. This machine is x86; deploy target is wasm
SIMD128. **None of the NEON wins are buildable or benchable here.** They are
left unimplemented and flagged.

The only kernel live on x86/wasm is the generic-Highway `XybToRgb`
(dec_xyb-inl.h:38). Two byte-exact wins apply there:

1. **EqualBias** — when `opsin_biases[0..2]` and `opsin_biases_cbrt[0..2]` are
   bit-equal (the DEFAULT opsin transform = the common case), use one neg-bias
   and one cbrt-bias `Set` instead of three each. Fewer live vector constants ->
   lower register pressure. Byte-exact by construction.

2. **Gray** — when the 3x3 inverse matrix has three bit-identical rows (the
   grayscale OUTPUT-encoding case built by `SetColorEncoding`'s srgb_to_luma),
   compute the matrix product once (3 muls) and store to all three planes
   instead of 9 muls. Byte-exact when rows are bit-identical. This is genuine
   grayscale-IMAGE decode; a user grayscale VIEW-filter on a colour image is a
   separate app-side concern and must NOT destroy the colour base.

## Results (byte-exact YES on both targets, both pairs)

| Pair                  | 4-lane SSSE3 (wasm proxy) | AVX2 8-lane |
|-----------------------|---------------------------|-------------|
| Generic vs EqualBias  | -2.3% .. -6.0%            | -6.2%       |
| Gray-gen vs Gray-spec | -17.2% .. -24.5%          | -7.9%       |

EqualBias: modest, somewhat noisy, but applies to the COMMON colour decode path.
Gray-spec: strong, larger on 4-lane (wasm) than AVX2 — mul-count reduction helps
more when lanes/registers are scarce. Applies only to grayscale-output decode.

## Caveat
4-lane SSSE3 is the closest native proxy for wasm SIMD128 (both 4xf32) but is
still a PROXY. A confirmed win must be re-measured on the real wasm build before
landing into the live `XybToRgb` / `stage_xyb` hot path (lesson: SSE/SSSE3 proxy
has lied about wasm transpose codegen before).

## Next (Round 2, build-gated)
Wire EqualBias + Gray dispatch into `XybToRgb`: detect both conditions ONCE per
`OpsinToLinear` / `stage_xyb` call (not per-vector), pass template flags, keep
the generic path for non-default transforms. Touches dec_xyb-inl.h + callers
(dec_xyb.cc, stage_xyb.cc, dec_modular.cc). Requires full wasm rebuild to
confirm codegen before merge — the expensive step.
