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

## Round 2 (2026-06-26) — wired into hot path + wasm compile-gate

Commit 4cf3a477: refactored `XybToRgb` -> `XybToRgbImpl<kEqualBias,kGray>`
(`<false,false>` = original verbatim, dec_modular unchanged). `DetectXybKernel`
computes both flags ONCE per conversion; shared `XybToRgbRow` + dispatch macro
route `OpsinToLinear` (aligned) and decode `XYBStage` (in-place, unaligned).

**Real wasm build (emscripten 4.0.14, -msimd128, foreach_target), via
`build-pgo.mjs --plain` with `JXL_PGO_LIBJXL_SRC` -> this worktree (third_party
junctioned from primary):**
- `dec_xyb.cc.o` (step 70/201) and `render_pipeline/stage_xyb.cc.o` (step
  123/201) **both compiled clean — zero errors.** COMPILE-GATE PASSED on the
  actual wasm SIMD128 target (4-lane, the deploy reality).
- Build then failed at `packages/jxl-wasm/src/bridge.cpp:2122` —
  `JxlDecoderSetProgressivePaintTarget` undeclared. This is the user's SEPARATE
  uncommitted paint-target WIP (bridge.cpp was already `M` at session start)
  calling a libjxl decoder API absent at source 680ec439. **Unrelated to
  dec_xyb; not touched.**

Empirical SHA byte-exact compare (encode OLD vs NEW) is blocked behind that
bridge.cpp/libjxl symbol mismatch (the user's in-flight work). Byte-exactness
remains: construction-guaranteed (same ops; EqualBias = bit-equal constants;
Gray = bit-identical rows) + standalone-bench-verified bit-identical (Round 1) +
wasm IEEE-float deterministic (no reordering). The unknown the rebuild was meant
to resolve — does it compile + codegen on real wasm — is answered: YES.

## Status
Round 1 (isolate/measure) + Round 2 (wire + wasm compile-gate) DONE on branch
`perf/dec-xyb-specialize` (commits 0fba1683, 4cf3a477). Remaining for merge:
end-to-end SHA byte-exact + perf delta on a linkable wasm build, which needs the
bridge.cpp/libjxl-symbol mismatch resolved first (user's call).
