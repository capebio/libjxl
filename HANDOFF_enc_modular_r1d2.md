# enc_modular R1–R4 + D2 — handoff

**Agent:** 2026-06-29. One-writer discipline: own worktree, unique branch, pushed;
**submodule main untouched, no gitlink bump** (integrator's job).

This is a follow-on to `HANDOFF_enc_modular.md` (the E1–E9 + Tier-2 byte-exact pass,
already on submodule main @10783f7e). It lands the ChatGPT items that pass left on
the table, plus deferred item D2.

## Branch pushed (hand off to integrator)

| Repo | Branch | Base | Status |
|------|--------|------|--------|
| submodule (capebio/libjxl) | `perf/enc-modular-r1to4-d2-jun29-q4z` | `10783f7e` | **VERIFIED byte-exact, pushed** |
| superproject (origin) | `verify/enc-modular-q4z-jun29` | `bench/enc-modular-verify-jun29-m7k2` | verify worktree only (reused FNV harness); nothing new to merge |

Worktrees: `C:\Foo\rcw-enc-modular-q4z` (submodule), `C:\Foo\rcw-verify` (superproject harness).

## What landed (all byte-exact)

- **R1 — two-phase single-group prepare.** `ComputeEncodingData` split the
  `PrepareStreamParams` fan-out into phase 1 (parallel, per-group id != 0 copies out
  of the immutable full image) + phase 2 (serial, the single-group Global stream id 0
  that *mutates* `stream_images_[0]`/`stream_options_[0]` via its RCT/palette/WP
  search). The old single-phase form raced those mutations against the copies and the
  concurrent `stream_options_[stream] = stream_options_[0]` reads — benign for output
  (the empty single-group per-group copies are discarded) but a real data race.
  Split is byte-exact and lets the Global RCT/`do_transform` use the pool;
  `PrepareStreamParams` gained a defaulted `transform_pool` (nullptr for per-group
  work to avoid nested-pool oversubscription).
- **R2 — AddACMetadata zero-channel construction.** `Image::Create(...,0)` then append
  ch0/1 (color-tile), ch2 (one entry per first-block), ch3 (full-res EPF), instead of
  `Image::Create(...,4)` allocating four full-res planes and discarding three. ch3 set
  `component=3` to match the old layout (ch0..2 were reassigned → default −1).
- **R3 — reserves:** `stream_images_`, `stream_params_`, `quantizers`.
- **R4 — correctness:** reject uint32+ integer extra channels before the UB `1u<<32`
  shift (mirrors the colour path); clear `gi_channel_[stream]` before append so encoder
  reuse can't accumulate stale channel indices (no-op for single use).
- **D2 — EstimateCost workspace reuse.** New `EstimateCostWorkspace` (aligned row
  buffer + histogram vector) threaded through the HWY_DYNAMIC_DISPATCH per-target impl;
  the per-group RCT search holds one workspace across its baseline + up to 18 trials
  instead of re-allocating the buffer + 17 histograms per call. The 1-arg `EstimateCost`
  wraps a temporary, so all other callers are unchanged.

## Verification

OLD = `C:\Tmp\libjxl-012tip` (clean 10783f7e), NEW = this worktree, both built via
`C:\Foo\rcw-verify\build-ab.ps1` (MSVC+ClangCL), run interleaved with the FNV-hash
harness `jxl_encdec_ab` + `run-ab.mjs`.

| Mode | Effort | Corpus | byte-exact |
|------|--------|--------|------------|
| lossy (VarDCT) | 3 | 8 real RAW/JPEG | **8/8 IDENTICAL** |
| lossless (modular) | 9 | 3 | **3/3 IDENTICAL** |

**11/11 files produce bit-identical encoded bytes OLD == NEW.** Lossy exercises
AddVarDCTDC/AddACMetadata (R2) + the two-phase prepare (R1); lossless effort 9
exercises the RCT search (R1 pool path) + EstimateCost workspace (D2) + WP selection.

**Timing: neutral-in-noise.** The harness is single-thread (deterministic for the
hash), so R1's pool parallelism cannot show (it needs multi-thread + a single-group
frame). The decode column — *unchanged code* — drifted ±6–7% per run (lossy 0.94x,
lossless 1.07x), i.e. environmental bias of the same magnitude as the encode deltas,
so encode is neutral within noise. Consistent with the prior pass: enc_modular hot
paths are a small fraction of encode time. R1/D2 are byte-exact structural wins
(fewer allocations, data race removed) → kept per the "not-a-regression + better
memory/efficiency" rule; output-identical, so nothing to discard.

## Deferred / rejected

Appended to superproject `Questions_deferred.md` and `docs/1 rejected optimizations.md`.
Key: subsampled AddVarDCTDC exact-alloc (rare 4:2:0 path); R1 multi-thread single-group
bench (open measurement); D1/D3 still deferred (D3 confirmed NOT byte-exact). Dropped:
y_to_c manual hoist (compiler hoists it), QuantizeChannel pow2 special-case,
DC X/B inner parallelism, work-ordered scheduling, ComputeTokens empty-stream filtering.

## Integrator notes

- Land `perf/enc-modular-r1to4-d2-jun29-q4z` on the submodule, then bump the
  superproject gitlink. No superproject product change in this pass.
- R1's `group_rect`-style assumptions and the two-phase reorder are byte-exact on the
  corpus but the single-group path is lightly covered (only `small_file`); run libjxl's
  own modular/streaming roundtrip tests before relying on the multi-thread pool path.
