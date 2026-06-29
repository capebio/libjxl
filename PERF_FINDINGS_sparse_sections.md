# Sparse `FrameDecoder::ProcessSections` dispatch — findings

Branch `Decode_12_big_imp`, commit `177591f0` (off submodule `main` `dcaea78f`).

## Change
`ProcessSections` dispatched `RunOnPool` over **every** DC group and **every**
AC group, with the per-task lambda no-opping on groups that had no section this
call. Replaced with compact decoder-owned runnable lists (`ps_dc_runnable_`,
`ps_ac_runnable_`); AC group-dec-cache storage is sized to the runnable count.
Also guarded an unsigned underflow in the `kLastPasses` cap.

The earlier per-call `vector<vector<size_t>>` alloc churn was **already** removed
(reusable `ps_*` flat scratch). This change targets the *remaining* cost: the
`RunOnPool` fan-out itself.

## Byte-exact (by construction, and verified)
AC/DC groups decode independently and the group-dec-cache slot is interchangeable
scratch, so dispatching a runnable subset (and re-indexing storage by compact
task id) cannot change output. Verified empirically — `decode_chunk_bench`
FNV-1a pixel hashes are **identical** OLD vs NEW for both test files across
chunk ∈ {8 KiB, 64 KiB, one-shot} and threads ∈ {0, 4}.

## Timing (native, clang-cl /O2, 20.5 MP RGBA `P2200619`)
Vehicle: `tools/decode_chunk_bench.cc` streams the `.jxl` in fixed chunks
(many `ProcessInput` cycles ⇒ many `ProcessSections` calls with few runnable
groups). A one-shot decode cannot show the change (all sections arrive at once
⇒ every group runnable). Numbers are median of interleaved per-process `min_ms`;
dev box was contended so per-sample variance is high (flipflop `trust:low`).

| Path | OLD | NEW | Δ |
|---|---|---|---|
| One-shot, 4 threads (common path) | 920 ms | 859 ms | neutral (within noise) |
| Streaming 8 KiB, **serial** | ~1560 ms (floor) | ~1515 ms (floor) | neutral |
| Streaming 8 KiB, **4 threads** | ~1120 ms | ~850 ms | **NEW ~24% faster** |

The threaded streaming win is corroborated by three independent measurements
(flipflop warm median 24.8%, interleaved median 24.4%, min-time floor ~23%).

## Why threaded-only
With a thread pool + fine chunks, OLD issues a `RunOnPool` over all (~336 for
20.5 MP) AC groups on **every** `ProcessSections` call — forcing a full
fork-join barrier even when ~1 group is actually runnable. Over the ~100+
section-feed calls a chunked stream produces, that is ~100+ needless barriers.
NEW dispatches only the runnable groups, so `RunOnPool` runs the 1–2 real tasks
without the full-pool fork-join. Serially there is no barrier, so the change is
neutral there (it does strictly-less loop work, lost in noise).

This is exactly the production WASM `simd-mt` decode tier. One-shot decode (the
common still-image path) is unaffected.

Flipflop journal: `docs/outputs/timing tests/flipflop/flipflopjournal.toon`
(record `decode_sparse_sections`).
