# enc_bit_writer optimization — findings

Branch: `perf/enc-bit-writer-append-jun30-w8k4` (capebio, off submodule main `00f4d7fc`).

## What changed (all byte-exact for emitted output)

| Item | Change | Why |
|------|--------|-----|
| `AppendUnaligned` | byte-aligned dst → single `memcpy` of all full bytes; unaligned dst → 7 source bytes per `Write(56)` instead of 1 byte per `Write(8)` | ~7× fewer read-modify-write 64-bit stores; real enc_ans histogram concat hits the unaligned path |
| `WithMaxBits` | `std::function` → forwarding-ref template (defined in header) | removes per-scope type erasure + possible closure heap-alloc + indirect call |
| `WithMaxBits` | always `ReclaimAndCharge` after a successful `Init` | fixes latent leak/dangling `current_allotment_` if `FinishedHistogram` fails |
| `AppendByteAligned` (both) | grow to `BitsWritten()/8 + payload + 1`, not `storage_.size()+…`; validate alignment before resize | stops leaking one accounted byte per repeated append; preserves enclosing reservation |
| `ZeroPadToByte` | advance `bits_written_` over already-zero pad bits instead of `Write(remainder,0)` | skips an RMW store; pad bits are already zero by the zero-tail invariant |
| `PrivateReclaim` | skip resize when `unused_bytes == 0` | avoids a no-op resize round-trip |

## Explicitly REJECTED

- `Write()`'s `uint64_t v = *p` → `memcpy(&v, p, 8)`: **bug**. `*p` is a single
  zero-extended byte load, not a 64-bit load. Widening it reads uninitialized
  trailing storage bytes and breaks the zero-tail invariant (the 8-byte store
  would no longer zero the bytes ahead). `Write()` left unchanged.

## Verification

- Standalone A/B harness `enc_bit_writer_append_ab.cc` (faithful model: garbage-
  initialized storage, +8 write-past-end slack, first-byte-zero, zero-tail kept
  only by the store) — **192 cases, 0 mismatches** across sizes {0,1,5,8,13,56,
  64,191,192,197,1024,4099} × dst offsets 0..7 × 1–2 chained appends + trailing
  write. OLD vs NEW byte-identical.
- Timing (clang -O3, interleaved ×6, 200k iters):
  - aligned/memcpy: 1.11× (8B) → **11.35× (512B)**
  - unaligned/56-bit: 1.31× (24B) → **5.06× (512B)**; no regression at any size.
- Real-compile (`-fsyntax-only`, real headers): `enc_bit_writer.cc` + callers
  `enc_ans / enc_frame / enc_toc / enc_context_map / enc_fields /
  enc_quant_weights / enc_chroma_from_luma` all clean with the templated API.
- `enc_bit_writer_test.cc` gains `AppendUnaligned` coverage (all 8 dst offsets,
  multi-block + partial-bit sources, back-to-back appends, trailing-write
  zero-tail assertion).

## Integrator gate

Full `enc_bit_writer_test` execution + WASM enc A/B byte-exact on a real RAW
stream (the templated `WithMaxBits` is codegen-dependent; confirm size/SHA
identical OLD vs NEW). gtest source not present in worktree third_party.

Build harness: `clang++ -O3 -std=c++17 tools/enc_bit_writer_append_ab.cc -o bw_ab`
