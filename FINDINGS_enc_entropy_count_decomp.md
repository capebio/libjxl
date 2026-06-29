# enc_entropy_coder.cc — count-decomposition pass (round 2)

Branch: `perf/enc-entropy-count-decomp-jun29-z4x` (capebio), off submodule main
**@10783f7e**. Worktree `C:\Foo\rcw-enc-entropy-z4`. Commit 38e6dd97.

Trunk @10783f7e already carries round-1 (`tmp_num_nzeroes` Image3I→Image3B,
interior-block skip `bx += covered_blocks_x()`, hoisted `kStrategyOrder`,
subsampling-aware reserve). This round is the next, orthogonal layer.

## What changed (byte-exact by construction)

Both nonzero counters dropped their SIMD masks in favour of the algebraic
identity `AC_nonzeros = all_nonzeros − LLF_nonzeros`:

1. **`NumNonZero8x8ExceptDC`** — one uniform mask-free SIMD pass over all 64
   coefficients, then subtract `(block[0] != 0)`. Removes the `dc_mask` stack
   object, its `Load`, the `AndNot`, and the first-row special case.

2. **`NumNonZeroExceptLLF`** — one contiguous mask-free SIMD pass over the whole
   `cx*cy*64` block (block is row-major and contiguous), then a small scalar
   subtract of the top-left `cx*cy` LLF rectangle. Removes the per-row
   `llf_mask` `LoadU` + `AndNot`. Also **robust for every transform width**: the
   old `llf_mask_lanes` only seeded `-1` lanes for `cx<=4` and was correct only
   because AC-residual LLF slots are already zero — the new form needs no such
   assumption (it subtracts the actual LLF nonzeros, which is 0 in practice).

Removed the now-unused `AndNot` using-declaration.

## Why byte-exact

`all_nonzeros` and `LLF_nonzeros` are disjoint counts over the same block, so
`AC_nz = all_nz − LLF_nz` is an identity. The SIMD zero-count reduction
(`Zero`/`Eq`/`VecFromMask`/`SumOfLanes`) is unchanged from the original — only
the masking was removed and a scalar subtract added. The stored density value
`ceil(nzeros/covered_blocks)` is unchanged.

## Verification

Clean A/B: both OLD (trunk file) and NEW (this change) built to plain enc WASM
from the **same worktree source** (file-flip isolates exactly this one file),
`build-pgo.mjs --plain`, tier `simd`.

`tools/enc-sha-flipflop.mjs OLD NEW`:

```
=== SHA256 CORRECTNESS (1024×768) ===
  e3: PASS  d6d18b6395feca16 == d6d18b6395feca16
  e5: PASS  dd5e4b58398a201b == dd5e4b58398a201b
  e6: PASS  c65271544aab0267 == c65271544aab0267
  e7: PASS  84206d16be80b997 == 84206d16be80b997
  e9: PASS  5128d6e941d14f9a == 5128d6e941d14f9a
  ALL PASS — byte-exact ✓
```

Timing (true-alternation, ratio = old/new min, >1 ⇒ NEW faster):
`256²e3 1.006 · 256²e5 1.002 · 512²e3 1.023 · 512²e5 1.013 · 1024²e3 0.978 ·
1024²e5 1.029` — **neutral within noise**, no regression (5 of 6 ≥ 1.0; the one
0.978 min is contradicted by its own median 1.016). As round-1 found,
tokenization is too small a slice of whole-encode to move the e2e number; value
is fewer kernel instructions (mask `Load` + `AndNot` gone) + width-robustness,
at zero output change. Kept under the "theoretically-better + non-regression"
rule.

## Status

PUSHED to `capebio/perf/enc-entropy-count-decomp-jun29-z4x`. NOT merged / NOT
gitlink-bumped — handoff to integrator. Single file:
`lib/jxl/enc_entropy_coder.cc`.

Deferred cross-file levers (quantizer-emitted nonzero counts, scan-order
coefficient staging, persistent token-reserve hint) → `Questions_deferred.md`.
Rejected micro-levers (maintained-bucket context, 4:4:4 split, perimeter density
writes, ring buffer, etc.) → `docs/1 rejected optimizations.md`.
