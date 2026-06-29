# Block-context-map: free size from fixing default-map detection

Branch: `perf/enc-ctxmap-largetx-split-jun29-z9k` (off submodule `main` @10783f7e)
Goal: shrink files with **zero reconstruction change** (decoded pixels
bit-identical; only entropy-coding/signalling changes). Not merged.

## Landed: fix stale default-ctx-map detection (`enc_context_map.cc`)

`EncodeBlockCtxMap`'s "is this the default map?" fast path tested
`ctx_map.size() == 21`, but the default `BlockCtxMap` is `3 * kNumOrders = 39`
entries (it was 21 when `kNumOrders == 7`). So the check **never fired** and
every default-map frame signalled the full 39-entry context map instead of the
1-bit "use default" flag. Fixed to `3 * kNumOrders`.

**Verification (native cjxl A/B, OLD `==21` vs NEW, 5 images × e{5,7,9}):**
decode SHA-256 **IDENTICAL in all 15 cases** (zero reconstruction change), and
size **never regresses** (3 wins, 0 losses):

| image | size Δ |
|---|---|
| ellipses (256×128, uses default map) | **−23 B (−0.5%)** at e5/e7/e9 |
| torture / hdr_room / splines / grayscale (large) | −0 B |

Encode time neutral (±a few % noise). Net corpus −69 B / −0.011%.

**Why the split between images:** there's a VarDCT context-map optimizer,
`FindBestBlockEntropyModel` (`enc_heuristics.cc:69`). For frames with
`tot ≥ (1<<10)·distance` blocks (≈256×256 at d1.0) it builds a *custom* map
(qf split + occurrence clustering), so those never used the default and the fix
can't help them. Small frames fall through to the default map — that's where
this fix pays off (thumbnails, previews, the O1 multi-blob TTFP model's small
blobs). It is a genuine latent-bug fix: free, correct, never-regressing.

## Tested & rejected: raise the cluster cap (9 → 11 luma)

`FindBestBlockEntropyModel` caps luma clusters at 9 (+5 chroma = 14), leaving 2
of the format's 16 context slots unused. Hypothesis: large detailed frames
could keep more classes separate for free (ANS clustering re-merges any that
don't pay off). **Measured on a 2048² image (65 536 blocks, hits the cap):
REGRESSED** — +1694 B (+0.24%) at e5, +1384 B (+0.20%) at e7, and +6–8% slower;
decode stayed identical. More block contexts cost more in map signalling +
extra histograms than the greedy ANS clusterer recovers. The cap of 9 is
well-tuned. **Reverted.**

## Considered & not pursued

- **Split large-transform contexts** (square vs rect): subsumed by
  `FindBestBlockEntropyModel` (already clusters orders for large frames) and
  hard-capped at 16 contexts (default already uses 15 → 1 free slot). Marginal.
- **`enc_heuristics.cc:83` bug**: the `decoding_speed_tier ≥ 1` branch assigns
  the simple 2-context map into a local **copy** (`auto bcm = *block_ctx_map;`)
  then returns, discarding it — so that tier silently keeps the default map.
  Real, but fixing it is output-changing (and would *enlarge* files: the simple
  map trades compression for decode speed) and only affects the niche
  `decoding_speed_tier ≥ 1` mode. Left alone.

## Reproduce

```
node tools/flipflop/ctxmap_eval.mjs C:/Tmp/acs-test/ctxbase C:/Tmp/acs-test/ctxfix   # the fix
node tools/flipflop/ctxcap_eval.mjs C:/Tmp/acs-test/ctxfix  C:/Tmp/acs-test/ctxcap    # rejected cap raise
```
