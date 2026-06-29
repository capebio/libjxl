# enc_xyb copy-elimination (ChatGPT 1A + 1B)

Branch: `perf/enc-xyb-copy-elim-jun29-k3w9` (capebio fork) off submodule `main @ 0ba69efd`.

## Change
`ToXYB` made two redundant full-image passes when the slow encoder also wants a
linear-sRGB copy (`want_linear == true`). Both are removed, byte-exact:

| Path | Old (2 passes) | New (1 pass) | Traffic removed |
|------|----------------|--------------|-----------------|
| 1A linear-sRGB input | `CopyImageTo(*image,linear)` + in-place `LinearSRGBToXYB` | `LinearSRGBToXYBAndCopy` (store source row → `linear`, write XYB → `image`) | 1 full source re-read (~12 B/px) + 1 pool pass |
| 1B general CMS input | `CopyImageTo(*linear,image)` + in-place `LinearSRGBToXYB` | `LinearSRGBToXYBFrom(*linear,image)` out-of-place | full intermediate copy (write+read, ~24 B/px) |

Per-pixel `LinearRGBToXYB` math is identical (no FP reassociation, no reorder),
so output is bit-identical. Common sRGB hot path (`SRGBToXYB` /
`SRGBToXYBAndLinear`) untouched. These branches fire only for linear-sRGB or
non-sRGB/wide-gamut inputs — the app's RAW→sRGB pipeline never reaches them.

## Verification (native cjxl A/B, OLD 0ba69efd vs NEW, static build, clang-cl 22.1.5)
Inputs (1024×1024, content incl. out-of-gamut values to exercise `ZeroIfNegative`):
- 1A: `lin.pfm` — PFM is read as linear sRGB → linear-sRGB branch.
- 1B: `adobe.png` — AdobeRGB1998 ICC embedded → non-sRGB → `ApplyColorTransform` branch.
Effort `-e 9 -d 1.0` (kitten/tortoise → `want_linear == true`).

### 1A — VERIFIED byte-exact
Genuinely distinct binaries (OLD enc_xyb @0ba69efd vs NEW; exe sha old `68e8a3c6…`
≠ new `7976f456…`), incremental swap in one static build tree.
`lin.pfm -e9 -d1.0` → both 14432 B, SHA `e740172563c65d60bdca` identical.
`lin.pfm -e8 -d1.0` → both 14457 B, SHA `dc3d2fc771154295c50f` identical.
Branch reachability confirmed in `enc_frame.cc:1592` — `linear` is non-null
(`want_linear`) iff VarDCT && `speed_tier <= kKitten` (e8/e9), so
`LinearSRGBToXYBAndCopy` is the path actually exercised here.

### 1B — VERIFIED byte-exact
Input `adobe.png` = 1024² with AdobeRGB1998 ICC embedded. Control test (same
pixels tagged AdobeRGB vs sRGB) → outputs differ (16244 vs 15724 B), proving the
profile is honored and `ToXYB` takes the `ApplyColorTransform` (CMS) branch, i.e.
`LinearSRGBToXYBFrom` is exercised.
A/B (OLD enc_xyb @0ba69efd vs NEW, distinct binaries old `F020BB0E…`/new `E2414F2B…`):
- `adobe.png -e9 -d1.0` → both 16244 B, SHA `023afe35570a07c6de7d` identical.
- `adobe.png -e8 -d1.0` → both 16294 B, SHA `bc54871112f93e6c37b1` identical.

## Decision
Both paths bit-identical to the originals. Timing (process-level, interleaved
×8, `lin.pfm -e8`): NEW vs OLD median −1.1%, mean within noise (copy-elim is
~0.03% of an e8 encode, unmeasurable at process scope). NEW does strictly fewer
memory ops → cannot regress and is theoretically better (memory/efficiency);
kept per rule 10. No dual implementation retained — NEW replaces the two-pass
sequences inline, so nothing to remove.

## Build/verify recipe
Static clang-cl/ninja, junction `third_party/*` from a populated checkout,
`-DBUILD_SHARED_LIBS=OFF` (avoids DLL-per-side capture), `-DJPEGXL_ENABLE_APNG=1
-DJPEGXL_BUNDLE_LIBPNG=1` (so cjxl reads the AdobeRGB PNG). vcvars needs the VS
*Installer* dir on PATH (for vswhere). OLD side = `git checkout 0ba69efd --
lib/jxl/enc_xyb.cc` then incremental `cmake --build`, capture exe, restore.
PFM is read as linear sRGB (drives 1A); `linear` is requested iff VarDCT &&
speed_tier ≤ kKitten (e8/e9).
