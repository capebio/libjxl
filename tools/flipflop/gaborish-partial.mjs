// flipflop cmd-mode harness: GaborishInverse OLD (full-plane scratch) vs
// NEW (xrect-sized tile scratch) over a PARTIAL rect — the streaming/group
// case where the optimization actually saves allocation + back-copy.
//
// The driver exe ignores the fractal {input}; it builds its own deterministic
// image internally and runs the chosen impl `REPS` times. Process wall-clock is
// the measure, so REPS is sized to dominate ~25 ms process startup.
//
// Run (engine lives in the main repo root):
//   node --expose-gc C:/Foo/raw-converter-wasm/flipflop.mjs \
//     C:/Foo/rcw-gaborish/tools/flipflop/gaborish-partial.mjs \
//     --sizes 512 --types mandel --rounds 512:11 --print \
//     --journal "C:/Foo/rcw-gaborish/tools/flipflop/gaborish-journal.toon"

const EXE = 'C:/Foo/rcw-gaborish/build-bench/gaborish_bench.exe';
const S = 1536;
const REPS = 80;

export const name = 'gaborish-partial-rect';
export const description =
  'GaborishInverse partial rect: full-plane scratch vs xrect tile scratch';

export const variants = [
  { name: 'old-fullplane-scratch', baseline: true,
    cmd: `"${EXE}" old partial ${S} ${REPS}` },
  { name: 'new-tile-scratch',
    cmd: `"${EXE}" new partial ${S} ${REPS}` },
];
