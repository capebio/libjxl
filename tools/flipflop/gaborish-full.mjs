// flipflop cmd-mode harness: GaborishInverse OLD (full-plane scratch) vs
// NEW (xrect-sized tile scratch) over the FULL-frame rect — the live browser
// (non-streaming) path. Expected ~neutral: temp is full-plane either way, so
// this arm documents that the change does not regress the shipped path.
//
// Run (engine lives in the main repo root):
//   node --expose-gc C:/Foo/raw-converter-wasm/flipflop.mjs \
//     C:/Foo/rcw-gaborish/tools/flipflop/gaborish-full.mjs \
//     --sizes 512 --types mandel --rounds 512:11 --print \
//     --journal "C:/Foo/rcw-gaborish/tools/flipflop/gaborish-journal.toon"

const EXE = 'C:/Foo/rcw-gaborish/build-bench/gaborish_bench.exe';
const S = 1536;
const REPS = 40;

export const name = 'gaborish-full-rect';
export const description =
  'GaborishInverse full rect: full-plane scratch vs xrect tile scratch (neutral)';

export const variants = [
  { name: 'old-fullplane-scratch', baseline: true,
    cmd: `"${EXE}" old full ${S} ${REPS}` },
  { name: 'new-tile-scratch',
    cmd: `"${EXE}" new full ${S} ${REPS}` },
];
