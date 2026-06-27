// flipflop cmd-mode harness: GaborishInverse OLD vs NEW with REALISTIC
// streaming geometry — plane is one DC-group patch (group + 8px kBlockDim
// border ring), rect is the group itself. This is what EncodeFrameStreaming
// actually passes (enc_frame.cc:1553 max_border=kBlockDim), so the rect is
// ~99% of the plane and only the border ring differs between OLD and NEW.
//
// Expected ~neutral: the tile-scratch only saves the 8px border ring here,
// unlike the synthetic quarter-plane `partial` harness.
//
//   node --expose-gc C:/Foo/raw-converter-wasm/flipflop.mjs \
//     C:/Foo/rcw-gaborish/tools/flipflop/gaborish-stream.mjs \
//     --sizes 512 --types mandel --rounds 512:11 --print \
//     --journal "C:/Foo/rcw-gaborish/tools/flipflop/gaborish-journal.toon"

const EXE = 'C:/Foo/rcw-gaborish/build-bench/gaborish_bench.exe';
const S = 2064;   // 2048 DC group + 2*8px border
const REPS = 22;

export const name = 'gaborish-stream-geom';
export const description =
  'GaborishInverse streaming geometry (group+8px border): full-plane vs tile scratch';

export const variants = [
  { name: 'old-fullplane-scratch', baseline: true,
    cmd: `"${EXE}" old stream ${S} ${REPS}` },
  { name: 'new-tile-scratch',
    cmd: `"${EXE}" new stream ${S} ${REPS}` },
];
