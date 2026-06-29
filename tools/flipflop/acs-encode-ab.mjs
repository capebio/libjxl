// flipflop cmd-mode harness: full cjxl encode, OLD (submodule main) vs NEW
// (perf/enc-acs-scratch-prune) — measures the wall-time effect of the
// enc_ac_strategy changes (64x64 scratch envelope, rate-before-loss, incumbent
// pruning in the merge/square paths, phantom-merge-entry removal).
//
// The changes are byte-exact: the encoded .jxl is SHA-256-identical between OLD
// and NEW (verified separately by tools/flipflop/acs_ab.mjs PART 1). This
// harness only times them. Effort 7 (squirrel) runs the full ACS search
// including the non-aligned merge passes; swap to e5/e9 to probe other tiers.
//
// Build recipe (static A/B is simpler, but the snapshots below are shared-lib
// builds with their DLLs co-located, which also works):
//   1. cmake -G Ninja -S <submodule> -B C:/Tmp/acs-build-ab -DCMAKE_BUILD_TYPE=Release \
//        -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DBUILD_TESTING=OFF \
//        -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_EXAMPLES=OFF (under vcvars64)
//   2. ninja -C C:/Tmp/acs-build-ab cjxl   (NEW)  -> snapshot exe+dlls to new/
//   3. git checkout main -- lib/jxl/enc_ac_strategy.{cc,h}; ninja ... cjxl (OLD)
//        -> snapshot to old/; then restore branch files.
//
// Run from the superproject root:
//   node --expose-gc flipflop.mjs \
//     C:/Foo/rcw-enc-acs/tools/flipflop/acs-encode-ab.mjs --print \
//     --journal "C:/Foo/rcw-enc-acs/tools/flipflop/acs-encode-journal.toon"

const OLD = "C:/Tmp/acs-test/old/cjxl.exe";
const NEW = "C:/Tmp/acs-test/new/cjxl.exe";
const DIR = "C:/Tmp/acs-test";
const EFFORT = 7;
const DIST = 1.0;

export const name = "acs-encode-prune";
export const description =
  `cjxl full encode e${EFFORT} d${DIST}: enc_ac_strategy OLD vs scratch+prune NEW (byte-exact)`;

export const corpus = () =>
  ["torture", "hdr_room", "splines"].map((n) => ({
    name: n,
    kind: "file",
    path: `${DIR}/${n}.ppm`,
  }));

export const variants = [
  {
    name: "old-main",
    baseline: true,
    cmd: `"${OLD}" {input} {output} -e ${EFFORT} -d ${DIST} --num_threads=1 --quiet`,
  },
  {
    name: "new-scratch-prune",
    cmd: `"${NEW}" {input} {output} -e ${EFFORT} -d ${DIST} --num_threads=1 --quiet`,
  },
];

// cmd-mode does not capture output; byte-exactness is asserted by acs_ab.mjs.
export const equal = () => true;
