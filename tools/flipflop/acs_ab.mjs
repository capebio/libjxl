// AC-strategy encoder A/B harness.
//   PART 1 (correctness): encode each corpus image with OLD and NEW cjxl at
//     several efforts and thread counts, compare SHA-256 of the .jxl output.
//     These four changes are meant to be byte-exact, so every case must match.
//   PART 2 (flipflop timing): interleaved A/B with start-rotation to cancel
//     thermal/scheduler drift; reports median encode wall-time OLD vs NEW.
//
// Usage: node acs_ab.mjs <cjxl_old.exe> <cjxl_new.exe>
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, existsSync } from "node:fs";

const [OLD, NEW] = process.argv.slice(2);
if (!OLD || !NEW) { console.error("need OLD NEW cjxl paths"); process.exit(2); }
const DIR = "C:\\Tmp\\acs-test";
const IMAGES = ["torture.ppm", "hdr_room.ppm", "splines.ppm",
                "grayscale_patches.ppm", "ellipses.ppm"];
const EFFORTS = [5, 7, 9];           // hare / squirrel / tortoise -> full ACS
const DIST = 1.0;

function enc(bin, img, e, threads, out) {
  const inP = `${DIR}\\${img}`;
  const args = [inP, out, "-e", String(e), "-d", String(DIST),
                `--num_threads=${threads}`, "--quiet"];
  const t0 = process.hrtime.bigint();
  const r = spawnSync(bin, args, { encoding: "buffer" });
  const t1 = process.hrtime.bigint();
  if (r.status !== 0) {
    return { ok: false, err: (r.stderr || Buffer.alloc(0)).toString().slice(-400) };
  }
  return { ok: true, ms: Number(t1 - t0) / 1e6 };
}
function sha(p) { return createHash("sha256").update(readFileSync(p)).digest("hex"); }

// ---------- PART 1: byte-exact correctness ----------
console.log("== PART 1: byte-exact A/B (SHA-256 of encoded .jxl) ==");
let pass = 0, fail = 0;
const failures = [];
for (const threads of [1, 0]) {            // 1 = single, 0 = all cores
  for (const img of IMAGES) {
    for (const e of EFFORTS) {
      const o = `${DIR}\\_old.jxl`, n = `${DIR}\\_new.jxl`;
      const ro = enc(OLD, img, e, threads, o);
      const rn = enc(NEW, img, e, threads, n);
      if (!ro.ok || !rn.ok) {
        fail++; failures.push(`${img} e${e} t${threads}: ENC FAIL ${ro.err||""}${rn.err||""}`);
        continue;
      }
      const so = sha(o), sn = sha(n);
      const szo = readFileSync(o).length, szn = readFileSync(n).length;
      if (so === sn) { pass++; console.log(`  OK   ${img.padEnd(22)} e${e} t${threads}  ${szo}B  ${so.slice(0,12)}`); }
      else { fail++; console.log(`  FAIL ${img.padEnd(22)} e${e} t${threads}  old=${szo}B(${so.slice(0,12)}) new=${szn}B(${sn.slice(0,12)})`);
             failures.push(`${img} e${e} t${threads}: SHA MISMATCH old=${so.slice(0,16)} new=${sn.slice(0,16)}`); }
    }
  }
}
console.log(`\nPART 1 result: ${pass} pass, ${fail} fail`);
if (fail) { console.log("FAILURES:\n" + failures.join("\n")); }

// ---------- PART 2: flipflop timing ----------
console.log("\n== PART 2: flipflop timing (interleaved, start-rotated) ==");
const REPS = 7;
const median = a => { const s=[...a].sort((x,y)=>x-y); return s[(s.length-1)>>1]; };
const rows = [];
for (const img of ["torture.ppm", "hdr_room.ppm", "splines.ppm"]) {
  for (const e of EFFORTS) {
    const tOld = [], tNew = [];
    for (let r = 0; r < REPS; r++) {
      // rotate which variant runs first each rep
      const order = (r % 2 === 0) ? [["OLD",OLD,tOld],["NEW",NEW,tNew]]
                                  : [["NEW",NEW,tNew],["OLD",OLD,tOld]];
      for (const [, bin, acc] of order) {
        const res = enc(bin, img, e, 1, `${DIR}\\_t.jxl`);
        if (res.ok) acc.push(res.ms);
      }
    }
    const mo = median(tOld), mn = median(tNew);
    const delta = (mo - mn) / mo * 100;
    rows.push({ img, e, mo, mn, delta });
    console.log(`  ${img.padEnd(22)} e${e}  OLD ${mo.toFixed(1)}ms  NEW ${mn.toFixed(1)}ms  ` +
                `${delta>=0?"-":"+"}${Math.abs(delta).toFixed(1)}% ${delta>=0?"faster":"slower"}`);
  }
}
writeFileSync(`${DIR}\\acs_ab_result.json`,
  JSON.stringify({ pass, fail, failures, timing: rows }, null, 2));
console.log("\nwrote acs_ab_result.json");
process.exit(fail ? 1 : 0);
