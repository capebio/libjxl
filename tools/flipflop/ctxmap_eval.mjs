// Evaluate the block-context-map default-detection fix (and any custom-map
// experiment). The encoded .jxl bytes CHANGE (smaller signaling), but the
// decoded pixels MUST be bit-identical — that is the correctness gate (zero
// reconstruction change => zero quality loss). Reports size delta + a hard
// decode-identical check + interleaved encode-time delta.
//
// Usage: node ctxmap_eval.mjs <baseline_dir> <fix_dir>
//   each dir: cjxl.exe + dlls; baseline_dir also has djxl.exe.
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";

const [BASE, FIX] = process.argv.slice(2);
const DIR = "C:\\Tmp\\acs-test";
const DJXL = `${BASE}\\djxl.exe`;
const IMAGES = ["torture", "hdr_room", "splines", "grayscale_patches", "ellipses"];
const EFFORTS = [5, 7, 9];
const DIST = 1.0;

const run = (exe, args) => {
  const t0 = process.hrtime.bigint();
  const r = spawnSync(exe, args, { encoding: "buffer" });
  const t1 = process.hrtime.bigint();
  return { ms: Number(t1 - t0) / 1e6, status: r.status,
           err: (r.stderr||Buffer.alloc(0)).toString().slice(-300) };
};
const enc = (dir, img, e, out) =>
  run(`${dir}\\cjxl.exe`, [`${DIR}\\${img}.ppm`, out, "-e", String(e),
       "-d", String(DIST), "--num_threads=1", "--quiet"]);
const sha = p => createHash("sha256").update(readFileSync(p)).digest("hex");
function decSha(jxl, ppm) { const r = run(DJXL, [jxl, ppm, "--quiet"]);
  return r.status === 0 ? sha(ppm) : `DECFAIL:${r.err}`; }
const median = a => { const s=[...a].sort((x,y)=>x-y); return s[(s.length-1)>>1]; };

console.log("== size + bit-identical-decode ==");
let totB=0, totF=0, decFail=0, sizeWin=0, sizeLoss=0;
const rows=[];
for (const img of IMAGES) {
  for (const e of EFFORTS) {
    const b=`${DIR}\\_b.jxl`, f=`${DIR}\\_f.jxl`;
    const rb=enc(BASE,img,e,b), rf=enc(FIX,img,e,f);
    if (rb.status||rf.status){ console.log(`  ENCFAIL ${img} e${e} ${rb.err}${rf.err}`); continue; }
    const sb=readFileSync(b).length, sf=readFileSync(f).length;
    const dB=decSha(b,`${DIR}\\_bd.ppm`), dF=decSha(f,`${DIR}\\_fd.ppm`);
    const identical = dB===dF && !dB.startsWith("DECFAIL");
    if(!identical) decFail++;
    if(sf<sb) sizeWin++; else if(sf>sb) sizeLoss++;
    totB+=sb; totF+=sf;
    const d=sb-sf, dpct=d/sb*100;
    rows.push({img,e,sb,sf,d,dpct,identical});
    console.log(`  ${img.padEnd(20)} e${e}  ${sb}->${sf}B  ${d>=0?"-":"+"}${Math.abs(d)}B (${dpct>=0?"-":"+"}${Math.abs(dpct).toFixed(3)}%)  decode:${identical?"IDENTICAL":"*** DIFFERS ***"}`);
  }
}
console.log(`\nsize: total ${totB}->${totF}B (saved ${totB-totF}B, ${((totB-totF)/totB*100).toFixed(3)}%); wins ${sizeWin} losses ${sizeLoss}`);
console.log(`decode-identical: ${decFail===0?"ALL OK":"*** "+decFail+" DIFFER ***"}`);

console.log("\n== encode time (interleaved, median of 5) ==");
for (const img of ["torture","hdr_room","splines"]) {
  for (const e of EFFORTS) {
    const tb=[],tf=[];
    for(let r=0;r<5;r++){ const order=r%2===0?[[BASE,tb],[FIX,tf]]:[[FIX,tf],[BASE,tb]];
      for(const[d,acc]of order){const res=enc(d,img,e,`${DIR}\\_t.jxl`); if(res.status===0)acc.push(res.ms);} }
    const mb=median(tb),mf=median(tf),dt=(mf-mb)/mb*100;
    console.log(`  ${img.padEnd(20)} e${e}  base ${mb.toFixed(0)}ms fix ${mf.toFixed(0)}ms (${dt>=0?"+":""}${dt.toFixed(1)}%)`);
  }
}
writeFileSync(`${DIR}\\ctxmap_eval.json`, JSON.stringify(rows,null,2));
console.log("\nwrote ctxmap_eval.json");
process.exit(decFail?1:0);
