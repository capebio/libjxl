// Measure DECODE speed of the decoding_speed_tier>=1 fix.
// buggy cjxl keeps the 15-ctx default map; fixed cjxl installs the 2-ctx simple
// map. Both encode at --faster_decoding=1; decoded pixels are bit-identical.
// Fewer block contexts => less entropy-decode work => should decode faster.
// We time djxl on each produced .jxl (interleaved, start-rotated, median).
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
const BUG = "C:\\Tmp\\acs-test\\ctxfix";        // buggy: discards simple map
const FIX = "C:\\Tmp\\acs-test\\ctxsimplefix";  // fixed: uses 2-ctx simple map
const DJXL = `${BUG}\\djxl.exe`;                 // decoder is the same for both
const DIR = "C:\\Tmp\\acs-test";
const run = (e,a) => { const t0=process.hrtime.bigint(); const r=spawnSync(e,a,{encoding:"buffer"});
  return { ms:Number(process.hrtime.bigint()-t0)/1e6, status:r.status }; };
const enc = (dir,img,out) => run(`${dir}\\cjxl.exe`,[`${DIR}\\${img}.ppm`,out,"-e","7","-d","1.0",
  "--faster_decoding=1","--num_threads=1","--quiet"]);
const dec = (jxl,ppm) => run(DJXL,[jxl,ppm,"--num_threads=1","--quiet"]);
const sha = p => createHash("sha256").update(readFileSync(p)).digest("hex");
const median = a => { const s=[...a].sort((x,y)=>x-y); return s[(s.length-1)>>1]; };
const REPS = 17;
console.log("== decode speed @ --faster_decoding=1 (buggy 15-ctx vs fixed 2-ctx) ==");
for (const img of ["big2048","torture","splines","hdr_room"]) {
  const b=`${DIR}\\_db.jxl`, f=`${DIR}\\_df.jxl`;
  enc(BUG,img,b); enc(FIX,img,f);
  const sb=readFileSync(b).length, sf=readFileSync(f).length;
  // confirm decode identical
  dec(b,`${DIR}\\_dbp.ppm`); dec(f,`${DIR}\\_dfp.ppm`);
  const ident = sha(`${DIR}\\_dbp.ppm`)===sha(`${DIR}\\_dfp.ppm`);
  const tb=[], tf=[];
  for (let r=0;r<REPS;r++){
    const order = r%2===0 ? [[b,tb],[f,tf]] : [[f,tf],[b,tb]];
    for (const [jxl,acc] of order){ const x=dec(jxl,`${DIR}\\_dt.ppm`); if(x.status===0) acc.push(x.ms); }
  }
  const mb=median(tb), mf=median(tf), dpct=(mf-mb)/mb*100;
  console.log(`  ${img.padEnd(10)} size ${sb}->${sf}B  decode buggy ${mb.toFixed(1)}ms  fixed ${mf.toFixed(1)}ms  ` +
    `${dpct<=0?"-":"+"}${Math.abs(dpct).toFixed(1)}% ${dpct<=0?"FASTER":"slower"}  decode:${ident?"identical":"*DIFFER*"}`);
}
