// Isolate the nb_clusters cap raise: BASE = ==21-fix only, NEW = +cap raise.
// big2048 exercises the cap (wants 32 clusters). Decode must stay identical.
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
const [BASE, NEW] = process.argv.slice(2);
const DIR = "C:\\Tmp\\acs-test";
const DJXL = `${BASE}\\djxl.exe`;
const run=(e,a)=>{const t0=process.hrtime.bigint();const r=spawnSync(e,a,{encoding:"buffer"});return{ms:Number(process.hrtime.bigint()-t0)/1e6,status:r.status};};
const enc=(d,img,e,o)=>run(`${d}\\cjxl.exe`,[`${DIR}\\${img}.ppm`,o,"-e",String(e),"-d","1.0","--num_threads=1","--quiet"]);
const sha=p=>createHash("sha256").update(readFileSync(p)).digest("hex");
const decSha=(j,p)=>{const r=run(DJXL,[j,p,"--quiet"]);return r.status===0?sha(p):"DECFAIL";};
const median=a=>{const s=[...a].sort((x,y)=>x-y);return s[(s.length-1)>>1];};
// (image, efforts)
const cases=[["big2048",[5,7]],["torture",[7,9]],["hdr_room",[7]],["splines",[7]],["ellipses",[7]]];
console.log("== cap raise: size + bit-identical decode ==");
let tB=0,tN=0,fail=0;
for(const[img,effs]of cases)for(const e of effs){
  const b=`${DIR}\\_cb.jxl`,n=`${DIR}\\_cn.jxl`;
  if(enc(BASE,img,e,b).status||enc(NEW,img,e,n).status){console.log(`ENCFAIL ${img} e${e}`);continue;}
  const sb=readFileSync(b).length,sn=readFileSync(n).length;
  const id=decSha(b,`${DIR}\\_cbd.ppm`)===decSha(n,`${DIR}\\_cnd.ppm`);
  if(!id)fail++; tB+=sb;tN+=sn;
  const d=sb-sn;
  console.log(`  ${img.padEnd(12)} e${e}  ${sb}->${sn}B  ${d>=0?"-":"+"}${Math.abs(d)}B (${(d/sb*100>=0?"-":"+")}${Math.abs(d/sb*100).toFixed(3)}%)  decode:${id?"IDENTICAL":"*** DIFFERS ***"}`);
}
console.log(`\nsize total ${tB}->${tN}B (saved ${tB-tN}B, ${((tB-tN)/tB*100).toFixed(3)}%); decode ${fail?("*** "+fail+" DIFFER ***"):"ALL IDENTICAL"}`);
console.log("\n== encode time (interleaved median of 5) ==");
for(const[img,effs]of [["big2048",[5,7]],["torture",[7]],["splines",[7]]])for(const e of effs){
  const tb=[],tn=[];
  for(let r=0;r<5;r++){const o=r%2===0?[[BASE,tb],[NEW,tn]]:[[NEW,tn],[BASE,tb]];for(const[d,a]of o){const x=enc(d,img,e,`${DIR}\\_ct.jxl`);if(x.status===0)a.push(x.ms);}}
  const mb=median(tb),mn=median(tn),dt=(mn-mb)/mb*100;
  console.log(`  ${img.padEnd(12)} e${e}  base ${mb.toFixed(0)}ms cap ${mn.toFixed(0)}ms (${dt>=0?"+":""}${dt.toFixed(1)}%)`);
}
process.exit(fail?1:0);
