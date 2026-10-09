const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('pascal.html','utf8');
const src=html.split('id="app-source">')[1].split('</script>')[0];
const babel=require('C:/Users/DNMod/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/lib/transform/babelBundle.js');
babel.babelParse(src,'pascal.jsx');
console.log('Full application JSX syntax: passed');
const pure=src.slice(0,src.indexOf('const ResidueStrip ='));
const registry=src.slice(src.indexOf('const MAP_MODES ='),src.indexOf('// Return increment'));
vm.runInNewContext(pure+'\n'+registry+`
for(const m of [2,3,8,30,65537,99991]) {
 const t=buildTriangle(35,m);
 for(let n=0;n<35;n++) {let b=1n;for(let k=0;k<=n;k++) {
  assert.equal(t[n*(n+1)/2+k],Number(b%BigInt(m)));
  b=b*BigInt(n-k)/BigInt(k+1);
 }}
}
const a=new Uint16Array([1,1,1,1,8,1]);
const b=new Uint16Array([1,1,1,1,0,1]);
const delta=mapFieldRaw(b,3,2,'temporal',{},null,{residues:a});
assert.equal(delta.field[4],8);
const hist={size:6,frames:[b]};
const params={magWindow:4,magAmp:1,magSource:'velocity',magDisplay:'pure'};
const low=mapFieldRaw(a,3,20,'magnify',params,hist);
const high=mapFieldRaw(a,3,20,'magnify',{...params,magAmp:2},hist);
assert(low.field[4]>0); assert(high.field[4]>low.field[4]);assert.equal(high.max,1);
for(const mode of ['math']) {
 const f=buildCombinedField(buildTriangle(20,7),20,7,['value','bandpass'],'average',{bandLo:6,bandHi:6});
 assert(f.field.every(Number.isFinite));
 assert(f.field.some(x=>Math.abs(x-.5)>.01));
}
`,{React:{},assert});
console.log('Exact binomials, reverse delta, current-frame magnification, gain and equal-radius bandpass: passed');
if(process.argv.includes('--browser')) (async()=>{
 const {chromium}=require('C:/Users/DNMod/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
 const browser=await chromium.launch({channel:'msedge',headless:true});
 try {
 const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('file:///'+process.cwd().replace(/\\/g,'/')+'/pascal.html');
 await page.getByText('MODULUS STATUS',{exact:true}).waitFor({timeout:60000});
 await page.getByRole('button',{name:/CARRY TERRACES/}).click();
 await page.getByText('MOD 8: COMPOSITE',{exact:true}).waitFor();
 await page.getByRole('button',{name:/FACTOR OVERLAP/}).click();
 await page.getByText('MOD 30: COMPOSITE',{exact:true}).waitFor();
 await page.getByText('More experiments',{exact:true}).click();
 assert.equal(errors.length,0,errors.join('\n'));
 console.log('Fresh browser boot, new presets and experiment disclosure: passed');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
