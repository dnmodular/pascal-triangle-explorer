// Checks pascal-core.html's math against exact BigInt binomials, its Grassmannian lens against
// an independent q-Pascal recurrence, and its motion detector against synthetic motion and noise.
// Run: node verify-pascal-core.cjs            (from this folder)
//      node verify-pascal-core.cjs out.zip    (also writes a sample ZIP to inspect)
const fs = require('fs'), vm = require('vm'), assert = require('assert'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, 'pascal-core.html'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
new Function(script); // whole app parses
const core = script.split('// <core>')[1].split('// </core>')[0].replace(/^.*\n/, ''); // drop the marker line's comment
const ctx = { assert, TextEncoder, console };
const names = 'stripeField stripePatches matchPatches driftSums driftFrom walkField baseCarryField buildTriangle buildScaled scaledBig scaledTarget scaledEq scaledTheta scaledZeros residueLevel residueLevels kummerField crtField carryField lucasField phaseField digitField qField tablesFor factorize factorP sequenceList parseIsolation maskFor paintFlat paintRelief reliefGeom zipStore crc32 censusField newCensus binomBig carries BG rational stepMu newMotion motionAlloc motionStep motionVelocity blurWin pow10 LNF';
vm.runInNewContext(core + `\n;globalThis.__ = { ${names.split(' ').join(', ')} };`, ctx);
const C = ctx.__;

const binomRows = (N) => { const rows = [[1n]]; for (let n = 1; n < N; n++) { const p = rows[n - 1], r = [1n]; for (let k = 1; k < n; k++) r.push(p[k - 1] + p[k]); r.push(1n); rows.push(r); } return rows; };
const B = binomRows(160);
const nu = (b, p) => { let e = 0; const P = BigInt(p); while (b % P === 0n) { b /= P; e++; } return e; };
let checks = 0;

// 1. whole-modulus residues
for (const m of [2, 3, 4, 8, 30, 97, 360, 65537, 99991]) {
  const N = 90, t = C.buildTriangle(N, m);
  for (let n = 0; n < N; n++) for (let k = 0; k <= n; k++, checks++) assert.equal(t[n * (n + 1) / 2 + k], Number(B[n][k] % BigInt(m)), `C(${n},${k}) mod ${m}`);
}
// 2. fractional moduli at any depth: S = Q·C mod P in all three storage tiers, and
//    C mod μ = S/Q = C − μ·floor(C/μ) exactly; zeros exactly where P | C
const tiers = {};
for (const [units, d] of [[101n, 1], [1025n, 2], [105n, 1], [99999n, 3], [999999999n, 4], [5000000001n, 9], [10n ** 16n + 1n, 15],
  [10n ** 41n + 1n, 40], [10n * 10n ** 120n + 37n, 120], [99999n * 10n ** 200n + 123456789n, 200]]) {
  const { P, Q } = C.rational(units, d), N = 50, sc = C.buildScaled(N, P, Q);
  tiers[sc.kind] = (tiers[sc.kind] || 0) + 1;
  assert.equal(P * C.pow10(d), units * Q, 'P/Q equals units/10^d');
  const zeros = C.scaledZeros(sc, 0, N * (N + 1) / 2), th = C.scaledTheta(sc, 0, N * (N + 1) / 2);
  for (let n = 0; n < N; n++) for (let k = 0; k <= n; k++, checks++) {
    const i = n * (n + 1) / 2 + k, c = B[n][k], QC = Q * c, want = QC - P * (QC / P);
    assert.equal(C.scaledBig(sc, i), want, `C(${n},${k}) mod ${units}/10^${d} [${sc.kind}]`);
    assert.equal(zeros[i] === 1, c % P === 0n, 'μ divides C exactly when the numerator does');
    const exact = Number((want << 64n) / P) / 2 ** 64;
    assert(Math.abs(th[i] - exact) < 1e-12, `phase ${th[i]} vs ${exact}`);
  }
  if (d === 40) { // exact comparisons in the multi-word tier
    const one = C.scaledTarget(sc, Q);
    assert.equal(C.scaledEq(sc, 0, one), true, 'C(0,0)=1 -> S = Q');
    const mask = C.maskFor(sc, 0, N * (N + 1) / 2, new Set([1]));
    for (let n = 0; n < N; n++) for (let k = 0; k <= n; k++, checks++) assert.equal(mask[n * (n + 1) / 2 + k], B[n][k] * Q % P === Q ? 1 : 0);
  }
}
assert(tiers.int && tiers.f64 && tiers.limbs, 'all three storage tiers exercised: ' + JSON.stringify(tiers));
{ // residue levels at whole moduli reproduce the familiar coloring order
  const lv = (v, m, s) => C.residueLevel(v, m, 1, s);
  assert.equal(lv(0, 7, true), 0); assert.equal(lv(1, 7, true), 1); assert.equal(lv(2, 7, true), 2); assert.equal(lv(6, 7, true), 255);
  assert.equal(lv(6, 7, false), 255); assert.equal(lv(1, 2, true), 1);
  const { P, Q } = C.rational(10n ** 41n + 1n, 40), sc = C.buildScaled(30, P, Q), { idx } = C.residueLevels(sc, 0, 465, 10, true);
  assert.equal(idx[0], 1, 'C(0,0)=1 is white at 40 decimals');
}
// 3. Kummer depth = p-adic valuation, and the cyclotomic form of it:
//    ν_p(C(n,k)) = Σ_j e_{p^j},  e_d = ⌊n/d⌋ − ⌊k/d⌋ − ⌊(n−k)/d⌋  (multiplicity of Φ_d in [n k]_q)
for (const p of [2, 3, 5, 7, 11]) {
  const N = 120, f = C.kummerField(N, p);
  for (let n = 0; n < N; n++) for (let k = 0; k <= n; k++, checks++) {
    const v = nu(B[n][k], p);
    assert.equal(f[n * (n + 1) / 2 + k], v, `nu_${p} C(${n},${k})`);
    let e = 0; for (let d = p; d <= n; d *= p) e += Math.floor(n / d) - Math.floor(k / d) - Math.floor((n - k) / d);
    assert.equal(e, v, 'cyclotomic carries');
  }
}
// 4. CRT layers = divisibility by each prime power
for (const m of [12, 30, 72, 360, 2310]) {
  const fs_ = C.factorize(m), N = 100, f = C.crtField(N, fs_);
  for (let n = 0; n < N; n++) for (let k = 0; k <= n; k++) fs_.forEach(([p, a], b) => {
    checks++;
    assert.equal((f[n * (n + 1) / 2 + k] >> b) & 1, B[n][k] % BigInt(p ** a) === 0n ? 1 : 0, `${p}^${a} | C(${n},${k})`);
  });
}
// 5. Lucas: the product of every digit's factor is the residue mod p
for (const p of [2, 3, 5, 7, 13]) {
  const N = 150, t = C.buildTriangle(N, p), layers = [];
  for (let L = 0; p ** L < N; L++) layers.push(C.lucasField(N, p, L));
  for (let i = 0; i < t.length; i++, checks++) assert.equal(layers.reduce((acc, f) => acc * f[i] % p, 1), t[i], `lucas p=${p} cell ${i}`);
}
// 6. discrete log tables: g generates (Z/p)*, and the phase field is dark exactly on multiples of p
for (const p of [3, 5, 7, 11, 13, 101, 9973, 99991]) {
  const T = C.tablesFor(p);
  let x = 1;
  for (let j = 0; j < p - 1; j++, checks++) { assert.equal(T.ind[x], j); x = x * T.g % p; }
  assert.equal(x, 1, `g=${T.g} has order p-1 mod ${p}`);
}
{
  const t = C.buildTriangle(80, 7), f = C.phaseField(t, 7);
  for (let i = 0; i < t.length; i++, checks++) assert.equal(f[i] === 0, t[i] === 0);
}
// 7. first carry digit = smallest j with (k mod p^j) + ((n-k) mod p^j) >= p^j
for (const p of [2, 3, 5]) {
  const N = 140, f = C.carryField(N, p);
  for (let n = 0; n < N; n++) for (let k = 0; k <= n; k++, checks++) {
    let want = 0;
    for (let j = 1, pj = p; pj <= 2 * n + 1; j++, pj *= p) if ((k % pj) + ((n - k) % pj) >= pj) { want = j; break; }
    assert.equal(f[n * (n + 1) / 2 + k], want, `first carry base ${p} C(${n},${k})`);
    assert.equal(want > 0, B[n][k] % BigInt(p) === 0n);
  }
}
// 8. p-adic digit j of C(n,k) itself
for (const [p, J] of [[5, 0], [5, 1], [3, 2], [2, 4]]) {
  const N = 100, f = C.digitField(C.buildTriangle(N, p ** (J + 1)), p, J);
  for (let n = 0; n < N; n++) for (let k = 0; k <= n; k++, checks++) assert.equal(f[n * (n + 1) / 2 + k], Number((B[n][k] / BigInt(p ** J)) % BigInt(p)));
}
// 9. census counts each modulus once
{
  const cs = C.newCensus(), N = 40, total = N * (N + 1) / 2;
  for (const m of [2, 3, 2, 4]) C.censusField(cs, C.scaledZeros(C.buildScaled(N, BigInt(m), 1n), 0, total), N, String(m));
  assert.equal(cs.count, 3);
  for (let n = 0; n < N; n++) for (let k = 0; k <= n; k++, checks++) {
    const want = [2, 3, 4].filter((m) => B[n][k] % BigInt(m) === 0n).length;
    assert.equal(cs.sum[n * (n + 1) / 2 + k], want);
  }
}
// 10. factoring numerators: the product is exact and the listed factors are prime
for (const P of [1001n, 2n ** 10n * 3n ** 4n * 99991n, 10n ** 41n + 1n, 99999n * 10n ** 200n + 123456789n, 1000000007n * 998244353n]) {
  const f = C.factorP(P);
  let prod = f.rest;
  for (const [p, a] of f.factors) { prod *= BigInt(p) ** BigInt(a); assert.equal(C.factorize(p).length === 1 && C.factorize(p)[0][1] === 1, true, `${p} prime`); }
  assert.equal(prod, P);
  checks++;
}
assert.deepEqual(C.factorP(1001n).factors, [[7, 1], [11, 1], [13, 1]]);
// 11. Grassmannian lens: [n k]_q at q = e^{2πiQ/P} against the q-Pascal recurrence
//     [n k] = [n−1 k−1] + q^k [n−1 k], and its zeros against q-Lucas (k mod P > n mod P)
let qMax = 0;
for (const [units, d] of [[6n, 0], [7n, 0], [10n, 0], [1025n, 2], [25n, 1], [73n, 1], [10n ** 31n + 1n, 30], [100003n, 4]]) {
  const { P, Q } = C.rational(units, d), N = 64, total = N * (N + 1) / 2, q = C.qField(N, 0, P, Q, true);
  const ang = 2 * Math.PI * Number(Q % P) / Number(P), Pn = Number(P);
  const re = new Float64Array(total), im = new Float64Array(total);
  for (let n = 0; n < N; n++) {
    const row = n * (n + 1) / 2, prev = row - n;
    re[row] = 1; re[row + n] = 1; im[row] = 0; im[row + n] = 0;
    for (let k = 1; k < n; k++) {
      const cr = Math.cos(ang * k), ci = Math.sin(ang * k), ar = re[prev + k], ai = im[prev + k];
      re[row + k] = re[prev + k - 1] + cr * ar - ci * ai;
      im[row + k] = im[prev + k - 1] + cr * ai + ci * ar;
    }
  }
  for (let n = 0; n < N; n++) for (let k = 0; k <= n; k++, checks++) {
    const i = n * (n + 1) / 2 + k, cnk = Number(B[n][k]), mag = Math.hypot(re[i], im[i]);
    const qLucasZero = Pn <= n && (k % Pn) > (n % Pn);
    assert.equal(q.lnAbs[i] === -Infinity, qLucasZero, `q-Lucas zero at μ=${units}/10^${d}, (${n},${k})`);
    if (qLucasZero) { assert(mag < 1e-6 * cnk + 1e-9, 'direct recurrence is ~0 there too'); continue; }
    // the double-precision reference is only good to ~1e-15·C(n,k) absolute (it misses e.g. the true
    // |[10 1]_q| ≈ 1e-30 at μ = 10 + 1e-30, which the lens gets right), so scale the tolerance to it
    const r = Math.exp(q.lnAbs[i]), err = Math.abs(r - mag) / (Math.max(mag, r) + 1e-10 * cnk);
    qMax = Math.max(qMax, err);
    assert(err < 1e-6, `|[${n} ${k}]_q| ${r} vs ${mag} (μ=${units}/10^${d})`);
    if (mag > 1e-6 * cnk) { // phase, with the q^{k(n−k)/2} rotation included
      const ph = Math.atan2(im[i], re[i]) / (2 * Math.PI), dph = Math.abs(((q.theta[i] - ph) % 1 + 1.5) % 1 - 0.5);
      assert(dph < 1e-6, `phase (${n},${k}) ${q.theta[i]} vs ${ph}`);
    }
    if (Pn <= n) { // q-Lucas magnitude: C(⌊n/P⌋, ⌊k/P⌋) · |[n mod P, k mod P]_q|
      const j0 = (n % Pn) * ((n % Pn) + 1) / 2 + (k % Pn), lucas = Number(B[Math.floor(n / Pn)][Math.floor(k / Pn)]) * Math.exp(q.lnAbs[j0]);
      assert(Math.abs(lucas - r) <= 1e-9 * r + 1e-12, 'q-Lucas magnitude');
    }
  }
}
// at q → 1 (huge μ) the lens returns the ordinary binomials
{
  const q = C.qField(40, 0, 10n ** 12n, 1n, false);
  for (let n = 0; n < 40; n++) for (let k = 0; k <= n; k++, checks++) {
    const i = n * (n + 1) / 2 + k;
    assert(Math.abs(Math.exp(q.lnAbs[i]) / Number(B[n][k]) - 1) < 1e-6, 'q→1 gives C(n,k)');
  }
}
// zone-plate fringes: ½(1 + cos 2π·k(n−k)/2μ), dark on the q-Lucas zeros
{
  const { P, Q } = C.rational(305n, 1), N = 60, f = C.qField(N, 0, P, Q, 'fringe'), mu = 30.5;
  for (let n = 0; n < N; n++) for (let k = 0; k <= n; k++, checks++) {
    const i = n * (n + 1) / 2 + k, ph = k * (n - k) / (2 * mu), want = Math.round(127.5 * (1 + Math.cos(2 * Math.PI * (ph - Math.floor(ph)))));
    assert(Math.abs(f.idx[i] - want) <= 1, `fringe (${n},${k})`);
  }
}
// 12a. quantum walk: unitary rows, mirror symmetry, fronts at ±cos θ (Konno's limit theorem)
let walkFront = '';
for (const th of [Math.PI / 2, Math.PI / 3, Math.PI / 4, 0.3, Math.PI / 7]) {
  const N = 400, w = C.walkField(N, 0, th);
  for (let n = 0; n < N; n++) {
    let s = 0; for (let k = 0; k <= n; k++) s += w.prob[n * (n + 1) / 2 + k];
    assert(Math.abs(s - 1) < 1e-10, `row ${n} carries probability 1 (θ=${th}): ${s}`);
    checks++;
  }
  const n = N - 1, row = n * (n + 1) / 2, edge = n * Math.cos(th);
  let outside = 0, best = 0, kBest = 0;
  for (let k = 0; k <= n; k++) {
    const p = w.prob[row + k], x = Math.abs(2 * k - n);
    if (x > 1.08 * edge + 2) outside += p;
    if (p > best) { best = p; kBest = k; }
    if (Math.abs(th - Math.PI / 4) < 1e-12) assert(Math.abs(p - w.prob[row + n - k]) < 1e-12, 'Hadamard walk is mirror-symmetric');
  }
  assert(outside < 0.01, `probability outside the cone ±cos θ: ${outside} (θ=${th})`);
  if (th < Math.PI / 2) {
    const xb = Math.abs(2 * kBest - n) / edge;
    assert(xb > 0.85 && xb <= 1.05, `brightest front at ${xb.toFixed(3)}·cos θ (θ=${th})`);
    walkFront += `${(th * 180 / Math.PI).toFixed(0)}°: front at ${xb.toFixed(3)}·n·cos θ; `;
  }
}
// 12b. carries in base μ: Kummer at primes, digit-addition carries at any whole base
const digitCarries = (n, k, b) => { let a = k, c = n - k, carry = 0, cnt = 0; while (a > 0 || c > 0 || carry) { const s = (a % b) + (c % b) + carry; carry = s >= b ? 1 : 0; cnt += carry; a = Math.floor(a / b); c = Math.floor(c / b); } return cnt; };
for (const b of [2, 3, 5, 6, 7, 10, 12]) {
  const N = 150, f = C.baseCarryField(N, b).raw, kum = [2, 3, 5, 7].includes(b) ? C.kummerField(N, b) : null;
  for (let n = 0; n < N; n++) for (let k = 0; k <= n; k++, checks++) {
    const i = n * (n + 1) / 2 + k;
    assert.equal(f[i], digitCarries(n, k, b), `carries in base ${b} at (${n},${k})`);
    if (kum) assert.equal(f[i], kum[i]);
  }
}
{ // between simple fractions the field changes a little at a time; at a simple fraction like
  // 5/2 every multiple of 5 sits exactly on a tile edge, so stepping off it flips them together
  const cmp = (x, y) => { const a = C.baseCarryField(200, x).raw, b = C.baseCarryField(200, y).raw; let d = 0; for (let i = 0; i < a.length; i++) d += a[i] !== b[i] ? 1 : 0; return d / a.length; };
  const smooth = cmp(2.5137, 2.5138), snap = cmp(2.5, 2.5001);
  assert(smooth < 0.01 && snap > 0.2, `generic step changes ${smooth}, step off 5/2 changes ${snap}`);
}
// 12c. stripes: runs of similar phase, the doubling below them, and the drift tracker
let stripeInfo = '';
{
  const N = 128, tol = 0.04, close = (a, b) => { let d = Math.abs(a - b); d = Math.min(d, 1 - d); return d < tol; };
  const { P, Q } = C.rational(199n, 0), th = C.scaledTheta(C.buildScaled(N, P, Q), 0, N * (N + 1) / 2);
  const sf = C.stripeField(th, N, 0, tol, 3);
  let striped = 0, doubled = 0, children = 0;
  for (let r = 0, j = 0; r < N; r++) for (let c = 0; c <= r; c++, j++, checks++) {
    // brute-force run length through the cell
    let a = c, b = c; const row = r * (r + 1) / 2;
    while (a > 0 && close(th[row + a - 1], th[row + a])) a--;
    while (b < r && close(th[row + b], th[row + b + 1])) b++;
    assert.equal(sf.run[j], b - a + 1, `run through (${r},${c})`);
    assert.equal(sf.flag[j], b - a + 1 >= 3 ? 1 : 0);
    striped += sf.flag[j];
    // two similar parents put (about) twice their phase in the cell below: Pascal's rule
    if (r > 0 && c > 0 && c < r && close(th[j - r - 1], th[j - r])) {
      children++;
      const want = th[j - r - 1] + th[j - r]; let d = Math.abs(th[j] - (want - Math.floor(want)));
      d = Math.min(d, 1 - d);
      if (d < 1e-9) doubled++;
    }
  }
  assert.equal(doubled, children, 'every cell under two similar parents holds their sum');
  // tracker: a known shift is recovered; fresh noise and the real prime sweep show no direction
  const pf = (theta) => C.stripePatches(C.stripeField(theta, N, 0, tol, 3).flag, theta, N, 0, 3);
  const shifted = []; for (let t = 0; t < 30; t++) { const a = new Float64Array(th.length).fill(NaN); for (let r = 0, j = 0; r < N; r++) for (let c = 0; c <= r; c++, j++) if (c - t >= 0) a[j] = th[r * (r + 1) / 2 + c - t]; shifted.push(pf(a)); }
  const run = (frames) => { const s = []; for (let i = 1; i < frames.length; i++) s.push(C.driftSums(frames[i - 1], frames[i], 6)); return C.driftFrom(s); };
  const dShift = run(shifted), dNoise = run(Array.from({ length: 40 }, () => pf(Float64Array.from({ length: th.length }, Math.random))));
  const primes = [197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401];
  const dPrimes = run(primes.map((p) => pf(C.scaledTheta(C.buildScaled(N, BigInt(p), 1n), 0, th.length))));
  assert(Math.abs(dShift.vx - 1) < 0.05 && Math.abs(dShift.vy) < 0.05 && dShift.z > 20, 'known shift recovered: ' + JSON.stringify(dShift));
  assert(dNoise.z < 3, 'noise has no direction: z = ' + dNoise.z);
  checks += 3;
  stripeInfo = `stripes at mod 199: ${striped} cells; ${children} cells under similar parents, all exact sums; tracker: shift ${dShift.vx.toFixed(3)} cells/frame (z ${dShift.z.toFixed(0)}), noise z ${dNoise.z.toFixed(1)}, primes 197–401 z ${dPrimes.z.toFixed(1)}`;
}
// 12. sequences, fine stepping (exact, any depth), filters
const head = (id, n = 8) => Array.from(C.sequenceList(id).subarray(0, n)).join(',');
assert.equal(head('2p'), '4,6,10,14,22,26,34,38');
assert.equal(head('3p'), '6,9,15,21,33,39,51,57');
assert.equal(head('pq'), '6,10,14,15,21,22,26,33');
assert.equal(head('pqr'), '30,42,66,70,78,102,105,110');
assert.equal(head('pow2', 20), '2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536');
assert.equal(Array.from(C.sequenceList('step', 5, 7).subarray(0, 3)).join(','), '5,12,19');
{
  const walk = (id, d, from, steps, dir = 1, sub0 = 0n) => {
    const a = C.sequenceList(id), out = [];
    let pos = { m: from, sub: sub0 };
    for (let s = 0; s < steps && pos; s++) { const u = BigInt(pos.m) * C.pow10(d) + pos.sub; out.push(u.toString()); pos = C.stepMu(pos, dir, a, d); }
    return out.join(',');
  };
  assert.equal(walk('all', 1, 10, 12), '100,101,102,103,104,105,106,107,108,109,110,111');
  assert.equal(walk('prime', 0, 7, 4), '7,11,13,17');
  assert.equal(walk('all', 2, 11, 3, -1), '1100,1099,1098');
  const big = C.pow10(60);
  assert.equal(walk('all', 60, 10, 3, 1, big - 2n), [10n * big + big - 2n, 10n * big + big - 1n, 11n * big].join(','), '60 decimals cross a whole modulus exactly');
}
const iso = (s, m) => Array.from(C.parseIsolation(s, m) || []).sort((a, b) => a - b).join(',');
assert.equal(iso('1/2', 5), '2'); assert.equal(iso('1/2', 6), '2,3');
assert.equal(iso('n, n-1, 0', 10), '0,8,9'); assert.equal(iso('99', 10), ''); assert.equal(C.parseIsolation('', 10), null);
// 13. painters: flat brick geometry, relief writes
{
  const N = 3, pix = new Uint32Array(2 * N * N), col = Uint32Array.from([11, 12, 13, 14, 15, 16]);
  C.paintFlat(pix, N, 0, col);
  const show = (a) => Array.from(a).map((v) => v === C.BG ? '.' : v - 10).join('');
  assert.equal(show(pix.subarray(0, 6)), '..11..');
  assert.equal(show(pix.subarray(6, 12)), '.2233.');
  assert.equal(show(pix.subarray(12, 18)), '445566');
  const g = C.reliefGeom(N, 0, 1, 0), rp = new Uint32Array(g.W * g.H);
  C.paintRelief(rp, g, N, 0, col, new Float32Array(6));
  assert.deepEqual(Array.from(rp), Array.from(pix)); // zero height relief at unit 1 = the flat picture
}
// 14. motion detector: synthetic translations are found with the right direction; noise is not
{
  const N = 48, n = N * (N + 1) / 2, rnd = (() => { let s = 12345; return () => (s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32; })();
  const field = Array.from({ length: 200 }, () => Array.from({ length: 200 }, rnd)); // random pattern to move around
  const run = (thetaAt, mode = 'flow', frames = 30) => {
    const ms = C.newMotion(); C.motionAlloc(ms, 'k', n);
    for (let t = 0; t < frames; t++) {
      const th = new Float64Array(n);
      for (let r = 0, j = 0; r < N; r++) for (let c = 0; c <= r; c++, j++) th[j] = thetaAt(r, c, t);
      C.motionStep(ms, th, N, 0, { mode, tau: 6, trail: 0, pool: 0 });
    }
    let sx = 0, sy = 0, e = 0; // strength-weighted mean direction and mean strength
    for (let j = 0; j < n; j++) { const a = ms.H[j] / 256 * 2 * Math.PI; sx += ms.T[j] * Math.cos(a); sy += ms.T[j] * Math.sin(a); e += ms.T[j]; }
    return { strength: e / n, angle: Math.atan2(sy, sx) * 180 / Math.PI, coherence: Math.hypot(sx, sy) / (e || 1), ms };
  };
  const X = (r, c) => c - r / 2, Y = (r) => r * 0.8660254;               // cell centre, screen units
  const frac = (v) => v - Math.floor(v);
  const right = run((r, c, t) => field[r + 20][c - t + 100]);           // random pattern slides right one cell per frame
  const downRight = run((r, c, t) => field[r - t + 100][c - t + 100]);  // slides down-right (screen angle 60°)
  const slow = run((r, c, t) => frac(0.07 * (X(r, c) - 0.3 * t)));       // smooth wave, 0.3 cells/frame to the right
  const slowUp = run((r, c, t) => frac(0.05 * (Y(r) + 0.02 * t)));       // very slow wave moving up (−90°)
  const noise = run(() => rnd());                                        // fresh static every frame
  const drift = run((r, c, t) => (field[r][c] + t * field[r + 1][c]) % 1, 'drift'); // each cell spins at its own steady rate
  const driftNoise = run(() => rnd(), 'drift');
  const median = (ms) => { const v = []; for (let r = 2, j0 = 3; r < N - 2; j0 += r + 1, r++) for (let c = 2; c <= r - 2; c++) v.push(Math.hypot(...C.motionVelocity(ms, j0 + c))); return v.sort((a, b) => a - b)[v.length >> 1]; };
  const sSlow = median(run((r, c, t) => frac(0.07 * (X(r, c) - 0.3 * t)), 'flow').ms), sSlowUp = median(run((r, c, t) => frac(0.05 * (Y(r) + 0.02 * t)), 'flow').ms);
  // half the triangle moving, half static: the moving half lights, the static half does not
  const ms = C.newMotion(); C.motionAlloc(ms, 'k', n);
  for (let t = 0; t < 30; t++) {
    const th = new Float64Array(n);
    for (let r = 0, j = 0; r < N; r++) for (let c = 0; c <= r; c++, j++) th[j] = r < N / 2 ? frac(0.07 * (X(r, c) - 0.3 * t)) : rnd();
    C.motionStep(ms, th, N, 0, { mode: 'flow', tau: 6, trail: 0, pool: 1 });
  }
  let lit = [0, 0], cnt = [0, 0];
  for (let r = 0, j = 0; r < N; r++) for (let c = 0; c <= r; c++, j++) { const h = r < N / 2 ? 0 : 1; cnt[h]++; lit[h] += ms.T[j] > 0.3 ? 1 : 0; }
  assert(Math.abs(sSlow - 0.3) < 0.01 && Math.abs(sSlowUp - 0.02) < 0.002, `speeds ${sSlow} (want 0.3), ${sSlowUp} (want 0.02)`);
  // random textures break the smooth-phase assumption: weaker, but found, in the right direction, well above noise
  assert(right.strength > 5 * noise.strength && Math.abs(right.angle) < 10 && right.coherence > 0.8, 'rightward motion: ' + right.strength);
  assert(downRight.strength > 5 * noise.strength && Math.abs(downRight.angle - 60) < 10, 'down-right motion: ' + downRight.angle);
  assert(slow.strength > 0.8 && Math.abs(slow.angle) < 5, 'slow smooth wave');
  assert(slowUp.strength > 0.8 && Math.abs(slowUp.angle + 90) < 5, 'very slow upward wave');
  assert(noise.strength < 0.03, 'noise stays dark: ' + noise.strength);
  assert(drift.strength > 0.9 && driftNoise.strength < 0.05, `steady spin found (${drift.strength.toFixed(2)}), noise rejected (${driftNoise.strength.toFixed(3)})`);
  assert(lit[0] / cnt[0] > 0.9 && lit[1] / cnt[1] < 0.05, `moving half lit ${lit[0]}/${cnt[0]}, static half lit ${lit[1]}/${cnt[1]}`);
  checks += 8;
  console.log(`motion: speeds ${sSlow.toFixed(4)} / ${sSlowUp.toFixed(5)} (true 0.3 / 0.02); slow waves ${slow.strength.toFixed(2)}, ${slowUp.strength.toFixed(2)}; ` +
    `textures ${right.strength.toFixed(2)} @ ${right.angle.toFixed(1)}°, ${downRight.strength.toFixed(2)} @ ${downRight.angle.toFixed(1)}°; noise ${noise.strength.toFixed(3)}; ` +
    `drift ${drift.strength.toFixed(2)} vs ${driftNoise.strength.toFixed(3)}; half/half ${(lit[0] / cnt[0]).toFixed(2)} vs ${(lit[1] / cnt[1]).toFixed(2)}`);
}
// 15. ZIP
assert.equal(C.crc32(new TextEncoder().encode('hello')), 0x3610a686);
const zipParts = C.zipStore([{ name: 'a.txt', data: new TextEncoder().encode('hello') }, { name: 'dir/b.txt', data: new TextEncoder().encode('pascal\n') }]);
const zip = Buffer.concat(zipParts.map((p) => Buffer.from(p)));
assert.equal(zip.readUInt32LE(zip.length - 22), 0x06054b50);
assert.equal(zip.readUInt16LE(zip.length - 12), 2);
if (process.argv[2]) fs.writeFileSync(process.argv[2], zip);

console.log(`quantum walk: ${walkFront}`);
console.log(stripeInfo);
console.log(`storage tiers exercised: ${JSON.stringify(tiers)}; Grassmannian lens max relative error ${qMax.toExponential(1)}`);
console.log(`pascal-core: ${checks.toLocaleString()} checks passed (residues at 0–200 decimals, Kummer + cyclotomic carries, CRT, Lucas, q-Lucas, discrete log, first carry, p-adic digits, census, factoring, filter, motion), plus stepping, painters and ZIP.`);
