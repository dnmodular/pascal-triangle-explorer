// ============================================================================
// measures.wgsl — scalar field passes over the grid.
//
// Every measure is one entry point that reads the cell grid and writes a raw
// field into buf_a; its normalization (division by the field's maximum) happens
// in the blend pass after reduce_max has found that maximum. Adding a measure
// means adding one entry point here and one registry entry in layers.rs.
//
// Also here: reduce_max (field maximum into maxbuf[slot]) and blur_scalar
// (one hex-neighbourhood averaging pass over the triangular lattice).
// ============================================================================

struct PassParams {
  slot: u32,
  count: u32,
  mode: u32,
  flag: u32,
  f0: f32,
  f1: f32,
  f2: f32,
  f3: f32,
  hist_start: u32,
  hist_len: u32,
  hist_cap: u32,
  stride: u32,
};

@group(1) @binding(0) var<storage, read_write> buf_a: array<f32>;
@group(1) @binding(1) var<storage, read_write> buf_b: array<f32>;
@group(1) @binding(2) var<storage, read_write> buf_c: array<f32>;
@group(1) @binding(3) var<storage, read_write> buf_d: array<f32>;
@group(1) @binding(4) var<uniform> PP: PassParams;
@group(1) @binding(5) var<storage, read_write> counts: array<atomic<u32>>; // RARITY histogram

// Slots in P.dparams are filled by the registry's `derive` functions (layers.rs);
// the DSLOT_<LABEL> constants are generated from the registry (gpu::dslot_constants)
// and prepended to this file. DSLOT_LUCAS.x = p^L, DSLOT_PDIGIT.x = p^J, each 0 when
// the power does not fit a u32.

// KUMMER: carries adding k and n−k in base p = ν_p(C(n,k)), p the smallest prime factor.
@compute @workgroup_size(16, 16)
fn measure_kummer(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cells[idx];
  buf_a[idx] = select(0.0, f32(cell_nu(c)), cell_valid(c));
}

// LUCAS: the base-p digit pair (n_L, k_L) at scale L, 0 where k_L > n_L (a Lucas zero).
@compute @workgroup_size(16, 16)
fn measure_lucas(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cell_of(gid.x, gid.y);
  if (c.valid == 0u) { buf_a[idx] = 0.0; return; }
  let p = P.p;
  let pl = P.dparams[DSLOT_LUCAS].x;   // p^L from the CPU: no loop, no power here
  var nd = 0u;
  var kd = 0u;
  if (pl != 0u) {
    nd = (c.n / pl) % p;
    kd = (c.k / pl) % p;
  }
  buf_a[idx] = select(f32(nd * p + kd + 1u), 0.0, kd > nd);
}

// CRT: bitmask of the prime-power factors dividing C(n,k).
@compute @workgroup_size(16, 16)
fn measure_crt(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cells[idx];
  buf_a[idx] = select(0.0, f32(cell_mask(c)), cell_valid(c));
}

// DLOG: 1 + index of the residue mod p to a primitive root; 0 where p divides.
@compute @workgroup_size(16, 16)
fn measure_dlog(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cells[idx];
  if (!cell_valid(c)) { buf_a[idx] = 0.0; return; }
  let p = P.p;
  if (p == 2u) {
    buf_a[idx] = f32(c.x & 1u);
    return;
  }
  let vp = c.x % p;
  buf_a[idx] = select(f32(tables[P.dlog_off + vp] + 1u), 0.0, vp == 0u);
}

// P DIGIT: base-p digit J of the residue, ⌊v / p^J⌋ mod p.
@compute @workgroup_size(16, 16)
fn measure_padic(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cells[idx];
  if (!cell_valid(c)) { buf_a[idx] = 0.0; return; }
  let p = P.p;
  let pj = P.dparams[DSLOT_PDIGIT].x;   // p^J from the CPU
  var v = 0u;
  if (pj != 0u) { v = (c.x / pj) % p; }
  buf_a[idx] = f32(v);
}

// DENSITY: nonzero indicator; the registry then hex-blurs it R passes.
@compute @workgroup_size(16, 16)
fn measure_density(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cells[idx];
  buf_a[idx] = select(0.0, 1.0, cell_valid(c) && c.x != 0u);
}

// ---------------------------------------------------------------------------
// reduce_max: maxbuf[PP.slot] = max over buf_a (of |buf_a| when PP.flag = 1) (non-negative floats compare as
// their bit patterns, so an integer atomicMax is exact). The workgroup reduces
// through a plain shared array; only thread 0 touches the global atomic.
// ---------------------------------------------------------------------------
var<workgroup> wg_vals: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn reduce_max(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  var bits = 0u;
  if (gid.x < P.grid_w && gid.y < P.grid_h) {
    let raw = buf_a[gid.y * P.grid_w + gid.x];
    // PP.flag = 1: signed field, reduce its magnitude
    let v = select(raw, abs(raw), PP.flag == 1u);
    if (v > 0.0) { bits = bitcast<u32>(v); }
  }
  wg_vals[lid] = bits;
  workgroupBarrier();
  // Tree reduction: 8 uniform steps over the 256 shared values.
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) { wg_vals[lid] = max(wg_vals[lid], wg_vals[lid + s]); }
    workgroupBarrier();
  }
  if (lid == 0u && wg_vals[0] > 0u) { atomicMax(&maxbuf[PP.slot], wg_vals[0]); }
}

// ---------------------------------------------------------------------------
// blur_scalar: one neighbour-average pass over the lattice, buf_a → buf_b.
// Accumulation order matches the reference: up-left, up-right, down-left,
// down-right, left, right. Cells outside the triangle are skipped and left as is.
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn blur_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  if (!cell_valid(cells[idx])) { buf_b[idx] = buf_a[idx]; return; }
  var s = buf_a[idx];
  var cnt = 1.0;
  for (var w = 0u; w < 6u; w = w + 1u) {
    let ni = neighbor_index(i32(gid.x), i32(gid.y), w);
    if (ni >= 0) { s = s + buf_a[ni]; cnt = cnt + 1.0; }
  }
  buf_b[idx] = s / cnt;
}

// ===========================================================================
// Measures ported from the reference's mapFieldRaw. Every loop below has a
// constant bound; anything loop-derived (powers, tables) comes from the CPU.
// ===========================================================================

// Sum of one u32 per invocation over the workgroup (8 uniform tree steps), added
// to maxbuf[slot] by invocation 0. Must be reached by every invocation.
var<workgroup> wg_sum: array<u32, 256>;

fn workgroup_count_into(slot: u32, lid: u32, value: u32) {
  wg_sum[lid] = value;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) { wg_sum[lid] = wg_sum[lid] + wg_sum[lid + s]; }
    workgroupBarrier();
  }
  if (lid == 0u && wg_sum[0] > 0u) { atomicAdd(&maxbuf[slot], wg_sum[0]); }
}

// GRADIENT: wrap-distance from each cell to its two parents (n−1, k−1) and (n−1, k),
// evaluated in closed form, so it is exact at every zoom.
@compute @workgroup_size(16, 16)
fn measure_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cell_of(gid.x, gid.y);
  if (c.valid == 0u) { buf_a[idx] = 0.0; return; }
  let v = cells[idx].x;
  var g = 0u;
  if (c.n > 0u) {
    if (c.k > 0u) { g = g + wrap_dist(v, residue_at(c.n - 1u, c.k - 1u)); }
    if (c.k < c.n) { g = g + wrap_dist(v, residue_at(c.n - 1u, c.k)); }
  }
  buf_a[idx] = f32(g);
}

// ---------------------------------------------------------------------------
// DIST→0: breadth-first lattice distance to the nearest zero, as Jacobi
// relaxation. init: 0 on zeros, BIG elsewhere, and counts the zeros in view;
// step (dispatched PP.count times by the driver): d = min(d, 1 + min neighbours);
// finish: distances not reached within PP.count steps read PP.count, and a view
// without any zero reads 0 everywhere (the reference's "unreachable").
// ---------------------------------------------------------------------------
const DIST_BIG: f32 = 1.0e9;

@compute @workgroup_size(16, 16)
fn distzero_init(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  var zero = 0u;
  if (gid.x < P.grid_w && gid.y < P.grid_h) {
    let idx = gid.y * P.grid_w + gid.x;
    let c = cells[idx];
    let is_zero = cell_valid(c) && c.x == 0u;
    buf_a[idx] = select(DIST_BIG, 0.0, is_zero);
    zero = select(0u, 1u, is_zero);
  }
  workgroup_count_into(PP.slot, lid, zero);
}

@compute @workgroup_size(16, 16)
fn distzero_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  var d = buf_a[idx];
  if (cell_valid(cells[idx])) {
    for (var w = 0u; w < 6u; w = w + 1u) {
      let ni = neighbor_index(i32(gid.x), i32(gid.y), w);
      if (ni >= 0) { d = min(d, buf_a[ni] + 1.0); }
    }
  }
  buf_b[idx] = d;
}

@compute @workgroup_size(16, 16)
fn distzero_finish(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let zeros = atomicLoad(&maxbuf[PP.slot]);
  let valid = cell_valid(cells[idx]);
  let d = min(buf_a[idx], f32(PP.count));
  buf_a[idx] = select(0.0, d, valid && zeros > 0u);
}

// ---------------------------------------------------------------------------
// MAGNITUDE: ln C(n,k) of the true integer, ≥ 0. Exact table differences below
// LNF_TABLE (ln i! from the CPU); above it a Stirling form arranged so the large
// terms cancel analytically instead of in f32:
//   ln n! − ln m! = (m + ½)·(−log1p(−j/n)) + j·ln n − j + 1/(12n) − 1/(12m),  m = n − j
// ---------------------------------------------------------------------------
fn lnfact_table(x: u32) -> f32 { return bitcast<f32>(tables[P.lnf_off + x]); }

// Stirling series for ln Γ(x + 1).
fn lnfact_stirling(xf: f32) -> f32 {
  return (xf + 0.5) * log(xf) - xf + 0.9189385 + 1.0 / (12.0 * xf);
}

fn lnfact(x: u32) -> f32 {
  if (x < LNF_TABLE) { return lnfact_table(x); }
  return lnfact_stirling(f32(x));
}

// log(1 + y) without cancellation for small y (Kahan's trick).
fn log1p_safe(y: f32) -> f32 {
  let u = 1.0 + y;
  if (u == 1.0) { return y; }
  return log(u) * y / (u - 1.0);
}

fn ln_binom(n: u32, k: u32) -> f32 {
  let j = min(k, n - k);
  let m = n - j;
  if (n < LNF_TABLE) { return lnfact_table(n) - lnfact_table(j) - lnfact_table(m); }
  var d = 0.0;   // ln n! − ln m!
  if (m < LNF_TABLE) {
    d = lnfact_stirling(f32(n)) - lnfact_table(m);
  } else {
    let nf = f32(n);
    let mf = f32(m);
    let jf = f32(j);
    d = (mf + 0.5) * (-log1p_safe(-jf / nf)) + jf * log(nf) - jf + 1.0 / (12.0 * nf) - 1.0 / (12.0 * mf);
  }
  return d - lnfact(j);
}

@compute @workgroup_size(16, 16)
fn measure_magnitude(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cell_of(gid.x, gid.y);
  if (c.valid == 0u) { buf_a[idx] = 0.0; return; }
  buf_a[idx] = max(ln_binom(c.n, c.k), 0.0);
}

// ---------------------------------------------------------------------------
// VARIANCE: 1 − |mean of e^{2πi v/m}| over a hex neighbourhood. init writes the
// cosine and sine fields (the driver blurs each R passes), combine forms the
// circular variance.
// ---------------------------------------------------------------------------
const TAU: f32 = 6.2831853;

@compute @workgroup_size(16, 16)
fn variance_init(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cells[idx];
  let a = TAU * f32(c.x) / f32(P.m);
  buf_a[idx] = select(0.0, cos(a), cell_valid(c));
  buf_b[idx] = select(0.0, sin(a), cell_valid(c));
}

@compute @workgroup_size(16, 16)
fn variance_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let bc = buf_a[idx];
  let bs = buf_b[idx];
  let v = max(1.0 - sqrt(bc * bc + bs * bs), 0.0);
  buf_d[idx] = select(0.0, v, cell_valid(cells[idx]));
}

// ---------------------------------------------------------------------------
// RARITY: −log2 of each residue's frequency among the cells in view:
//   log2(cells in view) − log2(count of this residue).
// count: atomic histogram of residues (cleared by the driver) and the cell total.
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn rarity_count(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  var one = 0u;
  if (gid.x < P.grid_w && gid.y < P.grid_h) {
    let c = cells[gid.y * P.grid_w + gid.x];
    if (cell_valid(c)) {
      atomicAdd(&counts[c.x], 1u);
      one = 1u;
    }
  }
  workgroup_count_into(PP.slot, lid, one);
}

@compute @workgroup_size(16, 16)
fn rarity_lookup(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cells[idx];
  if (!cell_valid(c)) { buf_a[idx] = 0.0; return; }
  let total = f32(atomicLoad(&maxbuf[PP.slot]));
  let n = f32(max(atomicLoad(&counts[c.x]), 1u));
  buf_a[idx] = max(log2(total) - log2(n), 0.0);
}

// ---------------------------------------------------------------------------
// MOIRÉ: local covariance of the zero sets of the two smallest distinct primes p
// and q: blur(zp·zq) − blur(zp)·blur(zq). Signed; flat 0 on prime powers.
// DSLOT_MOIRE = (p, q, has_q).
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn moire_init(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cells[idx];
  let d = P.dparams[DSLOT_MOIRE];
  var zp = 0.0;
  var zq = 0.0;
  if (cell_valid(c) && d.z == 1u) {
    zp = select(0.0, 1.0, c.x % d.x == 0u);
    zq = select(0.0, 1.0, c.x % d.y == 0u);
  }
  buf_a[idx] = zp;
  buf_b[idx] = zq;
  buf_c[idx] = zp * zq;
}

@compute @workgroup_size(16, 16)
fn moire_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let v = buf_c[idx] - buf_a[idx] * buf_b[idx];
  buf_d[idx] = select(0.0, v, cell_valid(cells[idx]));
}

// PRIMES: 1-based prime index of the residue (spectrum) or 0/1 (mask), from a
// CPU-built table. DSLOT_PRIMES.x = 1 for mask.
@compute @workgroup_size(16, 16)
fn measure_primes(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cells[idx];
  if (!cell_valid(c)) { buf_a[idx] = 0.0; return; }
  let rank = tables[P.prank_off + c.x];
  let mask = P.dparams[DSLOT_PRIMES].x == 1u;
  buf_a[idx] = select(f32(rank), select(0.0, 1.0, rank > 0u), mask);
}

// ---------------------------------------------------------------------------
// STRIPES: length of the periodic A…B…A run through each cell along a lattice
// direction. Position s is consistent when v(s) ≈ v(s + 2P) and v(s) ≉ v(s + P)
// (wrap distance ≤ tol, > tol). The value is the run of consistent positions
// starting at s plus the run ending just before s, exactly as the reference's
// forward and backward chain DP, but walked per cell in closed form: at most
// STRIPE_CAP positions each way (longer runs read as STRIPE_CAP). DSLOT_STRIPES = (P·dr, P·dc, tol).
// ---------------------------------------------------------------------------
const STRIPE_CAP: u32 = 256u;

struct LatticePos { n: u32, k: u32, ok: bool };

fn lattice_step(n: u32, k: u32, j: i32, dr: u32, dc: u32) -> LatticePos {
  let r = i32(n) + j * i32(dr);
  let c = i32(k) + j * i32(dc);
  let ok = r >= 0 && c >= 0 && c <= r;
  return LatticePos(u32(max(r, 0)), u32(max(c, 0)), ok);
}

fn residue_or_zero(p: LatticePos) -> u32 {
  return select(0u, residue_at(p.n, p.k), p.ok);
}

fn stripe_consistent(a: u32, b: u32, c: u32, ok: bool, tol: u32) -> bool {
  return ok && wrap_dist(a, c) <= tol && wrap_dist(a, b) > tol;
}

@compute @workgroup_size(16, 16)
fn measure_stripes(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let cell = cell_of(gid.x, gid.y);
  if (cell.valid == 0u) { buf_a[idx] = 0.0; return; }
  let d = P.dparams[DSLOT_STRIPES];
  let dr = d.x;
  let dc = d.y;
  let tol = d.z;
  let n = cell.n;
  let k = cell.k;
  // forward: window (v_j, v_{j+1}, v_{j+2}) starting at j = 0
  let p1 = lattice_step(n, k, 1, dr, dc);
  let p2 = lattice_step(n, k, 2, dr, dc);
  let v0 = cells[idx].x;
  let v1 = residue_or_zero(p1);
  var a = v0;
  var b = v1;
  var c = residue_or_zero(p2);
  var okb = p1.ok;
  var okc = p2.ok;
  var fwd = 0u;
  for (var j = 0u; j < STRIPE_CAP; j = j + 1u) {
    if (!stripe_consistent(a, b, c, okb && okc, tol)) { break; }
    fwd = fwd + 1u;
    let pn = lattice_step(n, k, i32(j) + 3, dr, dc);
    a = b;
    b = c;
    c = residue_or_zero(pn);
    okb = okc;
    okc = pn.ok;
  }
  // backward: positions j = −1, −2, … with window (v_j, v_{j+1}, v_{j+2})
  var x = v0;       // v_{j+1}
  var y = v1;       // v_{j+2}
  var oky = p1.ok;
  var bwd = 0u;
  for (var t = 1u; t <= STRIPE_CAP; t = t + 1u) {
    let pj = lattice_step(n, k, -i32(t), dr, dc);
    if (!pj.ok) { break; }
    let w = residue_at(pj.n, pj.k);
    if (!stripe_consistent(w, x, y, oky, tol)) { break; }
    bwd = bwd + 1u;
    y = x;
    x = w;
    oky = true;
  }
  buf_a[idx] = f32(fwd + bwd);
}

// LAG DIFF: wrap-distance between each cell and the cell LAG steps away along the
// chosen direction (0 when that cell is outside the triangle). DSLOT_LAGDIFF = (L·dr, L·dc).
@compute @workgroup_size(16, 16)
fn measure_lagdiff(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let cell = cell_of(gid.x, gid.y);
  if (cell.valid == 0u) { buf_a[idx] = 0.0; return; }
  let d = P.dparams[DSLOT_LAGDIFF];
  let n2 = cell.n + d.x;
  let k2 = cell.k + d.y;
  var out = 0u;
  if (k2 <= n2) { out = wrap_dist(cells[idx].x, residue_at(n2, k2)); }
  buf_a[idx] = f32(out);
}
