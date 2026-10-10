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

// Parameter slots (see layers.rs: ParamDef::slot).
const SLOT_LUCAS_SCALE: u32 = 0u;
const SLOT_PADIC_DIGIT: u32 = 1u;

// p^t as u32, or 0 when it does not fit (the reference's Math.pow then floors to 0 digits).
fn pow_u32(p: u32, t: u32) -> u32 {
  var r = 1u;
  for (var i = 0u; i < t; i = i + 1u) {
    if (r > 0xffffffffu / p) { return 0u; }
    r = r * p;
  }
  return r;
}

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
  if (!c.valid) { buf_a[idx] = 0.0; return; }
  let p = P.p;
  let L = u32(clamp(iparam(SLOT_LUCAS_SCALE), 0, 12));
  let pl = pow_u32(p, L);
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
  let J = u32(clamp(iparam(SLOT_PADIC_DIGIT), 0, 12));
  let pj = pow_u32(p, J);
  buf_a[idx] = select(f32((c.x / pj) % p), 0.0, pj == 0u);
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
// reduce_max: maxbuf[PP.slot] = max over buf_a (non-negative floats compare as
// their bit patterns, so an integer atomicMax is exact).
// ---------------------------------------------------------------------------
var<workgroup> wg_max: atomic<u32>;

@compute @workgroup_size(16, 16)
fn reduce_max(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  if (lid == 0u) { atomicStore(&wg_max, 0u); }
  workgroupBarrier();
  if (gid.x < P.grid_w && gid.y < P.grid_h) {
    let v = buf_a[gid.y * P.grid_w + gid.x];
    if (v > 0.0) { atomicMax(&wg_max, bitcast<u32>(v)); }
  }
  workgroupBarrier();
  if (lid == 0u) { atomicMax(&maxbuf[PP.slot], atomicLoad(&wg_max)); }
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
