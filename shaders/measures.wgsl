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

// Slots in P.dparams filled by the registry's `derive` functions (layers.rs).
const DSLOT_LUCAS: u32 = 0u;  // .x = p^L, or 0 when it does not fit a u32; .y = L
const DSLOT_PADIC: u32 = 1u;  // .x = p^J, or 0 when it does not fit a u32; .y = J

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
  let pj = P.dparams[DSLOT_PADIC].x;   // p^J from the CPU
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
// reduce_max: maxbuf[PP.slot] = max over buf_a (non-negative floats compare as
// their bit patterns, so an integer atomicMax is exact). The workgroup reduces
// through a plain shared array; only thread 0 touches the global atomic.
// ---------------------------------------------------------------------------
var<workgroup> wg_vals: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn reduce_max(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  var bits = 0u;
  if (gid.x < P.grid_w && gid.y < P.grid_h) {
    let v = buf_a[gid.y * P.grid_w + gid.x];
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
