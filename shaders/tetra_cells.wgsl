// ============================================================================
// tetra_cells.wgsl — the tetra stack's residues (binom.wgsl is prepended).
//
// One thread per cell (m, n, k) of the stack, in the CPU's enumeration: slices
// in order, and inside a slice the strided rows i = 0.. with cells j = 0..i
// (n = i·stride, k = j·stride). Each slice carries its own modulus: the factor
// descriptors from `factor_off` (their tables shared between slices with the
// same prime power), recombined by Garner's CRT exactly as the 2D passes do.
//
// Nonzero cells are appended to `nonzero` and zeros to `zeros`; the two counts
// are the instance counts of the indirect draws that follow, so the cubes are
// drawn without a CPU round trip. Every loop has a constant bound.
// ============================================================================

struct CellParams {
  n_slices: u32,
  stride: u32,
  total: u32,
  row_threads: u32,   // threads per row of the 2D dispatch (x groups × 64)
};

struct Slice {
  m: u32,
  rows: u32,          // strided rows
  cell_base: u32,     // first cell of the slice in the enumeration
  nfactors: u32,
  factor_off: u32,    // factors[factor_off ..] (garner constants per slice)
  level_off: u32,     // residue → palette level bytes at levels[level_off + r] (draw pass)
  height: f32,        // y = −height
  ox: f32,            // the shape's shift of the slice in x and z (draw pass)
  oz: f32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<uniform> C: CellParams;
@group(0) @binding(1) var<storage, read> tables: array<u32>;
@group(0) @binding(2) var<storage, read> slices: array<Slice>;
@group(0) @binding(3) var<storage, read> factors: array<Factor>;
@group(0) @binding(4) var<storage, read_write> nonzero: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> zeros: array<vec2<u32>>;
// two DrawIndexedIndirect argument blocks: instance counts at words 1 and 6
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>, 10>;

const MAX_FACTORS_T: u32 = 9u;
const SLICE_SEARCH: u32 = 16u;   // binary search steps: 2^16 > any slice count

fn slice_residue(n: u32, k: u32, s: Slice) -> u32 {
  var x = 0u;
  for (var f = 0u; f < MAX_FACTORS_T; f = f + 1u) {
    if (f >= s.nfactors) { break; }
    let F = factors[s.factor_off + f];
    let res = binom_mod_prime_power(n, k, F, false);
    if (f == 0u) {
      x = res.r;
    } else {
      let xq = x % F.q;
      let t = res.r + F.q - xq;               // in (0, 2q)
      let diff = select(t, t - F.q, t >= F.q);
      let cc = mm(diff, F.garner_inv, F);     // plain (r_f − x)·Q^{−1} mod q_f
      x = x + F.garner_q * cc;
    }
  }
  return x;
}

@compute @workgroup_size(64)
fn tetra_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x + gid.y * C.row_threads;
  if (g >= C.total || C.n_slices == 0u) { return; }
  // the last slice whose cell_base ≤ g (slice 0 starts at 0)
  var lo = 0u;
  var hi = C.n_slices;
  for (var it = 0u; it < SLICE_SEARCH; it = it + 1u) {
    if (hi - lo <= 1u) { break; }
    let mid = (lo + hi) / 2u;
    if (slices[mid].cell_base <= g) { lo = mid; } else { hi = mid; }
  }
  let s = slices[lo];
  let l = g - s.cell_base;
  // row i with i(i+1)/2 ≤ l < (i+1)(i+2)/2: the f32 root is within one, then exact
  var i = u32(max(floor((sqrt(8.0 * f32(l) + 1.0) - 1.0) * 0.5), 0.0));
  for (var a = 0u; a < 2u; a = a + 1u) {
    if (i > 0u && i * (i + 1u) / 2u > l) { i = i - 1u; }
  }
  for (var a = 0u; a < 2u; a = a + 1u) {
    if ((i + 1u) * (i + 2u) / 2u <= l) { i = i + 1u; }
  }
  let j = l - i * (i + 1u) / 2u;
  let n = i * C.stride;
  let k = j * C.stride;
  let r = slice_residue(n, k, s);
  let w = vec2<u32>(n | (k << 16u), lo | (r << 16u));
  if (r != 0u) {
    nonzero[atomicAdd(&counters[1], 1u)] = w;
  } else {
    zeros[atomicAdd(&counters[6], 1u)] = w;
  }
}
