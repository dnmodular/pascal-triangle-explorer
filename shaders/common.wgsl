// ============================================================================
// common.wgsl — prepended to every pass.
//
// Bind group 0 is shared by the whole pipeline: the frame parameters, the
// number-theory tables, the color tables and the cell grid. The math lives in
// clearly separated functions:
//
//   mul64 / mont_mul            32-bit Montgomery multiplication (no u64 in WGSL)
//   mm / from_mont              multiplication mod one prime-power factor
//   binom_mod_prime_power       Granville's theorem (Lucas when e = 1) + Kummer carries
//   garner_combine (in value)   Chinese remainder theorem, Garner form
//   row_of / k0_of / cell_of    the grid texel → (n, k) mapping
//   texel_of                    the inverse: pixel → grid texel
// ============================================================================

struct Factor {
  p: u32,          // prime
  e: u32,          // exponent
  q: u32,          // p^e
  qinv: u32,       // Montgomery -q^{-1} mod 2^32 (odd q only)
  one: u32,        // 1 in Montgomery form (R mod q); plain 1 when p = 2
  p_mont: u32,     // p in Montgomery form; plain p when p = 2
  tab_off: u32,    // tables[tab_off .. +q] = (x!)_p, then [+q .. +2q] = its inverse
  minus_one: u32,  // 1 when (p^e !)_p ≡ -1 (every case except p = 2, e ≥ 3)
  garner_q: u32,   // product of the previous factors' q (1 for the first factor)
  garner_inv: u32, // garner_q^{-1} mod q, Montgomery form (plain when p = 2)
  ppow_off: u32,   // tables[ppow_off + t] = p^t mod q for t < e (Montgomery form; plain when p = 2)
  pad1: u32,
};

struct Params {
  grid_w: u32,
  grid_h: u32,
  canvas_w: u32,
  canvas_h: u32,

  n0: i32,          // row at the top edge of the view (floor), may be negative
  x0i: i32,         // floor of the world x at the left edge
  ss: u32,          // sub-samples per pixel edge in the resolve pass
  nfactors: u32,

  y0f: f32,         // fractional parts of the view origin
  x0f: f32,
  sx: f32,          // cells per texel (≥ 1)
  sy: f32,          // rows per texel (≥ 1)

  cells_per_px: f32,
  rows_per_px: f32,
  lnf_off: u32,     // tables[lnf_off + i] = ln(i!) as f32 bits, i < LNF_TABLE
  prank_off: u32,   // tables[prank_off + v] = 1-based prime index of v (0 = not prime), v < m

  m: u32,           // the modulus
  p: u32,           // its smallest prime factor
  dlog_off: u32,    // tables[dlog_off + x] = discrete log of x mod p (p odd)
  blend_mode: u32,  // 0 average, 1 multiply, 2 max, 3 screen

  color_mode: u32,  // 0 math ramp, 1 palette blend
  wants_value: u32, // VALUE layer folded into the blend
  n_measures: u32,  // active measures (fields 0..n_measures)
  value_style: u32, // 0 residue → palette, 1 zero/nonzero mask

  field_active: u32, // the field pipeline ran: resolve reads the working field
  field_stride: u32, // elements per field slot
  signed_mask: u32,  // bit s set when measure slot s is signed (normalized around 0.5)
  pad_d: u32,

  dparams: array<vec4<u32>, 8>, // per-layer values derived on the CPU (LayerDef::derive / dslot)
  factors: array<Factor, 9>,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> tables: array<u32>;
@group(0) @binding(2) var<storage, read> levels: array<u32>;   // residue → color level, 4 per word
@group(0) @binding(3) var<storage, read> palette: array<u32>;  // 256 res colors, 256 ramp colors, 256 hues, 256 saturations
@group(0) @binding(4) var<storage, read_write> cells: array<vec2<u32>>; // x = residue, y = packed (nu | mask << 8 | valid << 31)
@group(0) @binding(5) var<storage, read_write> maxbuf: array<atomic<u32>>; // per-slot field maxima as f32 bits

const CELL_VALID: u32 = 0x80000000u;
const MAX_FACTORS: u32 = 9u;     // distinct primes of a u32 modulus
const MAX_DIGITS: u32 = 32u;     // base-p digits of a u32 (p ≥ 2)
const LNF_TABLE: u32 = 4096u;    // ln(i!) is tabulated below this

fn cell_valid(c: vec2<u32>) -> bool { return (c.y & CELL_VALID) != 0u; }
fn cell_nu(c: vec2<u32>) -> u32 { return c.y & 0xffu; }
fn cell_mask(c: vec2<u32>) -> u32 { return (c.y >> 8u) & 0xffffu; }

fn level_of(residue: u32) -> u32 {
  return (levels[residue >> 2u] >> (8u * (residue & 3u))) & 0xffu;
}

fn unpack_rgb(c: u32) -> vec3<f32> {
  return vec3<f32>(f32((c >> 16u) & 0xffu), f32((c >> 8u) & 0xffu), f32(c & 0xffu)) / 255.0;
}

const BG_COLOR: vec3<f32> = vec3<f32>(10.0 / 255.0, 10.0 / 255.0, 10.0 / 255.0);
const LIGHT_COLOR: vec3<f32> = vec3<f32>(236.0 / 255.0, 236.0 / 255.0, 226.0 / 255.0);

// ---------------------------------------------------------------------------
// 32-bit modular multiplication.
// ---------------------------------------------------------------------------

// Full 64-bit product of two u32 as (lo, hi), from four 16-bit partial products.
fn mul64(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xffffu;
  let a1 = a >> 16u;
  let b0 = b & 0xffffu;
  let b1 = b >> 16u;
  let p00 = a0 * b0;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let p11 = a1 * b1;
  let mid = p01 + (p00 >> 16u);            // < 2^32: (2^16-1)^2 + 2^16 - 1
  let mid2 = (mid & 0xffffu) + p10;        // < 2^32 likewise
  let lo = (mid2 << 16u) | (p00 & 0xffffu);
  let hi = p11 + (mid >> 16u) + (mid2 >> 16u);
  return vec2<u32>(lo, hi);
}

// Montgomery REDC with R = 2^32: a·b·R^{-1} mod q for odd q < 2^31.
// With both inputs in Montgomery form (x·R) the result is in Montgomery form too.
fn mont_mul(a: u32, b: u32, q: u32, qinv: u32) -> u32 {
  let t = mul64(a, b);
  let mq = mul64(t.x * qinv, q);           // t.x * qinv wraps mod 2^32 on purpose
  let lo = t.x + mq.x;                     // ≡ 0 mod 2^32 by construction
  let carry = select(0u, 1u, lo < t.x);
  var u = t.y + mq.y + carry;              // < 2q
  if (u >= q) { u = u - q; }
  return u;
}

// a·b in the arithmetic of factor F: Montgomery for odd q, wrap-and-mask for 2^e.
fn mm(a: u32, b: u32, F: Factor) -> u32 {
  if (F.p == 2u) { return (a * b) & (F.q - 1u); }
  return mont_mul(a, b, F.q, F.qinv);
}

// Montgomery form → plain integer.
fn from_mont(x: u32, F: Factor) -> u32 {
  if (F.p == 2u) { return x; }
  return mont_mul(x, 1u, F.q, F.qinv);
}

// ---------------------------------------------------------------------------
// C(n,k) mod p^e by Granville's theorem; Lucas's theorem is the e = 1 case.
//
// Walk the base-p digits of n and k (the digits of r = n − k and the carries of
// k + r come from the subtraction n − k with borrow, so only two numbers are
// divided). At digit j the e-digit windows N_j, K_j, R_j index the tables of
// (x!)_p and its inverse:
//
//    C(n,k) / p^ν ≡ (±1)^{carries at positions ≥ e−1} · ∏_j (N_j!)_p / ((K_j!)_p (R_j!)_p)
//
// ν = total carries = ν_p(C(n,k)) (Kummer). Once ν reaches e the residue is 0;
// the loop then only keeps counting carries if the caller wants the full ν.
// ---------------------------------------------------------------------------
struct BinomRes { r: u32, nu: u32 };

fn binom_mod_prime_power(n: u32, k: u32, F: Factor, full_nu: bool) -> BinomRes {
  let p = F.p;
  let e = F.e;
  let q = F.q;
  let is2 = (p == 2u);
  let prime = (e == 1u);
  let qm = q - 1u;
  var nn = n;
  var kk = k;
  var carry = 0u;      // carry into digit j (= borrow of n − k)
  var nu = 0u;         // carries so far
  var sc = 0u;         // carries at digit positions ≥ e − 1 (the sign)
  var acc = F.one;     // running product, Montgomery form
  var zero = false;
  var done = false;
  var j = 0u;
  for (var it = 0u; it < MAX_DIGITS; it = it + 1u) {
    if (nn == 0u || done) { break; }
    // One division per number per digit; the digit is the remainder. For p = 2
    // everything is shifts and masks. The e-digit windows N_j, K_j are the digits
    // themselves when e = 1 (Lucas), otherwise one more remainder each.
    var nq = 0u;
    var kq = 0u;
    var nj = 0u;
    var kj = 0u;
    var nw = 0u;
    var kw = 0u;
    if (is2) {
      nq = nn >> 1u; kq = kk >> 1u; nj = nn & 1u; kj = kk & 1u; nw = nn & qm; kw = kk & qm;
    } else {
      nq = nn / p; kq = kk / p; nj = nn - nq * p; kj = kk - kq * p;
      if (prime) { nw = nj; kw = kj; } else { nw = nn % q; kw = kk % q; }
    }
    let c_out = select(0u, 1u, nj < kj + carry);
    if (!zero) {
      let t = nw + q - kw - carry;            // in [1, 2q)
      let rw = select(t, t - q, t >= q);      // R_j = N_j − K_j − carry (mod q)
      acc = mm(acc, tables[F.tab_off + nw], F);
      acc = mm(acc, tables[F.tab_off + q + kw], F);
      acc = mm(acc, tables[F.tab_off + q + rw], F);
      if (j + 1u >= e) { sc = sc + c_out; }
    }
    nu = nu + c_out;
    if (nu >= e) {
      zero = true;
      // The residue is 0; keep walking only when the caller wants the full carry count.
      done = !full_nu;
    }
    carry = c_out;
    nn = nq;
    kk = kq;
    j = j + 1u;
  }
  var r = 0u;
  if (!zero) {
    // ν < e here, and p^ν mod q comes from a CPU-built table: no power loop.
    r = mm(acc, tables[F.ppow_off + nu], F);
    r = from_mont(r, F);
    if (F.minus_one == 1u && (sc & 1u) == 1u) { r = select(q - r, 0u, r == 0u); }
  }
  return BinomRes(r, nu);
}

// ---------------------------------------------------------------------------
// C(n,k) mod m for the whole modulus: one residue per prime-power factor,
// recombined by Garner's form of the Chinese remainder theorem
//   x_1 = r_1,   x_f = x_{f−1} + Q_{f−1}·((r_f − x_{f−1})·Q_{f−1}^{−1} mod q_f)
// so every intermediate stays below m. Also returns ν_p for the smallest prime
// (full count only when want_nu) and the per-factor divisibility mask.
// ---------------------------------------------------------------------------
struct ResidueInfo { x: u32, nu0: u32, mask: u32 };

fn residue_info(n: u32, k: u32, want_nu: bool) -> ResidueInfo {
  var x = 0u;
  var nu0 = 0u;
  var mask = 0u;
  for (var f = 0u; f < MAX_FACTORS; f = f + 1u) {
    if (f >= P.nfactors) { break; }
    let F = P.factors[f];
    let res = binom_mod_prime_power(n, k, F, want_nu && f == 0u);
    if (f == 0u) {
      nu0 = res.nu;
      x = res.r;
    } else {
      let xq = x % F.q;
      let t = res.r + F.q - xq;               // in (0, 2q)
      let diff = select(t, t - F.q, t >= F.q);
      let cc = mm(diff, F.garner_inv, F);     // plain (r_f − x)·Q^{−1} mod q_f
      x = x + F.garner_q * cc;
    }
    if (res.nu >= F.e) { mask = mask | (1u << f); }
  }
  return ResidueInfo(x, nu0, mask);
}

// C(n,k) mod m (0 outside the triangle).
fn residue_at(n: u32, k: u32) -> u32 {
  if (k > n) { return 0u; }
  return residue_info(n, k, false).x;
}

// Wrap-around distance between two residues mod m.
fn wrap_dist(a: u32, b: u32) -> u32 {
  let d = select(b - a, a - b, a >= b);
  return min(d, P.m - d);
}

// ---------------------------------------------------------------------------
// Grid geometry. Texel (i, j) samples row n = n0 + ⌊(j + ½)·sy⌋ and cell
// k = k0(n) + ⌊(i + ½)·sx⌋, where k0(n) = ⌊x0 + n/2 + ½⌋ is the leftmost cell
// of row n touching the view. All large quantities stay in integers.
// ---------------------------------------------------------------------------
const MAX_ROW_F: f32 = 1.0e9;

fn row_of(j: u32) -> i32 {
  let r = floor((f32(j) + 0.5) * P.sy);
  return P.n0 + i32(clamp(r, 0.0, 2.0e9));
}

fn k0_of(n: u32) -> i32 {
  let half = select(0.0, 0.5, (n & 1u) == 1u);
  return P.x0i + i32(n >> 1u) + i32(floor(P.x0f + half + 0.5));
}

struct Cell { n: u32, k: u32, valid: u32 };

fn cell_of(i: u32, j: u32) -> Cell {
  var c = Cell(0u, 0u, 0u);
  let n = row_of(j);
  if (n < 0 || f32(n) > MAX_ROW_F) { return c; }
  let nu = u32(n);
  c.n = nu;
  let kf = floor((f32(i) + 0.5) * P.sx);
  if (kf > MAX_ROW_F) { return c; }
  let off = u32(kf);
  let k0 = k0_of(nu);
  if (k0 >= 0) {
    let k0u = u32(k0);
    if (k0u <= nu && off <= nu - k0u) {
      c.k = k0u + off;
      c.valid = 1u;
    }
  } else {
    let neg = u32(-k0);
    if (off >= neg && off - neg <= nu) {
      c.k = off - neg;
      c.valid = 1u;
    }
  }
  return c;
}

// Inverse mapping for the resolve pass: continuous pixel position → texel, or (-1,-1).
fn texel_of(px: f32, py: f32) -> vec2<i32> {
  let yf = P.y0f + py * P.rows_per_px;
  if (yf < 0.0) { return vec2<i32>(-1, -1); }
  let jf = floor(yf / P.sy);
  if (jf >= f32(P.grid_h)) { return vec2<i32>(-1, -1); }
  let j = i32(jf);
  let n = row_of(u32(j));
  if (n < 0) { return vec2<i32>(-1, -1); }
  let nu = u32(n);
  // x0i − k0(n) + ⌊n/2⌋ is −⌊x0f + half + ½⌋ ∈ {−1, 0}: tiny, so f32 is exact.
  let ibase = P.x0i - k0_of(nu) + i32(nu >> 1u);
  let half = select(0.0, 0.5, (nu & 1u) == 1u);
  let xf = P.x0f + px * P.cells_per_px;
  let iq = floor((f32(ibase) + xf + half + 0.5) / P.sx);
  if (iq < 0.0 || iq >= f32(P.grid_w)) { return vec2<i32>(-1, -1); }
  return vec2<i32>(i32(iq), j);
}

// Lattice neighbourhood of texel (i, j): the six neighbours of cell (n, k) are
// (n, k±1), (n−1, k−1), (n−1, k), (n+1, k), (n+1, k+1). Rows above and below are
// offset by the change in k0, so the brick pattern is followed exactly when a
// texel is a cell. Returns the texel index or -1.
fn neighbor_index(i: i32, j: i32, which: u32) -> i32 {
  let W = i32(P.grid_w);
  let H = i32(P.grid_h);
  var ni = i;
  var nj = j;
  if (which < 2u) {           // 0 up-left, 1 up-right
    nj = j - 1;
    if (nj < 0) { return -1; }
    let n = row_of(u32(j));
    let nup = row_of(u32(nj));
    if (n < 0 || nup < 0) { return -1; }
    ni = i + (k0_of(u32(n)) - k0_of(u32(nup))) - 1 + i32(which);
  } else if (which < 4u) {    // 2 down-left, 3 down-right
    nj = j + 1;
    if (nj >= H) { return -1; }
    let n = row_of(u32(j));
    let ndn = row_of(u32(nj));
    if (n < 0 || ndn < 0) { return -1; }
    ni = i - (k0_of(u32(ndn)) - k0_of(u32(n))) + i32(which) - 2;
  } else if (which == 4u) {   // left
    ni = i - 1;
  } else {                    // right
    ni = i + 1;
  }
  if (ni < 0 || ni >= W) { return -1; }
  let idx = nj * W + ni;
  if (!cell_valid(cells[idx])) { return -1; }
  return idx;
}
