// ============================================================================
// common.wgsl — prepended to every pass.
//
// Bind group 0 is shared by the whole pipeline: the frame parameters, the
// number-theory tables, the color tables and the cell grid. The math lives in
// clearly separated functions:
//
//   (binom.wgsl, appended)      Montgomery multiplication, Granville / Lucas, Kummer
//   garner_combine (in value)   Chinese remainder theorem, Garner form
//   row_of / k0_of / cell_of    the grid texel → (n, k) mapping
//   texel_of                    the inverse: pixel → grid texel
// ============================================================================

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
  qscale: u32,     // a decimal modulus μ = m/q: q mod m (cells hold q·r mod m); 1 when whole

  style_flags: u32,  // resolve decorations: bit 0 cell outlines, bit 1 glow around zeros
  glow_rgb: u32,     // 0xRRGGBB of the glow (the palette ramp at 0.8)
  row_limit: u32,    // rows at or past this are not drawn (the fitted view's last row + 1; 0 = all)
  pad_f: u32,

  nfactors_b: u32,   // a morph to a second whole modulus m_b (m_b = 0: no morph): its factors,
  m_b: u32,
  morph_t: f32,      // how far the frame is from m to m_b, 0..1
  levels_b_off: u32, // its residue levels at levels[levels_b_off + r]

  dparams: array<vec4<u32>, 8>, // per-layer values derived on the CPU (LayerDef::derive / dslot)
  factors: array<Factor, 9>,
  factors_b: array<Factor, 9>,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> tables: array<u32>;
@group(0) @binding(2) var<storage, read> levels: array<u32>;   // residue → color level, 4 per word
@group(0) @binding(3) var<storage, read> palette: array<u32>;  // 256 res colors, 256 ramp colors, 256 hues, 256 saturations
@group(0) @binding(4) var<storage, read_write> cells: array<vec2<u32>>; // x = residue, y = packed (nu | mask << 8 | valid << 31)
@group(0) @binding(5) var<storage, read_write> maxbuf: array<atomic<u32>>; // per-slot field maxima as f32 bits

const CELL_VALID: u32 = 0x80000000u;
const MAX_FACTORS: u32 = 9u;     // distinct primes of a u32 modulus
const LNF_TABLE: u32 = 4096u;    // ln(i!) is tabulated below this

fn cell_valid(c: vec2<u32>) -> bool { return (c.y & CELL_VALID) != 0u; }
fn cell_nu(c: vec2<u32>) -> u32 { return c.y & 0xffu; }
fn cell_mask(c: vec2<u32>) -> u32 { return (c.y >> 8u) & 0xffffu; }

fn level_of(residue: u32) -> u32 {
  return (levels[residue >> 2u] >> (8u * (residue & 3u))) & 0xffu;
}

// The color offset: the level painted instead of `level`, from the map the CPU builds
// (palette::level_map) after the 1024 palette words; the identity at offset 0.
fn shift_level(level: u32) -> u32 {
  return (palette[1024u + (level >> 2u)] >> (8u * (level & 3u))) & 0xffu;
}

fn unpack_rgb(c: u32) -> vec3<f32> {
  return vec3<f32>(f32((c >> 16u) & 0xffu), f32((c >> 8u) & 0xffu), f32(c & 0xffu)) / 255.0;
}

const BG_COLOR: vec3<f32> = vec3<f32>(10.0 / 255.0, 10.0 / 255.0, 10.0 / 255.0);
const LIGHT_COLOR: vec3<f32> = vec3<f32>(236.0 / 255.0, 236.0 / 255.0, 226.0 / 255.0);

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

// C(n,k) mod m_b, the morph's second modulus (factors_b, the same Garner recombination).
fn residue_b(n: u32, k: u32) -> u32 {
  var x = 0u;
  for (var f = 0u; f < MAX_FACTORS; f = f + 1u) {
    if (f >= P.nfactors_b) { break; }
    let F = P.factors_b[f];
    let res = binom_mod_prime_power(n, k, F, false);
    if (f == 0u) {
      x = res.r;
    } else {
      let xq = x % F.q;
      let t = res.r + F.q - xq;
      let diff = select(t, t - F.q, t >= F.q);
      let cc = mm(diff, F.garner_inv, F);
      x = x + F.garner_q * cc;
    }
  }
  return x;
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
