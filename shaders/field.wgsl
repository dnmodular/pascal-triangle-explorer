// ============================================================================
// field.wgsl — the working field: blend, transforms and resolve.
//
// This file is compiled twice. FT is the field element type:
//   MATH RAMP      FT = f32          one value per texel, painted through the ramp
//   PALETTE BLEND  FT = vec4<f32>    the palette image (rgb), transforms per channel
// The FIELD_PRELUDE marker below is replaced with the variant's helper functions.
//
// Passes (fixed order): blend → [AVERAGE] → [BAND PASS] → [AFTERIMAGE] → resolve.
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

const MAX_MEASURE_SLOTS: u32 = 8u;
const HIST_CAP: u32 = 64u;
const MAX_SS: u32 = 4u;

{{FIELD_PRELUDE}}

@group(1) @binding(0) var<storage, read_write> fa: array<FT>;      // source / current
@group(1) @binding(1) var<storage, read_write> fb: array<FT>;      // destination
@group(1) @binding(2) var<storage, read_write> fc: array<FT>;      // aux: cut blur, history ring, adaptation
@group(1) @binding(3) var<storage, read_write> fields: array<f32>; // the measures' fields, slot-strided
@group(1) @binding(4) var<uniform> PP: PassParams;
@group(1) @binding(5) var out_tex: texture_storage_2d<rgba8unorm, write>;

// hslToRgb from the reference (h degrees, s and l in 0..1), channels rounded to 8 bits.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
  let c = (1.0 - abs(2.0 * l - 1.0)) * s;
  let h360 = h - 360.0 * floor(h / 360.0);        // h mod 360, in [0, 360)
  let hp = h360 / 60.0;
  let hp2 = hp - 2.0 * floor(hp / 2.0);            // hp mod 2
  let x = c * (1.0 - abs(hp2 - 1.0));
  var rgb = vec3<f32>(0.0);
  if (hp < 1.0) { rgb = vec3<f32>(c, x, 0.0); }
  else if (hp < 2.0) { rgb = vec3<f32>(x, c, 0.0); }
  else if (hp < 3.0) { rgb = vec3<f32>(0.0, c, x); }
  else if (hp < 4.0) { rgb = vec3<f32>(0.0, x, c); }
  else if (hp < 5.0) { rgb = vec3<f32>(x, 0.0, c); }
  else { rgb = vec3<f32>(c, 0.0, x); }
  let m0 = l - c / 2.0;
  return floor((rgb + m0) * 255.0 + 0.5) / 255.0;
}

// ---------------------------------------------------------------------------
// blend: fold the active measures (normalized by their maxima) and optionally
// the residue ratio into one 0..1 value per texel, then turn it into FT.
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn blend_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cells[idx];
  let valid = cell_valid(c);
  let denom = max(f32(P.m) - 1.0, 1.0);
  let nm = P.n_measures;
  var b = 0.0;
  if (nm == 0u) {
    // AVERAGE / BAND PASS alone (or with VALUE): operate on the residue ratio.
    b = f32(c.x) / denom;
  } else {
    let mode = P.blend_mode;
    var acc = select(0.0, 1.0, mode == 1u);
    var nl = 0.0;
    for (var s = 0u; s < MAX_MEASURE_SLOTS; s = s + 1u) {
      if (s >= nm) { break; }
      let mx = bitcast<f32>(atomicLoad(&maxbuf[s]));
      let inv = select(1.0 / mx, 1.0, mx == 0.0);
      let raw = fields[s * P.field_stride + idx] * inv;
      // signed measures are centered on 0.5, as in the reference's buildCombinedField
      let v = select(raw, 0.5 + 0.5 * clamp(raw, -1.0, 1.0), ((P.signed_mask >> s) & 1u) == 1u);
      acc = fold(acc, v, mode);
      nl = nl + 1.0;
    }
    if (P.wants_value == 1u) {
      acc = fold(acc, f32(c.x) / denom, mode);
      nl = nl + 1.0;
    }
    b = select(acc, acc / nl, mode == 0u);
  }
  fb[idx] = ft_from_blend(b, valid, c.x, nm > 0u);
}

fn fold(acc: f32, v: f32, mode: u32) -> f32 {
  if (mode == 1u) { return acc * v; }                       // multiply: intersection
  if (mode == 2u) { return max(acc, v); }                   // max
  if (mode == 3u) { return 1.0 - (1.0 - acc) * (1.0 - v); } // screen: union
  return acc + v;                                           // average (sum, divided later)
}

// ---------------------------------------------------------------------------
// blur_field: one hex-neighbourhood pass over the lattice, fa → fb.
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn blur_field(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  if (!cell_valid(cells[idx])) { fb[idx] = fa[idx]; return; }
  var s = fa[idx];
  var cnt = 1.0;
  for (var w = 0u; w < 6u; w = w + 1u) {
    let ni = neighbor_index(i32(gid.x), i32(gid.y), w);
    if (ni >= 0) { s = s + fa[ni]; cnt = cnt + 1.0; }
  }
  fb[idx] = s / cnt;
}

// ---------------------------------------------------------------------------
// temporal_main: AVERAGE over recent frames. fc is a ring of past under-fields
// (frame t at fc[t·stride + idx]); the current frame counts as one.
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn temporal_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  var s = fa[idx];
  for (var t = 0u; t < HIST_CAP; t = t + 1u) {
    if (t >= PP.hist_len) { break; }
    let f = (PP.hist_start + t) % PP.hist_cap;
    s = s + fc[f * PP.stride + idx];
  }
  fb[idx] = s / (1.0 + f32(PP.hist_len));
}

// ---------------------------------------------------------------------------
// BAND PASS: fb = fa − fc (keep blur minus cut blur), then reduce_max_abs finds
// max |d| and normalize_signed recenters it on 0.5.
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn band_diff(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  fb[idx] = fa[idx] - fc[idx];
}

var<workgroup> wg_abs: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn reduce_max_abs(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  var bits = 0u;
  if (gid.x < P.grid_w && gid.y < P.grid_h) {
    let idx = gid.y * P.grid_w + gid.x;
    if (cell_valid(cells[idx])) {
      let v = ft_max_abs(fa[idx]);
      if (v > 0.0) { bits = bitcast<u32>(v); }
    }
  }
  wg_abs[lid] = bits;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) { wg_abs[lid] = max(wg_abs[lid], wg_abs[lid + s]); }
    workgroupBarrier();
  }
  if (lid == 0u && wg_abs[0] > 0u) { atomicMax(&maxbuf[PP.slot], wg_abs[0]); }
}

@compute @workgroup_size(16, 16)
fn normalize_signed(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let mx = bitcast<f32>(atomicLoad(&maxbuf[PP.slot]));
  let inv = select(1.0 / mx, 1.0, mx == 0.0);
  fa[idx] = ft_splat(0.5) + fa[idx] * (0.5 * inv);
}

// ---------------------------------------------------------------------------
// AFTERIMAGE: the ghost is the complement of what past frames fatigued.
//   out = cur·(1 − w) + (1 − A)·w
// and the adaptation A follows cur as an envelope follower (attack when rising,
// decay when falling), stepped only when the frame advanced (PP.flag bit 0).
// PP.flag bit 1: A holds a value (else the unfatigued baseline 0.5 is used).
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn afterimage_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let w = PP.f0;
  let atk = PP.f1;
  let dec = PP.f2;
  let cur = fa[idx];
  var a = ft_splat(0.5);
  if ((PP.flag & 2u) != 0u) { a = fc[idx]; }
  fb[idx] = cur * (1.0 - w) + (ft_splat(1.0) - a) * w;
  if ((PP.flag & 1u) != 0u) {
    // attack where the signal rose above the adaptation, decay where it fell
    // (step(a, cur) = 1 where cur ≥ a; at equality the update is zero either way)
    fc[idx] = a + (cur - a) * mix(ft_splat(dec), ft_splat(atk), step(a, cur));
  }
}

// ---------------------------------------------------------------------------
// tween_capture: at the start of a morph, fb = what was on screen
//   mix(fb (start of the previous morph), fa (previous true field), PP.f0)
// PP.f0 = 1 when no morph was running, which copies the previous true field.
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn tween_capture(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  fb[idx] = select(mix(fb[idx], fa[idx], PP.f0), fa[idx], PP.f0 >= 1.0);
}

// ---------------------------------------------------------------------------
// resolve: canvas pixels → colors. ss × ss sub-samples per pixel, each mapped to
// its grid texel; the same shader serves the live view and the export.
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn resolve_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.canvas_w || gid.y >= P.canvas_h) { return; }
  let ss = clamp(P.ss, 1u, MAX_SS);
  var sum = vec3<f32>(0.0);
  for (var a = 0u; a < MAX_SS; a = a + 1u) {
    if (a >= ss) { break; }
    for (var b = 0u; b < MAX_SS; b = b + 1u) {
      if (b >= ss) { break; }
      let px = f32(gid.x) + (f32(a) + 0.5) / f32(ss);
      let py = f32(gid.y) + (f32(b) + 0.5) / f32(ss);
      sum = sum + sample_color(px, py);
    }
  }
  textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(sum / f32(ss * ss), 1.0));
}

fn sample_color(px: f32, py: f32) -> vec3<f32> {
  let t = texel_of(px, py);
  if (t.x < 0) { return BG_COLOR; }
  if (P.row_limit != 0u) {
    // a fitted view draws its rows only (the canvas past them is background)
    let yf = P.y0f + py * P.rows_per_px;
    if (f32(P.n0) + floor(yf) >= f32(P.row_limit)) { return BG_COLOR; }
  }
  let idx = u32(t.y) * P.grid_w + u32(t.x);
  let c = cells[idx];
  if (!cell_valid(c)) { return BG_COLOR; }
  var col: vec3<f32>;
  if (P.field_active == 1u) {
    // display-only morph (PP.flag bit 0): fc holds the field shown when it began
    var f = fa[idx];
    if ((PP.flag & 1u) != 0u) { f = mix(fc[idx], fa[idx], PP.f0); }
    col = ft_resolve_color(f);
  } else if (P.value_style == 1u) {
    col = select(BG_COLOR, LIGHT_COLOR, c.x != 0u);
  } else {
    col = unpack_rgb(palette[shift_level(level_of(c.x))]);
  }
  if (P.style_flags != 0u) {
    col = decorate(col, px, py, t);
  }
  return col;
}

// ---------------------------------------------------------------------------
// Cell decorations (outlines, glow around zeros). Only when a texel is one cell
// (sx = sy = 1) and cells are a few pixels wide; they fade in between 3 and 6 px.
// Geometry in pixels relative to the cell's top-left corner: the cell is
// [0, cw] × [0, ch]; the rows above and below are offset by half a cell.
// ---------------------------------------------------------------------------
const OUTLINE_RGB: vec3<f32> = vec3<f32>(235.0 / 255.0, 238.0 / 255.0, 250.0 / 255.0);
const OUTLINE_ALPHA: f32 = 0.12;  // the square scanner's rgba(235, 238, 250, .12)
const GLOW_MAX_PX: f32 = 10.0;    // its shadowBlur = min(10, 0.8 · cell)
const GLOW_STRENGTH: f32 = 0.55;

// Position of (px, py) inside its cell, in [0, 1)²: the fractional parts of the
// coordinates texel_of floors (with sx = sy = 1).
fn cell_uv(px: f32, py: f32, nu: u32) -> vec2<f32> {
  let yf = P.y0f + py * P.rows_per_px;
  let ibase = P.x0i - k0_of(nu) + i32(nu >> 1u);
  let half = select(0.0, 0.5, (nu & 1u) == 1u);
  let xf = P.x0f + px * P.cells_per_px;
  return vec2<f32>(fract(f32(ibase) + xf + half + 0.5), fract(yf));
}

// Distance (px) from p to the rectangle [lo, hi].
fn rect_dist(p: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>) -> f32 {
  return length(max(vec2<f32>(0.0), max(lo - p, p - hi)));
}

// True when cell (nn, kk) is in the triangle and m divides C(nn, kk). The residue comes
// from the grid when it holds the cell (texel row j + dn), else from the closed form:
// the grid has no margin on the left and top, and zeros just off-canvas still glow.
fn is_zero_cell(j: i32, dn: i32, nn: i32, kk: i32) -> bool {
  if (nn < 0 || kk < 0 || kk > nn || f32(nn) > MAX_ROW_F) { return false; }
  let jj = j + dn;
  if (jj >= 0 && jj < i32(P.grid_h)) {
    let ii = kk - k0_of(u32(nn));
    if (ii >= 0 && ii < i32(P.grid_w)) {
      let c = cells[u32(jj) * P.grid_w + u32(ii)];
      if (cell_valid(c)) { return c.x == 0u; }
    }
  }
  return residue_at(u32(nn), u32(kk)) == 0u;
}

fn decorate(col_in: vec3<f32>, px: f32, py: f32, t: vec2<i32>) -> vec3<f32> {
  if (P.sx != 1.0 || P.sy != 1.0) { return col_in; }
  let cw = 1.0 / P.cells_per_px;
  let ch = 1.0 / P.rows_per_px;
  let size = min(cw, ch);
  let fade = smoothstep(3.0, 6.0, size);
  if (fade <= 0.0) { return col_in; }
  let cell = cell_of(u32(t.x), u32(t.y));
  if (cell.valid == 0u) { return col_in; }
  let n = i32(cell.n);
  let k = i32(cell.k);
  let uv = cell_uv(px, py, cell.n);
  let p = uv * vec2<f32>(cw, ch);
  var col = col_in;
  if ((P.style_flags & 2u) != 0u) {
    // Light from every other zero cell within r = min(10 px, 0.8 cell): with r below
    // one row and one cell, only these ten can reach (the same row ±1, and rows n ± 1
    // from k − 2 to k + 2 shifted by half a cell). Zero cells glow too, from their
    // neighbours, so a void's lattice lights up as in the scanner.
    var nb = array<vec2<i32>, 10>(
      vec2<i32>(-1, -2), vec2<i32>(-1, -1), vec2<i32>(-1, 0), vec2<i32>(-1, 1),
      vec2<i32>(0, -1), vec2<i32>(0, 1),
      vec2<i32>(1, -1), vec2<i32>(1, 0), vec2<i32>(1, 1), vec2<i32>(1, 2));
    let r = min(GLOW_MAX_PX, 0.8 * size);
    var d = 1.0e9;
    for (var w = 0u; w < 10u; w = w + 1u) {
      let dn = nb[w].x;
      let dk = nb[w].y;
      // the neighbour's rectangle: rows are offset by half a cell per row step
      let lo = vec2<f32>((f32(dk) - 0.5 * f32(dn)) * cw, f32(dn) * ch);
      let dist = rect_dist(p, lo, lo + vec2<f32>(cw, ch));
      if (dist >= min(d, r)) { continue; }
      if (is_zero_cell(t.y, dn, n + dn, k + dk)) { d = dist; }
    }
    if (d < r) {
      let g = 1.0 - d / r;
      // screen blend: light only ever brightens
      let a = GLOW_STRENGTH * g * g * fade;
      col = vec3<f32>(1.0) - (vec3<f32>(1.0) - col) * (vec3<f32>(1.0) - unpack_rgb(P.glow_rgb) * a);
    }
  }
  if ((P.style_flags & 1u) != 0u) {
    // a 1 px line centred on each boundary, by coverage: a sample at distance e from
    // its cell's edge is covered by max(0, 1 − e) of a pixel-wide box
    let edge = min(min(p.x, cw - p.x), min(p.y, ch - p.y));
    let cover = clamp(1.0 - edge, 0.0, 1.0);
    col = mix(col, OUTLINE_RGB, OUTLINE_ALPHA * fade * cover);
  }
  return col;
}
