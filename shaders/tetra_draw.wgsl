// ============================================================================
// tetra_draw.wgsl — drawing the tetra stack.
//
// Each cube is one instance of a 36-index cube over 8 corners (the corner is
// decoded from the vertex index, no vertex buffer). The edges are drawn in the
// fragment shader: on a face, two of the cube's local coordinates vary and the
// distance to the nearest edge is the middle of the three distances to the
// faces, measured in pixels with fwidth. Fragments away from the edges are
// discarded, so the solid reads as a wireframe and its zero tunnels show.
//
// Cubes at or above the fill height are filled instead, in their palette color:
// the slice's residue → level table, its color-offset map, then the palette's
// residue colors, as in the 2D view. The 3D space fills everything; the preview
// tile fills from the top down while it is hovered. Cubes inside the hover
// brush (or the hovered slice) are highlighted.
//
//   cube_vs / cube_fs   the cubes (wire, or filled under the brush); the ghost
//                       pass draws the zeros with the same shaders (flag bit 2)
//   pick_vs / pick_fs   the ID pass: the cube under one pixel, as its words
//   line_vs / line_fs   slice outlines (solid) and skipped moduli (dashed)
// ============================================================================

struct DrawParams {
  view_proj: mat4x4<f32>,
  hover: vec4<f32>,     // xyz centre of the hovered cube, w = 1 when one is hovered
  half: vec4<f32>,      // cube half extents, w = brush radius (world units)
  wire: vec4<f32>,      // wire rgb, a = wire opacity
  hover_slice: u32,     // slice of the hovered cube
  flags: u32,           // bit 0 whole slice, bit 1 mask lens, bit 2 ghost pass, bit 3 flat view
  line_px: f32,         // wire half width in px
  ghost_alpha: f32,
  fill: vec4<f32>,      // x: cubes whose centre is at or above this height are filled in color
  peel_a: vec4<u32>,    // peeled away: k < x, n − k < y, last row − n < z (cells), slice < w
  peel_b: vec4<u32>,    // x: slices from the base, y: slices in the stack (0 = none), z: stride, w: last cube index
};

struct Slice {
  m: u32,
  rows: u32,
  cell_base: u32,
  nfactors: u32,
  factor_off: u32,
  level_off: u32,
  kind: u32,
  row_n: u32,
  ox: f32,          // cell (n, k) sits at o + n·nv + (k, 0, 0)
  oy: f32,
  oz: f32,
  pad0: u32,
  nx: f32,
  ny: f32,
  nz: f32,
  pad1: u32,
};

@group(0) @binding(0) var<uniform> D: DrawParams;
@group(0) @binding(1) var<storage, read> slices: array<Slice>;
@group(0) @binding(2) var<storage, read> levels: array<u32>;   // residue levels, 4 per word
@group(0) @binding(3) var<storage, read> colors: array<u32>;   // 256 residue colors, then 64 words of offset map per slice
@group(0) @binding(4) var<storage, read> cubes: array<vec2<u32>>;

const SQRT3_2: f32 = 0.8660254;
const FLAG_SLICE: u32 = 1u;
const FLAG_MASK: u32 = 2u;
const FLAG_GHOST: u32 = 4u;
const FLAG_FLAT: u32 = 8u;
const FLAG_ZERO: u32 = 16u;  // the zeros drawn solid (the flat views: no gaps)   // a flat view: faces toward the viewer keep their true color
const LIGHT: vec3<f32> = vec3<f32>(236.0 / 255.0, 236.0 / 255.0, 226.0 / 255.0);

fn unpack_rgb(c: u32) -> vec3<f32> {
  return vec3<f32>(f32((c >> 16u) & 0xffu), f32((c >> 8u) & 0xffu), f32(c & 0xffu)) / 255.0;
}

fn cube_center(c: vec2<u32>, s: Slice) -> vec3<f32> {
  let n = f32(c.x & 0xffffu);
  let k = f32(c.x >> 16u);
  return vec3<f32>(s.ox + n * s.nx + k, s.oy + n * s.ny, s.oz + n * s.nz);
}

// Peeled cubes are not drawn (the stack's sides come off layer by layer). The
// thresholds are the CPU's: layers × stride on the triangles' sides, layers of slices
// at the top and the base.
fn peeled(c: vec2<u32>, si: u32, s: Slice) -> bool {
  if (s.kind != 0u) { return false; }
  let n = c.x & 0xffffu;
  let k = c.x >> 16u;
  let last = s.row_n + (s.rows - 1u) * D.peel_b.z;
  return (D.peel_a.x > 0u && k < D.peel_a.x)
      || (D.peel_a.y > 0u && n - k < D.peel_a.y)
      || (D.peel_a.z > 0u && last - n < D.peel_a.z)
      || (D.peel_a.w > 0u && si < D.peel_a.w)
      || (D.peel_b.x > 0u && D.peel_b.y - 1u - si < D.peel_b.x);
}

// Outside the clip volume: a peeled cube's triangles collapse and are clipped.
const NOWHERE: vec4<f32> = vec4<f32>(0.0, 0.0, -2.0, 1.0);

// The zero passes (ghosts, solid zeros) read the zeros, stored from the back of the
// cube buffer.
fn cube_index(ii: u32) -> u32 {
  return select(ii, D.peel_b.w - ii, (D.flags & (FLAG_GHOST | FLAG_ZERO)) != 0u);
}

const ZERO_RGB: vec3<f32> = vec3<f32>(0.16, 0.16, 0.18);

fn corner_of(vi: u32) -> vec3<f32> {
  return vec3<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u), f32((vi >> 2u) & 1u));
}

// The palette color of residue r in slice si: level, offset map, color.
fn residue_rgb(si: u32, s: Slice, r: u32) -> vec3<f32> {
  if ((D.flags & FLAG_MASK) != 0u) {
    return select(vec3<f32>(10.0 / 255.0), LIGHT, r != 0u);
  }
  let b = s.level_off + r;
  let level = (levels[b >> 2u] >> (8u * (b & 3u))) & 0xffu;
  let mw = colors[256u + si * 64u + (level >> 2u)];
  let shifted = (mw >> (8u * (level & 3u))) & 0xffu;
  return unpack_rgb(colors[shifted]);
}

struct CubeOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) local: vec3<f32>,
  @location(1) color: vec3<f32>,
  @location(2) world: vec3<f32>,
  @location(3) @interpolate(flat) lit: u32,   // 0 wire, 1 filled, 2 filled and highlighted
};

@vertex
fn cube_vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> CubeOut {
  let c = cubes[cube_index(ii)];
  let si = c.y & 0xffffu;
  let s = slices[si];
  let center = cube_center(c, s);
  let corner = corner_of(vi);
  let world = center + (corner * 2.0 - 1.0) * D.half.xyz;
  var lit = false;
  if (D.hover.w > 0.5) {
    if ((D.flags & FLAG_SLICE) != 0u) {
      lit = si == D.hover_slice;
    } else {
      let d = center - D.hover.xyz;
      lit = dot(d, d) <= D.half.w * D.half.w + 1e-3;
    }
  }
  var out: CubeOut;
  out.pos = select(D.view_proj * vec4<f32>(world, 1.0), NOWHERE, peeled(c, si, s));
  out.local = corner;
  out.world = world;
  let filled = center.y >= D.fill.x || lit || (D.flags & FLAG_ZERO) != 0u;
  out.lit = select(select(0u, 1u, filled), 2u, lit);
  out.color = D.wire.rgb;
  if (filled && (D.flags & FLAG_GHOST) == 0u) {
    out.color = residue_rgb(si, s, c.y >> 16u);
  }
  if ((D.flags & FLAG_ZERO) != 0u) {
    out.color = ZERO_RGB;
  }
  return out;
}

@fragment
fn cube_fs(in: CubeOut) -> @location(0) vec4<f32> {
  // derivatives first, in uniform control flow
  let d = min(in.local, vec3<f32>(1.0) - in.local);
  let mid = d.x + d.y + d.z - max(max(d.x, d.y), d.z) - min(min(d.x, d.y), d.z);
  let fw = max(fwidth(mid), 1e-6);
  let edge = mid / fw;                       // px to the nearest edge of this face
  let lw = fwidth(in.local);
  let cube_px = 1.0 / max(max(lw.x, lw.y), max(lw.z, 1e-6));
  let n = normalize(cross(dpdx(in.world), dpdy(in.world)));
  let ink = 1.0 - smoothstep(D.line_px - 0.5, D.line_px + 0.5, edge);

  if ((D.flags & FLAG_GHOST) != 0u) {
    let a = ink * D.ghost_alpha * select(1.0, 3.0, in.lit == 2u);
    if (a < 0.004) { discard; }
    return vec4<f32>(in.color * a, a);       // premultiplied, added
  }
  if (in.lit >= 1u) {
    // filled: flat shading from the face normal, with darker edges; highlighted
    // cubes (the hover brush) lighten and get bright edges
    let l = normalize(vec3<f32>(0.35, 0.85, -0.4));
    // (in the flat views a face turned to the viewer shows the palette color itself, as
    // in the 2D view; faces turned away, as the hinge swings, still shade)
    let lit_shade = 0.55 + 0.45 * abs(dot(n, l));
    let shade = select(lit_shade, mix(lit_shade, 1.0, abs(n.z)), (D.flags & FLAG_FLAT) != 0u);
    var col = in.color * shade;
    var edge_col = col * 0.35;
    if (in.lit == 2u) {
      col = mix(col, vec3<f32>(1.0), 0.3);
      edge_col = vec3<f32>(1.0);
    }
    return vec4<f32>(mix(col, edge_col, ink * 0.8), 1.0);
  }
  // wire: thin cubes far away merge into a translucent glow instead of a solid wall
  let a = ink * D.wire.a * clamp(cube_px / 10.0, 0.35, 1.0);
  if (a < 0.03) { discard; }
  return vec4<f32>(in.color * a, a);
}

// ---- picking: the frontmost nonzero cube under one pixel -----------------------

struct PickOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(flat) id: vec2<u32>,
};

@vertex
fn pick_vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PickOut {
  let c = cubes[cube_index(ii)];
  let si = c.y & 0xffffu;
  let s = slices[si];
  let world = cube_center(c, s) + (corner_of(vi) * 2.0 - 1.0) * D.half.xyz;
  var out: PickOut;
  out.pos = select(D.view_proj * vec4<f32>(world, 1.0), NOWHERE, peeled(c, si, s));
  out.id = vec2<u32>(c.x, c.y | 0x80000000u);
  return out;
}

@fragment
fn pick_fs(in: PickOut) -> @location(0) vec2<u32> {
  return in.id;
}

// ---- lines ----------------------------------------------------------------------

struct LineIn {
  @location(0) pos: vec3<f32>,
  @location(1) color: vec4<f32>,
  @location(2) dash: f32,           // < 0 solid; else dashes of one unit
};

struct LineOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) dash: f32,
};

@vertex
fn line_vs(v: LineIn) -> LineOut {
  var out: LineOut;
  out.pos = D.view_proj * vec4<f32>(v.pos, 1.0);
  out.color = v.color;
  out.dash = v.dash;
  return out;
}

@fragment
fn line_fs(in: LineOut) -> @location(0) vec4<f32> {
  if (in.dash >= 0.0 && fract(in.dash) > 0.5) { discard; }
  return vec4<f32>(in.color.rgb * in.color.a, in.color.a);
}
