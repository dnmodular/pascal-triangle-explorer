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
//   sphere_fs / sphere_pick_fs   spheres: a quad facing the camera, the sphere found
//                       per pixel by a ray test, its depth written
//
// A cell is a cube (8 corners, 36 indices), a rhombic dodecahedron (12 rhombi of 4
// corners each, 72 indices; the corners face by face in RHOMBIC, appended by the
// renderer) or a sphere (a quad, 6 indices), by flag.
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
  peel_b: vec4<u32>,    // x: layers from the base, y: layers in all, z: stride, w: last cube index
  eye: vec4<f32>,       // the camera: eye (w 1 perspective, 0 orthographic), forward, right, up
  fwd: vec4<f32>,
  right: vec4<f32>,
  up: vec4<f32>,
  shell: vec4<f32>,     // x: the last removed layers' gap (world), y: 1 when they are drawn
  nrm: array<vec4<f32>, 5>,   // the sides' outward normals: left, right, front, top, base
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
  k0: u32,          // a line's first k
  nx: f32,
  ny: f32,
  nz: f32,
  peel_id: u32,     // the row the front side's peel counts from | layer << 16
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
const FLAG_FLAT: u32 = 8u;   // a flat view: faces toward the viewer keep their true color, no edges
const FLAG_ZERO: u32 = 16u;  // the zeros drawn solid (the flat views: no gaps)
const FLAG_RHOMBIC: u32 = 32u;  // cells are rhombic dodecahedra (half.x = the stride)
const FLAG_SPHERE: u32 = 64u;   // cells are spheres (half.x = the radius)
const FLAG_TURNED: u32 = 128u;  // the top face's frame: (x, y, z) → (x, −z, y)
const LIGHT: vec3<f32> = vec3<f32>(236.0 / 255.0, 236.0 / 255.0, 226.0 / 255.0);

fn unpack_rgb(c: u32) -> vec3<f32> {
  return vec3<f32>(f32((c >> 16u) & 0xffu), f32((c >> 8u) & 0xffu), f32(c & 0xffu)) / 255.0;
}

fn cube_center(c: vec2<u32>, s: Slice) -> vec3<f32> {
  let n = f32(c.x & 0xffffu);
  let k = f32(c.x >> 16u);
  return vec3<f32>(s.ox + n * s.nx + k, s.oy + n * s.ny, s.oz + n * s.nz);
}

// Peeled cubes are not drawn (the stack's sides come off layer by layer, and a face's
// edges). The thresholds are the CPU's: layers × stride on the triangles' sides, layers
// of slices (a face's lines) at the top and the base. The diamond's are all 0.
fn peeled(c: vec2<u32>, si: u32, s: Slice) -> bool {
  let n = c.x & 0xffffu;
  let k = c.x >> 16u;
  let last = s.peel_id & 0xffffu;
  let layer = s.peel_id >> 16u;
  return (D.peel_a.x > 0u && k < D.peel_a.x)
      || (D.peel_a.y > 0u && n - k < D.peel_a.y)
      || (D.peel_a.z > 0u && last - n < D.peel_a.z)
      || (D.peel_a.w > 0u && layer < D.peel_a.w)
      || (D.peel_b.x > 0u && D.peel_b.y - 1u - layer < D.peel_b.x);
}

// The same, every side one layer shallower: still peeled there means deeper than the
// side's last removed layer.
fn peeled_shallower(c: vec2<u32>, s: Slice) -> bool {
  let n = c.x & 0xffffu;
  let k = c.x >> 16u;
  let last = s.peel_id & 0xffffu;
  let layer = s.peel_id >> 16u;
  let st = D.peel_b.z;
  let a = vec3<u32>(select(0u, D.peel_a.x - st, D.peel_a.x > st), select(0u, D.peel_a.y - st, D.peel_a.y > st), select(0u, D.peel_a.z - st, D.peel_a.z > st));
  let top = select(0u, D.peel_a.w - 1u, D.peel_a.w > 1u);
  let base = select(0u, D.peel_b.x - 1u, D.peel_b.x > 1u);
  return (a.x > 0u && k < a.x)
      || (a.y > 0u && n - k < a.y)
      || (a.z > 0u && last - n < a.z)
      || (top > 0u && layer < top)
      || (base > 0u && D.peel_b.y - 1u - layer < base);
}

// Where a cell goes: xyz its offset, w 1 when it is drawn. A cell on the stack stays;
// a peeled one is drawn only in a side's last removed layer while those show, pulled
// out along the normal of every side whose last layer holds it (an edge cell along
// two), so the sheets stay joined.
fn placement(c: vec2<u32>, si: u32, s: Slice) -> vec4<f32> {
  if (!peeled(c, si, s)) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
  if (D.shell.y < 0.5 || peeled_shallower(c, s)) { return vec4<f32>(0.0); }
  let n = c.x & 0xffffu;
  let k = c.x >> 16u;
  let last = s.peel_id & 0xffffu;
  let layer = s.peel_id >> 16u;
  var dir = vec3<f32>(0.0);
  if (D.peel_a.x > 0u && k < D.peel_a.x) { dir = dir + D.nrm[0].xyz; }
  if (D.peel_a.y > 0u && n - k < D.peel_a.y) { dir = dir + D.nrm[1].xyz; }
  if (D.peel_a.z > 0u && last - n < D.peel_a.z) { dir = dir + D.nrm[2].xyz; }
  if (D.peel_a.w > 0u && layer < D.peel_a.w) { dir = dir + D.nrm[3].xyz; }
  if (D.peel_b.x > 0u && D.peel_b.y - 1u - layer < D.peel_b.x) { dir = dir + D.nrm[4].xyz; }
  return vec4<f32>(dir * D.shell.x, 1.0);
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

struct CellVertex {
  offset: vec3<f32>,   // from the cell's centre
  local: vec3<f32>,    // a cube's corner in 0..1; a rhombus's (u, v, 0); a quad's (±1, ±1, 0)
};

// Vertex vi of a cell, by the cell flags.
fn cell_vertex(vi: u32) -> CellVertex {
  var out: CellVertex;
  if ((D.flags & FLAG_RHOMBIC) != 0u) {
    var p = RHOMBIC[vi] * D.half.x;
    if ((D.flags & FLAG_TURNED) != 0u) { p = vec3<f32>(p.x, -p.z, p.y); }
    let c = vi & 3u;
    out.offset = p;
    out.local = vec3<f32>(select(0.0, 1.0, c == 1u || c == 2u), select(0.0, 1.0, c >= 2u), 0.0);
  } else if ((D.flags & FLAG_SPHERE) != 0u) {
    // a little over the radius: in perspective the silhouette is a hair wider
    let q = vec2<f32>(select(-1.0, 1.0, (vi & 1u) != 0u), select(-1.0, 1.0, (vi & 2u) != 0u));
    out.offset = (D.right.xyz * q.x + D.up.xyz * q.y) * D.half.x * 1.25;
    out.local = vec3<f32>(q, 0.0);
  } else {
    let corner = corner_of(vi);
    out.offset = (corner * 2.0 - 1.0) * D.half.xyz;
    out.local = corner;
  }
  return out;
}

// The ray through a point of a sphere's quad, against the sphere: the entry point and
// how squarely it is hit (1 head on, 0 at the rim), or w < 0 for a miss.
fn sphere_ray(world: vec3<f32>, center: vec3<f32>) -> vec4<f32> {
  let r = D.half.x;
  var o = world - D.fwd.xyz * (4.0 * r);
  var d = D.fwd.xyz;
  if (D.eye.w > 0.5) {
    o = D.eye.xyz;
    d = normalize(world - o);
  }
  let oc = o - center;
  let b = dot(oc, d);
  let h = b * b - (dot(oc, oc) - r * r);
  if (h < 0.0) { return vec4<f32>(0.0, 0.0, 0.0, -1.0); }
  let p = o + d * (-b - sqrt(h));
  return vec4<f32>(p, -dot((p - center) / r, d));
}

fn depth_of(p: vec3<f32>) -> f32 {
  let c = D.view_proj * vec4<f32>(p, 1.0);
  return c.z / c.w;
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
  @location(4) @interpolate(flat) center: vec3<f32>,
};

@vertex
fn cube_vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> CubeOut {
  let c = cubes[cube_index(ii)];
  let si = c.y & 0xffffu;
  let s = slices[si];
  let place = placement(c, si, s);
  let center = cube_center(c, s) + place.xyz;
  let cv = cell_vertex(vi);
  let world = center + cv.offset;
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
  out.pos = select(D.view_proj * vec4<f32>(world, 1.0), NOWHERE, place.w < 0.5);
  out.local = cv.local;
  out.world = world;
  out.center = center;
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
  var ink = 1.0 - smoothstep(D.line_px - 0.5, D.line_px + 0.5, edge);
  var n = normalize(cross(dpdx(in.world), dpdy(in.world)));
  let sphere = (D.flags & FLAG_SPHERE) != 0u;
  var facing = 1.0;
  if (sphere) {
    // a sphere's ghost ring, or a flat disk in a planar view: the quad beyond the disk
    // dropped, the normal the sphere's, the rim darkened
    let hit = sphere_ray(in.world, in.center);
    if (hit.w < 0.0) { discard; }
    facing = hit.w;
    ink = 1.0 - smoothstep(0.0, 0.35, hit.w);
    n = (hit.xyz - in.center) / D.half.x;
  }

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
    var shade = select(lit_shade, mix(lit_shade, 1.0, abs(n.z)), (D.flags & FLAG_FLAT) != 0u);
    if (sphere) {
      // (lit as sphere_fs lights a solid sphere)
      shade = 0.5 + 0.5 * max(dot(n, l), 0.0) + 0.12 * facing;
    }
    var col = in.color * shade;
    var edge_col = col * 0.35;
    // the flat views tile cell against cell: no dark edges, only the highlight's (a
    // sphere keeps its rim)
    let edged = select(ink, 0.0, (D.flags & FLAG_FLAT) != 0u && in.lit != 2u && !sphere);
    if (in.lit == 2u) {
      col = mix(col, vec3<f32>(1.0), 0.3);
      edge_col = vec3<f32>(1.0);
    }
    return vec4<f32>(mix(col, edge_col, edged * 0.8), 1.0);
  }
  // wire: thin cubes far away merge into a translucent glow instead of a solid wall
  let a = ink * D.wire.a * clamp(cube_px / 10.0, 0.35, 1.0);
  if (a < 0.03) { discard; }
  return vec4<f32>(in.color * a, a);
}

struct SphereOut {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

// A filled sphere: lit from the same light as the cubes, its rim darkened (white under
// the highlight).
@fragment
fn sphere_fs(in: CubeOut) -> SphereOut {
  let hit = sphere_ray(in.world, in.center);
  if (hit.w < 0.0) { discard; }
  let n = (hit.xyz - in.center) / D.half.x;
  let l = normalize(vec3<f32>(0.35, 0.85, -0.4));
  var col = in.color * (0.5 + 0.5 * max(dot(n, l), 0.0) + 0.12 * hit.w);
  let rim = 1.0 - smoothstep(0.0, 0.35, hit.w);
  col = mix(col, col * 0.35, rim * 0.8);
  if (in.lit == 2u) {
    col = mix(mix(col, vec3<f32>(1.0), 0.3), vec3<f32>(1.0), rim * 0.8);
  }
  var out: SphereOut;
  out.color = vec4<f32>(col, 1.0);
  out.depth = depth_of(hit.xyz);
  return out;
}

// ---- picking: the frontmost nonzero cube under one pixel -----------------------

struct PickOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(flat) id: vec2<u32>,
  @location(1) world: vec3<f32>,
  @location(2) @interpolate(flat) center: vec3<f32>,
};

@vertex
fn pick_vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PickOut {
  let c = cubes[cube_index(ii)];
  let si = c.y & 0xffffu;
  let s = slices[si];
  let center = cube_center(c, s);
  let world = center + cell_vertex(vi).offset;
  var out: PickOut;
  out.pos = select(D.view_proj * vec4<f32>(world, 1.0), NOWHERE, peeled(c, si, s));
  out.id = vec2<u32>(c.x, c.y | 0x80000000u);
  out.world = world;
  out.center = center;
  return out;
}

@fragment
fn pick_fs(in: PickOut) -> @location(0) vec2<u32> {
  // (a sphere drawn as a flat disk: the quad's corners are not it)
  if ((D.flags & FLAG_SPHERE) != 0u && sphere_ray(in.world, in.center).w < 0.0) { discard; }
  return in.id;
}

struct SpherePick {
  @location(0) id: vec2<u32>,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn sphere_pick_fs(in: PickOut) -> SpherePick {
  let hit = sphere_ray(in.world, in.center);
  if (hit.w < 0.0) { discard; }
  var out: SpherePick;
  out.id = in.id;
  out.depth = depth_of(hit.xyz);
  return out;
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
