//! The tetra stack: Pascal's triangle mod m for many moduli at once, one horizontal
//! slice per modulus. Slice m holds rows 0..m−1 of the triangle mod m, so slices grow
//! linearly with m and the stack is a tetrahedron (mod 2, two rows, is the apex).
//!
//! Cell (m, n, k) sits at x = k − n/2, z = n·√3/2 (the 2D brick layout laid flat) and
//! y = −height(m): the slice's height is m itself (TRUE spacing: skipped moduli leave
//! gaps, the edges stay straight) or its index in the selection (PACKED: no gaps, the
//! sides curve because slice size no longer tracks height).
//!
//! This module is the CPU side: which moduli are stacked, the level of detail that keeps
//! the cube count in budget, the camera, and the reference ray cast the picking test
//! compares the GPU's ID buffer against. [`crate::tetra_gpu`] renders it.

use crate::math::{factor_string, factorize, Modulus};
use crate::sweep::Sweep;

/// Largest modulus a slice can have (n, k and the residue are packed in 16 bits each).
pub const TETRA_MAX_M: u32 = 4096;
/// √3 / 2: the row spacing of the brick layout.
pub const SQRT3_2: f32 = 0.866_025_4;
/// Cube half extents as a fraction of the cell pitch (a small gap between cubes).
pub const CUBE_FILL: f32 = 0.45;
/// Half height of a cube; slices are one unit apart at their closest.
pub const CUBE_HALF_Y: f32 = 0.42;
/// Cube-count budgets offered in the panel.
pub const BUDGETS: [u32; 6] = [250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000];

/// How slices are spaced vertically.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Spacing {
    /// Slice height = m: skipped moduli leave gaps and the edges stay straight.
    #[default]
    True,
    /// Slice height = its index in the selection: no gaps, curved sides.
    Packed,
}

impl Spacing {
    pub fn label(self) -> &'static str {
        match self {
            Spacing::True => "true spacing",
            Spacing::Packed => "packed",
        }
    }
    pub fn toggled(self) -> Spacing {
        match self {
            Spacing::True => Spacing::Packed,
            Spacing::Packed => Spacing::True,
        }
    }
}

/// Everything the panel's TETRA STACK section sets.
#[derive(Clone, Debug, PartialEq)]
pub struct TetraConfig {
    /// The selection: `start..=end` filtered by the mode (all, primes, composites, even,
    /// odd, interval / sequences). Only the selection fields are used, not playback.
    pub sweep: Sweep,
    pub spacing: Spacing,
    /// Zero cells as faint ghost cubes.
    pub ghosts: bool,
    /// Hover brush radius in cell units (0 = the hovered cube only).
    pub brush: f32,
    /// Color the whole hovered slice instead of a brush.
    pub color_slice: bool,
    /// Most cubes drawn; a larger selection gets a cell stride.
    pub budget: u32,
    /// The preview tile turns slowly.
    pub spin: bool,
    /// Show the preview tile on the 2D canvas.
    pub tile: bool,
}

impl Default for TetraConfig {
    fn default() -> Self {
        let mut sweep = Sweep::default();
        sweep.end = 48;
        TetraConfig { sweep, spacing: Spacing::True, ghosts: false, brush: 2.0, color_slice: false, budget: 1_000_000, spin: true, tile: true }
    }
}

impl TetraConfig {
    /// What the stack's geometry depends on (the scene is rebuilt when it changes).
    pub fn scene_key(&self) -> SceneKey {
        let s = &self.sweep;
        (s.mode, s.prime_order, s.seq, s.step, s.start, s.end, self.spacing, self.budget)
    }
}

pub type SceneKey = (crate::sweep::ModMode, crate::sweep::PrimeOrder, crate::sweep::Seq, u32, u32, u32, Spacing, u32);

/// The moduli in `start..=end` (clamped to 2..=TETRA_MAX_M) that the selection mode
/// visits, ascending.
pub fn selected_moduli(sweep: &Sweep) -> Vec<u32> {
    let mut sw = sweep.clone();
    let (mut lo, mut hi) = (sw.start.clamp(2, TETRA_MAX_M), sw.end.clamp(2, TETRA_MAX_M));
    if lo > hi {
        std::mem::swap(&mut lo, &mut hi);
    }
    // the interval walks are anchored at the start: the lower bound
    sw.start = lo;
    sw.end = hi;
    (lo..=hi).filter(|&m| sw.snap(m as i64) == m).collect()
}

/// Strided rows of slice m: rows 0, s, 2s, … below m.
pub fn strided_rows(m: u32, stride: u32) -> u32 {
    (m - 1) / stride + 1
}

/// Cells of a slice at stride s: row i·s holds cells 0, s, …, i·s.
pub fn strided_cells(m: u32, stride: u32) -> u64 {
    let r = strided_rows(m, stride) as u64;
    r * (r + 1) / 2
}

/// The smallest cell stride that keeps the stack within `budget` cubes.
pub fn stride_for(mods: &[u32], budget: u32) -> u32 {
    let total = |s: u32| mods.iter().map(|&m| strided_cells(m, s)).sum::<u64>();
    let full = total(1);
    if full <= budget as u64 {
        return 1;
    }
    // cells shrink like 1/s²: start just below the estimate and walk up
    let mut s = ((full as f64 / budget.max(1) as f64).sqrt().floor() as u32).max(1);
    while s > 1 && total(s - 1) <= budget as u64 {
        s -= 1;
    }
    while total(s) > budget as u64 {
        s += 1;
    }
    s
}

/// One slice as laid out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slice {
    pub m: u32,
    /// Distance below the top of the stack (y = −height).
    pub height: f32,
    /// Strided rows and cells.
    pub rows: u32,
    pub cells: u32,
    /// First cell of this slice in the stack's cell enumeration.
    pub cell_base: u32,
}

/// The stacked slices at one level of detail.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub slices: Vec<Slice>,
    pub spacing: Spacing,
    /// Every `stride`-th row and cell is drawn (1 = all).
    pub stride: u32,
    /// Cells in the enumeration (nonzero cubes + zeros).
    pub total: u32,
    /// Cells of the selection at stride 1.
    pub full_total: u64,
}

impl Layout {
    pub fn new(mods: &[u32], spacing: Spacing, budget: u32) -> Layout {
        let stride = stride_for(mods, budget);
        let mut slices = Vec::with_capacity(mods.len());
        let mut base = 0u32;
        for (i, &m) in mods.iter().enumerate() {
            let cells = strided_cells(m, stride) as u32;
            let height = match spacing {
                Spacing::True => m as f32,
                Spacing::Packed => i as f32,
            };
            slices.push(Slice { m, height, rows: strided_rows(m, stride), cells, cell_base: base });
            base += cells;
        }
        let full_total = mods.iter().map(|&m| strided_cells(m, 1)).sum();
        Layout { slices, spacing, stride, total: base, full_total }
    }

    /// Cube half extents (x, y, z) in world units.
    pub fn half_extents(&self) -> [f32; 3] {
        let s = self.stride as f32;
        [CUBE_FILL * s, CUBE_HALF_Y, CUBE_FILL * s * SQRT3_2]
    }

    /// World bounds of the cube centres' slices, padded by a cube.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for s in &self.slices {
            let n_last = ((s.rows - 1) * self.stride) as f32;
            for p in [[0.0, -s.height, 0.0], [-n_last / 2.0, -s.height, n_last * SQRT3_2], [n_last / 2.0, -s.height, n_last * SQRT3_2]] {
                for a in 0..3 {
                    lo[a] = lo[a].min(p[a]);
                    hi[a] = hi[a].max(p[a]);
                }
            }
        }
        if self.slices.is_empty() {
            return ([-1.0; 3], [1.0; 3]);
        }
        let h = self.half_extents();
        (sub(lo, h), add(hi, h))
    }

    /// The slice index of modulus m.
    pub fn slice_of(&self, m: u32) -> Option<usize> {
        self.slices.binary_search_by_key(&m, |s| s.m).ok()
    }

    /// Cell number `g` of the enumeration → (slice, n, k), as the compute shader maps it.
    pub fn cell(&self, g: u32) -> Option<(usize, u32, u32)> {
        if g >= self.total {
            return None;
        }
        let si = self.slices.partition_point(|s| s.cell_base <= g) - 1;
        let s = &self.slices[si];
        let l = (g - s.cell_base) as u64;
        let mut i = (((8 * l + 1) as f64).sqrt() as u64 - 1) / 2;
        while i * (i + 1) / 2 > l {
            i -= 1;
        }
        while (i + 1) * (i + 2) / 2 <= l {
            i += 1;
        }
        let j = l - i * (i + 1) / 2;
        Some((si, i as u32 * self.stride, j as u32 * self.stride))
    }

    /// Centre of cube (slice, n, k).
    pub fn center(&self, slice: usize, n: u32, k: u32) -> [f32; 3] {
        cell_center(self.slices[slice].height, n, k)
    }

    /// The moduli between the first and last slice that the selection skips (drawn as
    /// dashed outlines in true spacing).
    pub fn missing(&self) -> Vec<u32> {
        if self.spacing != Spacing::True || self.slices.len() < 2 {
            return Vec::new();
        }
        self.slices.windows(2).flat_map(|w| (w[0].m + 1)..w[1].m).collect()
    }

    /// Consecutive slices more than one modulus apart: (upper m, lower m), labeled with
    /// the gap `lower − upper` in true spacing.
    pub fn gaps(&self) -> Vec<(u32, u32)> {
        if self.spacing != Spacing::True {
            return Vec::new();
        }
        self.slices.windows(2).filter(|w| w[1].m - w[0].m > 1).map(|w| (w[0].m, w[1].m)).collect()
    }

    /// The outline of a slice of modulus m at height h: the triangle around its cubes.
    pub fn outline(&self, m: u32, h: f32) -> [[f32; 3]; 3] {
        let n_last = ((strided_rows(m, self.stride) - 1) * self.stride) as f32;
        let e = self.half_extents();
        let pad = e[0] + 0.25;
        let y = -h;
        let zl = n_last * SQRT3_2 + e[2] + 0.25;
        [[0.0, y, -e[2] - 0.6], [-n_last / 2.0 - pad - 0.4, y, zl], [n_last / 2.0 + pad + 0.4, y, zl]]
    }
}

/// Centre of cell (n, k) of a slice at height h.
pub fn cell_center(h: f32, n: u32, k: u32) -> [f32; 3] {
    [k as f32 - n as f32 / 2.0, -h, n as f32 * SQRT3_2]
}

/// Packed instance words, as the compute pass writes them and the pick pass returns them:
/// `n | k << 16` and `slice | residue << 16`.
pub fn pack_cube(slice: u32, n: u32, k: u32, residue: u32) -> [u32; 2] {
    [n | (k << 16), slice | (residue << 16)]
}

/// The cube a pick (or instance) word pair names.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CubeId {
    pub slice: u32,
    pub n: u32,
    pub k: u32,
    pub residue: u32,
}

impl CubeId {
    pub fn unpack(w: [u32; 2]) -> CubeId {
        CubeId { n: w[0] & 0xffff, k: w[0] >> 16, slice: w[1] & 0xffff, residue: (w[1] >> 16) & 0x7fff }
    }
}

/// The HUD line for a cube of modulus m.
pub fn describe(m: u32, c: &CubeId) -> String {
    let fs = factorize(m);
    let f = if fs.len() == 1 && fs[0].e == 1 { "prime".to_string() } else { factor_string(&fs) };
    format!("m {m} ({f}) · n {} · k {} · C(n,k) ≡ {} (mod {m})", c.n, c.k, c.residue)
}

// ---------------------------------------------------------------------------------------
// Camera: an orbit around a target, perspective, wgpu clip space (depth 0..1).
// ---------------------------------------------------------------------------------------

/// Column-major 4×4 (WGSL's `mat4x4<f32>` layout).
pub type Mat4 = [[f32; 4]; 4];

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn normalize(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt().max(1e-20);
    scale(a, 1.0 / l)
}

pub fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut r = [[0f32; 4]; 4];
    for (c, col) in r.iter_mut().enumerate() {
        for (row, v) in col.iter_mut().enumerate() {
            *v = (0..4).map(|i| a[i][row] * b[c][i]).sum();
        }
    }
    r
}

/// `m · (p, 1)` → clip coordinates.
pub fn transform(m: &Mat4, p: [f32; 3]) -> [f32; 4] {
    let mut r = [0f32; 4];
    for (row, v) in r.iter_mut().enumerate() {
        *v = m[0][row] * p[0] + m[1][row] * p[1] + m[2][row] * p[2] + m[3][row];
    }
    r
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub target: [f32; 3],
    /// Around the vertical axis, and above the horizon (radians).
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    /// Vertical field of view (radians).
    pub fov_y: f32,
    /// Scene radius, for the clip planes.
    pub radius: f32,
}

impl Camera {
    /// A view of the whole stack, from beside its base and a little above, where its
    /// four corners all show.
    pub fn framing(bounds: ([f32; 3], [f32; 3])) -> Camera {
        let (lo, hi) = bounds;
        let target = scale(add(lo, hi), 0.5);
        let radius = (dot(sub(hi, lo), sub(hi, lo)).sqrt() * 0.5).max(1.0);
        let fov_y = 40f32.to_radians();
        // the solid fills far less than its bounding sphere: frame it a little inside
        let dist = radius / (fov_y * 0.5).tan() * 1.1;
        Camera { target, yaw: 2.5, pitch: 0.38, dist, fov_y, radius }
    }

    pub fn eye(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        // yaw 0 looks along +z from the front (−z side): the apex row faces the viewer
        add(self.target, scale([sy * cp, sp, -cy * cp], self.dist))
    }

    fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let f = normalize(sub(self.target, self.eye()));
        let r = normalize(cross(f, [0.0, 1.0, 0.0]));
        let u = cross(r, f);
        (f, r, u)
    }

    pub fn clip_planes(&self) -> (f32, f32) {
        let far = self.dist + self.radius * 2.0 + 10.0;
        let near = ((self.dist - self.radius) * 0.5).max(self.dist * 0.002).max(0.02);
        (near, far)
    }

    pub fn view(&self) -> Mat4 {
        let (f, r, u) = self.basis();
        let e = self.eye();
        [[r[0], u[0], -f[0], 0.0], [r[1], u[1], -f[1], 0.0], [r[2], u[2], -f[2], 0.0], [-dot(r, e), -dot(u, e), dot(f, e), 1.0]]
    }

    pub fn proj(&self, aspect: f32) -> Mat4 {
        let (near, far) = self.clip_planes();
        let t = 1.0 / (self.fov_y * 0.5).tan();
        let a = far / (near - far);
        [[t / aspect.max(1e-6), 0.0, 0.0, 0.0], [0.0, t, 0.0, 0.0], [0.0, 0.0, a, -1.0], [0.0, 0.0, a * near, 0.0]]
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        mat_mul(&self.proj(aspect), &self.view())
    }

    /// The view-projection that maps pixel (px, py) of a w × h target onto a 1 × 1 one
    /// (the pick pass): the pixel's centre becomes the only sample.
    pub fn pick_view_proj(&self, w: u32, h: u32, px: f32, py: f32) -> Mat4 {
        let cx = (px + 0.5) / w as f32 * 2.0 - 1.0;
        let cy = 1.0 - (py + 0.5) / h as f32 * 2.0;
        // clip' = (W·(x − cx·w), H·(y − cy·w), z, w)
        let pick: Mat4 = [[w as f32, 0.0, 0.0, 0.0], [0.0, h as f32, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [-cx * w as f32, -cy * h as f32, 0.0, 1.0]];
        mat_mul(&pick, &self.view_proj(w as f32 / h as f32))
    }

    /// The ray through the centre of pixel (px, py) of a w × h target.
    pub fn ray(&self, w: u32, h: u32, px: f32, py: f32) -> ([f32; 3], [f32; 3]) {
        let (f, r, u) = self.basis();
        let t = (self.fov_y * 0.5).tan();
        let aspect = w as f32 / h as f32;
        let x = ((px + 0.5) / w as f32 * 2.0 - 1.0) * t * aspect;
        let y = (1.0 - (py + 0.5) / h as f32 * 2.0) * t;
        (self.eye(), normalize(add(f, add(scale(r, x), scale(u, y)))))
    }

    /// Drag to orbit (points of pointer travel).
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.55, 1.55);
    }

    /// Right- or shift-drag to pan: the target moves with the pointer (`view_h` points
    /// tall viewport).
    pub fn pan(&mut self, dx: f32, dy: f32, view_h: f32) {
        let (_, r, u) = self.basis();
        let per_pt = 2.0 * self.dist * (self.fov_y * 0.5).tan() / view_h.max(1.0);
        self.target = add(self.target, add(scale(r, -dx * per_pt), scale(u, dy * per_pt)));
    }

    /// Wheel to zoom (positive = closer).
    pub fn zoom(&mut self, scroll: f32) {
        self.dist = (self.dist * (-scroll * 0.0015).exp()).clamp(0.5, self.radius * 20.0 + 100.0);
    }

    /// Screen position (px) of a world point in a w × h target, None behind the eye.
    pub fn project(&self, w: u32, h: u32, p: [f32; 3]) -> Option<(f32, f32)> {
        let c = transform(&self.view_proj(w as f32 / h as f32), p);
        if c[3] <= 1e-6 {
            return None;
        }
        Some(((c[0] / c[3] * 0.5 + 0.5) * w as f32, (0.5 - c[1] / c[3] * 0.5) * h as f32))
    }
}

/// Slab test: distance along the ray to the box, if it is hit in front of the origin.
pub fn ray_box(o: [f32; 3], d: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> Option<f32> {
    let (mut t0, mut t1) = (0f32, f32::MAX);
    for a in 0..3 {
        if d[a].abs() < 1e-12 {
            if o[a] < lo[a] || o[a] > hi[a] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d[a];
        let (mut ta, mut tb) = ((lo[a] - o[a]) * inv, (hi[a] - o[a]) * inv);
        if ta > tb {
            std::mem::swap(&mut ta, &mut tb);
        }
        t0 = t0.max(ta);
        t1 = t1.min(tb);
        if t0 > t1 {
            return None;
        }
    }
    Some(t0)
}

/// What a CPU ray cast hits: the nearest nonzero cube, and the next-nearest distance
/// (a test ray is only meaningful when the two are well apart).
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    pub cube: CubeId,
    pub t: f32,
    pub next_t: f32,
}

/// The reference ray cast over every nonzero cube of the layout (CPU residues).
pub fn ray_cast(layout: &Layout, mods: &[Modulus], o: [f32; 3], d: [f32; 3]) -> Option<RayHit> {
    let he = layout.half_extents();
    let mut best: Option<RayHit> = None;
    let mut second = f32::MAX;
    for (si, s) in layout.slices.iter().enumerate() {
        let tri = layout.outline(s.m, s.height);
        let lo = [tri[1][0], -s.height - he[1], tri[0][2]];
        let hi = [tri[2][0], -s.height + he[1], tri[1][2]];
        if ray_box(o, d, lo, hi).is_none() {
            continue;
        }
        let md = &mods[si];
        for i in 0..s.rows {
            let n = i * layout.stride;
            for j in 0..=i {
                let k = j * layout.stride;
                let c = cell_center(s.height, n, k);
                let Some(t) = ray_box(o, d, sub(c, he), add(c, he)) else { continue };
                let r = md.binom(n, k);
                if r == 0 {
                    continue;
                }
                match best {
                    Some(b) if t >= b.t => second = second.min(t),
                    _ => {
                        if let Some(b) = best {
                            second = second.min(b.t);
                        }
                        best = Some(RayHit { cube: CubeId { slice: si as u32, n, k, residue: r }, t, next_t: 0.0 });
                    }
                }
            }
        }
    }
    best.map(|b| RayHit { next_t: second, ..b })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sweep::ModMode;

    #[test]
    fn selections_follow_the_mode() {
        let mut c = TetraConfig::default();
        c.sweep.start = 2;
        c.sweep.end = 20;
        assert_eq!(selected_moduli(&c.sweep), (2..=20).collect::<Vec<_>>());
        c.sweep.mode = ModMode::Prime;
        assert_eq!(selected_moduli(&c.sweep), vec![2, 3, 5, 7, 11, 13, 17, 19]);
        c.sweep.mode = ModMode::Composite;
        assert_eq!(selected_moduli(&c.sweep), vec![4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]);
        c.sweep.end = 99_999;
        assert_eq!(*selected_moduli(&c.sweep).last().unwrap(), TETRA_MAX_M);
        // a reversed range is the same range, and interval walks start at its low end
        c.sweep.mode = ModMode::Interval;
        c.sweep.step = 5;
        c.sweep.start = 40;
        c.sweep.end = 10;
        assert_eq!(selected_moduli(&c.sweep), vec![10, 15, 20, 25, 30, 35, 40]);
    }

    #[test]
    fn true_spacing_leaves_prime_gaps() {
        let l = Layout::new(&[2, 3, 5, 7, 11], Spacing::True, u32::MAX);
        assert_eq!(l.slices.iter().map(|s| s.height).collect::<Vec<_>>(), vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        assert_eq!(l.gaps(), vec![(3, 5), (5, 7), (7, 11)]);
        assert_eq!(l.missing(), vec![4, 6, 8, 9, 10]);
        let p = Layout::new(&[2, 3, 5, 7, 11], Spacing::Packed, u32::MAX);
        assert_eq!(p.slices.iter().map(|s| s.height).collect::<Vec<_>>(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(p.gaps().is_empty() && p.missing().is_empty());
    }

    #[test]
    fn the_stride_keeps_the_budget() {
        let mods: Vec<u32> = (2..=400).collect();
        let full: u64 = mods.iter().map(|&m| strided_cells(m, 1)).sum();
        assert!(full > 10_000_000);
        for budget in [100_000u32, 1_000_000, 4_000_000] {
            let s = stride_for(&mods, budget);
            let l = Layout::new(&mods, Spacing::True, budget);
            assert_eq!(l.stride, s);
            assert!(l.total as u64 <= budget as u64, "{budget}: {} cells", l.total);
            let looser: u64 = mods.iter().map(|&m| strided_cells(m, s - 1)).sum();
            assert!(s == 1 || looser > budget as u64, "stride {s} is not the smallest");
        }
        assert_eq!(stride_for(&mods, u32::MAX), 1);
    }

    #[test]
    fn the_cell_enumeration_covers_every_strided_cell() {
        for budget in [u32::MAX, 120] {
            let l = Layout::new(&[2, 5, 9, 30], Spacing::True, budget);
            let stride = l.stride;
            assert_eq!(stride > 1, budget == 120, "stride {stride}");
            let mut seen = std::collections::HashSet::new();
            for g in 0..l.total {
                let (si, n, k) = l.cell(g).unwrap();
                let m = l.slices[si].m;
                assert!(n < m && k <= n && n.is_multiple_of(stride) && k.is_multiple_of(stride));
                assert!(seen.insert((si, n, k)));
            }
            let expect: u64 = l.slices.iter().map(|s| strided_cells(s.m, stride)).sum();
            assert_eq!(seen.len() as u64, expect);
            assert!(l.cell(l.total).is_none());
        }
    }

    #[test]
    fn the_camera_projects_and_casts_consistently() {
        let l = Layout::new(&(2..=30).collect::<Vec<_>>(), Spacing::True, u32::MAX);
        let mut cam = Camera::framing(l.bounds());
        cam.orbit(40.0, -10.0);
        let (w, h) = (640, 400);
        for p in [[0.0, -10.0, 3.0], [5.0, -25.0, 20.0], [-7.0, -2.0, 1.0]] {
            let (px, py) = cam.project(w, h, p).unwrap();
            let (o, d) = cam.ray(w, h, px - 0.5, py - 0.5);
            // the ray through the projected point passes through the point
            let v = sub(p, o);
            let along = dot(v, d);
            let off = sub(v, scale(d, along));
            assert!(dot(off, off).sqrt() < 1e-3 * along, "{p:?}: miss by {}", dot(off, off).sqrt());
        }
        // the pick projection puts the pixel's centre at the centre of clip space
        let p = [5.0, -25.0, 20.0];
        let (px, py) = cam.project(w, h, p).unwrap();
        let c = transform(&cam.pick_view_proj(w, h, px - 0.5, py - 0.5), p);
        assert!((c[0] / c[3]).abs() < 1e-2 && (c[1] / c[3]).abs() < 1e-2, "{c:?}");
    }

    #[test]
    fn cube_words_round_trip() {
        let w = pack_cube(4095, 4095, 17, 4094);
        assert_eq!(CubeId::unpack(w), CubeId { slice: 4095, n: 4095, k: 17, residue: 4094 });
        let w = [w[0], w[1] | 0x8000_0000];
        assert_eq!(CubeId::unpack(w).residue, 4094);
    }
}
