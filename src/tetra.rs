//! The tetra stack: Pascal's triangle mod m for many moduli at once, one horizontal
//! slice per modulus. Slice m holds rows 0..m−1 of the triangle mod m, so slices grow
//! linearly with m and the stack is a tetrahedron (mod 2, two rows, is the apex).
//!
//! Cell (m, n, k) sits at x = k − n/2, z = n·√3/2 (the 2D brick layout laid flat),
//! shifted by the slice's offset, and y = −height(m). A slice of side s = m − 1 sits
//! s·√(2/3) below the apex, the height of a regular tetrahedron of edge s, so with the
//! slices' centroids on one vertical axis the stack is equilateral ([`Shape::Regular`]).
//! The other shapes anchor a corner or the middle of an edge instead, which makes one
//! or two faces vertical (flush). With TRUE spacing the height follows m itself, so
//! skipped moduli leave gaps and the edges stay straight; PACKED uses the slice's index,
//! with no gaps, and the sides curve because slice size no longer tracks height.
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
/// Cube half extent per unit of stride: the same on every axis, so a cube is a cube.
/// The tightest pitch is between slices (√(2/3) ≈ 0.816 per modulus), so 0.78 wide
/// leaves a sliver there and more room along the rows (1) and between them (√3/2).
pub const CUBE_HALF: f32 = 0.39;
/// √(2/3): the height of a regular tetrahedron per unit of edge, so the vertical step
/// from one modulus to the next.
pub const LEVEL: f32 = 0.816_496_6;
/// Cube-count budgets offered in the panel; `u32::MAX` = every cell, up to what the GPU
/// holds (the default: nothing is left out unless the range is too big for it).
pub const BUDGETS: [u32; 5] = [1_000_000, 4_000_000, 16_000_000, 32_000_000, u32::MAX];
/// The preview tile's budget: it turns all the time, so it draws a light stack.
pub const TILE_BUDGET: u32 = 1_000_000;

/// How the slices line up: their centroids (a regular tetrahedron), one corner (two
/// faces vertical, meeting in a vertical edge) or the middle of one edge (one face
/// vertical). The corners and edges are those of the 2D triangle: the apex (row 0), the
/// left and right ends of the last row; the front edge is the last row (the face the
/// default camera looks at, which the front-face view lays flat).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Shape {
    #[default]
    Regular,
    CornerApex,
    CornerLeft,
    CornerRight,
    FaceFront,
    FaceLeft,
    FaceRight,
}

impl Shape {
    pub const ALL: [Shape; 7] = [Shape::Regular, Shape::CornerApex, Shape::CornerLeft, Shape::CornerRight, Shape::FaceFront, Shape::FaceLeft, Shape::FaceRight];

    pub fn label(self) -> &'static str {
        match self {
            Shape::Regular => "equilateral",
            Shape::CornerApex => "flush corner: apex",
            Shape::CornerLeft => "flush corner: left",
            Shape::CornerRight => "flush corner: right",
            Shape::FaceFront => "flush face: front",
            Shape::FaceLeft => "flush face: left",
            Shape::FaceRight => "flush face: right",
        }
    }

    pub fn note(self) -> &'static str {
        match self {
            Shape::Regular => "A regular tetrahedron: every slice centred on one vertical axis, all four faces equilateral and equally tilted.",
            Shape::CornerApex => "The row-0 corners of all slices on one vertical line: the left and right faces stand vertical and meet in a vertical edge.",
            Shape::CornerLeft => "The left ends of the last rows on one vertical line: the left and front faces stand vertical.",
            Shape::CornerRight => "The right ends of the last rows on one vertical line: the right and front faces stand vertical.",
            Shape::FaceFront => "The middles of the last rows on one vertical line: the front face (the last rows) stands vertical.",
            Shape::FaceLeft => "The middles of the left edges (k = 0) on one vertical line: the left face stands vertical.",
            Shape::FaceRight => "The middles of the right edges (k = n) on one vertical line: the right face stands vertical.",
        }
    }

    /// The (x, z) shift of a slice whose drawn last row is row `side` (in the 2D layout
    /// its apex is at the origin and that row at z = side·√3/2): the anchored point moves
    /// to the axis. `side` is m − 1, or the last strided row under a cell stride, so the
    /// drawn corner cubes are the ones on the axis.
    pub fn offset(self, side: u32) -> (f32, f32) {
        let s = side as f32;
        let back = s * SQRT3_2;
        match self {
            Shape::Regular => (0.0, -back * 2.0 / 3.0),
            Shape::CornerApex => (0.0, 0.0),
            Shape::CornerLeft => (s / 2.0, -back),
            Shape::CornerRight => (-s / 2.0, -back),
            Shape::FaceFront => (0.0, -back),
            Shape::FaceLeft => (s / 4.0, -back / 2.0),
            Shape::FaceRight => (-s / 4.0, -back / 2.0),
        }
    }
}

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

/// Everything the 3D space's options panel sets (and the main panel's tile toggles).
#[derive(Clone, Debug, PartialEq)]
pub struct TetraConfig {
    /// The selection: `start..=end` filtered by the mode (all, primes, composites, even,
    /// odd, interval / sequences). Only the selection fields are used, not playback.
    pub sweep: Sweep,
    pub spacing: Spacing,
    pub shape: Shape,
    /// What the 3D space shows: the stack, its front face, or the diamond.
    pub view: View,
    /// Peeling the stack's sides away, layer by layer (the stack view only).
    pub peel: Peel,
    /// The options panel is open in the 3D space.
    pub options: bool,
    /// Zero cells as faint ghost cubes.
    pub ghosts: bool,
    /// Hover highlight radius in cell units (0 = the hovered cube only).
    pub brush: f32,
    /// Highlight the whole hovered slice instead of a brush.
    pub color_slice: bool,
    /// Most cubes drawn (u32::MAX = all the GPU holds); a larger selection gets a stride.
    pub budget: u32,
    /// The face and the diamond draw their zeros as dark cubes: no gaps.
    pub solid_zeros: bool,
    /// The preview tile turns slowly.
    pub spin: bool,
    /// Show the preview tile on the 2D canvas.
    pub tile: bool,
}

impl Default for TetraConfig {
    fn default() -> Self {
        let mut sweep = Sweep::default();
        sweep.end = 48;
        TetraConfig { sweep, spacing: Spacing::True, shape: Shape::Regular, view: View::Stack, peel: Peel::default(), options: true, ghosts: false, brush: 2.0, color_slice: false, budget: u32::MAX, solid_zeros: true, spin: true, tile: true }
    }
}

impl TetraConfig {
    /// What the geometry of `view` depends on (the scene is rebuilt when it changes).
    pub fn scene_key(&self, view: View) -> SceneKey {
        let s = &self.sweep;
        (s.mode, s.prime_order, s.seq, s.step, s.start, s.end, self.spacing, self.budget, self.shape, view)
    }
}

pub type SceneKey = (crate::sweep::ModMode, crate::sweep::PrimeOrder, crate::sweep::Seq, u32, u32, u32, Spacing, u32, Shape, View);

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

/// The last drawn row of slice m at stride s (m − 1 at stride 1).
pub fn strided_side(m: u32, stride: u32) -> u32 {
    (strided_rows(m, stride) - 1) * stride
}

/// Cells of a slice at stride s: row i·s holds cells 0, s, …, i·s.
pub fn strided_cells(m: u32, stride: u32) -> u64 {
    let r = strided_rows(m, stride) as u64;
    r * (r + 1) / 2
}

/// Cells of row m − 1 of slice m at stride s (cells 0, s, … of the last row).
pub fn strided_row_cells(m: u32, stride: u32) -> u64 {
    ((m - 1) / stride + 1) as u64
}

/// How far below the top selection index i (modulus m) sits: a stack slice's height,
/// or a face row's depth (the face is equilateral, so its rows are √3/2 apart).
pub fn level_depth(view: View, spacing: Spacing, i: usize, m: u32) -> f32 {
    match (view, spacing) {
        (View::Stack, Spacing::True) => true_height(m),
        (View::Stack, Spacing::Packed) => i as f32 * LEVEL,
        (_, Spacing::True) => (m - 1) as f32 * SQRT3_2,
        (_, Spacing::Packed) => i as f32 * SQRT3_2,
    }
}

/// The selection indices drawn at stride s. Walking down from the apex (always drawn, so
/// the top of the stack does not change when the range's end does), a slice or face row
/// is kept when it sits at least one cube (s cells) below the last one kept, so cubes
/// never overlap and a sparse selection whose slices are far apart keeps all of them.
/// The base (the stack's base slice, the diamond's base) is always drawn too: when it is
/// less than a cube below the last one kept, it takes that one's place.
pub fn drawn(view: View, mods: &[u32], spacing: Spacing, stride: u32) -> Vec<usize> {
    let len = mods.len();
    if len == 0 {
        return Vec::new();
    }
    if stride <= 1 {
        // neighbouring moduli are at least √(2/3) apart, more than a cube (0.78)
        return (0..len).collect();
    }
    let need = 2.0 * CUBE_HALF * stride as f32 - 1e-4;
    let depth = |i: usize| level_depth(view, spacing, i, mods[i]);
    let mut out = vec![0];
    let mut last = depth(0);
    for i in 1..len {
        if depth(i) - last >= need {
            out.push(i);
            last = depth(i);
        }
    }
    // the base was not kept, so it is less than a cube below the last kept one, which
    // is a cube below the one before it: the base can take its place
    if *out.last().unwrap() != len - 1 {
        *out.last_mut().unwrap() = len - 1;
    }
    out
}

/// Cells a view draws at stride s.
pub fn view_cells(view: View, mods: &[u32], spacing: Spacing, stride: u32) -> u64 {
    let len = mods.len();
    let rows = drawn(view, mods, spacing, stride).into_iter();
    match view {
        View::Stack => rows.map(|i| strided_cells(mods[i], stride)).sum(),
        View::Face => rows.map(|i| strided_row_cells(mods[i], stride)).sum(),
        View::Diamond => rows.map(|i| if i + 1 == len { strided_cells(mods[i], stride) } else { strided_row_cells(mods[i], stride) }).sum(),
    }
}

/// The smallest stride that keeps a view within `budget` cubes. A stride s draws slices
/// (or face rows) at least s cells apart and every s-th row and cell in them, so the
/// cubes stay cubes, s cells wide.
pub fn view_stride(view: View, mods: &[u32], spacing: Spacing, budget: u32) -> u32 {
    let budget = budget.max(1) as u64;
    let view_cells = |view, mods, s| view_cells(view, mods, spacing, s);
    let full = view_cells(view, mods, 1);
    if full <= budget {
        return 1;
    }
    // a stack shrinks like 1/s³, a face like 1/s²: start a little below the estimate
    let power = if view == View::Stack { 1.0 / 3.0 } else { 0.5 };
    let mut s = (((full as f64 / budget as f64).powf(power) * 0.8).floor() as u32).max(1);
    while view_cells(view, mods, s) > budget {
        s += 1;
    }
    while s > 1 && view_cells(view, mods, s - 1) <= budget {
        s -= 1;
    }
    s
}

/// The stack's stride in true spacing (for callers that only know the stack).
pub fn stride_for(mods: &[u32], budget: u32) -> u32 {
    view_stride(View::Stack, mods, Spacing::True, budget)
}

/// A side of the stack that peels: the three sides of every triangle (as faces of the
/// tetrahedron) and its two ends, the apex and the base.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Side {
    /// k = 0 in every slice (all 1s).
    #[default]
    Left,
    /// k = n in every slice (all 1s).
    Right,
    /// The last rows (the front face).
    Front,
    /// Slices from the apex down.
    Top,
    /// Slices from the base up.
    Base,
}

impl Side {
    pub const ALL: [Side; 5] = [Side::Left, Side::Right, Side::Front, Side::Top, Side::Base];

    pub fn name(self) -> &'static str {
        match self {
            Side::Left => "left",
            Side::Right => "right",
            Side::Front => "front",
            Side::Top => "top",
            Side::Base => "base",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Side::Left => "left (k = 0)",
            Side::Right => "right (k = n)",
            Side::Front => "front (last rows)",
            Side::Top => "top (from the apex)",
            Side::Base => "base (from the bottom)",
        }
    }
}

/// Peeling: which sides, and how many layers are off. A layer is one row or cell of the
/// stride on the triangles' sides (k, n − k, last row − n) and one drawn slice at the top
/// or the base, so a stride never makes a layer peel nothing.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Peel {
    /// The sides that peel, in `Side::ALL` order; the unchecked ones stay.
    pub sides: [bool; 5],
    pub layers: u32,
    /// The animation: a layer comes off every 1/speed s while playing, and when the
    /// last cube is gone it starts again from nothing peeled.
    pub playing: bool,
    pub speed: f32,
}

impl Default for Peel {
    fn default() -> Self {
        Peel { sides: [false; 5], layers: 0, playing: false, speed: 6.0 }
    }
}

impl Peel {
    /// Whether any side peels.
    pub fn active(&self) -> bool {
        self.sides.iter().any(|&on| on)
    }

    /// Layers off each side, in `Side::ALL` order (0 = untouched).
    pub fn depths(&self) -> [u32; 5] {
        self.sides.map(|on| if on { self.layers } else { 0 })
    }

    /// The checked sides.
    pub fn checked(&self) -> impl Iterator<Item = Side> + '_ {
        Side::ALL.into_iter().zip(self.sides).filter(|&(_, on)| on).map(|(sd, _)| sd)
    }
}

/// A snapped camera direction for the stack: straight down at the apex, straight up at
/// the base, or square onto one of the three sides (the left, right and front faces).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Snap {
    Top,
    Bottom,
    Left,
    Right,
    Front,
}

impl Snap {
    pub const ALL: [Snap; 5] = [Snap::Top, Snap::Bottom, Snap::Left, Snap::Right, Snap::Front];

    pub fn label(self) -> &'static str {
        match self {
            Snap::Top => "top",
            Snap::Bottom => "base",
            Snap::Left => "left",
            Snap::Right => "right",
            Snap::Front => "front",
        }
    }
}

/// What the 3D space shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum View {
    /// The tetrahedron: one slice per modulus.
    #[default]
    Stack,
    /// The face the slices' last rows make, laid flat: row m − 1 of the triangle mod m,
    /// for every m, which is again a triangle.
    Face,
    /// The face with the base slice (the last modulus, all its rows) hinged down along
    /// their shared row: the tetrahedron's net of two faces, a diamond.
    Diamond,
}

impl View {
    pub const ALL: [View; 3] = [View::Stack, View::Face, View::Diamond];

    pub fn label(self) -> &'static str {
        match self {
            View::Stack => "stack",
            View::Face => "front face",
            View::Diamond => "diamond",
        }
    }

    pub fn note(self) -> &'static str {
        match self {
            View::Stack => "The tetrahedron: one slice per modulus, rows 0..m−1 of the triangle mod m.",
            View::Face => "The last rows of all the slices, laid flat: row m − 1 of the triangle mod m for every m. They form a triangle again, with row r = C(r, k) mod (r + 1).",
            View::Diamond => "The front face with the base slice (the last modulus, every row) hinged down along the row they share: two triangles, a diamond. The hinge swings open when the view is chosen.",
        }
    }

    /// Seen straight on, as a picture (an orthographic camera that pans and zooms).
    pub fn flat(self) -> bool {
        self != View::Stack
    }

    pub fn next(self) -> View {
        match self {
            View::Stack => View::Face,
            View::Face => View::Diamond,
            View::Diamond => View::Stack,
        }
    }
}

/// The angle the diamond's base slice starts folded at, out of the face's plane: the
/// tetrahedron's own fold, 180° minus its dihedral angle arccos(1/3).
pub const FOLD: f32 = std::f32::consts::PI - 1.230_959_4;

/// Which cells a slice holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SliceKind {
    /// Rows 0, s, 2s, … of the triangle mod m, each with cells 0, s, …, n.
    Triangle,
    /// Row n only (the face's row m − 1), cells 0, s, …
    Row(u32),
}

/// One slice as laid out. Cell (n, k) of it sits at `origin + n·nvec + k·(1, 0, 0)`, so
/// one mapping covers a slice in the stack, a row of the face and the hinged base.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slice {
    pub m: u32,
    pub kind: SliceKind,
    /// Strided rows (1 for a row) and cells.
    pub rows: u32,
    pub cells: u32,
    /// First cell of this slice in the cell enumeration.
    pub cell_base: u32,
    /// A triangle's first drawn row: its rows are n0, n0 + s, …, so the diamond's base
    /// ends exactly on row m − 1, the hinge it shares with the face (0 elsewhere).
    pub n0: u32,
    pub origin: [f32; 3],
    pub nvec: [f32; 3],
    /// Its index in the whole selection.
    pub index: usize,
}

impl Slice {
    /// How far below the top it sits (the stack's slice height, a face row's depth).
    pub fn height(&self) -> f32 {
        -self.origin[1]
    }

    /// Its last drawn row.
    pub fn last_row(&self, stride: u32) -> u32 {
        match self.kind {
            SliceKind::Triangle => self.n0 + (self.rows - 1) * stride,
            SliceKind::Row(n) => n,
        }
    }

    /// The last drawn cell of its last row.
    pub fn last_k(&self, stride: u32) -> u32 {
        match self.kind {
            SliceKind::Triangle => (self.rows - 1) * stride,
            SliceKind::Row(_) => (self.cells - 1) * stride,
        }
    }
}

/// The slices of one view at one level of detail.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub view: View,
    /// The whole selection, drawn or not.
    pub mods: Vec<u32>,
    pub slices: Vec<Slice>,
    pub spacing: Spacing,
    pub shape: Shape,
    /// Every `stride`-th slice, row and cell is drawn (1 = all).
    pub stride: u32,
    /// Cells in the enumeration (nonzero cubes + zeros).
    pub total: u32,
    /// Cells of the view at stride 1.
    pub full_total: u64,
    /// The diamond's hinge: the base slice's angle out of the face's plane (0 = flat).
    pub fold: f32,
}

/// The stack's row direction: rows run back along z, half a cell left per row.
const STACK_N: [f32; 3] = [-0.5, 0.0, SQRT3_2];

impl Layout {
    pub fn new(mods: &[u32], spacing: Spacing, budget: u32) -> Layout {
        Layout::with_shape(mods, spacing, Shape::Regular, budget)
    }

    pub fn with_shape(mods: &[u32], spacing: Spacing, shape: Shape, budget: u32) -> Layout {
        Layout::build(View::Stack, mods, spacing, shape, budget)
    }

    /// The slices of `view`; the diamond starts folded (see [`Layout::set_fold`]).
    pub fn build(view: View, mods: &[u32], spacing: Spacing, shape: Shape, budget: u32) -> Layout {
        let stride = view_stride(view, mods, spacing, budget);
        let len = mods.len();
        let mut l = Layout { view, mods: mods.to_vec(), slices: Vec::new(), spacing, shape, stride, total: 0, full_total: view_cells(view, mods, spacing, 1), fold: 0.0 };
        let mut base = 0u32;
        for i in drawn(view, mods, spacing, stride) {
            let m = mods[i];
            // the diamond's base keeps its last row (the hinge) under a stride
            let n0 = if view == View::Diamond && i + 1 == len { (m - 1) % stride } else { 0 };
            let (kind, rows, cells, origin, nvec) = match view {
                View::Stack => {
                    let h = l.stack_height(i, m);
                    let (ox, oz) = shape.offset(strided_side(m, stride));
                    (SliceKind::Triangle, strided_rows(m, stride), strided_cells(m, stride), [ox, -h, oz], STACK_N)
                }
                View::Diamond if i + 1 == len => (SliceKind::Triangle, strided_rows(m, stride), strided_cells(m, stride), [0.0; 3], [0.0; 3]),
                View::Face | View::Diamond => (SliceKind::Row(m - 1), 1, strided_row_cells(m, stride), [0.0, -l.face_y(i, m), 0.0], [-0.5, 0.0, 0.0]),
            };
            l.slices.push(Slice { m, kind, rows, cells: cells as u32, cell_base: base, n0, origin, nvec, index: i });
            base += cells as u32;
        }
        l.total = base;
        if view == View::Diamond {
            l.set_fold(FOLD);
        }
        l
    }

    /// The stack's slice height of selection index i (modulus m).
    fn stack_height(&self, i: usize, m: u32) -> f32 {
        level_depth(View::Stack, self.spacing, i, m)
    }

    /// How far below the face's apex row m − 1 lies: the face is an equilateral
    /// triangle, so rows are √3/2 apart.
    fn face_y(&self, i: usize, m: u32) -> f32 {
        level_depth(View::Face, self.spacing, i, m)
    }

    /// Swing the diamond's base slice about the row it shares with the face: 0 lays it
    /// flat below the face, [`FOLD`] folds it back to where it sits in the tetrahedron.
    pub fn set_fold(&mut self, angle: f32) {
        if self.view != View::Diamond {
            return;
        }
        let Some(i) = self.slices.len().checked_sub(1) else { return };
        let s = self.slices[i];
        let m = s.m;
        let y_edge = -self.face_y(s.index, m);
        // the hinge is row m − 1 itself (the base's rows end on it at any stride)
        let n_last = s.last_row(self.stride) as f32;
        let (sn, cs) = angle.sin_cos();
        // row n is (n_last − n)·√3/2 from the hinge, along (−cos φ, −sin φ) in (y, z)
        let nvec = [-0.5, SQRT3_2 * cs, SQRT3_2 * sn];
        let origin = [0.0, y_edge - n_last * SQRT3_2 * cs, -n_last * SQRT3_2 * sn];
        self.slices[i].origin = origin;
        self.slices[i].nvec = nvec;
        self.fold = angle;
    }

    /// The same layout with the diamond lying flat (what the camera frames).
    pub fn at_rest(&self) -> Layout {
        let mut l = self.clone();
        l.set_fold(0.0);
        l
    }

    /// Cube half extents (x, y, z) in world units: a cube, s cells wide.
    pub fn half_extents(&self) -> [f32; 3] {
        [CUBE_HALF * self.stride as f32; 3]
    }

    /// The centres of a slice's corner cubes.
    pub fn slice_corners(&self, si: usize) -> Vec<[f32; 3]> {
        let s = &self.slices[si];
        let (n_last, k_last) = (s.last_row(self.stride), s.last_k(self.stride));
        match s.kind {
            SliceKind::Triangle => vec![self.center(si, s.n0, 0), self.center(si, n_last, 0), self.center(si, n_last, k_last)],
            SliceKind::Row(n) => vec![self.center(si, n, 0), self.center(si, n, k_last)],
        }
    }

    /// World bounds of the cubes.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for si in 0..self.slices.len() {
            for p in self.slice_corners(si) {
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
        if let SliceKind::Row(n) = s.kind {
            return Some((si, n, l as u32 * self.stride));
        }
        let mut i = (((8 * l + 1) as f64).sqrt() as u64 - 1) / 2;
        while i * (i + 1) / 2 > l {
            i -= 1;
        }
        while (i + 1) * (i + 2) / 2 <= l {
            i += 1;
        }
        let j = l - i * (i + 1) / 2;
        Some((si, s.n0 + i as u32 * self.stride, j as u32 * self.stride))
    }

    /// Every cell of slice si, as (n, k).
    pub fn slice_cells(&self, si: usize) -> Vec<(u32, u32)> {
        let s = &self.slices[si];
        match s.kind {
            SliceKind::Row(n) => (0..s.cells).map(|j| (n, j * self.stride)).collect(),
            SliceKind::Triangle => (0..s.rows).flat_map(|i| (0..=i).map(move |j| (i, j))).map(|(i, j)| (s.n0 + i * self.stride, j * self.stride)).collect(),
        }
    }

    /// Centre of cube (slice, n, k).
    pub fn center(&self, slice: usize, n: u32, k: u32) -> [f32; 3] {
        let s = &self.slices[slice];
        add(add(s.origin, scale(s.nvec, n as f32)), [k as f32, 0.0, 0.0])
    }

    /// Whether cell (slice, n, k) is peeled away (the stack only), as the shaders decide.
    pub fn peeled(&self, si: usize, n: u32, k: u32, depths: [u32; 5]) -> bool {
        if self.view != View::Stack {
            return false;
        }
        let s = &self.slices[si];
        let st = self.stride;
        let last = s.last_row(st);
        let n_slices = self.slices.len() as u32;
        let si = si as u32;
        (depths[0] > 0 && k < depths[0] * st)
            || (depths[1] > 0 && n - k < depths[1] * st)
            || (depths[2] > 0 && last - n < depths[2] * st)
            || (depths[3] > 0 && si < depths[3])
            || (depths[4] > 0 && n_slices - 1 - si < depths[4])
    }

    /// The fewest layers that peel the whole stack away with `peel`'s sides.
    pub fn peel_extent(&self, peel: &Peel) -> u32 {
        let mut p = *peel;
        p.layers = 1;
        let unit = p.depths();
        if self.view != View::Stack || unit.iter().all(|&d| d == 0) {
            return 0;
        }
        let st = self.stride;
        let n_slices = self.slices.len() as u32;
        let mut extent = 0u32;
        for (si, s) in self.slices.iter().enumerate() {
            let last = s.last_row(st);
            let si = si as u32;
            // a cell survives L layers while every peeled side is at least L layers away
            let slice_d = [si, n_slices - 1 - si];
            let slice_min = (3..5).filter(|&i| unit[i] > 0).map(|i| slice_d[i - 3] + 1).min().unwrap_or(u32::MAX);
            if slice_min <= extent {
                continue;
            }
            for (n, k) in self.slice_cells(si as usize) {
                let cell_d = [k / st, (n - k) / st, (last - n) / st];
                let cell_min = (0..3).filter(|&i| unit[i] > 0).map(|i| cell_d[i] + 1).min().unwrap_or(u32::MAX);
                extent = extent.max(cell_min.min(slice_min));
            }
        }
        extent
    }

    /// A camera looking square onto a side of the stack (or straight down / up),
    /// framing the whole stack, from the stack's own corners so every shape works.
    pub fn snap_camera(&self, snap: Snap) -> Camera {
        let mut cam = Camera::framing(self.bounds());
        let (Some(first), Some(last)) = (self.slices.first(), self.slices.last()) else { return cam };
        let apex = self.corners(first.m, first.height())[0];
        let [b0, bl, br] = self.corners(last.m, last.height());
        // outward normal of the plane through a, b, c, away from the opposite corner
        let outward = |a: [f32; 3], b: [f32; 3], c: [f32; 3], away: [f32; 3]| {
            let mut nrm = normalize(cross(sub(b, a), sub(c, a)));
            if dot(nrm, sub(away, a)) > 0.0 {
                nrm = scale(nrm, -1.0);
            }
            nrm
        };
        let dir = match snap {
            Snap::Top => [0.0, 1.0, 0.0],
            Snap::Bottom => [0.0, -1.0, 0.0],
            Snap::Left => outward(apex, b0, bl, br),
            Snap::Right => outward(apex, b0, br, bl),
            Snap::Front => outward(apex, bl, br, b0),
        };
        // eye = target + dir·dist, with eye = target + (sin yaw·cos pitch, sin pitch, −cos yaw·cos pitch)
        cam.pitch = dir[1].clamp(-1.0, 1.0).asin().clamp(-1.55, 1.55);
        if dir[0].abs() + dir[2].abs() > 1e-4 {
            cam.yaw = dir[0].atan2(-dir[2]);
        }
        cam
    }

    /// The stack height a modulus has (or would have, skipped) in this spacing.
    pub fn height_of(&self, m: u32) -> f32 {
        match self.spacing {
            Spacing::True => true_height(m),
            Spacing::Packed => self.mods.binary_search(&m).map_or(0.0, |i| i as f32 * LEVEL),
        }
    }

    /// The stack's corner cube centres for a slice of modulus m at height h (drawn or
    /// skipped): the apex, the left and the right end of its last (strided) row.
    pub fn corners(&self, m: u32, h: f32) -> [[f32; 3]; 3] {
        let n_last = strided_side(m, self.stride);
        let (ox, oz) = self.shape.offset(n_last);
        [cell_center(h, ox, oz, 0, 0), cell_center(h, ox, oz, n_last, 0), cell_center(h, ox, oz, n_last, n_last)]
    }

    /// The moduli between the first and last of the selection that it skips (drawn as
    /// dashed outlines in the stack, true spacing). Slices left out by the level of
    /// detail are not skipped.
    pub fn missing(&self) -> Vec<u32> {
        if self.view != View::Stack || self.spacing != Spacing::True || self.mods.len() < 2 {
            return Vec::new();
        }
        self.mods.windows(2).flat_map(|w| (w[0] + 1)..w[1]).collect()
    }

    /// Consecutive moduli of the selection more than one apart: (upper m, lower m),
    /// labeled with the gap `lower − upper` in the stack, true spacing.
    pub fn gaps(&self) -> Vec<(u32, u32)> {
        if self.view != View::Stack || self.spacing != Spacing::True {
            return Vec::new();
        }
        self.mods.windows(2).filter(|w| w[1] - w[0] > 1).map(|w| (w[0], w[1])).collect()
    }

    /// The outline of a stack slice of modulus m at height h: the triangle around its
    /// cubes, its edges pushed out past them.
    pub fn outline(&self, m: u32, h: f32) -> [[f32; 3]; 3] {
        let [a, l, r] = self.corners(m, h);
        // an edge moves out by d when the corners move 2d away from the centroid
        let d = 2.0 * (self.half_extents()[0] + 0.3);
        [add(a, [0.0, 0.0, -d]), add(l, [-d * SQRT3_2, 0.0, d * 0.5]), add(r, [d * SQRT3_2, 0.0, d * 0.5])]
    }
}

/// The height of modulus m in true spacing: s·√(2/3) for side s = m − 1.
pub fn true_height(m: u32) -> f32 {
    m.saturating_sub(1) as f32 * LEVEL
}

/// Centre of cell (n, k) of a stack slice at height h shifted by (ox, oz).
pub fn cell_center(h: f32, ox: f32, oz: f32, n: u32, k: u32) -> [f32; 3] {
    [k as f32 - n as f32 / 2.0 + ox, -h, n as f32 * SQRT3_2 + oz]
}

/// Whether row m − 1 of the triangle mod m alternates 1, m − 1, 1, … that is,
/// C(m − 1, k) ≡ (−1)^k (mod m) for every k. It does exactly when m is prime.
pub fn face_row_alternates(md: &Modulus) -> bool {
    let m = md.m;
    (0..m).all(|k| md.binom(m - 1, k) == if k % 2 == 0 { 1 % m } else { m - 1 })
}

/// Zero cells in row m − 1 of the triangle mod m.
pub fn face_row_zeros(md: &Modulus) -> u32 {
    (0..md.m).filter(|&k| md.binom(md.m - 1, k) == 0).count() as u32
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
    /// Orthographic half height in world units (the flat views); 0 = perspective.
    pub ortho: f32,
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
        Camera { target, yaw: 2.5, pitch: 0.38, dist, fov_y, radius, ortho: 0.0 }
    }

    /// A flat view: straight at the z = 0 plane from the front (+z, x to the right),
    /// orthographic, the bounds filling a w × h window (px) between `top` and `bottom`
    /// px left free for the overlays (the HUD above, the 2D thumbnail below).
    pub fn flat(bounds: ([f32; 3], [f32; 3]), w: f32, h: f32, top: f32, bottom: f32) -> Camera {
        let (lo, hi) = bounds;
        let (w, h) = (w.max(1.0), h.max(1.0));
        let radius = (dot(sub(hi, lo), sub(hi, lo)).sqrt() * 0.5).max(1.0);
        let avail = (h - top - bottom).max(h * 0.3);
        let (content_w, content_h) = (hi[0] - lo[0] + 2.0, hi[1] - lo[1] + 2.0);
        // world units per pixel: the content fits the free height and the width
        let per_px = (content_h / avail).max(content_w / (w * 0.96));
        let half_h = per_px * h / 2.0;
        // the content's centre sits in the middle of the free band, (top − bottom)/2 px
        // below the window's centre (insets scaled down if they had to give way)
        let (top, bottom) = (top.max(0.0), bottom.max(0.0));
        let give = if top + bottom > 0.0 { ((h - avail) / (top + bottom)).min(1.0) } else { 0.0 };
        let shift = (top - bottom) * give / 2.0 * per_px;
        let target = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0 + shift, 0.0];
        Camera { target, yaw: std::f32::consts::PI, pitch: 0.0, dist: radius * 2.0 + 10.0, fov_y: 40f32.to_radians(), radius, ortho: half_h }
    }

    pub fn eye(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        // yaw 0 looks along +z from the −z side (behind the apex rows); the default yaw
        // looks at the last rows' side, the front face
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
        if self.ortho > 0.0 {
            return ((self.dist - self.radius * 2.0 - 5.0).max(0.01), far);
        }
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
        if self.ortho > 0.0 {
            // view z in [−near, −far] → depth [0, 1], w = 1
            let (sx, sy) = (1.0 / (self.ortho * aspect.max(1e-6)), 1.0 / self.ortho);
            let a = 1.0 / (near - far);
            return [[sx, 0.0, 0.0, 0.0], [0.0, sy, 0.0, 0.0], [0.0, 0.0, a, 0.0], [0.0, 0.0, a * near, 1.0]];
        }
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
        let aspect = w as f32 / h as f32;
        let nx = (px + 0.5) / w as f32 * 2.0 - 1.0;
        let ny = 1.0 - (py + 0.5) / h as f32 * 2.0;
        if self.ortho > 0.0 {
            let o = add(self.eye(), add(scale(r, nx * self.ortho * aspect), scale(u, ny * self.ortho)));
            return (o, f);
        }
        let t = (self.fov_y * 0.5).tan();
        (self.eye(), normalize(add(f, add(scale(r, nx * t * aspect), scale(u, ny * t)))))
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
        let span = if self.ortho > 0.0 { self.ortho } else { self.dist * (self.fov_y * 0.5).tan() };
        let per_pt = 2.0 * span / view_h.max(1.0);
        self.target = add(self.target, add(scale(r, -dx * per_pt), scale(u, dy * per_pt)));
    }

    /// Wheel to zoom (positive = closer).
    pub fn zoom(&mut self, scroll: f32) {
        let f = (-scroll * 0.0015).exp();
        if self.ortho > 0.0 {
            self.ortho = (self.ortho * f).clamp(0.5, self.radius * 10.0 + 50.0);
        } else {
            self.dist = (self.dist * f).clamp(0.5, self.radius * 20.0 + 100.0);
        }
    }

    /// Zoom keeping the point under pixel (px, py) of a w × h target in place (flat
    /// views; a perspective camera zooms about its target).
    pub fn zoom_at(&mut self, w: u32, h: u32, px: f32, py: f32, scroll: f32) {
        if self.ortho <= 0.0 {
            return self.zoom(scroll);
        }
        let before = self.ray(w, h, px, py).0;
        self.zoom(scroll);
        let after = self.ray(w, h, px, py).0;
        self.target = add(self.target, sub(before, after));
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

/// The reference ray cast over every nonzero cube of the layout (CPU residues;
/// `mods[i]` is slice i's modulus).
pub fn ray_cast(layout: &Layout, mods: &[Modulus], o: [f32; 3], d: [f32; 3]) -> Option<RayHit> {
    ray_cast_peeled(layout, mods, o, d, [0; 5])
}

/// The same through a peel: peeled cubes are not there.
pub fn ray_cast_peeled(layout: &Layout, mods: &[Modulus], o: [f32; 3], d: [f32; 3], peel: [u32; 5]) -> Option<RayHit> {
    let he = layout.half_extents();
    let mut best: Option<RayHit> = None;
    let mut second = f32::MAX;
    for (si, md) in mods.iter().enumerate().take(layout.slices.len()) {
        // skip slices whose box the ray misses
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in layout.slice_corners(si) {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a] - he[a]);
                hi[a] = hi[a].max(p[a] + he[a]);
            }
        }
        if ray_box(o, d, lo, hi).is_none() {
            continue;
        }
        for (n, k) in layout.slice_cells(si) {
            if layout.peeled(si, n, k, peel) {
                continue;
            }
            let c = layout.center(si, n, k);
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
        let h = |l: &Layout| l.slices.iter().map(|s| s.height() / LEVEL).map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>();
        assert_eq!(h(&l), vec![1.0, 2.0, 4.0, 6.0, 10.0]);
        assert_eq!(l.gaps(), vec![(3, 5), (5, 7), (7, 11)]);
        assert_eq!(l.missing(), vec![4, 6, 8, 9, 10]);
        let p = Layout::new(&[2, 3, 5, 7, 11], Spacing::Packed, u32::MAX);
        assert_eq!(h(&p), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(p.gaps().is_empty() && p.missing().is_empty());
    }

    #[test]
    fn the_equilateral_stack_is_a_regular_tetrahedron() {
        let mods: Vec<u32> = (2..=60).collect();
        // with a cell stride the drawn slices are what stays centred
        let strided = Layout::with_shape(&mods, Spacing::True, Shape::Regular, 4000);
        assert!(strided.stride > 1);
        for sl in &strided.slices {
            let c = strided.corners(sl.m, sl.height());
            let (cx, cz) = ((c[0][0] + c[1][0] + c[2][0]) / 3.0, (c[0][2] + c[1][2] + c[2][2]) / 3.0);
            assert!(cx.abs() < 1e-3 && cz.abs() < 1e-3, "strided m {}: centroid ({cx}, {cz})", sl.m);
        }
        let l = Layout::with_shape(&mods, Spacing::True, Shape::Regular, u32::MAX);
        // the apex (side 0, m = 1) is the origin; the base is the last slice's corners
        let base = l.corners(60, l.slices.last().unwrap().height());
        let apex = [0.0f32, -true_height(1), 0.0];
        let len = |a: [f32; 3], b: [f32; 3]| dot(sub(a, b), sub(a, b)).sqrt();
        let edges = [len(apex, base[0]), len(apex, base[1]), len(apex, base[2]), len(base[0], base[1]), len(base[1], base[2]), len(base[2], base[0])];
        for e in edges {
            assert!((e - 59.0).abs() < 1e-3, "edges {edges:?}");
        }
        // every slice's centroid on the vertical axis
        for sl in &l.slices {
            let c = l.corners(sl.m, sl.height());
            let cx = (c[0][0] + c[1][0] + c[2][0]) / 3.0;
            let cz = (c[0][2] + c[1][2] + c[2][2]) / 3.0;
            assert!(cx.abs() < 1e-3 && cz.abs() < 1e-3, "m {}: centroid ({cx}, {cz})", sl.m);
        }
    }

    #[test]
    fn flush_shapes_stand_their_faces_vertical() {
        let mods: Vec<u32> = (2..=40).collect();
        // which corners stay on the vertical axis through the stack
        // at stride 1 and under a cell stride (whose drawn corners are what lines up)
        let on_axis = |shape: Shape, pick: &dyn Fn([[f32; 3]; 3]) -> [f32; 3]| {
            [u32::MAX, 3000].iter().all(|&budget| {
                let l = Layout::with_shape(&mods, Spacing::True, shape, budget);
                assert_eq!(l.stride > 1, budget == 3000);
                l.slices.iter().all(|s| {
                    let p = pick(l.corners(s.m, s.height()));
                    let c = l.center(l.slice_of(s.m).unwrap(), 0, 0);
                    // the corners are the drawn cubes' centres
                    assert!(sub(c, l.corners(s.m, s.height())[0]).iter().all(|v| v.abs() < 1e-4));
                    p[0].abs() < 1e-3 && p[2].abs() < 1e-3
                })
            })
        };
        let mid = |a: [f32; 3], b: [f32; 3]| scale(add(a, b), 0.5);
        assert!(on_axis(Shape::CornerApex, &|c| c[0]));
        assert!(on_axis(Shape::CornerLeft, &|c| c[1]));
        assert!(on_axis(Shape::CornerRight, &|c| c[2]));
        assert!(on_axis(Shape::FaceFront, &|c| mid(c[1], c[2])));
        assert!(on_axis(Shape::FaceLeft, &|c| mid(c[0], c[1])));
        assert!(on_axis(Shape::FaceRight, &|c| mid(c[0], c[2])));
        assert!(!on_axis(Shape::Regular, &|c| c[0]));
    }

    #[test]
    fn the_stride_keeps_the_budget() {
        let mods: Vec<u32> = (2..=400).collect();
        assert!(view_cells(View::Stack, &mods, Spacing::True, 1) > 10_000_000);
        for view in View::ALL {
            for budget in [20_000u32, 100_000, 1_000_000, 4_000_000] {
                let l = Layout::build(view, &mods, Spacing::True, Shape::Regular, budget);
                let s = l.stride;
                assert!(l.total as u64 <= budget as u64, "{view:?} {budget}: {} cells", l.total);
                assert!(s == 1 || view_cells(view, &mods, Spacing::True, s - 1) > budget as u64, "{view:?}: stride {s} is not the smallest");
                // the last modulus is always drawn: the base of the stack and the diamond
                assert_eq!(l.slices.last().unwrap().m, 400);
            }
        }
        assert_eq!(stride_for(&mods, u32::MAX), 1);
    }

    #[test]
    fn strided_cubes_stay_cubes_and_never_overlap() {
        // the reported trouble: wide ranges made slabs (wide, one slice tall). With the
        // stride on the slices as well, every cube is s × s × s and fits its pitch.
        for mods in [(2..=600).collect::<Vec<u32>>(), crate::sweep::prime_tables().primes.iter().copied().filter(|&p| p <= 2000).collect()] {
            for spacing in [Spacing::True, Spacing::Packed] {
                for view in View::ALL {
                    let l = Layout::build(view, &mods, spacing, Shape::Regular, 200_000).at_rest();
                    let he = l.half_extents();
                    assert!(he[0] == he[1] && he[1] == he[2], "a cube");
                    assert!((he[0] - CUBE_HALF * l.stride as f32).abs() < 1e-6);
                    let side = 2.0 * he[0];
                    // within a slice: rows and cells are at least a cube apart
                    let s = l.stride as f32;
                    assert!(s * 1.0 >= side && s * SQRT3_2 >= side);
                    // between slices (the stack) or face rows: at least a cube apart too
                    let ys: Vec<f32> = l.slices.iter().filter(|sl| view == View::Stack || matches!(sl.kind, SliceKind::Row(_))).map(|sl| sl.origin[1]).collect();
                    for w in ys.windows(2) {
                        assert!(w[0] - w[1] >= side - 1e-3, "{view:?} {spacing:?} stride {}: pitch {} < cube {side}", l.stride, w[0] - w[1]);
                    }
                }
            }
        }
    }

    #[test]
    fn the_cell_enumeration_covers_every_strided_cell() {
        for view in View::ALL {
            for budget in [u32::MAX, 120] {
                let l = Layout::build(view, &[2, 5, 9, 13, 30], Spacing::True, Shape::Regular, budget);
                let stride = l.stride;
                assert_eq!(stride > 1, budget == 120 && view_cells(view, &l.mods, Spacing::True, 1) > 120, "{view:?} stride {stride}");
                let mut seen = std::collections::HashSet::new();
                for g in 0..l.total {
                    let (si, n, k) = l.cell(g).unwrap();
                    let m = l.slices[si].m;
                    assert!(n < m && k <= n && k.is_multiple_of(stride));
                    match l.slices[si].kind {
                        SliceKind::Row(r) => assert_eq!((n, r), (m - 1, m - 1)),
                        SliceKind::Triangle => assert!((n - l.slices[si].n0).is_multiple_of(stride)),
                    }
                    assert!(seen.insert((si, n, k)));
                }
                let expect: usize = (0..l.slices.len()).map(|si| l.slice_cells(si).len()).sum();
                assert_eq!(seen.len(), expect);
                assert_eq!(l.total as usize, expect);
                assert!(l.cell(l.total).is_none());
            }
        }
    }

    #[test]
    fn the_face_is_the_last_rows_and_the_diamond_unfolds_the_base() {
        let mods: Vec<u32> = (2..=30).collect();
        let face = Layout::build(View::Face, &mods, Spacing::True, Shape::Regular, u32::MAX);
        // one row per modulus: row m − 1, m cells, an equilateral triangle of rows √3/2 apart
        assert_eq!(face.slices.len(), mods.len());
        for (sl, &m) in face.slices.iter().zip(&mods) {
            assert_eq!((sl.m, sl.kind, sl.cells), (m, SliceKind::Row(m - 1), m));
            let si = face.slice_of(m).unwrap();
            let (a, b) = (face.center(si, m - 1, 0), face.center(si, m - 1, m - 1));
            assert!((a[0] + b[0]).abs() < 1e-4, "row {m} is centred");
            assert!((a[1] + (m - 1) as f32 * SQRT3_2).abs() < 1e-3 && a[2] == 0.0);
        }
        // the diamond: the face's rows above (the last modulus's row is the base's own),
        // and the whole base slice
        let mut d = Layout::build(View::Diamond, &mods, Spacing::True, Shape::Regular, u32::MAX);
        let last = d.slices.len() - 1;
        assert_eq!(d.slices[last].kind, SliceKind::Triangle);
        assert_eq!(d.slices[last].m, 30);
        assert!(d.slices[..last].iter().all(|sl| sl.m < 30 && matches!(sl.kind, SliceKind::Row(_))));
        assert_eq!(d.total as u64, view_cells(View::Diamond, &mods, Spacing::True, 1));
        // folded: the base meets the face along its last row at the tetrahedron's dihedral
        // angle arccos(1/3)
        assert!((d.fold - FOLD).abs() < 1e-6);
        let hinge = d.center(last, 29, 0);
        let tip = d.center(last, 0, 0);
        assert!((hinge[1] + 29.0 * SQRT3_2).abs() < 1e-3 && hinge[2].abs() < 1e-3, "the last row is the hinge: {hinge:?}");
        let into_base = normalize(sub(tip, scale(add(hinge, d.center(last, 29, 29)), 0.5)));
        let cos_dihedral = dot(into_base, [0.0, 1.0, 0.0]);
        assert!((cos_dihedral - 1.0 / 3.0).abs() < 1e-3, "dihedral cos {cos_dihedral}");
        // flat: a rhombus of two equilateral triangles, the base's apex straight below
        d.set_fold(0.0);
        let tip = d.center(last, 0, 0);
        assert!(tip[0].abs() < 1e-3 && tip[2].abs() < 1e-4);
        assert!((tip[1] + 2.0 * 29.0 * SQRT3_2).abs() < 1e-3, "the base's apex is a whole face below: {tip:?}");
        for n in 0..30u32 {
            // row n of the base is as wide as row n of the face, mirrored about the hinge
            let (l0, r0) = (d.center(last, n, 0), d.center(last, n, n));
            assert!((l0[0] + r0[0]).abs() < 1e-3 && (r0[0] - l0[0] - n as f32).abs() < 1e-3);
            assert!((l0[1] - (hinge[1] - (29 - n) as f32 * SQRT3_2)).abs() < 1e-3);
        }
    }

    #[test]
    fn the_apex_does_not_move_when_the_range_end_does() {
        // the reported trouble: 2–487 and 2–488 drew different slices all the way up
        for spacing in [Spacing::True, Spacing::Packed] {
            for (a, b) in [((2..=487).collect::<Vec<u32>>(), (2..=488).collect::<Vec<u32>>()), (crate::sweep::prime_tables().primes.iter().copied().filter(|&p| p <= 486).collect(), crate::sweep::prime_tables().primes.iter().copied().filter(|&p| p <= 487).collect())] {
                let (la, lb) = (Layout::build(View::Stack, &a, spacing, Shape::Regular, 1_000_000), Layout::build(View::Stack, &b, spacing, Shape::Regular, 1_000_000));
                assert_eq!(la.stride, lb.stride);
                let (ma, mb): (Vec<u32>, Vec<u32>) = (la.slices.iter().map(|s| s.m).collect(), lb.slices.iter().map(|s| s.m).collect());
                // the same slices from the apex down to the last couple, which make room
                // for each range's own base
                let common = ma.len().min(mb.len()) - 2;
                assert_eq!(ma[..common], mb[..common], "{spacing:?}");
                assert_eq!(ma[0], a[0]);
                assert_eq!((*ma.last().unwrap(), *mb.last().unwrap()), (*a.last().unwrap(), *b.last().unwrap()));
            }
        }
    }

    #[test]
    fn peels_take_whole_layers_and_the_extent_clears_the_stack() {
        for budget in [u32::MAX, 3_000] {
            let l = Layout::build(View::Stack, &(2..=30).collect::<Vec<_>>(), Spacing::True, Shape::Regular, budget);
            let cells: Vec<(usize, u32, u32)> = (0..l.total).map(|g| l.cell(g).unwrap()).collect();
            // each side alone, every side but one, and all five
            let mut sets: Vec<[bool; 5]> = (0..5).map(|i| std::array::from_fn(|j| j == i)).collect();
            sets.extend((0..5).map(|i| std::array::from_fn(|j| j != i)));
            sets.push([true; 5]);
            for sides in sets {
                {
                    let mut p = Peel { sides, ..Peel::default() };
                    let extent = l.peel_extent(&p);
                    assert!(extent > 0, "{sides:?}");
                    let left = |layers: u32| {
                        let q = Peel { layers, ..p };
                        cells.iter().filter(|&&(si, n, k)| !l.peeled(si, n, k, q.depths())).count()
                    };
                    assert_eq!(left(0), cells.len());
                    // each layer takes something off, until the extent takes everything
                    let mut prev = cells.len();
                    for layers in 1..=extent {
                        let now = left(layers);
                        assert!(now < prev, "{sides:?} stride {}: layer {layers} took nothing", l.stride);
                        prev = now;
                    }
                    assert_eq!(prev, 0, "{sides:?}: {extent} layers leave cubes");
                    // one layer of a side is exactly its outer cells
                    if let [side] = p.checked().collect::<Vec<_>>()[..] {
                        p.layers = 1;
                        let st = l.stride;
                        for &(si, n, k) in &cells {
                            let s = &l.slices[si];
                            let outer = match side {
                                Side::Left => k == 0,
                                Side::Right => k == n,
                                Side::Front => n == s.last_row(st),
                                Side::Top => si == 0,
                                Side::Base => si == l.slices.len() - 1,
                            };
                            assert_eq!(l.peeled(si, n, k, p.depths()), outer, "{side:?} ({si}, {n}, {k})");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn snapped_cameras_look_square_onto_the_faces() {
        // the equilateral shape, the flush ones, and packed spacing (prime selections make
        // it squat): each snap faces the plane from the top slice to the bottom one
        let primes: Vec<u32> = (2..=487).filter(|&m| crate::math::is_prime(m)).collect();
        let cases = [(Shape::Regular, Spacing::True, (2..=40).collect::<Vec<u32>>()), (Shape::CornerApex, Spacing::True, (2..=40).collect()), (Shape::FaceFront, Spacing::True, (2..=40).collect()), (Shape::Regular, Spacing::Packed, primes.clone()), (Shape::FaceLeft, Spacing::Packed, primes)];
        for (shape, spacing, mods) in cases {
            let l = Layout::build(View::Stack, &mods, spacing, shape, 1_000_000);
            let first = l.slices[0];
            let last = *l.slices.last().unwrap();
            let apex = l.corners(first.m, first.height())[0];
            let [b0, bl, br] = l.corners(last.m, last.height());
            for (snap, face) in [(Snap::Left, [apex, b0, bl]), (Snap::Right, [apex, b0, br]), (Snap::Front, [apex, bl, br])] {
                let cam = l.snap_camera(snap);
                let view = normalize(sub(cam.eye(), cam.target));
                // the view direction is perpendicular to both of the face's edges
                for e in [sub(face[1], face[0]), sub(face[2], face[0])] {
                    assert!(dot(view, normalize(e)).abs() < 1e-3, "{shape:?} {snap:?}: not square on");
                }
                // and from outside: the face's other corner is behind it
                let other = [apex, b0, bl, br].into_iter().find(|p| !face.contains(p)).unwrap();
                assert!(dot(view, sub(other, face[0])) < 0.0, "{shape:?} {snap:?}: looking from inside");
            }
            assert!(l.snap_camera(Snap::Top).pitch > 1.5 && l.snap_camera(Snap::Bottom).pitch < -1.5);
        }
    }

    #[test]
    fn a_sparse_selection_keeps_its_slices_under_a_stride() {
        // every 100th modulus: slices far apart, so the stride thins rows and cells only
        let mods: Vec<u32> = (0..41).map(|i| 2 + 100 * i).collect();
        for view in View::ALL {
            let l = Layout::build(view, &mods, Spacing::True, Shape::Regular, 1_000_000);
            // (the flat views of 41 rows fit without a stride; the stack needs one)
            assert_eq!(l.stride > 1, view != View::Face, "{view:?}: stride {}", l.stride);
            assert_eq!(l.slices.len(), mods.len(), "{view:?}: stride {} dropped slices", l.stride);
            assert!(l.total <= 1_000_000);
        }
        // a dense one is still thinned to keep the cubes apart
        let dense: Vec<u32> = (2..=600).collect();
        let l = Layout::build(View::Stack, &dense, Spacing::True, Shape::Regular, 200_000);
        assert!(l.slices.len() < dense.len());
    }

    #[test]
    fn the_diamonds_hinge_is_row_m_minus_1_at_any_stride() {
        for (hi, budget) in [(1000u32, 250_000u32), (2000, 1_000_000), (997, 40_000)] {
            let mods: Vec<u32> = (2..=hi).collect();
            let mut d = Layout::build(View::Diamond, &mods, Spacing::True, Shape::Regular, budget);
            assert!(d.stride > 1);
            let bi = d.slices.len() - 1;
            let b = d.slices[bi];
            assert_eq!((b.m, b.last_row(d.stride)), (hi, hi - 1), "stride {}", d.stride);
            // every cell of the base is a real cell of the triangle mod M
            assert!(d.slice_cells(bi).iter().all(|&(n, k)| n < hi && k <= n));
            d.set_fold(0.0);
            // the hinge row sits where the face's row M − 1 would, and base row M − 1 − js
            // is on the face rows' lattice: x = k − n/2
            let hinge = d.center(bi, hi - 1, 0);
            assert!((hinge[1] + (hi - 1) as f32 * SQRT3_2).abs() < 1e-2);
            for (n, k) in d.slice_cells(bi).into_iter().step_by(97) {
                let c = d.center(bi, n, k);
                assert!((c[0] - (k as f32 - n as f32 / 2.0)).abs() < 1e-3);
                assert!((c[1] - (hinge[1] - (hi - 1 - n) as f32 * SQRT3_2)).abs() < 1e-2);
            }
        }
    }

    #[test]
    fn a_face_row_alternates_exactly_when_its_modulus_is_prime() {
        // C(m − 1, k) ≡ (−1)^k (mod m) for every k ⇔ m prime
        for m in 2..=600u32 {
            let md = Modulus::new(m);
            assert_eq!(face_row_alternates(&md), crate::math::is_prime(m), "m = {m}");
        }
        // zeros are rare: none until 30 = 2·3·5, and (checked here to 1200) only for
        // moduli with three or more distinct prime factors, though not all of them (42)
        let omega = |m: u32| crate::math::factorize(m).len();
        let first = (2..=1200u32).find(|&m| face_row_zeros(&Modulus::new(m)) > 0);
        assert_eq!(first, Some(30));
        assert_eq!(face_row_zeros(&Modulus::new(30)), 10);
        assert_eq!(face_row_zeros(&Modulus::new(42)), 0);
        for m in 2..=1200u32 {
            if omega(m) < 3 {
                assert_eq!(face_row_zeros(&Modulus::new(m)), 0, "m = {m}");
            }
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
    fn the_flat_camera_looks_straight_at_the_face() {
        let l = Layout::build(View::Diamond, &(2..=40).collect::<Vec<_>>(), Spacing::True, Shape::Regular, u32::MAX).at_rest();
        let (w, h) = (800u32, 600u32);
        let mut cam = Camera::flat(l.bounds(), w as f32, h as f32, 0.0, 0.0);
        // every cube projects inside the window, x to the right and y up
        let (lo, hi) = l.bounds();
        for p in [lo, hi, [lo[0], hi[1], 0.0], [hi[0], lo[1], 0.0]] {
            let (px, py) = cam.project(w, h, [p[0], p[1], 0.0]).unwrap();
            assert!((0.0..=w as f32).contains(&px) && (0.0..=h as f32).contains(&py), "{p:?} → ({px}, {py})");
        }
        let (a, _) = cam.project(w, h, [-5.0, 0.0, 0.0]).unwrap();
        let cam0 = cam;
        let (b, _) = cam.project(w, h, [5.0, 0.0, 0.0]).unwrap();
        assert!(a < b, "+x is to the right");
        // rays are parallel, straight in (−z), and land where they project
        let (o, d) = cam.ray(w, h, 100.0, 200.0);
        assert!((d[2] + 1.0).abs() < 1e-5);
        let t = -o[2] / d[2];
        let hit = add(o, scale(d, t));
        let (px, py) = cam.project(w, h, hit).unwrap();
        assert!((px - 100.5).abs() < 0.05 && (py - 200.5).abs() < 0.05, "({px}, {py})");
        // the pick projection still centres the pixel
        let c = transform(&cam0.pick_view_proj(w, h, px - 0.5, py - 0.5), hit);
        // zooming about a pixel keeps the point under it
        let before = cam.ray(w, h, 300.0, 120.0).0;
        cam.zoom_at(w, h, 300.0, 120.0, 400.0);
        let after = cam.ray(w, h, 300.0, 120.0).0;
        assert!((before[0] - after[0]).abs() < 1e-3 && (before[1] - after[1]).abs() < 1e-3);
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
