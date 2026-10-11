//! The "modular wave" overlay: a translucent square over the canvas that plays one row
//! of the triangle as a wave.
//!
//! For the active row n and modulus m, the coefficients are r_k = C(n, k) mod m and each
//! sits at frequency f_k = n − 2k, the expansion (2 cos θ)^n = Σ C(n,k) e^{i(n−2k)θ}
//! filtered through mod m:
//!
//! ```text
//! w(θ) = Σ_k r_k · cos(f_k·θ + φ(t)),   θ ∈ [−π, π]
//! ```
//!
//! A full row is symmetric (r_k = r_{n−k}), so w(θ) = cos φ · Σ r_k cos(f_k θ): the
//! phase drift makes the wave breathe (swell, flatten, invert) rather than travel.
//!
//! Coefficients live in maps keyed by frequency, so rows of different lengths line up.
//! A source change only rebuilds `target`; every frame the `displayed` coefficients
//! ease toward it with frame-rate-independent exponential smoothing, and the plot is
//! normalized by a smoothed running max of |w|, so the wave morphs instead of snapping.
//!
//! API: [`WaveOverlay::set_source`] (cheap, idempotent; call every frame with the app's
//! state), [`WaveOverlay::set_viewport`], [`WaveOverlay::update`] (easing, sampling and
//! geometry, once per frame) and [`WaveOverlay::draw`] (adds the prepared shapes to an
//! egui painter, after everything else on the canvas).

use std::collections::HashMap;
use std::sync::Arc;

use egui::epaint::tessellator::Path;
use egui::epaint::text::{LayoutJob, TextWrapping};
use egui::epaint::{Mesh, PathStroke};
use egui::{Color32, CornerRadius, FontId, Galley, Pos2, Rect, Shape, Stroke, StrokeKind, Vec2};

use crate::math::{Modulus, MOD_CAP};

/// Default phase drift φ'(t) in rad/s (see [`OverlayConfig::drift`]).
pub const PHASE_DRIFT: f32 = 0.3;
/// Time constant of every easing in the overlay (coefficients, normalization, fade).
pub const TAU: f32 = 0.12;
/// Displayed entries are dropped once both their value and their target are below this.
const DROP_EPS: f32 = 1e-4;
/// Rows longer than this draw only their `TOP_K` largest coefficients (by magnitude),
/// which bounds the per-frame work for rows of up to 16,384 terms; the plot is then an
/// approximation of the full sum. See `select_active` for how morphs stay smooth.
const LARGE_ROW: u64 = 2000;
const TOP_K: usize = 256;
/// Rows up to this are built with Pascal's rule mod m (O(n²) from scratch, O(n) per
/// forward step). Longer rows use the app's closed form (Lucas / Granville / CRT,
/// verified against Pascal's rule in the GPU tests), which never forms a full binomial.
const PASCAL_MAX_ROW: u64 = 2048;
/// Most coefficients enumerated for one row. Rows longer than this (the app reaches
/// rows near 10⁹) use the window of k on screen, clamped to this many entries.
const ROW_ENUM_CAP: u64 = 16_384;
/// Most cells a selected region averages, and most rows it walks: larger regions are
/// sampled (every s-th row, every c-th cell of a long row), and the note says so.
const REGION_CELL_BUDGET: u64 = 65_536;
const REGION_ROW_BUDGET: u64 = 4_096;
/// A region is summed exactly when stepping Pascal's rule over its light cone costs at
/// most this many additions (rows × cone width; ~20–40 ms), and its first row, when
/// past PASCAL_MAX_ROW, at most REGION_CELL_BUDGET closed-form cells.
const REGION_EXACT_BUDGET: u64 = 24_000_000;
/// Sample count: ~2 per pixel of plot width (≈ 400–500 at the default size), within
/// these bounds.
const MIN_SAMPLES: usize = 256;
const MAX_SAMPLES: usize = 1024;
/// Floor of the normalization, so an all-zero row (m = 1, deep zeros) stays flat.
const NORM_FLOOR: f32 = 1e-3;
/// Below this fade the overlay is neither computed nor drawn.
const FADE_EPS: f32 = 0.004;
/// Inner padding of the box, and the heights of the text strips above and below the plot.
const PAD: f32 = 10.0;
const LABEL_H: f32 = 15.0;
const NOTE_H: f32 = 13.0;
const FORMULA_H: f32 = 14.0;
/// The formula line is shown on boxes at least this big, when it fits unclipped.
const FORMULA_MIN_SIDE: f32 = 200.0;
/// Boxes smaller than this (tiny canvases) are not drawn at all.
const MIN_DRAW_SIDE: f32 = 2.0 * PAD + LABEL_H + 32.0;

const WAVE_RGB: [u8; 3] = [0x4f, 0xc3, 0xf7];
const FORMULA: &str = "C(n,k) mod m → Σ r_k cos((n−2k)θ)";

/// Which corner of the canvas the overlay sits in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Corner {
    TopRight,
    TopLeft,
    BottomRight,
    BottomLeft,
}

impl Corner {
    pub const ALL: [Corner; 4] = [Corner::TopRight, Corner::TopLeft, Corner::BottomRight, Corner::BottomLeft];
    pub fn label(self) -> &'static str {
        match self {
            Corner::TopRight => "top right",
            Corner::TopLeft => "top left",
            Corner::BottomRight => "bottom right",
            Corner::BottomLeft => "bottom left",
        }
    }
}

/// Placement, size and motion of the overlay. Lengths are in egui points.
#[derive(Clone, Copy, Debug)]
pub struct OverlayConfig {
    pub corner: Corner,
    /// Gap to the canvas edges.
    pub margin: f32,
    /// Extra gap at the top edge (keeps the canvas zoom buttons clear).
    pub top_clearance: f32,
    /// Extra gap at the bottom edge.
    pub bottom_clearance: f32,
    /// Side as a fraction of min(window width, height), clamped to [min_side, max_side]
    /// and then to the canvas.
    pub size_frac: f32,
    pub min_side: f32,
    pub max_side: f32,
    /// Phase drift in rad/s; 0 freezes φ.
    pub drift: f32,
    /// Subtle fill between the wave and the zero line.
    pub fill: bool,
    /// Ease from one wave to the next (coefficients and scale); off snaps at once.
    pub smooth: bool,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        OverlayConfig { corner: Corner::TopRight, margin: 16.0, top_clearance: 26.0, bottom_clearance: 0.0, size_frac: 0.28, min_side: 160.0, max_side: 420.0, drift: PHASE_DRIFT, fill: true, smooth: true }
    }
}

/// What the current `target` describes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Source {
    /// One row; k_lo..=k_hi is the whole row unless it is longer than ROW_ENUM_CAP.
    Row { row: u64, modulus: u64, k_lo: u64, k_hi: u64 },
    /// A block of cells: rows n_lo..=n_hi, and in each the cells whose centre lies in
    /// x2_lo..=x2_hi, in half cells (x2 = 2k − n, so a cell's frequency n − 2k is −x2).
    /// The rectangle between two marked cells; the wave is their cells' average.
    Region { modulus: u64, n_lo: u64, n_hi: u64, x2_lo: i64, x2_hi: i64 },
}

/// How a region was sampled: cells used, the region's (estimated) size, and the
/// largest row / cell strides.
#[derive(Clone, Copy, Default, Debug)]
struct RegionStats {
    cells: u64,
    total: u64,
    row_step: u64,
    cell_step: u64,
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The smallest stride ≥ `step` coprime with 2m. Strided rows (and cells) then run
/// through every residue class mod 2 and mod every prime power of m: an even row stride
/// would keep rows of one parity only — and a row's frequencies n − 2k all share its
/// parity — while a stride sharing a prime with m would follow the triangle's
/// self-similar blocks.
fn coprime_step(step: u64, m: u64) -> u64 {
    let g = 2 * m.max(1);
    let mut s = step.max(1);
    while s > 1 && gcd(s, g) != 1 {
        s += 1;
    }
    s
}

/// Cells of row `n` whose x2 = 2k − n lies in lo..=hi: (first x2, count), stepping by 2.
fn region_row(n: u64, lo: i64, hi: i64) -> (i64, u64) {
    let ni = n as i64;
    let mut a = lo.max(-ni);
    let b = hi.min(ni);
    if (a - ni).rem_euclid(2) != 0 {
        a += 1; // x2 has the parity of n
    }
    if a > b {
        return (a, 0);
    }
    (a, ((b - a) / 2 + 1) as u64)
}

/// Row n of Pascal's triangle mod m, built and cached with Pascal's rule in u64.
#[derive(Default)]
struct PascalRow {
    n: u64,
    m: u64,
    row: Vec<u64>,
    valid: bool,
}

impl PascalRow {
    /// Row `n` mod `m` (any m ≥ 1). Steps forward or backward from the cached row when
    /// that is cheaper than restarting from row 0.
    fn get(&mut self, n: u64, m: u64) -> &[u64] {
        let cur = self.n;
        let back_cheaper = n < cur && (cur - n).saturating_mul(cur) < n.saturating_mul(n) / 2;
        if !(self.valid && self.m == m && (n >= cur || back_cheaper)) {
            self.row.clear();
            self.row.push(1 % m);
            self.n = 0;
            self.m = m;
            self.valid = true;
        }
        while self.n > n {
            // Pascal's rule inverted, in place, left to right: old[k] = new[k] − old[k−1]
            let len = self.row.len();
            for k in 1..len - 1 {
                let (a, b) = (self.row[k], self.row[k - 1]);
                self.row[k] = if a >= b { a - b } else { a.wrapping_sub(b).wrapping_add(m) };
            }
            self.row.pop();
            self.n -= 1;
        }
        while self.n < n {
            // in place, right to left: new[k] = old[k] + old[k−1] (mod m)
            self.row.push(1 % m);
            let len = self.row.len();
            for k in (1..len - 1).rev() {
                let (a, b) = (self.row[k], self.row[k - 1]);
                let s = a.wrapping_add(b);
                // a, b < m: on overflow the true sum is s + 2⁶⁴ ≥ m, so subtracting m
                // with wrapping gives the residue either way
                self.row[k] = if s < a || s >= m { s.wrapping_sub(m) } else { s };
            }
            self.n += 1;
        }
        &self.row
    }
}

/// Where everything goes inside the box.
struct Layout {
    bx: Rect,
    plot: Rect,
    label: Rect,
    note: Option<Rect>,
    formula: Option<Rect>,
}

/// A displayed coefficient as `select_active` ranks it.
#[derive(Clone, Copy)]
struct Ranked {
    f: i64,
    value: f32,
    /// |target| for the current row's terms, |value| for terms fading out.
    key: f32,
    current: bool,
}

pub struct WaveOverlay {
    pub config: OverlayConfig,
    visible: bool,
    /// Eased visibility, 0..1.
    fade: f32,
    source: Option<Source>,
    /// `target` (and the texts) are stale: rebuilt on the next visible `update`.
    dirty: bool,
    /// Only the texts are stale (the pin changed; the coefficients did not).
    texts_dirty: bool,
    pinned: bool,
    /// The last rebuild reached the row (false: modulus too large for a long row).
    in_range: bool,
    /// How the current region (if any) was sampled.
    region_stats: RegionStats,
    pascal: PascalRow,
    /// Closed form for long rows: the app's (see `share_closed_form`) or our own.
    closed: Option<Arc<Modulus>>,
    target: HashMap<i64, f32>,
    displayed: HashMap<i64, f32>,
    /// The coefficients drawn this frame, and scratch for choosing them (both reused).
    active: Vec<(i64, f32)>,
    ranked: Vec<Ranked>,
    /// w(θ_j), j = 0..samples (reused).
    samples: Vec<f32>,
    /// cos and sin of 2π i / P for i = 0..P, P = samples − 1 (rebuilt when P changes).
    cos_t: Vec<f32>,
    sin_t: Vec<f32>,
    /// Phase-free sums Σ c·cos(2π b j / P) and Σ c·sin(…) per sample (reused).
    csum: Vec<f32>,
    ssum: Vec<f32>,
    /// Active coefficients folded by frequency mod P (reused).
    buckets: Vec<f32>,
    phase: f32,
    /// Smoothed running max of |w| over θ and the phase: the envelope max_θ |Σ c e^{ifθ}|.
    norm: f32,
    /// Geometry, from `set_viewport`.
    viewport: Rect,
    window: Vec2,
    pixels_per_point: f32,
    /// Prepared shapes: the fill under the wave and the anti-aliased polyline. Held in
    /// Arcs that egui drops after tessellation, so `Arc::make_mut` reuses the buffers.
    fill_mesh: Arc<Mesh>,
    line_mesh: Arc<Mesh>,
    path: Path,
    points: Vec<Pos2>,
    /// Label text and a second, smaller line (the k window of a deep row, an all-zero
    /// row), rebuilt only when the source changes.
    label: String,
    note: String,
    /// The formula fit the box when it was last laid out (decided in `draw`).
    formula_fits: bool,
    settled: bool,
}

impl Default for WaveOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl WaveOverlay {
    pub fn new() -> Self {
        WaveOverlay {
            config: OverlayConfig::default(),
            visible: true,
            fade: 0.0,
            source: None,
            dirty: false,
            texts_dirty: false,
            pinned: false,
            in_range: true,
            region_stats: RegionStats::default(),
            pascal: PascalRow::default(),
            closed: None,
            target: HashMap::new(),
            displayed: HashMap::new(),
            active: Vec::new(),
            ranked: Vec::new(),
            samples: Vec::new(),
            cos_t: Vec::new(),
            sin_t: Vec::new(),
            csum: Vec::new(),
            ssum: Vec::new(),
            buckets: Vec::new(),
            phase: 0.0,
            norm: 0.0,
            viewport: Rect::NOTHING,
            window: Vec2::ZERO,
            pixels_per_point: 1.0,
            fill_mesh: Arc::new(Mesh::default()),
            line_mesh: Arc::new(Mesh::default()),
            path: Path::default(),
            points: Vec::new(),
            label: String::new(),
            note: String::new(),
            formula_fits: true,
            settled: false,
        }
    }

    pub fn visible(&self) -> bool {
        self.visible
    }

    pub fn set_visible(&mut self, on: bool) {
        self.visible = on;
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// True while the overlay changes from frame to frame (fading, easing or drifting),
    /// i.e. while the app should keep requesting redraws for it.
    pub fn is_animating(&self) -> bool {
        if self.layout().is_none() {
            return false; // nothing to show (no viewport, or one too small for the box)
        }
        if self.visible {
            self.fade < 1.0 || self.dirty || self.texts_dirty || !self.settled || self.config.drift != 0.0
        } else {
            self.fade > 0.0
        }
    }

    /// The screen rectangle of the overlay box (for hit testing), if it is drawn.
    pub fn box_rect(&self) -> Option<Rect> {
        if self.fade <= FADE_EPS {
            return None;
        }
        self.layout().map(|l| l.bx)
    }

    /// Point the overlay at row `row` of the triangle mod `modulus`. Cheap and idempotent:
    /// it only records the request; the coefficients are rebuilt by the next visible
    /// `update`, and only when the row, the modulus or (for rows too long to enumerate)
    /// the k window changed. `k_window` is the part of the row on screen; it is used only
    /// when the row has more than ROW_ENUM_CAP entries.
    pub fn set_source(&mut self, row: u64, modulus: u64, k_window: (u64, u64)) {
        let len = row.saturating_add(1);
        let (k_lo, k_hi) = if len <= ROW_ENUM_CAP {
            (0, row)
        } else {
            let lo = k_window.0.min(row);
            let hi = k_window.1.clamp(lo, row);
            // centre a window of ROW_ENUM_CAP entries on the visible range
            let mid = lo + (hi - lo) / 2;
            let lo2 = mid.saturating_sub(ROW_ENUM_CAP / 2).min(len - ROW_ENUM_CAP);
            (lo2, lo2 + ROW_ENUM_CAP - 1)
        };
        let src = Source::Row { row, modulus, k_lo, k_hi };
        if self.source == Some(src) {
            return;
        }
        // A new window of the same long row only changes the target when the window
        // moved a lot; keeps the morph calm while panning along a deep row.
        if let Some(Source::Row { row: or, modulus: om, k_lo: ok, .. }) = self.source {
            if or == row && om == modulus && len > ROW_ENUM_CAP && ok.abs_diff(k_lo) < ROW_ENUM_CAP / 8 {
                return;
            }
        }
        self.source = Some(src);
        self.dirty = true;
    }

    /// Point the overlay at the block of cells between two marked cells `a` and `b`
    /// ((n, k) each): rows between theirs, and in each the cells whose centre lies
    /// between their centres. The wave is the average of those cells' terms,
    /// (1/N) Σ r_{n,k} cos((n − 2k)θ + φ), binned by frequency like a row's. Cheap and
    /// idempotent, like `set_source`.
    pub fn set_region(&mut self, modulus: u64, a: (u64, u64), b: (u64, u64)) {
        let x2 = |(n, k): (u64, u64)| 2 * k as i64 - n as i64;
        let src = Source::Region { modulus, n_lo: a.0.min(b.0), n_hi: a.0.max(b.0), x2_lo: x2(a).min(x2(b)), x2_hi: x2(a).max(x2(b)) };
        if self.source != Some(src) {
            self.source = Some(src);
            self.dirty = true;
        }
    }

    /// Lend the overlay the app's closed form for the current modulus, so long rows do not
    /// build a second copy of its tables. Cheap (an Arc clone when it changed).
    pub fn share_closed_form(&mut self, md: &Arc<Modulus>) {
        if !self.closed.as_ref().is_some_and(|c| Arc::ptr_eq(c, md)) {
            self.closed = Some(md.clone());
        }
    }

    /// Mark the row as pinned by the user (shown in the label; the coefficients stay).
    pub fn set_pinned(&mut self, pinned: bool) {
        if pinned != self.pinned {
            self.pinned = pinned;
            self.texts_dirty = true;
        }
    }

    /// Canvas rectangle and window size (points) and pixels per point. Cheap; call every
    /// frame (handles resizes).
    pub fn set_viewport(&mut self, canvas: Rect, window: Vec2, pixels_per_point: f32) {
        self.viewport = canvas;
        self.window = window;
        self.pixels_per_point = if pixels_per_point.is_finite() { pixels_per_point.max(0.1) } else { 1.0 };
    }

    /// Rebuild the label and note for the current source and target.
    fn rebuild_texts(&mut self) {
        use std::fmt::Write;
        let Some(src) = self.source else { return };
        if let Source::Region { modulus: m, n_lo, n_hi, .. } = src {
            self.label.clear();
            let rows = if n_lo == n_hi { format!("row {}", group(n_lo)) } else { format!("rows {}–{}", group(n_lo), group(n_hi)) };
            let _ = write!(self.label, "{rows} · mod {} · selection", group(m));
            self.note.clear();
            let st = self.region_stats;
            if !self.in_range {
                self.note.push_str("beyond the overlay's range");
            } else {
                if st.cells < st.total {
                    let _ = write!(self.note, "average of {} of ~{} cells (sampled)", group(st.cells), group(st.total));
                } else {
                    let _ = write!(self.note, "average of {} cell{}", group(st.cells), if st.cells == 1 { "" } else { "s" });
                }
                if self.target.is_empty() && st.cells > 0 {
                    self.note.push_str(" · every r is 0");
                }
            }
            return;
        }
        let Source::Row { row: n, modulus: m, k_lo, k_hi } = src else { return };
        self.label.clear();
        let _ = write!(self.label, "row {} · mod {}", group(n), group(m));
        if self.pinned {
            self.label.push_str(" · pinned");
        }
        // the verdict goes first, so a narrow box elides the k range rather than it
        self.note.clear();
        if !self.in_range {
            self.note.push_str("beyond the overlay's range");
        } else if self.target.is_empty() {
            self.note.push_str("every r_k is 0");
        }
        if n.saturating_add(1) > ROW_ENUM_CAP {
            if !self.note.is_empty() {
                self.note.push_str(" · ");
            }
            let _ = write!(self.note, "k {}–{}", group(k_lo), group(k_hi));
        }
    }

    /// Fill `target` with the nonzero r_k of `src`; false when the row is out of reach.
    fn rebuild_target(&mut self, src: Source) -> bool {
        self.target.clear();
        let (n, m, k_lo, k_hi) = match src {
            Source::Row { row, modulus, k_lo, k_hi } => (row, modulus, k_lo, k_hi),
            Source::Region { modulus, n_lo, n_hi, x2_lo, x2_hi } => return self.rebuild_region(modulus, n_lo, n_hi, x2_lo, x2_hi),
        };
        if m <= 1 {
            // every residue mod 1 is 0: an empty target, so the wave eases to a flat line
            return true;
        }
        let ni = n as i64;
        if n <= PASCAL_MAX_ROW {
            let row = self.pascal.get(n, m);
            for (k, &r) in row.iter().enumerate() {
                if r != 0 {
                    self.target.insert(ni - 2 * k as i64, r as f32);
                }
            }
        } else if m <= MOD_CAP as u64 && n <= u32::MAX as u64 {
            let m32 = m as u32;
            if self.closed.as_ref().map(|c| c.m) != Some(m32) {
                self.closed = Some(Arc::new(Modulus::new(m32)));
            }
            let md = self.closed.as_ref().expect("closed form just built");
            for k in k_lo..=k_hi {
                let r = md.binom(n as u32, k as u32);
                if r != 0 {
                    self.target.insert(ni - 2 * k as i64, r as f32);
                }
            }
        } else {
            // Moduli above the app's cap have no closed-form tables; a deep row would
            // need O(n²) Pascal steps, so it is left flat rather than stalling the UI.
            log::debug!("wave overlay: row {n} mod {m} is beyond the overlay's range");
            return false;
        }
        // every target frequency gets a displayed entry to ease from (allocates only
        // here, on a source change)
        for &f in self.target.keys() {
            self.displayed.entry(f).or_insert(0.0);
        }
        true
    }

    /// Fill `target` with a region's cells (see `Source::Region`): each cell's r at its
    /// frequency. The wave is their average; the target keeps the sum, which differs by
    /// the constant 1/N only (the plot is normalized), so no average term is so small
    /// that it falls under DROP_EPS.
    ///
    /// Rows up to PASCAL_MAX_ROW are summed exactly (Pascal's rule walks them anyway).
    /// Deeper rows go through the closed form, cell by cell, so past the budget they are
    /// sampled by whole rows with a stride coprime with 2m (see `coprime_step`), each
    /// sampled row bringing its full spectrum and weighing for the rows it stands for.
    /// Only rows longer than ROW_ENUM_CAP are thinned, each starting one cell further
    /// on, so the thinning does not comb out the same frequencies every row.
    fn rebuild_region(&mut self, m: u64, n_lo: u64, n_hi: u64, x2_lo: i64, x2_hi: i64) -> bool {
        let mut stats = RegionStats { cells: 0, total: 0, row_step: 1, cell_step: 1 };
        let in_range = m <= 1 || n_hi <= PASCAL_MAX_ROW || (m <= MOD_CAP as u64 && n_hi <= u32::MAX as u64);
        if !in_range {
            log::debug!("wave overlay: region to row {n_hi} mod {m} is beyond the overlay's range");
            self.region_stats = stats;
            return false;
        }
        if m > 1 && n_hi > PASCAL_MAX_ROW {
            let exact = self.region_cone(m, n_lo, n_hi, x2_lo, x2_hi, &mut stats);
            if exact {
                stats.total = stats.cells;
            }
            if exact || self.region_bands(m, n_lo, n_hi, x2_lo, x2_hi, &mut stats) {
                self.region_stats = stats;
                for &f in self.target.keys() {
                    self.displayed.entry(f).or_insert(0.0);
                }
                return true;
            }
        }
        // the exact part
        if n_lo <= PASCAL_MAX_ROW {
            for n in n_lo..=n_hi.min(PASCAL_MAX_ROW) {
                let (x0, count) = region_row(n, x2_lo, x2_hi);
                stats.total += count;
                stats.cells += count;
                if count == 0 || m <= 1 {
                    continue;
                }
                let row = self.pascal.get(n, m);
                for i in 0..count {
                    let x2 = x0 + 2 * i as i64;
                    let r = row[((x2 + n as i64) / 2) as usize];
                    if r != 0 {
                        *self.target.entry(-x2).or_insert(0.0) += r as f32;
                    }
                }
            }
        }
        // the deep part, sampled within the budget
        let deep_lo = n_lo.max(PASCAL_MAX_ROW + 1);
        if n_hi >= deep_lo {
            if m > 1 {
                let m32 = m as u32;
                if self.closed.as_ref().map(|c| c.m) != Some(m32) {
                    self.closed = Some(Arc::new(Modulus::new(m32)));
                }
            }
            let rows = n_hi - deep_lo + 1;
            let probe = rows.div_ceil(REGION_ROW_BUDGET).max(1);
            let est: u64 = (deep_lo..=n_hi).step_by(probe as usize).map(|n| region_row(n, x2_lo, x2_hi).1.min(ROW_ENUM_CAP)).sum::<u64>() * probe;
            let row_step = coprime_step(probe.max(est.div_ceil(REGION_CELL_BUDGET)), m);
            stats.row_step = row_step;
            let mut n = deep_lo;
            while n <= n_hi {
                let (x0, count) = region_row(n, x2_lo, x2_hi);
                stats.total += count * row_step;
                if count > 0 {
                    let cell_step = coprime_step(count.div_ceil(ROW_ENUM_CAP), m);
                    stats.cell_step = stats.cell_step.max(cell_step);
                    // the thinning's phase moves on by one cell per sampled row
                    let first = (((n - deep_lo) / row_step) % cell_step).min(count - 1);
                    let weight = (row_step * cell_step) as f32;
                    if m > 1 {
                        let md = self.closed.as_ref().expect("closed form built above");
                        for i in (first..count).step_by(cell_step as usize) {
                            let x2 = x0 + 2 * i as i64;
                            let r = md.binom(n as u32, ((x2 + n as i64) / 2) as u32);
                            if r != 0 {
                                *self.target.entry(-x2).or_insert(0.0) += r as f32 * weight;
                            }
                        }
                    }
                    stats.cells += (count - first).div_ceil(cell_step);
                }
                n = match n.checked_add(row_step) {
                    Some(v) => v,
                    None => break,
                };
            }
        }
        stats.total = stats.total.max(stats.cells);
        self.region_stats = stats;
        for &f in self.target.keys() {
            self.displayed.entry(f).or_insert(0.0);
        }
        true
    }

    /// A deep region exactly, if affordable: row n_lo over the k the region's cells
    /// depend on (from the closed form, or Pascal's rule when shallow), then Pascal's rule
    /// down to n_hi. Cell (n, k) depends on row n_lo from k − (n − n_lo) to k, so the
    /// window starts (n_hi − n_lo) cells left of the region's left edge at n_hi; values
    /// left of that light cone go stale and are never read. False (nothing touched) when
    /// over the budgets.
    fn region_cone(&mut self, m: u64, n_lo: u64, n_hi: u64, x2_lo: i64, x2_hi: i64, stats: &mut RegionStats) -> bool {
        self.region_cone_weighted(m, n_lo, n_hi, x2_lo, x2_hi, 1.0, REGION_EXACT_BUDGET, stats)
    }

    /// `region_cone` with every cell counted `weight` times, within `budget` additions.
    #[allow(clippy::too_many_arguments)] // the region's bounds, and how to count it
    fn region_cone_weighted(&mut self, m: u64, n_lo: u64, n_hi: u64, x2_lo: i64, x2_hi: i64, weight: f32, budget: u64, stats: &mut RegionStats) -> bool {
        let rows = n_hi - n_lo + 1;
        let k_first = |n: u64| ((x2_lo + n as i64 + 1).div_euclid(2)).max(0) as u64; // ⌈(x2_lo + n)/2⌉
        let k_last = |n: u64| ((x2_hi + n as i64).div_euclid(2)).min(n as i64); // ⌊(x2_hi + n)/2⌋
        if k_last(n_hi) < 0 {
            return false; // nothing of the region inside the triangle: let the slow path say so
        }
        let lo = k_first(n_hi).saturating_sub(rows - 1);
        let hi = k_last(n_hi) as u64;
        if hi < lo {
            return false;
        }
        let width = hi - lo + 1;
        let first_row_cells = width.min(n_lo.saturating_sub(lo) + 1);
        if rows.saturating_mul(width) > budget || (n_lo > PASCAL_MAX_ROW && first_row_cells > REGION_CELL_BUDGET) {
            return false;
        }
        // row n_lo on [lo, hi] (zeros past its end)
        let mut seg = vec![0u64; width as usize];
        if n_lo <= PASCAL_MAX_ROW {
            let row = self.pascal.get(n_lo, m);
            for (i, v) in seg.iter_mut().enumerate() {
                if let Some(&r) = row.get(lo as usize + i) {
                    *v = r;
                }
            }
        } else {
            let m32 = m as u32;
            if self.closed.as_ref().map(|c| c.m) != Some(m32) {
                self.closed = Some(Arc::new(Modulus::new(m32)));
            }
            let md = self.closed.as_ref().expect("closed form just built");
            for (i, v) in seg.iter_mut().enumerate() {
                let k = lo + i as u64;
                if k <= n_lo {
                    *v = md.binom(n_lo as u32, k as u32) as u64;
                }
            }
        }
        let mut n = n_lo;
        loop {
            // the region's cells of row n
            let (a, b) = (k_first(n), k_last(n));
            if b >= a as i64 {
                for k in a..=b as u64 {
                    let r = seg[(k - lo) as usize];
                    stats.cells += 1;
                    if r != 0 {
                        *self.target.entry(n as i64 - 2 * k as i64).or_insert(0.0) += r as f32 * weight;
                    }
                }
            }
            if n == n_hi {
                break;
            }
            // row n + 1, in place, right to left (as PascalRow::get)
            for i in (1..seg.len()).rev() {
                let (x, y) = (seg[i], seg[i - 1]);
                let s = x.wrapping_add(y);
                seg[i] = if s < x || s >= m { s.wrapping_sub(m) } else { s };
            }
            n += 1;
        }
        true
    }

    /// A deep region too large to sum: BANDS bands of consecutive rows spread evenly
    /// over it, each summed exactly with `region_cone_weighted` and weighed for the rows
    /// it stands for. Consecutive rows keep both parities and the local structure that
    /// isolated sample rows lose. False when even one band is unaffordable.
    fn region_bands(&mut self, m: u64, n_lo: u64, n_hi: u64, x2_lo: i64, x2_hi: i64, stats: &mut RegionStats) -> bool {
        const BANDS: u64 = 16;
        let rows = n_hi - n_lo + 1;
        let cells_per_row = (((x2_hi - x2_lo).max(0) / 2 + 1) as u64).min(n_hi + 1);
        // band height h with BANDS · h · (cells + h) ≤ budget
        let per_band = REGION_EXACT_BUDGET / BANDS;
        let mut h = (per_band / (cells_per_row + 1)).clamp(1, rows / BANDS);
        while h > 1 && h * (cells_per_row + h) > per_band {
            h /= 2;
        }
        if h * (cells_per_row + h) > per_band || h * BANDS >= rows {
            return false;
        }
        let weight = rows as f32 / (h * BANDS) as f32;
        let (cells0, mut total) = (stats.cells, 0u64);
        for i in 0..BANDS {
            let start = n_lo + i * (rows - h) / (BANDS - 1);
            let mut band = RegionStats::default();
            if !self.region_cone_weighted(m, start, start + h - 1, x2_lo, x2_hi, weight, per_band, &mut band) {
                // undo the bands already added: the caller falls back to row sampling
                self.target.clear();
                stats.cells = cells0;
                return false;
            }
            stats.cells += band.cells;
            total += band.cells;
        }
        stats.total += (total as f32 * weight) as u64;
        stats.row_step = rows / (h * BANDS);
        true
    }

    /// The box and its parts for the current viewport (None when there is no room).
    fn layout(&self) -> Option<Layout> {
        let c = &self.config;
        let vp = self.viewport;
        if !vp.is_positive() {
            return None;
        }
        let basis = if self.window.x > 0.0 && self.window.y > 0.0 { self.window.x.min(self.window.y) } else { vp.width().min(vp.height()) };
        let avail_w = (vp.width() - 2.0 * c.margin).max(1.0);
        let avail_h = (vp.height() - 2.0 * c.margin - c.top_clearance - c.bottom_clearance).max(1.0);
        let side = (c.size_frac * basis).clamp(c.min_side, c.max_side).min(avail_w).min(avail_h);
        if side.is_nan() || side < MIN_DRAW_SIDE {
            return None;
        }
        let x = match c.corner {
            Corner::TopRight | Corner::BottomRight => vp.max.x - c.margin - side,
            Corner::TopLeft | Corner::BottomLeft => vp.min.x + c.margin,
        };
        let y = match c.corner {
            Corner::TopRight | Corner::TopLeft => vp.min.y + c.margin + c.top_clearance,
            Corner::BottomRight | Corner::BottomLeft => vp.max.y - c.margin - c.bottom_clearance - side,
        };
        let bx = Rect::from_min_size(Pos2::new(x, y), Vec2::splat(side));
        let inner = bx.shrink(PAD);
        let show_formula = side >= FORMULA_MIN_SIDE && self.formula_fits;
        let formula_h = if show_formula { FORMULA_H } else { 0.0 };
        let label = Rect::from_min_max(inner.min, Pos2::new(inner.max.x, inner.min.y + LABEL_H));
        let note = (!self.note.is_empty()).then(|| Rect::from_min_max(Pos2::new(inner.min.x, label.max.y), Pos2::new(inner.max.x, label.max.y + NOTE_H)));
        let top = note.map_or(label.max.y, |r| r.max.y);
        let plot = Rect::from_min_max(Pos2::new(inner.min.x, top), Pos2::new(inner.max.x, (inner.max.y - formula_h).max(top + 1.0)));
        let formula = show_formula.then(|| Rect::from_min_max(Pos2::new(inner.min.x, inner.max.y - FORMULA_H), inner.max));
        Some(Layout { bx, plot, label, note, formula })
    }

    /// Advance the animation by the real frame time `dt` (seconds): fade, phase drift,
    /// coefficient easing, sampling, normalization and the prepared geometry. Call once
    /// per frame, after `set_source` and `set_viewport`.
    pub fn update(&mut self, dt: f32) {
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        let alpha = 1.0 - (-dt / TAU).exp();
        let want = if self.visible { 1.0 } else { 0.0 };
        self.fade += (want - self.fade) * alpha;
        if (self.fade - want).abs() < FADE_EPS {
            self.fade = want;
        }
        let drift = if self.config.drift.is_finite() { self.config.drift } else { 0.0 };
        self.phase = (self.phase + drift * dt).rem_euclid(std::f32::consts::TAU);
        if !self.visible && self.fade <= FADE_EPS {
            return; // hidden: nothing to compute
        }
        if self.dirty {
            self.dirty = false;
            if let Some(src) = self.source {
                self.in_range = self.rebuild_target(src);
            }
            self.texts_dirty = true;
            self.settled = false;
        }
        if self.texts_dirty {
            self.texts_dirty = false;
            self.rebuild_texts();
        }

        // ease every displayed coefficient toward its target (0 when absent); without
        // smoothing, snap
        let ease = if self.config.smooth { alpha } else { 1.0 };
        let target = &self.target;
        let mut moving = false;
        for (f, v) in self.displayed.iter_mut() {
            let t = target.get(f).copied().unwrap_or(0.0);
            *v += (t - *v) * ease;
            if (t - *v).abs() > DROP_EPS * (1.0 + t.abs()) {
                moving = true;
            }
        }
        self.displayed.retain(|f, v| v.abs() >= DROP_EPS || target.get(f).is_some_and(|t| t.abs() >= DROP_EPS));

        self.select_active();

        // sample count from the plot width in pixels
        let Some(Layout { plot, .. }) = self.layout() else {
            self.settled = true; // no room to draw: nothing to animate
            return;
        };
        let px = (plot.width() * self.pixels_per_point).max(1.0);
        let n_samples = ((px * 2.0).round() as usize).clamp(MIN_SAMPLES, MAX_SAMPLES);
        if self.samples.len() != n_samples {
            self.samples.resize(n_samples, 0.0);
        }
        let period = n_samples - 1; // θ_j = −π + 2π j / period, j = 0..=period
        if self.cos_t.len() != period {
            let step = std::f64::consts::TAU / period as f64;
            self.cos_t.clear();
            self.sin_t.clear();
            self.cos_t.extend((0..period).map(|i| (step * i as f64).cos() as f32));
            self.sin_t.extend((0..period).map(|i| (step * i as f64).sin() as f32));
        }
        // e^{i f θ_j} = (−1)^f · e^{2πi ((f mod P)·j mod P) / P}: exact for any integer
        // frequency, so rows near 10⁹ need no large trig arguments, and frequencies with
        // the same f mod P fold into one bucket (at most P² terms per frame).
        let p64 = period as i64;
        self.buckets.clear();
        self.buckets.resize(period, 0.0);
        for &(f, c) in &self.active {
            self.buckets[f.rem_euclid(p64) as usize] += if f.rem_euclid(2) == 1 { -c } else { c };
        }
        // C_j = Σ c cos(x), S_j = Σ c sin(x) with x = 2π b j / P; then
        // w(θ_j) = Σ c cos(x + φ) = cos φ · C_j − sin φ · S_j
        self.csum.clear();
        self.csum.resize(n_samples, 0.0);
        self.ssum.clear();
        self.ssum.resize(n_samples, 0.0);
        for (b, &c) in self.buckets.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            let mut i = 0usize;
            for (cs, ss) in self.csum.iter_mut().zip(self.ssum.iter_mut()) {
                *cs += c * self.cos_t[i];
                *ss += c * self.sin_t[i];
                i += b;
                if i >= period {
                    i -= period;
                }
            }
        }
        let (cp, sp) = (self.phase.cos(), self.phase.sin());
        let mut env2 = 0.0f32;
        for ((s, &cs), &ss) in self.samples.iter_mut().zip(&self.csum).zip(&self.ssum) {
            *s = cp * cs - sp * ss;
            env2 = env2.max(cs * cs + ss * ss);
        }

        // Normalize by a smoothed running max of |w|, taken over the phase as well: the
        // envelope max_j |C_j + i S_j| does not move with φ, so the drift shows as the
        // wave breathing (a whole row is w = cos φ · C) and peaks never clip.
        let raw = env2.sqrt().max(NORM_FLOOR);
        if !(self.norm > 0.0 && self.norm.is_finite()) {
            self.norm = raw;
        } else {
            self.norm += (raw - self.norm) * ease;
        }
        self.norm = self.norm.max(NORM_FLOOR);
        if (raw - self.norm).abs() > 1e-3 * raw {
            moving = true;
        }
        self.settled = !moving;
        self.build_geometry(plot);
    }

    /// The coefficients to draw this frame. Rows up to LARGE_ROW draw every displayed
    /// term. Longer rows draw the TOP_K largest terms of the current row (ranked by their
    /// target) plus the TOP_K largest still fading out from earlier rows (ranked by their
    /// value). Easing scales each group uniformly, so neither ranking changes during a
    /// morph and nothing pops in or out; ties break by |f| and then f, which keeps the
    /// choice deterministic and the ±f pairs of a symmetric row together.
    fn select_active(&mut self) {
        self.active.clear();
        // (a region's terms are averages over many rows, with no few dominant ones; the
        // frequency folding already bounds its per-frame cost)
        let long_row = matches!(self.source, Some(Source::Row { row, .. }) if row > LARGE_ROW);
        if !long_row || self.displayed.len() <= TOP_K {
            self.active.extend(self.displayed.iter().map(|(&f, &v)| (f, v)));
            return;
        }
        self.ranked.clear();
        for (&f, &value) in &self.displayed {
            let t = self.target.get(&f).copied().unwrap_or(0.0);
            let current = t != 0.0;
            self.ranked.push(Ranked { f, value, key: if current { t.abs() } else { value.abs() }, current });
        }
        // current-row terms first
        let mut split = 0;
        for i in 0..self.ranked.len() {
            if self.ranked[i].current {
                self.ranked.swap(i, split);
                split += 1;
            }
        }
        let order = |a: &Ranked, b: &Ranked| b.key.total_cmp(&a.key).then(a.f.unsigned_abs().cmp(&b.f.unsigned_abs())).then(b.f.cmp(&a.f));
        let (cur, old) = self.ranked.split_at_mut(split);
        for group in [cur, old] {
            let mut keep = group.len().min(TOP_K);
            if group.len() > TOP_K {
                group.select_nth_unstable_by(TOP_K, order);
                // the first term left out is the partner of the last term kept: keep both
                let last = group[..TOP_K].iter().copied().max_by(|a, b| order(a, b)).expect("TOP_K > 0");
                if group[TOP_K].f == -last.f {
                    keep += 1;
                }
            }
            self.active.extend(group[..keep].iter().map(|r| (r.f, r.value)));
        }
    }

    /// Fill and polyline meshes for the current samples (buffers reused across frames).
    fn build_geometry(&mut self, plot: Rect) {
        let fade = self.fade;
        let n = self.samples.len();
        let cy = plot.center().y;
        let half = plot.height() * 0.5 * 0.92;
        let inv = 1.0 / self.norm.max(NORM_FLOOR);
        self.points.clear();
        for (j, &s) in self.samples.iter().enumerate() {
            let x = plot.min.x + plot.width() * j as f32 / (n - 1).max(1) as f32;
            let y = (cy - s * inv * half).clamp(plot.min.y, plot.max.y);
            self.points.push(Pos2::new(x, y));
        }
        let [r, g, b] = WAVE_RGB;
        let fill = Arc::make_mut(&mut self.fill_mesh);
        fill.clear();
        if self.config.fill {
            let fill_col = Color32::from_rgba_unmultiplied(r, g, b, (0.15 * 255.0 * fade).round() as u8);
            for (j, p) in self.points.iter().enumerate() {
                fill.colored_vertex(*p, fill_col);
                fill.colored_vertex(Pos2::new(p.x, cy), fill_col);
                if j > 0 {
                    let a = 2 * (j as u32 - 1);
                    fill.add_triangle(a, a + 1, a + 2);
                    fill.add_triangle(a + 1, a + 3, a + 2);
                }
            }
        }
        let line = Arc::make_mut(&mut self.line_mesh);
        line.clear();
        if self.points.len() >= 2 {
            self.path.clear();
            self.path.add_open_points(&self.points);
            let stroke = PathStroke::from(Stroke::new(2.0, Color32::from_rgba_unmultiplied(r, g, b, (255.0 * fade).round() as u8)));
            self.path.stroke_open(1.0 / self.pixels_per_point, &stroke, line);
        }
    }

    /// Add the overlay to `painter`, on top of whatever the painter already holds.
    pub fn draw(&mut self, painter: &egui::Painter) {
        if self.fade <= FADE_EPS {
            return;
        }
        let Some(Layout { bx, plot, label: label_rect, note: note_rect, formula: formula_rect }) = self.layout() else { return };
        let fade = self.fade;
        let a = |x: f32| (x * 255.0 * fade).round() as u8;
        painter.rect_filled(bx, CornerRadius::same(8), Color32::from_rgba_unmultiplied(0, 0, 0, a(0.62)));
        painter.rect_stroke(bx, CornerRadius::same(8), Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, a(0.15))), StrokeKind::Inside);
        let zero = Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, a(0.18)));
        painter.line_segment([Pos2::new(plot.min.x, plot.center().y), Pos2::new(plot.max.x, plot.center().y)], zero);
        painter.add(Shape::mesh(self.fill_mesh.clone()));
        painter.add(Shape::mesh(self.line_mesh.clone()));

        // Text, elided to the box width. Laid out every frame through egui's galley cache
        // (a hash lookup, plus a copy of each short string), so it follows font-atlas
        // rebuilds such as a theme switch.
        let width = label_rect.width();
        let line = |text: &str, size: f32| -> Arc<Galley> {
            let mut job = LayoutJob::simple_singleline(text.to_owned(), FontId::monospace(size), Color32::WHITE);
            job.wrap = TextWrapping::truncate_at_width(width);
            painter.layout_job(job)
        };
        painter.galley_with_override_text_color(label_rect.min, line(&self.label, 11.0), Color32::from_rgba_unmultiplied(221, 221, 221, a(0.9)));
        if let Some(nr) = note_rect {
            painter.galley_with_override_text_color(nr.min, line(&self.note, 10.0), Color32::from_rgba_unmultiplied(187, 187, 187, a(0.7)));
        }
        if bx.width() >= FORMULA_MIN_SIDE {
            let formula = line(FORMULA, 10.0);
            let fits = !formula.elided;
            if fits != self.formula_fits {
                // the plot height depends on this: lay out again next frame
                self.formula_fits = fits;
                self.settled = false;
            }
            if let (Some(fr), true) = (formula_rect, fits) {
                painter.galley_with_override_text_color(Pos2::new(fr.min.x, fr.max.y - formula.size().y), formula, Color32::from_rgba_unmultiplied(187, 187, 187, a(0.7)));
            }
        }
    }

    // ---- read access for tests and diagnostics ----

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn norm(&self) -> f32 {
        self.norm
    }

    pub fn fade(&self) -> f32 {
        self.fade
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn note(&self) -> &str {
        &self.note
    }

    pub fn displayed(&self, f: i64) -> f32 {
        self.displayed.get(&f).copied().unwrap_or(0.0)
    }

    pub fn target(&self, f: i64) -> f32 {
        self.target.get(&f).copied().unwrap_or(0.0)
    }

    pub fn displayed_len(&self) -> usize {
        self.displayed.len()
    }

    pub fn active_len(&self) -> usize {
        self.active.len()
    }
}

/// `1234567` → `1,234,567`.
fn group(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::binom_exact;

    fn overlay() -> WaveOverlay {
        let mut w = WaveOverlay::new();
        w.set_viewport(Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 700.0)), Vec2::new(1300.0, 760.0), 1.0);
        w
    }

    /// Run `secs` of frames of length `dt`.
    fn run(w: &mut WaveOverlay, secs: f32, dt: f32) {
        for _ in 0..(secs / dt).round() as usize {
            w.update(dt);
        }
    }

    #[test]
    fn pascal_row_matches_true_binomials_mod_m() {
        let mut pr = PascalRow::default();
        let huge = u64::MAX - 58; // the largest prime below 2⁶⁴: sums overflow u64
        for m in [1u64, 2, 3, 4, 6, 7, 10, 12, 30, 97, 65536, 1_000_000_007, 1 << 63, huge] {
            for n in 0..=30u64 {
                let row = pr.get(n, m).to_vec();
                assert_eq!(row.len() as u64, n + 1);
                for (k, &r) in row.iter().enumerate() {
                    let exact = binom_exact(n as u32, k as u32).unwrap();
                    assert_eq!(r as u128, exact % m as u128, "C({n},{k}) mod {m}");
                }
            }
            // stepping backwards (inverse Pascal rule, or a restart for big jumps)
            for n in (0..=29u64).rev().chain([30, 3, 28, 27, 0, 30, 2]) {
                let row = pr.get(n, m).to_vec();
                let want: Vec<u128> = (0..=n as u32).map(|k| binom_exact(n as u32, k).unwrap() % m as u128).collect();
                assert_eq!(row.iter().map(|&x| x as u128).collect::<Vec<_>>(), want, "row {n} mod {m}");
            }
        }
        // deep enough that C(n,k) overflows u64 many times over, forward and back
        for n in [125u64, 124, 110, 125] {
            let row = pr.get(n, huge).to_vec();
            for (k, &r) in row.iter().enumerate() {
                assert_eq!(r as u128, binom_exact(n as u32, k as u32).unwrap() % huge as u128, "C({n},{k}) mod {huge}");
            }
        }
    }

    #[test]
    fn closed_form_rows_match_pascal_rows() {
        // the long-row path (closed form) agrees with Pascal's rule where both apply
        let mut pr = PascalRow::default();
        for m in [2u32, 9, 30, 101] {
            let md = Modulus::new(m);
            let n = PASCAL_MAX_ROW + 1;
            let row = pr.get(n, m as u64).to_vec();
            for (k, &r) in row.iter().enumerate() {
                assert_eq!(md.binom(n as u32, k as u32) as u64, r, "C({n},{k}) mod {m}");
            }
        }
    }

    #[test]
    fn target_is_keyed_by_frequency() {
        let mut w = overlay();
        w.set_source(4, 7, (0, 4));
        w.update(0.0);
        // row 4: 1 4 6 4 1 at frequencies 4, 2, 0, −2, −4
        for (f, c) in [(4, 1.0), (2, 4.0), (0, 6.0), (-2, 4.0), (-4, 1.0)] {
            assert_eq!(w.target(f), c);
        }
        assert_eq!(w.target(1), 0.0);
        assert_eq!(w.label(), "row 4 · mod 7");
        assert_eq!(w.note(), "");
        w.set_pinned(true);
        assert!(!w.dirty, "pinning only relabels; the coefficients are not recomputed");
        w.update(0.0);
        assert_eq!(w.label(), "row 4 · mod 7 · pinned");
        // a long row, through the closed form: C(2050, k) mod 7 at f = 2050 − 2k
        let mut w = overlay();
        w.set_source(2050, 7, (0, 2050));
        w.update(0.0);
        let md = Modulus::new(7);
        for k in [0u32, 1, 2, 7, 49, 1000, 1025, 2050] {
            assert_eq!(w.target(2050 - 2 * k as i64), md.binom(2050, k) as f32, "k={k}");
        }
    }

    #[test]
    fn set_source_is_lazy_and_idempotent() {
        let mut w = overlay();
        w.set_source(12, 5, (0, 12));
        assert_eq!(w.target(12), 0.0, "nothing is built before update");
        w.update(1.0 / 60.0);
        assert_eq!(w.target(12), 1.0);
        let label = w.label().as_ptr();
        w.set_source(12, 5, (0, 12));
        assert!(!w.dirty, "the same source is a no-op");
        w.update(1.0 / 60.0);
        assert_eq!(w.label().as_ptr(), label);
        // hidden overlays do no work
        w.set_visible(false);
        run(&mut w, 2.0, 1.0 / 60.0);
        w.set_source(13, 5, (0, 13));
        w.update(1.0 / 60.0);
        assert!(w.dirty && w.target(13) == 0.0);
    }

    #[test]
    fn smoothing_converges_to_target_frame_rate_independently() {
        for dt in [1.0 / 30.0, 1.0 / 144.0] {
            let mut w = overlay();
            w.set_source(6, 5, (0, 6));
            run(&mut w, 2.0, dt);
            for f in [-6i64, -4, -2, 0, 2, 4, 6] {
                assert!((w.displayed(f) - w.target(f)).abs() < 1e-3, "dt={dt}: f={f} {} vs {}", w.displayed(f), w.target(f));
            }
            // a new row eases away from the old one, and old frequencies decay to nothing
            w.set_source(7, 5, (0, 7));
            w.update(0.01);
            assert!(w.displayed(5) > 0.0 && w.displayed(5) < w.target(5), "eases, does not snap");
            assert!(w.displayed(6) > 0.0, "the old row fades out");
            run(&mut w, 3.0, dt);
            assert_eq!(w.displayed(6), 0.0, "row 7 has only odd frequencies");
            assert!((w.displayed(5) - w.target(5)).abs() < 1e-3);
        }
        // the same elapsed time gives the same state at different frame rates
        let state = |dt: f32| {
            let mut w = overlay();
            w.set_source(10, 11, (0, 10));
            w.update(0.0);
            run(&mut w, 0.3, dt);
            (w.displayed(0), w.norm(), w.fade())
        };
        let (a, b) = (state(1.0 / 30.0), state(1.0 / 240.0));
        assert!(a.0 > 1.0, "C(10,5) mod 11 = 10 is on its way");
        assert!((a.0 - b.0).abs() < 1e-3 * a.0 && (a.2 - b.2).abs() < 1e-3, "{a:?} vs {b:?}");
        // the normalization chases a moving max, so it agrees only approximately
        assert!((a.1 - b.1).abs() < 0.05 * a.1, "{a:?} vs {b:?}");
    }

    #[test]
    fn edge_cases_produce_no_nans() {
        for (n, m) in [(0u64, 1u64), (0, 7), (5, 1), (10, 1), (3, 2), (100, u64::from(u32::MAX)), (40, u64::MAX), (5000, 1_000_003), (1_000_000_000, 7), (u64::from(u32::MAX), 2), (u64::MAX, 3)] {
            let mut w = overlay();
            w.set_source(n, m, (0, n.min(1000)));
            run(&mut w, 0.5, 1.0 / 60.0);
            assert!(w.samples().iter().all(|s| s.is_finite()), "n={n} m={m}");
            assert!(w.norm().is_finite() && w.norm() > 0.0, "n={n} m={m}");
            if m == 1 {
                assert!(w.samples().iter().all(|s| *s == 0.0), "m = 1 is a flat line");
                assert_eq!(w.note(), "every r_k is 0");
            }
        }
        // n = 0: a single coefficient at frequency 0, so w(θ) = cos φ everywhere
        let mut w = overlay();
        w.set_source(0, 7, (0, 0));
        run(&mut w, 2.0, 1.0 / 60.0);
        let s = w.samples();
        assert!(s.iter().all(|x| (x - s[0]).abs() < 1e-4), "row 0 is constant in θ");
        // odd frame times, a zero-sized window and resizes stay finite
        let mut w = overlay();
        w.set_source(33, 6, (0, 33));
        for dt in [0.0, -1.0, f32::NAN, f32::INFINITY, 1e-7, 5.0] {
            w.update(dt);
        }
        for size in [Vec2::new(0.0, 0.0), Vec2::new(90.0, 4000.0), Vec2::new(5000.0, 3000.0)] {
            w.set_viewport(Rect::from_min_size(Pos2::ZERO, size), size, 2.0);
            w.update(1.0 / 60.0);
            assert!(w.samples().iter().all(|s| s.is_finite()) && w.norm().is_finite(), "{size:?}");
            assert!(w.samples().len() <= MAX_SAMPLES);
        }
    }

    #[test]
    fn samples_match_the_direct_sum() {
        let mut w = overlay();
        w.set_source(9, 11, (0, 9));
        run(&mut w, 4.0, 1.0 / 60.0);
        let s = w.samples().to_vec();
        let period = s.len() - 1;
        let phase = w.phase;
        for (j, &got) in s.iter().enumerate().step_by(37) {
            let theta = -std::f64::consts::PI + 2.0 * std::f64::consts::PI * j as f64 / period as f64;
            let want: f64 = (0..=9i64).map(|k| w.displayed(9 - 2 * k) as f64 * ((9 - 2 * k) as f64 * theta + phase as f64).cos()).sum();
            assert!((got as f64 - want).abs() < 1e-3 * (1.0 + want.abs()), "j={j}: {got} vs {want}");
        }
    }

    #[test]
    fn long_rows_draw_only_the_largest_coefficients() {
        let mut w = overlay();
        w.set_source(3000, 999_983, (0, 3000));
        run(&mut w, 0.2, 1.0 / 60.0);
        assert!(w.displayed_len() > TOP_K);
        assert!((TOP_K..=TOP_K + 1).contains(&w.active_len()), "{}", w.active_len());
        // short rows draw everything
        let mut w = overlay();
        w.set_source(1500, 999_983, (0, 1500));
        run(&mut w, 0.2, 1.0 / 60.0);
        assert_eq!(w.active_len(), 1501);
        // rows beyond the enumeration cap use a window of k around the visible part
        let mut w = overlay();
        w.set_source(1_000_000_000, 30, (499_999_000, 500_001_000));
        let window = |w: &WaveOverlay| match w.source {
            Some(Source::Row { k_lo, k_hi, .. }) => (k_lo, k_hi),
            other => panic!("{other:?}"),
        };
        let (k_lo, k_hi) = window(&w);
        assert_eq!(k_hi - k_lo + 1, ROW_ENUM_CAP);
        assert!(k_lo <= 499_999_000 && k_hi >= 500_001_000);
        // ... clamped to the row
        w.set_source(1_000_000_000, 30, (999_999_990, 1_000_000_000));
        let (k_lo, k_hi) = window(&w);
        assert_eq!((k_hi, k_hi - k_lo + 1), (1_000_000_000, ROW_ENUM_CAP));
        run(&mut w, 0.2, 1.0 / 60.0);
        assert_eq!(w.label(), "row 1,000,000,000 · mod 30");
        assert!(w.note().starts_with("k 999,983,617–1,000,000,000"), "{}", w.note());
        // deep rows are mostly zeros: say so instead of drawing an unexplained flat line
        let mut w = overlay();
        w.set_source(587_896_253, 30, (0, 587_896_253));
        w.update(1.0 / 60.0);
        if w.displayed_len() == 0 {
            assert!(w.note().starts_with("every r_k is 0 · k "), "{}", w.note());
        }
    }

    #[test]
    fn long_row_morphs_do_not_jump() {
        // rows 3001 → 3002 share no frequency (opposite parity): the old row's top terms
        // must fade out while the new row's fade in, not swap in one frame
        let mut w = overlay();
        w.config.drift = 0.0;
        w.set_source(3001, 999_983, (0, 3001));
        run(&mut w, 3.0, 1.0 / 60.0);
        w.set_source(3002, 999_983, (0, 3002));
        let mut prev: Option<Vec<f32>> = None;
        let mut worst = 0.0f32;
        for _ in 0..120 {
            w.update(1.0 / 60.0);
            let norm = w.norm();
            let cur: Vec<f32> = w.samples().iter().map(|s| s / norm).collect();
            if let Some(p) = &prev {
                worst = worst.max(p.iter().zip(&cur).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max));
            }
            prev = Some(cur);
        }
        assert!(worst < 0.25, "largest frame-to-frame change {worst} of full scale");
        // ties (every r_k = 1 mod 2) still pick a symmetric, deterministic set
        let pick = || {
            let mut w = overlay();
            w.set_source(4095, 2, (0, 4095));
            run(&mut w, 0.5, 1.0 / 60.0);
            let mut f: Vec<i64> = w.active.iter().map(|a| a.0).collect();
            f.sort_unstable();
            f
        };
        let a = pick();
        assert!(a.iter().all(|f| a.binary_search(&-f).is_ok()), "±f pairs stay together");
        assert_eq!(a, pick());
    }

    #[test]
    fn normalization_follows_the_envelope() {
        // a full drift cycle on a whole row: the wave breathes (|w| ∝ |cos φ|) and the
        // normalized peaks never pass ±1, so nothing is clipped
        let mut w = overlay();
        w.set_source(20, 7, (0, 20));
        run(&mut w, 2.0, 1.0 / 60.0);
        let (mut lo, mut hi) = (f32::MAX, 0.0f32);
        for _ in 0..(std::f32::consts::TAU / PHASE_DRIFT * 60.0) as usize {
            w.update(1.0 / 60.0);
            let peak = w.samples().iter().fold(0.0f32, |a, s| a.max(s.abs())) / w.norm();
            assert!(peak <= 1.0 + 1e-4, "peak {peak} at φ = {}", w.phase);
            lo = lo.min(peak);
            hi = hi.max(peak);
        }
        assert!(hi > 0.99 && lo < 0.05, "breathes from {lo} to {hi}");
    }

    #[test]
    fn tiny_canvases_draw_nothing() {
        let mut w = overlay();
        w.set_source(9, 4, (0, 9));
        for size in [Vec2::new(300.0, 70.0), Vec2::new(300.0, 90.0), Vec2::new(60.0, 500.0)] {
            w.set_viewport(Rect::from_min_size(Pos2::new(300.0, 0.0), size), size + Vec2::splat(100.0), 1.0);
            run(&mut w, 0.5, 1.0 / 60.0);
            assert!(w.box_rect().is_none(), "{size:?}");
            assert!(!w.is_animating(), "{size:?}");
        }
        // and come back when there is room again
        w.set_viewport(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0)), Vec2::new(1100.0, 660.0), 1.0);
        w.update(1.0 / 60.0);
        assert!(w.box_rect().is_some() && w.is_animating());
    }

    #[test]
    fn region_waves_average_their_cells() {
        // brute force: every cell of rows 3..=9 whose centre lies between the marks
        let (m, a, b) = (5u64, (9u64, 2u64), (3u64, 2u64)); // x2 = 2k − n: −5 and 1
        let mut w = overlay();
        w.set_region(m, a, b);
        w.update(1.0 / 60.0);
        let mut want: HashMap<i64, f64> = HashMap::new();
        let mut cells = 0u64;
        for n in 3..=9u32 {
            for k in 0..=n {
                let x2 = 2 * k as i64 - n as i64;
                if (-5..=1).contains(&x2) {
                    cells += 1;
                    *want.entry(n as i64 - 2 * k as i64).or_default() += (binom_exact(n, k).unwrap() % m as u128) as f64;
                }
            }
        }
        // the target holds the cells' sum: the average times N
        for f in -12..=12i64 {
            let got = w.target(f) as f64 / cells as f64;
            let exp = want.get(&f).copied().unwrap_or(0.0) / cells as f64;
            assert!((got - exp).abs() < 1e-6, "f = {f}: {got} vs {exp}");
        }
        assert_eq!(w.label(), "rows 3–9 · mod 5 · selection");
        assert_eq!(w.note(), format!("average of {cells} cells"));
        // the marks in either order, and one cell alone
        let mut w2 = overlay();
        w2.set_region(m, b, a);
        assert_eq!(w2.source, w.source);
        w2.set_region(7, (6, 2), (6, 2));
        w2.update(1.0 / 60.0);
        assert_eq!(w2.target(2), 1.0, "C(6,2) = 15 ≡ 1 mod 7, at frequency 2");
        assert_eq!(w2.target(4), 0.0);
        assert_eq!(w2.note(), "average of 1 cell");
        // a region off the triangle has no cells and stays flat
        w2.set_region(7, (6, 0), (2, 0));
        w2.update(1.0 / 60.0);
        assert!(w2.samples().iter().all(|s| s.is_finite()));
    }

    /// The exact average (as a sum per frequency) of a region, by brute force.
    fn region_exact(m: u64, a: (u64, u64), b: (u64, u64)) -> HashMap<i64, f64> {
        let (n_lo, n_hi) = (a.0.min(b.0), a.0.max(b.0));
        let x2 = |(n, k): (u64, u64)| 2 * k as i64 - n as i64;
        let (lo, hi) = (x2(a).min(x2(b)), x2(a).max(x2(b)));
        let mut pr = PascalRow::default();
        let mut out = HashMap::new();
        for n in n_lo..=n_hi {
            let row = pr.get(n, m).to_vec();
            for (k, &r) in row.iter().enumerate() {
                let x = 2 * k as i64 - n as i64;
                if (lo..=hi).contains(&x) && r != 0 {
                    *out.entry(-x).or_insert(0.0) += r as f64;
                }
            }
        }
        out
    }

    /// Relative L2 distance between two spectra after scaling each to unit length (the
    /// plot is normalized, so only the shape counts).
    fn shape_error(a: &HashMap<i64, f64>, b: &HashMap<i64, f64>) -> f64 {
        let na = a.values().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
        let nb = b.values().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
        let keys: std::collections::HashSet<i64> = a.keys().chain(b.keys()).copied().collect();
        keys.iter().map(|f| (a.get(f).unwrap_or(&0.0) / na - b.get(f).unwrap_or(&0.0) / nb).powi(2)).sum::<f64>().sqrt()
    }

    #[test]
    fn big_regions_are_sampled_within_budget_and_keep_their_shape() {
        // the reviewer's cases, within Pascal's range: summed exactly now
        for (m, a, b) in [(3u64, (0u64, 0u64), (511u64, 511u64)), (7, (0, 0), (511, 511)), (2, (0, 0), (999, 999)), (5, (0, 0), (1023, 0))] {
            let mut w = overlay();
            w.set_region(m, a, b);
            w.update(1.0 / 60.0);
            let st = w.region_stats;
            assert_eq!(st.cells, st.total, "{st:?}");
            let got: HashMap<i64, f64> = w.target.iter().map(|(&f, &v)| (f, v as f64)).collect();
            let err = shape_error(&got, &region_exact(m, a, b));
            assert!(err < 1e-6, "m = {m} {a:?}–{b:?}: shape error {err}");
        }
        // deep regions within the cone budget are exact too
        for (m, a, b) in [(3u64, (2100u64, 0u64), (4000u64, 4000u64)), (7, (2100, 300), (4000, 2500)), (2, (3000, 1400), (3010, 1600)), (6, (2049, 0), (2049, 2049))] {
            let mut w = overlay();
            w.set_region(m, a, b);
            w.update(1.0 / 60.0);
            let st = w.region_stats;
            assert_eq!(st.cells, st.total, "{st:?}");
            let got: HashMap<i64, f64> = w.target.iter().map(|(&f, &v)| (f, v as f64)).collect();
            let err = shape_error(&got, &region_exact(m, a, b));
            assert!(err < 1e-5, "m = {m} {a:?}–{b:?}: shape error {err}");
        }
        // past it they are sampled in bands of whole consecutive rows, summed exactly:
        // both parities kept, and the shape near the exact average
        for (m, a, b) in [(3u64, (3000u64, 0u64), (9000u64, 9000u64)), (7, (3000, 0), (9000, 9000)), (2, (2500, 0), (8500, 8500))] {
            let mut w = overlay();
            w.set_region(m, a, b);
            let t = std::time::Instant::now();
            w.update(1.0 / 60.0);
            let took = t.elapsed();
            let st = w.region_stats;
            assert!(st.cells < st.total, "{st:?}");
            assert!(w.note().contains("sampled"), "{}", w.note());
            let got: HashMap<i64, f64> = w.target.iter().map(|(&f, &v)| (f, v as f64)).collect();
            assert!(got.keys().any(|f| f.rem_euclid(2) == 0) && got.keys().any(|f| f.rem_euclid(2) == 1), "both parities");
            let err = shape_error(&got, &region_exact(m, a, b));
            assert!(err < 0.35, "m = {m} {a:?}–{b:?}: shape error {err} ({st:?})");
            assert!(took.as_secs_f32() < 1.0, "{took:?}");
        }
        // the row-sampling fallback for regions too wide for a band: strides coprime
        // with 2m and both parities
        let mut w = overlay();
        w.set_region(3, (500_000, 0), (900_000, 900_000));
        w.update(1.0 / 60.0);
        let st = w.region_stats;
        assert!(gcd(st.row_step, 6) == 1 && gcd(st.cell_step, 6) == 1 && st.cells < st.total, "{st:?}");
        assert!(w.target.keys().any(|f| f.rem_euclid(2) == 0) && w.target.keys().any(|f| f.rem_euclid(2) == 1));
        // deep rows through the closed form, a wide window, still bounded
        let mut w = overlay();
        w.set_region(30, (1_000_000_000, 400_000_000), (999_000_000, 600_000_000));
        let t = std::time::Instant::now();
        w.update(1.0 / 60.0);
        assert!(w.region_stats.cells <= REGION_CELL_BUDGET + REGION_CELL_BUDGET / 8, "{:?}", w.region_stats);
        assert!(t.elapsed().as_secs_f32() < 5.0);
        assert!(w.samples().iter().all(|s| s.is_finite()));
    }

    #[test]
    fn wide_small_modulus_regions_keep_every_term() {
        // averages over tens of thousands of cells used to fall under DROP_EPS; and a
        // region past row 2000 draws all of its terms (no top-K)
        for (m, a, b) in [(2u64, (1500u64, 0u64), (1530u64, 1530u64)), (3, (1500, 0), (2400, 2400))] {
            let mut w = overlay();
            w.set_region(m, a, b);
            run(&mut w, 1.0, 1.0 / 60.0);
            let exact = region_exact(m, a, b);
            let st = w.region_stats;
            if st.cells == st.total {
                assert_eq!(w.active_len(), exact.len(), "every frequency drawn");
            }
            let got: HashMap<i64, f64> = w.displayed.iter().map(|(&f, &v)| (f, v as f64)).collect();
            let err = shape_error(&got, &exact);
            assert!(err < 0.5, "m = {m}: shape error {err}");
        }
    }

    #[test]
    fn without_smoothing_waves_snap() {
        let mut w = overlay();
        w.config.smooth = false;
        w.set_source(6, 5, (0, 6));
        w.update(1.0 / 60.0);
        for f in [-6i64, -4, -2, 0, 2, 4, 6] {
            assert_eq!(w.displayed(f), w.target(f), "f = {f}");
        }
        let norm6 = w.norm();
        w.set_source(7, 5, (0, 7));
        w.update(1.0 / 60.0);
        assert_eq!(w.displayed(6), 0.0, "the old row is gone at once");
        assert_eq!(w.displayed(5), w.target(5));
        assert!(w.norm() != norm6);
        // and with it, one frame only gets part of the way
        let mut w = overlay();
        w.set_source(6, 5, (0, 6));
        w.update(1.0 / 60.0);
        assert!(w.displayed(6) < w.target(6));
    }

    #[test]
    fn hiding_fades_out() {
        let mut w = overlay();
        w.set_source(8, 3, (0, 8));
        run(&mut w, 1.0, 1.0 / 60.0);
        assert!(w.box_rect().is_some());
        assert!(w.is_animating(), "the phase drifts");
        w.toggle();
        w.update(1.0 / 60.0);
        assert!(w.fade() > 0.5, "fades, does not pop");
        run(&mut w, 2.0, 1.0 / 60.0);
        assert!(w.box_rect().is_none());
        assert!(!w.is_animating(), "a hidden overlay needs no redraws");
        // with the drift off, a settled overlay stops asking for frames
        let mut w = overlay();
        w.config.drift = 0.0;
        w.set_source(8, 3, (0, 8));
        run(&mut w, 3.0, 1.0 / 60.0);
        assert!(!w.is_animating());
    }

    #[test]
    fn box_follows_the_window_and_stays_inside_the_canvas() {
        let mut w = overlay();
        w.set_source(5, 3, (0, 5));
        run(&mut w, 1.0, 1.0 / 60.0);
        // 28% of min(1300, 760) = 212.8
        let b = w.box_rect().unwrap();
        assert!((b.width() - 212.8).abs() < 1e-3 && (b.width() - b.height()).abs() < 1e-3, "{b:?}");
        assert_eq!(b.max.x, 1000.0 - 16.0);
        // clamped to [160, 420]
        w.set_viewport(Rect::from_min_size(Pos2::ZERO, Vec2::new(3000.0, 2000.0)), Vec2::new(3300.0, 2100.0), 1.0);
        assert_eq!(w.box_rect().unwrap().width(), 420.0);
        w.set_viewport(Rect::from_min_size(Pos2::ZERO, Vec2::new(400.0, 300.0)), Vec2::new(500.0, 360.0), 1.0);
        assert_eq!(w.box_rect().unwrap().width(), 160.0);
        // and never larger than the canvas
        let canvas = Rect::from_min_size(Pos2::new(300.0, 0.0), Vec2::new(150.0, 900.0));
        w.set_viewport(canvas, Vec2::new(450.0, 960.0), 1.0);
        assert!(canvas.contains_rect(w.box_rect().unwrap()));
        for c in Corner::ALL {
            w.config.corner = c;
            w.set_viewport(Rect::from_min_size(Pos2::new(300.0, 0.0), Vec2::new(980.0, 736.0)), Vec2::new(1280.0, 800.0), 1.0);
            let b = w.box_rect().unwrap();
            assert!(Rect::from_min_size(Pos2::new(300.0, 0.0), Vec2::new(980.0, 736.0)).contains_rect(b), "{c:?}");
        }
    }
}

