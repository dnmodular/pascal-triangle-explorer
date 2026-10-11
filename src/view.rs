//! Camera and the cell/texel geometry shared by the CPU (hover, numbers, status) and
//! the shaders (uniforms).
//!
//! World coordinates: cell `(n, k)` is a brick of width 1 (cell widths) centered at
//! `x = k − n/2`, occupying rows `[n, n + 1)` in units of rows; a row is `ROW_H = √3/2`
//! cell widths tall, so the bricks are equilateral-triangle-shaped tiles. Row `n + 1` is
//! offset half a cell from row `n`.
//!
//! The view is stored in f64 (`x0`, `y0`, `ppc`) and handed to the GPU split into integer
//! and fractional parts, so rows into the billions stay exact in 32-bit shader math.

/// Row height in cell widths (√3 × half a cell width).
pub const ROW_H: f64 = 0.866_025_403_784_438_6;
/// Deepest row the view can reach (a billion rows; the shader keeps `x0 + n/2` inside i32).
pub const MAX_ROW: f64 = 1_000_000_000.0;
pub const MIN_PPC: f64 = 1e-7;
/// Largest zoom: enough for a 2-row fit to fill an 8K-tall canvas.
pub const MAX_PPC: f64 = 4096.0;
/// Padding `fit` leaves at the canvas's sides, and the height of the status bar (its
/// own strip under the canvas, so the canvas is exactly the visible view).
pub const FIT_PAD: f64 = 12.0;
pub const STATUS_H: f64 = 18.0;
/// Slack for row edges that fall on a canvas edge up to float rounding.
const EDGE_EPS: f64 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// World x (cell widths) at the left edge of the canvas.
    pub x0: f64,
    /// World y (rows) at the top edge of the canvas.
    pub y0: f64,
    /// Pixels per cell width (zoom).
    pub ppc: f64,
}

impl View {
    /// Fit rows `0..rows` into a `w × h` canvas: the apex on the top edge, centered
    /// horizontally. When height decides the scale (any canvas wider than the triangle's
    /// 2 : √3), row `rows − 1` ends exactly on the bottom edge, so the view shows those
    /// rows and nothing more; a narrower canvas shows the triangle's full width and
    /// further rows below it.
    pub fn fit(w: f64, h: f64, rows: f64) -> View {
        let rows = rows.max(2.0);
        let aw = (w - 2.0 * FIT_PAD).max(1.0);
        let ah = h.max(1.0);
        let ppc = (aw / rows).min(ah / (rows * ROW_H)).clamp(MIN_PPC, MAX_PPC);
        View { x0: -w / (2.0 * ppc), y0: 0.0, ppc }
    }

    pub fn cells_per_px(&self) -> f64 {
        1.0 / self.ppc
    }

    pub fn rows_per_px(&self) -> f64 {
        1.0 / (self.ppc * ROW_H)
    }

    pub fn pan(&mut self, dx_px: f64, dy_px: f64) {
        self.x0 -= dx_px * self.cells_per_px();
        self.y0 -= dy_px * self.rows_per_px();
        self.clamp();
    }

    /// Zoom by `factor` keeping the world point under pixel `(px, py)` fixed.
    pub fn zoom_at(&mut self, px: f64, py: f64, factor: f64) {
        let new_ppc = (self.ppc * factor).clamp(MIN_PPC, MAX_PPC);
        let wx = self.x0 + px * self.cells_per_px();
        let wy = self.y0 + py * self.rows_per_px();
        self.ppc = new_ppc;
        self.x0 = wx - px * self.cells_per_px();
        self.y0 = wy - py * self.rows_per_px();
        self.clamp();
    }

    pub fn clamp(&mut self) {
        self.ppc = self.ppc.clamp(MIN_PPC, MAX_PPC);
        self.y0 = self.y0.clamp(-MAX_ROW, MAX_ROW);
        self.x0 = self.x0.clamp(-MAX_ROW, MAX_ROW);
    }

    /// Rows visible in a canvas of height `h`.
    pub fn rows_visible(&self, h: f64) -> f64 {
        h * self.rows_per_px()
    }

    /// The row under canvas pixel row `py` (inside or outside the triangle), if any.
    pub fn row_at(&self, py: f64) -> Option<u32> {
        let wy = self.y0 + py * self.rows_per_px();
        (0.0..MAX_ROW).contains(&wy).then(|| wy.floor() as u32)
    }

    /// The `k` range of row `n` that intersects a canvas of width `w` (clamped to the
    /// row, so it is never empty).
    pub fn visible_k(&self, n: u32, w: f64) -> (u32, u32) {
        let nf = n as f64;
        let lo = (self.x0 + nf / 2.0 - 0.5).ceil().clamp(0.0, nf);
        let hi = (self.x0 + w * self.cells_per_px() + nf / 2.0 + 0.5).floor().clamp(lo, nf);
        (lo as u32, hi as u32)
    }

    /// The cell under pixel `(px, py)`, if inside the triangle.
    pub fn cell_at(&self, px: f64, py: f64) -> Option<(u32, u32)> {
        let n = self.row_at(py)? as f64;
        let wx = self.x0 + px * self.cells_per_px();
        let k = (wx + n / 2.0 + 0.5).floor();
        if k < 0.0 || k > n {
            return None;
        }
        Some((n as u32, k as u32))
    }

    /// Screen rectangle `(x, y, w, h)` of cell `(n, k)`.
    pub fn cell_rect(&self, n: u32, k: u32) -> (f64, f64, f64, f64) {
        let x = (k as f64 - n as f64 / 2.0 - 0.5 - self.x0) * self.ppc;
        let y = (n as f64 - self.y0) * self.ppc * ROW_H;
        (x, y, self.ppc, self.ppc * ROW_H)
    }

    /// Inclusive range of (non-negative) rows that intersect a canvas of height `h`.
    pub fn visible_rows(&self, h: f64) -> Option<(u32, u32)> {
        let lo = self.y0.max(0.0);
        let hi = (self.y0 + h * self.rows_per_px()).min(MAX_ROW);
        if hi <= 0.0 || lo >= hi {
            return None;
        }
        // a row edge on the canvas edge (up to rounding) does not make the next row visible
        let first = (lo + EDGE_EPS).floor().min(hi);
        Some((first as u32, ((hi - EDGE_EPS).ceil() - 1.0).max(first) as u32))
    }

    /// Inclusive range of rows wholly inside a canvas of height `h` (None when no row
    /// fits completely, e.g. zoomed in past a single row).
    pub fn full_rows(&self, h: f64) -> Option<(u32, u32)> {
        let lo = (self.y0 - EDGE_EPS).ceil().max(0.0);
        let hi = ((self.y0 + h * self.rows_per_px() + EDGE_EPS).floor() - 1.0).min(MAX_ROW);
        (hi >= lo).then_some((lo as u32, hi as u32))
    }

    /// Number of cells inside the triangle that intersect the `w × h` canvas (exact for
    /// up to 200k visible rows, sampled every stride-th row beyond that).
    pub fn visible_cells(&self, w: f64, h: f64) -> u64 {
        let Some((n_lo, n_hi)) = self.visible_rows(h) else { return 0 };
        let x1 = self.x0 + w * self.cells_per_px();
        let rows = (n_hi - n_lo) as u64 + 1;
        let stride = (rows / 200_000).max(1);
        let mut total = 0u64;
        let mut n = n_lo as u64;
        while n <= n_hi as u64 {
            let nf = n as f64;
            let k_lo = (self.x0 + nf / 2.0 - 0.5).ceil().max(0.0);
            let k_hi = (x1 + nf / 2.0 + 0.5).floor().min(nf);
            if k_hi >= k_lo {
                total += (k_hi - k_lo) as u64 + 1;
            }
            n += stride;
        }
        total * stride
    }
}

/// How the visible region is tiled into the grid the whole GPU pipeline runs on.
///
/// Texel `(i, j)` samples row `n = n0 + ⌊(j + ½)·sy⌋` and cell `k = k0(n) + ⌊(i + ½)·sx⌋`
/// where `k0(n) = ⌊x0 + n/2 + ½⌋` is the leftmost cell of that row touching the view.
/// Zoomed in (`sx = sy = 1`) a texel is exactly one cell and neighbours on the grid are
/// lattice neighbours; zoomed out a texel is `1/ss` of a pixel and samples one cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridSpec {
    pub w: u32,
    pub h: u32,
    /// Cells per texel horizontally, rows per texel vertically (both ≥ 1).
    pub sx: f32,
    pub sy: f32,
    /// First row (`⌊y0⌋`) and the fraction above it; may be negative above the apex.
    pub n0: i32,
    pub y0f: f32,
    /// `⌊x0⌋` and its fraction.
    pub x0i: i32,
    pub x0f: f32,
    pub cells_per_px: f32,
    pub rows_per_px: f32,
    /// Sub-samples per pixel edge in the resolve pass (1 when zoomed in).
    pub ss: u32,
    pub canvas_w: u32,
    pub canvas_h: u32,
}

impl GridSpec {
    /// `ss_pref` is the user's supersampling choice; the grid falls back to 1 sample
    /// when the texel count would exceed `max_texels`.
    pub fn new(view: &View, canvas_w: u32, canvas_h: u32, ss_pref: u32, max_texels: u64) -> GridSpec {
        let cpp = view.cells_per_px();
        let rpp = view.rows_per_px();
        // Cells at least a pixel wide and rows at least a pixel tall: one texel per cell.
        let zoomed_in = view.ppc * ROW_H >= 1.0;
        let mut ss = if zoomed_in { 1 } else { ss_pref.clamp(1, 4) };
        loop {
            let sx = (cpp / ss as f64).max(1.0);
            let sy = (rpp / ss as f64).max(1.0);
            let w = (canvas_w as f64 * cpp / sx).ceil() as u32 + 2;
            let h = (canvas_h as f64 * rpp / sy).ceil() as u32 + 2;
            if (w as u64) * (h as u64) > max_texels && ss > 1 {
                ss -= 1;
                continue;
            }
            let n0 = view.y0.floor();
            let x0i = view.x0.floor();
            return GridSpec {
                w,
                h,
                sx: sx as f32,
                sy: sy as f32,
                n0: n0 as i32,
                y0f: (view.y0 - n0) as f32,
                x0i: x0i as i32,
                x0f: (view.x0 - x0i) as f32,
                cells_per_px: cpp as f32,
                rows_per_px: rpp as f32,
                ss,
                canvas_w,
                canvas_h,
            };
        }
    }

    pub fn texels(&self) -> u64 {
        self.w as u64 * self.h as u64
    }

    /// Leftmost cell index of row `n` touching the view (`⌊x0 + n/2 + ½⌋`), in the same
    /// integer/fraction arithmetic the shader uses.
    pub fn k0(&self, n: u32) -> i64 {
        let half = if n & 1 == 1 { 0.5f32 } else { 0.0 };
        self.x0i as i64 + (n >> 1) as i64 + (self.x0f + half + 0.5).floor() as i64
    }

    /// CPU twin of the shader's `cell_of(i, j)`: the sampled `(n, k)` of a texel, or
    /// `None` outside the triangle.
    pub fn cell_of(&self, i: u32, j: u32) -> Option<(u32, u32)> {
        let n = self.n0 as i64 + ((j as f32 + 0.5) * self.sy).floor() as i64;
        if n < 0 || n > MAX_ROW as i64 {
            return None;
        }
        let k = self.k0(n as u32) + ((i as f32 + 0.5) * self.sx).floor() as i64;
        if k < 0 || k > n {
            return None;
        }
        Some((n as u32, k as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_and_cell_lookup_roundtrip() {
        let v = View::fit(800.0, 600.0, 64.0);
        for n in 0..64u32 {
            for k in 0..=n {
                let (x, y, w, h) = v.cell_rect(n, k);
                assert_eq!(v.cell_at(x + w / 2.0, y + h / 2.0), Some((n, k)));
            }
        }
        assert_eq!(v.cell_at(400.0, 2.0), Some((0, 0)), "the apex touches the top edge");
        // fitting R rows shows exactly rows 0..R−1, whole (no part of row R)
        for (w, h, rows) in [(800.0, 600.0, 64.0), (980.0, 736.0, 2.0), (980.0, 736.0, 128.0), (1280.0, 700.0, 7.0)] {
            let v = View::fit(w, h, rows);
            let last = rows as u32 - 1;
            assert_eq!(v.visible_rows(h), Some((0, last)), "{w}x{h} {rows} rows");
            assert_eq!(v.full_rows(h), Some((0, last)), "{w}x{h} {rows} rows");
        }
        // a canvas narrower than the triangle shows its full width and more rows below
        let v = View::fit(300.0, 900.0, 64.0);
        assert!(v.visible_rows(900.0).unwrap().1 > 63);
        // zoomed into the middle of one row: visible, but not whole
        let v = View { x0: 0.0, y0: 10.3, ppc: 1000.0 };
        assert_eq!(v.visible_rows(500.0), Some((10, 10)));
        assert_eq!(v.full_rows(500.0), None);
    }

    #[test]
    fn row_and_k_window_lookup() {
        let v = View::fit(800.0, 600.0, 64.0);
        // every cell's row is found from its centre, inside the triangle or beside it
        for n in [0u32, 1, 31, 63] {
            let (_, y, _, h) = v.cell_rect(n, 0);
            assert_eq!(v.row_at(y + h / 2.0), Some(n));
        }
        assert_eq!(v.row_at(-1e6), None);
        // the fitted view shows whole rows
        assert_eq!(v.visible_k(40, 800.0), (0, 40));
        // a view panned off to one side clamps to an end of the row, never inverts
        let right = View { x0: 1e6, ..v };
        let left = View { x0: -1e6, ..v };
        assert_eq!(right.visible_k(40, 800.0), (40, 40));
        assert_eq!(left.visible_k(40, 800.0), (0, 0));
        // zoomed into the middle of a deep row
        let deep = View { x0: -10.0, y0: 1e9 - 100.0, ppc: 20.0 };
        let (lo, hi) = deep.visible_k(999_999_950, 800.0);
        assert!(lo < 499_999_975 && hi > 499_999_975 && hi - lo <= 42, "{lo}..{hi}");
    }

    #[test]
    fn grid_is_one_texel_per_cell_when_zoomed_in() {
        let v = View { x0: -40.0, y0: 0.0, ppc: 10.0 };
        let g = GridSpec::new(&v, 800, 600, 2, 1 << 24);
        assert_eq!((g.sx, g.sy, g.ss), (1.0, 1.0, 1));
        assert_eq!(g.w, 82);
        for j in 0..g.h {
            for i in 0..g.w {
                if let Some((n, k)) = g.cell_of(i, j) {
                    assert_eq!(n, j);
                    let k0 = (v.x0 + n as f64 / 2.0 + 0.5).floor() as i64;
                    assert_eq!(k as i64, k0 + i as i64);
                }
            }
        }
    }

    #[test]
    fn zoom_keeps_cursor_point_fixed() {
        let mut v = View::fit(800.0, 600.0, 100.0);
        let before = v.cell_at(300.0, 250.0);
        v.zoom_at(300.0, 250.0, 1.0001);
        assert_eq!(v.cell_at(300.0, 250.0), before);
    }
}
