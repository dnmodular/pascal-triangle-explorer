//! Everything the user can change, and how the modulus moves: the model the UI edits and
//! the renderer reads.

use super::*;

/// The ghost wave's settings (see [`WaveOverlay::set_ghost`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GhostWave {
    pub(crate) on: bool,
    pub(crate) modulus: u32,
    /// Its first and last rows, typed in (either order).
    pub(crate) rows: (u32, u32),
}

impl Default for GhostWave {
    fn default() -> Self {
        GhostWave { on: false, modulus: 2, rows: (0, 10) }
    }
}

impl GhostWave {
    /// The ghost's (modulus, rows), or None when it is off.
    pub(crate) fn source(&self) -> Option<(u64, (u64, u64))> {
        let (a, b) = self.rows;
        self.on.then_some((self.modulus.max(2) as u64, (a.min(b) as u64, a.max(b) as u64)))
    }
}

/// Picking rows whose average wave the overlay plays (like the pin, but for every cell
/// of the rows from a red start mark's to a blue end mark's).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum WaveSelect {
    #[default]
    Off,
    /// Armed: the next click on a cell marks the start (red).
    PickStart,
    /// The next click marks the end (blue).
    PickEnd((u32, u32)),
    /// The overlay plays the cells between the two marks.
    Active((u32, u32), (u32, u32)),
}

/// How a fraction μ = P/Q colors its cells (see `CanvasStyle::phase`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum FracColors {
    /// By the residue's value C mod μ: continuous in μ, but below μ ≈ 3 nearly one color.
    Value,
    /// By phase: the scaled residue Q·(C mod μ) colored as mod P colors it.
    Phase,
    /// Phase below μ = 3, value from there.
    #[default]
    Auto,
}

impl FracColors {
    pub(crate) const ALL: [FracColors; 3] = [FracColors::Auto, FracColors::Value, FracColors::Phase];
    pub(crate) fn label(self) -> &'static str {
        match self {
            FracColors::Value => "value",
            FracColors::Phase => "phase",
            FracColors::Auto => "auto",
        }
    }
    pub(crate) fn tip(self) -> &'static str {
        match self {
            FracColors::Value => "Fractions are colored by the residue's value x = C mod μ along the ramp: continuous as μ glides, but below μ ≈ 3 the values crowd one end of the ramp, so the picture is nearly one color (10/7 and 10/9 look alike)",
            FracColors::Phase => "Fractions μ = P/Q are colored by phase: the cell's Q·(C mod μ), a whole number below P, colored as mod P colors it. Every fraction with numerator P then uses mod P's colors, shuffled by Q (10/7 and 10/9 differ)",
            FracColors::Auto => "Phase below μ = 3, where the value colors run out of range; value from 3 up",
        }
    }
}

/// Everything the user can change.
pub(crate) struct Model {
    /// The whole modulus the sweep is at, and how far past it a decimal modulus is:
    /// μ = m + sub·10⁻ᵈ (d = the sweep's decimals; sub = 0 for a whole modulus).
    pub(crate) m: u32,
    pub(crate) sub: u64,
    /// A morph's second end: the next visited whole modulus and how far μ is from m to
    /// it (the sweep's morph mode, between two whole moduli).
    pub(crate) morph: Option<(Arc<Modulus>, f32)>,
    /// The super-prime walk's position (visit mode "super primes"), and a fraction typed
    /// in that is not on the sweep's grid (then shown as it is; a step leaves it).
    pub(crate) sp_index: usize,
    pub(crate) free_fraction: bool,
    /// The exact decimal at the sweep's position needed a numerator past MOD_CAP: the
    /// nearest fraction that fits is drawn.
    pub(crate) rounded: bool,
    /// μ's closed form: the whole m, or the decimal μ = P/Q (see `Modulus::decimal`).
    /// Shared with the wave overlay, which reads long rows through the same closed form.
    pub(crate) modulus: Arc<Modulus>,
    pub(crate) m_text: String,
    pub(crate) scheme: Scheme,
    pub(crate) palette: Palette,
    /// Color offset and cell decorations (outlines, glow around zeros).
    pub(crate) style: CanvasStyle,
    pub(crate) frac_colors: FracColors,
    /// The offset cycles on its own while this is on: one full turn per period.
    pub(crate) offset_playing: bool,
    pub(crate) offset_period: f32,
    /// The offset field is being typed into or dragged: the cycle waits.
    pub(crate) offset_editing: bool,
    pub(crate) layers: LayerState,
    pub(crate) view: View,
    pub(crate) fit_rows: f64,
    pub(crate) auto_fit: bool,
    /// Rows parented to the modulus: rows = m × link_factor, refitted whenever m moves
    /// (the reference's LINK). While the sweep plays they follow too, unless held.
    pub(crate) link_rows: bool,
    pub(crate) link_factor: f64,
    pub(crate) link_hold_sweep: bool,
    /// The canvas size (px) of the last frame, for fits made outside the canvas UI.
    pub(crate) canvas: (f64, f64),
    /// The canvas alone, borderless over the whole screen (F11; Esc returns).
    pub(crate) fullscreen: bool,
    pub(crate) ss: u32,
    pub(crate) show_numbers: bool,
    pub(crate) export_dir: String,
    pub(crate) export_note: String,
    pub(crate) pending_export: bool,
    /// Save SVG was clicked: rows to write once the frame has rendered (the valuation
    /// coloring reads the frame's max ν back).
    pub(crate) pending_svg: Option<u32>,
    pub(crate) hover: Option<(u32, u32)>,
    /// The row under the pointer, inside the triangle or beside it.
    pub(crate) hover_row: Option<u32>,
    pub(crate) sweep: Sweep,
    /// When the play timer last advanced the modulus.
    pub(crate) last_step: Option<Instant>,
    /// The "modular wave" overlay and the row it plays: the pinned row (click a cell),
    /// else the hovered row, else the last visible row.
    pub(crate) wave: WaveOverlay,
    pub(crate) pinned_row: Option<u32>,
    pub(crate) wave_select: WaveSelect,
    pub(crate) wave_row: u32,
    /// The ghost wave: shown, its modulus and its rows (first, last).
    pub(crate) ghost: GhostWave,
    /// The pointer is over the overlay box: the canvas under it gets no hover.
    pub(crate) over_wave: bool,
    /// The pin and selection before the latest click (a double click undoes its first
    /// click, whichever it was).
    pub(crate) pin_before_click: Option<u32>,
    pub(crate) select_before_click: WaveSelect,
    /// The canvas is being dragged (the wave keeps its row while the view pans).
    pub(crate) dragging: bool,
    /// A widget had keyboard focus or a popup was open last frame (its Esc is not ours).
    pub(crate) focus_last_frame: bool,
    /// The tetra stack: its settings, preview tile and 3D space.
    pub(crate) tetra: TetraView,
}

impl Model {
    pub(crate) fn new() -> Self {
        let m = 7;
        Model {
            m,
            sub: 0,
            morph: None,
            sp_index: 0,
            free_fraction: false,
            rounded: false,
            modulus: Arc::new(Modulus::new(m)),
            m_text: m.to_string(),
            scheme: Scheme::Spectral,
            palette: Palette::new(Scheme::Spectral),
            style: CanvasStyle::default(),
            frac_colors: FracColors::default(),
            offset_playing: false,
            offset_period: 4.0,
            offset_editing: false,
            layers: LayerState::default(),
            view: View::fit(800.0, 600.0, 128.0),
            fit_rows: 128.0,
            auto_fit: true,
            link_rows: false,
            link_factor: 1.0,
            link_hold_sweep: false,
            canvas: (0.0, 0.0),
            fullscreen: false,
            ss: 2,
            show_numbers: true,
            export_dir: std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| ".".into()),
            export_note: String::new(),
            pending_export: false,
            pending_svg: None,
            hover: None,
            hover_row: None,
            sweep: Sweep::default(),
            last_step: None,
            wave: WaveOverlay::new(),
            pinned_row: None,
            wave_select: WaveSelect::Off,
            wave_row: 0,
            ghost: GhostWave::default(),
            over_wave: false,
            pin_before_click: None,
            select_before_click: WaveSelect::Off,
            dragging: false,
            focus_last_frame: false,
            tetra: TetraView::new(),
        }
    }

    pub(crate) fn set_m(&mut self, m: u32) {
        self.set_m_with(m, true);
    }

    /// Set the (whole) modulus; with `follow`, linked rows follow it.
    pub(crate) fn set_m_with(&mut self, m: u32, follow: bool) {
        self.set_pos(m, 0, follow);
    }

    /// Go to μ = p/q exactly (lowest terms taken; p ≥ 1 within MOD_CAP, so μ > 0), off
    /// the sweep's grid; with `follow`, linked rows follow it.
    pub(crate) fn set_fraction(&mut self, p: u32, q: u32, follow: bool) {
        // (lowest terms first: the tables are built only when μ changes)
        let g = math::gcd_u64(p as u64, q.max(1) as u64).max(1) as u32;
        let (p, q) = (p / g, q.max(1) / g);
        let changed = (p, q) != (self.modulus.m, self.modulus.q) || self.morph.is_some();
        if changed {
            self.modulus = if q == 1 { self.whole(p) } else { Arc::new(Modulus::decimal(p, q)) };
        }
        self.m = (p / q).max(1);
        self.sub = 0;
        self.morph = None;
        self.free_fraction = q > 1;
        self.m_text = self.modulus.label();
        if changed && follow {
            self.apply_link();
        }
    }

    /// Go to super prime number `i` of the walk.
    pub(crate) fn set_super(&mut self, i: usize, follow: bool) {
        let list = self.sweep.super_list();
        let Some(&(p, q)) = list.get(i.min(list.len().saturating_sub(1))) else { return };
        self.sp_index = i.min(list.len() - 1);
        self.set_fraction(p, q, follow);
        if (self.modulus.m, self.modulus.q) != (p, q) {
            // an unreduced fraction of the walk: shown as listed (10/4 draws 5/2)
            self.m_text = format!("{p}/{q}");
        }
    }

    /// The walk is a list of fractions (super primes, or mode All by numerator).
    pub(crate) fn super_mode(&self) -> bool {
        self.sweep.fraction_mode()
    }

    /// Go to μ = m + sub·10⁻ᵈ; with `follow`, linked rows follow it. A μ whose
    /// numerator would pass MOD_CAP falls back to the whole m.
    pub(crate) fn set_pos(&mut self, m: u32, sub: u64, follow: bool) {
        let d = self.sweep.decimals;
        let sub = if d == 0 { 0 } else { sub };
        // below 1 the position is 0.sub (μ = 0 itself is no modulus: C mod 0 is C)
        let (m, sub) = match (m.min(MOD_CAP), sub) {
            (0, 0) if d > 0 => (0, 1),
            (0, 0) => (1, 0),
            x => x,
        };
        // (a morph runs between whole moduli: below 1 there is none to start from)
        let morphing = self.sweep.morph && sub > 0 && m >= 1;
        if sub == 0 || morphing {
            self.rounded = false;
        }
        let changed = (m, sub) != (self.m, self.sub) || (sub == 0 && (self.modulus.is_decimal() || self.morph.is_some())) || (sub > 0 && morphing != self.morph.is_some()) || self.free_fraction;
        self.free_fraction = false;
        if changed {
            self.m = m;
            self.sub = sub;
            if morphing {
                // from the whole m to the next visited modulus, t of the way
                let gap = self.sweep.gap_after(m);
                let t = sub as f64 / (gap as u64 * self.sweep.fine_units()) as f64;
                // (both ends are reused while the walk stays in this gap)
                let (base, target) = (self.whole(m), self.whole((m + gap).min(MOD_CAP)));
                self.modulus = base;
                self.morph = Some((target, t as f32));
            } else if sub == 0 {
                self.modulus = self.whole(m);
                self.morph = None;
                self.rounded = false;
            } else {
                self.morph = None;
                let (md, rounded) = Modulus::from_decimal_nearest(m, sub, d, MOD_CAP);
                self.modulus = Arc::new(md);
                self.rounded = rounded;
            }
        }
        self.m_text = self.mu_text();
        if changed && follow {
            self.apply_link();
        }
    }

    /// μ as text: the decimal modulus exactly, or the sweep's position m + sub·10⁻ᵈ
    /// (a morph's, or one whose exact numerator is past the cap).
    pub(crate) fn mu_text(&self) -> String {
        if self.morph.is_some() || (self.rounded && self.sub > 0) {
            let u = self.sweep.fine_units();
            math::decimal_text(self.m as u64 * u + self.sub, u)
        } else {
            self.modulus.label()
        }
    }

    /// μ's value: the modulus drawn, or the sweep's position while that is a morph or
    /// a rounded decimal.
    pub(crate) fn mu(&self) -> f64 {
        if self.morph.is_some() || (self.rounded && self.sub > 0) {
            self.m as f64 + self.sub as f64 / self.sweep.fine_units() as f64
        } else {
            self.modulus.mu()
        }
    }

    /// The whole modulus m's tables: the ones already held (the modulus drawn, or a
    /// morph's far end) when they are m's, else built (O(m) for a prime m).
    pub(crate) fn whole(&self, m: u32) -> Arc<Modulus> {
        let held = |md: &Arc<Modulus>| md.m == m && md.q == 1;
        if held(&self.modulus) {
            return self.modulus.clone();
        }
        match &self.morph {
            Some((b, _)) if held(b) => b.clone(),
            _ => Arc::new(Modulus::new(m)),
        }
    }

    /// The style the frame is drawn in: the canvas style with the fraction colors
    /// resolved for this μ.
    pub(crate) fn frame_style(&self) -> CanvasStyle {
        let phase = match self.frac_colors {
            FracColors::Value => false,
            FracColors::Phase => true,
            FracColors::Auto => self.modulus.mu() < 3.0,
        };
        CanvasStyle { phase: phase && self.modulus.is_decimal(), ..self.style }
    }

    /// What the frame morphs to, for the renderer.
    pub(crate) fn morph_target(&self) -> Option<(&Modulus, f32)> {
        self.morph.as_ref().map(|(b, t)| (&**b, *t))
    }

    /// A typed modulus, maybe decimal ("10.37"): the whole part snaps to the selection
    /// mode, extra digits raise the decimals (within what the range allows), and the
    /// fraction is kept within the gap to the next visited modulus.
    pub(crate) fn goto_text(&mut self, text: &str) -> bool {
        let t = text.trim();
        // a fraction, "37/5": exactly that μ (in the super-prime walk, its place there)
        if let Some((a, b)) = t.split_once('/') {
            let (Ok(a), Ok(b)) = (a.trim().parse::<u64>(), b.trim().parse::<u64>()) else { return false };
            let g = math::gcd_u64(a, b).max(1);
            let (p, q) = (a / g, b / g);
            if q == 0 || p == 0 || p > MOD_CAP as u64 || q > u32::MAX as u64 {
                return false;
            }
            if self.super_mode() {
                if let Some(i) = self.sweep.super_nearest(p as u32, q as u32) {
                    self.sp_index = i;
                }
            }
            self.set_fraction(p as u32, q as u32, true);
            return true;
        }
        if self.super_mode() {
            // a number in the super-prime walk: its nearest super prime
            let Ok(v) = t.parse::<f64>() else { return false };
            let q = 1_000_000u32;
            let Some(i) = self.sweep.super_nearest((v.max(0.0) * q as f64).round().min(u32::MAX as f64) as u32, q) else { return false };
            self.set_super(i, true);
            return true;
        }
        let (ip, fp) = t.split_once('.').unwrap_or((t, ""));
        let (Ok(whole), true) = (ip.parse::<i64>(), fp.chars().all(|c| c.is_ascii_digit())) else { return false };
        let fp = fp.trim_end_matches('0');
        // 0.x: below 1, under every mode's first modulus
        let m = if whole == 0 && !fp.is_empty() { 0 } else { self.sweep.snap(whole) };
        if fp.is_empty() || (m as i64) != whole {
            self.set_m(m);
            return true;
        }
        let want = fp.len() as u32;
        let max = self.sweep.max_decimals();
        if want > self.sweep.decimals {
            self.sweep.decimals = want.min(max);
        }
        let d = self.sweep.decimals;
        if d == 0 {
            self.set_m(m);
            return true;
        }
        let digits: String = fp.chars().chain(std::iter::repeat('0')).take(d as usize).collect();
        let sub = digits.parse::<u64>().unwrap_or(0);
        let top = self.sweep.gap_after(m) as u64 * self.sweep.fine_units();
        self.set_pos(m, sub.min(top - 1), true);
        true
    }

    /// The rows drawn on a canvas `h` px tall: the visible ones, and while fitted none
    /// past the fit (the canvas below it is background).
    pub(crate) fn shown_rows(&self, h: f64) -> Option<(u32, u32)> {
        let (a, b) = self.view.visible_rows(h)?;
        if !self.auto_fit {
            return Some((a, b));
        }
        let last = (self.fit_rows.round() as u32).saturating_sub(1);
        (a <= last).then_some((a, b.min(last)))
    }

    /// The row count the link gives for the current modulus.
    pub(crate) fn linked_rows(&self) -> f64 {
        (self.mu() * self.link_factor).round().clamp(2.0, 1.0e9)
    }

    /// With rows linked, set them from the modulus and refit. A view the user zoomed
    /// or panned (auto_fit off) keeps its zoom relative to the fit and the same spot of
    /// the triangle, scaled with it (the world origin is the apex), so stepping or
    /// sweeping a linked modulus never throws the view away.
    pub(crate) fn apply_link(&mut self) {
        if !self.link_rows {
            return;
        }
        let old_rows = self.fit_rows;
        self.fit_rows = self.linked_rows();
        let (cw, ch) = self.canvas;
        if !(cw > 0.0 && ch > 0.0) {
            self.auto_fit = true; // the first frame fits
            return;
        }
        if self.auto_fit || old_rows <= 0.0 {
            self.fit(cw, ch);
            return;
        }
        let zoom = self.view.ppc / View::fit(cw, ch, old_rows).ppc;
        let v = self.view;
        // the canvas centre in world units (cells, rows), per row of the old count
        let rx = (v.x0 + cw / 2.0 / v.ppc) / old_rows;
        let ry = (v.y0 + ch / 2.0 / (v.ppc * ROW_H)) / old_rows;
        let ppc = (View::fit(cw, ch, self.fit_rows).ppc * zoom).clamp(MIN_PPC, MAX_PPC);
        self.view = View { x0: rx * self.fit_rows - cw / 2.0 / ppc, y0: ry * self.fit_rows - ch / 2.0 / (ppc * ROW_H), ppc };
    }

    /// Rows typed in: with the link on, the modulus moves instead (m = rows / factor,
    /// snapped to the selection mode) and the rows follow it. The modulus is rounded in
    /// the direction of the edit, and an edit that would snap back to the same modulus
    /// takes the next one that way, so a ±1 nudge always moves.
    pub(crate) fn set_rows(&mut self, rows: f64) {
        if self.link_rows {
            // (not signum: it calls 0 positive)
            let dir = if rows > self.fit_rows { 1 } else if rows < self.fit_rows { -1 } else { 0 };
            if dir == 0 {
                return;
            }
            let exact = rows / self.link_factor;
            let target = if dir > 0 { exact.ceil() } else { exact.floor() };
            let before = self.m;
            self.goto(target as i64);
            if self.m == before {
                let v = self.sweep.step_by(self.m, dir);
                self.set_m(v);
            }
            self.apply_link();
        } else {
            self.fit_rows = rows.clamp(2.0, 1.0e9);
            self.fit(self.canvas.0, self.canvas.1);
        }
    }

    /// Go to `m` snapped to the active selection mode (typed value, slider).
    pub(crate) fn goto(&mut self, m: i64) {
        if self.super_mode() {
            // the super prime nearest the whole modulus
            if let Some(i) = self.sweep.super_nearest(m.clamp(1, MOD_CAP as i64) as u32, 1) {
                self.set_super(i, true);
            }
            return;
        }
        let v = self.sweep.snap(m);
        self.set_m(v);
    }

    /// `n` steps through the active selection mode (sign = direction); stops playback.
    pub(crate) fn step(&mut self, n: i32) {
        self.sweep.playing = false;
        if self.super_mode() {
            let i = self.sweep.super_step(self.sp_index, n);
            self.set_super(i, true);
            return;
        }
        let (m, sub) = self.sweep.step_fine_by((self.m, self.sub), n);
        self.set_pos(m, sub, true);
    }

    /// Advance the play timer; true when the modulus moved.
    pub(crate) fn play_tick(&mut self, now: Instant) -> bool {
        if !self.sweep.playing {
            self.last_step = None;
            return false;
        }
        let due = self.last_step.map(|t| now.duration_since(t).as_millis() as u32 >= self.sweep.speed_ms).unwrap_or(true);
        if !due {
            return false;
        }
        self.last_step = Some(now);
        if self.super_mode() {
            let before = self.sp_index;
            let i = self.sweep.super_tick(before);
            self.set_super(i, !self.link_hold_sweep);
            return i != before;
        }
        let before = (self.m, self.sub);
        let (m, sub) = self.sweep.tick_fine(before);
        self.set_pos(m, sub, !self.link_hold_sweep);
        (m, sub) != before
    }

    /// Turn the color offset by `dt` seconds' worth while it plays. The offset is cyclic
    /// (a whole turn changes nothing), so it wraps into [0, 1) and repeats forever.
    pub(crate) fn advance_offset(&mut self, dt: f32) {
        if !self.offset_playing || self.offset_editing || self.offset_period.is_nan() || self.offset_period <= 0.0 || !dt.is_finite() {
            return;
        }
        self.style.offset = (self.style.offset + dt.max(0.0) / self.offset_period).rem_euclid(1.0);
    }

    /// A click on `cell` while picking a selection: the first marks the start, the
    /// second the end.
    pub(crate) fn select_click(&mut self, cell: (u32, u32)) {
        self.wave_select = match self.wave_select {
            WaveSelect::PickStart => WaveSelect::PickEnd(cell),
            WaveSelect::PickEnd(a) => WaveSelect::Active(a, cell),
            other => other,
        };
    }

    /// Back to the defaults (the export folder stays), fitted to the canvas.
    pub(crate) fn reset(&mut self, cw: f64, ch: f64) {
        let dir = std::mem::take(&mut self.export_dir);
        *self = Model::new();
        self.export_dir = dir;
        self.fit(cw, ch);
    }

    pub(crate) fn set_scheme(&mut self, s: Scheme) {
        if s != self.scheme {
            self.scheme = s;
            self.palette = Palette::new(s);
        }
    }

    pub(crate) fn fit(&mut self, cw: f64, ch: f64) {
        self.view = View::fit(cw, ch, self.fit_rows);
        self.auto_fit = true;
    }

    pub(crate) fn inspector_text(&self, n: u32, k: u32) -> String {
        let md = &self.modulus;
        let mut lines = vec![format!("n = {n}, k = {k}")];
        match math::binom_exact(n, k) {
            Some(b) if b.to_string().len() <= 34 => lines.push(format!("C(n,k) = {b}")),
            _ => {
                let lg = math::log10_binom(n, k);
                let e = lg.floor();
                lines.push(format!("C(n,k) ≈ {:.2}e{}", 10f64.powf(lg - e), e as i64));
            }
        }
        if let Some((b, t)) = &self.morph {
            let (ra, rb) = (md.binom(n, k), b.binom(n, k));
            lines.push(format!("μ = {}: a morph {:.0}% of the way from mod {} to mod {}", self.mu_text(), t * 100.0, md.m, b.m));
            lines.push(format!("C mod {} = {ra}   →   C mod {} = {rb}", md.m, b.m));
            return lines.join("\n");
        }
        if md.is_decimal() {
            // μ = P/Q: C mod μ = (Q·C mod P)/Q, exact; μ divides C exactly when P does
            let s = md.scaled(n, k);
            lines.push(format!("C mod {} = {}{}", md.label(), math::decimal_text(s as u64, md.q as u64), if s == 0 { "   μ divides C" } else { "" }));
            lines.push(format!("μ = {}/{}: C mod μ = ({}·C mod {})/{}", md.m, md.q, md.q, md.m, md.q));
            if md.is_prime() {
                lines.push(format!("super prime: {} is prime, so the zeros are mod {}'s (Lucas, blocks of {} rows); {} only reshuffles the colors", md.m, md.m, md.m, md.q));
            }
            return lines.join("\n");
        }
        if md.is_one() {
            lines.push("C mod 1 = 0   (1 divides everything)".into());
            return lines.join("\n");
        }
        let r = md.binom(n, k);
        lines.push(format!("C mod {} = {}{}", md.m, r, if r == 0 { "   m divides C" } else { "" }));
        let composite = md.factors.len() > 1 || md.factors[0].f.e > 1;
        for t in &md.factors {
            let nu = math::carries(n, k, t.f.p);
            let mut s = format!("ν{} = {}", sub(t.f.p), nu);
            if composite {
                s.push_str(&format!("   ({} {})", math::factor_string(&[t.f]), if nu >= t.f.e { "divides" } else { "does not divide" }));
            }
            lines.push(s);
        }
        if let Some(terms) = math::lucas_string(n, k, md.p, 14) {
            lines.push(format!("Lucas: {terms} ≡ {} (mod {})", md.factors[0].binom(n, k) % md.p, md.p));
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pascal::sweep::SuperPrimes;

    fn model() -> Model {
        let mut m = Model::new();
        m.canvas = (980.0, 736.0);
        m
    }

    #[test]
    fn rows_follow_the_modulus_when_linked() {
        let mut md = model();
        md.set_m(11);
        assert_eq!(md.fit_rows, 128.0, "unlinked rows stay");
        md.link_rows = true;
        md.apply_link();
        assert_eq!(md.fit_rows, 11.0);
        assert!(md.auto_fit);
        // typed, stepped and slid moduli all carry the rows
        md.goto(250);
        assert_eq!(md.fit_rows, 250.0);
        md.step(1);
        assert_eq!((md.m, md.fit_rows), (251, 251.0));
        // the factor scales; tiny factors still give 2 rows
        md.link_factor = 2.5;
        md.apply_link();
        assert_eq!(md.fit_rows, 628.0);
        md.link_factor = 0.01;
        md.apply_link();
        assert_eq!(md.fit_rows, 3.0);
        md.set_m(7);
        assert_eq!(md.fit_rows, 2.0);
        // rows typed while linked move the modulus (rows ÷ factor), snapped to the mode
        md.link_factor = 1.0;
        md.set_rows(97.0);
        assert_eq!((md.m, md.fit_rows), (97, 97.0));
        md.sweep.mode = pascal::sweep::ModMode::Prime;
        md.set_rows(100.0);
        assert!(Modulus::new(md.m).is_prime() && md.fit_rows == md.m as f64, "m = {}", md.m);
        // the view is refitted to the new rows
        assert_eq!(md.view, View::fit(980.0, 736.0, md.fit_rows));
        // unlinking leaves the rows where they are; rows typed then are just rows
        md.link_rows = false;
        let m_before = md.m;
        md.set_rows(40.0);
        assert_eq!((md.m, md.fit_rows), (m_before, 40.0));
        md.set_m(200);
        assert_eq!(md.fit_rows, 40.0);
    }

    #[test]
    fn the_sweep_drags_linked_rows_unless_held() {
        let mut md = model();
        md.link_rows = true;
        md.set_m(20);
        md.sweep.playing = true;
        let t0 = Instant::now();
        assert!(md.play_tick(t0));
        assert_eq!(md.fit_rows, md.m as f64, "rows follow the playing sweep");
        md.link_hold_sweep = true;
        let held = md.fit_rows;
        assert!(md.play_tick(t0 + std::time::Duration::from_secs(1)));
        assert_ne!(md.m as f64, held);
        assert_eq!(md.fit_rows, held, "held while playing");
        // a typed modulus still moves them
        md.goto(64);
        assert_eq!(md.fit_rows, 64.0);
    }

    #[test]
    fn linked_row_nudges_always_move_the_modulus() {
        let mut md = model();
        md.link_rows = true;
        md.set_m(8);
        md.link_factor = 2.5;
        md.apply_link();
        assert_eq!(md.fit_rows, 20.0);
        md.set_rows(21.0); // 21 / 2.5 = 8.4: up means 9
        assert_eq!(md.m, 9);
        md.set_rows(md.fit_rows - 1.0);
        assert_eq!(md.m, 8, "and back down");
        // modes that snap back to the current modulus take the next one that way
        md.link_factor = 1.0;
        md.sweep.mode = pascal::sweep::ModMode::Prime;
        md.set_m(7);
        md.set_rows(8.0);
        assert_eq!((md.m, md.fit_rows), (11, 11.0));
        md.set_rows(10.0);
        assert_eq!(md.m, 7);
        md.sweep.mode = pascal::sweep::ModMode::Odd;
        md.set_rows(6.0);
        assert_eq!(md.m, 5);
        md.set_rows(6.0);
        assert_eq!(md.m, 7);
        // the same count again is not a nudge (a deferred field reports changed()
        // while its text is being typed, with the old number)
        for _ in 0..3 {
            md.set_rows(md.fit_rows);
        }
        assert_eq!((md.m, md.fit_rows), (7, 7.0));
    }

    #[test]
    fn linked_steps_keep_a_zoomed_view() {
        let mut md = model();
        md.link_rows = true;
        md.set_m(100);
        let fit100 = View::fit(980.0, 736.0, 100.0);
        assert_eq!(md.view, fit100);
        // zoom 3× into a spot left of the apex, as the wheel would
        md.view.zoom_at(300.0, 200.0, 3.0);
        md.auto_fit = false;
        let rel = |v: &View, rows: f64| ((v.x0 + 490.0 / v.ppc) / rows, (v.y0 + 368.0 / (v.ppc * ROW_H)) / rows);
        let (zoom, spot) = (md.view.ppc / fit100.ppc, rel(&md.view, 100.0));
        md.step(1);
        assert_eq!(md.fit_rows, 101.0);
        let fit101 = View::fit(980.0, 736.0, 101.0);
        assert!((md.view.ppc / fit101.ppc - zoom).abs() < 1e-9, "the zoom stays");
        let spot2 = rel(&md.view, 101.0);
        assert!((spot.0 - spot2.0).abs() < 1e-9 && (spot.1 - spot2.1).abs() < 1e-9, "{spot:?} vs {spot2:?}");
        assert!(!md.auto_fit);
        // a fitted view simply refits
        md.fit(980.0, 736.0);
        md.step(1);
        assert_eq!(md.view, View::fit(980.0, 736.0, 102.0));
    }

    /// One egui frame of a single field at the origin, with these input events.
    fn field_frame(ctx: &egui::Context, v: &mut u32, live: bool, events: Vec<egui::Event>) -> Rect {
        let input = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(400.0, 200.0))), events, ..Default::default() };
        let mut rect = Rect::NOTHING;
        let _ = ctx.run_ui(input, |ui| {
            rect = mono_drag_with(ui, v, 2..=100_000u32, 1.0, live).rect;
        });
        rect
    }

    #[test]
    fn deferred_fields_apply_typed_values_on_enter() {
        for live in [false, true] {
            let ctx = egui::Context::default();
            let mut v = 7u32;
            let r = field_frame(&ctx, &mut v, live, vec![]);
            let p = r.center();
            let press = |down| egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: down, modifiers: Default::default() };
            field_frame(&ctx, &mut v, live, vec![egui::Event::PointerMoved(p), press(true)]);
            field_frame(&ctx, &mut v, live, vec![press(false)]);
            field_frame(&ctx, &mut v, live, vec![]);
            // the click selected the text: type over it, one key per frame
            for ch in ["5", "0", "0"] {
                field_frame(&ctx, &mut v, live, vec![egui::Event::Text(ch.into())]);
            }
            if live {
                assert_eq!(v, 500, "live fields follow each key");
            } else {
                assert_eq!(v, 7, "nothing applies before Enter");
                let enter = egui::Event::Key { key: egui::Key::Enter, physical_key: None, pressed: true, repeat: false, modifiers: Default::default() };
                field_frame(&ctx, &mut v, live, vec![enter]);
                field_frame(&ctx, &mut v, live, vec![]);
                assert_eq!(v, 500, "Enter applies the typed value");
            }
        }
    }

    #[test]
    fn the_offset_cycles_while_it_plays() {
        let mut md = model();
        md.offset_period = 4.0;
        md.advance_offset(1.0);
        assert_eq!(md.style.offset, 0.0, "not playing: nothing moves");
        md.offset_playing = true;
        md.advance_offset(1.0);
        assert!((md.style.offset - 0.25).abs() < 1e-6);
        for _ in 0..3 {
            md.advance_offset(1.0);
        }
        assert!(md.style.offset.abs() < 1e-5 || (md.style.offset - 1.0).abs() < 1e-5, "a whole turn: {}", md.style.offset);
        md.advance_offset(5.0);
        assert!((0.0..1.0).contains(&md.style.offset), "wraps, round and round");
        md.style.offset = -0.9; // set by hand while playing: carries on from there
        md.advance_offset(0.4);
        assert!((md.style.offset - 0.2).abs() < 1e-5, "{}", md.style.offset);
        md.offset_playing = false;
        md.advance_offset(1.0);
        assert!((md.style.offset - 0.2).abs() < 1e-5);
    }

    #[test]
    fn decimals_glide_the_modulus_between_visited_ones() {
        let mut md = Model::new();
        md.set_m(10);
        md.sweep.decimals = 2;
        md.step(1);
        assert_eq!((md.m, md.sub, md.m_text.as_str()), (10, 1, "10.01"));
        assert_eq!((md.modulus.m, md.modulus.q), (1001, 100));
        md.step(49);
        assert_eq!((md.m_text.as_str(), md.modulus.m, md.modulus.q), ("10.5", 21, 2), "10.50 in lowest terms");
        md.step(50);
        assert_eq!((md.m, md.sub, md.m_text.as_str()), (11, 0, "11"), "the glide lands on the next modulus");
        assert!(!md.modulus.is_decimal());
        md.step(-1);
        assert_eq!(md.m_text, "10.99");
        // typed: the whole part snaps, extra digits raise the decimals
        assert!(md.goto_text("12.375"));
        assert_eq!((md.sweep.decimals, md.m_text.as_str()), (3, "12.375"));
        assert!(md.goto_text("7"));
        assert_eq!(md.m_text, "7");
        assert!(!md.goto_text("7.x"));
        // primes: 13.9 → 14 is not a stop; the gap runs on to 17
        md.sweep.mode = ModMode::Prime;
        md.sweep.decimals = 1;
        md.set_m(13);
        md.step(39);
        assert_eq!(md.m_text, "16.9");
        md.step(1);
        assert_eq!(md.m_text, "17");
        // playback through a gap, and the rows follow μ
        md.sweep.playing = true;
        md.sweep.speed_ms = 16;
        let t0 = std::time::Instant::now();
        assert!(md.play_tick(t0));
        assert_eq!(md.m_text, "17.1");
    }

    #[test]
    fn morphs_run_between_the_whole_moduli() {
        let mut md = Model::new();
        md.sweep.morph = true;
        md.sweep.decimals = 1;
        md.set_m(10);
        md.step(3);
        let (b, t) = md.morph.clone().expect("a morph");
        assert_eq!((md.m_text.as_str(), md.modulus.m, md.modulus.q, b.m), ("10.3", 10, 1, 11));
        assert!((t - 0.3).abs() < 1e-6);
        assert!(md.inspector_text(5, 2).contains("C mod 10 = 0   →   C mod 11 = 10"));
        md.step(7);
        assert!(md.morph.is_none(), "it lands on 11");
        assert_eq!(md.m_text, "11");
        // primes: 7.5 is a quarter of the way from 7 to 11
        md.sweep.mode = ModMode::Prime;
        md.set_m(7);
        md.step(10);
        let (b, t) = md.morph.clone().unwrap();
        assert_eq!((b.m, md.m_text.as_str()), (11, "8"));
        assert!((t - 0.25).abs() < 1e-6);
        // switching the mode off draws the exact decimal again
        md.sweep.morph = false;
        md.set_pos(7, 5, true);
        assert!(md.morph.is_none() && md.modulus.is_decimal());
    }

    #[test]
    fn mod_one_and_its_decimals() {
        let mut md = Model::new();
        md.set_m(1);
        assert_eq!((md.m, md.modulus.binom(9, 4), md.m_text.as_str()), (1, 0, "1"));
        assert!(md.inspector_text(9, 4).contains("C mod 1 = 0"));
        // 1.5 = 3/2: C mod 1.5 is 0, 0.5 or 1
        assert!(md.goto_text("1.5"));
        assert_eq!((md.modulus.m, md.modulus.q), (3, 2));
        for n in 0..30 {
            for k in 0..=n {
                let r = md.modulus.residue(n, k);
                assert!([0.0, 0.5, 1.0].contains(&r), "C({n},{k}) mod 1.5 = {r}");
            }
        }
        assert!(md.goto_text("1.25") && md.m_text == "1.25");
        // a morph from mod 1 (all zeros) to mod 2
        md.sweep.morph = true;
        md.sweep.decimals = 1;
        md.set_m(1);
        md.step(5);
        assert_eq!(md.morph.as_ref().map(|(b, _)| b.m), Some(2));
        assert!(md.modulus.is_one());
    }

    #[test]
    fn fractions_typed_in_and_the_super_prime_walk() {
        let mut md = Model::new();
        // any mode: "37/5" is μ = 7.4 exactly, "23/3" stays a fraction, "74/10" reduces
        assert!(md.goto_text("37/5"));
        assert_eq!((md.modulus.m, md.modulus.q, md.m_text.as_str()), (37, 5, "7.4"));
        assert!(md.inspector_text(40, 3).contains("super prime: 37 is prime"));
        assert!(md.goto_text("23/3"));
        assert_eq!((md.m_text.as_str(), md.m), ("23/3", 7));
        assert!(md.goto_text("74/10") && md.m_text == "7.4");
        assert!(md.goto_text("3/5") && md.m_text == "0.6", "below 1");
        assert!(!md.goto_text("0/5"));
        assert!(!md.goto_text("5/0"));
        assert!(md.goto_text("23/3"));
        // a step leaves the fraction for the grid
        md.step(1);
        assert!(!md.modulus.is_decimal() && md.m_text == "8");
        // the walk: decimals 3 around 7, by value; 37/5 typed lands on its place
        md.sweep.mode = ModMode::SuperPrime;
        md.sweep.start = 7;
        md.sweep.end = 8;
        md.sweep.decimals = 3;
        assert!(md.goto_text("37/5"));
        let i = md.sp_index;
        assert_eq!(md.sweep.super_list()[i], (37, 5));
        md.step(1);
        let next = md.sweep.super_list()[i + 1];
        assert_eq!((md.modulus.m, md.modulus.q), next);
        assert!(next.0 as f64 / next.1 as f64 > 7.4);
        // typed decimals find their nearest super prime
        assert!(md.goto_text("7.3"));
        assert_eq!((md.modulus.m, md.modulus.q), (73, 10));
        // playback walks the list
        md.sweep.playing = true;
        md.sweep.speed_ms = 16;
        assert!(md.play_tick(std::time::Instant::now()));
        assert_eq!(md.sp_index, md.sweep.super_nearest(73, 10).unwrap() + 1);
    }

    #[test]
    fn nine_decimals_and_all_by_numerator() {
        let mut md = Model::new();
        md.sweep.decimals = 9;
        md.set_m(7);
        md.step(1);
        // 7.000000001 needs 7000000001: past the cap, so the nearest fraction that fits
        assert!(md.rounded && md.m_text == "7.000000001" && !md.modulus.is_decimal());
        // six exact digits near 7 (lowest terms: 7.123456 = 111304/15625)
        assert!(md.goto_text("7.123456") && !md.rounded && md.modulus.label() == "7.123456");
        // mode All by numerator: every fraction of the range, coarsest numerators first
        md.sweep.decimals = 1;
        md.sweep.start = 7;
        md.sweep.end = 8;
        md.sweep.all_order = SpOrder::Numerator;
        md.goto(7);
        assert_eq!((md.modulus.m, md.modulus.q), (7, 1));
        md.step(3);
        assert_eq!((md.modulus.m, md.modulus.q, md.m_text.as_str()), (36, 5, "7.2"));
        md.step(1);
        assert_eq!(md.m_text, "7.4");
    }

    #[test]
    fn the_tetra_stack_keeps_whole_moduli() {
        // its own sweep never glides: a decimal setting there changes nothing
        let mut c = pascal::tetra::TetraConfig::default();
        let whole = pascal::tetra::selected_moduli(&c.sweep);
        c.sweep.decimals = 3;
        assert_eq!(pascal::tetra::selected_moduli(&c.sweep), whole);
    }

    #[test]
    fn the_ghost_plays_the_rows_typed_in() {
        // rows 1–5 mod 5, typed either way round
        let mut g = GhostWave { on: true, modulus: 5, rows: (1, 5) };
        assert_eq!(g.source(), Some((5, (1, 5))));
        g.rows = (5, 1);
        assert_eq!(g.source(), Some((5, (1, 5))));
        g.on = false;
        assert_eq!(g.source(), None);
    }

    #[test]
    fn selections_take_a_start_then_an_end() {
        let mut md = model();
        md.select_click((5, 2));
        assert_eq!(md.wave_select, WaveSelect::Off, "clicks do nothing until armed");
        md.wave_select = WaveSelect::PickStart;
        md.select_click((9, 3));
        assert_eq!(md.wave_select, WaveSelect::PickEnd((9, 3)));
        md.select_click((4, 1));
        assert_eq!(md.wave_select, WaveSelect::Active((9, 3), (4, 1)));
        md.select_click((7, 7));
        assert_eq!(md.wave_select, WaveSelect::Active((9, 3), (4, 1)), "an active selection keeps its marks");
    }

    #[test]
    fn linked_refits_respect_the_zoom_cap() {
        // zoomed all the way in at m = 40, then m = 2: the kept relative zoom would
        // pass MAX_PPC (and the in-cell numbers' font with it)
        let mut md = model();
        md.link_rows = true;
        md.set_m(40);
        for _ in 0..200 {
            md.view.zoom_at(490.0, 368.0, 1.5);
        }
        md.auto_fit = false;
        assert_eq!(md.view.ppc, MAX_PPC);
        md.set_m(2);
        assert!(md.view.ppc <= MAX_PPC && md.view.ppc >= MIN_PPC, "{}", md.view.ppc);
    }

    #[test]
    fn a_fitted_view_shows_its_rows_only() {
        // a canvas narrower than the triangle: the fit is width-limited, and rows past
        // the fit are background, not counted
        let mut md = model();
        md.canvas = (300.0, 900.0);
        md.fit_rows = 64.0;
        md.fit(300.0, 900.0);
        assert!(md.view.visible_rows(900.0).unwrap().1 > 63);
        assert_eq!(md.shown_rows(900.0), Some((0, 63)));
        // zoomed or panned, the whole triangle is back
        md.view.zoom_at(150.0, 450.0, 1.2);
        md.auto_fit = false;
        assert!(md.shown_rows(900.0).unwrap().1 > 63);
    }

    #[test]
    fn typing_the_offset_holds_the_cycle() {
        let mut md = model();
        md.offset_playing = true;
        md.offset_editing = true;
        md.style.offset = 0.125;
        md.advance_offset(1.0);
        assert_eq!(md.style.offset, 0.125);
        md.offset_editing = false;
        md.advance_offset(1.0);
        assert!((md.style.offset - 0.375).abs() < 1e-6);
    }

    #[test]
    fn reset_unlinks() {
        let mut md = model();
        md.link_rows = true;
        md.link_factor = 3.0;
        md.reset(980.0, 736.0);
        assert!(!md.link_rows && md.link_factor == 1.0 && md.fit_rows == 128.0);
    }

    #[test]
    fn below_one_phase_colors_and_unreduced_fractions() {
        let mut md = Model::new();
        // 10/7 and 10/9: by value nearly one color, by phase mod 10's colors reshuffled
        assert!(md.goto_text("10/7"));
        assert!(md.frame_style().phase, "auto: phase below 3");
        let colors = |md: &Model| {
            let st = md.frame_style();
            let map = md.palette.level_map(md.modulus.m, st.offset);
            let q = if st.phase { 1 } else { md.modulus.q };
            (0..md.modulus.m).map(|s| md.palette.residue_color_scaled(s, md.modulus.m, q, &map)).collect::<Vec<_>>()
        };
        let a = colors(&md);
        assert!(md.goto_text("10/9"));
        let b = colors(&md);
        // one table, mod 10's colors by s; a cell with C ≡ 3 (mod 10) has s = 7·3 mod 10
        // = 1 in 10/7, s = 9·3 mod 10 = 7 in 10/9: different colors
        assert_eq!(a, b);
        assert_ne!(a[3 * 7 % 10], b[3 * 9 % 10]);
        md.frac_colors = FracColors::Value;
        assert!(!md.frame_style().phase);
        let v = colors(&md);
        let distinct = |c: &[[u8; 3]]| { let mut c = c.to_vec(); c.sort(); c.dedup(); c.len() };
        assert!(distinct(&v) <= 3, "by value below 2: void, white, one ramp color");
        assert_eq!(distinct(&a), 10);
        md.frac_colors = FracColors::Auto;
        assert!(md.goto_text("37/5") && !md.frame_style().phase, "auto: value from 3");
        assert!(md.goto_text("7") && !md.frame_style().phase, "whole moduli are the same either way");

        // decimals from 0: 0.7 = 7/10, and the walk's first frame is 0.1
        md.sweep.decimals = 1;
        md.sweep.start = 0;
        md.sweep.end = 3;
        assert!(md.goto_text("0.7"));
        assert_eq!((md.m, md.sub, md.modulus.m, md.modulus.q, md.m_text.as_str()), (0, 7, 7, 10, "0.7"));
        md.step(-6);
        assert_eq!((md.m, md.sub), (0, 1));
        md.step(-1);
        assert_eq!((md.m, md.sub), (0, 1), "μ = 0 is no modulus");
        md.set_pos(0, 0, false);
        assert_eq!((md.m, md.sub), (0, 1));
        md.step(9);
        assert_eq!((md.m, md.sub, md.m_text.as_str()), (1, 0, "1"));
        // a morph below 1 draws the exact residue (no whole modulus to morph from)
        md.sweep.morph = true;
        md.set_pos(0, 5, false);
        assert!(md.morph.is_none() && md.modulus.label() == "0.5");

        // unreduced: numerator 10 shows every Q; 10/4 draws 5/2 and says 10/4
        md.sweep.morph = false;
        md.sweep.mode = ModMode::All;
        md.sweep.all_order = SpOrder::Numerator;
        md.sweep.super_primes = SuperPrimes { denoms: SpDenoms::Any, qmax: 12, order: SpOrder::Numerator, unreduced: true };
        md.sweep.start = 0;
        md.sweep.end = 128;
        let i = md.sweep.super_list().iter().position(|&f| f == (10, 4)).unwrap();
        md.set_super(i, false);
        assert_eq!((md.modulus.m, md.modulus.q, md.m_text.as_str()), (5, 2, "10/4"));
        assert!(md.goto_text(&md.m_text.clone()) && md.modulus.label() == "2.5");
    }

    #[test]
    fn kinship_steps_through_relatives() {
        let mut md = Model::new();
        md.sweep.mode = ModMode::Kinship;
        md.sweep.start = 2;
        md.sweep.end = 24;
        md.set_super(0, false);
        let mut seen = vec![md.modulus.m];
        for _ in 0..8 {
            md.step(1);
            seen.push(md.modulus.m);
        }
        assert_eq!(seen, vec![2, 4, 8, 16, 24, 12, 6, 3, 9]);
        assert!(!md.modulus.is_decimal() && md.m_text == "9");
        md.step(-2);
        assert_eq!(md.modulus.m, 6);
        // a typed modulus lands on its place in the walk
        assert!(md.goto_text("15"));
        assert_eq!(md.sweep.super_list()[md.sp_index], (15, 1));
        md.step(1);
        assert_eq!(md.modulus.m, 5);
    }

    #[test]
    fn tables_are_reused_and_fractions_keep_their_value() {
        let mut md = Model::new();
        // a morph keeps both ends' tables while the walk stays in the gap, and landing
        // on the far end takes the tables the morph already held
        md.sweep.decimals = 1;
        md.sweep.morph = true;
        md.set_pos(7, 1, false);
        let (base, far) = (md.modulus.clone(), md.morph.as_ref().unwrap().0.clone());
        assert_eq!((base.m, far.m), (7, 8));
        md.set_pos(7, 2, false);
        assert!(Arc::ptr_eq(&md.modulus, &base) && Arc::ptr_eq(&md.morph.as_ref().unwrap().0, &far));
        md.set_pos(8, 0, false);
        assert!(Arc::ptr_eq(&md.modulus, &far) && md.morph.is_none());
        md.sweep.morph = false;
        // the same fraction again builds nothing
        assert!(md.goto_text("37/5"));
        let held = md.modulus.clone();
        md.set_fraction(74, 10, false);
        assert!(Arc::ptr_eq(&md.modulus, &held));
        // rows linked to a fraction follow its value, not its whole part
        assert!((md.mu() - 7.4).abs() < 1e-12);
        md.link_rows = true;
        md.link_factor = 10.0;
        assert_eq!(md.linked_rows(), 74.0);
        assert!(md.goto_text("7/10") && (md.mu() - 0.7).abs() < 1e-12);
        // a denominator past u32 is refused, not truncated
        assert!(!md.goto_text("7/4294967301"));
        assert_eq!(math::fraction_label(74, 10), "7.4");
        assert_eq!(math::fraction_label(10, 4), "2.5");
        assert_eq!(math::fraction_label(23, 3), "23/3");
    }
}
