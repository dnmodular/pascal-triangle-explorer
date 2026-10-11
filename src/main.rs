//! The desktop app: a winit window, an egui control panel on the left and the GPU
//! canvas filling the rest. All math and rendering live in the `pascal` library; this
//! file is the UI, the camera interaction and the export plumbing.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Vec2};
use pascal::export;
use pascal::gpu::{CanvasStyle, FrameInput, Gpu, Renderer, Tween};
use pascal::layers::{AvgMode, BlendMode, FieldColorMode, Kind, LayerId, LayerState, Preset, REGISTRY};
use pascal::math::{self, Modulus, MOD_CAP};
use pascal::palette::{Palette, Scheme};
use pascal::sweep::{ModMode, PrimeOrder, Seq, Sweep};
use pascal::view::{GridSpec, View, MAX_PPC, MIN_PPC, ROW_H, STATUS_H};
use pascal::wave_overlay::{Corner, WaveOverlay};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

const PANEL_W: f32 = 300.0;
/// Height of the sweep bar under the canvas (two rows of controls).
const SWEEP_BAR_H: f32 = 64.0;
const CANVAS_BG: Color32 = Color32::from_rgb(10, 10, 10);
/// Cells at least this wide (px) get their residue printed inside.
const NUMBERS_MIN_PX: f64 = 18.0;
const NUMBERS_MAX_CELLS: u64 = 9000;
/// Largest in-cell number, in points (egui's glyph atlas grows with every size drawn).
const NUMBERS_MAX_PT: f64 = 96.0;

/// Picking a block of cells whose average wave the overlay plays (like the pin, but
/// for a rectangle of cells between a red start mark and a blue end mark).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum WaveSelect {
    #[default]
    Off,
    /// Armed: the next click on a cell marks the start (red).
    PickStart,
    /// The next click marks the end (blue).
    PickEnd((u32, u32)),
    /// The overlay plays the cells between the two marks.
    Active((u32, u32), (u32, u32)),
}

/// Everything the user can change.
struct Model {
    m: u32,
    /// Shared with the wave overlay, which reads long rows through the same closed form.
    modulus: Arc<Modulus>,
    m_text: String,
    scheme: Scheme,
    palette: Palette,
    /// Color offset and cell decorations (outlines, glow around zeros).
    style: CanvasStyle,
    /// The offset cycles on its own while this is on: one full turn per period.
    offset_playing: bool,
    offset_period: f32,
    /// The offset field is being typed into or dragged: the cycle waits.
    offset_editing: bool,
    layers: LayerState,
    view: View,
    fit_rows: f64,
    auto_fit: bool,
    /// Rows parented to the modulus: rows = m × link_factor, refitted whenever m moves
    /// (the reference's LINK). While the sweep plays they follow too, unless held.
    link_rows: bool,
    link_factor: f64,
    link_hold_sweep: bool,
    /// The canvas size (px) of the last frame, for fits made outside the canvas UI.
    canvas: (f64, f64),
    /// The canvas alone, borderless over the whole screen (F11; Esc returns).
    fullscreen: bool,
    ss: u32,
    show_numbers: bool,
    export_dir: String,
    export_note: String,
    pending_export: bool,
    /// Save SVG was clicked: rows to write once the frame has rendered (the valuation
    /// coloring reads the frame's max ν back).
    pending_svg: Option<u32>,
    hover: Option<(u32, u32)>,
    /// The row under the pointer, inside the triangle or beside it.
    hover_row: Option<u32>,
    sweep: Sweep,
    /// When the play timer last advanced the modulus.
    last_step: Option<Instant>,
    /// The "modular wave" overlay and the row it plays: the pinned row (click a cell),
    /// else the hovered row, else the last visible row.
    wave: WaveOverlay,
    pinned_row: Option<u32>,
    wave_select: WaveSelect,
    wave_row: u32,
    /// The pointer is over the overlay box: the canvas under it gets no hover.
    over_wave: bool,
    /// The pin and selection before the latest click (a double click undoes its first
    /// click, whichever it was).
    pin_before_click: Option<u32>,
    select_before_click: WaveSelect,
    /// The canvas is being dragged (the wave keeps its row while the view pans).
    dragging: bool,
    /// A widget had keyboard focus or a popup was open last frame (its Esc is not ours).
    focus_last_frame: bool,
}

impl Model {
    fn new() -> Self {
        let m = 7;
        Model {
            m,
            modulus: Arc::new(Modulus::new(m)),
            m_text: m.to_string(),
            scheme: Scheme::Spectral,
            palette: Palette::new(Scheme::Spectral),
            style: CanvasStyle::default(),
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
            over_wave: false,
            pin_before_click: None,
            select_before_click: WaveSelect::Off,
            dragging: false,
            focus_last_frame: false,
        }
    }

    fn set_m(&mut self, m: u32) {
        self.set_m_with(m, true);
    }

    /// Set the modulus; with `follow`, linked rows follow it.
    fn set_m_with(&mut self, m: u32, follow: bool) {
        let m = m.clamp(2, MOD_CAP);
        let changed = m != self.m;
        if changed {
            self.m = m;
            self.modulus = Arc::new(Modulus::new(m));
        }
        self.m_text = m.to_string();
        if changed && follow {
            self.apply_link();
        }
    }

    /// The rows drawn on a canvas `h` px tall: the visible ones, and while fitted none
    /// past the fit (the canvas below it is background).
    fn shown_rows(&self, h: f64) -> Option<(u32, u32)> {
        let (a, b) = self.view.visible_rows(h)?;
        if !self.auto_fit {
            return Some((a, b));
        }
        let last = (self.fit_rows.round() as u32).saturating_sub(1);
        (a <= last).then_some((a, b.min(last)))
    }

    /// The row count the link gives for the current modulus.
    fn linked_rows(&self) -> f64 {
        (self.m as f64 * self.link_factor).round().clamp(2.0, 1.0e9)
    }

    /// With rows linked, set them from the modulus and refit. A view the user zoomed
    /// or panned (auto_fit off) keeps its zoom relative to the fit and the same spot of
    /// the triangle, scaled with it (the world origin is the apex), so stepping or
    /// sweeping a linked modulus never throws the view away.
    fn apply_link(&mut self) {
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
    fn set_rows(&mut self, rows: f64) {
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
    fn goto(&mut self, m: i64) {
        let v = self.sweep.snap(m);
        self.set_m(v);
    }

    /// `n` steps through the active selection mode (sign = direction); stops playback.
    fn step(&mut self, n: i32) {
        self.sweep.playing = false;
        let v = self.sweep.step_by(self.m, n);
        self.set_m(v);
    }

    /// Advance the play timer; true when the modulus moved.
    fn play_tick(&mut self, now: Instant) -> bool {
        if !self.sweep.playing {
            self.last_step = None;
            return false;
        }
        let due = self.last_step.map(|t| now.duration_since(t).as_millis() as u32 >= self.sweep.speed_ms).unwrap_or(true);
        if !due {
            return false;
        }
        self.last_step = Some(now);
        let v = self.sweep.tick(self.m);
        let moved = v != self.m;
        self.set_m_with(v, !self.link_hold_sweep);
        moved
    }

    /// Turn the color offset by `dt` seconds' worth while it plays. The offset is cyclic
    /// (a whole turn changes nothing), so it wraps into [0, 1) and repeats forever.
    fn advance_offset(&mut self, dt: f32) {
        if !self.offset_playing || self.offset_editing || self.offset_period.is_nan() || self.offset_period <= 0.0 || !dt.is_finite() {
            return;
        }
        self.style.offset = (self.style.offset + dt.max(0.0) / self.offset_period).rem_euclid(1.0);
    }

    /// A click on `cell` while picking a selection: the first marks the start, the
    /// second the end.
    fn select_click(&mut self, cell: (u32, u32)) {
        self.wave_select = match self.wave_select {
            WaveSelect::PickStart => WaveSelect::PickEnd(cell),
            WaveSelect::PickEnd(a) => WaveSelect::Active(a, cell),
            other => other,
        };
    }

    /// Back to the defaults (the export folder stays), fitted to the canvas.
    fn reset(&mut self, cw: f64, ch: f64) {
        let dir = std::mem::take(&mut self.export_dir);
        *self = Model::new();
        self.export_dir = dir;
        self.fit(cw, ch);
    }

    fn set_scheme(&mut self, s: Scheme) {
        if s != self.scheme {
            self.scheme = s;
            self.palette = Palette::new(s);
        }
    }

    fn fit(&mut self, cw: f64, ch: f64) {
        self.view = View::fit(cw, ch, self.fit_rows);
        self.auto_fit = true;
    }

    /// The key an inspector line or in-cell number shows for a cell.
    fn cell_value(&self, n: u32, k: u32) -> Option<String> {
        match self.layers.preset() {
            Some(Preset::Residue) | Some(Preset::Mask) => Some(self.modulus.binom(n, k).to_string()),
            Some(Preset::Valuation) => Some(math::carries(n, k, self.modulus.p).to_string()),
            None => None,
        }
    }

    fn inspector_text(&self, n: u32, k: u32) -> String {
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

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    canvas_tex: Option<egui::TextureId>,
    canvas_size: (u32, u32),
}

/// Everything that changes what the GPU renders: modulus, palette, style, layers, camera,
/// supersampling and canvas size.
type FrameKey = (u32, Scheme, CanvasStyle, LayerState, View, u32, (u32, u32));

struct App {
    gfx: Option<Gfx>,
    model: Model,
    last_inputs: Option<FrameKey>,
    frame_ms: f32,
    /// Redraw counter: egui may run the UI closure more than once per frame, but
    /// input (keys, wheel, drag, clicks) must move the camera only once.
    frame_nr: u64,
    surface_errors: u32,
    fatal: bool,
    /// Modulus of the last rendered frame, and when the running tween began.
    last_m: u32,
    tween_t0: Option<Instant>,
    /// The display texture holds the settled true frame of `last_inputs` (no tween, not
    /// the advancing render), so a redraw that changes none of them (overlay animation,
    /// hover) can skip the GPU passes.
    canvas_current: bool,
    /// When the last frame started, and whether it asked for the next one right away
    /// (then the gap is real frame time; after an idle gap one nominal frame is used).
    last_frame: Option<Instant>,
    continuous: bool,
    /// Frames since `fps_t0` (logged at debug level once a second).
    fps_frames: u32,
    fps_t0: Option<Instant>,
    /// A delayed repaint egui asked for.
    repaint_at: Option<Instant>,
    /// Whether the window is in (borderless) fullscreen, to follow `Model::fullscreen`.
    fullscreen_applied: bool,
}

const USAGE: &str = "pascal-triangle-explorer [options]
  --m N              modulus (default 7)
  --rows R           rows to fit (default 128)
  --palette NAME     spectral | viridis | magma | inferno | turbo | cividis | coolwarm | mono
                     | ember | ocean | moss
  --offset X         fractional color offset (−1..1)
  --link             rows follow the modulus (rows = m)
  --outlines         outline each cell when zoomed in
  --glow             glow around zero cells when zoomed in
  --mode MODE        residue | mask | valuation
  --visit MODE       sweep selection: all | primes | composites | even | odd | interval
  --layers a,b,c     active layers: value kummer lucas crt dlog pdigit gradient distzero magnitude
                     density variance rarity moire primes stripes lagdiff average bandpass afterimage
  --field MODE       math | palette (field color mode)
  --export FILE.png  render once without a window and write a PNG
  --svg FILE.svg     write rows 0..R−1 (--rows, at most 1024) as an SVG (residue / mask / valuation)
  --size WxH         canvas size for --export (default 1600x1000)
  --check-shaders    compile every shader pipeline one by one and report what the driver rejects";

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model = Model::new();
    let mut export: Option<PathBuf> = None;
    let mut svg: Option<PathBuf> = None;
    let mut size = (1600u32, 1000u32);
    let mut i = 0;
    while i < args.len() {
        let next = || args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--m" => model.set_m(next().parse().unwrap_or(7)),
            "--rows" => model.fit_rows = next().parse::<f64>().unwrap_or(128.0).max(2.0),
            "--palette" => {
                if let Some(s) = Scheme::ALL.iter().find(|s| s.id() == next()) {
                    model.set_scheme(*s);
                }
            }
            "--mode" => match next().as_str() {
                "mask" => model.layers.preset_mask(),
                "valuation" => model.layers.preset_valuation(),
                _ => model.layers.preset_residue(),
            },
            "--layers" => {
                model.layers.active.clear();
                for name in next().split(',') {
                    let name = name.trim().to_lowercase().replace([' ', '-', '_'], "");
                    let clean = |x: &str| x.to_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>();
                    match REGISTRY.iter().find(|d| clean(d.label) == clean(&name) || clean(&format!("{:?}", d.id)) == clean(&name)) {
                        Some(d) => model.layers.set_active(d.id, true),
                        None => {
                            eprintln!("unknown layer {name}\n{USAGE}");
                            std::process::exit(2);
                        }
                    }
                }
            }
            "--field" => {
                model.layers.color_mode = if next() == "palette" { FieldColorMode::PaletteBlend } else { FieldColorMode::MathRamp };
            }
            "--visit" => {
                if let Some(m) = ModMode::ALL.iter().find(|m| m.label() == next()) {
                    model.sweep.mode = *m;
                }
            }
            "--export" => export = Some(PathBuf::from(next())),
            "--svg" => svg = Some(PathBuf::from(next())),
            // as the panel's control: −1..1
            "--offset" => model.style.offset = next().parse::<f32>().ok().filter(|v| v.is_finite()).unwrap_or(0.0).clamp(-1.0, 1.0),
            "--link" => {
                model.link_rows = true;
                i += 1;
                continue;
            }
            "--outlines" | "--glow" => {
                if args[i] == "--outlines" {
                    model.style.outlines = true;
                } else {
                    model.style.glow = true;
                }
                i += 1;
                continue;
            }
            "--size" => {
                let v = next();
                if let Some((w, h)) = v.split_once('x') {
                    size = (w.parse().unwrap_or(1600), h.parse().unwrap_or(1000));
                }
            }
            "--check-shaders" => {
                check_shaders();
                return;
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                return;
            }
            other => {
                eprintln!("unknown option {other}\n{USAGE}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    model.apply_link();
    if let Some(path) = svg {
        let rows = model.fit_rows.round().clamp(1.0, export::SVG_MAX_ROWS as f64) as u32;
        // headless: the valuation is normalized over the exported rows
        let Some(colors) = export::preset_colors(rows, &model.modulus, &model.palette, &model.layers, &model.style, None) else {
            eprintln!("--svg draws the residue, mask and valuation colorings only");
            std::process::exit(2);
        };
        match export::write_svg(&path, &export::svg_document(&colors, &model.style)) {
            Ok(()) => eprintln!("wrote rows 0–{} to {}", rows - 1, path.display()),
            Err(e) => {
                eprintln!("export failed: {e}");
                std::process::exit(1);
            }
        }
        if export.is_none() {
            return;
        }
    }
    if let Some(path) = export {
        headless_export(model, size, &path);
        return;
    }
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App { gfx: None, model, last_inputs: None, frame_ms: 0.0, frame_nr: 0, surface_errors: 0, fatal: false, last_m: 0, tween_t0: None, canvas_current: false, last_frame: None, continuous: false, fps_frames: 0, fps_t0: None, repaint_at: None, fullscreen_applied: false };
    event_loop.run_app(&mut app).expect("run");
}

/// The wgpu instance. Debug info (OpName/OpSource/OpLine) is left out of the SPIR-V
/// unless `WGPU_DEBUG=1`: it only helps graphics debuggers and has tripped driver
/// compilers before. `WGPU_BACKEND` and `WGPU_VALIDATION` are honored as usual.
fn instance_descriptor(window: Option<Arc<Window>>) -> wgpu::InstanceDescriptor {
    let mut desc = match window {
        Some(w) => wgpu::InstanceDescriptor::new_with_display_handle(Box::new(w)),
        None => wgpu::InstanceDescriptor::new_without_display_handle(),
    };
    desc.flags = wgpu::InstanceFlags::empty().with_env();
    desc.with_env()
}

/// `--check-shaders`: name the entry points the driver's shader compiler rejects.
fn check_shaders() {
    let instance = wgpu::Instance::new(instance_descriptor(None));
    let adapter = pollster::block_on(Gpu::request_adapter(&instance, None)).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let info = adapter.get_info();
    println!("adapter: {} ({:?}, driver {} {})", info.name, info.backend, info.driver, info.driver_info);
    println!("pascal-triangle-explorer {} on wgpu 30; each entry point is compiled on a fresh device", env!("CARGO_PKG_VERSION"));
    let mut failed = 0;
    let mut unchecked_ok = 0;
    for c in pascal::gpu::check_shaders(&adapter) {
        match c.result {
            Ok(()) => println!("  ok    {}::{}", c.module, c.entry),
            Err(e) => {
                failed += 1;
                println!("  FAIL  {}::{}\n        {}", c.module, c.entry, e.replace('\n', "\n        "));
                match c.unchecked_result {
                    Some(Ok(())) => {
                        unchecked_ok += 1;
                        println!("        ok without naga runtime checks (PASCAL_UNCHECKED_SHADERS=1)");
                    }
                    Some(Err(e2)) => println!("        also fails without runtime checks: {}", e2.replace('\n', "\n        ")),
                    None => {}
                }
            }
        }
    }
    if failed == 0 {
        println!("all pipelines compiled");
    } else {
        println!("{failed} pipeline(s) rejected by the driver. Please report this output together with the driver version.");
        if unchecked_ok == failed {
            println!("Every failing pipeline compiles without naga's runtime-check wrappers: run the app with PASCAL_UNCHECKED_SHADERS=1.");
        }
        println!("Other workarounds: WGPU_BACKEND=gl (Linux/Windows) or WGPU_BACKEND=dx12 (Windows) use a different shader compiler path; a newer or older driver branch may also compile it.");
        std::process::exit(1);
    }
}

/// Write rows `0..rows` as an SVG into the export folder; returns the note to show.
/// `valuation_max` is the live view's max ν, so the valuation colors match the screen.
fn save_svg(model: &Model, rows: u32, valuation_max: Option<f32>) -> String {
    let Some(colors) = export::preset_colors(rows, &model.modulus, &model.palette, &model.layers, &model.style, valuation_max) else {
        return "SVG export draws the residue, mask and valuation colorings only.".into();
    };
    let dir = PathBuf::from(model.export_dir.trim());
    let path = export::export_path(&dir, model.m, (0, rows - 1), model.scheme, &model.layers, "svg");
    match export::write_svg(&path, &export::svg_document(&colors, &model.style)) {
        Ok(()) => format!("Saved {rows} rows to {}", path.display()),
        Err(e) => format!("Export failed: {e}"),
    }
}

/// `--export`: the same pipeline as the window, on a headless device.
fn headless_export(mut model: Model, size: (u32, u32), path: &std::path::Path) {
    let gpu = Gpu::new_headless().unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    eprintln!("adapter: {} ({:?})", gpu.info.name, gpu.info.backend);
    let mut renderer = Renderer::new(gpu.device.clone(), gpu.queue.clone());
    model.fit(size.0 as f64, size.1 as f64);
    let grid = GridSpec::new(&model.view, size.0, size.1, model.ss, renderer.max_texels());
    renderer.render_blocking(&FrameInput { grid, modulus: &model.modulus, palette: &model.palette, layers: &model.layers, advance: true, tween: Tween::Off, style: model.style });
    let (w, h, rgba) = renderer.read_output_rgba();
    match export::write_png(path, w, h, &rgba) {
        Ok(()) => eprintln!("wrote {}×{} px to {}", w, h, path.display()),
        Err(e) => {
            eprintln!("export failed: {e}");
            std::process::exit(1);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("Pascal sieve").with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("window"));
        let instance = wgpu::Instance::new(instance_descriptor(Some(window.clone())));
        let surface = instance.create_surface(window.clone()).expect("surface");
        let gpu = pollster::block_on(Gpu::new_async(instance, Some(&surface))).unwrap_or_else(|e| panic!("{e}"));
        log::info!("adapter: {} ({:?})", gpu.info.name, gpu.info.backend);
        gpu.device.set_device_lost_callback(|reason, msg| {
            eprintln!("GPU device lost ({reason:?}): {msg}");
            eprintln!("If the driver printed a shader compiler error above, run with --check-shaders to see which pipeline it rejects, and try WGPU_BACKEND=gl (or dx12 on Windows) as a workaround.");
        });
        let size = window.inner_size();
        let caps = surface.get_capabilities(&gpu.adapter);
        // A non-sRGB surface keeps egui on its gamma path, so the canvas bytes reach the
        // screen unchanged.
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let mut config = surface.get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1)).expect("surface config");
        config.format = format;
        config.view_formats = vec![];
        surface.configure(&gpu.device, &config);

        let egui_ctx = egui::Context::default();
        setup_style(&egui_ctx);
        let egui_state = egui_winit::State::new(egui_ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), window.theme(), Some(gpu.device.limits().max_texture_dimension_2d as usize));
        // no dithering: the canvas bytes are already 8-bit and must reach the screen exactly
        let egui_renderer = egui_wgpu::Renderer::new(&gpu.device, format, egui_wgpu::RendererOptions { dithering: false, ..Default::default() });
        let mut renderer = Renderer::new(gpu.device.clone(), gpu.queue.clone());
        renderer.enable_timing();
        self.gfx = Some(Gfx { window, surface, config, gpu, renderer, egui_ctx, egui_state, egui_renderer, canvas_tex: None, canvas_size: (0, 0) });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gfx) = self.gfx.as_mut() else { return };
        let response = gfx.egui_state.on_window_event(&gfx.window, &event);
        // egui flags every event it saw for a repaint, RedrawRequested included; honoring
        // that one would redraw in a loop forever (continuous frames are asked for below).
        if response.repaint && !matches!(event, WindowEvent::RedrawRequested) {
            gfx.window.request_redraw();
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                // the OS or window manager may leave fullscreen on its own: follow it
                if self.fullscreen_applied && gfx.window.fullscreen().is_none() {
                    self.fullscreen_applied = false;
                    self.model.fullscreen = false;
                }
                if size.width > 0 && size.height > 0 {
                    gfx.config.width = size.width;
                    gfx.config.height = size.height;
                    gfx.surface.configure(&gfx.gpu.device, &gfx.config);
                }
                gfx.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                self.redraw();
                if self.fatal {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    /// egui asks for delayed repaints (tooltips, cursor blink): sleep until then.
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        match self.repaint_at {
            Some(t) if Instant::now() >= t => {
                self.repaint_at = None;
                if let Some(gfx) = &self.gfx {
                    gfx.window.request_redraw();
                }
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            Some(t) => event_loop.set_control_flow(ControlFlow::WaitUntil(t)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }
}

impl App {
    fn redraw(&mut self) {
        let t0 = Instant::now();
        let Some(gfx) = self.gfx.as_mut() else { return };
        let frame = match gfx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                gfx.surface.configure(&gfx.gpu.device, &gfx.config);
                gfx.window.request_redraw();
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                gfx.window.request_redraw();
                return;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                self.surface_errors += 1;
                if self.surface_errors == 1 {
                    eprintln!("The surface returned a validation error; the GPU device is probably lost (see messages above).");
                }
                if self.surface_errors >= 3 {
                    eprintln!("Giving up after repeated surface errors. Run with --check-shaders for a diagnosis.");
                    self.fatal = true;
                }
                return;
            }
        };
        self.surface_errors = 0;
        let ppp = egui_winit::pixels_per_point(&gfx.egui_ctx, &gfx.window);
        let raw_input = gfx.egui_state.take_egui_input(&gfx.window);
        let gpu_ms = gfx.renderer.gpu_ms();
        self.model.play_tick(Instant::now());
        // real frame time for the overlay's easing (one nominal frame after an idle gap)
        let now = Instant::now();
        let dt = match self.last_frame {
            Some(t) if self.continuous => now.duration_since(t).as_secs_f32().min(0.25),
            _ => 1.0 / 60.0,
        };
        self.last_frame = Some(now);
        self.model.advance_offset(dt);
        self.fps_frames += 1;
        let fps_t0 = *self.fps_t0.get_or_insert(now);
        if now.duration_since(fps_t0).as_secs_f32() >= 1.0 {
            log::debug!("{} frames in {:.2} s", self.fps_frames, now.duration_since(fps_t0).as_secs_f32());
            self.fps_frames = 0;
            self.fps_t0 = Some(now);
        }
        // The panel runs before the canvas is laid out: give it last frame's canvas size
        // (the first frame estimates it), so its Fit and Reset fit the real canvas.
        let mut canvas_px = if gfx.canvas_size.0 > 0 {
            gfx.canvas_size
        } else if self.model.fullscreen {
            (gfx.config.width.max(1), gfx.config.height.max(1))
        } else {
            (gfx.config.width.saturating_sub((PANEL_W * ppp) as u32).max(1), gfx.config.height.saturating_sub(((SWEEP_BAR_H + STATUS_H as f32) * ppp) as u32).max(1))
        };
        let model = &mut self.model;
        let frame_ms = self.frame_ms;
        self.frame_nr += 1;
        let frame_nr = self.frame_nr;
        let mut handled_frame = 0u64;
        let ctx = gfx.egui_ctx.clone();
        let Gfx { renderer, egui_renderer, gpu, canvas_tex, canvas_size, .. } = gfx;

        let full_output = ctx.run_ui(raw_input, |ui| {
            // fullscreen: the canvas alone (F11 / Esc)
            if !model.fullscreen {
                // ---- left panel ----
                egui::Panel::left("panel").exact_size(PANEL_W).resizable(false).show(ui, |ui| {
                    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                        ui.set_max_width(PANEL_W - 24.0);
                        panel_ui(ui, model, canvas_px, frame_ms);
                    });
                });

                // ---- sweep bar under the canvas: fixed height, fixed slots ----
                let bar_fill = ui.visuals().panel_fill;
                egui::Panel::bottom("sweep")
                    .exact_size(SWEEP_BAR_H)
                    .resizable(false)
                    .frame(egui::Frame::NONE.fill(bar_fill).inner_margin(egui::Margin::symmetric(10, 6)))
                    .show(ui, |ui| sweep_bar(ui, model));

                // ---- status bar between them: its own strip, so the canvas is exactly the view ----
                egui::Panel::bottom("status")
                    .exact_size(STATUS_H as f32)
                    .resizable(false)
                    .frame(egui::Frame::NONE.fill(Color32::from_gray(14)).inner_margin(egui::Margin::symmetric(8, 0)))
                    .show(ui, |ui| status_bar(ui, model, gpu_ms, frame_ms));
            }

            // ---- canvas ----
            egui::CentralPanel::default().frame(egui::Frame::NONE.fill(CANVAS_BG)).show(ui, |ui| {
                let rect = ui.available_rect_before_wrap();
                let w = ((rect.width() * ppp).round() as u32).max(1);
                let h = ((rect.height() * ppp).round() as u32).max(1);
                canvas_px = (w, h);
                model.canvas = (w as f64, h as f64);
                let mut needs_tex_update = false;
                if *canvas_size != (w, h) {
                    renderer.ensure_canvas(w, h);
                    *canvas_size = (w, h);
                    needs_tex_update = true;
                    if model.auto_fit {
                        model.fit(w as f64, h as f64);
                    }
                }
                let id = match *canvas_tex {
                    Some(id) if !needs_tex_update => id,
                    Some(id) => {
                        egui_renderer.update_egui_texture_from_wgpu_texture(&gpu.device, renderer.display_view(), wgpu::FilterMode::Nearest, id);
                        id
                    }
                    None => {
                        let id = egui_renderer.register_native_texture(&gpu.device, renderer.display_view(), wgpu::FilterMode::Nearest);
                        *canvas_tex = Some(id);
                        id
                    }
                };
                let response = ui.allocate_rect(rect, Sense::click_and_drag());
                ui.painter().image(id, rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                let handle = handled_frame != frame_nr;
                handled_frame = frame_nr;
                canvas_interaction(ui, model, &response, rect, ppp, handle);
                canvas_overlays(ui, model, rect, ppp, handle);
                wave_overlay(ui, model, rect, ppp, dt, handle);
            });
        });
        let egui::FullOutput { platform_output, mut textures_delta, shapes, pixels_per_point, viewport_output } = full_output;
        gfx.egui_state.handle_platform_output(&gfx.window, platform_output);

        // ---- GPU frame ----
        let grid = GridSpec::new(&model.view, canvas_px.0, canvas_px.1, model.ss, gfx.renderer.max_texels());
        // a fitted view draws its rows only
        model.style.row_limit = if model.auto_fit { model.fit_rows.round().clamp(1.0, u32::MAX as f64) as u32 } else { 0 };
        let inputs = (model.m, model.scheme, model.style, model.layers.clone(), model.view, model.ss, canvas_px);
        let changed = self.last_inputs.as_ref() != Some(&inputs);
        // a new color offset alone recolors the same frame: the history layers (temporal
        // AVERAGE, AFTERIMAGE) must not count it as a new one
        let recolor_only = changed && self.last_inputs.as_ref().is_some_and(|l| (l.0, l.1, CanvasStyle { offset: 0.0, ..l.2 }, &l.3, l.4, l.5, l.6) == (inputs.0, inputs.1, CanvasStyle { offset: 0.0, ..inputs.2 }, &inputs.3, inputs.4, inputs.5, inputs.6));
        let advance = changed && !recolor_only;
        self.last_inputs = Some(inputs);
        // Display-only tween between moduli (smoothstep over tween_ms, as in the reference).
        let now = Instant::now();
        let tween = if model.sweep.tween_ms == 0 {
            self.tween_t0 = None;
            Tween::Off
        } else if model.m != self.last_m && self.last_m != 0 {
            self.tween_t0 = Some(now);
            Tween::Start(0.0)
        } else if let Some(t0) = self.tween_t0 {
            let t = (now.duration_since(t0).as_secs_f32() * 1000.0 / model.sweep.tween_ms as f32).min(1.0);
            if t >= 1.0 {
                self.tween_t0 = None;
            }
            Tween::Continue(t * t * (3.0 - 2.0 * t))
        } else {
            Tween::Off
        };
        self.last_m = model.m;
        let mut encoder = gfx.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        let tween_off = matches!(tween, Tween::Off);
        let render_canvas = changed || !tween_off || !self.canvas_current;
        if render_canvas {
            gfx.renderer.render(&mut encoder, &FrameInput { grid, modulus: &model.modulus, palette: &model.palette, layers: &model.layers, advance, tween, style: model.style });
            gfx.renderer.flush_pass_params();
        }
        // After a change, render once more without advancing: the history layers
        // (temporal AVERAGE, AFTERIMAGE) then settle on the image an export reproduces.
        self.canvas_current = tween_off && !advance;

        // ---- egui paint ----
        let paint_jobs = ctx.tessellate(shapes, pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [gfx.config.width, gfx.config.height], pixels_per_point };
        for (id, deltas) in &textures_delta.set {
            for delta in deltas {
                gfx.egui_renderer.update_texture(&gfx.gpu.device, &gfx.gpu.queue, *id, delta);
            }
        }
        let extra = gfx.egui_renderer.update_buffers(&gfx.gpu.device, &gfx.gpu.queue, &mut encoder, &paint_jobs, &screen);
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut rpass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            gfx.egui_renderer.render(&mut rpass, &paint_jobs, &screen);
        }
        gfx.gpu.queue.submit(extra.into_iter().chain(std::iter::once(encoder.finish())));
        if render_canvas {
            gfx.renderer.begin_timing_readback();
        }
        gfx.window.pre_present_notify();
        gfx.gpu.queue.present(frame);
        for id in &textures_delta.free {
            gfx.egui_renderer.free_texture(id);
        }
        textures_delta.clear();

        if model.pending_export {
            model.pending_export = false;
            // Render the true frame (no tween) with the same pipeline, then read it back.
            gfx.renderer.render_blocking(&FrameInput { grid, modulus: &model.modulus, palette: &model.palette, layers: &model.layers, advance: false, tween: Tween::Off, style: model.style });
            self.tween_t0 = None;
            let (w, h, rgba) = gfx.renderer.read_output_rgba();
            let rows = model.shown_rows(h as f64).unwrap_or((0, 0));
            let dir = PathBuf::from(model.export_dir.trim());
            let path = export::export_path(&dir, model.m, rows, model.scheme, &model.layers, "png");
            model.export_note = match export::write_png(&path, w, h, &rgba) {
                Ok(()) => format!("Saved {w}×{h} px to {}", path.display()),
                Err(e) => format!("Export failed: {e}"),
            };
            gfx.window.request_redraw();
        }

        if model.fullscreen != self.fullscreen_applied {
            gfx.window.set_fullscreen(model.fullscreen.then_some(winit::window::Fullscreen::Borderless(None)));
            self.fullscreen_applied = model.fullscreen;
            gfx.window.request_redraw();
        }
        if let Some(rows) = model.pending_svg.take() {
            let vmax = (model.layers.preset() == Some(Preset::Valuation)).then(|| gfx.renderer.read_max()[0]);
            model.export_note = save_svg(model, rows, vmax);
            gfx.window.request_redraw();
        }

        self.frame_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let delay = viewport_output.get(&egui::ViewportId::ROOT).map(|v| v.repaint_delay).unwrap_or(std::time::Duration::MAX);
        self.continuous = delay.is_zero() || model.sweep.playing || self.tween_t0.is_some() || !self.canvas_current || model.offset_playing || model.wave.is_animating();
        if self.continuous {
            gfx.window.request_redraw();
            self.repaint_at = None;
        } else {
            self.repaint_at = Instant::now().checked_add(delay);
        }
    }
}

// ---------------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------------

fn header(ui: &mut egui::Ui, text: &str) {
    ui.add_space(2.0);
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &text.to_uppercase(),
        0.0,
        egui::TextFormat { font_id: FontId::proportional(11.0), extra_letter_spacing: 0.8, color: ui.visuals().weak_text_color(), ..Default::default() },
    );
    ui.label(job);
    ui.add_space(2.0);
}

fn note(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(egui::RichText::new(text).size(12.0).color(ui.visuals().weak_text_color())).wrap());
}

/// A number field in the monospace font.
fn mono_drag<N: egui::emath::Numeric>(ui: &mut egui::Ui, v: &mut N, range: std::ops::RangeInclusive<N>, speed: f64) -> egui::Response {
    mono_drag_with(ui, v, range, speed, true)
}

/// `mono_drag`; with `live` false a typed value applies on Enter or focus loss only.
fn mono_drag_with<N: egui::emath::Numeric>(ui: &mut egui::Ui, v: &mut N, range: std::ops::RangeInclusive<N>, speed: f64, live: bool) -> egui::Response {
    let old = ui.style().override_font_id.clone();
    ui.style_mut().override_font_id = Some(FontId::monospace(12.0));
    let r = ui.add(egui::DragValue::new(v).range(range).speed(speed).update_while_editing(live));
    ui.style_mut().override_font_id = old;
    r
}

/// A single monospace line that never wraps (long text is cut with …, full text on hover),
/// for readouts that change while you work.
fn mono_line(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.add(egui::Label::new(egui::RichText::new(text).font(FontId::monospace(12.0)).color(ui.visuals().weak_text_color())).truncate())
}

fn mono_note(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(egui::RichText::new(text).font(FontId::monospace(12.0)).color(ui.visuals().weak_text_color())).wrap());
}

fn section<R>(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    ui.separator();
    header(ui, title);
    let r = add(ui);
    ui.add_space(4.0);
    r
}

fn panel_ui(ui: &mut egui::Ui, model: &mut Model, canvas_px: (u32, u32), _frame_ms: f32) {
    ui.spacing_mut().item_spacing = Vec2::new(6.0, 5.0);
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Pascal sieve").size(15.0).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("Reset").on_hover_text("Every setting back to its default (the export folder stays)").clicked() {
                model.reset(canvas_px.0 as f64, canvas_px.1 as f64);
            }
        });
    });
    note(ui, "C(n,k) mod m. Every pixel computes its own residue on the GPU — Lucas's theorem for primes, Granville's for prime powers, the Chinese remainder theorem for the rest — so there is no row above to wait for.");

    section(ui, "Modulus", |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("m").color(ui.visuals().weak_text_color()));
            let resp = ui.add(egui::TextEdit::singleline(&mut model.m_text).font(FontId::monospace(12.0)).desired_width(90.0));
            if resp.lost_focus() || (resp.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                match model.m_text.trim().parse::<i64>() {
                    Ok(v) => model.goto(v),
                    Err(_) => model.m_text = model.m.to_string(),
                }
            }
            if ui.add(egui::Button::new("◀").small()).on_hover_text("One step back in the selection mode (←)").clicked() {
                model.step(-1);
            }
            if ui.add(egui::Button::new("▶").small()).on_hover_text("One step forward in the selection mode (→)").clicked() {
                model.step(1);
            }
        });
        let mut m = model.m;
        ui.spacing_mut().slider_width = ui.available_width() - 8.0;
        let slider = ui.add(egui::Slider::new(&mut m, 2..=MOD_CAP).logarithmic(true).show_value(false));
        if slider.drag_started() {
            model.sweep.playing = false;
        }
        if slider.changed() {
            model.goto(m as i64);
        }
        let fs = &model.modulus.factors.iter().map(|t| t.f).collect::<Vec<_>>();
        let info = if model.modulus.is_prime() { format!("{} is prime", model.m) } else { format!("{} = {}", model.m, math::factor_string(fs)) };
        // one fixed line, so nothing below moves while the modulus changes
        mono_line(ui, &info).on_hover_text("Zeros are where every prime-power factor of m divides C(n,k); the CRT layer shows which ones do.");
    });

    section(ui, "View", |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("rows").color(ui.visuals().weak_text_color()));
            let shown = model.fit_rows.round() as u32;
            let mut rows = shown;
            let tip = if model.link_rows { "Linked to the modulus: a new row count moves m to rows ÷ factor" } else { "Rows to fit" };
            // while linked a typed count moves the modulus, so apply it once, on Enter or
            // focus loss (each keystroke would snap and rewrite the text being typed)
            let resp = mono_drag_with(ui, &mut rows, 2..=1_000_000_000u32, 1.0, !model.link_rows).on_hover_text(tip);
            // a deferred field reports changed() for every keystroke of its text; act on
            // the number only
            if resp.changed() && rows != shown {
                model.canvas = (canvas_px.0 as f64, canvas_px.1 as f64);
                model.set_rows(rows.max(2) as f64);
            }
            if ui.button("Fit").on_hover_text("Fit this many rows (F)").clicked() {
                model.fit(canvas_px.0 as f64, canvas_px.1 as f64);
            }
        });
        ui.horizontal(|ui| {
            let mut on = model.link_rows;
            let resp = ui.checkbox(&mut on, "rows follow m").on_hover_text("Parent the rows to the modulus: rows = m × factor, refitted whenever m changes (typed, stepped or swept). With the link on, editing rows or ↑/↓ moves the modulus instead.");
            if resp.changed() {
                model.link_rows = on;
                model.apply_link();
            }
            ui.label(egui::RichText::new("×").color(ui.visuals().weak_text_color()));
            let f = ui.add_enabled_ui(model.link_rows, |ui| mono_drag(ui, &mut model.link_factor, 0.01..=1000.0, 0.01).on_hover_text("rows = m × this")).inner;
            if f.changed() {
                model.apply_link();
            }
        });
        ui.add_enabled_ui(model.link_rows, |ui| {
            ui.checkbox(&mut model.link_hold_sweep, "hold rows while sweeping").on_hover_text("Play or frame-step the sweep (|◀ ▶|) without the row count following the modulus; typed moduli, ◀ ▶ and the arrow keys still move the rows");
        });
        if let Some((a, b)) = model.shown_rows(canvas_px.1 as f64) {
            mono_line(ui, &format!("rows {}–{} · {:.3} px/cell", group(a as u64), group(b as u64), model.view.ppc));
        }
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("supersample").color(ui.visuals().weak_text_color()));
            egui::ComboBox::from_id_salt("ss").selected_text(match model.ss {
                1 => "off",
                2 => "2 × 2",
                _ => "3 × 3",
            }).show_ui(ui, |ui| {
                ui.selectable_value(&mut model.ss, 1, "off");
                ui.selectable_value(&mut model.ss, 2, "2 × 2");
                ui.selectable_value(&mut model.ss, 3, "3 × 3");
            });
        });
        note(ui, "When cells are smaller than a pixel, each pixel averages this many samples. Zoomed in, cells stay crisp.");
        if ui.button("Fullscreen").on_hover_text("The canvas alone, over the whole screen: panel, sweep bar, status and buttons hidden (F11; Esc returns)").clicked() {
            model.fullscreen = true;
        }
        ui.checkbox(&mut model.show_numbers, "numbers when zoomed in");
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let mut on = model.wave.visible();
            if ui.checkbox(&mut on, "wave overlay").on_hover_text("Show or hide the modular wave (W)").changed() {
                model.wave.set_visible(on);
            }
            let corner = &mut model.wave.config.corner;
            egui::ComboBox::from_id_salt("wave corner").selected_text(corner.label()).show_ui(ui, |ui| {
                for c in Corner::ALL {
                    ui.selectable_value(corner, c, c.label());
                }
            });
        });
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("drift").color(ui.visuals().weak_text_color()));
            mono_drag(ui, &mut model.wave.config.drift, 0.0..=3.0, 0.01).on_hover_text("Phase drift φ′ in rad/s; 0 holds the phase");
            ui.label(egui::RichText::new("rad/s").color(ui.visuals().weak_text_color()));
            ui.checkbox(&mut model.wave.config.fill, "fill");
            if let Some(n) = model.pinned_row {
                if ui.small_button(format!("unpin {}", group(n as u64))).on_hover_text("Back to the hovered / last visible row (Esc)").clicked() {
                    model.pinned_row = None;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut model.wave.config.smooth, "smooth").on_hover_text("Ease from one wave to the next; off snaps straight to the new one");
            match model.wave_select {
                WaveSelect::Off => {
                    if ui.small_button("select cells").on_hover_text("Average the waves of a block of cells: click a start cell (red mark), then an end cell (blue mark)").clicked() {
                        model.wave_select = WaveSelect::PickStart;
                    }
                }
                WaveSelect::PickStart | WaveSelect::PickEnd(_) => {
                    if ui.small_button("cancel").on_hover_text("Stop picking cells (Esc)").clicked() {
                        model.wave_select = WaveSelect::Off;
                    }
                }
                WaveSelect::Active(..) => {
                    if ui.small_button("clear selection").on_hover_text("Back to the pinned / hovered row (Esc)").clicked() {
                        model.wave_select = WaveSelect::Off;
                    }
                }
            }
        });
        let status = match model.wave_select {
            WaveSelect::Off => "selection: none".to_string(),
            WaveSelect::PickStart => "click the start cell (red)".to_string(),
            WaveSelect::PickEnd((n, k)) => format!("start ({n}, {k}) · click the end cell (blue)"),
            WaveSelect::Active((n1, k1), (n2, k2)) => format!("cells ({n1}, {k1}) – ({n2}, {k2})"),
        };
        mono_line(ui, &status);
        note(ui, "Plays one row as w(θ) = Σ (C(n,k) mod m)·cos((n−2k)θ + φ): the row under the pointer, else the last visible row. Click a cell to pin its row; a selection plays the average of its cells instead.");
    });

    section(ui, "Color", |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("palette").color(ui.visuals().weak_text_color()));
            let mut s = model.scheme;
            egui::ComboBox::from_id_salt("scheme").selected_text(s.label()).show_ui(ui, |ui| {
                for sc in Scheme::ALL {
                    ui.selectable_value(&mut s, sc, sc.label());
                }
            });
            model.set_scheme(s);
        });
        let preset = model.layers.preset();
        let mut pick = preset;
        ui.radio_value(&mut pick, Some(Preset::Residue), "residue → palette");
        ui.radio_value(&mut pick, Some(Preset::Mask), "zero / nonzero mask");
        ui.radio_value(&mut pick, Some(Preset::Valuation), "p-adic valuation (Kummer heatmap)");
        if pick != preset {
            match pick {
                Some(Preset::Residue) => model.layers.preset_residue(),
                Some(Preset::Mask) => model.layers.preset_mask(),
                Some(Preset::Valuation) => model.layers.preset_valuation(),
                None => {}
            }
        }
        note(ui, match model.scheme {
            Scheme::Spectral => "Zeros are void, 1 is near-white, the other residues run red → violet.",
            _ => "Zeros are void; residues 1..m−1 run along the ramp.",
        });
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("offset").color(ui.visuals().weak_text_color()));
            let r = mono_drag(ui, &mut model.style.offset, -1.0..=1.0, 0.005).on_hover_text("Fractional color offset: moves every residue color this far along the palette, wrapping around (0 and spectral's 1 stay)");
            model.offset_editing = r.has_focus() || r.dragged();
            let playing = model.offset_playing;
            let tip = if playing { "Stop cycling the offset" } else { "Cycle the offset: it turns round and round, one full turn per period, as long as this is on" };
            if ui.add(egui::Button::new(if playing { "❚❚" } else { "▶" }).small().selected(playing)).on_hover_text(tip).clicked() {
                model.offset_playing = !playing;
            }
            mono_drag(ui, &mut model.offset_period, 0.1..=120.0, 0.05).on_hover_text("Seconds per full turn of the cycling offset");
            ui.label(egui::RichText::new("s / turn").color(ui.visuals().weak_text_color()));
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut model.style.outlines, "outlines").on_hover_text("A faint outline around each cell once cells are a few pixels wide");
            ui.checkbox(&mut model.style.glow, "glow").on_hover_text("A soft halo around zero cells — where m divides C(n,k) — once cells are a few pixels wide");
        });
    });

    section(ui, "Layers", |ui| {
        note(ui, "Measures produce a 0..1 field per cell and blend together; transforms run on top in a fixed order.");
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("field color").color(ui.visuals().weak_text_color()));
            ui.selectable_value(&mut model.layers.color_mode, FieldColorMode::MathRamp, "math ramp");
            ui.selectable_value(&mut model.layers.color_mode, FieldColorMode::PaletteBlend, "palette blend");
        });
        note(ui, match model.layers.color_mode {
            FieldColorMode::MathRamp => "The blended field is painted through the palette's ramp.",
            FieldColorMode::PaletteBlend => "Measures light the palette image; transforms run per RGB channel, so a red region ghosts cyan.",
        });
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("blend").color(ui.visuals().weak_text_color()));
            egui::ComboBox::from_id_salt("blend").selected_text(model.layers.blend.label()).show_ui(ui, |ui| {
                for b in BlendMode::ALL {
                    ui.selectable_value(&mut model.layers.blend, b, b.label());
                }
            });
        });
        note(ui, model.layers.blend.note());
        let mut last_kind = None;
        for d in REGISTRY {
            if last_kind != Some(d.kind) {
                last_kind = Some(d.kind);
                if d.kind == Kind::Transform {
                    ui.add_space(4.0);
                    header(ui, "transforms, in order");
                }
            }
            let mut on = model.layers.is_active(d.id);
            let resp = ui.checkbox(&mut on, egui::RichText::new(d.label).font(FontId::proportional(13.0)));
            if resp.changed() {
                model.layers.set_active(d.id, on);
            }
            resp.on_hover_text(d.desc);
            if on {
                ui.indent(d.id as u8, |ui| {
                    ui.set_max_width(ui.available_width());
                    if d.id == LayerId::Average {
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut model.layers.avg_mode, AvgMode::Spatial, "spatial");
                            ui.selectable_value(&mut model.layers.avg_mode, AvgMode::Temporal, "temporal");
                        });
                    }
                    for p in d.params {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(p.label).color(ui.visuals().weak_text_color()));
                            let mut v = model.layers.param(p.key);
                            if !p.choices.is_empty() {
                                let before = v;
                                egui::ComboBox::from_id_salt(p.key).selected_text(p.choices[v.clamp(0, p.choices.len() as i32 - 1) as usize]).show_ui(ui, |ui| {
                                    for (i, c) in p.choices.iter().enumerate() {
                                        ui.selectable_value(&mut v, i as i32, *c);
                                    }
                                });
                                if v != before {
                                    model.layers.params.insert(p.key, v);
                                }
                                return;
                            }
                            let (lo, hi) = if p.key == "avgAmount" && model.layers.avg_mode == AvgMode::Spatial { (1, 12) } else { (p.min, p.max) };
                            let speed = if p.max > 1000 { 1.0 } else { 0.1 };
                            if mono_drag(ui, &mut v, lo..=hi, speed).changed() {
                                model.layers.params.insert(p.key, v);
                            }
                        });
                    }
                    note(ui, first_sentence(d.desc));
                });
            }
        }
    });

    section(ui, "Export", |ui| {
        ui.horizontal(|ui| {
            if ui.button("Save PNG").on_hover_text("The canvas at its native pixel size, from the same shader as the live view").clicked() {
                model.pending_export = true;
            }
            let rows = model.fit_rows.round().max(1.0) as u32;
            let svg_ok = rows <= export::SVG_MAX_ROWS && model.layers.preset().is_some();
            let tip = format!("Rows 0–{} as vector cells (residue, mask and valuation colorings, up to {} rows; outlines included, glow not)", rows - 1, export::SVG_MAX_ROWS);
            if ui.add_enabled(svg_ok, egui::Button::new("Save SVG")).on_hover_text(tip).on_disabled_hover_text(format!("SVG export draws the residue, mask and valuation colorings of up to {} rows", export::SVG_MAX_ROWS)).clicked() {
                model.pending_svg = Some(rows);
            }
        });
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("folder").color(ui.visuals().weak_text_color()));
            ui.add(egui::TextEdit::singleline(&mut model.export_dir).font(FontId::monospace(12.0)).desired_width(f32::INFINITY));
        });
        mono_line(ui, &format!("{} × {} px", canvas_px.0, canvas_px.1));
        if !model.export_note.is_empty() {
            mono_note(ui, &model.export_note);
        }
    });

    ui.separator();
    note(ui, "Keys: space play/pause · ←/→ step · PgUp/PgDn ±10 steps · ↑/↓ rows · shift = ×10 · F fit. Drag to pan, wheel to zoom, double-click to fit. Hover a cell to inspect it; click it to pin its row for the wave (Esc releases) · W wave on/off · F11 fullscreen (Esc returns).");
}

/// Subscript digits: 7 → ₇.
fn sub(n: u32) -> String {
    const SUB: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
    n.to_string().chars().map(|c| SUB[c as usize - '0' as usize]).collect()
}

/// A child area of exactly `w` points wide, so its contents can change (a combo box
/// appearing, a number growing) without moving anything after it.
fn slot<R>(ui: &mut egui::Ui, w: f32, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let h = ui.spacing().interact_size.y;
    ui.allocate_ui_with_layout(Vec2::new(w, h), egui::Layout::left_to_right(egui::Align::Center), |ui| {
        ui.set_min_size(Vec2::new(w, h));
        ui.set_max_width(w);
        add(ui)
    })
    .inner
}

fn bar_label(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).color(ui.visuals().weak_text_color()));
}

/// Fixed-size monospace number field. A typed value applies on Enter or focus loss
/// (dragging stays live): the sweep reads these every frame, and a half-typed value
/// (the "5" of "500" clamped to a 16 ms frame) would already drive it.
fn bar_drag<N: egui::emath::Numeric>(ui: &mut egui::Ui, w: f32, v: &mut N, range: std::ops::RangeInclusive<N>, speed: f64) -> egui::Response {
    let old = ui.style().override_font_id.clone();
    ui.style_mut().override_font_id = Some(FontId::monospace(12.0));
    let h = ui.spacing().interact_size.y;
    let r = ui.add_sized([w, h], egui::DragValue::new(v).range(range).speed(speed).update_while_editing(false));
    ui.style_mut().override_font_id = old;
    r
}

fn bar_button(ui: &mut egui::Ui, label: &str, selected: bool, tip: &str) -> bool {
    let h = ui.spacing().interact_size.y;
    ui.add_sized([30.0, h], egui::Button::new(egui::RichText::new(label).font(FontId::monospace(12.0))).selected(selected)).on_hover_text(tip).clicked()
}

fn bar_sep(ui: &mut egui::Ui) {
    ui.add_space(6.0);
    ui.separator();
    ui.add_space(6.0);
}

/// The sweep controls as a two-row bar under the canvas. Every control sits in a
/// fixed-width slot and explanatory text lives in tooltips, so nothing reflows while
/// the modulus or the mode changes.
fn sweep_bar(ui: &mut egui::Ui, model: &mut Model) {
    let sw = &mut model.sweep;
    let mut action: Option<i32> = None; // Some(0) = jump to start, Some(n) = step n frames
    let before = (sw.mode, sw.prime_order, sw.seq, sw.step, sw.start);
    ui.spacing_mut().item_spacing = Vec2::new(4.0, 6.0);

    // ---- row 1: transport · selection mode · mode options · range ----
    ui.horizontal(|ui| {
        bar_label(ui, "SWEEP");
        ui.add_space(4.0);
        if bar_button(ui, "⇤", false, "Jump to the range start") {
            sw.playing = false;
            action = Some(0);
        }
        if bar_button(ui, "◀◀", sw.playing && sw.dir < 0, "Play in reverse") {
            sw.dir = -1;
            sw.playing = true;
        }
        if bar_button(ui, if sw.playing { "❚❚" } else { "▶" }, false, "Play / pause (space)") {
            sw.playing = !sw.playing;
        }
        if bar_button(ui, "▶▶", sw.playing && sw.dir > 0, "Play forward") {
            sw.dir = 1;
            sw.playing = true;
        }
        if bar_button(ui, "◀|", false, "Step back one frame (←)") {
            action = Some(-1);
        }
        if bar_button(ui, "|▶", false, "Step forward one frame (→)") {
            action = Some(1);
        }
        bar_sep(ui);
        bar_label(ui, "visit");
        egui::ComboBox::from_id_salt("modmode")
            .width(96.0)
            .selected_text(sw.mode.label())
            .show_ui(ui, |ui| {
                for m in ModMode::ALL {
                    ui.selectable_value(&mut sw.mode, m, m.label()).on_hover_text(m.note());
                }
            })
            .response
            .on_hover_text(sw.mode.note());
        // the mode's own options, in a slot of constant width
        slot(ui, 270.0, |ui| match sw.mode {
            ModMode::Prime => {
                egui::ComboBox::from_id_salt("primeorder").width(150.0).selected_text(sw.prime_order.label()).show_ui(ui, |ui| {
                    for o in PrimeOrder::ALL {
                        ui.selectable_value(&mut sw.prime_order, o, o.label());
                    }
                });
            }
            ModMode::Interval => {
                let prev = sw.seq;
                let note = if sw.seq == Seq::Fixed {
                    "Constant step from the range start."
                } else if sw.seq.absolute() {
                    "Lands directly on the sequence's values at or above the range start."
                } else {
                    "Increments follow the sequence from the range start; zero increments are skipped."
                };
                egui::ComboBox::from_id_salt("seq")
                    .width(150.0)
                    .selected_text(sw.seq.label())
                    .show_ui(ui, |ui| {
                        for q in Seq::ALL {
                            ui.selectable_value(&mut sw.seq, q, q.label());
                        }
                    })
                    .response
                    .on_hover_text(note);
                // as in the reference, the 2×Pn and 3×Pn walks start on their first term
                if sw.seq != prev {
                    match sw.seq {
                        Seq::Semi2p => sw.start = 4,
                        Seq::Semi3p => sw.start = 6,
                        _ => {}
                    }
                }
                if sw.seq == Seq::Fixed {
                    bar_label(ui, "step");
                    bar_drag(ui, 56.0, &mut sw.step, 1..=10000u32, 0.2);
                }
            }
            _ => {}
        });
        bar_sep(ui);
        bar_label(ui, "range");
        bar_drag(ui, 70.0, &mut sw.start, 2..=MOD_CAP, 1.0);
        bar_label(ui, "to");
        bar_drag(ui, 70.0, &mut sw.end, 2..=MOD_CAP, 1.0);
    });

    // ---- row 2: frame speed · loop · tween · what the range visits ----
    ui.horizontal(|ui| {
        ui.add_space(54.0);
        bar_label(ui, "frame");
        ui.spacing_mut().slider_width = 110.0;
        // the slider spans 16 ms–1 s; the field takes any value typed or dragged up to 10 s
        ui.add(egui::Slider::new(&mut sw.speed_ms, 16..=1000).show_value(false).clamping(egui::SliderClamping::Never));
        bar_drag(ui, 56.0, &mut sw.speed_ms, 16..=10_000u32, 1.0).on_hover_text("Time per frame while playing, in ms: click to type, drag to change");
        bar_label(ui, "ms");
        ui.checkbox(&mut sw.looping, "loop").on_hover_text("Wrap at the range ends; off stops on the last frame.");
        bar_sep(ui);
        bar_label(ui, "tween");
        bar_drag(ui, 56.0, &mut sw.tween_ms, 0..=3000u32, 5.0).on_hover_text("Display-only transition between moduli: fields morph, plain residue coloring cross-dissolves. Exports, layer history and the inspector use the true frames. 0 = hard cuts. With rows following m the view changes too, so frames cross-fade rather than morph.");
        bar_label(ui, "ms");
        bar_sep(ui);
        let preview = sw.preview(8);
        let frames = sw.frames_in_range(100_000);
        let mut pv: Vec<String> = preview.iter().map(|v| v.to_string()).collect();
        if preview.len() >= 8 {
            pv.push("…".into());
        }
        let text = format!("{} frame{} · {}", group(frames as u64), if frames == 1 { "" } else { "s" }, pv.join(", "));
        mono_line(ui, &text);
    });

    let changed = before != (sw.mode, sw.prime_order, sw.seq, sw.step, sw.start);
    match action {
        Some(0) => {
            let (lo, hi) = model.sweep.bounds();
            let v = if model.sweep.dir > 0 { lo } else { hi };
            model.set_m(v);
        }
        Some(n) => {
            // the sweep's own frame steps, like playback, leave held rows alone
            model.sweep.playing = false;
            let v = model.sweep.step_by(model.m, n);
            model.set_m_with(v, !model.link_hold_sweep);
        }
        None => {}
    }
    // re-snap when the mode or its parameters changed, as the reference does
    if changed {
        model.goto(model.m as i64);
    }
}

fn first_sentence(s: &str) -> &str {
    match s.find(". ") {
        Some(i) => &s[..i + 1],
        None => s,
    }
}

/// `1234567` → `1,234,567`.
fn group(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

// ---------------------------------------------------------------------------------
// Canvas interaction and overlays
// ---------------------------------------------------------------------------------

fn canvas_interaction(ui: &mut egui::Ui, model: &mut Model, response: &egui::Response, rect: Rect, ppp: f32, handle: bool) {
    // the canvas in whole pixels, as the GPU and the fits use it (rect × ppp is
    // fractional at fractional scale factors)
    let (cw, ch) = model.canvas;
    let to_px = |p: Pos2| ((p.x - rect.min.x) as f64 * ppp as f64, (p.y - rect.min.y) as f64 * ppp as f64);

    if handle && response.dragged_by(egui::PointerButton::Primary) {
        let d = response.drag_delta();
        if d != Vec2::ZERO {
            model.view.pan(d.x as f64 * ppp as f64, d.y as f64 * ppp as f64);
            model.auto_fit = false;
        }
    }
    if handle && response.double_clicked() {
        model.fit(cw, ch);
    }
    if handle && response.hovered() {
        let (scroll, zoom, pointer) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta(), i.pointer.hover_pos()));
        if let Some(p) = pointer {
            let (px, py) = to_px(p);
            let mut factor = (scroll as f64 * 0.003).exp();
            if zoom != 1.0 {
                factor *= zoom as f64;
            }
            if factor != 1.0 {
                model.view.zoom_at(px, py, factor);
                model.auto_fit = false;
            }
        }
    }
    let pointer = response.hover_pos().filter(|_| response.hovered() && !response.dragged());
    model.over_wave = pointer.is_some_and(|p| model.wave.box_rect().is_some_and(|b| b.contains(p)));
    let pointer = pointer.filter(|_| !model.over_wave).map(to_px);
    // a fitted view shows its rows only: nothing to hover below them
    let shown = |n: u32| !model.auto_fit || (n as f64) < model.fit_rows.round();
    model.hover = pointer.and_then(|(px, py)| model.view.cell_at(px, py)).filter(|c| shown(c.0));
    model.hover_row = pointer.and_then(|(_, py)| model.view.row_at(py)).filter(|&n| shown(n));
    model.dragging = response.dragged();
    // A click on a cell pins its row for the wave overlay; a click on the pinned row (or
    // Esc) releases it. A double click (fit) restores the pin from before its first click.
    if handle && response.double_clicked() {
        model.pinned_row = model.pin_before_click;
        model.wave_select = model.select_before_click;
    } else if handle && response.clicked() {
        model.pin_before_click = model.pinned_row;
        model.select_before_click = model.wave_select;
        if matches!(model.wave_select, WaveSelect::PickStart | WaveSelect::PickEnd(_)) {
            // picking a selection: clicks mark cells instead of pinning rows
            if let Some(cell) = model.hover {
                model.select_click(cell);
            }
        } else {
            if let Some((n, _)) = model.hover {
                model.pinned_row = if model.pinned_row == Some(n) { None } else { Some(n) };
            }
        }
    }

    // F11 even while a text field has focus (none of them takes it)
    if handle && ui.input(|i| i.key_pressed(egui::Key::F11)) {
        model.fullscreen = !model.fullscreen;
    }

    // Keyboard, when no text field has focus.
    if handle && !ui.ctx().egui_wants_keyboard_input() {
        let (space, shift, right, left, pgup, pgdn, up, down, fit, wave, esc) = ui.input(|i| {
            (
                i.key_pressed(egui::Key::Space),
                i.modifiers.shift,
                i.num_presses(egui::Key::ArrowRight),
                i.num_presses(egui::Key::ArrowLeft),
                i.num_presses(egui::Key::PageUp),
                i.num_presses(egui::Key::PageDown),
                i.num_presses(egui::Key::ArrowUp),
                i.num_presses(egui::Key::ArrowDown),
                i.key_pressed(egui::Key::F),
                i.key_pressed(egui::Key::W),
                i.key_pressed(egui::Key::Escape),
            )
        });
        if wave {
            model.wave.toggle();
        }
        // Esc that left a text field or closed a popup is not an unpin
        if esc && !model.focus_last_frame {
            // fullscreen goes first, then a selection (or picking one), then the pin
            if model.fullscreen {
                model.fullscreen = false;
            } else if model.wave_select != WaveSelect::Off {
                model.wave_select = WaveSelect::Off;
            } else {
                model.pinned_row = None;
            }
        }
        let big: i64 = if shift { 10 } else { 1 };
        if space {
            model.sweep.playing = !model.sweep.playing;
        }
        // ←/→ one step through the selection mode, PgUp/PgDn ten; shift ×10
        let dm = (right as i64 - left as i64) * big + (pgup as i64 - pgdn as i64) * 10 * big;
        if dm != 0 {
            model.step(dm.clamp(-100_000, 100_000) as i32);
        }
        let dr = (up as i64 - down as i64) * big;
        if dr != 0 {
            if model.link_rows {
                // rows are the modulus's child: step the modulus, the rows follow
                model.step(dr.clamp(-100_000, 100_000) as i32);
            } else {
                model.fit_rows = (model.fit_rows + dr as f64).clamp(2.0, 1.0e9);
                model.fit(cw, ch);
            }
        }
        if fit {
            model.fit(cw, ch);
        }
    }
    if handle {
        model.focus_last_frame = ui.ctx().memory(|m| m.focused().is_some()) || ui.ctx().any_popup_open();
    }
}

fn canvas_overlays(ui: &mut egui::Ui, model: &mut Model, rect: Rect, ppp: f32, handle: bool) {
    let painter = ui.painter_at(rect);
    // the canvas in whole pixels, as the GPU and the fits use it (rect × ppp is
    // fractional at fractional scale factors)
    let (cw, ch) = model.canvas;
    let to_pt = |x: f64, y: f64| Pos2::new(rect.min.x + (x / ppp as f64) as f32, rect.min.y + (y / ppp as f64) as f32);

    // Residue numbers inside cells, when they are big enough.
    if model.show_numbers && model.view.ppc >= NUMBERS_MIN_PX && model.view.visible_cells(cw, ch) <= NUMBERS_MAX_CELLS {
        if let Some((n_lo, n_hi)) = model.shown_rows(ch) {
            // capped, and stepped (four sizes an octave) so a smooth zoom reuses glyphs
            // instead of rasterizing a new size every frame
            let pt = ((model.view.ppc * 0.34).min(model.view.ppc * ROW_H * 0.55) / ppp as f64).clamp(4.0, NUMBERS_MAX_PT);
            let font = FontId::monospace(((pt.log2() * 4.0).round() / 4.0).exp2() as f32);
            let x1 = model.view.x0 + cw / model.view.ppc;
            let level_map = model.palette.level_map(model.m, model.style.offset);
            for n in n_lo..=n_hi {
                let k_lo = (model.view.x0 + n as f64 / 2.0 - 0.5).ceil().max(0.0) as i64;
                let k_hi = (x1 + n as f64 / 2.0 + 0.5).floor().min(n as f64) as i64;
                for k in k_lo..=k_hi {
                    let k = k as u32;
                    let Some(text) = model.cell_value(n, k) else { break };
                    let r = model.modulus.binom(n, k);
                    let (rgb, show) = match model.layers.preset() {
                        Some(Preset::Residue) => (model.palette.residue_color_mapped(r, model.m, &level_map), r != 0),
                        Some(Preset::Mask) => (pascal::palette::LIGHT, r != 0),
                        Some(Preset::Valuation) => {
                            let nu = math::carries(n, k, model.modulus.p) as f64;
                            let max = ((n_hi.max(1) as f64).ln() / (model.modulus.p as f64).ln()).floor().max(1.0);
                            (model.palette.ramp[((nu / max).clamp(0.0, 1.0) * 255.0).round() as usize], true)
                        }
                        None => break,
                    };
                    if !show {
                        continue;
                    }
                    let lum = 0.299 * rgb[0] as f64 + 0.587 * rgb[1] as f64 + 0.114 * rgb[2] as f64;
                    let color = if lum > 140.0 { Color32::BLACK } else { Color32::WHITE };
                    let (x, y, w, h) = model.view.cell_rect(n, k);
                    painter.text(to_pt(x + w / 2.0, y + h / 2.0), Align2::CENTER_CENTER, text, font.clone(), color);
                }
            }
        }
    }

    // Hover outline + inspector.
    if let Some((n, k)) = model.hover {
        let (x, y, w, h) = model.view.cell_rect(n, k);
        let r = Rect::from_min_max(to_pt(x, y), to_pt(x + w, y + h));
        painter.rect_stroke(r, 0.0, egui::Stroke::new(3.0, Color32::BLACK), egui::StrokeKind::Outside);
        painter.rect_stroke(r, 0.0, egui::Stroke::new(1.5, Color32::WHITE), egui::StrokeKind::Outside);
        let text = model.inspector_text(n, k);
        let galley = painter.layout(text, FontId::monospace(12.0), Color32::from_rgb(221, 221, 221), (rect.width() - 16.0).min(520.0));
        let size = galley.size() + Vec2::new(18.0, 12.0);
        let mut pos = rect.min + Vec2::new(8.0, 8.0);
        // keep clear of the wave overlay: below it, else just above the status strip
        if let Some(b) = model.wave.box_rect() {
            let floor = rect.max.y;
            let clear = |p: Pos2| {
                let r = Rect::from_min_size(p, size);
                !r.intersects(b.expand(4.0)) && r.max.y <= floor
            };
            if !clear(pos) {
                let below = Pos2::new(pos.x, b.max.y + 8.0);
                let bottom = Pos2::new(pos.x, floor - size.y - 8.0);
                pos = [below, bottom].into_iter().find(|p| clear(*p)).unwrap_or(pos);
            }
        }
        let bg = Rect::from_min_size(pos, size);
        painter.rect_filled(bg, 0.0, Color32::from_black_alpha(200));
        painter.galley(pos + Vec2::new(9.0, 6.0), galley, Color32::from_rgb(221, 221, 221));
    }

    // −/+/Fit at the top right (not in fullscreen).
    let btn_h = 22.0;
    let mut x = rect.max.x - 8.0;
    let buttons: &[(&str, u8)] = if model.fullscreen { &[] } else { &[("Fit", 2u8), ("+", 1), ("−", 0)] };
    for &(label, action) in buttons {
        let w = if action == 2 { 34.0 } else { 24.0 };
        x -= w;
        let r = Rect::from_min_size(Pos2::new(x, rect.min.y + 8.0), Vec2::new(w, btn_h));
        if ui.put(r, egui::Button::new(label)).clicked() && handle {
            match action {
                0 => model.view.zoom_at(cw / 2.0, ch / 2.0, 1.0 / 1.4),
                1 => model.view.zoom_at(cw / 2.0, ch / 2.0, 1.4),
                _ => model.fit(cw, ch),
            }
            if action != 2 {
                model.auto_fit = false;
            }
        }
        x -= 4.0;
    }
}

/// The status bar under the canvas: modulus, the rows and cells in view, timings.
fn status_bar(ui: &mut egui::Ui, model: &Model, gpu_ms: Option<f32>, frame_ms: f32) {
    let (cw, ch) = model.canvas;
    let mut parts = vec![format!("m = {}", model.m)];
    if let Some((a, b)) = model.shown_rows(ch) {
        parts.push(format!("rows {}–{}", group(a as u64), group(b as u64)));
    }
    parts.push(format!("{} cells", group(model.view.visible_cells(cw, ch))));
    if let Some(g) = gpu_ms {
        parts.push(format!("gpu {g:.1} ms"));
    }
    parts.push(format!("frame {frame_ms:.1} ms"));
    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
        ui.add(egui::Label::new(egui::RichText::new(parts.join("  ·  ")).font(FontId::monospace(11.0)).color(Color32::from_rgb(187, 187, 187))).truncate());
    });
}

/// The "modular wave" overlay, drawn last so it sits on top of the canvas. The state is
/// advanced once per frame (egui may run the UI closure twice); drawing happens every pass.
fn wave_overlay(ui: &mut egui::Ui, model: &mut Model, rect: Rect, ppp: f32, dt: f32, handle: bool) {
    // the canvas in whole pixels, as the GPU and the fits use it (rect × ppp is
    // fractional at fractional scale factors)
    let (cw, ch) = model.canvas;
    if handle {
        let row = if let Some(n) = model.pinned_row {
            n
        } else if let Some(n) = model.hover_row {
            n
        } else if model.over_wave || model.dragging {
            model.wave_row // on the overlay, or panning: keep playing the same row
        } else {
            // the last row wholly in view (a sliver of the next one does not count), and
            // while fitted never past the rows asked for
            let last = model.view.full_rows(ch).or(model.view.visible_rows(ch)).map_or(model.wave_row, |r| r.1);
            if model.auto_fit { last.min((model.fit_rows.round() as u32).saturating_sub(1)) } else { last }
        };
        model.wave_row = row;
        model.wave.share_closed_form(&model.modulus);
        if let WaveSelect::Active(a, b) = model.wave_select {
            model.wave.set_region(model.m as u64, (a.0 as u64, a.1 as u64), (b.0 as u64, b.1 as u64));
        } else {
            let (k_lo, k_hi) = model.view.visible_k(row, cw);
            model.wave.set_source(row as u64, model.m as u64, (k_lo as u64, k_hi as u64));
        }
        model.wave.set_pinned(model.pinned_row.is_some());
        model.wave.set_viewport(rect, ui.ctx().viewport_rect().size(), ppp);
        model.wave.update(dt);
    }
    let painter = ui.painter_at(rect);
    // the selection: its rectangle, a red start mark and a blue end mark (the end
    // follows the pointer while it is being picked)
    let to_pt = |x: f64, y: f64| Pos2::new(rect.min.x + (x / ppp as f64) as f32, rect.min.y + (y / ppp as f64) as f32);
    let cell_box = |(n, k): (u32, u32)| {
        let (x, y, w, h) = model.view.cell_rect(n, k);
        Rect::from_min_max(to_pt(x, y), to_pt(x + w, y + h))
    };
    let (start, end) = match model.wave_select {
        WaveSelect::PickEnd(a) => (Some(a), model.hover),
        WaveSelect::Active(a, b) => (Some(a), Some(b)),
        _ => (None, None),
    };
    if let (Some(a), Some(b)) = (start, end) {
        let r = cell_box(a).union(cell_box(b));
        painter.rect_stroke(r, 0.0, egui::Stroke::new(1.0, Color32::from_white_alpha(140)), egui::StrokeKind::Outside);
    }
    for (cell, rgb) in [(start, (235, 64, 64)), (end, (70, 130, 255))] {
        if let Some(c) = cell {
            let b = cell_box(c);
            // at least 8 pt across, so a mark shows when cells are tiny
            let b = if b.width() < 8.0 { Rect::from_center_size(b.center(), Vec2::splat(8.0)) } else { b };
            let col = Color32::from_rgb(rgb.0, rgb.1, rgb.2);
            painter.rect_filled(b, 0.0, col.gamma_multiply(0.35));
            painter.rect_stroke(b, 0.0, egui::Stroke::new(2.0, col), egui::StrokeKind::Outside);
        }
    }
    // ticks at the canvas edges mark the row the wave plays (solid when pinned)
    let fade = model.wave.fade();
    if fade > 0.0 && !matches!(model.wave_select, WaveSelect::Active(..)) {
        let (_, y, _, h) = model.view.cell_rect(model.wave_row, 0);
        let yc = rect.min.y + ((y + h / 2.0) / ppp as f64) as f32;
        if (rect.min.y..=rect.max.y).contains(&yc) {
            let a = if model.pinned_row.is_some() { 0.95 } else { 0.55 };
            let stroke = egui::Stroke::new(2.0, Color32::from_rgba_unmultiplied(0x4f, 0xc3, 0xf7, (a * fade * 255.0) as u8));
            painter.line_segment([Pos2::new(rect.min.x, yc), Pos2::new(rect.min.x + 10.0, yc)], stroke);
            painter.line_segment([Pos2::new(rect.max.x - 10.0, yc), Pos2::new(rect.max.x, yc)], stroke);
        }
    }
    model.wave.draw(&painter);
}

// ---------------------------------------------------------------------------------
// Style and fonts
// ---------------------------------------------------------------------------------

fn setup_style(ctx: &egui::Context) {
    ctx.set_fonts(system_fonts());
    ctx.all_styles_mut(|s| {
        use egui::TextStyle::*;
        s.text_styles.insert(Small, FontId::proportional(11.0));
        s.text_styles.insert(Body, FontId::proportional(13.0));
        s.text_styles.insert(Monospace, FontId::monospace(12.0));
        s.text_styles.insert(Button, FontId::proportional(13.0));
        s.text_styles.insert(Heading, FontId::proportional(15.0));
        s.spacing.item_spacing = Vec2::new(6.0, 5.0);
        s.spacing.button_padding = Vec2::new(8.0, 3.0);
        s.spacing.interact_size.y = 20.0;
        s.spacing.combo_width = 120.0;
        for w in [&mut s.visuals.widgets.noninteractive, &mut s.visuals.widgets.inactive, &mut s.visuals.widgets.hovered, &mut s.visuals.widgets.active, &mut s.visuals.widgets.open] {
            w.corner_radius = egui::CornerRadius::same(2);
        }
        s.visuals.slider_trailing_fill = false;
    });
}

/// egui's bundled fonts, with the OS sans-serif and monospace fonts in front when a
/// readable file can be found (best effort; the bundled fonts always remain as fallback).
fn system_fonts() -> egui::FontDefinitions {
    let mut defs = egui::FontDefinitions::default();
    let candidates: &[(&str, egui::FontFamily, u32)] = &[
        // Linux
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", egui::FontFamily::Proportional, 0),
        ("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", egui::FontFamily::Proportional, 0),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", egui::FontFamily::Proportional, 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", egui::FontFamily::Monospace, 0),
        ("/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf", egui::FontFamily::Monospace, 0),
        ("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", egui::FontFamily::Monospace, 0),
        // macOS
        ("/System/Library/Fonts/Supplemental/Arial.ttf", egui::FontFamily::Proportional, 0),
        ("/System/Library/Fonts/Helvetica.ttc", egui::FontFamily::Proportional, 0),
        ("/System/Library/Fonts/Menlo.ttc", egui::FontFamily::Monospace, 0),
        ("/System/Library/Fonts/Monaco.ttf", egui::FontFamily::Monospace, 0),
        // Windows
        ("C:\\Windows\\Fonts\\segoeui.ttf", egui::FontFamily::Proportional, 0),
        ("C:\\Windows\\Fonts\\arial.ttf", egui::FontFamily::Proportional, 0),
        ("C:\\Windows\\Fonts\\consola.ttf", egui::FontFamily::Monospace, 0),
        ("C:\\Windows\\Fonts\\cour.ttf", egui::FontFamily::Monospace, 0),
    ];
    let mut have = std::collections::HashSet::new();
    for (path, family, index) in candidates {
        if have.contains(family) {
            continue;
        }
        let Ok(bytes) = std::fs::read(path) else { continue };
        if ab_glyph::FontRef::try_from_slice_and_index(&bytes, *index).is_err() {
            continue;
        }
        let name = format!("system-{}", path.replace(['/', '\\', ':'], "_"));
        let mut data = egui::FontData::from_owned(bytes);
        data.index = *index;
        defs.font_data.insert(name.clone(), Arc::new(data));
        defs.families.entry(family.clone()).or_default().insert(0, name);
        have.insert(family.clone());
    }
    defs
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
