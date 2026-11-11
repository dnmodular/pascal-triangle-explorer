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
use pascal::sweep::{KinOrder, ModMode, PrimeOrder, Seq, SpDenoms, SpOrder, Sweep};
use pascal::view::{GridSpec, View, MAX_PPC, MIN_PPC, ROW_H, STATUS_H};
use pascal::wave_overlay::{Corner, WaveOverlay};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

mod canvas;
mod model;
mod panel;
mod sweep_bar;
mod tetra_ui;
mod widgets;
use canvas::*;
use model::*;
use panel::*;
use sweep_bar::*;
use tetra_ui::{TetraCache, TetraView};
use widgets::*;

const PANEL_W: f32 = 300.0;
/// Height of the sweep bar under the canvas (two rows of controls).
const SWEEP_BAR_H: f32 = 64.0;
const CANVAS_BG: Color32 = Color32::from_rgb(10, 10, 10);
/// Cells at least this wide (px) get their residue printed inside.
const NUMBERS_MIN_PX: f64 = 18.0;
const NUMBERS_MAX_CELLS: u64 = 9000;
/// Largest in-cell number, in points (egui's glyph atlas grows with every size drawn).
const NUMBERS_MAX_PT: f64 = 96.0;

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
    tetra: pascal::tetra_gpu::TetraRenderer,
    tetra_tex: Option<egui::TextureId>,
    tetra_cache: TetraCache,
}

/// Everything that changes what the GPU renders: modulus, palette, style, layers, camera,
/// supersampling and canvas size.
type FrameKey = ((u32, u32, (u32, u32)), Scheme, CanvasStyle, LayerState, View, u32, (u32, u32));

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
    last_m: (u32, u32),
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
  --m N              modulus: whole, decimal (10.5, 0.7) or fraction (37/5) (default 7)
  --morph            draw a decimal --m as a morph between the whole moduli around it
  --rows R           rows to fit (default 128)
  --palette NAME     spectral | viridis | magma | inferno | turbo | cividis | coolwarm | mono
                     | ember | ocean | moss
  --offset X         fractional color offset (−1..1)
  --link             rows follow the modulus (rows = m)
  --outlines         outline each cell when zoomed in
  --glow             glow around zero cells when zoomed in
  --mode MODE        residue | mask | valuation
  --visit MODE       sweep selection: all | primes | composites | even | odd | interval
                     | super primes | kinship
  --kin ORDER        kinship order: one-factor | look-alike
  --fraction-colors  value | phase | auto (how fractions color)
  --layers a,b,c     active layers: value kummer lucas crt dlog pdigit gradient distzero magnitude
                     density variance rarity moire primes stripes lagdiff magnify average bandpass
                     afterimage
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
    let mut m_text: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let next = || args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            // a whole or decimal modulus ("10.5"), applied once every flag is read
            "--m" => m_text = Some(next()),
            "--morph" => {
                model.sweep.morph = true;
                i += 1;
                continue;
            }
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
            // the kinship walk's order: one-factor or look-alike
            "--kin" => {
                let v = next().replace('-', " ");
                if let Some(o) = KinOrder::ALL.into_iter().find(|o| o.label() == v) {
                    model.sweep.kin_order = o;
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
            "--fraction-colors" => {
                let v = next();
                model.frac_colors = FracColors::ALL.into_iter().find(|f| f.label() == v).unwrap_or_default();
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
    if let Some(text) = m_text {
        if !model.goto_text(&text) {
            model.set_m(7);
        }
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
    let mut app = App { gfx: None, model, last_inputs: None, frame_ms: 0.0, frame_nr: 0, surface_errors: 0, fatal: false, last_m: (0, 1), tween_t0: None, canvas_current: false, last_frame: None, continuous: false, fps_frames: 0, fps_t0: None, repaint_at: None, fullscreen_applied: false };
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
    renderer.render_blocking(&FrameInput { grid, modulus: &model.modulus, palette: &model.palette, layers: &model.layers, advance: true, tween: Tween::Off, style: model.frame_style(), morph: model.morph_target() });
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
        let tetra = pascal::tetra_gpu::TetraRenderer::new(gpu.device.clone(), gpu.queue.clone());
        self.gfx = Some(Gfx { window, surface, config, gpu, renderer, egui_ctx, egui_state, egui_renderer, canvas_tex: None, canvas_size: (0, 0), tetra, tetra_tex: None, tetra_cache: TetraCache::default() });
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
        let real_dt = self.last_frame.map_or(1.0 / 60.0, |t| now.duration_since(t).as_secs_f32().min(0.25));
        let dt = if self.continuous { real_dt } else { 1.0 / 60.0 };
        self.last_frame = Some(now);
        self.model.advance_offset(dt);
        // the tile spins on its own ~30 fps timer, not continuous frames: real time
        self.model.tetra.advance(if self.model.tetra.spinning() { real_dt.min(0.1) } else { dt });
        // the tetra stack's pick from an earlier frame
        if let Some(hit) = gfx.tetra.poll_pick() {
            self.model.tetra.land_pick(hit);
        }
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
        let Gfx { renderer, egui_renderer, gpu, canvas_tex, canvas_size, tetra_tex, .. } = gfx;
        let tetra_tex_id = *tetra_tex;
        let canvas_tex_id = *canvas_tex;

        let full_output = ctx.run_ui(raw_input, |ui| {
            // the tetra stack's own space: nothing else on screen
            if model.tetra.open {
                let handle = handled_frame != frame_nr;
                handled_frame = frame_nr;
                tetra_ui::space(ui, model, ppp, tetra_tex_id, canvas_tex_id, *canvas_size, handle);
                return;
            }
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
                tetra_ui::tile(ui, model, rect, ppp, tetra_tex_id);
            });
        });
        let egui::FullOutput { platform_output, mut textures_delta, shapes, pixels_per_point, viewport_output } = full_output;
        gfx.egui_state.handle_platform_output(&gfx.window, platform_output);

        // ---- GPU frame ----
        let grid = GridSpec::new(&model.view, canvas_px.0, canvas_px.1, model.ss, gfx.renderer.max_texels());
        // a fitted view draws its rows only
        model.style.row_limit = if model.auto_fit { model.fit_rows.round().clamp(1.0, u32::MAX as f64) as u32 } else { 0 };
        let inputs = ((model.modulus.m, model.modulus.q, model.morph.as_ref().map_or((0, 0), |(b, t)| (b.m, t.to_bits()))), model.scheme, model.frame_style(), model.layers.clone(), model.view, model.ss, canvas_px);
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
        } else if (model.modulus.m, model.modulus.q) != self.last_m && self.last_m.0 != 0 {
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
        self.last_m = (model.modulus.m, model.modulus.q);
        let mut encoder = gfx.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        let tween_off = matches!(tween, Tween::Off);
        let render_canvas = changed || !tween_off || !self.canvas_current;
        // the tetra stack, into its own target (before egui samples it)
        if tetra_ui::render(model, &mut gfx.tetra_cache, &mut gfx.tetra, &mut encoder, ctx.dragged_id().is_some()) || (gfx.tetra_tex.is_none() && gfx.tetra.target_view().is_some()) {
            let view = gfx.tetra.target_view().expect("target");
            match gfx.tetra_tex {
                Some(id) => gfx.egui_renderer.update_egui_texture_from_wgpu_texture(&gfx.gpu.device, view, wgpu::FilterMode::Linear, id),
                None => gfx.tetra_tex = Some(gfx.egui_renderer.register_native_texture(&gfx.gpu.device, view, wgpu::FilterMode::Linear)),
            }
        }
        if render_canvas {
            gfx.renderer.render(&mut encoder, &FrameInput { grid, modulus: &model.modulus, palette: &model.palette, layers: &model.layers, advance, tween, style: model.frame_style(), morph: model.morph_target() });
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
        gfx.tetra.start_pick_map();
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
            gfx.renderer.render_blocking(&FrameInput { grid, modulus: &model.modulus, palette: &model.palette, layers: &model.layers, advance: false, tween: Tween::Off, style: model.frame_style(), morph: model.morph_target() });
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
        self.continuous = delay.is_zero() || model.sweep.playing || self.tween_t0.is_some() || !self.canvas_current || model.offset_playing || (!model.tetra.open && model.wave.is_animating()) || gfx.tetra.pick_in_flight() || model.tetra.filling();
        if self.continuous {
            gfx.window.request_redraw();
            self.repaint_at = None;
        } else {
            // the preview tile turns at about 30 frames a second
            let delay = if model.tetra.spinning() { delay.min(std::time::Duration::from_millis(33)) } else { delay };
            self.repaint_at = Instant::now().checked_add(delay);
        }
    }
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

