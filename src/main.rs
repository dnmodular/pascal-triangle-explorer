//! The desktop app: a winit window, an egui control panel on the left and the GPU
//! canvas filling the rest. All math and rendering live in the `pascal` library; this
//! file is the UI, the camera interaction and the export plumbing.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Vec2};
use pascal::export;
use pascal::gpu::{FrameInput, Gpu, Renderer, Tween};
use pascal::layers::{AvgMode, BlendMode, FieldColorMode, Kind, LayerId, LayerState, Preset, REGISTRY};
use pascal::math::{self, Modulus, MOD_CAP};
use pascal::palette::{Palette, Scheme};
use pascal::sweep::{ModMode, PrimeOrder, Seq, Sweep};
use pascal::view::{GridSpec, View, ROW_H, STATUS_H};
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

/// Everything the user can change.
struct Model {
    m: u32,
    /// Shared with the wave overlay, which reads long rows through the same closed form.
    modulus: Arc<Modulus>,
    m_text: String,
    scheme: Scheme,
    palette: Palette,
    layers: LayerState,
    view: View,
    fit_rows: f64,
    auto_fit: bool,
    ss: u32,
    show_numbers: bool,
    export_dir: String,
    export_note: String,
    pending_export: bool,
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
    wave_row: u32,
    /// The pointer is over the overlay box: the canvas under it gets no hover.
    over_wave: bool,
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
            layers: LayerState::default(),
            view: View::fit(800.0, 600.0, 128.0),
            fit_rows: 128.0,
            auto_fit: true,
            ss: 2,
            show_numbers: true,
            export_dir: std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| ".".into()),
            export_note: String::new(),
            pending_export: false,
            hover: None,
            hover_row: None,
            sweep: Sweep::default(),
            last_step: None,
            wave: WaveOverlay::new(),
            pinned_row: None,
            wave_row: 0,
            over_wave: false,
        }
    }

    fn set_m(&mut self, m: u32) {
        let m = m.clamp(2, MOD_CAP);
        if m != self.m {
            self.m = m;
            self.modulus = Arc::new(Modulus::new(m));
        }
        self.m_text = m.to_string();
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
        self.set_m(v);
        moved
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

/// Everything that changes what the GPU renders: modulus, palette, layers, camera,
/// supersampling and canvas size.
type FrameKey = (u32, Scheme, LayerState, View, u32, (u32, u32));

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
    /// The display texture holds the true frame of `last_inputs` (no tween in it), so a
    /// redraw that changes none of them (overlay animation, hover) can skip the GPU passes.
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
}

const USAGE: &str = "pascal-triangle-explorer [options]
  --m N              modulus (default 7)
  --rows R           rows to fit (default 128)
  --palette NAME     spectral | viridis | magma | inferno | turbo | cividis | coolwarm | mono
  --mode MODE        residue | mask | valuation
  --visit MODE       sweep selection: all | primes | composites | even | odd | interval
  --layers a,b,c     active layers: value kummer lucas crt dlog pdigit gradient distzero magnitude
                     density variance rarity moire primes stripes lagdiff average bandpass afterimage
  --field MODE       math | palette (field color mode)
  --export FILE.png  render once without a window and write a PNG
  --size WxH         canvas size for --export (default 1600x1000)
  --check-shaders    compile every shader pipeline one by one and report what the driver rejects";

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model = Model::new();
    let mut export: Option<PathBuf> = None;
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
    if let Some(path) = export {
        headless_export(model, size, &path);
        return;
    }
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App { gfx: None, model, last_inputs: None, frame_ms: 0.0, frame_nr: 0, surface_errors: 0, fatal: false, last_m: 0, tween_t0: None, canvas_current: false, last_frame: None, continuous: false, fps_frames: 0, fps_t0: None, repaint_at: None };
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
    renderer.render_blocking(&FrameInput { grid, modulus: &model.modulus, palette: &model.palette, layers: &model.layers, advance: true, tween: Tween::Off });
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
        let egui_renderer = egui_wgpu::Renderer::new(&gpu.device, format, egui_wgpu::RendererOptions::default());
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
        self.fps_frames += 1;
        let fps_t0 = *self.fps_t0.get_or_insert(now);
        if now.duration_since(fps_t0).as_secs_f32() >= 1.0 {
            log::debug!("{} frames in {:.2} s", self.fps_frames, now.duration_since(fps_t0).as_secs_f32());
            self.fps_frames = 0;
            self.fps_t0 = Some(now);
        }
        let mut canvas_px = (gfx.config.width.saturating_sub((PANEL_W * ppp) as u32).max(1), gfx.config.height.max(1));
        let model = &mut self.model;
        let frame_ms = self.frame_ms;
        self.frame_nr += 1;
        let frame_nr = self.frame_nr;
        let mut handled_frame = 0u64;
        let ctx = gfx.egui_ctx.clone();
        let Gfx { renderer, egui_renderer, gpu, canvas_tex, canvas_size, .. } = gfx;

        let full_output = ctx.run_ui(raw_input, |ui| {
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

            // ---- canvas ----
            egui::CentralPanel::default().frame(egui::Frame::NONE.fill(CANVAS_BG)).show(ui, |ui| {
                let rect = ui.available_rect_before_wrap();
                let w = ((rect.width() * ppp).round() as u32).max(1);
                let h = ((rect.height() * ppp).round() as u32).max(1);
                canvas_px = (w, h);
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
                canvas_overlays(ui, model, rect, ppp, gpu_ms, frame_ms, handle);
                wave_overlay(ui, model, rect, ppp, dt, handle);
            });
        });
        let egui::FullOutput { platform_output, mut textures_delta, shapes, pixels_per_point, viewport_output } = full_output;
        gfx.egui_state.handle_platform_output(&gfx.window, platform_output);

        // ---- GPU frame ----
        let grid = GridSpec::new(&model.view, canvas_px.0, canvas_px.1, model.ss, gfx.renderer.max_texels());
        let inputs = (model.m, model.scheme, model.layers.clone(), model.view, model.ss, canvas_px);
        let advance = self.last_inputs.as_ref() != Some(&inputs);
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
        let render_canvas = advance || !tween_off || !self.canvas_current;
        if render_canvas {
            gfx.renderer.render(&mut encoder, &FrameInput { grid, modulus: &model.modulus, palette: &model.palette, layers: &model.layers, advance, tween });
            gfx.renderer.flush_pass_params();
        }
        self.canvas_current = tween_off;

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
            gfx.renderer.render_blocking(&FrameInput { grid, modulus: &model.modulus, palette: &model.palette, layers: &model.layers, advance: false, tween: Tween::Off });
            self.tween_t0 = None;
            let (w, h, rgba) = gfx.renderer.read_output_rgba();
            let rows = model.view.visible_rows(h as f64).unwrap_or((0, 0));
            let dir = PathBuf::from(model.export_dir.trim());
            let path = export::export_path(&dir, model.m, rows, model.scheme, &model.layers);
            model.export_note = match export::write_png(&path, w, h, &rgba) {
                Ok(()) => format!("Saved {w}×{h} px to {}", path.display()),
                Err(e) => format!("Export failed: {e}"),
            };
            gfx.window.request_redraw();
        }

        self.frame_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let delay = viewport_output.get(&egui::ViewportId::ROOT).map(|v| v.repaint_delay).unwrap_or(std::time::Duration::MAX);
        self.continuous = delay.is_zero() || model.sweep.playing || self.tween_t0.is_some() || model.wave.is_animating();
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
    let old = ui.style().override_font_id.clone();
    ui.style_mut().override_font_id = Some(FontId::monospace(12.0));
    let r = ui.add(egui::DragValue::new(v).range(range).speed(speed));
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
    ui.label(egui::RichText::new("Pascal sieve").size(15.0).strong());
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
            let mut rows = model.fit_rows.round() as u32;
            let resp = mono_drag(ui, &mut rows, 2..=1_000_000_000u32, 1.0);
            if resp.changed() {
                model.fit_rows = rows.max(2) as f64;
                model.fit(canvas_px.0 as f64, canvas_px.1 as f64);
            }
            if ui.button("Fit").on_hover_text("Fit this many rows (F)").clicked() {
                model.fit(canvas_px.0 as f64, canvas_px.1 as f64);
            }
        });
        if let Some((a, b)) = model.view.visible_rows(canvas_px.1 as f64) {
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
        note(ui, "Plays one row as w(θ) = Σ (C(n,k) mod m)·cos((n−2k)θ + φ): the row under the pointer, else the last visible row. Click a cell to pin its row.");
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
        if ui.button("Save PNG").on_hover_text("The canvas at its native pixel size, from the same shader as the live view").clicked() {
            model.pending_export = true;
        }
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
    note(ui, "Keys: space play/pause · ←/→ step · PgUp/PgDn ±10 steps · ↑/↓ rows · shift = ×10 · F fit. Drag to pan, wheel to zoom, double-click to fit. Hover a cell to inspect it.");
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

/// Fixed-size monospace number field.
fn bar_drag<N: egui::emath::Numeric>(ui: &mut egui::Ui, w: f32, v: &mut N, range: std::ops::RangeInclusive<N>, speed: f64) -> egui::Response {
    let old = ui.style().override_font_id.clone();
    ui.style_mut().override_font_id = Some(FontId::monospace(12.0));
    let h = ui.spacing().interact_size.y;
    let r = ui.add_sized([w, h], egui::DragValue::new(v).range(range).speed(speed));
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
        ui.add(egui::Slider::new(&mut sw.speed_ms, 16..=1000).show_value(false));
        slot(ui, 56.0, |ui| {
            ui.label(egui::RichText::new(format!("{:>4} ms", sw.speed_ms)).font(FontId::monospace(12.0)));
        });
        ui.checkbox(&mut sw.looping, "loop").on_hover_text("Wrap at the range ends; off stops on the last frame.");
        bar_sep(ui);
        bar_label(ui, "tween");
        bar_drag(ui, 56.0, &mut sw.tween_ms, 0..=3000u32, 5.0).on_hover_text("Display-only transition between moduli: fields morph, plain residue coloring cross-dissolves. Exports, layer history and the inspector use the true frames. 0 = hard cuts.");
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
        Some(n) => model.step(n),
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
    let cw = (rect.width() * ppp) as f64;
    let ch = (rect.height() * ppp) as f64;
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
    model.hover = pointer.and_then(|(px, py)| model.view.cell_at(px, py));
    model.hover_row = pointer.and_then(|(_, py)| model.view.row_at(py));
    // A click on a cell pins its row for the wave overlay; a click on the pinned row (or
    // Esc) releases it. A double click (fit) toggles twice and leaves the pin as it was.
    if handle && response.clicked() {
        if let Some((n, _)) = model.hover {
            model.pinned_row = if model.pinned_row == Some(n) { None } else { Some(n) };
        }
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
        if esc {
            model.pinned_row = None;
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
            model.fit_rows = (model.fit_rows + dr as f64).clamp(2.0, 1.0e9);
            model.fit(cw, ch);
        }
        if fit {
            model.fit(cw, ch);
        }
    }
}

fn canvas_overlays(ui: &mut egui::Ui, model: &mut Model, rect: Rect, ppp: f32, gpu_ms: Option<f32>, frame_ms: f32, handle: bool) {
    let painter = ui.painter_at(rect);
    let cw = (rect.width() * ppp) as f64;
    let ch = (rect.height() * ppp) as f64;
    let to_pt = |x: f64, y: f64| Pos2::new(rect.min.x + (x / ppp as f64) as f32, rect.min.y + (y / ppp as f64) as f32);

    // Residue numbers inside cells, when they are big enough.
    if model.show_numbers && model.view.ppc >= NUMBERS_MIN_PX && model.view.visible_cells(cw, ch) <= NUMBERS_MAX_CELLS {
        if let Some((n_lo, n_hi)) = model.view.visible_rows(ch) {
            let font = FontId::monospace(((model.view.ppc * 0.34).min(model.view.ppc * ROW_H * 0.55) / ppp as f64) as f32);
            let x1 = model.view.x0 + cw / model.view.ppc;
            for n in n_lo..=n_hi {
                let k_lo = (model.view.x0 + n as f64 / 2.0 - 0.5).ceil().max(0.0) as i64;
                let k_hi = (x1 + n as f64 / 2.0 + 0.5).floor().min(n as f64) as i64;
                for k in k_lo..=k_hi {
                    let k = k as u32;
                    let Some(text) = model.cell_value(n, k) else { break };
                    let r = model.modulus.binom(n, k);
                    let (rgb, show) = match model.layers.preset() {
                        Some(Preset::Residue) => (model.palette.residue_color(r, model.m), r != 0),
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
        let pos = rect.min + Vec2::new(8.0, 8.0);
        let bg = Rect::from_min_size(pos, galley.size() + Vec2::new(18.0, 12.0));
        painter.rect_filled(bg, 0.0, Color32::from_black_alpha(200));
        painter.galley(pos + Vec2::new(9.0, 6.0), galley, Color32::from_rgb(221, 221, 221));
    }

    // −/+/Fit at the top right.
    let btn_h = 22.0;
    let mut x = rect.max.x - 8.0;
    for (label, action) in [("Fit", 2u8), ("+", 1), ("−", 0)] {
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

    // Status strip.
    let rows = model.view.visible_rows(ch);
    let cells = model.view.visible_cells(cw, ch);
    let mut parts = vec![format!("m = {}", model.m)];
    if let Some((a, b)) = rows {
        parts.push(format!("rows {}–{}", group(a as u64), group(b as u64)));
    }
    parts.push(format!("{} cells", group(cells)));
    if let Some(g) = gpu_ms {
        parts.push(format!("gpu {g:.1} ms"));
    }
    parts.push(format!("frame {frame_ms:.1} ms"));
    let strip = Rect::from_min_max(Pos2::new(rect.min.x, rect.max.y - STATUS_H as f32), rect.max);
    painter.rect_filled(strip, 0.0, Color32::from_black_alpha(150));
    painter.text(Pos2::new(strip.min.x + 8.0, strip.center().y), Align2::LEFT_CENTER, parts.join("  ·  "), FontId::monospace(11.0), Color32::from_rgb(187, 187, 187));
}

/// The "modular wave" overlay, drawn last so it sits on top of the canvas. The state is
/// advanced once per frame (egui may run the UI closure twice); drawing happens every pass.
fn wave_overlay(ui: &mut egui::Ui, model: &mut Model, rect: Rect, ppp: f32, dt: f32, handle: bool) {
    let cw = (rect.width() * ppp) as f64;
    let ch = (rect.height() * ppp) as f64;
    if handle {
        let row = if let Some(n) = model.pinned_row {
            n
        } else if let Some(n) = model.hover_row {
            n
        } else if model.over_wave {
            model.wave_row // the pointer is on the overlay: keep playing the same row
        } else {
            model.view.visible_rows(ch).map(|r| r.1).unwrap_or(model.wave_row)
        };
        model.wave_row = row;
        let (k_lo, k_hi) = model.view.visible_k(row, cw);
        model.wave.share_closed_form(&model.modulus);
        model.wave.set_source(row as u64, model.m as u64, (k_lo as u64, k_hi as u64));
        model.wave.set_pinned(model.pinned_row.is_some());
        model.wave.set_viewport(rect, ui.ctx().viewport_rect().size(), ppp);
        model.wave.update(dt);
    }
    let painter = ui.painter_at(rect);
    // ticks at the canvas edges mark the row the wave plays (solid when pinned)
    let fade = model.wave.fade();
    if fade > 0.0 {
        let (_, y, _, h) = model.view.cell_rect(model.wave_row, 0);
        let yc = rect.min.y + ((y + h / 2.0) / ppp as f64) as f32;
        if (rect.min.y..=rect.max.y - STATUS_H as f32).contains(&yc) {
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
