//! The desktop app: a winit window, an egui control panel on the left and the GPU
//! canvas filling the rest. All math and rendering live in the `pascal` library; this
//! file is the UI, the camera interaction and the export plumbing.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Vec2};
use pascal::export;
use pascal::gpu::{FrameInput, Gpu, Renderer};
use pascal::layers::{AvgMode, BlendMode, FieldColorMode, Kind, LayerId, LayerState, Preset, REGISTRY};
use pascal::math::{self, Modulus, MOD_CAP};
use pascal::palette::{Palette, Scheme};
use pascal::view::{GridSpec, View, ROW_H, STATUS_H};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

const PANEL_W: f32 = 300.0;
const CANVAS_BG: Color32 = Color32::from_rgb(10, 10, 10);
/// Cells at least this wide (px) get their residue printed inside.
const NUMBERS_MIN_PX: f64 = 18.0;
const NUMBERS_MAX_CELLS: u64 = 9000;

/// Everything the user can change.
struct Model {
    m: u32,
    modulus: Modulus,
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
}

impl Model {
    fn new() -> Self {
        let m = 7;
        Model {
            m,
            modulus: Modulus::new(m),
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
        }
    }

    fn set_m(&mut self, m: u32) {
        let m = m.clamp(2, MOD_CAP);
        if m != self.m {
            self.m = m;
            self.modulus = Modulus::new(m);
        }
        self.m_text = m.to_string();
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

struct App {
    gfx: Option<Gfx>,
    model: Model,
    last_inputs: Option<(u32, Scheme, LayerState, View, u32, (u32, u32))>,
    frame_ms: f32,
    /// Redraw counter: egui may run the UI closure more than once per frame, but
    /// input (keys, wheel, drag, clicks) must move the camera only once.
    frame_nr: u64,
}

const USAGE: &str = "pascal-triangle-explorer [options]
  --m N              modulus (default 7)
  --rows R           rows to fit (default 128)
  --palette NAME     spectral | viridis | magma | inferno | turbo | cividis | coolwarm | mono
  --mode MODE        residue | mask | valuation
  --layers a,b,c     active layers: value kummer lucas crt dlog pdigit density average bandpass afterimage
  --field MODE       math | palette (field color mode)
  --export FILE.png  render once without a window and write a PNG
  --size WxH         canvas size for --export (default 1600x1000)";

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
                    match REGISTRY.iter().find(|d| d.label.to_lowercase().replace(' ', "") == name) {
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
            "--export" => export = Some(PathBuf::from(next())),
            "--size" => {
                let v = next();
                if let Some((w, h)) = v.split_once('x') {
                    size = (w.parse().unwrap_or(1600), h.parse().unwrap_or(1000));
                }
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
    let mut app = App { gfx: None, model, last_inputs: None, frame_ms: 0.0, frame_nr: 0 };
    event_loop.run_app(&mut app).expect("run");
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
    renderer.render_blocking(&FrameInput { grid, modulus: &model.modulus, palette: &model.palette, layers: &model.layers, advance: true });
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
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(window.clone())));
        let surface = instance.create_surface(window.clone()).expect("surface");
        let gpu = pollster::block_on(Gpu::new_async(instance, Some(&surface))).unwrap_or_else(|e| panic!("{e}"));
        log::info!("adapter: {} ({:?})", gpu.info.name, gpu.info.backend);
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
        if response.repaint {
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
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
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
                log::error!("surface validation error");
                return;
            }
        };
        let ppp = egui_winit::pixels_per_point(&gfx.egui_ctx, &gfx.window);
        let raw_input = gfx.egui_state.take_egui_input(&gfx.window);
        let gpu_ms = gfx.renderer.gpu_ms();
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
                        egui_renderer.update_egui_texture_from_wgpu_texture(&gpu.device, &renderer.out_view_sampled, wgpu::FilterMode::Nearest, id);
                        id
                    }
                    None => {
                        let id = egui_renderer.register_native_texture(&gpu.device, &renderer.out_view_sampled, wgpu::FilterMode::Nearest);
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
            });
        });
        let egui::FullOutput { platform_output, mut textures_delta, shapes, pixels_per_point, viewport_output } = full_output;
        gfx.egui_state.handle_platform_output(&gfx.window, platform_output);

        // ---- GPU frame ----
        let grid = GridSpec::new(&model.view, canvas_px.0, canvas_px.1, model.ss, gfx.renderer.max_texels());
        let inputs = (model.m, model.scheme, model.layers.clone(), model.view, model.ss, canvas_px);
        let advance = self.last_inputs.as_ref() != Some(&inputs);
        self.last_inputs = Some(inputs);
        let mut encoder = gfx.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        gfx.renderer.render(&mut encoder, &FrameInput { grid, modulus: &model.modulus, palette: &model.palette, layers: &model.layers, advance });
        gfx.renderer.flush_pass_params();

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
        gfx.renderer.begin_timing_readback();
        gfx.window.pre_present_notify();
        gfx.gpu.queue.present(frame);
        for id in &textures_delta.free {
            gfx.egui_renderer.free_texture(id);
        }
        textures_delta.clear();

        if model.pending_export {
            model.pending_export = false;
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
        let repaint = viewport_output.get(&egui::ViewportId::ROOT).map(|v| v.repaint_delay.is_zero()).unwrap_or(false);
        if repaint {
            gfx.window.request_redraw();
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
                match model.m_text.trim().parse::<u32>() {
                    Ok(v) => model.set_m(v),
                    Err(_) => model.m_text = model.m.to_string(),
                }
            }
            if ui.add(egui::Button::new("◀").small()).on_hover_text("One step down (←)").clicked() {
                model.set_m(model.m.saturating_sub(1));
            }
            if ui.add(egui::Button::new("▶").small()).on_hover_text("One step up (→)").clicked() {
                model.set_m(model.m + 1);
            }
        });
        let mut m = model.m;
        ui.spacing_mut().slider_width = ui.available_width() - 8.0;
        if ui.add(egui::Slider::new(&mut m, 2..=MOD_CAP).logarithmic(true).show_value(false)).changed() {
            model.set_m(m);
        }
        let fs = &model.modulus.factors.iter().map(|t| t.f).collect::<Vec<_>>();
        let info = if model.modulus.is_prime() { format!("{} is prime", model.m) } else { format!("{} = {}", model.m, math::factor_string(fs)) };
        mono_note(ui, &info);
        if !model.modulus.is_prime() {
            note(ui, "Zeros are where every prime-power factor divides C(n,k); the CRT layer shows which ones do.");
        }
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
            mono_note(ui, &format!("rows {}–{} in view · {:.3} px per cell", group(a as u64), group(b as u64), model.view.ppc));
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
                            let (lo, hi) = if p.key == "avgAmount" && model.layers.avg_mode == AvgMode::Spatial { (1, 12) } else { (p.min, p.max) };
                            if mono_drag(ui, &mut v, lo..=hi, 0.1).changed() {
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
        mono_note(ui, &format!("{} × {} px", canvas_px.0, canvas_px.1));
        if !model.export_note.is_empty() {
            mono_note(ui, &model.export_note);
        }
    });

    ui.separator();
    note(ui, "Keys: ←/→ step m · PgUp/PgDn ±10 · ↑/↓ rows · shift = ×10 · F fit. Drag to pan, wheel to zoom, double-click to fit. Hover a cell to inspect it.");
}

/// Subscript digits: 7 → ₇.
fn sub(n: u32) -> String {
    const SUB: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
    n.to_string().chars().map(|c| SUB[c as usize - '0' as usize]).collect()
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
        if i > 0 && (s.len() - i) % 3 == 0 {
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
    model.hover = if response.hovered() && !response.dragged() {
        response.hover_pos().and_then(|p| {
            let (px, py) = to_px(p);
            model.view.cell_at(px, py)
        })
    } else {
        None
    };

    // Keyboard, when no text field has focus.
    if handle && !ui.ctx().egui_wants_keyboard_input() {
        let (shift, right, left, pgup, pgdn, up, down, fit) = ui.input(|i| {
            (
                i.modifiers.shift,
                i.num_presses(egui::Key::ArrowRight),
                i.num_presses(egui::Key::ArrowLeft),
                i.num_presses(egui::Key::PageUp),
                i.num_presses(egui::Key::PageDown),
                i.num_presses(egui::Key::ArrowUp),
                i.num_presses(egui::Key::ArrowDown),
                i.key_pressed(egui::Key::F),
            )
        });
        let big: i64 = if shift { 10 } else { 1 };
        let dm = (right as i64 - left as i64) * big + (pgup as i64 - pgdn as i64) * 10 * big;
        if dm != 0 {
            model.set_m((model.m as i64 + dm).clamp(2, MOD_CAP as i64) as u32);
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
