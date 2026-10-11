//! The tetra stack's UI: its panel section, the live preview tile on the 2D canvas and
//! the isolated 3D space (the stack, a HUD, and a thumbnail of the 2D view to go back).

use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};
use pascal::layers::Preset;
use pascal::math::Modulus;
use pascal::palette::Scheme;
use pascal::sweep::{ModMode, PrimeOrder, Seq};
use pascal::tetra::{describe, selected_moduli, stride_for, Shape, Camera, CubeId, Layout, SceneKey, Spacing, TetraConfig, BUDGETS, TETRA_MAX_M};
use pascal::tetra_gpu::{outline_vertices, DrawOptions, TetraRenderer, BACKGROUND};
use pascal::wave_overlay::Corner;

use super::{group, mono_drag, mono_drag_with, mono_line, note, Model};

/// Preview spin, rad/s.
const SPIN: f32 = 0.35;
/// Seconds for the tile's color to run from the apex to the base (and back).
const FILL_TIME: f32 = 0.35;

/// The tetra stack's state besides its settings.
#[derive(Default)]
pub struct TetraView {
    pub config: TetraConfig,
    /// The isolated 3D space is showing.
    pub open: bool,
    /// The 3D space's camera; None frames the stack on the next draw.
    pub camera: Option<Camera>,
    pub hover: Option<CubeId>,
    /// The pointer over the 3D space, in target pixels.
    pub pointer: Option<(f32, f32)>,
    /// The last built stack (what the HUD describes) and its cube counts.
    pub layout: Option<Layout>,
    pub counts: (u32, u32),
    /// Preview tile angle.
    pub spin: f32,
    /// Target size wanted this frame (px), and where the tile was drawn.
    pub target_px: (u32, u32),
    pub tile_rect: Option<Rect>,
    /// The pointer is over the tile; how far its color has run down (0 wire, 1 all).
    pub tile_hot: bool,
    fill: f32,
    /// Pointer and camera of the last pick, so an unchanged view is not picked again.
    last_pick: Option<((f32, f32), Camera)>,
    focus_last_frame: bool,
}

/// GPU-side caches of the app.
#[derive(Default)]
pub struct TetraCache {
    scene: Option<SceneKey>,
    colors: Option<(Scheme, u32)>,
    /// What the target last showed.
    drawn: Option<(DrawOptions, (u32, u32))>,
}

impl TetraView {
    pub fn new() -> TetraView {
        TetraView::default()
    }

    /// Whether anything shows the stack this frame (the space, or the tile).
    pub fn visible(&self) -> bool {
        self.open || (self.config.tile && self.tile_rect.is_some())
    }

    /// The camera the next draw uses: the space's, or the slowly turning preview.
    fn draw_camera(&mut self) -> Option<Camera> {
        let layout = self.layout.as_ref()?;
        if self.open {
            Some(*self.camera.get_or_insert_with(|| Camera::framing(layout.bounds())))
        } else {
            let mut c = Camera::framing(layout.bounds());
            c.yaw += self.spin;
            Some(c)
        }
    }

    /// A pick landed: keep it only if the pick it answers is still current (the pointer
    /// left, or the scene was rebuilt, while it was in flight).
    pub fn land_pick(&mut self, hit: Option<CubeId>) {
        if self.open && self.pointer.is_some() && self.last_pick.is_some() {
            self.hover = hit;
        }
    }

    pub fn advance(&mut self, dt: f32) {
        if !self.open && self.config.spin && self.visible() {
            self.spin = (self.spin + dt * SPIN) % std::f32::consts::TAU;
        }
        let want = if self.tile_hot && !self.open { 1.0 } else { 0.0 };
        let step = dt.max(0.0) / FILL_TIME;
        self.fill = if want > self.fill { (self.fill + step).min(1.0) } else { (self.fill - step).max(0.0) };
    }

    /// The tile's color is running down or back up.
    pub fn filling(&self) -> bool {
        // (the space draws everything filled: nothing to animate there)
        let want = if self.tile_hot { 1.0 } else { 0.0 };
        !self.open && self.visible() && self.fill != want
    }

    /// The fill height of the next draw: everything in the 3D space; in the tile, from
    /// the apex down as far as the hover has run (an ease-out, so it snaps in).
    fn fill_height(&self) -> f32 {
        if self.open {
            return f32::NEG_INFINITY;
        }
        let Some(l) = &self.layout else { return f32::INFINITY };
        let (Some(first), Some(last)) = (l.slices.first(), l.slices.last()) else { return f32::INFINITY };
        if self.fill <= 0.0 {
            return f32::INFINITY;
        }
        let top = -first.height + 1.0;
        let bottom = -last.height - 1.0;
        let e = 1.0 - (1.0 - self.fill).powi(3);
        top - e * (top - bottom)
    }

    /// The preview is turning: the app repaints it a few dozen times a second.
    pub fn spinning(&self) -> bool {
        !self.open && self.config.spin && self.visible()
    }
}

/// Rebuild the stack when its selection changed, recolor it when the palette or offset
/// did, then draw it into the target (and pick under the pointer in the 3D space).
/// Returns true when the target was recreated (re-register it with egui).
/// `dragging`: a widget is being dragged (a range field, say): the stack is rebuilt
/// when it lets go, not on every step.
pub fn render(model: &mut Model, cache: &mut TetraCache, r: &mut TetraRenderer, enc: &mut wgpu::CommandEncoder, dragging: bool) -> bool {
    if !model.tetra.visible() {
        return false;
    }
    let t = &mut model.tetra;
    let key = t.config.scene_key();
    // (no layout: the model was reset to a config the cache still holds)
    if (cache.scene != Some(key) && !dragging) || t.layout.is_none() {
        let mods = selected_moduli(&t.config.sweep);
        let layout = Layout::with_shape(&mods, t.config.spacing, t.config.shape, t.config.budget);
        let mds: Vec<Modulus> = mods.iter().map(|&m| Modulus::new(m)).collect();
        r.set_scene(&layout, &mds, &model.palette, model.style.offset);
        r.set_lines(&outline_vertices(&layout));
        t.counts = r.read_counts();
        // a new range is framed anew; a new spacing, shape or level of detail keeps the view
        let reframe = cache.scene.is_none_or(|k| (k.0, k.1, k.2, k.3, k.4, k.5) != (key.0, key.1, key.2, key.3, key.4, key.5));
        if reframe {
            t.camera = None;
        } else if let (Some(cam), Some(old)) = (t.camera.as_mut(), t.layout.as_ref()) {
            // same angle and relative zoom, around the new shape (packed is far shorter)
            let (was, now) = (Camera::framing(old.bounds()), Camera::framing(layout.bounds()));
            let pan = [cam.target[0] - was.target[0], cam.target[1] - was.target[1], cam.target[2] - was.target[2]];
            let scale = now.radius / was.radius;
            cam.target = [now.target[0] + pan[0] * scale, now.target[1] + pan[1] * scale, now.target[2] + pan[2] * scale];
            cam.dist *= now.dist / was.dist;
            cam.radius = now.radius;
        }
        t.layout = Some(layout);
        t.hover = None;
        t.last_pick = None;
        cache.scene = Some(key);
        cache.colors = Some((model.scheme, model.style.offset.to_bits()));
        cache.drawn = None;
    }
    let colors = (model.scheme, model.style.offset.to_bits());
    if cache.colors != Some(colors) {
        r.set_colors(&model.palette, model.style.offset);
        cache.colors = Some(colors);
        cache.drawn = None;
    }
    let recreated = r.ensure_target(t.target_px.0, t.target_px.1);
    let Some(camera) = t.draw_camera() else { return recreated };
    if t.open && t.pointer.is_none() {
        // off the stack: no hover, and a pick still in flight is dropped when it lands
        t.hover = None;
        t.last_pick = None;
    }
    let mask = model.layers.preset() == Some(Preset::Mask);
    let hover = if t.open { t.hover } else { None };
    let opts = DrawOptions { camera, fill: t.fill_height(), hover, brush: t.config.brush, color_slice: t.config.color_slice, ghosts: t.config.ghosts, mask };
    // an unchanged picture is not drawn again (hover, egui repaints)
    let drawn = (opts, t.target_px);
    if recreated || cache.drawn != Some(drawn) {
        r.draw(enc, &opts);
        cache.drawn = Some(drawn);
    }
    if t.open {
        if let Some(p) = t.pointer {
            if t.last_pick != Some((p, camera)) && r.encode_pick(enc, &camera, p.0, p.1) {
                t.last_pick = Some((p, camera));
            }
        }
    }
    recreated
}

// ---------------------------------------------------------------------------------
// Panel section
// ---------------------------------------------------------------------------------

pub fn panel(ui: &mut egui::Ui, model: &mut Model) {
    let t = &mut model.tetra;
    let c = &mut t.config;
    ui.horizontal(|ui| {
        if ui.button("Open 3D view").on_hover_text("The stack alone in a 3D space (or click the preview tile); Esc or the 2D thumbnail returns").clicked() {
            t.open = true;
        }
        ui.checkbox(&mut c.tile, "preview tile");
        ui.checkbox(&mut c.spin, "spin");
    });
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("moduli").color(ui.visuals().weak_text_color()));
        mono_drag_with(ui, &mut c.sweep.start, 2..=TETRA_MAX_M, 1.0, false).on_hover_text("First modulus (the apex slice)");
        ui.label(egui::RichText::new("to").color(ui.visuals().weak_text_color()));
        mono_drag_with(ui, &mut c.sweep.end, 2..=TETRA_MAX_M, 1.0, false).on_hover_text(format!("Last modulus (the base slice), at most {TETRA_MAX_M}"));
    });
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("visit").color(ui.visuals().weak_text_color()));
        egui::ComboBox::from_id_salt("tetra mode").width(96.0).selected_text(c.sweep.mode.label()).show_ui(ui, |ui| {
            for m in ModMode::ALL {
                ui.selectable_value(&mut c.sweep.mode, m, m.label()).on_hover_text(m.note());
            }
        });
        match c.sweep.mode {
            ModMode::Prime => {
                egui::ComboBox::from_id_salt("tetra prime order").width(120.0).selected_text(c.sweep.prime_order.label()).show_ui(ui, |ui| {
                    for o in PrimeOrder::ALL {
                        ui.selectable_value(&mut c.sweep.prime_order, o, o.label());
                    }
                });
            }
            ModMode::Interval => {
                egui::ComboBox::from_id_salt("tetra seq").width(110.0).selected_text(c.sweep.seq.label()).show_ui(ui, |ui| {
                    for q in Seq::ALL {
                        ui.selectable_value(&mut c.sweep.seq, q, q.label());
                    }
                });
                if c.sweep.seq == Seq::Fixed {
                    mono_drag_with(ui, &mut c.sweep.step, 1..=TETRA_MAX_M, 0.2, false).on_hover_text("Step from the first modulus");
                }
            }
            _ => {}
        }
    });
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("shape").color(ui.visuals().weak_text_color()));
        egui::ComboBox::from_id_salt("tetra shape").width(150.0).selected_text(c.shape.label()).show_ui(ui, |ui| {
            for sh in Shape::ALL {
                ui.selectable_value(&mut c.shape, sh, sh.label()).on_hover_text(sh.note());
            }
        }).response.on_hover_text(c.shape.note());
    });
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("spacing").color(ui.visuals().weak_text_color()));
        ui.selectable_value(&mut c.spacing, Spacing::True, "true").on_hover_text("Slice height = m: skipped moduli leave gaps (dashed, with the gap size) and the edges stay straight (G)");
        ui.selectable_value(&mut c.spacing, Spacing::Packed, "packed").on_hover_text("Slice height = its index: no gaps, but the sides curve because slice size no longer tracks height (G)");
    });
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("highlight").color(ui.visuals().weak_text_color()));
        ui.add_enabled_ui(!c.color_slice, |ui| mono_drag(ui, &mut c.brush, 0.0..=64.0, 0.05).on_hover_text("In the 3D space, cubes within this many cells of the hovered cube light up"));
        ui.checkbox(&mut c.color_slice, "whole slice").on_hover_text("Light up the whole hovered slice instead");
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut c.ghosts, "ghost zeros").on_hover_text("Zero cells as faint ghost cubes: the tunnels glow through the wire");
        ui.label(egui::RichText::new("cubes ≤").color(ui.visuals().weak_text_color()));
        egui::ComboBox::from_id_salt("tetra budget").width(70.0).selected_text(short(c.budget)).show_ui(ui, |ui| {
            for b in BUDGETS {
                ui.selectable_value(&mut c.budget, b, short(b));
            }
        }).response.on_hover_text("Most cubes drawn. A larger selection draws every s-th row and cell (the HUD says so) instead of freezing");
    });
    let mods = selected_moduli(&c.sweep);
    let text = match (mods.first(), mods.last()) {
        (Some(a), Some(b)) => {
            let s = stride_for(&mods, c.budget);
            let lod = if s > 1 { format!(" · stride {s}") } else { String::new() };
            format!("{} slices, m {a}–{b}{lod}", mods.len())
        }
        _ => "no moduli in the range".into(),
    };
    mono_line(ui, &text);
    note(ui, "One slice per modulus m: rows 0..m−1 of the triangle mod m, stacked downward, so the solid is a tetrahedron. Zero cells are left out, and their voids run through it as tunnels. Hover a cube to highlight the cells around it.");
}

fn short(n: u32) -> String {
    if n >= 1_000_000 {
        format!("{}M", n as f64 / 1e6)
    } else {
        format!("{}K", n / 1000)
    }
}

// ---------------------------------------------------------------------------------
// The preview tile on the 2D canvas
// ---------------------------------------------------------------------------------

/// Where the tile goes: a bottom corner the wave overlay does not use.
fn tile_rect(model: &Model, rect: Rect) -> Rect {
    let side = (rect.width().min(rect.height()) * 0.22).clamp(120.0, 220.0);
    let margin = 16.0;
    let left = model.wave.config.corner != Corner::BottomLeft;
    let x = if left { rect.min.x + margin } else { rect.max.x - margin - side };
    Rect::from_min_size(Pos2::new(x, rect.max.y - margin - side), Vec2::splat(side))
}

/// The live preview tile, styled like the wave overlay's box. Click to open the space.
pub fn tile(ui: &mut egui::Ui, model: &mut Model, rect: Rect, ppp: f32, tex: Option<egui::TextureId>) {
    if !model.tetra.config.tile || rect.width() < 300.0 || rect.height() < 240.0 {
        model.tetra.tile_rect = None;
        model.tetra.tile_hot = false;
        return;
    }
    let b = tile_rect(model, rect);
    model.tetra.tile_rect = Some(b);
    let img = b.shrink(6.0);
    model.tetra.target_px = (((img.width() * ppp).round() as u32).max(1), ((img.height() * ppp).round() as u32).max(1));
    let resp = ui.interact(b, ui.id().with("tetra tile"), Sense::click()).on_hover_text("Open the tetra stack in its own 3D space");
    let painter = ui.painter_at(rect);
    let hot = resp.hovered();
    model.tetra.tile_hot = hot;
    painter.rect_filled(b, CornerRadius::same(8), Color32::from_rgba_unmultiplied(0, 0, 0, 158));
    if let Some(id) = tex {
        painter.image(id, img, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
    }
    painter.rect_stroke(b, CornerRadius::same(8), Stroke::new(1.0, Color32::from_white_alpha(if hot { 90 } else { 38 })), StrokeKind::Inside);
    painter.text(img.min + Vec2::new(4.0, 3.0), Align2::LEFT_TOP, "TETRA STACK", FontId::monospace(10.0), Color32::from_rgba_unmultiplied(221, 221, 221, 220));
    let sub = match model.tetra.layout.as_ref().and_then(|l| Some((l.slices.first()?.m, l.slices.last()?.m))) {
        Some((a, z)) => format!("m {a}–{z} · click to open"),
        None => "click to open".into(),
    };
    painter.text(Pos2::new(img.min.x + 4.0, img.max.y - 3.0), Align2::LEFT_BOTTOM, sub, FontId::monospace(9.0), Color32::from_rgba_unmultiplied(187, 187, 187, 190));
    if resp.clicked() {
        model.tetra.open = true;
        // the space shows everything colored; the tile comes back as wire
        model.tetra.tile_hot = false;
        model.tetra.fill = 0.0;
    }
    if hot {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
}

// ---------------------------------------------------------------------------------
// The isolated 3D space
// ---------------------------------------------------------------------------------

pub fn space(ui: &mut egui::Ui, model: &mut Model, ppp: f32, tex: Option<egui::TextureId>, canvas_tex: Option<egui::TextureId>, canvas_px: (u32, u32), handle: bool) {
    let bg = Color32::from_rgb((BACKGROUND[0] * 255.0).round() as u8, (BACKGROUND[1] * 255.0).round() as u8, (BACKGROUND[2] * 255.0).round() as u8);
    egui::CentralPanel::default().frame(egui::Frame::NONE.fill(bg)).show(ui, |ui| {
        let rect = ui.available_rect_before_wrap();
        let (w, h) = (((rect.width() * ppp).round() as u32).max(1), ((rect.height() * ppp).round() as u32).max(1));
        model.tetra.target_px = (w, h);
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        if let Some(id) = tex {
            painter.image(id, rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }

        // the way back: a thumbnail of the 2D view
        let aspect = canvas_px.0.max(1) as f32 / canvas_px.1.max(1) as f32;
        // the canvas's aspect, at most 240 wide and 200 tall
        let tw = (rect.width() * 0.18).clamp(120.0, 240.0).min(200.0 * aspect);
        let th = tw / aspect;
        let thumb = Rect::from_min_size(Pos2::new(rect.max.x - 16.0 - tw, rect.max.y - 16.0 - th), Vec2::new(tw, th));
        let tresp = ui.interact(thumb, ui.id().with("tetra back"), Sense::click()).on_hover_text("Back to the 2D view (Esc)");
        painter.rect_filled(thumb.expand(4.0), CornerRadius::same(8), Color32::from_rgba_unmultiplied(0, 0, 0, 158));
        if let Some(id) = canvas_tex {
            painter.image(id, thumb, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }
        painter.rect_stroke(thumb.expand(4.0), CornerRadius::same(8), Stroke::new(1.0, Color32::from_white_alpha(if tresp.hovered() { 110 } else { 40 })), StrokeKind::Inside);
        painter.text(thumb.min + Vec2::new(4.0, 3.0), Align2::LEFT_TOP, "2D · Esc", FontId::monospace(10.0), Color32::from_rgba_unmultiplied(221, 221, 221, 220));
        if tresp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }

        if handle && ui.input(|i| i.key_pressed(egui::Key::F11)) {
            model.fullscreen = !model.fullscreen;
        }
        let t = &mut model.tetra;
        if handle {
            if tresp.clicked() {
                t.open = false;
            }
            if let Some(cam) = t.camera.as_mut() {
                let shift = ui.input(|i| i.modifiers.shift);
                let d = resp.drag_delta();
                if d != Vec2::ZERO {
                    if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) || (shift && resp.dragged_by(egui::PointerButton::Primary)) {
                        cam.pan(d.x, d.y, rect.height());
                    } else if resp.dragged_by(egui::PointerButton::Primary) {
                        cam.orbit(d.x, d.y);
                    }
                }
                if resp.hovered() {
                    let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
                    let s = scroll + if zoom != 1.0 { zoom.ln() * 600.0 } else { 0.0 };
                    if s != 0.0 {
                        cam.zoom(s);
                    }
                }
            }
            if resp.double_clicked() {
                t.camera = None;
            }
            if !ui.ctx().egui_wants_keyboard_input() {
                let (esc, g) = ui.input(|i| (i.key_pressed(egui::Key::Escape), i.key_pressed(egui::Key::G)));
                if esc && !t.focus_last_frame {
                    t.open = false;
                }
                if g {
                    t.config.spacing = t.config.spacing.toggled();
                }
            }
            t.focus_last_frame = ui.ctx().memory(|m| m.focused().is_some()) || ui.ctx().any_popup_open();
            let p = resp.hover_pos().filter(|p| resp.hovered() && !resp.dragged() && !thumb.expand(4.0).contains(*p));
            t.pointer = p.map(|p| ((p.x - rect.min.x) * ppp, (p.y - rect.min.y) * ppp));
        }

        // gap labels between slices more than one modulus apart (true spacing)
        if let (Some(l), Some(cam)) = (&t.layout, t.camera) {
            let mut last_y = f32::NEG_INFINITY;
            let mut drawn = 0;
            for (a, b) in l.gaps() {
                // between the two slices' apex corners
                let (pa, pb) = (l.corners(a, l.height_of(a))[0], l.corners(b, l.height_of(b))[0]);
                let mid = [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0, (pa[2] + pb[2]) / 2.0];
                let Some((x, y)) = cam.project(w, h, mid) else { continue };
                let p = Pos2::new(rect.min.x + x / ppp, rect.min.y + y / ppp);
                if !rect.contains(p) || (p.y - last_y).abs() < 12.0 || drawn > 400 {
                    continue;
                }
                last_y = p.y;
                drawn += 1;
                painter.text(p - Vec2::new(8.0, 0.0), Align2::RIGHT_CENTER, format!("+{}", b - a), FontId::monospace(10.0), Color32::from_rgba_unmultiplied(120, 220, 150, 170));
            }
        }

        // HUD
        let mut lines: Vec<String> = Vec::new();
        if let Some(l) = &t.layout {
            match (l.slices.first(), l.slices.last()) {
                (Some(a), Some(z)) => {
                    lines.push(format!("TETRA STACK · m {}–{} · {} slices · {}", a.m, z.m, l.slices.len(), t.config.sweep.mode.label()));
                    lines.push(format!("{} cubes, {} zeros{}", group(t.counts.0 as u64), group(t.counts.1 as u64), if t.config.ghosts { " (ghosts)" } else { " (hidden)" }));
                    if l.stride > 1 {
                        lines.push(format!("LOD: stride {} — 1 in {} rows and cells, {} of {} cells (raise the cube budget for more)", l.stride, l.stride, group(l.total as u64), group(l.full_total)));
                    }
                }
                _ => lines.push("TETRA STACK · no moduli in the range".into()),
            }
            lines.push(format!("shape: {}", l.shape.label()));
            lines.push(match l.spacing {
                Spacing::True => "true spacing: height follows m; gaps are skipped moduli (G)".into(),
                Spacing::Packed => "packed: height = slice index, so the sides curve: slice size no longer tracks height (G)".into(),
            });
            let lens = match model.layers.preset() {
                Some(Preset::Mask) => "mask",
                Some(Preset::Residue) => "residue",
                _ => "residue (this layer setup has no 3D form)",
            };
            lines.push(format!("lens: {lens} · palette {}", model.scheme.label()));
            if let Some(c) = t.hover {
                if let Some(s) = l.slices.get(c.slice as usize) {
                    lines.push(describe(s.m, &c));
                }
            }
        }
        lines.push("drag orbit · right / shift-drag pan · wheel zoom · double-click reset · Esc back".into());
        let galleys: Vec<_> = lines.iter().enumerate().map(|(i, s)| {
            let col = if i == 0 { Color32::from_rgb(160, 240, 180) } else { Color32::from_rgb(187, 200, 190) };
            painter.layout_no_wrap(s.clone(), FontId::monospace(11.0), col)
        }).collect();
        let hud_w = galleys.iter().map(|g| g.size().x).fold(0.0, f32::max);
        let hud_h: f32 = galleys.iter().map(|g| g.size().y + 2.0).sum();
        let hud = Rect::from_min_size(rect.min + Vec2::new(12.0, 12.0), Vec2::new(hud_w + 16.0, hud_h + 12.0));
        painter.rect_filled(hud, CornerRadius::same(6), Color32::from_rgba_unmultiplied(0, 0, 0, 150));
        let mut y = hud.min.y + 6.0;
        for g in galleys {
            let gh = g.size().y;
            painter.galley(Pos2::new(hud.min.x + 8.0, y), g, Color32::WHITE);
            y += gh + 2.0;
        }
    });
}
