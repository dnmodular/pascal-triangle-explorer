//! The tetra stack's UI: a small entry in the main panel, the live preview tile on the
//! 2D canvas, and the isolated 3D space with its own options panel, a HUD, and a
//! thumbnail of the 2D view to go back. The space shows the stack, its front face laid
//! flat, or the diamond (the face with the base slice hinged down beside it).

use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};
use pascal::layers::Preset;
use pascal::math::Modulus;
use pascal::palette::Scheme;
use pascal::sweep::{ModMode, PrimeOrder, Seq};
use pascal::tetra::{describe, face_row_alternates, face_row_zeros, selected_moduli, view_stride, Camera, CubeId, Layout, SceneKey, Shape, SliceKind, Spacing, TetraConfig, View, BUDGETS, FOLD, TETRA_MAX_M};
use pascal::tetra_gpu::{outline_vertices, DrawOptions, TetraRenderer, BACKGROUND};
use pascal::wave_overlay::Corner;

use super::{group, mono_drag, mono_drag_with, mono_line, note, Model};

/// Preview spin, rad/s.
const SPIN: f32 = 0.35;
/// Seconds for the tile's color to run from the apex to the base (and back).
const FILL_TIME: f32 = 0.35;
/// Seconds for the diamond's hinge to swing open.
const UNFOLD_TIME: f32 = 1.1;
/// Width of the space's options panel (points).
const OPTIONS_W: f32 = 270.0;

/// The tetra stack's state besides its settings.
#[derive(Default)]
pub struct TetraView {
    pub config: TetraConfig,
    /// The isolated 3D space is showing.
    pub open: bool,
    /// The 3D space's camera; None frames the view on the next draw.
    pub camera: Option<Camera>,
    pub hover: Option<CubeId>,
    /// The pointer over the 3D space, in target pixels.
    pub pointer: Option<(f32, f32)>,
    /// The last built layout (what the HUD describes) and its cube counts.
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
    /// How far the diamond's hinge has swung open (0 folded as in the tetrahedron, 1 flat).
    unfold: f32,
    /// The hovered face row's facts: (m, alternates ±1, zeros).
    row_stats: Option<(u32, bool, u32)>,
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

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl TetraView {
    pub fn new() -> TetraView {
        TetraView { unfold: 1.0, ..TetraView::default() }
    }

    /// Whether anything shows the stack this frame (the space, or the tile).
    pub fn visible(&self) -> bool {
        self.open || (self.config.tile && self.tile_rect.is_some())
    }

    /// What is drawn: the space's chosen view, or the stack in the tile.
    pub fn view(&self) -> View {
        if self.open {
            self.config.view
        } else {
            View::Stack
        }
    }

    /// Show `v` in the space (the diamond's hinge swings open again).
    pub fn set_view(&mut self, v: View) {
        if v == View::Diamond && self.config.view != View::Diamond {
            self.unfold = 0.0;
        }
        self.config.view = v;
    }

    /// Swing the diamond's hinge open again from the folded tetrahedron.
    pub fn replay_unfold(&mut self) {
        self.unfold = 0.0;
    }

    /// The camera the next draw uses: the space's, or the slowly turning preview.
    fn draw_camera(&mut self) -> Option<Camera> {
        let layout = self.layout.as_ref()?;
        if self.open {
            let aspect = self.target_px.0.max(1) as f32 / self.target_px.1.max(1) as f32;
            Some(*self.camera.get_or_insert_with(|| if layout.view.flat() { Camera::flat(layout.at_rest().bounds(), aspect) } else { Camera::framing(layout.bounds()) }))
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
        let dt = dt.max(0.0);
        if !self.open && self.config.spin && self.visible() {
            self.spin = (self.spin + dt * SPIN) % std::f32::consts::TAU;
        }
        let want = if self.tile_hot && !self.open { 1.0 } else { 0.0 };
        let step = dt / FILL_TIME;
        self.fill = if want > self.fill { (self.fill + step).min(1.0) } else { (self.fill - step).max(0.0) };
        if self.open && self.config.view == View::Diamond {
            self.unfold = (self.unfold + dt / UNFOLD_TIME).min(1.0);
        }
    }

    /// Something animates on its own: the tile's color running, or the hinge swinging.
    pub fn filling(&self) -> bool {
        // (the space draws everything filled: nothing to run there)
        let want = if self.tile_hot { 1.0 } else { 0.0 };
        let tile = !self.open && self.visible() && self.fill != want;
        let hinge = self.open && self.config.view == View::Diamond && self.unfold < 1.0;
        tile || hinge
    }

    /// The hinge angle of the diamond now: folded at the start, flat at the end.
    fn fold_angle(&self) -> f32 {
        FOLD * (1.0 - smoothstep(self.unfold))
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
        let top = -first.height() + 1.0;
        let bottom = -last.height() - 1.0;
        let e = 1.0 - (1.0 - self.fill).powi(3);
        top - e * (top - bottom)
    }

    /// The preview is turning: the app repaints it a few dozen times a second.
    pub fn spinning(&self) -> bool {
        !self.open && self.config.spin && self.visible()
    }
}

/// Rebuild the view when its selection changed, recolor it when the palette or offset
/// did, swing the diamond's hinge, then draw into the target (and pick under the
/// pointer in the 3D space). Returns true when the target was recreated (re-register it
/// with egui). `dragging`: a widget is being dragged (a range field, say): the scene is
/// rebuilt when it lets go, not on every step.
pub fn render(model: &mut Model, cache: &mut TetraCache, r: &mut TetraRenderer, enc: &mut wgpu::CommandEncoder, dragging: bool) -> bool {
    if !model.tetra.visible() {
        return false;
    }
    let t = &mut model.tetra;
    let view = t.view();
    let key = t.config.scene_key(view);
    // (no layout: the model was reset to a config the cache still holds)
    if (cache.scene != Some(key) && !dragging) || t.layout.is_none() {
        let mods = selected_moduli(&t.config.sweep);
        let mut layout = Layout::build(view, &mods, t.config.spacing, t.config.shape, t.config.budget);
        if view == View::Diamond {
            layout.set_fold(t.fold_angle());
        }
        let mds: Vec<Modulus> = layout.slices.iter().map(|s| Modulus::new(s.m)).collect();
        r.set_scene(&layout, &mds, &model.palette, model.style.offset);
        r.set_lines(&outline_vertices(&layout));
        t.counts = r.read_counts();
        let same_view = t.layout.as_ref().is_some_and(|l| l.view == view);
        // a new range or view is framed anew; in the stack a new spacing, shape or level
        // of detail keeps the angle and the relative zoom
        let reframe = !same_view || view.flat() || cache.scene.is_none_or(|k| (k.0, k.1, k.2, k.3, k.4, k.5) != (key.0, key.1, key.2, key.3, key.4, key.5));
        if reframe {
            t.camera = None;
        } else if let (Some(cam), Some(old)) = (t.camera.as_mut(), t.layout.as_ref()) {
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
    // the diamond's hinge: only the base slice's placement changes
    let want = t.fold_angle();
    if let Some(l) = t.layout.as_mut() {
        if l.view == View::Diamond && l.fold != want {
            l.set_fold(want);
            r.set_placement(l);
            t.last_pick = None;
            cache.drawn = None;
        }
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
    let opts = DrawOptions { camera, fill: t.fill_height(), hover, brush: t.config.brush, color_slice: t.config.color_slice, ghosts: t.config.ghosts, mask, flat: t.open && view.flat() };
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
// The main panel's entry: the way in, and the tile on the 2D canvas
// ---------------------------------------------------------------------------------

pub fn panel(ui: &mut egui::Ui, model: &mut Model) {
    let t = &mut model.tetra;
    ui.horizontal(|ui| {
        if ui.button("Open 3D view").on_hover_text("The tetra stack in its own space (or click the preview tile); Esc or the 2D thumbnail returns").clicked() {
            t.open = true;
        }
        ui.checkbox(&mut t.config.tile, "preview tile");
        ui.checkbox(&mut t.config.spin, "spin");
    });
    note(ui, "One slice per modulus m, rows 0..m−1 of the triangle mod m, stacked into a tetrahedron. Its options (range, shape, the front face and the diamond) are in the 3D view.");
}

// ---------------------------------------------------------------------------------
// The 3D space's options panel
// ---------------------------------------------------------------------------------

fn options(ui: &mut egui::Ui, t: &mut TetraView) {
    ui.spacing_mut().item_spacing = Vec2::new(6.0, 5.0);
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Tetra stack").size(15.0).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("hide ◀").on_hover_text("Hide the options (O)").clicked() {
                t.config.options = false;
            }
        });
    });
    ui.separator();
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("view").color(ui.visuals().weak_text_color()));
        for v in View::ALL {
            if ui.selectable_label(t.config.view == v, v.label()).on_hover_text(v.note()).clicked() {
                t.set_view(v);
            }
        }
    });
    note(ui, t.config.view.note());
    if t.config.view == View::Diamond && ui.small_button("unfold again (U)").on_hover_text("Fold the base slice back to where it sits in the tetrahedron and swing it open").clicked() {
        t.replay_unfold();
    }
    ui.separator();
    let c = &mut t.config;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("moduli").color(ui.visuals().weak_text_color()));
        mono_drag_with(ui, &mut c.sweep.start, 2..=TETRA_MAX_M, 1.0, false).on_hover_text("First modulus (the apex)");
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
                egui::ComboBox::from_id_salt("tetra prime order").width(110.0).selected_text(c.sweep.prime_order.label()).show_ui(ui, |ui| {
                    for o in PrimeOrder::ALL {
                        ui.selectable_value(&mut c.sweep.prime_order, o, o.label());
                    }
                });
            }
            ModMode::Interval => {
                egui::ComboBox::from_id_salt("tetra seq").width(100.0).selected_text(c.sweep.seq.label()).show_ui(ui, |ui| {
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
    ui.add_enabled_ui(c.view == View::Stack, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("shape").color(ui.visuals().weak_text_color()));
            egui::ComboBox::from_id_salt("tetra shape").width(150.0).selected_text(c.shape.label()).show_ui(ui, |ui| {
                for sh in Shape::ALL {
                    ui.selectable_value(&mut c.shape, sh, sh.label()).on_hover_text(sh.note());
                }
            }).response.on_hover_text(c.shape.note()).on_disabled_hover_text("The face and the diamond are flat: the shape is the stack's");
        });
    });
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("spacing").color(ui.visuals().weak_text_color()));
        ui.selectable_value(&mut c.spacing, Spacing::True, "true").on_hover_text("Height (or a face row's place) follows m: skipped moduli leave gaps (in the stack dashed, with the gap size) and the edges stay straight (G)");
        ui.selectable_value(&mut c.spacing, Spacing::Packed, "packed").on_hover_text("By index: no gaps, but the sides curve because slice size no longer tracks height (G)");
    });
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("highlight").color(ui.visuals().weak_text_color()));
        ui.add_enabled_ui(!c.color_slice, |ui| mono_drag(ui, &mut c.brush, 0.0..=64.0, 0.05).on_hover_text("Cubes within this many cells of the hovered cube light up"));
        ui.checkbox(&mut c.color_slice, "whole slice").on_hover_text("Light up the whole hovered slice (or face row) instead");
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut c.ghosts, "ghost zeros").on_hover_text("Zero cells as faint ghost cubes: the tunnels glow through the solid");
        ui.label(egui::RichText::new("cubes ≤").color(ui.visuals().weak_text_color()));
        egui::ComboBox::from_id_salt("tetra budget").width(70.0).selected_text(short(c.budget)).show_ui(ui, |ui| {
            for b in BUDGETS {
                ui.selectable_value(&mut c.budget, b, short(b));
            }
        }).response.on_hover_text("Most cubes drawn. A larger selection draws every s-th slice, row and cell, as cubes s cells wide (the HUD says so), instead of freezing");
    });
    let mods = selected_moduli(&c.sweep);
    let text = match (mods.first(), mods.last()) {
        (Some(a), Some(b)) => {
            let s = view_stride(c.view, &mods, c.budget);
            let lod = if s > 1 { format!(" · stride {s}") } else { String::new() };
            format!("{} moduli, {a}–{b}{lod}", mods.len())
        }
        _ => "no moduli in the range".into(),
    };
    mono_line(ui, &text);
    ui.separator();
    note(ui, "Keys: V next view · U unfold again · G spacing · O options · Esc back. Stack: drag orbit, right / shift-drag pan, wheel zoom. Face and diamond: drag pan, wheel zoom. Double-click reframes.");
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
    let sub = match model.tetra.layout.as_ref().and_then(|l| Some((*l.mods.first()?, *l.mods.last()?))) {
        Some((a, z)) => format!("m {a}–{z} · click to open"),
        None => "click to open".into(),
    };
    painter.text(Pos2::new(img.min.x + 4.0, img.max.y - 3.0), Align2::LEFT_BOTTOM, sub, FontId::monospace(9.0), Color32::from_rgba_unmultiplied(187, 187, 187, 190));
    if resp.clicked() {
        model.tetra.open = true;
        // the space shows everything colored; the tile comes back as wire
        model.tetra.tile_hot = false;
        model.tetra.fill = 0.0;
        if model.tetra.config.view == View::Diamond {
            model.tetra.unfold = 0.0;
        }
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
    if model.tetra.config.options {
        egui::Panel::left("tetra options").exact_size(OPTIONS_W).resizable(false).show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.set_max_width(OPTIONS_W - 24.0);
                options(ui, &mut model.tetra);
            });
        });
    }
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
        // the options, when hidden: a small button in the top right corner
        let mut show_options = false;
        let mut obtn = Rect::NOTHING;
        if !model.tetra.config.options {
            obtn = Rect::from_min_size(Pos2::new(rect.max.x - 12.0 - 96.0, rect.min.y + 12.0), Vec2::new(96.0, 22.0));
            let oresp = ui.put(obtn, egui::Button::new(egui::RichText::new("options ▶").monospace().size(11.0))).on_hover_text("Show the tetra stack's options (O)");
            show_options = oresp.clicked();
        }

        if handle && ui.input(|i| i.key_pressed(egui::Key::F11)) {
            model.fullscreen = !model.fullscreen;
        }
        let t = &mut model.tetra;
        if show_options {
            t.config.options = true;
        }
        let flat = t.config.view.flat();
        if handle {
            if tresp.clicked() {
                t.open = false;
            }
            if let Some(cam) = t.camera.as_mut() {
                let shift = ui.input(|i| i.modifiers.shift);
                let d = resp.drag_delta();
                if d != Vec2::ZERO {
                    let pan = flat || resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) || (shift && resp.dragged_by(egui::PointerButton::Primary));
                    if pan {
                        cam.pan(d.x, d.y, rect.height());
                    } else if resp.dragged_by(egui::PointerButton::Primary) {
                        cam.orbit(d.x, d.y);
                    }
                }
                if resp.hovered() {
                    let (scroll, zoom, at) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta(), i.pointer.hover_pos()));
                    let s = scroll + if zoom != 1.0 { zoom.ln() * 600.0 } else { 0.0 };
                    if s != 0.0 {
                        match at {
                            Some(p) if flat => cam.zoom_at(w, h, (p.x - rect.min.x) * ppp, (p.y - rect.min.y) * ppp, s),
                            _ => cam.zoom(s),
                        }
                    }
                }
            }
            if resp.double_clicked() {
                t.camera = None;
            }
            if !ui.ctx().egui_wants_keyboard_input() {
                let (esc, g, v, u, o) = ui.input(|i| (i.key_pressed(egui::Key::Escape), i.key_pressed(egui::Key::G), i.key_pressed(egui::Key::V), i.key_pressed(egui::Key::U), i.key_pressed(egui::Key::O)));
                if esc && !t.focus_last_frame {
                    t.open = false;
                }
                if g {
                    t.config.spacing = t.config.spacing.toggled();
                }
                if v {
                    let next = t.config.view.next();
                    t.set_view(next);
                }
                if u && t.config.view == View::Diamond {
                    t.replay_unfold();
                }
                if o {
                    t.config.options = !t.config.options;
                }
            }
            t.focus_last_frame = ui.ctx().memory(|m| m.focused().is_some()) || ui.ctx().any_popup_open();
            let p = resp.hover_pos().filter(|p| resp.hovered() && !resp.dragged() && !thumb.expand(4.0).contains(*p) && !obtn.contains(*p));
            t.pointer = p.map(|p| ((p.x - rect.min.x) * ppp, (p.y - rect.min.y) * ppp));
        }

        // labels: gaps in the stack, the moduli down the face's left edge
        if let (Some(l), Some(cam)) = (&t.layout, t.camera) {
            let to_pt = |p: [f32; 3]| cam.project(w, h, p).map(|(x, y)| Pos2::new(rect.min.x + x / ppp, rect.min.y + y / ppp));
            let label_col = Color32::from_rgba_unmultiplied(120, 220, 150, 170);
            let mut last_y = f32::NEG_INFINITY;
            let mut drawn = 0;
            if l.view == View::Stack {
                for (a, b) in l.gaps() {
                    // between the two slices' apex corners
                    let (pa, pb) = (l.corners(a, l.height_of(a))[0], l.corners(b, l.height_of(b))[0]);
                    let Some(p) = to_pt([(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0, (pa[2] + pb[2]) / 2.0]) else { continue };
                    if !rect.contains(p) || (p.y - last_y).abs() < 12.0 || drawn > 400 {
                        continue;
                    }
                    last_y = p.y;
                    drawn += 1;
                    painter.text(p - Vec2::new(8.0, 0.0), Align2::RIGHT_CENTER, format!("+{}", b - a), FontId::monospace(10.0), label_col);
                }
            } else {
                let gap = l.half_extents()[0] * 2.0 + 0.6;
                for (si, s) in l.slices.iter().enumerate() {
                    let SliceKind::Row(n) = s.kind else { continue };
                    let c = l.center(si, n, 0);
                    let Some(p) = to_pt([c[0] - gap, c[1], c[2]]) else { continue };
                    if !rect.contains(p) || (p.y - last_y).abs() < 11.0 || drawn > 400 {
                        continue;
                    }
                    last_y = p.y;
                    drawn += 1;
                    painter.text(p, Align2::RIGHT_CENTER, s.m.to_string(), FontId::monospace(10.0), label_col);
                }
                // the diamond's lower half: the base slice, all its rows, mod M
                if l.view == View::Diamond {
                    if let Some(bi) = l.slices.len().checked_sub(1) {
                        let b = &l.slices[bi];
                        let n_last = (b.rows - 1) * l.stride;
                        let tip = l.center(bi, 0, 0);
                        if let Some(p) = to_pt([tip[0], tip[1] - gap, tip[2]]) {
                            painter.text(p, Align2::CENTER_TOP, format!("mod {} · rows 0–{}", b.m, n_last), FontId::monospace(10.0), label_col);
                        }
                        let e = l.center(bi, n_last, n_last);
                        if let Some(p) = to_pt([e[0] + gap, e[1], e[2]]) {
                            painter.text(p, Align2::LEFT_CENTER, format!("◀ hinge: row {n_last} mod {}", b.m), FontId::monospace(10.0), label_col);
                        }
                    }
                }
            }
        }

        // the hovered face row's facts (computed once per row)
        if let (Some(c), Some(l)) = (t.hover, &t.layout) {
            if let Some(s) = l.slices.get(c.slice as usize) {
                if matches!(s.kind, SliceKind::Row(_)) && t.row_stats.is_none_or(|r| r.0 != s.m) {
                    let md = Modulus::new(s.m);
                    t.row_stats = Some((s.m, face_row_alternates(&md), face_row_zeros(&md)));
                }
            }
        }

        // HUD
        let mut lines: Vec<String> = Vec::new();
        if let Some(l) = &t.layout {
            match (l.mods.first(), l.mods.last()) {
                (Some(a), Some(z)) => {
                    lines.push(format!("TETRA STACK · {} · m {a}–{z} · {} moduli · {}", l.view.label(), l.mods.len(), t.config.sweep.mode.label()));
                    lines.push(format!("{} cubes, {} zeros{}", group(t.counts.0 as u64), group(t.counts.1 as u64), if t.config.ghosts { " (ghosts)" } else { " (hidden)" }));
                    if l.stride > 1 {
                        lines.push(format!("LOD: stride {} — 1 in {} slices, rows and cells, as cubes {} cells wide: {} of {} cells", l.stride, l.stride, l.stride, group(l.total as u64), group(l.full_total)));
                    }
                }
                _ => lines.push("TETRA STACK · no moduli in the range".into()),
            }
            match l.view {
                View::Stack => {
                    lines.push(format!("shape: {}", l.shape.label()));
                    lines.push(match l.spacing {
                        Spacing::True => "true spacing: height follows m; gaps are skipped moduli (G)".into(),
                        Spacing::Packed => "packed: height = slice index, so the sides curve: slice size no longer tracks height (G)".into(),
                    });
                }
                View::Face | View::Diamond => {
                    lines.push("front face: row m − 1 of the triangle mod m for each m, so row r = C(r, k) mod (r + 1)".into());
                    lines.push("a row alternates 1, m−1, 1, … exactly when m is prime; zeros need 3+ distinct prime factors (first at 30)".into());
                    if l.view == View::Diamond {
                        let deg = l.fold.to_degrees();
                        let hinge = if deg > 0.05 { format!(" · hinge {deg:.0}° open of {:.0}°", FOLD.to_degrees()) } else { String::new() };
                        lines.push(format!("below the hinge: the base slice mod {}, every row, unfolded{hinge}", l.mods.last().copied().unwrap_or(0)));
                    }
                    lines.push(match l.spacing {
                        Spacing::True => "true spacing: rows sit at m − 1, skipped moduli leave blank rows (G)".into(),
                        Spacing::Packed => "packed: one row per modulus of the selection, no gaps (G)".into(),
                    });
                }
            }
            let lens = match model.layers.preset() {
                Some(Preset::Mask) => "mask",
                Some(Preset::Residue) => "residue",
                _ => "residue (this layer setup has no 3D form)",
            };
            lines.push(format!("lens: {lens} · palette {}", model.scheme.label()));
            if let Some(c) = t.hover {
                if let Some(s) = l.slices.get(c.slice as usize) {
                    lines.push(describe(s.m, &c));
                    if let (SliceKind::Row(_), Some((m, alt, zeros))) = (s.kind, t.row_stats) {
                        if m == s.m {
                            let pattern = if alt { "alternates 1, m−1, 1, … (m is prime)" } else { "does not alternate (m is composite)" };
                            lines.push(format!("row {} mod {m}: {pattern} · {zeros} zero{}", m - 1, if zeros == 1 { "" } else { "s" }));
                        }
                    }
                }
            }
        }
        lines.push(if flat { "drag pan · wheel zoom · double-click reframe · V view · U unfold · O options · Esc back" } else { "drag orbit · right / shift-drag pan · wheel zoom · double-click reset · V view · O options · Esc back" }.into());
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
