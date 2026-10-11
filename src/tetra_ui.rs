//! The tetra stack's UI: its panel section, the live preview tile on the 2D canvas and
//! the isolated 3D space (the stack, a HUD, and a thumbnail of the 2D view to go back).

use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};
use pascal::layers::Preset;
use pascal::math::Modulus;
use pascal::palette::Scheme;
use pascal::sweep::{ModMode, PrimeOrder, Seq};
use pascal::tetra::{describe, selected_moduli, Camera, CubeId, Layout, SceneKey, Spacing, TetraConfig, BUDGETS, TETRA_MAX_M};
use pascal::tetra_gpu::{outline_vertices, DrawOptions, TetraRenderer, BACKGROUND};
use pascal::wave_overlay::Corner;

use super::{group, mono_drag, mono_line, note, Model};

/// Preview spin, rad/s.
const SPIN: f32 = 0.35;

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
    /// Pointer and camera of the last pick, so an unchanged view is not picked again.
    last_pick: Option<((f32, f32), Camera)>,
    focus_last_frame: bool,
}

/// GPU-side caches of the app.
#[derive(Default)]
pub struct TetraCache {
    scene: Option<SceneKey>,
    colors: Option<(Scheme, u32)>,
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

    pub fn advance(&mut self, dt: f32) {
        if !self.open && self.config.spin && self.visible() {
            self.spin = (self.spin + dt * SPIN) % std::f32::consts::TAU;
        }
    }

    /// The preview is turning: the app repaints it a few dozen times a second.
    pub fn spinning(&self) -> bool {
        !self.open && self.config.spin && self.visible()
    }
}

/// Rebuild the stack when its selection changed, recolor it when the palette or offset
/// did, then draw it into the target (and pick under the pointer in the 3D space).
/// Returns true when the target was recreated (re-register it with egui).
pub fn render(model: &mut Model, cache: &mut TetraCache, r: &mut TetraRenderer, enc: &mut wgpu::CommandEncoder) -> bool {
    if !model.tetra.visible() {
        return false;
    }
    let t = &mut model.tetra;
    let key = t.config.scene_key();
    if cache.scene != Some(key) {
        let mods = selected_moduli(&t.config.sweep);
        let layout = Layout::new(&mods, t.config.spacing, t.config.budget);
        let mds: Vec<Modulus> = mods.iter().map(|&m| Modulus::new(m)).collect();
        r.set_scene(&layout, &mds, &model.palette, model.style.offset);
        r.set_lines(&outline_vertices(&layout));
        t.counts = r.read_counts();
        // a new range is framed anew; a new spacing or level of detail keeps the view
        let reframe = cache.scene.is_none_or(|k| (k.0, k.1, k.2, k.3, k.4, k.5) != (key.0, key.1, key.2, key.3, key.4, key.5));
        if reframe {
            t.camera = None;
        }
        t.layout = Some(layout);
        t.hover = None;
        t.last_pick = None;
        cache.scene = Some(key);
        cache.colors = Some((model.scheme, model.style.offset.to_bits()));
    }
    let colors = (model.scheme, model.style.offset.to_bits());
    if cache.colors != Some(colors) {
        r.set_colors(&model.palette, model.style.offset);
        cache.colors = Some(colors);
    }
    let recreated = r.ensure_target(t.target_px.0, t.target_px.1);
    let Some(camera) = t.draw_camera() else { return recreated };
    let mask = model.layers.preset() == Some(Preset::Mask);
    let hover = if t.open { t.hover } else { None };
    r.draw(enc, &DrawOptions { camera, hover, brush: t.config.brush, color_slice: t.config.color_slice, ghosts: t.config.ghosts, mask });
    if t.open {
        match t.pointer {
            Some(p) if t.last_pick != Some((p, camera)) => {
                if r.encode_pick(enc, &camera, p.0, p.1) {
                    t.last_pick = Some((p, camera));
                }
            }
            Some(_) => {}
            None => {
                t.hover = None;
                t.last_pick = None;
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
        mono_drag(ui, &mut c.sweep.start, 2..=TETRA_MAX_M, 1.0).on_hover_text("First modulus (the apex slice)");
        ui.label(egui::RichText::new("to").color(ui.visuals().weak_text_color()));
        mono_drag(ui, &mut c.sweep.end, 2..=TETRA_MAX_M, 1.0).on_hover_text(format!("Last modulus (the base slice), at most {TETRA_MAX_M}"));
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
                    mono_drag(ui, &mut c.sweep.step, 1..=TETRA_MAX_M, 0.2).on_hover_text("Step from the first modulus");
                }
            }
            _ => {}
        }
    });
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("spacing").color(ui.visuals().weak_text_color()));
        ui.selectable_value(&mut c.spacing, Spacing::True, "true").on_hover_text("Slice height = m: skipped moduli leave gaps (dashed, with the gap size) and the edges stay straight (G)");
        ui.selectable_value(&mut c.spacing, Spacing::Packed, "packed").on_hover_text("Slice height = its index: no gaps, but the sides curve because slice size no longer tracks height (G)");
    });
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("brush").color(ui.visuals().weak_text_color()));
        ui.add_enabled_ui(!c.color_slice, |ui| mono_drag(ui, &mut c.brush, 0.0..=64.0, 0.05).on_hover_text("Cells within this many cells of the hovered cube show their palette color"));
        ui.checkbox(&mut c.color_slice, "whole slice").on_hover_text("Color the whole hovered slice instead of a brush");
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
            let l = t.layout.as_ref().filter(|l| l.slices.len() == mods.len());
            let lod = l.map_or(String::new(), |l| if l.stride > 1 { format!(" · stride {}", l.stride) } else { String::new() });
            format!("{} slices, m {a}–{b}{lod}", mods.len())
        }
        _ => "no moduli in the range".into(),
    };
    mono_line(ui, &text);
    note(ui, "One slice per modulus m: rows 0..m−1 of the triangle mod m, stacked downward, so the solid is a tetrahedron. Zero cells are left out, and their voids run through it as tunnels. Hover a cube to color the cells around it.");
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
        return;
    }
    let b = tile_rect(model, rect);
    model.tetra.tile_rect = Some(b);
    let img = b.shrink(6.0);
    model.tetra.target_px = (((img.width() * ppp).round() as u32).max(1), ((img.height() * ppp).round() as u32).max(1));
    let resp = ui.interact(b, ui.id().with("tetra tile"), Sense::click()).on_hover_text("Open the tetra stack in its own 3D space");
    let painter = ui.painter_at(rect);
    let hot = resp.hovered();
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
        let tw = (rect.width() * 0.18).clamp(120.0, 240.0);
        let th = (tw / aspect).clamp(60.0, 200.0);
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
                let mid = [0.0, -((a + b) as f32) / 2.0, 0.0];
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
            lines.push(match l.spacing {
                Spacing::True => "true spacing: height = m; gaps are skipped moduli (G)".into(),
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
