//! The tetra stack's UI: a small entry in the main panel, the live preview tile on the
//! 2D canvas, and the isolated 3D space with its own options panel, a HUD, and a
//! thumbnail of the 2D view to go back. The space shows the stack, one of its faces laid
//! flat (front, left, right, base, or the stack from the top), or the diamond (the
//! front face with the base slice hinged down beside it).

use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};
use pascal::layers::Preset;
use pascal::math::Modulus;
use pascal::palette::Scheme;
use pascal::sweep::{ModMode, PrimeOrder};
use pascal::tetra::{describe, Cell, Face, TILE_BUDGET, Side, Snap, face_row_alternates, face_row_zeros, selected_moduli, view_stride, Camera, CubeId, Layout, SceneKey, Shape, SliceKind, Spacing, TetraConfig, View, BUDGETS, FOLD, TETRA_MAX_M};
use pascal::tetra_gpu::{outline_vertices, DrawOptions, PickSpec, TetraRenderer, BACKGROUND};
use pascal::wave_overlay::Corner;

use super::{group, mono_drag, mono_drag_with, mono_line, note, Model};

/// Preview spin, rad/s.
const SPIN: f32 = 0.35;
/// Seconds for the tile's color to run from the apex to the base (and back).
const FILL_TIME: f32 = 0.35;
/// Seconds for the diamond's hinge to swing open.
const UNFOLD_TIME: f32 = 1.1;
/// Seconds for a snapped camera to glide into place.
const SNAP_TIME: f32 = 0.45;
/// What a peel extent was counted for: the sides, and the layout (view, face, total,
/// stride, slices).
type PeelKey = ([bool; 5], View, Face, u32, u32, usize);
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
    last_pick: Option<((f32, f32), Camera, PickSpec)>,
    /// The peel animation's clock (a layer comes off every 1/speed s), and the extent
    /// (layers that peel everything) for the sides and the layout it was counted for.
    peel_clock: f32,
    peel_extent: Option<(PeelKey, u32)>,
    /// A snapped camera gliding into place: from, to, progress 0..1.
    snap: Option<(Camera, Camera, f32)>,
    /// The face the camera is snapped to, until the view is turned by hand: a new shape
    /// (packed spacing, another range) re-snaps to the same face.
    snapped: Option<Snap>,
    focus_last_frame: bool,
    /// Most cubes one scene can hold on this GPU.
    max_cells: u32,
    /// The space laid itself out this frame (its target size is real, not the tile's).
    sized: bool,
    /// Pixels the flat views leave free above (the HUD) and below (the 2D thumbnail).
    flat_insets: (f32, f32),
    /// The scene the space last showed and its bounds: its camera survives a visit to
    /// the 2D view (whose tile shows the stack), and carries over a new spacing.
    space_key: Option<SceneKey>,
    space_bounds: Option<([f32; 3], [f32; 3])>,
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

    /// Enter the 3D space (the diamond's hinge swings open again), whichever way in.
    pub fn enter(&mut self) {
        self.open = true;
        self.sized = false;
        if self.config.view == View::Diamond {
            self.unfold = 0.0;
        }
    }

    /// The camera the next draw uses: the space's, or the slowly turning preview.
    fn draw_camera(&mut self) -> Option<Camera> {
        let layout = self.layout.as_ref()?;
        if self.open {
            let (w, h) = (self.target_px.0 as f32, self.target_px.1 as f32);
            let (top, bottom) = self.flat_insets;
            Some(*self.camera.get_or_insert_with(|| if layout.view.flat() { Camera::flat(layout.at_rest().bounds(), w, h, top, bottom) } else { Camera::framing(layout.bounds()) }))
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

    /// The layers that peel the whole stack away with the current sides.
    pub fn peel_extent(&mut self) -> u32 {
        let Some(l) = &self.layout else { return 0 };
        let p = self.config.peel;
        let key = (p.sides, l.view, l.face, l.total, l.stride, l.slices.len());
        match self.peel_extent {
            Some((k, e)) if k == key => e,
            _ => {
                let e = l.peel_extent(&p);
                self.peel_extent = Some((key, e));
                e
            }
        }
    }

    /// The cube budget of `view`'s scene: the setting within what the GPU holds, and a
    /// light one for the tile.
    fn budget_for(&self) -> u32 {
        let b = self.config.budget.min(self.cell_cap());
        if self.open {
            b
        } else {
            b.min(TILE_BUDGET)
        }
    }

    /// Most cubes one scene holds here (before the first render: the renderer's ceiling).
    fn cell_cap(&self) -> u32 {
        if self.max_cells == 0 {
            pascal::tetra_gpu::MAX_CELLS
        } else {
            self.max_cells
        }
    }

    /// The peel shows: the space's stack and faces (not the diamond).
    fn peeling(&self) -> bool {
        self.open && self.config.view != View::Diamond
    }

    /// The peel the next draw uses.
    fn peel_depths(&self) -> [u32; 5] {
        if self.peeling() {
            self.config.peel.depths()
        } else {
            [0; 5]
        }
    }

    /// How many layers into its face the face view is (its own side's peel): the face
    /// is rebuilt at that depth. The top is the stack itself and needs no rebuild.
    fn face_depth(&self) -> u32 {
        match self.config.face.side() {
            Some(i) if self.peeling() && self.config.view == View::Face => self.config.peel.depths()[i],
            _ => 0,
        }
    }

    /// Check a side to peel when none is: the face being looked at (the top's own side
    /// for the top), else the left.
    fn peel_something(&mut self) {
        let p = &mut self.config.peel;
        if !p.active() {
            let i = match self.config.view {
                View::Face => self.config.face.side().unwrap_or(3),
                _ => 0,
            };
            p.sides[i] = true;
        }
    }

    /// Step the peel by `d` layers by hand (it stops playing): clamped at nothing
    /// peeled and everything peeled, not wrapped.
    pub fn step_peel(&mut self, d: i32) {
        self.peel_something();
        let extent = self.peel_extent();
        let p = &mut self.config.peel;
        p.playing = false;
        p.layers = (p.layers as i64 + d as i64).clamp(0, extent as i64) as u32;
    }

    /// Play the peel forward or backward; the same direction again pauses it.
    pub fn play_peel(&mut self, backward: bool) {
        self.peel_something();
        let p = &mut self.config.peel;
        p.playing = !(p.playing && p.backward == backward);
        p.backward = backward;
        self.peel_clock = 0.0;
    }

    /// Glide the space's camera square onto a side (or straight down / up).
    pub fn start_snap(&mut self, snap: Snap) {
        let (Some(l), Some(from)) = (&self.layout, self.camera) else { return };
        if l.view != View::Stack {
            return;
        }
        let mut to = l.snap_camera(snap);
        if matches!(snap, Snap::Top | Snap::Bottom) {
            // straight down or up: keep the turn the view already has
            to.yaw = from.yaw;
        }
        self.snap = Some((from, to, 0.0));
        self.snapped = Some(snap);
    }

    pub fn advance(&mut self, dt: f32) {
        let dt = dt.max(0.0);
        // the camera gliding to a snapped view
        if let Some((from, to, t)) = self.snap.as_mut() {
            *t = (*t + dt / SNAP_TIME).min(1.0);
            let e = smoothstep(*t);
            let (from, to, done) = (*from, *to, *t >= 1.0);
            if let Some(cam) = self.camera.as_mut() {
                let turn = (to.yaw - from.yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
                cam.yaw = from.yaw + turn * e;
                cam.pitch = from.pitch + (to.pitch - from.pitch) * e;
                cam.dist = from.dist + (to.dist - from.dist) * e;
                for a in 0..3 {
                    cam.target[a] = from.target[a] + (to.target[a] - from.target[a]) * e;
                }
                cam.radius = to.radius;
            }
            if done {
                self.snap = None;
            }
        }
        // the peel, a layer at a time while it plays, round and round
        if self.config.peel.playing && self.peeling() && self.config.peel.active() {
            let speed = self.config.peel.speed.clamp(0.1, 120.0);
            self.peel_clock += dt * speed;
            let steps = self.peel_clock.floor();
            if steps >= 1.0 {
                self.peel_clock -= steps;
                let extent = self.peel_extent().max(1);
                let p = &mut self.config.peel;
                let (period, steps) = (extent + 1, steps as u32 % (extent + 1));
                p.layers = if p.backward { (p.layers.min(extent) + period - steps) % period } else { (p.layers + steps) % period };
            }
        }
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
        let peel = self.peeling() && self.config.peel.playing && self.config.peel.active();
        tile || hinge || peel || self.snap.is_some()
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
    // the frame the space opens on still has the tile's size: build and frame it once
    // the space has laid itself out
    if model.tetra.open && !model.tetra.sized {
        return false;
    }
    let t = &mut model.tetra;
    let view = t.view();
    // the budget this scene gets: the setting, within what the GPU holds; the tile turns
    // all the time, so it draws a light stack
    t.max_cells = r.max_cells();
    let budget = t.budget_for();
    let mut key = t.config.scene_key(view, t.face_depth());
    key.7 = budget;
    // keys equal but for the face's depth: the face is peeled a layer deeper or back
    let deeper = |k: Option<SceneKey>| k.is_some_and(|mut k| k.11 != key.11 && { k.11 = key.11; k == key });
    // (no layout: the model was reset to a config the cache still holds; a peel slider
    // being dragged still shows each layer)
    if (cache.scene != Some(key) && (!dragging || deeper(cache.scene))) || t.layout.is_none() {
        let mods = selected_moduli(&t.config.sweep);
        let mut layout = if view == View::Face { Layout::build_face(t.config.face, key.11, &mods, t.config.spacing, t.config.shape, budget) } else { Layout::build(view, &mods, t.config.spacing, t.config.shape, budget) };
        if view == View::Diamond {
            layout.set_fold(t.fold_angle());
        }
        let mds: Vec<Modulus> = layout.slices.iter().map(|s| Modulus::new(s.m)).collect();
        r.set_scene(&layout, &mds, &model.palette, model.style.offset);
        r.set_lines(&outline_vertices(&layout));
        t.counts = r.read_counts();
        // The space's camera: the tile's rebuilds leave it alone, and coming back to the
        // scene the space last showed keeps it. A new range or view is framed anew; in
        // the stack a new spacing, shape or level of detail keeps the angle and the
        // relative zoom.
        // The space's view is kept through every rebuild of the same view (a new range,
        // visit mode, spacing, shape or level of detail): the same angle, and the pan
        // and zoom relative to the new size. Only a new view (stack, face, diamond) or a
        // double-click frames it afresh.
        let mut resnap = None;
        if t.open && t.space_key != Some(key) && !deeper(t.space_key) {
            // (a face peeled deeper keeps the camera as it is: the layers come off in place)
            let same_view = t.space_key.is_some_and(|k| k.9 == view && k.10 == key.10);
            match (same_view, t.camera.as_mut(), t.space_bounds) {
                (true, Some(cam), Some(old)) => {
                    let new = layout.at_rest().bounds();
                    let (w, h) = (t.target_px.0 as f32, t.target_px.1 as f32);
                    let (top, bottom) = t.flat_insets;
                    let (was, now) = if view.flat() { (Camera::flat(old, w, h, top, bottom), Camera::flat(new, w, h, top, bottom)) } else { (Camera::framing(old), Camera::framing(new)) };
                    let pan = [cam.target[0] - was.target[0], cam.target[1] - was.target[1], cam.target[2] - was.target[2]];
                    let scale = now.radius / was.radius;
                    cam.target = [now.target[0] + pan[0] * scale, now.target[1] + pan[1] * scale, now.target[2] + pan[2] * scale];
                    cam.dist *= now.dist / was.dist;
                    if view.flat() {
                        cam.ortho *= now.ortho / was.ortho;
                    }
                    cam.radius = now.radius;
                    // a snapped view follows its face to the new shape
                    resnap = t.snapped;
                }
                _ => {
                    t.camera = None;
                    t.snapped = None;
                }
            }
        }
        if t.open {
            t.space_key = Some(key);
            t.space_bounds = Some(layout.at_rest().bounds());
        }
        t.layout = Some(layout);
        if let Some(sn) = resnap {
            t.start_snap(sn);
        }
        t.hover = None;
        t.last_pick = None;
        cache.scene = Some(key);
        cache.colors = Some((model.scheme, model.style.offset.to_bits()));
        cache.drawn = None;
    }
    // the space opened on the tile's scene (no rebuild): its camera belongs to it now
    if t.open && t.space_key != Some(key) && cache.scene == Some(key) {
        if let Some(l) = &t.layout {
            t.space_key = Some(key);
            t.space_bounds = Some(l.at_rest().bounds());
        }
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
    let peel = t.peel_depths();
    let solid_zeros = t.open && view.flat() && t.config.solid_zeros;
    // (the tile keeps its wire cubes)
    let cell = if t.open { t.config.cell } else { Cell::Cube };
    let opts = DrawOptions { camera, fill: t.fill_height(), hover, brush: t.config.brush, color_slice: t.config.color_slice, ghosts: t.config.ghosts, mask, flat: t.open && view.flat(), peel, solid_zeros, cell };
    // an unchanged picture is not drawn again (hover, egui repaints)
    let drawn = (opts, t.target_px);
    if recreated || cache.drawn != Some(drawn) {
        r.draw(enc, &opts);
        cache.drawn = Some(drawn);
    }
    if t.open {
        if let Some(p) = t.pointer {
            let spec = PickSpec { peel, zeros: solid_zeros, cell };
            if t.last_pick != Some((p, camera, spec)) && r.encode_pick(enc, &camera, p.0, p.1, spec) {
                t.last_pick = Some((p, camera, spec));
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
            t.enter();
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
    if t.config.view == View::Face {
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("face").color(ui.visuals().weak_text_color()));
            for f in Face::ALL {
                ui.selectable_value(&mut t.config.face, f, f.label()).on_hover_text(format!("{} (F)", f.note()));
            }
        });
        note(ui, t.config.face.note());
    }
    if t.config.view == View::Diamond && ui.small_button("unfold again (U)").on_hover_text("Fold the base slice back to where it sits in the tetrahedron and swing it open").clicked() {
        t.replay_unfold();
    }
    if t.config.view == View::Stack {
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("snap").color(ui.visuals().weak_text_color()));
            for (i, sn) in Snap::ALL.iter().enumerate() {
                let tip = match sn {
                    Snap::Top => "Straight down onto the apex",
                    Snap::Bottom => "Straight up at the base slice",
                    Snap::Left => "Square onto the left face (k = 0)",
                    Snap::Right => "Square onto the right face (k = n)",
                    Snap::Front => "Square onto the front face (the last rows)",
                };
                if ui.small_button(sn.label()).on_hover_text(format!("{tip} ({})", i + 1)).clicked() {
                    t.start_snap(*sn);
                }
            }
        });
    }
    if t.config.view != View::Diamond {
        ui.separator();
        peel_controls(ui, t);
    }
    ui.separator();
    let max_cells = t.cell_cap();
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
    });
    // the mode's own options on a line of their own (they would overflow the panel)
    if matches!(c.sweep.mode, ModMode::Prime | ModMode::Interval) {
        ui.horizontal_wrapped(|ui| match c.sweep.mode {
            ModMode::Prime => {
                egui::ComboBox::from_id_salt("tetra prime order").width(110.0).selected_text(c.sweep.prime_order.label()).show_ui(ui, |ui| {
                    for o in PrimeOrder::ALL {
                        ui.selectable_value(&mut c.sweep.prime_order, o, o.label());
                    }
                });
            }
            ModMode::Interval => super::seq_picker(ui, &mut c.sweep, "tetra seq", 110.0),
            _ => {}
        });
    }
    ui.add_enabled_ui(c.view == View::Stack || (c.view == View::Face && c.face == Face::Top), |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("shape").color(ui.visuals().weak_text_color()));
            egui::ComboBox::from_id_salt("tetra shape").width(150.0).selected_text(c.shape.label()).show_ui(ui, |ui| {
                for sh in Shape::ALL {
                    ui.selectable_value(&mut c.shape, sh, sh.label()).on_hover_text(sh.note());
                }
            }).response.on_hover_text(c.shape.note()).on_disabled_hover_text("The faces and the diamond are flat: the shape is the stack's (the top shows it)");
        });
    });
    ui.add_enabled_ui(c.view == View::Stack || (c.view == View::Face && c.face == Face::Top), |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("cell").color(ui.visuals().weak_text_color()));
            for cl in Cell::ALL {
                ui.selectable_value(&mut c.cell, cl, cl.label()).on_hover_text(format!("{} (C)", cl.note()));
            }
        });
    });
    if c.cell != Cell::Cube && (c.shape != Shape::Regular || c.spacing != Spacing::True) && (c.view == View::Stack || (c.view == View::Face && c.face == Face::Top)) {
        note(ui, "The rhombic and sphere cells fit the equilateral shape in true spacing: elsewhere the slices do not sit in each other's hollows, and the cells overlap or leave gaps.");
    }
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
        }).response.on_hover_text(format!("Most cubes drawn. \"all\" draws every cell of the range, up to {} cubes (what one scene holds here); past that, or under a smaller budget, every s-th slice, row and cell is drawn, as cubes s cells wide (the HUD says so). Fewer cubes turn faster", group(max_cells as u64)));
    });
    ui.checkbox(&mut c.solid_zeros, "solid zeros on the faces / diamond").on_hover_text("Draw the zeros of the flat views as dark cells, so the shapes have no gaps (the stack keeps its zeros as tunnels; from the top, holes show the slices below)");
    let mods = selected_moduli(&c.sweep);
    let text = match (mods.first(), mods.last()) {
        (Some(a), Some(b)) => {
            let s = view_stride(c.view, &mods, c.spacing, c.budget.min(max_cells));
            let lod = if s > 1 { format!(" · stride {s}") } else { String::new() };
            format!("{} moduli, {a}–{b}{lod}", mods.len())
        }
        _ => "no moduli in the range".into(),
    };
    mono_line(ui, &text);
    ui.separator();
    note(ui, "Keys: V next view · F next face · C next cell · 1–5 snap the stack's camera · ← → step the peel a layer (Shift: ten) · Home / End none / all · P play it, R rewind · U unfold again · G spacing · O options · Esc back. Stack: drag orbit, right / shift-drag pan, wheel zoom. Faces and diamond: drag pan, wheel zoom. Double-click reframes.");
}

/// The peel: which sides, how many layers, and the animation that takes them off one by
/// one and starts over.
fn peel_controls(ui: &mut egui::Ui, t: &mut TetraView) {
    let extent = t.peel_extent();
    let p = &mut t.config.peel;
    // (another view or face may peel away in fewer layers)
    if !p.playing && p.active() && extent > 0 {
        p.layers = p.layers.min(extent);
    }
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("peel").color(ui.visuals().weak_text_color()));
    });
    for (sd, on) in Side::ALL.into_iter().zip(p.sides.iter_mut()) {
        ui.checkbox(on, sd.label()).on_hover_text(format!("Peel the {} side; unchecked sides stay", sd.name()));
    }
    if !p.active() {
        return;
    }
    let (playing, backward) = (p.playing, p.backward);
    let mut act: Option<i32> = None;
    let mut play: Option<bool> = None;
    ui.horizontal(|ui| {
        let small = |ui: &mut egui::Ui, text: &str, on: bool, tip: &str| ui.add(egui::Button::new(egui::RichText::new(text).monospace()).small().selected(on)).on_hover_text(tip).clicked();
        if small(ui, "|◀", false, "Back to nothing peeled (Home)") {
            act = Some(i32::MIN / 2);
        }
        if small(ui, if playing && backward { "❚❚" } else { "◀" }, playing && backward, "Rewind: put the layers back one by one, and from nothing peeled start again from everything peeled (R)") {
            play = Some(true);
        }
        if small(ui, "◀|", false, "One layer back (←; Shift+← ten)") {
            act = Some(-1);
        }
        if small(ui, "|▶", false, "One layer more (→; Shift+→ ten)") {
            act = Some(1);
        }
        if small(ui, if playing && !backward { "❚❚" } else { "▶" }, playing && !backward, "Peel a layer at a time, and start over when nothing is left (P)") {
            play = Some(false);
        }
        if small(ui, "▶|", false, "Everything peeled (End)") {
            act = Some(i32::MAX / 2);
        }
    });
    ui.horizontal(|ui| {
        ui.spacing_mut().slider_width = 150.0;
        ui.add(egui::Slider::new(&mut p.layers, 0..=extent.max(1)).show_value(false)).on_hover_text("Layers peeled (← → step)");
        mono_line(ui, &format!("{} / {}", p.layers, extent));
    });
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("speed").color(ui.visuals().weak_text_color()));
        mono_drag(ui, &mut p.speed, 0.1..=120.0, 0.05).on_hover_text("Layers a second while it plays");
        ui.label(egui::RichText::new("layers / s").color(ui.visuals().weak_text_color()));
    });
    if let Some(d) = act {
        t.step_peel(d);
    }
    if let Some(b) = play {
        t.play_peel(b);
    }
}

fn short(n: u32) -> String {
    if n == u32::MAX {
        "all".into()
    } else if n >= 1_000_000 {
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
    model.tetra.sized = false;
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
        model.tetra.enter();
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
        model.tetra.sized = true;
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
        // the flat views keep clear of the HUD (up to its eleven lines) and the thumbnail
        model.tetra.flat_insets = ((12.0 + 12.0 + 11.0 * 15.0) * ppp, (th + 16.0 + 8.0) * ppp);
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
                        t.snapped = None;
                        t.snap = None;
                    } else if resp.dragged_by(egui::PointerButton::Primary) {
                        cam.orbit(d.x, d.y);
                        t.snapped = None;
                        t.snap = None;
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
                t.snapped = None;
                t.snap = None;
            }
            if !ui.ctx().egui_wants_keyboard_input() {
                let (esc, g, v, u, o, pk) = ui.input(|i| (i.key_pressed(egui::Key::Escape), i.key_pressed(egui::Key::G), i.key_pressed(egui::Key::V), i.key_pressed(egui::Key::U), i.key_pressed(egui::Key::O), i.key_pressed(egui::Key::P)));
                if t.config.view != View::Diamond {
                    let (rk, left, right, home, end, shift) = ui.input(|i| (i.key_pressed(egui::Key::R), i.key_pressed(egui::Key::ArrowLeft), i.key_pressed(egui::Key::ArrowRight), i.key_pressed(egui::Key::Home), i.key_pressed(egui::Key::End), i.modifiers.shift));
                    let n = if shift { 10 } else { 1 };
                    if pk {
                        t.play_peel(false);
                    }
                    if rk {
                        t.play_peel(true);
                    }
                    if left {
                        t.step_peel(-n);
                    }
                    if right {
                        t.step_peel(n);
                    }
                    if home {
                        t.step_peel(i32::MIN / 2);
                    }
                    if end {
                        t.step_peel(i32::MAX / 2);
                    }
                }
                if t.config.view == View::Stack {
                    let keys = [egui::Key::Num1, egui::Key::Num2, egui::Key::Num3, egui::Key::Num4, egui::Key::Num5];
                    for (key, sn) in keys.iter().zip(Snap::ALL) {
                        if ui.input(|i| i.key_pressed(*key)) {
                            t.start_snap(sn);
                        }
                    }
                }
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
                if ui.input(|i| i.key_pressed(egui::Key::C) && !i.modifiers.any()) {
                    let i = Cell::ALL.iter().position(|&c| c == t.config.cell).unwrap_or(0);
                    t.config.cell = Cell::ALL[(i + 1) % Cell::ALL.len()];
                }
                if ui.input(|i| i.key_pressed(egui::Key::F) && !i.modifiers.any()) && t.config.view == View::Face {
                    t.config.face = t.config.face.next();
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
                    if !matches!(s.kind, SliceKind::Line { .. }) {
                        continue;
                    }
                    // at the line's left end
                    let c = l.slice_corners(si).into_iter().min_by(|a, b| a[0].total_cmp(&b[0])).unwrap_or_default();
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
                        let (n_last, k_last) = (b.last_row(l.stride), b.last_k(l.stride));
                        let tip = l.center(bi, b.n0, 0);
                        if let Some(p) = to_pt([tip[0], tip[1] - gap, tip[2]]) {
                            painter.text(p, Align2::CENTER_TOP, format!("mod {} · rows {}–{}", b.m, b.n0, n_last), FontId::monospace(10.0), label_col);
                        }
                        let e = l.center(bi, n_last, k_last);
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
                if s.kind == SliceKind::row(s.m - 1) && t.row_stats.is_none_or(|r| r.0 != s.m) {
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
                    let what = if l.view == View::Face { format!("{} face", l.face.label()) } else { l.view.label().to_string() };
                    lines.push(format!("TETRA STACK · {what} · m {a}–{z} · {} moduli · {}", l.mods.len(), t.config.sweep.mode.label()));
                    lines.push(format!("{} cubes, {} zeros{}", group(t.counts.0 as u64), group(t.counts.1 as u64), if t.config.ghosts { " (ghosts)" } else { " (hidden)" }));
                    if l.stride > 1 {
                        let what = if l.view == View::Stack || (l.view == View::Face && matches!(l.face, Face::Top | Face::Base)) { "slices" } else { "face rows" };
                        lines.push(format!("LOD: stride {} — {} of {} {what} (at least a cube apart), 1 in {} rows and cells, cubes {} cells wide: {} of {} cells", l.stride, l.slices.len(), l.mods.len(), l.stride, l.stride, group(l.total as u64), group(l.full_total)));
                    }
                }
                _ => lines.push("TETRA STACK · no moduli in the range".into()),
            }
            match l.view {
                View::Stack => {
                    lines.push(format!("shape: {}", l.shape.label()));
                    match t.config.cell {
                        Cell::Rhombic => lines.push("cells: rhombic dodecahedra, the cannonball pile's own cells: no gaps, every step a whole rhombus (C)".into()),
                        Cell::Sphere => lines.push("cells: spheres, the cannonballs, each touching twelve neighbours (C)".into()),
                        Cell::Cube => {}
                    }
                    lines.push(match l.spacing {
                        Spacing::True => "true spacing: height follows m; gaps are skipped moduli (G)".into(),
                        Spacing::Packed => "packed: height = slice index, so the sides curve: slice size no longer tracks height (G)".into(),
                    });
                }
                View::Face if l.face != Face::Front => {
                    let d = t.face_depth() * l.stride;
                    lines.push(match l.face {
                        Face::Left if d == 0 => "left face: k = 0 of every slice, all 1s (peel it to go in: d layers in is C(n, d) mod m)".into(),
                        Face::Left => format!("left face, {d} in: k = {d} of every slice, C(n, {d}) mod m for {d} ≤ n < m"),
                        Face::Right if d == 0 => "right face: k = n of every slice, all 1s (peel it to go in: the left face mirrored)".into(),
                        Face::Right => format!("right face, {d} in: k = n − {d} of every slice, C(n, {d}) mod m mirrored"),
                        Face::Base => match l.slices.first() {
                            Some(b) if d == 0 => format!("base: the base slice, the whole triangle mod {}", b.m),
                            Some(b) => format!("base, {} layers up: the slice mod {}, the whole triangle", t.face_depth(), b.m),
                            None => "base: peeled through every slice".into(),
                        },
                        _ => "top: the stack seen straight down (orthographic), each cell the topmost cube over it".into(),
                    });
                    if l.face == Face::Top {
                        lines.push(format!("shape: {} · cells: {} · lower slices peek out around the upper ones", l.shape.label(), t.config.cell.label()));
                    }
                }
                View::Face | View::Diamond => {
                    let d = if l.view == View::Face { t.face_depth() * l.stride } else { 0 };
                    if d == 0 {
                        lines.push("front face: row m − 1 of the triangle mod m for each m, so row r = C(r, k) mod (r + 1)".into());
                        lines.push("a row alternates 1, m−1, 1, … exactly when m is prime; zeros need 3+ distinct prime factors (first at 30)".into());
                    } else {
                        lines.push(format!("front face, {d} in: row m − 1 − {d} of the triangle mod m for each m"));
                    }
                    if l.view == View::Diamond {
                        // l.fold is what is still folded: 0 lies flat
                        let (open, swinging) = ((FOLD - l.fold).to_degrees(), l.fold.to_degrees() > 0.05);
                        let hinge = if swinging { format!(", unfolding: hinge {open:.0}° open of {:.0}°", FOLD.to_degrees()) } else { ", unfolded flat".into() };
                        lines.push(format!("below the hinge: the base slice mod {}, every row{hinge}", l.mods.last().copied().unwrap_or(0)));
                    }
                    lines.push(match l.spacing {
                        Spacing::True => "true spacing: rows sit at m − 1, skipped moduli leave blank rows (G)".into(),
                        Spacing::Packed => "packed: one row per modulus of the selection, no gaps (G)".into(),
                    });
                }
            }
            let p = t.config.peel;
            if l.view != View::Diamond && p.active() && (p.layers > 0 || p.playing) {
                let names: Vec<&str> = p.checked().map(Side::name).collect();
                let how = if names.len() == Side::ALL.len() { "every side".to_string() } else { names.join(", ") };
                lines.push(format!("peeled: {} layer{} off {how}{} (← → step · P play · R rewind)", p.layers, if p.layers == 1 { "" } else { "s" }, if !p.playing { "" } else if p.backward { ", rewinding" } else { ", playing" }));
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
                    if let Some((m, alt, zeros)) = t.row_stats {
                        if m == s.m && s.kind == SliceKind::row(m - 1) {
                            let pattern = if alt { "alternates 1, m−1, 1, … (m is prime)" } else { "does not alternate (m is composite)" };
                            lines.push(format!("row {} mod {m}: {pattern} · {zeros} zero{}", m - 1, if zeros == 1 { "" } else { "s" }));
                        }
                    }
                }
            }
        }
        lines.push(match t.config.view {
            View::Stack => "drag orbit · right / shift-drag pan · wheel zoom · double-click reset · V view · O options · Esc back",
            View::Face => "drag pan · wheel zoom · double-click reframe · V view · F face · P peel · O options · Esc back",
            View::Diamond => "drag pan · wheel zoom · double-click reframe · V view · U unfold · O options · Esc back",
        }.into());
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
