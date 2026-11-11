//! The canvas: pointer and keys, the overlays drawn on it, the status strip and the wave.

use super::*;

pub(crate) fn canvas_interaction(ui: &mut egui::Ui, model: &mut Model, response: &egui::Response, rect: Rect, ppp: f32, handle: bool) {
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
    // over the wave box or the tetra tile: the canvas under them gets no hover
    model.over_wave = pointer.is_some_and(|p| model.wave.box_rect().is_some_and(|b| b.contains(p)) || model.tetra.tile_rect.is_some_and(|b| b.contains(p)));
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

pub(crate) fn canvas_overlays(ui: &mut egui::Ui, model: &mut Model, rect: Rect, ppp: f32, handle: bool) {
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
            let level_map = model.palette.level_map(model.modulus.m, model.style.offset);
            // (per frame, not per cell: the preset, the color denominator, the ν scale)
            let md = &*model.modulus;
            let preset = model.layers.preset();
            let color_q = if model.frame_style().phase { 1 } else { md.q };
            let nu_max = ((n_hi.max(1) as f64).ln() / (md.p as f64).ln()).floor().max(1.0);
            'rows: for n in n_lo..=n_hi {
                let k_lo = (model.view.x0 + n as f64 / 2.0 - 0.5).ceil().max(0.0) as i64;
                let k_hi = (x1 + n as f64 / 2.0 + 0.5).floor().min(n as f64) as i64;
                for k in k_lo..=k_hi {
                    let k = k as u32;
                    let (rgb, text) = match preset {
                        Some(Preset::Residue) | Some(Preset::Mask) => {
                            let r = md.scaled(n, k);
                            if r == 0 {
                                continue;
                            }
                            let rgb = if preset == Some(Preset::Mask) { pascal::palette::LIGHT } else { model.palette.residue_color_scaled(r, md.m, color_q, &level_map) };
                            (rgb, math::decimal_text(r as u64, md.q as u64))
                        }
                        Some(Preset::Valuation) => {
                            let nu = math::carries(n, k, md.p);
                            (model.palette.ramp[((nu as f64 / nu_max).clamp(0.0, 1.0) * 255.0).round() as usize], nu.to_string())
                        }
                        None => break 'rows,
                    };
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
pub(crate) fn status_bar(ui: &mut egui::Ui, model: &Model, gpu_ms: Option<f32>, frame_ms: f32) {
    let (cw, ch) = model.canvas;
    let mut parts = vec![if model.morph.is_some() {
        format!("μ = {} (morph)", model.mu_text())
    } else if model.modulus.is_decimal() {
        format!("μ = {}", model.modulus.label())
    } else {
        format!("m = {}", model.m)
    }];
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
pub(crate) fn wave_overlay(ui: &mut egui::Ui, model: &mut Model, rect: Rect, ppp: f32, dt: f32, handle: bool) {
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
        model.wave.set_ghost(model.ghost.source());
        // (the modulus given is μ's numerator; a decimal μ = P/Q scales by Q)
        model.wave.set_scale(model.modulus.q as u64);
        model.wave.set_morph(model.morph.as_ref().map(|(b, t)| (b.m as u64, *t)));
        if let WaveSelect::Active(a, b) = model.wave_select {
            // (whole rows for now: the marked cells pick the rows, every cell of them counts)
            model.wave.set_rows(model.modulus.m as u64, a.0 as u64, b.0 as u64);
        } else {
            let (k_lo, k_hi) = model.view.visible_k(row, cw);
            model.wave.set_source(row as u64, model.modulus.m as u64, (k_lo as u64, k_hi as u64));
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
        // the whole rows between the marks: a trapezoid from the top row's ends to the
        // bottom row's
        let (lo, hi) = (a.0.min(b.0), a.0.max(b.0));
        let (tl, tr, bl, br) = (cell_box((lo, 0)), cell_box((lo, lo)), cell_box((hi, 0)), cell_box((hi, hi)));
        let outline = vec![tl.left_top(), tr.right_top(), br.right_bottom(), bl.left_bottom()];
        painter.add(egui::Shape::closed_line(outline, egui::Stroke::new(1.0, Color32::from_white_alpha(140))));
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
            let stroke = egui::Stroke::new(2.0, Color32::from_rgba_unmultiplied(82, 242, 122, (a * fade * 255.0) as u8));
            painter.line_segment([Pos2::new(rect.min.x, yc), Pos2::new(rect.min.x + 10.0, yc)], stroke);
            painter.line_segment([Pos2::new(rect.max.x - 10.0, yc), Pos2::new(rect.max.x, yc)], stroke);
        }
    }
    model.wave.draw(&painter);
}
