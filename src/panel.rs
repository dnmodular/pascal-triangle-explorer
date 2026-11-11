//! The side panel: modulus, view, colors, layers, export.

use super::*;

pub(crate) fn panel_ui(ui: &mut egui::Ui, model: &mut Model, canvas_px: (u32, u32), _frame_ms: f32) {
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
                let text = model.m_text.clone();
                if !model.goto_text(&text) {
                    model.m_text = model.modulus.label();
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
        let slider = ui.add(egui::Slider::new(&mut m, 1..=MOD_CAP).logarithmic(true).show_value(false));
        if slider.drag_started() {
            model.sweep.playing = false;
        }
        if slider.changed() {
            model.goto(m as i64);
        }
        let fs = &model.modulus.factors.iter().map(|t| t.f).collect::<Vec<_>>();
        let md = &model.modulus;
        let info = if let Some((b, t)) = &model.morph {
            format!("μ = {}: morph {} → {}, {:.0}%", model.mu_text(), model.m, b.m, t * 100.0)
        } else if model.rounded {
            format!("μ ≈ {} (numerator past {}): drawn as {}", model.mu_text(), group(MOD_CAP as u64), md.label())
        } else if md.is_decimal() {
            let label = md.label();
            let frac = if label.contains('/') { String::new() } else { format!(" = {}/{}", md.m, md.q) };
            format!("μ = {label}{frac}{}", if md.is_prime() { " · super prime" } else { "" })
        } else if md.is_one() {
            "mod 1: every residue is 0".to_string()
        } else if md.is_prime() {
            format!("{} is prime", model.m)
        } else {
            format!("{} = {}", model.m, math::factor_string(fs))
        };
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
            ui.label(egui::RichText::new("scroll").color(ui.visuals().weak_text_color()));
            mono_drag(ui, &mut model.wave.config.scroll, -3.0..=3.0, 0.01).on_hover_text("The wave slides sideways at this many rad/s of θ (negative: to the left); 0 holds it still");
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
                    if ui.small_button("select cells").on_hover_text("Average the waves of whole rows: click a cell in the first row (red mark), then one in the last row (blue mark); every cell of those rows and the rows between counts").clicked() {
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
            WaveSelect::Active((n1, _), (n2, _)) => {
                let (lo, hi) = (n1.min(n2), n1.max(n2));
                if lo == hi { format!("row {lo}, whole") } else { format!("rows {lo}–{hi}, whole") }
            }
        };
        mono_line(ui, &status);
        wave_rows(ui, model);
        ghost_controls(ui, model);
        note(ui, "Plays one row as w(θ) = Σ (C(n,k) mod m)·cos((n−2k)θ + φ): the row under the pointer, else the last visible row. Click a cell to pin its row; a selection plays the average of every cell of its rows instead.");
    });

    section(ui, "Tetra stack", |ui| tetra_ui::panel(ui, model));

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
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("fractions by").color(ui.visuals().weak_text_color()));
            let fc = &mut model.frac_colors;
            egui::ComboBox::from_id_salt("frac colors").width(70.0).selected_text(fc.label()).show_ui(ui, |ui| {
                for x in FracColors::ALL {
                    ui.selectable_value(fc, x, x.label()).on_hover_text(x.tip());
                }
            }).response.on_hover_text(fc.tip());
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
