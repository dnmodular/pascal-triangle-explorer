//! The sweep bar under the canvas: transport, visit modes, range, and the wave rows.

use super::*;

/// The main wave's rows, typed in: the selection's first and last rows (or the row
/// playing, when there is no selection). Typing either one makes them the selection.
pub(crate) fn wave_rows(ui: &mut egui::Ui, model: &mut Model) {
    let (mut a, mut b) = main_rows(model);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("rows").color(ui.visuals().weak_text_color()));
        let ra = mono_drag_with(ui, &mut a, 0..=u32::MAX - 1, 0.1, false).on_hover_text("First row of the selection (type it, or drag); the selection takes every cell of each row");
        ui.label(egui::RichText::new("to").color(ui.visuals().weak_text_color()));
        let rb = mono_drag_with(ui, &mut b, 0..=u32::MAX - 1, 0.1, false).on_hover_text("Last row of the selection");
        if ra.changed() || rb.changed() {
            model.wave_select = WaveSelect::Active((a.min(b), 0), (a.max(b), 0));
        }
    });
}

/// The rows the main wave plays: the selection's, or the row playing.
pub(crate) fn main_rows(model: &Model) -> (u32, u32) {
    match model.wave_select {
        WaveSelect::Active(a, b) => (a.0.min(b.0), a.0.max(b.0)),
        _ => (model.wave_row, model.wave_row),
    }
}

/// The ghost wave's controls: on, its modulus, and its own first and last rows.
pub(crate) fn ghost_controls(ui: &mut egui::Ui, model: &mut Model) {
    let main = main_rows(model);
    let g = &mut model.ghost;
    ui.horizontal(|ui| {
        let was = g.on;
        ui.checkbox(&mut g.on, "ghost wave").on_hover_text("A second wave, drawn faintly (violet) behind the green one in the same box and scroll: other rows, another modulus, to compare their shapes. Each wave is scaled to the box on its own.");
        if g.on && !was {
            // start from what plays now, in this modulus: then change either
            g.modulus = model.m;
            g.rows = main;
        }
        if g.on {
            ui.label(egui::RichText::new("mod").color(ui.visuals().weak_text_color()));
            mono_drag_with(ui, &mut g.modulus, 2..=MOD_CAP, 0.1, false).on_hover_text("The ghost's modulus: it stays here while the main wave's modulus moves");
            if ui.small_button("= m").on_hover_text("The current modulus").clicked() {
                g.modulus = model.m;
            }
        }
    });
    if !g.on {
        return;
    }
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("rows").color(ui.visuals().weak_text_color()));
        mono_drag_with(ui, &mut g.rows.0, 0..=u32::MAX - 1, 0.1, false).on_hover_text("The ghost's first row (type it, or drag)");
        ui.label(egui::RichText::new("to").color(ui.visuals().weak_text_color()));
        mono_drag_with(ui, &mut g.rows.1, 0..=u32::MAX - 1, 0.1, false).on_hover_text("The ghost's last row");
        if ui.small_button("← main").on_hover_text("Take the main wave's rows").clicked() {
            g.rows = main;
        }
    });
}

/// The fraction walks' controls in the sweep bar: the super primes' denominators and
/// order (`super_primes`), or mode All's order, with the denominators when it walks
/// every fraction by numerator.
pub(crate) fn fraction_controls(ui: &mut egui::Ui, sw: &mut Sweep, super_primes: bool) {
    let order_tip = |o: SpOrder| match (o, super_primes) {
        (SpOrder::Value, true) => "Along the number line: every super prime of the range in ascending order",
        (SpOrder::Numerator, true) => "By numerator: each prime P with all its cousins P/Q (smallest P first, the strongest structure: zeros in blocks of P rows), each cousin to the left of P",
        (SpOrder::Value, false) => "The decimal grid along the number line: whole steps, or 10⁻ᵈ steps with decimals",
        (SpOrder::Numerator, false) => "Every fraction P/Q of the range in lowest terms, by numerator: P decides the zeros (μ divides C exactly when P does), so the walk runs from the coarsest structures to the finest, each P with its cousins P/Q",
    };
    if !super_primes {
        let o = &mut sw.all_order;
        egui::ComboBox::from_id_salt("all order").width(100.0).selected_text(o.label()).show_ui(ui, |ui| {
            for x in SpOrder::ALL {
                ui.selectable_value(o, x, x.label()).on_hover_text(order_tip(x));
            }
        }).response.on_hover_text(order_tip(*o));
        if sw.all_order == SpOrder::Value {
            return;
        }
    }
    // (decimal denominators need one decimal at least: the field shows it)
    if sw.super_primes.denoms == SpDenoms::Decimal {
        sw.decimals = sw.decimals.max(1);
    }
    let sp = &mut sw.super_primes;
    egui::ComboBox::from_id_salt("sp denoms").width(86.0).selected_text(match sp.denoms {
        SpDenoms::Decimal => "decimal",
        SpDenoms::Any => "any Q",
    }).show_ui(ui, |ui| {
        ui.selectable_value(&mut sp.denoms, SpDenoms::Decimal, "decimal").on_hover_text("Denominators dividing 10ᵈ, d = the decimals setting (1 at least): the decimal fractions, 7.4 = 37/5");
        ui.selectable_value(&mut sp.denoms, SpDenoms::Any, "any Q").on_hover_text("Every denominator up to Q: 23/3, 43/6, …");
    }).response.on_hover_text("Which denominators the fractions P/Q take");
    if sp.denoms == SpDenoms::Any {
        bar_label(ui, "Q ≤");
        bar_drag(ui, 44.0, &mut sp.qmax, 1..=10_000u32, 0.1);
    }
    ui.checkbox(&mut sp.unreduced, "unreduced").on_hover_text("Also list the fractions not in lowest terms: 10/2, 10/4, 10/6, 10/8 beside 10/3, 10/7, 10/9. Each is the same μ as its lowest terms (10/4 = 5/2) and draws the same picture, so every numerator shows all its denominators");
    if super_primes {
        egui::ComboBox::from_id_salt("sp order").width(100.0).selected_text(sp.order.label()).show_ui(ui, |ui| {
            for o in SpOrder::ALL {
                ui.selectable_value(&mut sp.order, o, o.label()).on_hover_text(order_tip(o));
            }
        });
    }
}

/// The interval sequence picker: the presets in a list (picking the current one again
/// resets it to its standard values), then the sequence's knobs, and ↺ when they are
/// off the standard. The 2D sweep bar and the tetra panel share it.
pub(crate) fn seq_picker(ui: &mut egui::Ui, sw: &mut pascal::sweep::Sweep, id: &str, width: f32) {
    let note = if sw.seq == Seq::Fixed {
        "Constant step from the range start."
    } else if sw.seq.absolute() {
        "Lands directly on the sequence's values at or above the range start."
    } else {
        "Increments follow the sequence from the range start; zero increments are skipped."
    };
    egui::ComboBox::from_id_salt(id)
        .width(width)
        .selected_text(sw.seq_name())
        .show_ui(ui, |ui| {
            for q in Seq::ALL {
                let tip = if q == sw.seq { "Pick it again to reset its knobs to the standard values" } else { "Its standard values; the knobs beside the list open it up" };
                if ui.selectable_label(sw.seq == q, q.label()).on_hover_text(tip).clicked() {
                    sw.choose_seq(q);
                }
            }
        })
        .response
        .on_hover_text(format!("{note} Picking the selected sequence again resets it."));
    if sw.seq == Seq::Fixed {
        bar_label(ui, "step");
        bar_drag(ui, 56.0, &mut sw.step, 1..=10000u32, 0.2);
        return;
    }
    let mut k = sw.knobs();
    for (v, kn) in k.iter_mut().zip(sw.seq.knobs()) {
        if kn.flag {
            let mut on = *v == 1;
            ui.checkbox(&mut on, kn.name).on_hover_text(kn.tip);
            *v = on as u32;
        } else {
            bar_label(ui, kn.name);
            let w = if kn.hi >= 1000 { 52.0 } else { 36.0 };
            bar_drag(ui, w, v, kn.lo..=kn.hi, 0.1).on_hover_text(format!("{} (standard {})", kn.tip, kn.standard));
        }
    }
    sw.set_knobs(k);
    if !sw.standard_knobs() && bar_button(ui, "↺", false, &format!("Back to the standard {} (or pick it again in the list)", sw.seq.label())) {
        sw.choose_seq(sw.seq);
    }
}

/// The sweep controls as a two-row bar under the canvas. Every control sits in a
/// fixed-width slot and explanatory text lives in tooltips, so nothing reflows while
/// the modulus or the mode changes.
pub(crate) fn sweep_bar(ui: &mut egui::Ui, model: &mut Model) {
    let sw = &mut model.sweep;
    let mut action: Option<i32> = None; // Some(0) = jump to start, Some(n) = step n frames
    let before = (sw.mode, sw.prime_order, sw.seq, sw.step, sw.knobs(), sw.start, sw.end, sw.decimals, sw.morph, sw.super_primes, sw.all_order, sw.kin_order);
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
        slot(ui, 340.0, |ui| match sw.mode {
            ModMode::Prime => {
                egui::ComboBox::from_id_salt("primeorder").width(150.0).selected_text(sw.prime_order.label()).show_ui(ui, |ui| {
                    for o in PrimeOrder::ALL {
                        ui.selectable_value(&mut sw.prime_order, o, o.label());
                    }
                });
            }
            // (as in the reference, the 2×Pn and 3×Pn walks start on their first term)
            ModMode::Interval => seq_picker(ui, sw, "seq", 130.0),
            ModMode::SuperPrime => fraction_controls(ui, sw, true),
            ModMode::Kinship => {
                let o = &mut sw.kin_order;
                egui::ComboBox::from_id_salt("kin order").width(100.0).selected_text(o.label()).show_ui(ui, |ui| {
                    for x in KinOrder::ALL {
                        ui.selectable_value(o, x, x.label()).on_hover_text(x.tip());
                    }
                }).response.on_hover_text(o.tip());
            }
            ModMode::All => fraction_controls(ui, sw, false),
            _ => {}
        });
        bar_sep(ui);
        bar_label(ui, "range");
        bar_drag(ui, 70.0, &mut sw.start, 0..=MOD_CAP, 1.0);
        bar_label(ui, "to");
        bar_drag(ui, 70.0, &mut sw.end, 1..=MOD_CAP, 1.0);
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
        bar_label(ui, "decimals");
        let max = sw.max_decimals();
        sw.decimals = sw.decimals.min(max);
        bar_drag(ui, 36.0, &mut sw.decimals, 0..=max, 0.05).on_hover_text(format!(
            "Steps between the visited moduli: with d decimals (up to {max}) each step moves μ by 10⁻ᵈ through the gap to the next one, an analog glide from modulus to modulus (C mod μ = C − μ⌊C/μ⌋, exact at any row). Shallow rows slide smoothly; deeper ones turn faster, about C(n,k)·10⁻ᵈ/μ² a step. 0 = whole moduli. The exact residue needs μ's numerator within {}: past it (712.3456 at 4 decimals) the nearest fraction that fits is drawn and the panel says so; a morph is exact at every decimal.",
            group(MOD_CAP as u64)
        ));
        ui.checkbox(&mut sw.morph, "morph").on_hover_text("How the decimal steps between two visited moduli are drawn. Off: the exact real residue C mod μ (zeros jump with μ's numerator, deep rows turn over every step). On: a morph from the lower modulus to the next, each cell's color sliding along the palette from its color mod m to its color mod m′ (through the colors between), zeros fading in and out, the wave blended the same way: every in-between frame is its own, to stop on, play or export.");
        bar_sep(ui);
        let text = if sw.fraction_mode() {
            let what = match sw.mode {
                ModMode::SuperPrime => "super primes",
                ModMode::Kinship => "moduli",
                _ => "fractions",
            };
            let cap = match (sw.mode, sw.kin_order) {
                (ModMode::Kinship, KinOrder::Lattice) => pascal::sweep::KIN_LATTICE_MAX,
                (ModMode::Kinship, KinOrder::LookAlike) => pascal::sweep::KIN_LOOK_MAX,
                _ => pascal::sweep::SP_MAX,
            };
            let list = sw.super_list();
            let mut pv: Vec<String> = list.iter().take(6).map(|&(p, q)| math::fraction_label(p, q)).collect();
            if list.len() > 6 {
                pv.push("…".into());
            }
            let capped = if list.len() >= cap { " (capped)" } else { "" };
            format!("{} {what}{capped} · {}", group(list.len() as u64), pv.join(", "))
        } else {
            let preview = sw.preview(8);
            let frames = sw.frames_in_range(100_000);
            let mut pv: Vec<String> = preview.iter().map(|v| v.to_string()).collect();
            if preview.len() >= 8 {
                pv.push("…".into());
            }
            format!("{} frame{} · {}", group(frames as u64), if frames == 1 { "" } else { "s" }, pv.join(", "))
        };
        mono_line(ui, &text);
    });

    let changed = before != (sw.mode, sw.prime_order, sw.seq, sw.step, sw.knobs(), sw.start, sw.end, sw.decimals, sw.morph, sw.super_primes, sw.all_order, sw.kin_order);
    match action {
        Some(0) if model.super_mode() => {
            let last = model.sweep.super_list().len().saturating_sub(1);
            model.set_super(if model.sweep.dir > 0 { 0 } else { last }, true);
        }
        Some(n) if model.super_mode() => {
            model.sweep.playing = false;
            let i = model.sweep.super_step(model.sp_index, n);
            model.set_super(i, !model.link_hold_sweep);
        }
        Some(0) => {
            let hi = model.sweep.bounds().1;
            let (m, sub) = if model.sweep.dir > 0 { model.sweep.first_pos() } else { (hi, 0) };
            model.set_pos(m, sub, true);
        }
        Some(n) => {
            // the sweep's own frame steps, like playback, leave held rows alone
            model.sweep.playing = false;
            let (m, sub) = model.sweep.step_fine_by((model.m, model.sub), n);
            model.set_pos(m, sub, !model.link_hold_sweep);
        }
        None => {}
    }
    // re-snap when the mode or its parameters changed, as the reference does (a new
    // number of decimals lands on the whole modulus; the super-prime walk on the super
    // prime nearest μ)
    if changed {
        if model.super_mode() {
            let (p, q) = (model.modulus.m, model.modulus.q);
            if let Some(i) = model.sweep.super_nearest(p, q) {
                model.set_super(i, true);
            }
        } else {
            model.goto(model.m as i64);
        }
    }
}
