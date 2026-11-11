//! Small egui helpers shared by the panel, the sweep bar and the tetra options.

use super::*;

pub(crate) fn header(ui: &mut egui::Ui, text: &str) {
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

pub(crate) fn note(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(egui::RichText::new(text).size(12.0).color(ui.visuals().weak_text_color())).wrap());
}

/// A number field in the monospace font.
pub(crate) fn mono_drag<N: egui::emath::Numeric>(ui: &mut egui::Ui, v: &mut N, range: std::ops::RangeInclusive<N>, speed: f64) -> egui::Response {
    mono_drag_with(ui, v, range, speed, true)
}

/// `mono_drag`; with `live` false a typed value applies on Enter or focus loss only.
pub(crate) fn mono_drag_with<N: egui::emath::Numeric>(ui: &mut egui::Ui, v: &mut N, range: std::ops::RangeInclusive<N>, speed: f64, live: bool) -> egui::Response {
    let old = ui.style().override_font_id.clone();
    ui.style_mut().override_font_id = Some(FontId::monospace(12.0));
    let r = ui.add(egui::DragValue::new(v).range(range).speed(speed).update_while_editing(live));
    ui.style_mut().override_font_id = old;
    r
}

/// A single monospace line that never wraps (long text is cut with …, full text on hover),
/// for readouts that change while you work.
pub(crate) fn mono_line(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.add(egui::Label::new(egui::RichText::new(text).font(FontId::monospace(12.0)).color(ui.visuals().weak_text_color())).truncate())
}

pub(crate) fn mono_note(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(egui::RichText::new(text).font(FontId::monospace(12.0)).color(ui.visuals().weak_text_color())).wrap());
}

pub(crate) fn section<R>(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    ui.separator();
    header(ui, title);
    let r = add(ui);
    ui.add_space(4.0);
    r
}

/// Subscript digits: 7 → ₇.
pub(crate) fn sub(n: u32) -> String {
    const SUB: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
    n.to_string().chars().map(|c| SUB[c as usize - '0' as usize]).collect()
}

/// A child area of exactly `w` points wide, so its contents can change (a combo box
/// appearing, a number growing) without moving anything after it.
pub(crate) fn slot<R>(ui: &mut egui::Ui, w: f32, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let h = ui.spacing().interact_size.y;
    ui.allocate_ui_with_layout(Vec2::new(w, h), egui::Layout::left_to_right(egui::Align::Center), |ui| {
        ui.set_min_size(Vec2::new(w, h));
        ui.set_max_width(w);
        add(ui)
    })
    .inner
}

pub(crate) fn bar_label(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).color(ui.visuals().weak_text_color()));
}

/// Fixed-size monospace number field. A typed value applies on Enter or focus loss
/// (dragging stays live): the sweep reads these every frame, and a half-typed value
/// (the "5" of "500" clamped to a 16 ms frame) would already drive it.
pub(crate) fn bar_drag<N: egui::emath::Numeric>(ui: &mut egui::Ui, w: f32, v: &mut N, range: std::ops::RangeInclusive<N>, speed: f64) -> egui::Response {
    let old = ui.style().override_font_id.clone();
    ui.style_mut().override_font_id = Some(FontId::monospace(12.0));
    let h = ui.spacing().interact_size.y;
    let r = ui.add_sized([w, h], egui::DragValue::new(v).range(range).speed(speed).update_while_editing(false));
    ui.style_mut().override_font_id = old;
    r
}

pub(crate) fn bar_button(ui: &mut egui::Ui, label: &str, selected: bool, tip: &str) -> bool {
    let h = ui.spacing().interact_size.y;
    ui.add_sized([30.0, h], egui::Button::new(egui::RichText::new(label).font(FontId::monospace(12.0))).selected(selected)).on_hover_text(tip).clicked()
}

pub(crate) fn bar_sep(ui: &mut egui::Ui) {
    ui.add_space(6.0);
    ui.separator();
    ui.add_space(6.0);
}

pub(crate) fn first_sentence(s: &str) -> &str {
    match s.find(". ") {
        Some(i) => &s[..i + 1],
        None => s,
    }
}

/// `1234567` → `1,234,567`.
pub(crate) fn group(n: u64) -> String {
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
