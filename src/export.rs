//! Exports. PNG: the canvas texture, straight from the same resolve pass the live view
//! shows, so it can never differ from the screen. SVG: the triangle's first rows as
//! vector cells (the square scanner's "Save SVG"), for the residue, mask and valuation
//! colorings, which the CPU can reproduce exactly.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use crate::gpu::CanvasStyle;
use crate::layers::{Kind, LayerState, Preset, REGISTRY};
use crate::math::{carries, Modulus};
use crate::palette::{offset_shift, Palette, Scheme, BG, LIGHT};
use crate::view::ROW_H;

/// Most rows an SVG export draws (≈ 525k cells, a file of about 15 MB).
pub const SVG_MAX_ROWS: u32 = 1024;

pub fn write_png(path: &Path, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut enc = png::Encoder::new(BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(std::io::Error::other)?;
    writer.write_image_data(rgba).map_err(std::io::Error::other)?;
    Ok(())
}

/// `pascal_m7_rows0-342_spectral_kummer+density.png` (or `.svg`, per `ext`), in `dir`,
/// never overwriting.
pub fn export_path(dir: &Path, m: u32, rows: (u32, u32), scheme: Scheme, layers: &LayerState, ext: &str) -> PathBuf {
    let mut tags: Vec<String> = REGISTRY
        .iter()
        .filter(|d| d.kind != Kind::Value && layers.is_active(d.id))
        .map(|d| d.label.to_lowercase().replace(' ', ""))
        .collect();
    if tags.is_empty() {
        tags.push("residue".into());
    }
    let base = format!("pascal_m{}_rows{}-{}_{}_{}", m, rows.0, rows.1, scheme.id(), tags.join("+"));
    let mut path = dir.join(format!("{base}.{ext}"));
    let mut i = 2;
    while path.exists() {
        path = dir.join(format!("{base}_{i}.{ext}"));
        i += 1;
    }
    path
}

/// Cell width in SVG units (the height is `CELL · ROW_H`), and the margin around the
/// triangle.
const SVG_CELL: f64 = 10.0;
const SVG_PAD: f64 = 12.0;

/// The colors of rows `0..rows` as the live view paints them for the preset `layers`
/// selects, row by row (`None` when the layer stack is not one of the three presets:
/// measure fields exist only on the GPU). Valuation is ν/max ν through the ramp, with
/// the max over these rows and the GPU's f32 rounding.
pub fn preset_colors(rows: u32, md: &Modulus, palette: &Palette, layers: &LayerState, style: &CanvasStyle) -> Option<Vec<Vec<[u8; 3]>>> {
    let preset = layers.preset()?;
    let shift = offset_shift(style.offset);
    let mut max_nu = 0u32;
    if preset == Preset::Valuation {
        for n in 0..rows {
            for k in 0..=n {
                max_nu = max_nu.max(carries(n, k, md.p).min(255));
            }
        }
    }
    let inv = if max_nu == 0 { 1.0f32 } else { 1.0 / max_nu as f32 };
    Some(
        (0..rows)
            .map(|n| {
                (0..=n)
                    .map(|k| match preset {
                        Preset::Residue => palette.residue_color_shifted(md.binom(n, k), md.m, shift),
                        Preset::Mask => {
                            if md.binom(n, k) != 0 {
                                LIGHT
                            } else {
                                BG
                            }
                        }
                        Preset::Valuation => {
                            let v = (carries(n, k, md.p).min(255) as f32 * inv).clamp(0.0, 1.0);
                            palette.ramp[(v * 255.0 + 0.5).floor() as usize]
                        }
                    })
                    .collect()
            })
            .collect(),
    )
}

/// Rows `0..rows` as an SVG document: one path per color (a subpath per cell), cells as
/// wide as `SVG_CELL` and `ROW_H` of that tall, odd rows offset by half a cell as on the
/// canvas; with `style.outlines`, every cell gets the canvas's faint outline. The glow
/// is a raster effect and is left out.
pub fn svg_document(colors: &[Vec<[u8; 3]>], style: &CanvasStyle) -> String {
    let rows = colors.len();
    let ch = SVG_CELL * ROW_H;
    let w = rows as f64 * SVG_CELL + 2.0 * SVG_PAD;
    let h = rows as f64 * ch + 2.0 * SVG_PAD;
    // cells grouped by color, in a fixed order so equal inputs give equal files
    let mut groups: BTreeMap<[u8; 3], String> = BTreeMap::new();
    for (n, row) in colors.iter().enumerate() {
        let y = SVG_PAD + n as f64 * ch;
        for (k, &c) in row.iter().enumerate() {
            if c == BG && !style.outlines {
                continue; // the background already is the void color
            }
            let x = SVG_PAD + (k as f64 + (rows - 1 - n) as f64 / 2.0) * SVG_CELL;
            let d = groups.entry(c).or_default();
            let _ = write!(d, "M{}{}h{}v{}h-{}z", num(x), sep_num(y), num(SVG_CELL), num(ch), num(SVG_CELL));
        }
    }
    let mut out = String::new();
    let _ = write!(out, r#"<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">"#, num(w), num(h));
    let _ = write!(out, r#"<rect width="{}" height="{}" fill="{}"/>"#, num(w), num(h), hex(BG));
    let stroke = if style.outlines { r##" stroke="#ebeefa" stroke-opacity="0.12" stroke-width="0.6""## } else { "" };
    for (c, d) in &groups {
        let _ = write!(out, r#"<path fill="{}"{stroke} d="{d}"/>"#, hex(*c));
    }
    out.push_str("</svg>\n");
    out
}

fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// A coordinate with at most three decimals, trailing zeros trimmed.
fn num(x: f64) -> String {
    let s = format!("{x:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".into() } else { s.to_string() }
}

/// `num` with the separator SVG path data needs before it.
fn sep_num(x: f64) -> String {
    format!(" {}", num(x))
}

pub fn write_svg(path: &Path, svg: &str) -> std::io::Result<()> {
    std::fs::write(path, svg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::binom_exact;

    #[test]
    fn svg_cells_and_colors() {
        let md = Modulus::new(3);
        let pal = Palette::new(Scheme::Spectral);
        let mut layers = LayerState::default();
        layers.preset_residue();
        let style = CanvasStyle::default();
        let colors = preset_colors(9, &md, &pal, &layers, &style).unwrap();
        // every cell's color is the residue color of the true binomial
        for (n, row) in colors.iter().enumerate() {
            assert_eq!(row.len(), n + 1);
            for (k, &c) in row.iter().enumerate() {
                let r = (binom_exact(n as u32, k as u32).unwrap() % 3) as u32;
                assert_eq!(c, pal.residue_color(r, 3));
            }
        }
        let svg = svg_document(&colors, &style);
        assert!(svg.starts_with("<svg") && svg.ends_with("</svg>\n"));
        // one subpath per nonzero cell: rows 0..9 of mod 3 have 9·10/2 − zeros cells
        let zeros = colors.iter().flatten().filter(|&&c| c == BG).count();
        assert_eq!(svg.matches('M').count(), 45 - zeros);
        // apex centred: row 0's cell starts at pad + (rows − 1)/2 cells
        assert!(svg.contains(&format!("M{} 12h10", num(12.0 + 4.0 * 10.0))));
        // outlines draw the zero cells too
        let lined = svg_document(&colors, &CanvasStyle { outlines: true, ..style });
        assert_eq!(lined.matches('M').count(), 45);
        assert!(lined.contains("stroke-opacity"));
        // the color offset reaches the export
        let shifted = preset_colors(9, &md, &pal, &layers, &CanvasStyle { offset: 0.5, ..style }).unwrap();
        assert_ne!(shifted, colors);
        // field stacks are refused, the mask and valuation presets are not
        let mut fields = LayerState::default();
        fields.set_active(crate::layers::LayerId::Density, true);
        assert!(preset_colors(9, &md, &pal, &fields, &style).is_none());
        layers.preset_mask();
        assert!(preset_colors(9, &md, &pal, &layers, &style).unwrap()[4].iter().all(|&c| c == LIGHT || c == BG));
        layers.preset_valuation();
        let val = preset_colors(9, &md, &pal, &layers, &style).unwrap();
        assert_eq!(val[0][0], pal.ramp[0], "ν = 0 at the apex");
    }
}
