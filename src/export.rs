//! PNG export of the canvas texture. The pixels come straight from the same resolve
//! pass the live view shows, so an export can never differ from the screen.

use std::io::BufWriter;
use std::path::{Path, PathBuf};

use crate::layers::{Kind, LayerState, REGISTRY};
use crate::palette::Scheme;

pub fn write_png(path: &Path, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut enc = png::Encoder::new(BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(std::io::Error::other)?;
    writer.write_image_data(rgba).map_err(std::io::Error::other)?;
    Ok(())
}

/// `pascal_m7_rows0-342_spectral_kummer+density.png`, in `dir`, never overwriting.
pub fn export_path(dir: &Path, m: u32, rows: (u32, u32), scheme: Scheme, layers: &LayerState) -> PathBuf {
    let mut tags: Vec<String> = REGISTRY
        .iter()
        .filter(|d| d.kind != Kind::Value && layers.is_active(d.id))
        .map(|d| d.label.to_lowercase().replace(' ', ""))
        .collect();
    if tags.is_empty() {
        tags.push("residue".into());
    }
    let base = format!("pascal_m{}_rows{}-{}_{}_{}", m, rows.0, rows.1, scheme.id(), tags.join("+"));
    let mut path = dir.join(format!("{base}.png"));
    let mut i = 2;
    while path.exists() {
        path = dir.join(format!("{base}_{i}.png"));
        i += 1;
    }
    path
}
