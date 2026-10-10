//! GPU-vs-CPU verification. Renders regions of the triangle with the real shader
//! pipeline on whatever adapter is available (a software Vulkan/GL device in CI),
//! reads the results back and compares them cell for cell with the plain additive
//! rule. Everything must match exactly.
//!
//! Run with `cargo test --test gpu_verify` (or plain `cargo test`).

use pascal::gpu::{FrameInput, Gpu, Renderer};
use pascal::layers::{AvgMode, BlendMode, FieldColorMode, LayerId, LayerState};
use pascal::math::{carries, triangle_dp, Modulus};
use pascal::palette::{Palette, Scheme};
use pascal::view::{GridSpec, View, ROW_H};

fn device() -> (Gpu, Renderer) {
    let gpu = Gpu::new_headless().expect("a GPU adapter (install a software Vulkan driver such as Mesa lavapipe for headless runs)");
    eprintln!("adapter: {} ({:?}, {:?})", gpu.info.name, gpu.info.device_type, gpu.info.backend);
    let r = Renderer::new(gpu.device.clone(), gpu.queue.clone());
    (gpu, r)
}

/// A view in which every texel is exactly one cell and rows `0..rows` are all visible.
fn cell_grid(rows: u32) -> (View, GridSpec) {
    let view = View { x0: -(rows as f64) / 2.0 - 1.0, y0: 0.0, ppc: 2.0 };
    let canvas_w = rows * 2 + 4;
    let canvas_h = (rows as f64 * ROW_H * 2.0).ceil() as u32 + 2;
    let grid = GridSpec::new(&view, canvas_w, canvas_h, 1, 1 << 26);
    assert_eq!((grid.sx, grid.sy, grid.ss), (1.0, 1.0, 1));
    (view, grid)
}

fn frame<'a>(grid: GridSpec, md: &'a Modulus, pal: &'a Palette, layers: &'a LayerState) -> FrameInput<'a> {
    FrameInput { grid, modulus: md, palette: pal, layers, advance: true }
}

const MODULI: &[u32] = &[
    2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 25, 27, 30, 32, 49, 60, 64, 81, 125, 128, 210, 243, 256, 343, 360, 1024, 2310, 4096, 65535, 65536, 65537, 99991, 524288, 999983, 999999, 1_000_000,
];

#[test]
fn residues_match_additive_rule() {
    let (_gpu, mut r) = device();
    let pal = Palette::new(Scheme::Spectral);
    let layers = LayerState::default();
    let rows = 160u32;
    let (_, grid) = cell_grid(rows);
    for &m in MODULI {
        let md = Modulus::new(m);
        let dp = triangle_dp(rows as usize, m);
        r.render_blocking(&frame(grid, &md, &pal, &layers));
        let cells = r.read_cells();
        let mut seen = 0usize;
        for j in 0..grid.h {
            for i in 0..grid.w {
                let c = cells[(j * grid.w + i) as usize];
                let valid = c[1] & 0x8000_0000 != 0;
                match grid.cell_of(i, j) {
                    Some((n, k)) if n < rows => {
                        assert!(valid, "m={m}: texel ({i},{j}) = cell ({n},{k}) should be valid");
                        let want = dp[(n * (n + 1) / 2 + k) as usize];
                        assert_eq!(c[0], want, "m={m}: C({n},{k}) mod {m}");
                        // packed carries for the smallest prime and the per-factor divisibility mask
                        let nu = c[1] & 0xff;
                        assert_eq!(nu, carries(n, k, md.p).min(255), "m={m}: nu_{} C({n},{k})", md.p);
                        let mask = (c[1] >> 8) & 0xffff;
                        for (f, t) in md.factors.iter().enumerate() {
                            let divides = carries(n, k, t.f.p) >= t.f.e;
                            assert_eq!((mask >> f) & 1 == 1, divides, "m={m}: {}^{} | C({n},{k})", t.f.p, t.f.e);
                        }
                        seen += 1;
                    }
                    Some(_) => {}
                    None => assert!(!valid, "m={m}: texel ({i},{j}) is outside the triangle"),
                }
            }
        }
        assert_eq!(seen, (rows * (rows + 1) / 2) as usize, "m={m}: every cell of rows 0..{rows} is in the grid");
    }
}

#[test]
fn deep_rows_match_cpu_closed_form() {
    // Far beyond the reach of the additive rule: compare the shader with the CPU
    // Granville/CRT implementation (itself checked against the additive rule above).
    let (_gpu, mut r) = device();
    let pal = Palette::new(Scheme::Spectral);
    let layers = LayerState::default();
    let rows = 96u32;
    for &m in &[2u32, 3, 7, 8, 9, 12, 30, 360, 65536, 65537, 99991, 1_000_000] {
        let md = Modulus::new(m);
        for &(n0, left) in &[(1_000_000u32, true), (123_456_789, false), (999_999_900, true), (999_999_900, false)] {
            // left edge of the triangle, or its middle
            let x0 = if left { -(n0 as f64) / 2.0 - 2.0 } else { -40.0 };
            let view = View { x0, y0: n0 as f64, ppc: 2.0 };
            let grid = GridSpec::new(&view, rows * 2, (rows as f64 * ROW_H * 2.0).ceil() as u32 + 2, 1, 1 << 26);
            assert_eq!((grid.sx, grid.sy), (1.0, 1.0));
            r.render_blocking(&frame(grid, &md, &pal, &layers));
            let cells = r.read_cells();
            let mut seen = 0;
            for j in 0..grid.h {
                for i in 0..grid.w {
                    let c = cells[(j * grid.w + i) as usize];
                    if let Some((n, k)) = grid.cell_of(i, j) {
                        assert!(c[1] & 0x8000_0000 != 0);
                        assert_eq!(c[0], md.binom(n, k), "m={m}: C({n},{k})");
                        assert_eq!(c[1] & 0xff, carries(n, k, md.p).min(255));
                        seen += 1;
                    } else {
                        assert!(c[1] & 0x8000_0000 == 0);
                    }
                }
            }
            assert!(seen > rows * 10, "m={m}: region at row {n0} should contain cells");
        }
    }
}

// ---------------------------------------------------------------------------------
// CPU twins of the field passes, on the same grid, for the layer pipeline tests.
// ---------------------------------------------------------------------------------

struct CpuGrid {
    grid: GridSpec,
    md: Modulus,
    /// residue per texel (None outside the triangle)
    cells: Vec<Option<(u32, u32, u32)>>, // (n, k, residue)
}

impl CpuGrid {
    fn new(grid: GridSpec, m: u32) -> Self {
        let md = Modulus::new(m);
        let mut cells = Vec::with_capacity(grid.texels() as usize);
        for j in 0..grid.h {
            for i in 0..grid.w {
                cells.push(grid.cell_of(i, j).map(|(n, k)| (n, k, md.binom(n, k))));
            }
        }
        Self { grid, md, cells }
    }

    fn neighbor(&self, i: i32, j: i32, which: u32) -> Option<usize> {
        let (w, h) = (self.grid.w as i32, self.grid.h as i32);
        let row = |jj: i32| -> Option<u32> { self.cells[(jj * w) as usize..((jj + 1) * w) as usize].iter().flatten().next().map(|c| c.0) };
        let k0 = |n: u32| self.grid.k0(n);
        let (ni, nj) = match which {
            0 | 1 => {
                let nj = j - 1;
                if nj < 0 {
                    return None;
                }
                let (n, nup) = (row(j)?, row(nj)?);
                let d = (k0(n) - k0(nup)) as i32;
                (if which == 0 { i + d - 1 } else { i + d }, nj)
            }
            2 | 3 => {
                let nj = j + 1;
                if nj >= h {
                    return None;
                }
                let (n, ndn) = (row(j)?, row(nj)?);
                let d = (k0(ndn) - k0(n)) as i32;
                (if which == 2 { i - d } else { i - d + 1 }, nj)
            }
            4 => (i - 1, j),
            _ => (i + 1, j),
        };
        if ni < 0 || ni >= w {
            return None;
        }
        let idx = (nj * w + ni) as usize;
        self.cells[idx].map(|_| idx)
    }

    fn hex_blur(&self, src: &[f32], passes: u32) -> Vec<f32> {
        let mut cur = src.to_vec();
        for _ in 0..passes {
            let mut next = cur.clone();
            for j in 0..self.grid.h as i32 {
                for i in 0..self.grid.w as i32 {
                    let idx = (j * self.grid.w as i32 + i) as usize;
                    if self.cells[idx].is_none() {
                        continue;
                    }
                    let mut s = cur[idx];
                    let mut cnt = 1.0f32;
                    for which in 0..6 {
                        if let Some(ni) = self.neighbor(i, j, which) {
                            s += cur[ni];
                            cnt += 1.0;
                        }
                    }
                    next[idx] = s / cnt;
                }
            }
            cur = next;
        }
        cur
    }

    fn measure(&self, id: LayerId, layers: &LayerState) -> Vec<f32> {
        let p = self.md.p;
        let raw: Vec<f32> = self
            .cells
            .iter()
            .map(|c| match *c {
                None => 0.0,
                Some((n, k, v)) => match id {
                    LayerId::Kummer => carries(n, k, p) as f32,
                    LayerId::Lucas => {
                        let l = layers.param("lucasScale") as u32;
                        let pl = (p as u64).checked_pow(l).filter(|&x| x <= u32::MAX as u64);
                        let (nd, kd) = pl.map(|pl| ((n as u64 / pl % p as u64) as u32, (k as u64 / pl % p as u64) as u32)).unwrap_or((0, 0));
                        if kd > nd {
                            0.0
                        } else {
                            (nd * p + kd + 1) as f32
                        }
                    }
                    LayerId::Crt => {
                        let mut mask = 0u32;
                        for (f, t) in self.md.factors.iter().enumerate() {
                            if carries(n, k, t.f.p) >= t.f.e {
                                mask |= 1 << f;
                            }
                        }
                        mask as f32
                    }
                    LayerId::Dlog => {
                        if p == 2 {
                            (v & 1) as f32
                        } else {
                            let vp = v % p;
                            if vp == 0 {
                                0.0
                            } else {
                                (self.md.dlog[vp as usize] + 1) as f32
                            }
                        }
                    }
                    LayerId::PDigit => {
                        let jd = layers.param("padicDigit") as u32;
                        match (p as u64).checked_pow(jd).filter(|&x| x <= u32::MAX as u64) {
                            Some(pj) => ((v as u64 / pj) % p as u64) as f32,
                            None => 0.0,
                        }
                    }
                    LayerId::Density => (v != 0) as u32 as f32,
                    _ => unreachable!(),
                },
            })
            .collect();
        if id == LayerId::Density {
            self.hex_blur(&raw, layers.param("densityR") as u32)
        } else {
            raw
        }
    }

    fn blend(&self, layers: &LayerState) -> Vec<f32> {
        let plan = layers.plan();
        let denom = (self.md.m as f32 - 1.0).max(1.0);
        let fields: Vec<(Vec<f32>, f32)> = plan
            .measures
            .iter()
            .map(|d| {
                let f = self.measure(d.id, layers);
                let mx = f.iter().cloned().fold(0.0f32, f32::max);
                (f, if mx == 0.0 { 1.0 } else { mx })
            })
            .collect();
        (0..self.cells.len())
            .map(|idx| {
                let Some((_, _, v)) = self.cells[idx] else { return 0.0 };
                if fields.is_empty() {
                    return v as f32 / denom;
                }
                let fold = |acc: f32, x: f32| match layers.blend {
                    BlendMode::Average => acc + x,
                    BlendMode::Multiply => acc * x,
                    BlendMode::Max => acc.max(x),
                    BlendMode::Screen => 1.0 - (1.0 - acc) * (1.0 - x),
                };
                let mut acc = if layers.blend == BlendMode::Multiply { 1.0 } else { 0.0 };
                let mut nl = 0.0;
                for (f, mx) in &fields {
                    acc = fold(acc, f[idx] * (1.0 / mx));
                    nl += 1.0;
                }
                if plan.wants_value {
                    acc = fold(acc, v as f32 / denom);
                    nl += 1.0;
                }
                if layers.blend == BlendMode::Average {
                    acc / nl
                } else {
                    acc
                }
            })
            .collect()
    }
}

fn assert_close(gpu: &[f32], cpu: &[f32], tol: f32, what: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{what}: length");
    let mut worst = 0.0f32;
    for (i, (a, b)) in gpu.iter().zip(cpu).enumerate() {
        let d = (a - b).abs();
        assert!(d <= tol, "{what}: texel {i}: gpu {a} vs cpu {b}");
        worst = worst.max(d);
    }
    eprintln!("{what}: max |gpu − cpu| = {worst:e}");
}

#[test]
fn measures_match_cpu() {
    let (_gpu, mut r) = device();
    let pal = Palette::new(Scheme::Viridis);
    let rows = 100u32;
    let (_, grid) = cell_grid(rows);
    for &(m, id) in &[
        (2u32, LayerId::Kummer),
        (3, LayerId::Kummer),
        (30, LayerId::Kummer),
        (5, LayerId::Lucas),
        (12, LayerId::Crt),
        (360, LayerId::Crt),
        (7, LayerId::Dlog),
        (2, LayerId::Dlog),
        (49, LayerId::Dlog),
        (25, LayerId::PDigit),
        (27, LayerId::PDigit),
        (3, LayerId::Density),
        (6, LayerId::Density),
    ] {
        let cg = CpuGrid::new(grid, m);
        let mut layers = LayerState::default();
        layers.set_active(LayerId::Value, false);
        layers.set_active(id, true);
        if id == LayerId::Lucas {
            layers.params.insert("lucasScale", 1);
        }
        r.render_blocking(&frame(grid, &cg.md, &pal, &layers));
        let got = r.read_field(0);
        let want = cg.measure(id, &layers);
        let tol = if id == LayerId::Density { 1e-5 } else { 0.0 };
        assert_close(&got, &want, tol, &format!("m={m} {id:?}"));
        let mx = r.read_max()[0];
        assert!((mx - want.iter().cloned().fold(0.0, f32::max)).abs() <= tol, "m={m} {id:?}: max");
    }
}

#[test]
fn blend_modes_match_cpu() {
    let (_gpu, mut r) = device();
    let pal = Palette::new(Scheme::Viridis);
    let (_, grid) = cell_grid(80);
    let cg = CpuGrid::new(grid, 30);
    for blend in BlendMode::ALL {
        for wants_value in [false, true] {
            let mut layers = LayerState::default();
            layers.set_active(LayerId::Value, wants_value);
            layers.set_active(LayerId::Kummer, true);
            layers.set_active(LayerId::Crt, true);
            layers.set_active(LayerId::Density, true);
            layers.blend = blend;
            r.render_blocking(&frame(grid, &cg.md, &pal, &layers));
            assert_close(&r.read_work(), &cg.blend(&layers), 2e-6, &format!("{blend:?} value={wants_value}"));
        }
    }
}

#[test]
fn transforms_match_cpu() {
    let (_gpu, mut r) = device();
    let pal = Palette::new(Scheme::CoolWarm);
    let (_, grid) = cell_grid(80);
    let cg = CpuGrid::new(grid, 6);
    // AVERAGE (spatial, 3 passes) on top of DENSITY
    let mut layers = LayerState::default();
    layers.set_active(LayerId::Value, false);
    layers.set_active(LayerId::Density, true);
    layers.set_active(LayerId::Average, true);
    layers.params.insert("avgAmount", 3);
    r.render_blocking(&frame(grid, &cg.md, &pal, &layers));
    let under = cg.blend(&layers);
    assert_close(&r.read_work(), &cg.hex_blur(&under, 3), 2e-6, "average ×3");

    // BAND PASS keep 1 / cut 4 on the same field, after AVERAGE
    layers.set_active(LayerId::BandPass, true);
    layers.params.insert("bandLo", 1);
    layers.params.insert("bandHi", 4);
    r.render_blocking(&frame(grid, &cg.md, &pal, &layers));
    let cur = cg.hex_blur(&under, 3);
    let lo = cg.hex_blur(&cur, 1);
    let hi = cg.hex_blur(&cur, 4);
    let d: Vec<f32> = lo.iter().zip(&hi).map(|(a, b)| a - b).collect();
    let mx = d.iter().zip(&cg.cells).filter(|(_, c)| c.is_some()).map(|(x, _)| x.abs()).fold(0.0f32, f32::max);
    let want: Vec<f32> = d.iter().map(|x| 0.5 + 0.5 * x / mx).collect();
    assert_close(&r.read_work(), &want, 2e-5, "band pass");

    // AFTERIMAGE: first frame ghosts the neutral baseline, the second the adapted signal
    let mut layers = LayerState::default();
    layers.set_active(LayerId::Value, false);
    layers.set_active(LayerId::Kummer, true);
    layers.set_active(LayerId::Afterimage, true);
    layers.params.insert("aiStrength", 50);
    layers.params.insert("aiAttack", 2);
    layers.params.insert("aiDecay", 4);
    r.reset_history();
    r.render_blocking(&frame(grid, &cg.md, &pal, &layers));
    let f = cg.blend(&layers);
    let want1: Vec<f32> = f.iter().map(|&c| c * 0.5 + 0.5 * 0.5).collect();
    assert_close(&r.read_work(), &want1, 2e-6, "afterimage frame 1");
    let a1: Vec<f32> = f.iter().map(|&c| 0.5 + (c - 0.5) * if c > 0.5 { 0.5 } else { 0.25 }).collect();
    r.render_blocking(&frame(grid, &cg.md, &pal, &layers));
    let want2: Vec<f32> = f.iter().zip(&a1).map(|(&c, &a)| c * 0.5 + (1.0 - a) * 0.5).collect();
    assert_close(&r.read_work(), &want2, 2e-6, "afterimage frame 2");

    // temporal AVERAGE over 3 frames: identical frames average to themselves
    let mut layers = LayerState::default();
    layers.set_active(LayerId::Value, false);
    layers.set_active(LayerId::Kummer, true);
    layers.set_active(LayerId::Average, true);
    layers.avg_mode = AvgMode::Temporal;
    layers.params.insert("avgAmount", 3);
    r.reset_history();
    for _ in 0..3 {
        r.render_blocking(&frame(grid, &cg.md, &pal, &layers));
    }
    assert_close(&r.read_work(), &cg.blend(&layers), 2e-6, "temporal average of identical frames");
}

#[test]
fn resolve_colors_match_palette_tables() {
    let (_gpu, mut r) = device();
    let rows = 60u32;
    let (view, grid) = cell_grid(rows);
    for scheme in [Scheme::Spectral, Scheme::Turbo, Scheme::Mono] {
        let pal = Palette::new(scheme);
        for &m in &[2u32, 7, 12, 1000] {
            let md = Modulus::new(m);
            let layers = LayerState::default();
            r.render_blocking(&frame(grid, &md, &pal, &layers));
            let (w, h, rgba) = r.read_output_rgba();
            assert_eq!((w, h), (grid.canvas_w, grid.canvas_h));
            let mut checked = 0;
            for py in 0..h {
                for px in 0..w {
                    let want = match view.cell_at(px as f64 + 0.5, py as f64 + 0.5) {
                        Some((n, k)) => pal.residue_color(md.binom(n, k), m),
                        None => pascal::palette::BG,
                    };
                    let o = ((py * w + px) * 4) as usize;
                    assert_eq!(&rgba[o..o + 3], &want, "{scheme:?} m={m} pixel ({px},{py})");
                    checked += 1;
                }
            }
            assert!(checked > 0);
        }
    }
    // math ramp: KUMMER through the ramp, and palette-blend colors are in range
    let pal = Palette::new(Scheme::Magma);
    let md = Modulus::new(8);
    let mut layers = LayerState::default();
    layers.set_active(LayerId::Value, false);
    layers.set_active(LayerId::Kummer, true);
    r.render_blocking(&frame(grid, &md, &pal, &layers));
    let field = r.read_work();
    let (w, _, rgba) = r.read_output_rgba();
    for py in 0..grid.canvas_h {
        for px in 0..w {
            if let Some((n, k)) = view.cell_at(px as f64 + 0.5, py as f64 + 0.5) {
                if n >= rows {
                    continue;
                }
                let i = (k as i64 - grid.k0(n)) as u32;
                let v = field[(n * grid.w + i) as usize].clamp(0.0, 1.0);
                let lvl = (v * 255.0 + 0.5).floor() as usize;
                let o = ((py * w + px) * 4) as usize;
                assert_eq!(&rgba[o..o + 3], &pal.ramp[lvl], "ramp pixel ({px},{py})");
            }
        }
    }
    layers.color_mode = FieldColorMode::PaletteBlend;
    layers.set_active(LayerId::Average, true);
    r.render_blocking(&frame(grid, &md, &pal, &layers));
    let (_, _, rgba2) = r.read_output_rgba();
    assert!(rgba2.chunks(4).any(|c| c[..3] != pascal::palette::BG), "palette blend paints something");
}
