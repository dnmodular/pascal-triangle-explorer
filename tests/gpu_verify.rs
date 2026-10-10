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
    FrameInput { grid, modulus: md, palette: pal, layers, advance: true, tween: pascal::gpu::Tween::Off, style: Default::default() }
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

    fn residue(&self, n: u32, k: u32) -> u32 {
        if k > n { 0 } else { self.md.binom(n, k) }
    }

    fn wrap(&self, a: u32, b: u32) -> u32 {
        let d = a.abs_diff(b);
        d.min(self.md.m - d)
    }

    fn per_cell(&self, f: impl Fn(u32, u32, u32) -> f32) -> Vec<f32> {
        self.cells.iter().map(|c| c.map(|(n, k, v)| f(n, k, v)).unwrap_or(0.0)).collect()
    }

    /// Breadth-first lattice distance to the nearest zero over the grid's neighbour
    /// graph, saturated at `cap`; 0 everywhere when the view holds no zero.
    fn dist_to_zero(&self) -> Vec<f32> {
        let (w, h) = (self.grid.w as i32, self.grid.h as i32);
        let cap = 256u32.min(self.grid.w.max(self.grid.h));
        let mut dist = vec![u32::MAX; self.cells.len()];
        let mut queue = std::collections::VecDeque::new();
        for (i, c) in self.cells.iter().enumerate() {
            if matches!(c, Some((_, _, 0))) {
                dist[i] = 0;
                queue.push_back(i);
            }
        }
        if queue.is_empty() {
            return vec![0.0; self.cells.len()];
        }
        while let Some(i) = queue.pop_front() {
            let (ii, jj) = ((i as i32) % w, (i as i32) / w);
            let _ = h;
            for which in 0..6 {
                if let Some(ni) = self.neighbor(ii, jj, which) {
                    if dist[ni] == u32::MAX {
                        dist[ni] = dist[i] + 1;
                        queue.push_back(ni);
                    }
                }
            }
        }
        self.cells.iter().zip(&dist).map(|(c, &d)| if c.is_some() { d.min(cap) as f32 } else { 0.0 }).collect()
    }

    /// ln C(n,k) in f64 (sum of logs below 4096, Stirling above), clamped at 0.
    fn ln_binom(n: u32, k: u32) -> f64 {
        fn lnf(x: u32) -> f64 {
            if x < 4096 {
                (2..=x).map(|i| (i as f64).ln()).sum()
            } else {
                let x = x as f64;
                (x + 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * x) - 1.0 / (360.0 * x * x * x)
            }
        }
        let j = k.min(n - k);
        (lnf(n) - lnf(j) - lnf(n - j)).max(0.0)
    }

    /// The reference's STRIPES DP (forward chain F, backward chain G, value F[i] +
    /// G[pred]) on the additive triangle of `rows_total` rows. Returns (value, F, G)
    /// keyed like the grid.
    fn stripes_reference(&self, period: u32, dir: i32, tol: u32, rows_total: usize) -> (Vec<f32>, u32) {
        let (dr, dc) = pascal::layers::direction_step(dir);
        let m = self.md.m;
        let tri = triangle_dp(rows_total, m);
        let at = |r: usize, c: usize| tri[r * (r + 1) / 2 + c];
        let wrap = |a: u32, b: u32| {
            let d = a.abs_diff(b);
            d.min(m - d)
        };
        let n_tot = rows_total;
        let total = n_tot * (n_tot + 1) / 2;
        let (sr, sc) = ((dr * period) as usize, (dc * period) as usize);
        let mut cflag = vec![false; total];
        for r in 0..n_tot {
            let (r1, r2) = (r + sr, r + 2 * sr);
            for c in 0..=r {
                let (c1, c2) = (c + sc, c + 2 * sc);
                if r1 >= n_tot || c1 > r1 || r2 >= n_tot || c2 > r2 {
                    continue;
                }
                let (v0, vp, v2) = (at(r, c), at(r1, c1), at(r2, c2));
                if wrap(v0, v2) <= tol && wrap(v0, vp) > tol {
                    cflag[r * (r + 1) / 2 + c] = true;
                }
            }
        }
        let mut f = vec![0u32; total];
        for r in (0..n_tot).rev() {
            for c in (0..=r).rev() {
                let i = r * (r + 1) / 2 + c;
                if !cflag[i] {
                    continue;
                }
                let (r1, c1) = (r + sr, c + sc);
                f[i] = 1 + if r1 < n_tot && c1 <= r1 { f[r1 * (r1 + 1) / 2 + c1] } else { 0 };
            }
        }
        let mut g = vec![0u32; total];
        for r in 0..n_tot {
            for c in 0..=r {
                let i = r * (r + 1) / 2 + c;
                if !cflag[i] {
                    continue;
                }
                let ok = r >= sr && c >= sc && c - sc <= r - sr;
                g[i] = 1 + if ok { g[(r - sr) * (r - sr + 1) / 2 + c - sc] } else { 0 };
            }
        }
        let mut longest = 0;
        let vals = self
            .cells
            .iter()
            .map(|cell| {
                let Some((n, k, _)) = *cell else { return 0.0 };
                let (r, c) = (n as usize, k as usize);
                let i = r * (r + 1) / 2 + c;
                let ok = r >= sr && c >= sc && c - sc <= r - sr;
                let gb = if ok { g[(r - sr) * (r - sr + 1) / 2 + c - sc] } else { 0 };
                longest = longest.max(f[i]).max(gb);
                (f[i] + gb) as f32
            })
            .collect();
        (vals, longest)
    }

    fn measure(&self, id: LayerId, layers: &LayerState) -> Vec<f32> {
        let p = self.md.p;
        let m = self.md.m;
        match id {
            LayerId::Gradient => {
                return self.per_cell(|n, k, v| {
                    let mut g = 0;
                    if n > 0 {
                        if k > 0 {
                            g += self.wrap(v, self.residue(n - 1, k - 1));
                        }
                        if k < n {
                            g += self.wrap(v, self.residue(n - 1, k));
                        }
                    }
                    g as f32
                })
            }
            LayerId::LagDiff => {
                let (dr, dc) = pascal::layers::direction_step(layers.param("lagDir"));
                let lag = layers.param("lagLen") as u32;
                return self.per_cell(|n, k, v| {
                    let (n2, k2) = (n + lag * dr, k + lag * dc);
                    if k2 <= n2 { self.wrap(v, self.residue(n2, k2)) as f32 } else { 0.0 }
                });
            }
            LayerId::Magnitude => return self.per_cell(|n, k, _| Self::ln_binom(n, k) as f32),
            LayerId::DistZero => return self.dist_to_zero(),
            LayerId::Primes => {
                let t = pascal::sweep::prime_tables();
                let mask = layers.param("primesMode") == 1;
                return self.per_cell(|_, _, v| {
                    let r = t.rank(v);
                    if mask { (r > 0) as u32 as f32 } else { r as f32 }
                });
            }
            LayerId::Rarity => {
                let mut counts = std::collections::HashMap::new();
                let mut total = 0u32;
                for (_, _, v) in self.cells.iter().flatten() {
                    *counts.entry(*v).or_insert(0u32) += 1;
                    total += 1;
                }
                return self.per_cell(|_, _, v| ((total as f64).log2() - (counts[&v] as f64).log2()).max(0.0) as f32);
            }
            LayerId::Variance => {
                // same f32 bits as the shader's TAU literal 6.2831853
                let ang = |v: u32| std::f32::consts::TAU * v as f32 / m as f32;
                let r = layers.param("varR") as u32;
                let bc = self.hex_blur(&self.per_cell(|_, _, v| ang(v).cos()), r);
                let bs = self.hex_blur(&self.per_cell(|_, _, v| ang(v).sin()), r);
                return (0..self.cells.len()).map(|i| if self.cells[i].is_some() { (1.0 - (bc[i] * bc[i] + bs[i] * bs[i]).sqrt()).max(0.0) } else { 0.0 }).collect();
            }
            LayerId::Moire => {
                let q = self.md.factors.get(1).map(|t| t.f.p);
                let r = layers.param("moireR") as u32;
                let Some(q) = q else { return vec![0.0; self.cells.len()] };
                let zp = self.per_cell(|_, _, v| (v % p == 0) as u32 as f32);
                let zq = self.per_cell(|_, _, v| (v % q == 0) as u32 as f32);
                let zpq = self.per_cell(|_, _, v| (v % p == 0 && v % q == 0) as u32 as f32);
                let (cp, cq, cpq) = (self.hex_blur(&zp, r), self.hex_blur(&zq, r), self.hex_blur(&zpq, r));
                return (0..self.cells.len()).map(|i| if self.cells[i].is_some() { cpq[i] - cp[i] * cq[i] } else { 0.0 }).collect();
            }
            _ => {}
        }
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
        let fields: Vec<(Vec<f32>, f32, bool)> = plan
            .measures
            .iter()
            .map(|d| {
                let f = self.measure(d.id, layers);
                let mx = f.iter().map(|x| if d.signed { x.abs() } else { *x }).fold(0.0f32, f32::max);
                (f, if mx == 0.0 { 1.0 } else { mx }, d.signed)
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
                for (f, mx, signed) in &fields {
                    let raw = f[idx] * (1.0 / mx);
                    acc = fold(acc, if *signed { 0.5 + 0.5 * raw.clamp(-1.0, 1.0) } else { raw });
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
    for &(m, id, key, val) in &[
        (2u32, LayerId::Kummer, "", 0),
        (3, LayerId::Kummer, "", 0),
        (30, LayerId::Kummer, "", 0),
        (5, LayerId::Lucas, "lucasScale", 1),
        (5, LayerId::Lucas, "lucasScale", 0),
        (3, LayerId::Lucas, "lucasScale", 3),
        (7, LayerId::Lucas, "lucasScale", 12), // 7^12 overflows u32: every cell reads 1
        (12, LayerId::Crt, "", 0),
        (360, LayerId::Crt, "", 0),
        (7, LayerId::Dlog, "", 0),
        (2, LayerId::Dlog, "", 0),
        (49, LayerId::Dlog, "", 0),
        (25, LayerId::PDigit, "padicDigit", 1),
        (27, LayerId::PDigit, "padicDigit", 2),
        (8, LayerId::PDigit, "padicDigit", 0),
        (5, LayerId::PDigit, "padicDigit", 12), // 5^12 fits: digit 12 of residues < 25 is 0
        (2, LayerId::PDigit, "padicDigit", 12),
        (7, LayerId::PDigit, "padicDigit", 12), // 7^12 overflows u32: every cell reads 0
        (3, LayerId::Density, "", 0),
        (6, LayerId::Density, "", 0),
    ] {
        let cg = CpuGrid::new(grid, m);
        let mut layers = LayerState::default();
        layers.set_active(LayerId::Value, false);
        layers.set_active(id, true);
        if !key.is_empty() {
            layers.params.insert(key, val);
        }
        r.render_blocking(&frame(grid, &cg.md, &pal, &layers));
        let got = r.read_field(0);
        let want = cg.measure(id, &layers);
        let tol = if id == LayerId::Density { 1e-5 } else { 0.0 };
        assert_close(&got, &want, tol, &format!("m={m} {id:?} {key}={val}"));
        let mx = r.read_max()[0];
        assert!((mx - want.iter().cloned().fold(0.0, f32::max)).abs() <= tol, "m={m} {id:?} {key}={val}: max");
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

fn tween_frame<'a>(grid: GridSpec, md: &'a Modulus, pal: &'a Palette, layers: &'a LayerState, tween: pascal::gpu::Tween) -> FrameInput<'a> {
    FrameInput { grid, modulus: md, palette: pal, layers, advance: false, tween, style: Default::default() }
}

fn assert_rgba_close(a: &[u8], b: &[u8], tol: u8, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: size");
    let worst = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
    assert!(worst <= tol, "{what}: worst channel difference {worst} > {tol}");
}

#[test]
fn tween_is_display_only_and_lands_on_the_true_frame() {
    use pascal::gpu::Tween;
    let (_gpu, mut r) = device();
    let pal = Palette::new(Scheme::Spectral);
    let (_, grid) = cell_grid(90);
    let (m7, m11) = (Modulus::new(7), Modulus::new(11));
    for (what, field) in [("dissolve (VALUE)", false), ("morph (KUMMER)", true)] {
        let mut layers = LayerState::default();
        if field {
            layers.set_active(LayerId::Value, false);
            layers.set_active(LayerId::Kummer, true);
        }
        // the true frames
        r.render_blocking(&tween_frame(grid, &m11, &pal, &layers, Tween::Off));
        let (_, _, true11) = r.read_output_rgba();
        r.render_blocking(&tween_frame(grid, &m7, &pal, &layers, Tween::Off));
        let (_, _, shown7) = r.read_display_rgba();
        // start a transition 7 -> 11: e = 0 still shows 7
        r.render_blocking(&tween_frame(grid, &m11, &pal, &layers, Tween::Start(0.0)));
        let (_, _, d0) = r.read_display_rgba();
        assert_rgba_close(&d0, &shown7, 0, &format!("{what}: e = 0 shows the previous frame"));
        // halfway is between the two
        r.render_blocking(&tween_frame(grid, &m11, &pal, &layers, Tween::Continue(0.5)));
        let (_, _, dh) = r.read_display_rgba();
        assert!(dh != shown7 && dh != true11, "{what}: e = 0.5 is a blend");
        // e = 1 is exactly the true frame
        r.render_blocking(&tween_frame(grid, &m11, &pal, &layers, Tween::Continue(1.0)));
        let (_, _, d1) = r.read_display_rgba();
        assert_rgba_close(&d1, &true11, 0, &format!("{what}: e = 1 is the true frame"));
        // an Off render is always the true frame, in both textures
        r.render_blocking(&tween_frame(grid, &m11, &pal, &layers, Tween::Off));
        let (_, _, f) = r.read_output_rgba();
        assert_rgba_close(&f, &true11, 0, &format!("{what}: Off output"));
    }
    // a retarget mid-transition starts from what is on screen
    let layers = LayerState::default();
    let m13 = Modulus::new(13);
    r.render_blocking(&tween_frame(grid, &m7, &pal, &layers, Tween::Off));
    r.render_blocking(&tween_frame(grid, &m11, &pal, &layers, Tween::Start(0.0)));
    r.render_blocking(&tween_frame(grid, &m11, &pal, &layers, Tween::Continue(0.5)));
    let (_, _, mid) = r.read_display_rgba();
    r.render_blocking(&tween_frame(grid, &m13, &pal, &layers, Tween::Start(0.0)));
    let (_, _, d0) = r.read_display_rgba();
    assert_rgba_close(&d0, &mid, 0, "retarget starts from the blended picture");
    // the same for a morph
    let mut layers = LayerState::default();
    layers.set_active(LayerId::Value, false);
    layers.set_active(LayerId::Kummer, true);
    r.render_blocking(&tween_frame(grid, &m7, &pal, &layers, Tween::Off));
    r.render_blocking(&tween_frame(grid, &m11, &pal, &layers, Tween::Start(0.0)));
    r.render_blocking(&tween_frame(grid, &m11, &pal, &layers, Tween::Continue(0.5)));
    let (_, _, mid) = r.read_display_rgba();
    r.render_blocking(&tween_frame(grid, &m13, &pal, &layers, Tween::Start(0.0)));
    let (_, _, d0) = r.read_display_rgba();
    assert_rgba_close(&d0, &mid, 0, "morph retarget starts from the blended field");
}

#[test]
fn ported_measures_match_cpu() {
    let (_gpu, mut r) = device();
    let pal = Palette::new(Scheme::Viridis);
    let rows = 100u32;
    let (_, grid) = cell_grid(rows);
    // (modulus, layer, parameters, tolerance)
    type Case = (u32, LayerId, &'static [(&'static str, i32)], f32);
    let cases: &[Case] = &[
        (7, LayerId::Gradient, &[], 0.0),
        (12, LayerId::Gradient, &[], 0.0),
        (65537, LayerId::Gradient, &[], 0.0),
        (2, LayerId::DistZero, &[], 0.0),
        (3, LayerId::DistZero, &[], 0.0),
        (30, LayerId::DistZero, &[], 0.0),
        (99991, LayerId::DistZero, &[], 0.0), // no zero in view: 0 everywhere
        (5, LayerId::Magnitude, &[], 2e-3),
        (5, LayerId::Variance, &[("varR", 4)], 2e-5),
        (30, LayerId::Variance, &[("varR", 1)], 2e-5),
        (7, LayerId::Rarity, &[], 2e-5),
        (360, LayerId::Rarity, &[], 2e-5),
        (6, LayerId::Moire, &[("moireR", 4)], 2e-6),
        (30, LayerId::Moire, &[("moireR", 2)], 2e-6),
        (9, LayerId::Moire, &[("moireR", 4)], 0.0), // prime power: flat
        (11, LayerId::Primes, &[("primesMode", 0)], 0.0),
        (100, LayerId::Primes, &[("primesMode", 1)], 0.0),
        (13, LayerId::LagDiff, &[("lagDir", 0), ("lagLen", 1)], 0.0),
        (13, LayerId::LagDiff, &[("lagDir", 1), ("lagLen", 7)], 0.0),
        (30, LayerId::LagDiff, &[("lagDir", 2), ("lagLen", 3)], 0.0),
    ];
    for &(m, id, params, tol) in cases {
        let cg = CpuGrid::new(grid, m);
        let mut layers = LayerState::default();
        layers.set_active(LayerId::Value, false);
        layers.set_active(id, true);
        for &(k, v) in params {
            layers.params.insert(k, v);
        }
        r.render_blocking(&frame(grid, &cg.md, &pal, &layers));
        let got = r.read_field(0);
        let want = cg.measure(id, &layers);
        assert_close(&got, &want, tol, &format!("m={m} {id:?} {params:?}"));
        let signed = pascal::layers::def(id).signed;
        let mx = r.read_max()[0];
        let want_mx = want.iter().map(|x| if signed { x.abs() } else { *x }).fold(0.0, f32::max);
        assert!((mx - want_mx).abs() <= tol.max(want_mx * 1e-5), "m={m} {id:?}: max {mx} vs {want_mx}");
    }
}

#[test]
fn stripes_match_reference_dp() {
    let (_gpu, mut r) = device();
    let pal = Palette::new(Scheme::Viridis);
    let rows = 90u32;
    let (_, grid) = cell_grid(rows);
    for &(m, dir, period, tol) in &[(7u32, 0i32, 1u32, 0u32), (7, 1, 1, 0), (7, 2, 1, 0), (199, 0, 1, 4), (199, 2, 2, 6), (12, 1, 2, 0), (3, 0, 3, 0)] {
        let cg = CpuGrid::new(grid, m);
        let mut layers = LayerState::default();
        layers.set_active(LayerId::Value, false);
        layers.set_active(LayerId::Stripes, true);
        layers.params.insert("stripeDir", dir);
        layers.params.insert("stripePeriod", period as i32);
        layers.params.insert("stripeTol", tol as i32);
        r.render_blocking(&frame(grid, &cg.md, &pal, &layers));
        let got = r.read_field(0);
        // margin rows so the reference's bottom edge cannot shorten any chain in view
        let margin = (2 * period as usize) * 260;
        let (want, longest) = cg.stripes_reference(period, dir, tol, rows as usize + margin + 4);
        assert!(longest < 256, "m={m}: a run of {longest} reaches the shader's 256-step walk");
        eprintln!("stripes m={m} dir={dir} P={period} tol={tol}: longest run {longest}");
        assert_close(&got, &want, 0.0, &format!("stripes m={m} dir={dir} P={period} tol={tol}"));
    }
}

#[test]
fn deep_row_measures_stay_exact() {
    // Closed-form measures far below the additive rule's reach.
    let (_gpu, mut r) = device();
    let pal = Palette::new(Scheme::Viridis);
    let rows = 64u32;
    for &(m, n0) in &[(7u32, 1_000_000u32), (30, 123_456_789)] {
        let view = View { x0: -40.0, y0: n0 as f64, ppc: 2.0 };
        let grid = GridSpec::new(&view, rows * 2, (rows as f64 * ROW_H * 2.0).ceil() as u32 + 2, 1, 1 << 26);
        let cg = CpuGrid::new(grid, m);
        for id in [LayerId::Gradient, LayerId::LagDiff, LayerId::Magnitude] {
            let mut layers = LayerState::default();
            layers.set_active(LayerId::Value, false);
            layers.set_active(id, true);
            layers.params.insert("lagLen", 5);
            r.render_blocking(&frame(grid, &cg.md, &pal, &layers));
            let got = r.read_field(0);
            let want = cg.measure(id, &layers);
            let tol = if id == LayerId::Magnitude { want.iter().cloned().fold(0.0, f32::max) * 2e-5 } else { 0.0 };
            assert_close(&got, &want, tol, &format!("deep m={m} n0={n0} {id:?}"));
        }
    }
}

#[test]
fn signed_measure_blends_around_half() {
    let (_gpu, mut r) = device();
    let pal = Palette::new(Scheme::CoolWarm);
    let (_, grid) = cell_grid(80);
    let cg = CpuGrid::new(grid, 30);
    for blend in BlendMode::ALL {
        let mut layers = LayerState::default();
        layers.set_active(LayerId::Value, false);
        layers.set_active(LayerId::Moire, true);
        layers.set_active(LayerId::Density, true);
        layers.blend = blend;
        r.render_blocking(&frame(grid, &cg.md, &pal, &layers));
        assert_close(&r.read_work(), &cg.blend(&layers), 2e-5, &format!("moiré + density, {blend:?}"));
    }
}

/// CPU twin of the resolve pass's color offset, outlines and glow at one pixel centre,
/// built from the view's own cell rectangles rather than the shader's neighbour table:
/// the glow is the distance to the nearest other zero cell anywhere within two rows and
/// three cells, the outline the coverage of a 1 px line on the cell's boundary. None
/// when the pixel centre sits on a cell boundary.
fn decorated_pixel(view: &View, md: &Modulus, pal: &Palette, style: &pascal::gpu::CanvasStyle, px: f64, py: f64) -> Option<[u8; 3]> {
    let map = pal.level_map(md.m, style.offset);
    let Some((n, k)) = view.cell_at(px, py) else { return Some(pascal::palette::BG) };
    let mut col = pal.residue_color_mapped(md.binom(n, k), md.m, &map).map(|c| c as f64 / 255.0);
    let (x, y, cw, ch) = view.cell_rect(n, k);
    let edge = (px - x).min(x + cw - px).min(py - y).min(y + ch - py);
    if edge < 1e-4 {
        return None; // on a cell boundary
    }
    let size = cw.min(ch);
    let fade = {
        let t = ((size - 3.0) / 3.0).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    if style.glow {
        let r = (10.0f64).min(0.8 * size);
        let mut d = f64::INFINITY;
        for nn in n as i64 - 2..=n as i64 + 2 {
            for kk in k as i64 - 3..=k as i64 + 3 {
                if nn < 0 || kk < 0 || kk > nn || (nn, kk) == (n as i64, k as i64) || md.binom(nn as u32, kk as u32) != 0 {
                    continue;
                }
                let (rx, ry, rw, rh) = view.cell_rect(nn as u32, kk as u32);
                let dx = (rx - px).max(px - (rx + rw)).max(0.0);
                let dy = (ry - py).max(py - (ry + rh)).max(0.0);
                d = d.min((dx * dx + dy * dy).sqrt());
            }
        }
        if d < r {
            let g = 1.0 - d / r;
            let a = 0.55 * g * g * fade;
            let glow = pal.glow_color().map(|c| c as f64 / 255.0);
            col = [0, 1, 2].map(|i| 1.0 - (1.0 - col[i]) * (1.0 - glow[i] * a));
        }
    }
    if style.outlines {
        let cover = (1.0 - edge).clamp(0.0, 1.0);
        let o = [235.0 / 255.0, 238.0 / 255.0, 250.0 / 255.0];
        let a = 0.12 * fade * cover;
        col = [0, 1, 2].map(|i| col[i] + (o[i] - col[i]) * a);
    }
    Some(col.map(|c| (c.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8))
}

#[test]
fn offset_outlines_and_glow_match_cpu() {
    use pascal::gpu::CanvasStyle;
    let (_gpu, mut r) = device();
    // big cells (12 px) at awkward offsets: the apex in view; the triangle crossing the
    // left edge; the top edge cutting through row 30 (zeros just off-canvas must glow)
    let views = [View { x0: -7.37, y0: -0.29, ppc: 12.0 }, View { x0: -3.95, y0: -0.29, ppc: 12.0 }, View { x0: 2.02, y0: 30.01, ppc: 12.0 }];
    let (w, h) = (180u32, 150u32);
    let layers = LayerState::default();
    let styles = [
        CanvasStyle { offset: 0.3, outlines: false, glow: false },
        CanvasStyle { offset: 0.0, outlines: true, glow: false },
        CanvasStyle { offset: 0.0, outlines: false, glow: true },
        CanvasStyle { offset: -0.62, outlines: true, glow: true },
    ];
    let mut checked = 0usize;
    let mut decorated = [0usize; 4];
    for view in views {
        let grid = GridSpec::new(&view, w, h, 1, 1 << 24);
        assert_eq!((grid.sx, grid.sy), (1.0, 1.0));
        for scheme in [Scheme::Spectral, Scheme::Ember] {
            let pal = Palette::new(scheme);
            for m in [2u32, 3, 7] {
                let md = Modulus::new(m);
                for (si, style) in styles.into_iter().enumerate() {
                    r.render_blocking(&FrameInput { grid, modulus: &md, palette: &pal, layers: &layers, advance: true, tween: pascal::gpu::Tween::Off, style });
                    let (_, _, rgba) = r.read_output_rgba();
                    for py in 0..h {
                        for px in 0..w {
                            let (cx, cy) = (px as f64 + 0.5, py as f64 + 0.5);
                            let Some(want) = decorated_pixel(&view, &md, &pal, &style, cx, cy) else { continue };
                            let o = ((py * w + px) * 4) as usize;
                            let got = &rgba[o..o + 3];
                            let diff = got.iter().zip(want).map(|(a, b)| a.abs_diff(b)).max().unwrap();
                            assert!(diff <= 1, "{view:?} {scheme:?} m={m} {style:?} pixel ({px},{py}): {got:?} vs {want:?}");
                            checked += 1;
                            if decorated_pixel(&view, &md, &pal, &CanvasStyle::default(), cx, cy).is_some_and(|p| p != want) {
                                decorated[si] += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 1_000_000, "{checked}");
    // every style visibly changed a good share of the pixels
    assert!(decorated.iter().all(|&d| d > 5_000), "pixels changed per style: {decorated:?}");
    eprintln!("checked {checked} pixels; changed per style {decorated:?}");
    // with tiny cells the decorations fade out entirely: same image as no style
    let rows = 60u32;
    let (_, grid) = cell_grid(rows);
    let pal = Palette::new(Scheme::Spectral);
    let md = Modulus::new(7);
    r.render_blocking(&FrameInput { grid, modulus: &md, palette: &pal, layers: &layers, advance: true, tween: pascal::gpu::Tween::Off, style: CanvasStyle::default() });
    let (_, _, plain) = r.read_output_rgba();
    r.render_blocking(&FrameInput { grid, modulus: &md, palette: &pal, layers: &layers, advance: true, tween: pascal::gpu::Tween::Off, style: CanvasStyle { offset: 0.0, outlines: true, glow: true } });
    let (_, _, styled) = r.read_output_rgba();
    assert_eq!(plain, styled, "2 px cells: no outlines or glow");
}

#[test]
fn svg_colors_match_the_gpu() {
    // the SVG export's CPU colors equal what the resolve pass paints, per preset, on a
    // fitted view that shows a few rows past the exported ones (m = 3: row 81 has
    // ν = 4 while rows 0..79 reach 3, so the valuation needs the view's max)
    use pascal::gpu::CanvasStyle;
    let (_gpu, mut r) = device();
    let rows = 80u32;
    let (w, h) = (1000u32, 800u32);
    let view = View::fit(w as f64, h as f64, rows as f64);
    let grid = GridSpec::new(&view, w, h, 1, 1 << 26);
    assert_eq!((grid.sx, grid.sy), (1.0, 1.0));
    let md = Modulus::new(3);
    for (scheme, offset) in [(Scheme::Viridis, 0.17f32), (Scheme::Spectral, -0.4)] {
        let pal = Palette::new(scheme);
        let style = CanvasStyle { offset, outlines: false, glow: false };
        let mut layers = LayerState::default();
        for preset in 0..3 {
            match preset {
                0 => layers.preset_residue(),
                1 => layers.preset_mask(),
                _ => layers.preset_valuation(),
            }
            r.render_blocking(&FrameInput { grid, modulus: &md, palette: &pal, layers: &layers, advance: true, tween: pascal::gpu::Tween::Off, style });
            let vmax = (preset == 2).then(|| r.read_max()[0]);
            let colors = pascal::export::preset_colors(rows, &md, &pal, &layers, &style, vmax).expect("a preset");
            let (_, _, rgba) = r.read_output_rgba();
            let mut checked = 0;
            for py in 0..h {
                for px in 0..w {
                    let (cx, cy) = (px as f64 + 0.5, py as f64 + 0.5);
                    let Some((n, k)) = view.cell_at(cx, cy) else { continue };
                    if n >= rows {
                        continue;
                    }
                    // a fitted view has fractional px/cell: skip centres within float noise
                    // of a cell boundary, where f32 (GPU) and f64 (here) may pick sides
                    let (x, y, cw, ch) = view.cell_rect(n, k);
                    if (cx - x).min(x + cw - cx).min(cy - y).min(y + ch - cy) < 1e-3 {
                        continue;
                    }
                    let o = ((py * w + px) * 4) as usize;
                    assert_eq!(&rgba[o..o + 3], &colors[n as usize][k as usize], "{scheme:?} preset {preset} pixel ({px},{py}) cell ({n},{k})");
                    checked += 1;
                }
            }
            assert!(checked > 10_000);
            if preset == 2 {
                assert_eq!(vmax, Some(4.0), "the view reaches row 81");
            }
        }
    }
}
