//! The tetra stack on the GPU against the CPU: every residue the compute pass writes
//! (each slice with its own modulus) is checked against the additive rule, the cell
//! enumeration is checked complete, and the ID-buffer pick is checked against a CPU
//! ray cast through the same cubes.
//!
//! Run with `cargo test --test tetra_verify`.

use std::collections::HashMap;

use pascal::gpu::Gpu;
use pascal::math::{triangle_dp, Modulus};
use pascal::palette::{Palette, Scheme};
use pascal::sweep::{ModMode, Seq};
use pascal::tetra::{ray_cast, selected_moduli, Camera, CubeId, Layout, Spacing, TetraConfig};
use pascal::tetra_gpu::{DrawOptions, TetraRenderer, BACKGROUND};

fn device() -> (Gpu, TetraRenderer) {
    let gpu = Gpu::new_headless().expect("a GPU adapter (install a software Vulkan driver such as Mesa lavapipe for headless runs)");
    let r = TetraRenderer::new(gpu.device.clone(), gpu.queue.clone());
    (gpu, r)
}

fn scene(r: &mut TetraRenderer, mods: &[u32], spacing: Spacing, budget: u32) -> (Layout, Vec<Modulus>) {
    let layout = Layout::new(mods, spacing, budget);
    let mds: Vec<Modulus> = mods.iter().map(|&m| Modulus::new(m)).collect();
    r.set_scene(&layout, &mds, &Palette::new(Scheme::Spectral), 0.0);
    (layout, mds)
}

/// Every strided cell appears exactly once, in the nonzero list iff its residue is
/// nonzero, with the additive rule's residue.
fn check_residues(r: &mut TetraRenderer, mods: &[u32], spacing: Spacing, budget: u32) -> Layout {
    let (layout, _) = scene(r, mods, spacing, budget);
    let (nz, z) = r.read_cubes();
    assert_eq!((nz.len() + z.len()) as u32, layout.total, "every cell is written once");
    let dp: HashMap<u32, Vec<u32>> = layout.slices.iter().map(|s| (s.m, triangle_dp(s.m as usize, s.m))).collect();
    let mut seen = vec![false; layout.total as usize];
    let index_of = |c: &CubeId| {
        let s = &layout.slices[c.slice as usize];
        let (i, j) = (c.n / layout.stride, c.k / layout.stride);
        (s.cell_base + i * (i + 1) / 2 + j) as usize
    };
    for (list, nonzero) in [(&nz, true), (&z, false)] {
        for w in list.iter() {
            let c = CubeId::unpack(*w);
            let s = &layout.slices[c.slice as usize];
            assert!(c.n < s.m && c.k <= c.n && c.n.is_multiple_of(layout.stride) && c.k.is_multiple_of(layout.stride), "{c:?} in slice m={}", s.m);
            let want = dp[&s.m][(c.n * (c.n + 1) / 2 + c.k) as usize];
            assert_eq!(c.residue, want, "C({},{}) mod {}", c.n, c.k, s.m);
            assert_eq!(want != 0, nonzero, "C({},{}) mod {} in the wrong list", c.n, c.k, s.m);
            let g = index_of(&c);
            assert!(!seen[g], "{c:?} written twice");
            seen[g] = true;
        }
    }
    assert!(seen.iter().all(|&v| v));
    layout
}

#[test]
fn every_slice_matches_the_additive_rule() {
    let (_gpu, mut r) = device();
    // every modulus 2..=64: primes (Lucas), prime powers (Granville), composites (CRT)
    let all: Vec<u32> = (2..=64).collect();
    let l = check_residues(&mut r, &all, Spacing::True, u32::MAX);
    assert_eq!(l.stride, 1);
    // the selection modes
    let mut c = TetraConfig::default();
    c.sweep.start = 2;
    c.sweep.end = 150;
    for (mode, seq) in [(ModMode::Prime, Seq::Fixed), (ModMode::Composite, Seq::Fixed), (ModMode::Interval, Seq::Fib), (ModMode::Interval, Seq::Square)] {
        c.sweep.mode = mode;
        c.sweep.seq = seq;
        let mods = selected_moduli(&c.sweep);
        assert!(mods.len() >= 4, "{mode:?} {seq:?}: {mods:?}");
        check_residues(&mut r, &mods, Spacing::Packed, u32::MAX);
    }
}

#[test]
fn strided_and_deep_slices_match_too() {
    let (_gpu, mut r) = device();
    // a wide range under a small budget: a cell stride
    let mods: Vec<u32> = (2..=300).collect();
    let l = check_residues(&mut r, &mods, Spacing::True, 200_000);
    assert!(l.stride > 1, "stride {}", l.stride);
    // the deepest slices (moduli up to the cap, many digits per prime factor)
    let deep = [2048, 2310, 3125, 3600, 4093, 4095, 4096];
    let l = check_residues(&mut r, &deep, Spacing::True, 400_000);
    assert!(l.stride >= 1);
    // stride 1 on a few big slices in full: n and k near the 16-bit packing limits
    check_residues(&mut r, &[1024, 1331, 1458], Spacing::True, u32::MAX);
}

#[test]
fn the_pick_buffer_names_the_cube_the_ray_hits() {
    let (_gpu, mut r) = device();
    let mods: Vec<u32> = (2..=26).collect();
    let (layout, mds) = scene(&mut r, &mods, Spacing::True, u32::MAX);
    let (w, h) = (320u32, 240u32);
    r.ensure_target(w, h);
    let he = layout.half_extents();
    let mut tested = 0;
    let mut misses = 0;
    for (yaw, pitch, zoom) in [(0.0, 0.0, 0.0), (0.9, 0.3, 300.0), (-2.2, -0.6, 600.0), (2.8, 1.2, 0.0)] {
        let mut cam = Camera::framing(layout.bounds());
        cam.yaw += yaw;
        cam.pitch = (cam.pitch + pitch).clamp(-1.5, 1.5);
        cam.zoom(zoom);
        for py in (3..h).step_by(13) {
            for px in (5..w).step_by(11) {
                let (o, d) = cam.ray(w, h, px as f32, py as f32);
                let hit = ray_cast(&layout, &mds, o, d);
                let gpu = r.pick_blocking(&cam, px as f32, py as f32);
                match hit {
                    Some(hit) => {
                        // only rays well inside a face and well in front of the next cube
                        // decide anything: at an edge, rasterization may pick either cube
                        let s = &layout.slices[hit.cube.slice as usize];
                        let c = pascal::tetra::cell_center(s.height, hit.cube.n, hit.cube.k);
                        let p = [o[0] + d[0] * hit.t, o[1] + d[1] * hit.t, o[2] + d[2] * hit.t];
                        let mut sorted = (0..3).map(|a| he[a] - (p[a] - c[a]).abs()).collect::<Vec<_>>();
                        sorted.sort_by(f32::total_cmp);
                        // the hit point lies on a face (smallest margin ≈ 0); the other two
                        // margins are its distance to the face's edges
                        let clear = sorted[1] > 0.03 && hit.next_t - hit.t > 0.05;
                        if clear {
                            assert_eq!(gpu, Some(hit.cube), "pixel ({px},{py}), camera {cam:?}");
                            tested += 1;
                        }
                    }
                    None => {
                        if let Some(g) = gpu {
                            // the GPU hit something the CPU missed: it must be a grazing ray
                            let s = &layout.slices[g.slice as usize];
                            let c = pascal::tetra::cell_center(s.height, g.n, g.k);
                            let grow = 0.05;
                            let lo = [c[0] - he[0] - grow, c[1] - he[1] - grow, c[2] - he[2] - grow];
                            let hi = [c[0] + he[0] + grow, c[1] + he[1] + grow, c[2] + he[2] + grow];
                            assert!(pascal::tetra::ray_box(o, d, lo, hi).is_some(), "pixel ({px},{py}): GPU picked {g:?}, which the ray misses");
                        } else {
                            misses += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("{tested} hits and {misses} misses agreed");
    assert!(tested > 400, "only {tested} decisive hits");
    assert!(misses > 100, "only {misses} misses");
}

#[test]
fn the_stack_draws_green_wire_and_lit_brush_cells() {
    let (_gpu, mut r) = device();
    let mods: Vec<u32> = (2..=40).collect();
    let (layout, _) = scene(&mut r, &mods, Spacing::True, u32::MAX);
    let (w, h) = (400u32, 300u32);
    r.ensure_target(w, h);
    let cam = Camera::framing(layout.bounds());
    let mut o = DrawOptions { camera: cam, hover: None, brush: 3.0, color_slice: false, ghosts: false, mask: false };
    r.draw_blocking(&o);
    let (_, _, plain) = r.read_target_rgba();
    let bg = BACKGROUND.map(|c| (c * 255.0).round() as u8);
    let px = |img: &[u8], i: usize| [img[i * 4], img[i * 4 + 1], img[i * 4 + 2]];
    let n = (w * h) as usize;
    let green = (0..n).filter(|&i| {
        let c = px(&plain, i);
        c[1] > c[0] + 20 && c[1] > c[2] + 10
    }).count();
    let background = (0..n).filter(|&i| px(&plain, i) == bg).count();
    assert!(green > n / 50, "wire pixels: {green}");
    assert!(background > n / 4, "background pixels: {background}");
    // hover the cube under the image centre: the brush fills cells in palette colors
    let picked = r.pick_blocking(&cam, w as f32 / 2.0, h as f32 / 2.0).expect("a cube at the centre");
    o.hover = Some(picked);
    r.draw_blocking(&o);
    let (_, _, lit) = r.read_target_rgba();
    let changed = (0..n).filter(|&i| px(&plain, i) != px(&lit, i)).count();
    assert!(changed > 30, "the brush changed {changed} pixels");
    // ghosts add faint zeros: only brighter, never darker
    o.hover = None;
    o.ghosts = true;
    r.draw_blocking(&o);
    let (_, _, ghost) = r.read_target_rgba();
    let brighter = (0..n).filter(|&i| px(&ghost, i).iter().zip(px(&plain, i)).any(|(a, b)| *a > b)).count();
    assert!(brighter > 100, "ghost pixels: {brighter}");
}
