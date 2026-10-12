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
use pascal::tetra::{ray_cast, ray_cast_peeled, selected_moduli, Camera, CubeId, Face, Layout, Peel, Shape, Spacing, TetraConfig, View, FOLD};
use pascal::tetra_gpu::{DrawOptions, TetraRenderer, BACKGROUND};

fn device() -> (Gpu, TetraRenderer) {
    let gpu = Gpu::new_headless().expect("a GPU adapter (install a software Vulkan driver such as Mesa lavapipe for headless runs)");
    let r = TetraRenderer::new(gpu.device.clone(), gpu.queue.clone());
    (gpu, r)
}

fn scene(r: &mut TetraRenderer, mods: &[u32], spacing: Spacing, budget: u32) -> (Layout, Vec<Modulus>) {
    scene_of(r, View::Stack, mods, spacing, Shape::Regular, budget)
}

/// Build `view`, upload it and run the residue pass; the moduli of its slices.
fn scene_of(r: &mut TetraRenderer, view: View, mods: &[u32], spacing: Spacing, shape: Shape, budget: u32) -> (Layout, Vec<Modulus>) {
    let layout = Layout::build(view, mods, spacing, shape, budget);
    let mds: Vec<Modulus> = layout.slices.iter().map(|s| Modulus::new(s.m)).collect();
    r.set_scene(&layout, &mds, &Palette::new(Scheme::Spectral), 0.0);
    (layout, mds)
}

/// Every cell of the view appears exactly once, in the nonzero list iff its residue is
/// nonzero, with the additive rule's residue.
fn check_view(r: &mut TetraRenderer, view: View, mods: &[u32], spacing: Spacing, budget: u32) -> Layout {
    let (layout, _) = scene_of(r, view, mods, spacing, Shape::Regular, budget);
    let (nz, z) = r.read_cubes();
    assert_eq!((nz.len() + z.len()) as u32, layout.total, "{view:?}: every cell is written once");
    let index: HashMap<(usize, u32, u32), u32> = (0..layout.total).map(|g| {
        let (si, n, k) = layout.cell(g).unwrap();
        ((si, n, k), g)
    }).collect();
    // the additive rule per slice modulus (rows up to the largest n the slice draws)
    let dp: HashMap<u32, Vec<u32>> = layout.slices.iter().map(|s| (s.m, triangle_dp(s.m as usize, s.m))).collect();
    let mut seen = vec![false; layout.total as usize];
    for (list, nonzero) in [(&nz, true), (&z, false)] {
        for w in list.iter() {
            let c = CubeId::unpack(*w);
            let si = c.slice as usize;
            let s = &layout.slices[si];
            assert!(c.n < s.m && c.k <= c.n, "{c:?} in slice m={}", s.m);
            let want = dp[&s.m][(c.n * (c.n + 1) / 2 + c.k) as usize];
            assert_eq!(c.residue, want, "{view:?}: C({},{}) mod {}", c.n, c.k, s.m);
            assert_eq!(want != 0, nonzero, "{view:?}: C({},{}) mod {} in the wrong list", c.n, c.k, s.m);
            let g = *index.get(&(si, c.n, c.k)).unwrap_or_else(|| panic!("{view:?}: {c:?} is not a cell of the layout"));
            assert!(!seen[g as usize], "{c:?} written twice");
            seen[g as usize] = true;
        }
    }
    assert!(seen.iter().all(|&v| v));
    layout
}

fn check_residues(r: &mut TetraRenderer, mods: &[u32], spacing: Spacing, budget: u32) -> Layout {
    check_view(r, View::Stack, mods, spacing, budget)
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
fn the_face_and_the_diamond_match_too() {
    let (_gpu, mut r) = device();
    let all: Vec<u32> = (2..=200).collect();
    for view in [View::Face, View::Diamond] {
        let l = check_view(&mut r, view, &all, Spacing::True, u32::MAX);
        assert_eq!(l.stride, 1);
        // under a stride: every s-th row, every s-th cell, the base always there
        let l = check_view(&mut r, view, &all, Spacing::Packed, 3_000);
        assert!(l.stride > 1, "{view:?} stride {}", l.stride);
        assert_eq!(l.slices.last().unwrap().m, 200);
        // the diamond's base keeps its hinge row 199 even when 199 is off the stride
        if view == View::Diamond {
            let b = l.slices.last().unwrap();
            assert_eq!(b.last_row(l.stride), 199);
            assert!(b.n0 > 0, "199 is not a multiple of stride {}: the base starts at row {}", l.stride, b.n0);
        }
    }
    // a prime selection: the face's rows alternate 1, p − 1 (C(p−1, k) ≡ (−1)^k)
    let primes: Vec<u32> = (2..=150).filter(|&m| pascal::math::is_prime(m)).collect();
    let l = check_view(&mut r, View::Face, &primes, Spacing::True, u32::MAX);
    let (nz, z) = r.read_cubes();
    assert!(z.is_empty(), "no zeros on a prime face");
    for w in nz {
        let c = CubeId::unpack(w);
        let m = l.slices[c.slice as usize].m;
        assert_eq!(c.residue, if c.k.is_multiple_of(2) { 1 % m } else { m - 1 }, "C({}, {}) mod {m}", c.n, c.k);
    }
}

#[test]
fn the_pick_sees_through_the_peel() {
    let (_gpu, mut r) = device();
    let mods: Vec<u32> = (2..=26).collect();
    let (layout, mds) = scene_of(&mut r, View::Stack, &mods, Spacing::True, Shape::Regular, u32::MAX);
    let (w, h) = (320u32, 240u32);
    r.ensure_target(w, h);
    let he = layout.half_extents();
    // sides in Side::ALL order: left, right, front, top, base
    for (sides, layers) in [([true, false, false, false, false], 3), ([false, false, true, false, false], 5), ([false, false, false, true, false], 6), ([true, true, true, true, false], 2), ([true, false, true, false, true], 2)] {
        let peel = Peel { sides, layers, ..Peel::default() }.depths();
        let cam = Camera::framing(layout.bounds());
        let (mut tested, mut agreed_misses) = (0, 0);
        for py in (3..h).step_by(11) {
            for px in (5..w).step_by(9) {
                let (o, d) = cam.ray(w, h, px as f32, py as f32);
                let gpu = r.pick_blocking_peeled(&cam, px as f32, py as f32, peel);
                if let Some(g) = gpu {
                    assert!(!layout.peeled(g.slice as usize, g.n, g.k, peel), "{sides:?}: picked a peeled cube {g:?}");
                }
                match ray_cast_peeled(&layout, &mds, o, d, peel) {
                    Some(hit) => {
                        let c = layout.center(hit.cube.slice as usize, hit.cube.n, hit.cube.k);
                        let p = [o[0] + d[0] * hit.t, o[1] + d[1] * hit.t, o[2] + d[2] * hit.t];
                        let mut m = (0..3).map(|a| he[a] - (p[a] - c[a]).abs()).collect::<Vec<_>>();
                        m.sort_by(f32::total_cmp);
                        if m[1] > 0.03 && hit.next_t - hit.t > 0.05 {
                            assert_eq!(gpu, Some(hit.cube), "{sides:?} pixel ({px},{py})");
                            tested += 1;
                        }
                    }
                    None => {
                        if gpu.is_none() {
                            agreed_misses += 1;
                        }
                    }
                }
            }
        }
        eprintln!("{sides:?} {layers}: {tested} hits, {agreed_misses} misses agreed");
        assert!(tested > 40 && agreed_misses > 50, "too few decisive rays");
    }
}

#[test]
fn the_flat_views_draw_their_zeros_solid() {
    let (_gpu, mut r) = device();
    // rows m − 1 of m = 30, 60, … hold zeros: the face is full of holes without them
    let mods: Vec<u32> = (2..=64).collect();
    let (layout, _) = scene_of(&mut r, View::Face, &mods, Spacing::True, Shape::Regular, u32::MAX);
    let (_, zeros) = r.read_cubes();
    assert!(!zeros.is_empty());
    let (w, h) = (640u32, 480u32);
    r.ensure_target(w, h);
    let cam = Camera::flat(layout.bounds(), w as f32, h as f32, 0.0, 0.0);
    let mut o = DrawOptions { camera: cam, fill: f32::NEG_INFINITY, hover: None, brush: 0.0, color_slice: false, ghosts: false, mask: false, flat: true, peel: [0; 5], solid_zeros: false };
    r.draw_blocking(&o);
    let (_, _, holes) = r.read_target_rgba();
    o.solid_zeros = true;
    r.draw_blocking(&o);
    let (_, _, solid) = r.read_target_rgba();
    let changed = (0..(w * h) as usize).filter(|&i| holes[i * 4..i * 4 + 3] != solid[i * 4..i * 4 + 3]).count();
    assert!(changed > 20 * zeros.len(), "the zeros filled {changed} pixels");
    // and each zero is pickable where it sits
    for w0 in zeros.iter().step_by(7) {
        let c = CubeId::unpack(*w0);
        let p = layout.center(c.slice as usize, c.n, c.k);
        let (px, py) = cam.project(w, h, p).unwrap();
        let hit = r.pick_blocking_with(&cam, px - 0.5, py - 0.5, [0; 5], true);
        assert_eq!(hit, Some(c), "zero cube {c:?}");
        assert_eq!(r.pick_blocking(&cam, px - 0.5, py - 0.5), None, "without the zeros the pick sees the gap");
    }
}

#[test]
fn a_wide_range_draws_every_cell() {
    // the reported trouble: 2–500 drew every third slice, row and cell under the old
    // 1M budget. "all" (the default) now draws all 20.8M cells when the GPU holds them.
    let (_gpu, mut r) = device();
    // 2–500 where the device holds its 20.8M cubes (a desktop GPU), else the widest
    // range that fits (a software device caps a buffer at 128 MB)
    let cap = r.max_cells() as u64;
    let hi = (100..=500u32).rev().find(|&hi| (2..=hi).map(|m| (m as u64) * (m as u64 + 1) / 2).sum::<u64>() <= cap).unwrap();
    let mods: Vec<u32> = (2..=hi).collect();
    let budget = TetraConfig::default().budget.min(r.max_cells());
    let layout = Layout::build(View::Stack, &mods, Spacing::True, Shape::Regular, budget);
    eprintln!("max cells here {cap}: 2–{hi}, {} cells, stride {}", layout.total, layout.stride);
    assert!(hi >= 400, "the range should reach 400 at least");
    assert_eq!(layout.stride, 1, "every cell");
    assert_eq!(layout.slices.len(), mods.len(), "every slice");
    let mds: Vec<Modulus> = layout.slices.iter().map(|s| Modulus::new(s.m)).collect();
    r.set_scene(&layout, &mds, &Palette::new(Scheme::Spectral), 0.0);
    let (nz, z) = r.read_cubes();
    assert_eq!((nz.len() + z.len()) as u32, layout.total);
    for w in nz.iter().step_by(997).chain(z.iter().step_by(97)) {
        let c = CubeId::unpack(*w);
        assert_eq!(c.residue, mds[c.slice as usize].binom(c.n, c.k), "{c:?}");
    }
}

#[test]
fn strided_and_deep_slices_match_too() {
    let (_gpu, mut r) = device();
    // a wide range under a small budget: a stride on slices, rows and cells
    let mods: Vec<u32> = (2..=300).collect();
    let l = check_residues(&mut r, &mods, Spacing::True, 200_000);
    assert!(l.stride > 1, "stride {}", l.stride);
    assert!(l.slices.len() < mods.len() && l.slices.last().unwrap().m == 300);
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
    let small: Vec<u32> = (2..=26).collect();
    let orbit = |l: &Layout| -> Vec<Camera> {
        [(0.0, 0.0, 0.0), (0.9, 0.3, 300.0), (-2.2, -0.6, 600.0), (2.8, 1.2, 0.0)].iter().map(|&(yaw, pitch, zoom)| {
            let mut cam = Camera::framing(l.bounds());
            cam.yaw += yaw;
            cam.pitch = (cam.pitch + pitch).clamp(-1.5, 1.5);
            cam.zoom(zoom);
            cam
        }).collect()
    };
    pick_agrees(&mut r, View::Stack, &small, Spacing::True, Shape::Regular, u32::MAX, &orbit, 400, 100);
    pick_agrees(&mut r, View::Stack, &small, Spacing::Packed, Shape::Regular, u32::MAX, &orbit, 300, 100);
    // the flush shapes shift every slice differently
    pick_agrees(&mut r, View::Stack, &small, Spacing::True, Shape::CornerApex, u32::MAX, &orbit, 400, 100);
    pick_agrees(&mut r, View::Stack, &small, Spacing::True, Shape::FaceLeft, u32::MAX, &orbit, 400, 100);
    // cubes s cells wide, every s-th slice
    let l = pick_agrees(&mut r, View::Stack, &(2..=120).collect::<Vec<_>>(), Spacing::True, Shape::Regular, 20_000, &orbit, 300, 100);
    assert!(l.stride > 1, "stride {}", l.stride);
    // the flat views, straight on (orthographic), zoomed and panned
    let flat = |l: &Layout| -> Vec<Camera> {
        let base = Camera::flat(l.at_rest().bounds(), 320.0, 240.0, 0.0, 0.0);
        let mut near = base;
        near.zoom(700.0);
        near.pan(30.0, -20.0, 240.0);
        vec![base, near]
    };
    let face: Vec<u32> = (2..=40).collect();
    pick_agrees(&mut r, View::Face, &face, Spacing::True, Shape::Regular, u32::MAX, &flat, 200, 100);
    pick_agrees(&mut r, View::Diamond, &face, Spacing::True, Shape::Regular, u32::MAX, &flat, 200, 100);
    // and the diamond halfway through its hinge, seen at an angle
    let half_open = |l: &Layout| -> Vec<Camera> {
        let mut side = Camera::framing(l.bounds());
        side.yaw = 2.6;
        side.pitch = 0.3;
        let mut front = side;
        front.yaw = std::f32::consts::PI - 0.25;
        front.pitch = 0.15;
        vec![side, front]
    };
    // (a thin sheet seen at a slant: few rays clear the next cube by a margin)
    pick_agrees(&mut r, View::Diamond, &face, Spacing::True, Shape::Regular, u32::MAX, &half_open, 60, 30);
}

/// GPU picks against the CPU ray cast over a grid of pixels from each camera.
#[allow(clippy::too_many_arguments)]
fn pick_agrees(r: &mut TetraRenderer, view: View, mods: &[u32], spacing: Spacing, shape: Shape, budget: u32, cams: &dyn Fn(&Layout) -> Vec<Camera>, min_hits: usize, min_misses: usize) -> Layout {
    let (mut layout, mds) = scene_of(r, view, mods, spacing, shape, budget);
    if view == View::Diamond {
        // the flat cameras see it open; the angled one halfway
        let open = if cams(&layout).iter().all(|c| c.ortho > 0.0) { 0.0 } else { FOLD / 2.0 };
        layout.set_fold(open);
        r.set_placement(&layout);
    }
    let (w, h) = (320u32, 240u32);
    r.ensure_target(w, h);
    let he = layout.half_extents();
    let mut tested = 0;
    let mut misses = 0;
    for cam in cams(&layout) {
        for py in (3..h).step_by(13) {
            for px in (5..w).step_by(11) {
                let (o, d) = cam.ray(w, h, px as f32, py as f32);
                let hit = ray_cast(&layout, &mds, o, d);
                let gpu = r.pick_blocking(&cam, px as f32, py as f32);
                match hit {
                    Some(hit) => {
                        // only rays well inside a face and well in front of the next cube
                        // decide anything: at an edge, rasterization may pick either cube
                        let c = layout.center(hit.cube.slice as usize, hit.cube.n, hit.cube.k);
                        let p = [o[0] + d[0] * hit.t, o[1] + d[1] * hit.t, o[2] + d[2] * hit.t];
                        let mut sorted = (0..3).map(|a| he[a] - (p[a] - c[a]).abs()).collect::<Vec<_>>();
                        sorted.sort_by(f32::total_cmp);
                        // the hit point lies on a face (smallest margin ≈ 0); the other two
                        // margins are its distance to the face's edges
                        let clear = sorted[1] > 0.03 && hit.next_t - hit.t > 0.05;
                        if clear {
                            assert_eq!(gpu, Some(hit.cube), "{view:?}: pixel ({px},{py}), camera {cam:?}");
                            tested += 1;
                        }
                    }
                    None => {
                        if let Some(g) = gpu {
                            // the GPU hit something the CPU missed: it must be a grazing ray
                            let c = layout.center(g.slice as usize, g.n, g.k);
                            let grow = 0.05;
                            let lo = [c[0] - he[0] - grow, c[1] - he[1] - grow, c[2] - he[2] - grow];
                            let hi = [c[0] + he[0] + grow, c[1] + he[1] + grow, c[2] + he[2] + grow];
                            assert!(pascal::tetra::ray_box(o, d, lo, hi).is_some(), "{view:?}: pixel ({px},{py}): GPU picked {g:?}, which the ray misses");
                        } else {
                            misses += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("{view:?} {spacing:?}, stride {}: {tested} hits and {misses} misses agreed", layout.stride);
    assert!(tested > min_hits, "{view:?}: only {tested} decisive hits");
    assert!(misses > min_misses, "{view:?}: only {misses} misses");
    layout
}

#[test]
fn the_stack_draws_green_wire_and_lit_brush_cells() {
    let (_gpu, mut r) = device();
    let mods: Vec<u32> = (2..=40).collect();
    let (layout, _) = scene(&mut r, &mods, Spacing::True, u32::MAX);
    let (w, h) = (400u32, 300u32);
    r.ensure_target(w, h);
    let cam = Camera::framing(layout.bounds());
    let mut o = DrawOptions { camera: cam, fill: f32::INFINITY, hover: None, brush: 3.0, color_slice: false, ghosts: false, mask: false, flat: false, peel: [0; 5], solid_zeros: false };
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
    // filled: the palette colors replace the green wire, everywhere or from the top down
    o.hover = None;
    o.fill = f32::NEG_INFINITY;
    r.draw_blocking(&o);
    let (_, _, full) = r.read_target_rgba();
    let colored = |img: &[u8], i: usize| {
        let c = px(img, i);
        c != bg && !(c[1] > c[0] + 20 && c[1] > c[2] + 10)
    };
    let all = (0..n).filter(|&i| colored(&full, i)).count();
    assert!(all > n / 20, "filled pixels: {all}");
    let top = layout.slices[0].height();
    let bottom = layout.slices.last().unwrap().height();
    o.fill = -(top + bottom) / 2.0;
    r.draw_blocking(&o);
    let (_, _, half) = r.read_target_rgba();
    let part = (0..n).filter(|&i| colored(&half, i)).count();
    assert!(part > 30 && part < all, "half filled: {part} of {all}");
    // the filled half is the top: colored rows sit above the green ones on average
    let mean_row = |img: &[u8], pick: &dyn Fn(&[u8], usize) -> bool| {
        let v: Vec<usize> = (0..n).filter(|&i| pick(img, i)).map(|i| i / w as usize).collect();
        v.iter().sum::<usize>() as f64 / v.len().max(1) as f64
    };
    let green_px = |img: &[u8], i: usize| {
        let c = px(img, i);
        c[1] > c[0] + 20 && c[1] > c[2] + 10
    };
    assert!(mean_row(&half, &colored) < mean_row(&half, &green_px), "the color runs from the top");
    o.fill = f32::INFINITY;
    // ghosts add faint zeros: only brighter, never darker
    o.hover = None;
    o.ghosts = true;
    r.draw_blocking(&o);
    let (_, _, ghost) = r.read_target_rgba();
    let brighter = (0..n).filter(|&i| px(&ghost, i).iter().zip(px(&plain, i)).any(|(a, b)| *a > b)).count();
    assert!(brighter > 100, "ghost pixels: {brighter}");
}

/// Each face of the face view, peeled into: every cell written once with its residue,
/// on the face's own line (the front's row m − 1 − d, the left's k = d, the right's
/// n − k = d, the base's slice d up), and the pick seeing through the other sides' peel.
#[test]
fn the_faces_peel_on_the_gpu() {
    let (_gpu, mut r) = device();
    let mods: Vec<u32> = (2..=40).collect();
    let (w, h) = (320u32, 240u32);
    r.ensure_target(w, h);
    // sides in Side::ALL order: left, right, front, top, base
    let peels: [[bool; 5]; 4] = [[false; 5], [true, false, false, false, false], [true, true, true, false, false], [false, true, true, true, true]];
    for face in Face::ALL {
        for (budget, sides) in [(u32::MAX, peels[0]), (u32::MAX, peels[1]), (u32::MAX, peels[2]), (300, peels[3])] {
            for layers in [0u32, 1, 3] {
                let peel = Peel { sides, layers, ..Peel::default() }.depths();
                let depth = face.side().map_or(0, |i| peel[i]);
                let layout = Layout::build_face(face, depth, &mods, Spacing::True, Shape::Regular, budget);
                let mds: Vec<Modulus> = layout.slices.iter().map(|s| Modulus::new(s.m)).collect();
                r.set_scene(&layout, &mds, &Palette::new(Scheme::Spectral), 0.0);
                let (nz, z) = r.read_cubes();
                assert_eq!((nz.len() + z.len()) as u32, layout.total, "{face:?} {sides:?} {layers}");
                let st = layout.stride;
                let mut seen = std::collections::HashSet::new();
                for wd in nz.iter().chain(&z) {
                    let c = CubeId::unpack(*wd);
                    let s = &layout.slices[c.slice as usize];
                    assert!(c.n < s.m && c.k <= c.n, "{face:?}: {c:?} mod {}", s.m);
                    assert_eq!(c.residue, mds[c.slice as usize].binom(c.n, c.k), "{face:?}: {c:?}");
                    let d = depth * st;
                    match face {
                        Face::Front => assert_eq!(c.n, s.m - 1 - d),
                        Face::Left => assert_eq!(c.k, d),
                        Face::Right => assert_eq!(c.n - c.k, d),
                        Face::Base => assert_eq!(s.m, mods[mods.len() - 1 - depth as usize]),
                        Face::Top => {}
                    }
                    assert!(seen.insert((c.slice, c.n, c.k)), "{c:?} twice");
                }
                // the pick against the CPU ray cast, through the peel (three layers in)
                if layout.total == 0 || layers != 3 {
                    continue;
                }
                let cam = Camera::flat(layout.bounds(), w as f32, h as f32, 0.0, 0.0);
                let he = layout.half_extents();
                let mut agreed = 0;
                for py in (3..h).step_by(16) {
                    for px in (5..w).step_by(16) {
                        let (o, dir) = cam.ray(w, h, px as f32, py as f32);
                        let gpu = r.pick_blocking_peeled(&cam, px as f32, py as f32, peel);
                        if let Some(g) = gpu {
                            assert!(!layout.peeled(g.slice as usize, g.n, g.k, peel), "{face:?}: picked a peeled cube {g:?}");
                        }
                        if let Some(hit) = ray_cast_peeled(&layout, &mds, o, dir, peel) {
                            let c = layout.center(hit.cube.slice as usize, hit.cube.n, hit.cube.k);
                            let p = [o[0] + dir[0] * hit.t, o[1] + dir[1] * hit.t, o[2] + dir[2] * hit.t];
                            let mut m = (0..3).map(|a| he[a] - (p[a] - c[a]).abs()).collect::<Vec<_>>();
                            m.sort_by(f32::total_cmp);
                            if m[1] > 0.03 && hit.next_t - hit.t > 0.05 {
                                assert_eq!(gpu, Some(hit.cube), "{face:?} {sides:?} {layers}: pixel ({px},{py})");
                                agreed += 1;
                            }
                        }
                    }
                }
                assert!(agreed > 0 || nz.is_empty(), "{face:?} {sides:?} {layers}: no decisive rays");
            }
        }
    }
}
