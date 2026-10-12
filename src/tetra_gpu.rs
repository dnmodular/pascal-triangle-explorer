//! The tetra stack on the GPU: a compute pass that finds every cell's residue (each
//! slice with its own modulus, through the same Granville / CRT code as the 2D passes)
//! and sorts the cells into nonzero cubes and zeros, then instanced, indirect draws of
//! the cubes, an ID pass for picking, and the slice outlines.
//!
//! The residues depend only on the selection, so the compute pass runs when the
//! selection or its level of detail changes; colors (palette, offset) are looked up
//! while drawing.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::gpu::{GpuFactor, BINOM_WGSL};
use crate::math::{to_mont, Modulus};
use crate::palette::{level_map, level_map_words, residue_level, Palette, Scheme};
use crate::tetra::{rhombic_wgsl, Camera, Cell, CubeId, Layout, Mat4, SliceKind, View};

pub const TETRA_CELLS_WGSL: &str = include_str!("../shaders/tetra_cells.wgsl");
pub const TETRA_DRAW_WGSL: &str = include_str!("../shaders/tetra_draw.wgsl");

pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
/// Near-black with a hint of green.
pub const BACKGROUND: [f32; 3] = [6.0 / 255.0, 9.0 / 255.0, 7.0 / 255.0];
pub const WIRE: [f32; 4] = [0.32, 0.95, 0.48, 0.9];
const WG: u32 = 64;
const MAX_GROUPS_X: u32 = 32_768;
const CUBE_INDICES: [u16; 36] = [
    0, 2, 1, 1, 2, 3, // −z
    4, 5, 6, 5, 7, 6, // +z
    0, 1, 4, 1, 5, 4, // −y
    2, 6, 3, 3, 6, 7, // +y
    0, 4, 2, 2, 4, 6, // −x
    1, 3, 5, 3, 7, 5, // +x
];

/// Every cell shape's indices in one buffer: the cube's 36, then the rhombic
/// dodecahedron's 72 (two triangles per rhombus, corners 4f..4f + 3), then the sphere
/// quad's 6.
fn cell_indices() -> Vec<u16> {
    let mut v = CUBE_INDICES.to_vec();
    for f in 0..12u16 {
        let b = 4 * f;
        v.extend([b, b + 1, b + 2, b, b + 2, b + 3]);
    }
    v.extend([0, 1, 2, 2, 1, 3]);
    v
}

/// (index count, first index) of a cell shape in [`cell_indices`]. `flat`: a planar
/// view, seen straight on along −z: only what faces the viewer can show, so only that
/// is drawn. A box's +z face (6 indices of 36); a rhombic dodecahedron's three rhombi
/// turned up in the top face (faces 9–11 of [`crate::tetra::rhombic_faces`], away from
/// the step down: six more stand edge-on and three face away; 18 of 72); a sphere is a
/// flat disk either way.
fn cell_range(cell: Cell, flat: bool) -> (u32, u32) {
    match cell {
        Cell::Cube if flat => (6, 6),
        Cell::Cube => (36, 0),
        Cell::Rhombic if flat => (18, 36 + 6 * 9),
        Cell::Rhombic => (72, 36),
        Cell::Sphere => (6, 108),
    }
}

/// A planar view: the flat views' orthographic camera, which only pans and zooms and
/// always looks straight along −z (the faces, the base, the diamond, the top face).
/// There a cell is drawn as what of it can show from the front, a 2D shape: a box's
/// front face, a rhombic dodecahedron's three upper rhombi, a sphere's disk at its
/// centre's depth. Equal cells that do not overlap, seen along parallel rays, are in
/// front of one another exactly as their centres are (the plane between two centres
/// separates them), so the picture is the same as the solids'.
fn flat_view(camera: &Camera) -> bool {
    camera.ortho > 0.0 && camera.basis().0[2] < -0.9999
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct CellParams {
    n_slices: u32,
    stride: u32,
    total: u32,
    row_threads: u32,
    /// Reserved (0).
    flags: u32,
    pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuSlice {
    pub m: u32,
    pub rows: u32,
    pub cell_base: u32,
    pub nfactors: u32,
    pub factor_off: u32,
    pub level_off: u32,
    /// 0 a triangle of rows row_n, row_n + s, … (row_n = its first row, 0 but for the
    /// diamond's base and the base face); else a line of cells (row_n + j·dn·s, k0 +
    /// j·dk·s), kind = 1 | dn << 1 | dk << 2.
    pub kind: u32,
    pub row_n: u32,
    /// Cell (n, k) sits at origin + n·nvec + (k, 0, 0).
    pub origin: [f32; 3],
    pub k0: u32,
    pub nvec: [f32; 3],
    /// The peel's view of it: the row the front side counts from | its layer << 16.
    pub peel_id: u32,
}

impl GpuSlice {
    fn place(&mut self, s: &crate::tetra::Slice) {
        self.origin = s.origin;
        self.nvec = s.nvec;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct DrawParams {
    view_proj: Mat4,
    hover: [f32; 4],
    half: [f32; 4],
    wire: [f32; 4],
    hover_slice: u32,
    flags: u32,
    line_px: f32,
    ghost_alpha: f32,
    fill: [f32; 4],
    peel_a: [u32; 4],
    peel_b: [u32; 4],
    eye: [f32; 4],
    fwd: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
}

const FLAG_SLICE: u32 = 1;
const FLAG_MASK: u32 = 2;
const FLAG_GHOST: u32 = 4;
const FLAG_FLAT: u32 = 8;
const FLAG_ZERO: u32 = 16;
const FLAG_RHOMBIC: u32 = 32;
const FLAG_SPHERE: u32 = 64;
const FLAG_TURNED: u32 = 128;
/// Most cubes a scene holds, whatever the budget (8 bytes each, one buffer).
pub const MAX_CELLS: u32 = 48_000_000;

/// One line vertex: position, premultiplied-later RGBA, dash coordinate (< 0 solid).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
    pub dash: f32,
}

/// The number-theory side of a layout, as uploaded: per-slice descriptors, factor
/// descriptors (Garner constants per slice) and the (x!)_p tables, one copy per
/// distinct prime power.
pub struct SceneData {
    pub slices: Vec<GpuSlice>,
    pub factors: Vec<GpuFactor>,
    pub tables: Vec<u32>,
    /// Residue → palette level bytes of every slice, packed four per word.
    pub level_bytes: usize,
}

impl SceneData {
    pub fn new(layout: &Layout, mods: &[Modulus]) -> SceneData {
        let mut tables = Vec::new();
        let mut by_q: std::collections::HashMap<u32, (u32, u32)> = Default::default();
        let mut factors = Vec::new();
        let mut slices = Vec::with_capacity(layout.slices.len());
        let mut level_off = 0u32;
        for (s, md) in layout.slices.iter().zip(mods) {
            let factor_off = factors.len() as u32;
            for t in &md.factors {
                let q = t.f.q;
                let odd = t.f.p != 2;
                let conv = |x: u32| if odd { to_mont(x, q) } else { x };
                let (tab_off, ppow_off) = *by_q.entry(q).or_insert_with(|| {
                    let tab_off = tables.len() as u32;
                    tables.extend(t.fp.iter().map(|&x| conv(x)));
                    tables.extend(t.ifp.iter().map(|&x| conv(x)));
                    let ppow_off = tables.len() as u32;
                    let mut pw = 1u64;
                    for _ in 0..t.f.e {
                        tables.push(conv(pw as u32));
                        pw = pw * t.f.p as u64 % q as u64;
                    }
                    (tab_off, ppow_off)
                });
                factors.push(GpuFactor {
                    p: t.f.p,
                    e: t.f.e,
                    q,
                    qinv: t.qinv,
                    one: conv(1),
                    p_mont: conv(t.f.p % q),
                    tab_off,
                    minus_one: t.minus_one as u32,
                    garner_q: t.garner_q,
                    garner_inv: conv(t.garner_inv),
                    ppow_off,
                    pad1: 0,
                });
            }
            let (kind, row_n, k0) = match s.kind {
                SliceKind::Triangle => (0, s.n0, 0),
                SliceKind::Line { n, k, dn, dk } => (1 | dn << 1 | dk << 2, n, k),
            };
            let peel_id = s.front | s.layer << 16;
            slices.push(GpuSlice { m: s.m, rows: s.rows, cell_base: s.cell_base, nfactors: md.factors.len() as u32, factor_off, level_off, kind, row_n, origin: s.origin, k0, nvec: s.nvec, peel_id });
            level_off += s.m;
        }
        if tables.is_empty() {
            tables.push(0);
        }
        if factors.is_empty() {
            factors.push(GpuFactor::default());
        }
        if slices.is_empty() {
            slices.push(GpuSlice::default());
        }
        SceneData { slices, factors, tables, level_bytes: level_off as usize }
    }
}

/// Residue levels of every slice (bytes at `level_off + r`), packed four per word.
pub fn level_words(layout: &Layout, spectral: bool) -> Vec<u32> {
    let total: usize = layout.slices.iter().map(|s| s.m as usize).sum();
    let mut out = vec![0u32; total.div_ceil(4).max(1)];
    let mut b = 0usize;
    for s in &layout.slices {
        for r in 0..s.m {
            out[b / 4] |= (residue_level(r, s.m, spectral) as u32) << (8 * (b % 4));
            b += 1;
        }
    }
    out
}

/// The 256 residue colors, then the color-offset level map of every slice.
pub fn color_words(layout: &Layout, palette: &Palette, offset: f32) -> Vec<u32> {
    let mut w: Vec<u32> = palette.gpu_words()[..256].to_vec();
    let spectral = palette.scheme == Scheme::Spectral;
    for s in &layout.slices {
        w.extend_from_slice(&level_map_words(&level_map(s.m, spectral, offset)));
    }
    if layout.slices.is_empty() {
        w.extend_from_slice(&[0; 64]);
    }
    w
}

/// What one draw shows besides the cubes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawOptions {
    pub camera: Camera,
    /// Cubes whose centre is at or above this height (world y) are filled in their
    /// palette color, the rest are wire: `f32::NEG_INFINITY` fills all.
    pub fill: f32,
    /// The hovered cube, highlighted with its brush (or its slice).
    pub hover: Option<CubeId>,
    pub brush: f32,
    pub color_slice: bool,
    pub ghosts: bool,
    /// Zero / nonzero mask lens instead of the palette.
    pub mask: bool,
    /// A flat view (the face, the diamond): faces toward the viewer in true colors.
    pub flat: bool,
    /// Layers peeled off each [`crate::tetra::Side`] (the stack and the faces; 0 = none).
    pub peel: [u32; 5],
    /// The zeros as solid dark cubes instead of holes (the flat views).
    pub solid_zeros: bool,
    /// What the cells are drawn as (the stack and the top face; elsewhere boxes).
    pub cell: Cell,
}

/// What a pick sees: the peel, whether the zeros are solid (and pickable), the cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickSpec {
    pub peel: [u32; 5],
    pub zeros: bool,
    pub cell: Cell,
}

struct Target {
    w: u32,
    h: u32,
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

struct PickTarget {
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    staging: wgpu::Buffer,
}

struct Scene {
    layout: Layout,
    /// The slice table as uploaded (the diamond's hinge rewrites its last entry).
    gpu_slices: Vec<GpuSlice>,
    slices: wgpu::Buffer,
    /// The residue levels were built for spectral.
    spectral: bool,
    levels: wgpu::Buffer,
    colors: wgpu::Buffer,
    cubes: wgpu::Buffer,
    counters: wgpu::Buffer,
    draw_nz: wgpu::BindGroup,
    draw_z: wgpu::BindGroup,
    pick_nz: wgpu::BindGroup,
    pick_z: wgpu::BindGroup,
}

pub struct TetraRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    cells_layout: wgpu::BindGroupLayout,
    draw_layout: wgpu::BindGroupLayout,
    cells_pipe: wgpu::ComputePipeline,
    cube_pipe: wgpu::RenderPipeline,
    solid_pipe: wgpu::RenderPipeline,
    ghost_pipe: wgpu::RenderPipeline,
    pick_pipe: wgpu::RenderPipeline,
    line_pipe: wgpu::RenderPipeline,
    sphere_pipe: wgpu::RenderPipeline,
    sphere_pick_pipe: wgpu::RenderPipeline,
    /// The cells' indices: the cube's, the rhombic dodecahedron's, the sphere's quad.
    index_buf: wgpu::Buffer,
    draw_ubuf: wgpu::Buffer,
    ghost_ubuf: wgpu::Buffer,
    pick_ubuf: wgpu::Buffer,
    pick_z_ubuf: wgpu::Buffer,
    lines: Option<(wgpu::Buffer, u32)>,
    scene: Option<Scene>,
    target: Option<Target>,
    pick: PickTarget,
    /// A pick copied into `pick.staging` and being mapped; the flag turns true when done.
    pick_pending: Option<Arc<AtomicBool>>,
    /// `map_async` was called for the pending pick.
    pick_mapping: bool,
    pick_failed: Arc<AtomicBool>,
}

fn storage(binding: u32, read_only: bool, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: vis, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None }, count: None }
}

fn uniform(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: vis, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }
}

fn init(device: &wgpu::Device, label: &str, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage })
}

/// The bind group layouts: the compute pass, and the draws (vertex + fragment).
fn layouts(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    let c = wgpu::ShaderStages::COMPUTE;
    let cells = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tetra cells"),
        entries: &[uniform(0, c), storage(1, true, c), storage(2, true, c), storage(3, true, c), storage(4, false, c), storage(6, false, c)],
    });
    let v = wgpu::ShaderStages::VERTEX_FRAGMENT;
    let draw = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tetra draw"),
        entries: &[uniform(0, v), storage(1, true, wgpu::ShaderStages::VERTEX), storage(2, true, wgpu::ShaderStages::VERTEX), storage(3, true, wgpu::ShaderStages::VERTEX), storage(4, true, wgpu::ShaderStages::VERTEX)],
    });
    (cells, draw)
}

/// The shader modules exactly as compiled: the compute module and the draw module.
pub fn module_sources() -> [(&'static str, String); 2] {
    [("tetra-cells", format!("{BINOM_WGSL}\n{TETRA_CELLS_WGSL}")), ("tetra-draw", format!("{TETRA_DRAW_WGSL}\n{}", rhombic_wgsl()))]
}

/// Pipelines by name, for `--check-shaders`.
pub const PIPELINES: [&str; 8] = ["tetra_cells", "cube", "solid", "ghost", "pick", "line", "sphere", "sphere_pick"];

struct Pipes {
    cells: wgpu::ComputePipeline,
    cube: wgpu::RenderPipeline,
    /// The cubes when every one is filled: back faces culled (half the raster work).
    solid: wgpu::RenderPipeline,
    ghost: wgpu::RenderPipeline,
    pick: wgpu::RenderPipeline,
    line: wgpu::RenderPipeline,
    /// Spheres: the quads' fragments find the sphere and write its depth.
    sphere: wgpu::RenderPipeline,
    sphere_pick: wgpu::RenderPipeline,
}

fn depth_state(write: bool, compare: wgpu::CompareFunction) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: Some(write), depth_compare: Some(compare), stencil: Default::default(), bias: Default::default() }
}

const PREMULTIPLIED: wgpu::BlendState = wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;
const ADDITIVE: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
    alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
};

/// Build one pipeline (`which` from [`PIPELINES`]) or all of them.
fn build_pipes(device: &wgpu::Device, cells_layout: &wgpu::BindGroupLayout, draw_layout: &wgpu::BindGroupLayout, only: Option<&str>, unchecked: bool) -> Option<Pipes> {
    let [cells_src, draw_src] = module_sources();
    let module = |label: &str, src: &str| {
        let desc = wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.to_owned().into()) };
        if unchecked {
            // SAFETY: as for the 2D passes: every index is bounded in WGSL (instance
            // indices below the counts the compute pass wrote, slice indices from it,
            // table offsets built from the uploaded sizes) and every loop is bounded.
            unsafe { device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) }
        } else {
            device.create_shader_module(desc)
        }
    };
    let want = |n: &str| only.is_none_or(|o| o == n);
    let cells_layout_p = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("tetra cells"), bind_group_layouts: &[Some(cells_layout)], immediate_size: 0 });
    let draw_layout_p = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("tetra draw"), bind_group_layouts: &[Some(draw_layout)], immediate_size: 0 });
    let cells = want("tetra_cells").then(|| {
        let m = module(cells_src.0, &cells_src.1);
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("tetra_cells"), layout: Some(&cells_layout_p), module: &m, entry_point: Some("tetra_cells"), compilation_options: Default::default(), cache: None })
    });
    let draw_m = PIPELINES[1..].iter().any(|n| want(n)).then(|| module(draw_src.0, &draw_src.1));
    let render = |label: &str, vs: &str, fs: &str, format: wgpu::TextureFormat, blend: Option<wgpu::BlendState>, depth: wgpu::DepthStencilState, topology: wgpu::PrimitiveTopology, cull: Option<wgpu::Face>, buffers: &[Option<wgpu::VertexBufferLayout>]| {
        let m = draw_m.as_ref().expect("draw module");
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&draw_layout_p),
            vertex: wgpu::VertexState { module: m, entry_point: Some(vs), compilation_options: Default::default(), buffers },
            primitive: wgpu::PrimitiveState { topology, cull_mode: cull, ..Default::default() },
            depth_stencil: Some(depth),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState { module: m, entry_point: Some(fs), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })] }),
            multiview_mask: None,
            cache: None,
        })
    };
    let tri = wgpu::PrimitiveTopology::TriangleList;
    let cube = want("cube").then(|| render("cube", "cube_vs", "cube_fs", COLOR_FORMAT, Some(PREMULTIPLIED), depth_state(true, wgpu::CompareFunction::Less), tri, None, &[]));
    let solid = want("solid").then(|| render("solid", "cube_vs", "cube_fs", COLOR_FORMAT, Some(PREMULTIPLIED), depth_state(true, wgpu::CompareFunction::Less), tri, Some(wgpu::Face::Back), &[]));
    let ghost = want("ghost").then(|| render("ghost", "cube_vs", "cube_fs", COLOR_FORMAT, Some(ADDITIVE), depth_state(false, wgpu::CompareFunction::Always), tri, None, &[]));
    let pick = want("pick").then(|| render("pick", "pick_vs", "pick_fs", PICK_FORMAT, None, depth_state(true, wgpu::CompareFunction::Less), tri, Some(wgpu::Face::Back), &[]));
    let line_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4, 2 => Float32];
    let line_buf = wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<LineVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &line_attrs };
    let line = want("line").then(|| render("line", "line_vs", "line_fs", COLOR_FORMAT, Some(PREMULTIPLIED), depth_state(false, wgpu::CompareFunction::LessEqual), wgpu::PrimitiveTopology::LineList, None, &[Some(line_buf.clone())]));
    let sphere = want("sphere").then(|| render("sphere", "cube_vs", "sphere_fs", COLOR_FORMAT, Some(PREMULTIPLIED), depth_state(true, wgpu::CompareFunction::Less), tri, None, &[]));
    let sphere_pick = want("sphere_pick").then(|| render("sphere_pick", "pick_vs", "sphere_pick_fs", PICK_FORMAT, None, depth_state(true, wgpu::CompareFunction::Less), tri, None, &[]));
    Some(Pipes { cells: cells?, cube: cube?, solid: solid?, ghost: ghost?, pick: pick?, line: line?, sphere: sphere?, sphere_pick: sphere_pick? })
}

/// Compile one tetra pipeline on its own (for `--check-shaders`).
pub fn check_pipeline(device: &wgpu::Device, which: &str, unchecked: bool) {
    let (c, d) = layouts(device);
    let _ = build_pipes(device, &c, &d, Some(which), unchecked);
}

impl TetraRenderer {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> TetraRenderer {
        let (cells_layout, draw_layout) = layouts(&device);
        let unchecked = crate::gpu::unchecked_shaders_requested();
        let p = build_pipes(&device, &cells_layout, &draw_layout, None, unchecked).expect("all pipelines");
        let index_buf = init(&device, "cell indices", bytemuck::cast_slice(&cell_indices()), wgpu::BufferUsages::INDEX);
        let ubuf = |label| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: std::mem::size_of::<DrawParams>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let draw_ubuf = ubuf("tetra draw params");
        let ghost_ubuf = ubuf("tetra ghost params");
        let pick_ubuf = ubuf("tetra pick params");
        let pick_z_ubuf = ubuf("tetra zero pick params");
        let tex = |label, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor { label: Some(label), size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] })
        };
        let pick_color = tex("pick id", PICK_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let pick_depth = tex("pick depth", DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let pick = PickTarget {
            color_view: pick_color.create_view(&Default::default()),
            color: pick_color,
            depth_view: pick_depth.create_view(&Default::default()),
            staging: device.create_buffer(&wgpu::BufferDescriptor { label: Some("pick staging"), size: 256, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
        };
        TetraRenderer {
            cells_layout,
            draw_layout,
            cells_pipe: p.cells,
            cube_pipe: p.cube,
            solid_pipe: p.solid,
            ghost_pipe: p.ghost,
            pick_pipe: p.pick,
            line_pipe: p.line,
            sphere_pipe: p.sphere,
            sphere_pick_pipe: p.sphere_pick,
            index_buf,
            draw_ubuf,
            ghost_ubuf,
            pick_ubuf,
            pick_z_ubuf,
            lines: None,
            scene: None,
            target: None,
            pick,
            pick_pending: None,
            pick_mapping: false,
            pick_failed: Arc::new(AtomicBool::new(false)),
            device,
            queue,
        }
    }

    /// Most cubes one scene can hold here: [`MAX_CELLS`], or less when the device's
    /// largest storage binding or buffer (8 bytes a cube) is smaller.
    pub fn max_cells(&self) -> u32 {
        let l = self.device.limits();
        let by_dev = l.max_storage_buffer_binding_size.min(l.max_buffer_size) / 8;
        (MAX_CELLS as u64).min(by_dev) as u32
    }

    /// Upload a new selection and run the residue pass over it. `mods[i]` is the modulus
    /// of slice i.
    pub fn set_scene(&mut self, layout: &Layout, mods: &[Modulus], palette: &Palette, offset: f32) {
        let data = SceneData::new(layout, mods);
        let d = &*self.device;
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let tables = init(d, "tetra tables", bytemuck::cast_slice(&data.tables), st);
        let slices = init(d, "tetra slices", bytemuck::cast_slice(&data.slices), st);
        let factors = init(d, "tetra factors", bytemuck::cast_slice(&data.factors), st);
        let levels = init(d, "tetra levels", bytemuck::cast_slice(&level_words(layout, palette.scheme == Scheme::Spectral)), st);
        let colors = init(d, "tetra colors", bytemuck::cast_slice(&color_words(layout, palette, offset)), st);
        let cap = (layout.total.max(1) as u64) * 8;
        let out = |label| d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: cap, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        // one buffer: nonzero cubes from the front, zeros from the back
        let cubes = out("tetra cubes");
        let counters = init(d, "tetra draw args", bytemuck::cast_slice(&[36u32, 0, 0, 0, 0, 36, 0, 0, 0, 0]), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        let groups = layout.total.div_ceil(WG);
        let gx = groups.clamp(1, MAX_GROUPS_X);
        let gy = groups.div_ceil(gx).max(1);
        let params = CellParams { n_slices: layout.slices.len() as u32, stride: layout.stride, total: layout.total, row_threads: gx * WG, flags: 0, pad: [0; 3] };
        let pbuf = init(d, "tetra cell params", bytemuck::bytes_of(&params), wgpu::BufferUsages::UNIFORM);
        let cells_bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tetra cells"),
            layout: &self.cells_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tables.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: slices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: factors.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cubes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: counters.as_entire_binding() },
            ],
        });
        if layout.total > 0 {
            let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tetra cells") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("tetra cells"), timestamp_writes: None });
                pass.set_pipeline(&self.cells_pipe);
                pass.set_bind_group(0, &cells_bg, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            self.queue.submit(Some(enc.finish()));
        }
        let draw_bg = |ubuf: &wgpu::Buffer, cubes: &wgpu::Buffer| {
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tetra draw"),
                layout: &self.draw_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: slices.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: levels.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: colors.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: cubes.as_entire_binding() },
                ],
            })
        };
        let draw_nz = draw_bg(&self.draw_ubuf, &cubes);
        let draw_z = draw_bg(&self.ghost_ubuf, &cubes);
        let pick_nz = draw_bg(&self.pick_ubuf, &cubes);
        let pick_z = draw_bg(&self.pick_z_ubuf, &cubes);
        self.scene = Some(Scene { layout: layout.clone(), gpu_slices: data.slices.clone(), slices, spectral: palette.scheme == Scheme::Spectral, levels, colors, cubes, counters, draw_nz, draw_z, pick_nz, pick_z });
    }

    /// Move the slices of the same scene (the diamond's hinge swinging): only their
    /// placements change, the residues stay.
    pub fn set_placement(&mut self, layout: &Layout) {
        let Some(s) = &mut self.scene else { return };
        if s.layout.slices.len() != layout.slices.len() {
            return;
        }
        for (i, (g, l)) in s.gpu_slices.iter_mut().zip(&layout.slices).enumerate() {
            if g.origin != l.origin || g.nvec != l.nvec {
                g.place(l);
                let size = std::mem::size_of::<GpuSlice>();
                self.queue.write_buffer(&s.slices, (i * size) as u64, bytemuck::bytes_of(g));
            }
        }
        s.layout = layout.clone();
    }

    /// New palette or color offset for the same selection.
    pub fn set_colors(&mut self, palette: &Palette, offset: f32) {
        let Some(s) = &mut self.scene else { return };
        let words = color_words(&s.layout, palette, offset);
        self.queue.write_buffer(&s.colors, 0, bytemuck::cast_slice(&words));
        // the levels depend on the scheme only through spectral's white 1
        let spectral = palette.scheme == Scheme::Spectral;
        if spectral != s.spectral {
            self.queue.write_buffer(&s.levels, 0, bytemuck::cast_slice(&level_words(&s.layout, spectral)));
            s.spectral = spectral;
        }
    }

    pub fn has_scene(&self) -> bool {
        self.scene.is_some()
    }

    /// The outlines and dashes, in world space.
    pub fn set_lines(&mut self, verts: &[LineVertex]) {
        self.lines = (!verts.is_empty()).then(|| (init(&self.device, "tetra lines", bytemuck::cast_slice(verts), wgpu::BufferUsages::VERTEX), verts.len() as u32));
    }

    /// Size the render target; true when it was (re)created (re-register it with egui).
    pub fn ensure_target(&mut self, w: u32, h: u32) -> bool {
        let (w, h) = (w.max(1), h.max(1));
        if self.target.as_ref().is_some_and(|t| t.w == w && t.h == h) {
            return false;
        }
        let tex = |label, format, usage| {
            self.device.create_texture(&wgpu::TextureDescriptor { label: Some(label), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] })
        };
        let color = tex("tetra color", COLOR_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let depth = tex("tetra depth", DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT);
        self.target = Some(Target { w, h, color_view: color.create_view(&Default::default()), color, depth_view: depth.create_view(&Default::default()) });
        true
    }

    pub fn target_view(&self) -> Option<&wgpu::TextureView> {
        self.target.as_ref().map(|t| &t.color_view)
    }

    pub fn target_size(&self) -> (u32, u32) {
        self.target.as_ref().map_or((0, 0), |t| (t.w, t.h))
    }

    fn params(&self, o: &DrawOptions, view_proj: Mat4, ghost: bool) -> DrawParams {
        self.params_for(o, view_proj, ghost, false)
    }

    /// `zero`: the solid zeros pass (the zeros, drawn as dark cubes).
    fn params_for(&self, o: &DrawOptions, view_proj: Mat4, ghost: bool, zero: bool) -> DrawParams {
        let s = self.scene.as_ref().expect("scene");
        let he = s.layout.half_extents();
        let (hover, hover_slice) = match o.hover {
            Some(c) if (c.slice as usize) < s.layout.slices.len() => {
                let p = s.layout.center(c.slice as usize, c.n, c.k);
                ([p[0], p[1], p[2], 1.0], c.slice)
            }
            _ => ([0.0; 4], u32::MAX),
        };
        let cell = s.layout.draws(o.cell);
        let st = s.layout.stride as f32;
        let (he, mut flags) = match cell {
            Cell::Cube => (he, 0),
            Cell::Rhombic => ([st; 3], FLAG_RHOMBIC),
            Cell::Sphere => ([0.5 * st; 3], FLAG_SPHERE),
        };
        if s.layout.turned() {
            flags |= FLAG_TURNED;
        }
        if o.color_slice {
            flags |= FLAG_SLICE;
        }
        if o.mask {
            flags |= FLAG_MASK;
        }
        if ghost {
            flags |= FLAG_GHOST;
        }
        if o.flat {
            flags |= FLAG_FLAT;
        }
        if zero {
            flags |= FLAG_ZERO;
        }
        let brush = o.brush.max(0.0) * s.layout.stride as f32;
        let (st, d) = (s.layout.stride, o.peel);
        let last = s.layout.total.saturating_sub(1);
        let (peel_a, peel_b) = if s.layout.view != View::Diamond {
            ([d[0] * st, d[1] * st, d[2] * st, d[3]], [d[4], s.layout.layers, st, last])
        } else {
            ([0; 4], [0, 0, st, last])
        };
        let (f, r, u) = o.camera.basis();
        let e = o.camera.eye();
        let persp = if o.camera.ortho > 0.0 { 0.0 } else { 1.0 };
        DrawParams { view_proj, hover, half: [he[0], he[1], he[2], brush], wire: if ghost { [0.55, 0.75, 0.6, 1.0] } else { WIRE }, hover_slice, flags, line_px: 0.75, ghost_alpha: 0.07, fill: [o.fill.clamp(-1e30, 1e30), 0.0, 0.0, 0.0], peel_a, peel_b, eye: [e[0], e[1], e[2], persp], fwd: [f[0], f[1], f[2], 0.0], right: [r[0], r[1], r[2], 0.0], up: [u[0], u[1], u[2], 0.0] }
    }

    /// Point both indirect draws (the nonzero cubes, the zeros) at `cell`'s indices; the
    /// instance counts the compute pass wrote stay.
    fn use_cell(&self, s: &Scene, cell: Cell, camera: &Camera) {
        let (count, first) = cell_range(s.layout.draws(cell), flat_view(camera));
        for base in [0u64, 20] {
            self.queue.write_buffer(&s.counters, base, bytemuck::bytes_of(&count));
            self.queue.write_buffer(&s.counters, base + 8, bytemuck::bytes_of(&first));
        }
    }

    /// Draw the stack into the target: the cubes, the outlines, then the ghosts.
    pub fn draw(&self, enc: &mut wgpu::CommandEncoder, o: &DrawOptions) {
        let Some(t) = &self.target else { return };
        let aspect = t.w as f32 / t.h as f32;
        let vp = o.camera.view_proj(aspect);
        let bg = wgpu::Color { r: BACKGROUND[0] as f64, g: BACKGROUND[1] as f64, b: BACKGROUND[2] as f64, a: 1.0 };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tetra"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &t.color_view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(bg), store: wgpu::StoreOp::Store } })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &t.depth_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let Some(s) = &self.scene else { return };
        let p = self.params(o, vp, false);
        self.queue.write_buffer(&self.draw_ubuf, 0, bytemuck::bytes_of(&p));
        self.use_cell(s, o.cell, &o.camera);
        // (in a planar view a sphere is a flat disk at its centre's depth: no depth written)
        let sphere = s.layout.draws(o.cell) == Cell::Sphere && !flat_view(&o.camera);
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
        // all filled: no wire to see through, so back faces can go
        let solid = o.fill == f32::NEG_INFINITY;
        pass.set_pipeline(if sphere { &self.sphere_pipe } else if solid { &self.solid_pipe } else { &self.cube_pipe });
        pass.set_bind_group(0, &s.draw_nz, &[]);
        pass.draw_indexed_indirect(&s.counters, 0);
        if o.solid_zeros {
            // the zeros as dark cubes (the ghosts' uniforms are free: they are not drawn)
            let pz = self.params_for(o, vp, false, true);
            self.queue.write_buffer(&self.ghost_ubuf, 0, bytemuck::bytes_of(&pz));
            pass.set_pipeline(if sphere { &self.sphere_pipe } else { &self.solid_pipe });
            pass.set_bind_group(0, &s.draw_z, &[]);
            pass.draw_indexed_indirect(&s.counters, 20);
        }
        if let Some((buf, n)) = &self.lines {
            pass.set_pipeline(&self.line_pipe);
            pass.set_vertex_buffer(0, buf.slice(..));
            pass.draw(0..*n, 0..1);
        }
        drop(pass);
        if o.ghosts && !o.solid_zeros {
            self.draw_ghosts(enc, o, vp);
        }
    }

    /// The zeros as faint additive cubes over the drawn stack, with their own uniforms
    /// (the ghost flag). Not depth-tested: the wire in front would hide almost all of
    /// them, and the point is to see the tunnels through it as a haze.
    fn draw_ghosts(&self, enc: &mut wgpu::CommandEncoder, o: &DrawOptions, vp: Mat4) {
        let (Some(t), Some(s)) = (&self.target, &self.scene) else { return };
        let p = self.params(o, vp, true);
        self.queue.write_buffer(&self.ghost_ubuf, 0, bytemuck::bytes_of(&p));
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tetra ghosts"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &t.color_view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &t.depth_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }), stencil_ops: None }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_pipeline(&self.ghost_pipe);
        pass.set_bind_group(0, &s.draw_z, &[]);
        pass.draw_indexed_indirect(&s.counters, 20);
    }

    /// Encode the ID pass for pixel (px, py) of the target and copy its result out, unless
    /// a pick is still in flight. Call [`Self::start_pick_map`] after submitting.
    pub fn encode_pick(&mut self, enc: &mut wgpu::CommandEncoder, camera: &Camera, px: f32, py: f32, spec: PickSpec) -> bool {
        let PickSpec { peel, zeros, cell } = spec;
        if self.pick_pending.is_some() {
            return false;
        }
        let (Some(t), Some(s)) = (&self.target, &self.scene) else { return false };
        let o = DrawOptions { camera: *camera, fill: f32::INFINITY, hover: None, brush: 0.0, color_slice: false, ghosts: false, mask: false, flat: false, peel, solid_zeros: false, cell };
        self.use_cell(s, cell, camera);
        let sphere = s.layout.draws(cell) == Cell::Sphere && !flat_view(camera);
        let vp = camera.pick_view_proj(t.w, t.h, px, py);
        let p = self.params(&o, vp, false);
        self.queue.write_buffer(&self.pick_ubuf, 0, bytemuck::bytes_of(&p));
        if zeros {
            let pz = self.params_for(&o, vp, false, true);
            self.queue.write_buffer(&self.pick_z_ubuf, 0, bytemuck::bytes_of(&pz));
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tetra pick"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &self.pick.color_view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &self.pick.depth_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_pipeline(if sphere { &self.sphere_pick_pipe } else { &self.pick_pipe });
            pass.set_bind_group(0, &s.pick_nz, &[]);
            pass.draw_indexed_indirect(&s.counters, 0);
            if zeros {
                pass.set_bind_group(0, &s.pick_z, &[]);
                pass.draw_indexed_indirect(&s.counters, 20);
            }
        }
        enc.copy_texture_to_buffer(
            self.pick.color.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &self.pick.staging, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: None } },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        self.pick_pending = Some(Arc::new(AtomicBool::new(false)));
        true
    }

    /// After the submit that carried [`Self::encode_pick`]: start mapping the result.
    pub fn start_pick_map(&mut self) {
        if self.pick_mapping {
            return;
        }
        let Some(done) = self.pick_pending.clone() else { return };
        self.pick_mapping = true;
        let failed = self.pick_failed.clone();
        self.pick.staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            if r.is_err() {
                failed.store(true, Ordering::Release);
            }
            done.store(true, Ordering::Release);
        });
    }

    /// A pick is being read back (the app keeps drawing until it lands).
    pub fn pick_in_flight(&self) -> bool {
        self.pick_pending.is_some()
    }

    /// The finished pick, if any: `Some(None)` = nothing under the pixel.
    pub fn poll_pick(&mut self) -> Option<Option<CubeId>> {
        let _ = self.device.poll(wgpu::PollType::Poll);
        let done = self.pick_pending.as_ref()?.load(Ordering::Acquire);
        if !done {
            return None;
        }
        self.pick_pending = None;
        self.pick_mapping = false;
        if self.pick_failed.swap(false, Ordering::AcqRel) {
            return Some(None);
        }
        let w = {
            let data = self.pick.staging.slice(..8).get_mapped_range().expect("mapped");
            let v: Vec<u32> = bytemuck::pod_collect_to_vec(&data);
            [v[0], v[1]]
        };
        self.pick.staging.unmap();
        Some((w[1] & 0x8000_0000 != 0).then(|| CubeId::unpack(w)))
    }

    /// Pick synchronously (tests).
    pub fn pick_blocking(&mut self, camera: &Camera, px: f32, py: f32) -> Option<CubeId> {
        self.pick_blocking_peeled(camera, px, py, [0; 5])
    }

    /// Pick synchronously through a peel (tests).
    pub fn pick_blocking_peeled(&mut self, camera: &Camera, px: f32, py: f32, peel: [u32; 5]) -> Option<CubeId> {
        self.pick_blocking_with(camera, px, py, peel, false)
    }

    /// Pick synchronously, the zeros too when `zeros` (the solid zeros of the flat views).
    pub fn pick_blocking_with(&mut self, camera: &Camera, px: f32, py: f32, peel: [u32; 5], zeros: bool) -> Option<CubeId> {
        self.pick_blocking_cell(camera, px, py, peel, zeros, Cell::Cube)
    }

    /// Pick synchronously with the cells drawn as `cell` (tests).
    pub fn pick_blocking_cell(&mut self, camera: &Camera, px: f32, py: f32, peel: [u32; 5], zeros: bool, cell: Cell) -> Option<CubeId> {
        let mut enc = self.device.create_command_encoder(&Default::default());
        assert!(self.encode_pick(&mut enc, camera, px, py, PickSpec { peel, zeros, cell }), "no pick in flight");
        self.queue.submit(Some(enc.finish()));
        self.start_pick_map();
        loop {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            if let Some(r) = self.poll_pick() {
                return r;
            }
        }
    }

    fn read(&self, buf: &wgpu::Buffer, size: u64) -> Vec<u8> {
        self.read_at(buf, 0, size)
    }

    fn read_at(&self, buf: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
        if size == 0 {
            return Vec::new();
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("tetra readback"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, offset, &staging, 0, size);
        self.queue.submit(Some(enc.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let v = staging.slice(..).get_mapped_range().expect("mapped").to_vec();
        staging.unmap();
        v
    }

    /// The instance counts the compute pass wrote: (nonzero cubes, zeros).
    pub fn read_counts(&self) -> (u32, u32) {
        let Some(s) = &self.scene else { return (0, 0) };
        let w: Vec<u32> = bytemuck::pod_collect_to_vec(&self.read(&s.counters, 40));
        (w[1], w[6])
    }

    /// Every cube the compute pass wrote: (nonzero, zeros), as packed words.
    pub fn read_cubes(&self) -> (Vec<[u32; 2]>, Vec<[u32; 2]>) {
        let Some(s) = &self.scene else { return (Vec::new(), Vec::new()) };
        let (nz, z) = self.read_counts();
        let total = s.layout.total as u64;
        let front = bytemuck::pod_collect_to_vec::<u8, [u32; 2]>(&self.read_at(&s.cubes, 0, nz as u64 * 8));
        // the zeros sit at the back, the first one written last of all
        let mut back = bytemuck::pod_collect_to_vec::<u8, [u32; 2]>(&self.read_at(&s.cubes, (total - z as u64) * 8, z as u64 * 8));
        back.reverse();
        (front, back)
    }

    /// The target as tightly packed RGBA8 (tests).
    pub fn read_target_rgba(&self) -> (u32, u32, Vec<u8>) {
        let t = self.target.as_ref().expect("target");
        let bpr = (t.w * 4).div_ceil(256) * 256;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("tetra tex readback"), size: (bpr * t.h) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            t.color.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &staging, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: None } },
            wgpu::Extent3d { width: t.w, height: t.h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(enc.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let data = staging.slice(..).get_mapped_range().expect("mapped");
        let mut out = Vec::with_capacity((t.w * t.h * 4) as usize);
        for y in 0..t.h as usize {
            out.extend_from_slice(&data[y * bpr as usize..y * bpr as usize + t.w as usize * 4]);
        }
        drop(data);
        staging.unmap();
        (t.w, t.h, out)
    }

    /// Draw and wait (tests).
    pub fn draw_blocking(&self, o: &DrawOptions) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.draw(&mut enc, o);
        self.queue.submit(Some(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }
}

/// The outlines: a faint solid triangle around every slice and, in true spacing, a dashed
/// one for every skipped modulus.
pub fn outline_vertices(layout: &Layout) -> Vec<LineVertex> {
    let mut v = Vec::new();
    let mut tri = |pts: [[f32; 3]; 3], color: [f32; 4], dashed: bool| {
        for e in 0..3 {
            let (a, b) = (pts[e], pts[(e + 1) % 3]);
            let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
            // dashes about 1.5 units long, whole numbers of them per edge
            let dashes = if dashed { (len / 1.5).round().max(1.0) } else { -1.0 };
            v.push(LineVertex { pos: a, color, dash: if dashed { 0.0 } else { -1.0 } });
            v.push(LineVertex { pos: b, color, dash: dashes });
        }
    };
    if layout.view != View::Stack {
        return v;
    }
    for s in &layout.slices {
        tri(layout.outline(s.m, s.height()), [0.32, 0.95, 0.48, 0.16], false);
    }
    for m in layout.missing() {
        tri(layout.outline(m, layout.height_of(m)), [0.45, 0.8, 0.55, 0.22], true);
    }
    v
}
