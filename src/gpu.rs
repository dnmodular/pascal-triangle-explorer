//! The wgpu pipeline: one compute dispatch per pass over the grid, in the order the
//! layer plan dictates, and a final resolve pass that writes the canvas texture.
//!
//! ```text
//! value ──► cells ──► measure_i ──► fields[i] ──► reduce_max ──► maxbuf[i]
//!                       (blurred measures: hex blur passes in between)
//! blend(cells, fields, maxbuf) ──► work[0]  (under)
//! AVERAGE   spatial: hex blur passes          temporal: mean with the history ring
//! BAND PASS blur(keep) − blur(cut) → max|d| → recenter on 0.5
//! AFTERIMAGE complement ghost from the adaptation buffer (stepped per new frame)
//! resolve(cells, work[cur]) ──► rgba8 texture   (live view and export alike)
//! ```
//!
//! The working field has two element types — `f32` (MATH RAMP) and `vec4<f32>`
//! (PALETTE BLEND) — so `field.wgsl` is compiled twice with a different prelude.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::layers::{AvgMode, FieldColorMode, LayerId, LayerState, MeasureRun, ValueStyle};
use crate::math::{to_mont, Modulus, MAX_FACTORS};
use crate::palette::{residue_levels_packed, Palette, Scheme};
use crate::view::GridSpec;

const WG: u32 = 16;
/// Field slots available to measures.
pub const MAX_MEASURE_SLOTS: u32 = 8;
/// `maxbuf` entry used by BAND PASS normalization.
const BAND_MAX_SLOT: u32 = 8;
const MAXBUF_LEN: u64 = 16;
const PASS_STRIDE: u64 = 256;
const MAX_PASSES: u64 = 2048;
/// `maxbuf` counters used by DIST→0 (zeros in view) and RARITY (cells in view).
const ZERO_COUNT_SLOT: u32 = 12;
const CELL_COUNT_SLOT: u32 = 13;
/// Auxiliary scalar fields for multi-pass measures (plus one blur temporary in `tmp_buf`).
const AUX_FIELDS: u64 = 3;
/// Temporal AVERAGE keeps at most this many past frames.
pub const HIST_CAP: u32 = 64;


#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuFactor {
    pub p: u32,
    pub e: u32,
    pub q: u32,
    pub qinv: u32,
    pub one: u32,
    pub p_mont: u32,
    pub tab_off: u32,
    pub minus_one: u32,
    pub garner_q: u32,
    pub garner_inv: u32,
    pub ppow_off: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuParams {
    pub grid_w: u32,
    pub grid_h: u32,
    pub canvas_w: u32,
    pub canvas_h: u32,
    pub n0: i32,
    pub x0i: i32,
    pub ss: u32,
    pub nfactors: u32,
    pub y0f: f32,
    pub x0f: f32,
    pub sx: f32,
    pub sy: f32,
    pub cells_per_px: f32,
    pub rows_per_px: f32,
    pub lnf_off: u32,
    pub prank_off: u32,
    pub m: u32,
    pub p: u32,
    pub dlog_off: u32,
    pub blend_mode: u32,
    pub color_mode: u32,
    pub wants_value: u32,
    pub n_measures: u32,
    pub value_style: u32,
    pub field_active: u32,
    pub field_stride: u32,
    pub signed_mask: u32,
    pub res_shift: u32,
    pub style_flags: u32,
    pub glow_rgb: u32,
    pub pad_e: u32,
    pub pad_f: u32,
    pub dparams: [[u32; 4]; 8],
    pub factors: [GpuFactor; MAX_FACTORS],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PassParams {
    pub slot: u32,
    pub count: u32,
    pub mode: u32,
    pub flag: u32,
    pub f0: f32,
    pub f1: f32,
    pub f2: f32,
    pub f3: f32,
    pub hist_start: u32,
    pub hist_len: u32,
    pub hist_cap: u32,
    pub stride: u32,
}

/// Tables and constants for one modulus, as uploaded.
pub struct ModulusGpuData {
    pub factors: [GpuFactor; MAX_FACTORS],
    pub nfactors: u32,
    pub tables: Vec<u32>,
    pub dlog_off: u32,
    pub lnf_off: u32,
    pub prank_off: u32,
}

/// `ln(i!)` for `i < LNF_TABLE`, rounded to f32 (shared by the shader and the tests).
pub const LNF_TABLE: usize = 4096;
pub fn ln_factorial_table() -> &'static [f32] {
    static T: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut acc = 0f64;
        (0..LNF_TABLE)
            .map(|i| {
                if i >= 2 {
                    acc += (i as f64).ln();
                }
                acc as f32
            })
            .collect()
    })
}

impl ModulusGpuData {
    pub fn new(md: &Modulus) -> Self {
        let mut factors = [GpuFactor::default(); MAX_FACTORS];
        let mut tables = Vec::new();
        for (i, t) in md.factors.iter().enumerate() {
            let q = t.f.q;
            let odd = t.f.p != 2;
            let conv = |x: u32| if odd { to_mont(x, q) } else { x };
            let tab_off = tables.len() as u32;
            tables.extend(t.fp.iter().map(|&x| conv(x)));
            tables.extend(t.ifp.iter().map(|&x| conv(x)));
            // p^t mod q for t < e, so the shader never loops to form a power
            let ppow_off = tables.len() as u32;
            let mut pw = 1u64;
            for _ in 0..t.f.e {
                tables.push(conv(pw as u32));
                pw = pw * t.f.p as u64 % q as u64;
            }
            factors[i] = GpuFactor {
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
            };
        }
        let dlog_off = tables.len() as u32;
        tables.extend_from_slice(&md.dlog);
        // ln(i!) for i < LNF_TABLE, as f32 bits (MAGNITUDE)
        let lnf_off = tables.len() as u32;
        tables.extend(ln_factorial_table().iter().map(|v| v.to_bits()));
        // 1-based prime index of every residue 0..m (PRIMES)
        let prank_off = tables.len() as u32;
        let pt = crate::sweep::prime_tables();
        tables.extend((0..md.m).map(|v| pt.rank(v)));
        if tables.len() % 4 != 0 {
            tables.resize(tables.len() + 4 - tables.len() % 4, 0);
        }
        Self { factors, nfactors: md.factors.len() as u32, tables, dlog_off, lnf_off, prank_off }
    }
}

/// Adapter, device and queue.
pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub info: wgpu::AdapterInfo,
}

impl Gpu {
    /// A device without a window (tests, CI). Prefers a real GPU, accepts a software one.
    /// `WGPU_BACKEND` (e.g. `vulkan`, `gl`) selects a backend; otherwise all are tried.
    pub fn new_headless() -> Result<Gpu, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        pollster::block_on(Self::new_async(instance, None))
    }

    pub async fn new_async(instance: wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<Gpu, String> {
        let adapter = Self::request_adapter(&instance, surface).await?;
        let info = adapter.get_info();
        let (device, queue) = Self::request_device(&adapter).await?;
        Ok(Gpu { instance, adapter, device: Arc::new(device), queue: Arc::new(queue), info })
    }

    pub async fn request_adapter(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<wgpu::Adapter, String> {
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: surface,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no suitable GPU adapter: {e}"))
    }

    /// A device with the limits this renderer wants (large storage bindings) and
    /// timestamp queries when the adapter has them.
    pub async fn request_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue), String> {
        let al = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
            max_buffer_size: al.max_buffer_size,
            max_storage_buffers_per_shader_stage: al.max_storage_buffers_per_shader_stage.clamp(8, 16),
            max_texture_dimension_2d: al.max_texture_dimension_2d,
            ..Default::default()
        };
        let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("pascal"),
                required_features: features,
                required_limits: limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("request_device failed: {e}"))
    }
}

/// Which pipeline layout an entry point uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayoutKind {
    Value,
    Scalar,
    Field,
    Tween,
}

/// One shader module exactly as the renderer compiles it, with its entry points.
pub struct ModuleSource {
    pub label: &'static str,
    pub source: String,
    pub entries: Vec<(&'static str, LayoutKind)>,
}

/// The four modules: value, measures, and field.wgsl in its two element types.
pub fn module_sources() -> Vec<ModuleSource> {
    let measure_entries: Vec<(&'static str, LayoutKind)> = crate::layers::REGISTRY
        .iter()
        .filter_map(|d| d.run)
        .flat_map(|run| run.entries())
        .map(|entry| (entry, LayoutKind::Scalar))
        .chain([("reduce_max", LayoutKind::Scalar), ("blur_scalar", LayoutKind::Scalar)])
        .collect();
    let field_entries: Vec<(&'static str, LayoutKind)> =
        ["blend_main", "blur_field", "temporal_main", "band_diff", "reduce_max_abs", "normalize_signed", "afterimage_main", "tween_capture", "resolve_main"].into_iter().map(|e| (e, LayoutKind::Field)).collect();
    vec![
        ModuleSource { label: "value", source: format!("{COMMON_WGSL}\n{VALUE_WGSL}"), entries: vec![("value_main", LayoutKind::Value)] },
        ModuleSource { label: "measures", source: format!("{COMMON_WGSL}\n{}\n{MEASURES_WGSL}", dslot_constants()), entries: measure_entries },
        ModuleSource { label: "field-math", source: format!("{COMMON_WGSL}\n{}", FIELD_WGSL.replace("{{FIELD_PRELUDE}}", MATH_PRELUDE)), entries: field_entries.clone() },
        ModuleSource { label: "field-palette", source: format!("{COMMON_WGSL}\n{}", FIELD_WGSL.replace("{{FIELD_PRELUDE}}", PALETTE_PRELUDE)), entries: field_entries },
        ModuleSource { label: "tween", source: TWEEN_WGSL.to_string(), entries: vec![("tween_main", LayoutKind::Tween)] },
    ]
}

/// `const DSLOT_<ID>: u32 = n;` for every registry entry with a `derive`, so the
/// shader's slot numbers come from the registry and cannot drift from it.
pub fn dslot_constants() -> String {
    crate::layers::REGISTRY
        .iter()
        .filter(|d| d.derive.is_some())
        .map(|d| format!("const DSLOT_{}: u32 = {}u;\n", format!("{:?}", d.id).to_uppercase(), d.dslot))
        .collect()
}

/// Bind group and pipeline layouts shared by every pass.
struct Layouts {
    bgl0: wgpu::BindGroupLayout,
    bgl_scalar: wgpu::BindGroupLayout,
    bgl_field: wgpu::BindGroupLayout,
    bgl_tween: wgpu::BindGroupLayout,
    value: wgpu::PipelineLayout,
    scalar: wgpu::PipelineLayout,
    field: wgpu::PipelineLayout,
    tween: wgpu::PipelineLayout,
}

impl Layouts {
    fn new(device: &wgpu::Device) -> Layouts {
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shared"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
        });
        let pass_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: NonZeroU64::new(std::mem::size_of::<PassParams>() as u64),
            },
            count: None,
        };
        let bgl_scalar = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scalar passes"),
            entries: &[storage_entry(0, false), storage_entry(1, false), storage_entry(2, false), storage_entry(3, false), pass_entry(4), storage_entry(5, false)],
        });
        let bgl_field = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("field passes"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
                pass_entry(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let bgl_tween = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tween"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let tween = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("tween"), bind_group_layouts: &[Some(&bgl_tween)], immediate_size: 0 });
        let value = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("value"), bind_group_layouts: &[Some(&bgl0)], immediate_size: 0 });
        let scalar = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("scalar"), bind_group_layouts: &[Some(&bgl0), Some(&bgl_scalar)], immediate_size: 0 });
        let field = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("field"), bind_group_layouts: &[Some(&bgl0), Some(&bgl_field)], immediate_size: 0 });
        Layouts { bgl0, bgl_scalar, bgl_field, bgl_tween, value, scalar, field, tween }
    }

    fn for_kind(&self, kind: LayoutKind) -> &wgpu::PipelineLayout {
        match kind {
            LayoutKind::Value => &self.value,
            LayoutKind::Scalar => &self.scalar,
            LayoutKind::Field => &self.field,
            LayoutKind::Tween => &self.tween,
        }
    }
}

/// `PASCAL_UNCHECKED_SHADERS=1` compiles the shaders without naga's runtime-check
/// wrappers (index clamps, forced loop bounding). The shaders bound every index
/// themselves, so this is a legitimate workaround when a driver's compiler rejects
/// the wrapped form; `--check-shaders` reports whether it helps.
pub fn unchecked_shaders_requested() -> bool {
    std::env::var("PASCAL_UNCHECKED_SHADERS").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false)
}

fn create_module(device: &wgpu::Device, label: &str, source: &str, unchecked: bool) -> wgpu::ShaderModule {
    let desc = wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(source.to_owned().into()) };
    if unchecked {
        // SAFETY: every buffer index in these shaders is range-checked in WGSL before use
        // (grid bounds, table offsets derived from the uploaded sizes), and every loop
        // terminates by construction (digit loops divide by p ≥ 2 each step).
        unsafe { device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) }
    } else {
        device.create_shader_module(desc)
    }
}

fn make_pipeline(device: &wgpu::Device, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Result of compiling one entry point on the driver.
pub struct ShaderCheck {
    pub module: &'static str,
    pub entry: &'static str,
    /// With naga's runtime checks (the default build).
    pub result: Result<(), String>,
    /// Retry without the runtime-check wrappers, only when `result` failed.
    pub unchecked_result: Option<Result<(), String>>,
}

fn try_compile(adapter: &wgpu::Adapter, m: &ModuleSource, entry: &str, kind: LayoutKind, unchecked: bool) -> Result<(), String> {
    let (device, _queue) = pollster::block_on(Gpu::request_device(adapter))?;
    let lost = Arc::new(std::sync::Mutex::new(None::<String>));
    let lost2 = lost.clone();
    device.set_device_lost_callback(move |reason, msg| {
        *lost2.lock().unwrap() = Some(format!("device lost ({reason:?}): {msg}"));
    });
    let scope_validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let scope_internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let module = create_module(&device, m.label, &m.source, unchecked);
    let layouts = Layouts::new(&device);
    let _pipe = make_pipeline(&device, layouts.for_kind(kind), &module, entry);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let internal = pollster::block_on(scope_internal.pop());
    let validation = pollster::block_on(scope_validation.pop());
    if let Some(e) = validation {
        return Err(format!("{e}"));
    }
    if let Some(e) = internal {
        return Err(format!("{e}"));
    }
    if let Some(l) = lost.lock().unwrap().take() {
        return Err(l);
    }
    Ok(())
}

/// Compile every pipeline one at a time, each on a fresh device and inside error
/// scopes, so a driver that rejects a shader (e.g. NVIDIA's "NVVM compilation
/// failed") names the entry point instead of silently losing the device.
pub fn check_shaders(adapter: &wgpu::Adapter) -> Vec<ShaderCheck> {
    let mut out = Vec::new();
    for m in module_sources() {
        for (entry, kind) in &m.entries {
            let result = try_compile(adapter, &m, entry, *kind, false);
            let unchecked_result = if result.is_err() { Some(try_compile(adapter, &m, entry, *kind, true)) } else { None };
            out.push(ShaderCheck { module: m.label, entry, result, unchecked_result });
        }
    }
    out
}

struct FieldPipes {
    blend: wgpu::ComputePipeline,
    blur: wgpu::ComputePipeline,
    temporal: wgpu::ComputePipeline,
    band_diff: wgpu::ComputePipeline,
    reduce_max_abs: wgpu::ComputePipeline,
    normalize: wgpu::ComputePipeline,
    afterimage: wgpu::ComputePipeline,
    capture: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
}

/// What the renderer needs for one frame.
pub struct FrameInput<'a> {
    pub grid: GridSpec,
    pub modulus: &'a Modulus,
    pub palette: &'a Palette,
    pub layers: &'a LayerState,
    /// The frame counts as new for the history-keeping layers (temporal average,
    /// afterimage). False when only the UI repainted.
    pub advance: bool,
    /// Display-only transition into this frame (see [`Tween`]).
    pub tween: Tween,
    /// Color offset and cell decorations (part of the true frame: exports include them).
    pub style: CanvasStyle,
}

/// How the resolve pass colors and decorates cells: the appearance options of
/// `reference/ChatGPT-version-pascal-square-scanner`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct CanvasStyle {
    /// Fractional color offset: residue colors move this fraction along the palette
    /// ramp, wrapping (0 void and spectral's white 1 stay).
    pub offset: f32,
    /// A thin light outline around every cell once cells are a few pixels wide.
    pub outlines: bool,
    /// A soft halo in the palette's 0.8 color spilling from zero cells (where m divides
    /// C(n,k)) onto their neighbours, once cells are a few pixels wide.
    pub glow: bool,
}

/// `style_flags` bits.
pub const STYLE_OUTLINES: u32 = 1;
pub const STYLE_GLOW: u32 = 2;

/// A display-only transition between sweep frames. `e` is the eased progress 0..1.
/// `Start` marks the first frame of a new transition (it snapshots what is on screen);
/// `Continue` advances it. Exports render with `Off` and always see the true frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tween {
    Off,
    Start(f32),
    Continue(f32),
}

/// The canvas textures: the true frame the resolve pass writes, the display texture
/// the window shows (the frame, or a dissolve into it), and the dissolve's snapshot.
struct Canvas {
    frame: wgpu::Texture,
    frame_storage: wgpu::TextureView,
    frame_load: wgpu::TextureView,
    display: wgpu::Texture,
    display_storage: wgpu::TextureView,
    display_sampled: wgpu::TextureView,
    snap: wgpu::Texture,
    snap_load: wgpu::TextureView,
}

impl Canvas {
    fn new(device: &wgpu::Device, w: u32, h: u32) -> Canvas {
        let make = |label: &str, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage,
                view_formats: &[],
            })
        };
        let view = |t: &wgpu::Texture, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            t.create_view(&wgpu::TextureViewDescriptor { label: None, format: Some(format), usage: Some(usage), ..Default::default() })
        };
        use wgpu::TextureFormat::Rgba8Unorm;
        use wgpu::TextureUsages as U;
        let frame = make("canvas frame", U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC);
        let display = make("canvas display", U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC);
        let snap = make("canvas snapshot", U::TEXTURE_BINDING | U::COPY_DST);
        Canvas {
            frame_storage: view(&frame, Rgba8Unorm, U::STORAGE_BINDING),
            frame_load: view(&frame, Rgba8Unorm, U::TEXTURE_BINDING),
            display_storage: view(&display, Rgba8Unorm, U::STORAGE_BINDING),
            // a plain (non-sRGB) view: egui 0.36 samples textures as gamma-encoded bytes and,
            // on the app's non-sRGB surface, writes them out unchanged
            display_sampled: view(&display, Rgba8Unorm, U::TEXTURE_BINDING),
            snap_load: view(&snap, Rgba8Unorm, U::TEXTURE_BINDING),
            frame,
            display,
            snap,
        }
    }
}

struct EncodeOut {
    cur: usize,
    adapt_stepped: bool,
    hist_recorded: bool,
}

/// A sub-range of a buffer bound as one storage binding.
#[derive(Clone, Copy)]
struct Bind<'a> {
    buf: &'a wgpu::Buffer,
    offset: u64,
    size: u64,
}

impl<'a> Bind<'a> {
    fn whole(buf: &'a wgpu::Buffer) -> Self {
        Bind { buf, offset: 0, size: buf.size() }
    }
    fn resource(&self) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: self.buf, offset: self.offset, size: NonZeroU64::new(self.size) })
    }
}

pub struct Renderer {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    bgl0: wgpu::BindGroupLayout,
    bgl_scalar: wgpu::BindGroupLayout,
    bgl_field: wgpu::BindGroupLayout,
    pipe_value: wgpu::ComputePipeline,
    pipe_measures: HashMap<&'static str, wgpu::ComputePipeline>,
    pipe_reduce_max: wgpu::ComputePipeline,
    pipe_blur_scalar: wgpu::ComputePipeline,
    field_pipes: [FieldPipes; 2],

    params_buf: wgpu::Buffer,
    pass_buf: wgpu::Buffer,
    tables_buf: wgpu::Buffer,
    levels_buf: wgpu::Buffer,
    palette_buf: wgpu::Buffer,
    maxbuf: wgpu::Buffer,
    cells_buf: wgpu::Buffer,
    fields_buf: wgpu::Buffer,
    /// AUX_FIELDS scalar fields of `stride` elements each.
    scratch_buf: wgpu::Buffer,
    /// Blur and relaxation temporary (a separate buffer, so copies never alias).
    tmp_buf: wgpu::Buffer,
    /// Residue histogram (RARITY), atomic u32 per residue.
    counts_buf: wgpu::Buffer,
    work: [wgpu::Buffer; 4],
    hist_buf: wgpu::Buffer,
    adapt_buf: wgpu::Buffer,
    canvas: Canvas,
    bg0: wgpu::BindGroup,
    bgl_tween: wgpu::BindGroupLayout,
    pipe_tween: wgpu::ComputePipeline,
    tween_buf: wgpu::Buffer,
    /// The field shown at the start of a morph (`tween_capture` writes it).
    tween_prev: wgpu::Buffer,
    morphing: bool,
    dissolving: bool,
    frame_morph: Option<f32>,
    canvas_fresh: bool,
    last_e: f32,
    /// Grid and field variant of the last frame that ran the field pipeline.
    last_field: Option<(GridSpec, usize)>,

    grid: Option<GridSpec>,
    stride: u32,
    work_elem: u64,
    hist_cap: u32,
    hist_next: u32,
    hist_len: u32,
    adapt_valid: bool,
    uploaded_m: Option<(u32, bool)>,
    uploaded_scheme: Option<Scheme>,
    modulus_data: Option<ModulusGpuData>,
    pass_params: Vec<PassParams>,
    /// Which work buffer holds the final field after `render` (for tests).
    pub last_cur: usize,
    pub last_n_measures: u32,
    timing: Option<Timing>,
}

/// GPU timestamps around the frame's compute work (when the device supports them).
struct Timing {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    period_ns: f32,
    /// A readback is in flight (the staging buffer is mapped or being mapped).
    pending: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    last_ms: Option<f32>,
}

fn buffer(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage, mapped_at_creation: false })
}

const STORAGE_RW: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST).union(wgpu::BufferUsages::COPY_SRC);

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

const MATH_PRELUDE: &str = r#"
alias FT = f32;
fn ft_splat(x: f32) -> FT { return x; }
fn ft_max_abs(v: FT) -> f32 { return abs(v); }
fn ft_from_blend(b: f32, valid: bool, residue: u32, have_measures: bool) -> FT { return select(0.0, b, valid); }
fn ft_resolve_color(v: FT) -> vec3<f32> {
  let lvl = u32(floor(clamp(v, 0.0, 1.0) * 255.0 + 0.5));
  return unpack_rgb(palette[256u + lvl]);
}
"#;

const PALETTE_PRELUDE: &str = r#"
alias FT = vec4<f32>;
fn ft_splat(x: f32) -> FT { return vec4<f32>(x); }
fn ft_max_abs(v: FT) -> f32 { return max(max(abs(v.x), abs(v.y)), abs(v.z)); }
// The palette image: each cell's residue color, re-lit by the measures' blend as
// lightness (zeros keep their void color: measures never re-light them).
fn ft_from_blend(b: f32, valid: bool, residue: u32, have_measures: bool) -> FT {
  if (!valid) { return vec4<f32>(BG_COLOR, 0.0); }
  let lvl = shift_level(level_of(residue));
  if (have_measures && residue != 0u) {
    let q = floor(clamp(b, 0.0, 1.0) * 255.0 + 0.5);
    let h = bitcast<f32>(palette[512u + lvl]);
    let s = bitcast<f32>(palette[768u + lvl]);
    return vec4<f32>(hsl_to_rgb(h, s, (q / 255.0) * 0.92), 0.0);
  }
  return vec4<f32>(unpack_rgb(palette[lvl]), 0.0);
}
fn ft_resolve_color(v: FT) -> vec3<f32> {
  return floor(clamp(v.xyz, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0 + 0.5) / 255.0;
}
"#;

pub const COMMON_WGSL: &str = include_str!("../shaders/common.wgsl");
pub const VALUE_WGSL: &str = include_str!("../shaders/value.wgsl");
pub const MEASURES_WGSL: &str = include_str!("../shaders/measures.wgsl");
pub const FIELD_WGSL: &str = include_str!("../shaders/field.wgsl");
pub const TWEEN_WGSL: &str = include_str!("../shaders/tween.wgsl");

fn round_up(x: u64, to: u64) -> u64 {
    x.div_ceil(to) * to
}

impl Renderer {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Renderer {
        let layouts = Layouts::new(&device);
        let mut modules = HashMap::new();
        let mut pipes: HashMap<(&'static str, &'static str), wgpu::ComputePipeline> = HashMap::new();
        let unchecked = unchecked_shaders_requested();
        if unchecked {
            log::warn!("PASCAL_UNCHECKED_SHADERS is set: compiling shaders without naga's runtime checks");
        }
        for m in module_sources() {
            let module = create_module(&device, m.label, &m.source, unchecked);
            for (entry, kind) in &m.entries {
                pipes.insert((m.label, entry), make_pipeline(&device, layouts.for_kind(*kind), &module, entry));
            }
            modules.insert(m.label, module);
        }
        let take = |pipes: &mut HashMap<(&'static str, &'static str), wgpu::ComputePipeline>, m: &'static str, e: &'static str| pipes.remove(&(m, e)).unwrap_or_else(|| panic!("pipeline {m}::{e}"));
        let pipe_value = take(&mut pipes, "value", "value_main");
        let mut pipe_measures = HashMap::new();
        for d in crate::layers::REGISTRY {
            if let Some(run) = d.run {
                for entry in run.entries() {
                    pipe_measures.insert(entry, take(&mut pipes, "measures", entry));
                }
            }
        }
        let pipe_reduce_max = take(&mut pipes, "measures", "reduce_max");
        let pipe_blur_scalar = take(&mut pipes, "measures", "blur_scalar");
        let field_pipes = ["field-math", "field-palette"].map(|m| FieldPipes {
            blend: take(&mut pipes, m, "blend_main"),
            blur: take(&mut pipes, m, "blur_field"),
            temporal: take(&mut pipes, m, "temporal_main"),
            band_diff: take(&mut pipes, m, "band_diff"),
            reduce_max_abs: take(&mut pipes, m, "reduce_max_abs"),
            normalize: take(&mut pipes, m, "normalize_signed"),
            afterimage: take(&mut pipes, m, "afterimage_main"),
            capture: take(&mut pipes, m, "tween_capture"),
            resolve: take(&mut pipes, m, "resolve_main"),
        });
        let pipe_tween = take(&mut pipes, "tween", "tween_main");
        let Layouts { bgl0, bgl_scalar, bgl_field, bgl_tween, .. } = layouts;

        let params_buf = buffer(&device, "params", std::mem::size_of::<GpuParams>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let pass_buf = buffer(&device, "pass params", PASS_STRIDE * MAX_PASSES, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let tables_buf = buffer(&device, "tables", 1024, STORAGE_RW);
        let levels_buf = buffer(&device, "levels", 1024, STORAGE_RW);
        let palette_buf = buffer(&device, "palette", 4096, STORAGE_RW);
        let maxbuf = buffer(&device, "maxbuf", MAXBUF_LEN * 4, STORAGE_RW);
        let cells_buf = buffer(&device, "cells", 1024, STORAGE_RW);
        let fields_buf = buffer(&device, "fields", 1024, STORAGE_RW);
        let scratch_buf = buffer(&device, "scratch", 1024, STORAGE_RW);
        let counts_buf = buffer(&device, "counts", 1024, STORAGE_RW);
        let tmp_buf = buffer(&device, "tmp", 1024, STORAGE_RW);
        let work = [
            buffer(&device, "work0", 1024, STORAGE_RW),
            buffer(&device, "work1", 1024, STORAGE_RW),
            buffer(&device, "work2", 1024, STORAGE_RW),
            buffer(&device, "work3", 1024, STORAGE_RW),
        ];
        let hist_buf = buffer(&device, "history", 1024, STORAGE_RW);
        let adapt_buf = buffer(&device, "adapt", 1024, STORAGE_RW);
        let canvas = Canvas::new(&device, 16, 16);
        let tween_buf = buffer(&device, "tween", 16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let tween_prev = buffer(&device, "tween prev", 1024, STORAGE_RW);
        let bg0 = Self::make_bg0(&device, &bgl0, &params_buf, &tables_buf, &levels_buf, &palette_buf, &cells_buf, &maxbuf);

        Renderer {
            device,
            queue,
            bgl0,
            bgl_scalar,
            bgl_field,
            pipe_value,
            pipe_measures,
            pipe_reduce_max,
            pipe_blur_scalar,
            field_pipes,
            params_buf,
            pass_buf,
            tables_buf,
            levels_buf,
            palette_buf,
            maxbuf,
            cells_buf,
            fields_buf,
            scratch_buf,
            counts_buf,
            tmp_buf,
            work,
            hist_buf,
            adapt_buf,
            canvas,
            bg0,
            bgl_tween,
            pipe_tween,
            tween_buf,
            tween_prev,
            morphing: false,
            dissolving: false,
            frame_morph: None,
            canvas_fresh: true,
            last_e: 1.0,
            last_field: None,
            grid: None,
            stride: 0,
            work_elem: 4,
            hist_cap: 0,
            hist_next: 0,
            hist_len: 0,
            adapt_valid: false,
            uploaded_m: None,
            uploaded_scheme: None,
            modulus_data: None,
            pass_params: Vec::new(),
            last_cur: 0,
            last_n_measures: 0,
            timing: None,
        }
    }

    /// Turn on GPU frame timing (no-op when the device lacks timestamp queries).
    pub fn enable_timing(&mut self) {
        if !self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return;
        }
        let query_set = self.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("frame timing"), ty: wgpu::QueryType::Timestamp, count: 2 });
        let resolve_buf = buffer(&self.device, "timing resolve", 16, wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
        let staging = buffer(&self.device, "timing staging", 16, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        self.timing = Some(Timing { query_set, resolve_buf, staging, period_ns: self.queue.get_timestamp_period(), pending: None, last_ms: None });
    }

    /// GPU time of the most recently measured frame, in milliseconds.
    pub fn gpu_ms(&mut self) -> Option<f32> {
        let t = self.timing.as_mut()?;
        if let Some(rx) = &t.pending {
            let _ = self.device.poll(wgpu::PollType::Poll);
            if let Ok(res) = rx.try_recv() {
                if res.is_ok() {
                    if let Ok(view) = t.staging.slice(..).get_mapped_range() {
                        let v: &[u64] = bytemuck::cast_slice(&view);
                        if v.len() >= 2 && v[1] >= v[0] {
                            t.last_ms = Some((v[1] - v[0]) as f32 * t.period_ns / 1.0e6);
                        }
                    }
                }
                t.staging.unmap();
                t.pending = None;
            }
        }
        t.last_ms
    }

    /// Size the canvas texture before the UI references it (also done by `render`).
    pub fn ensure_canvas(&mut self, w: u32, h: u32) {
        if self.canvas.frame.width() != w.max(1) || self.canvas.frame.height() != h.max(1) {
            self.canvas = Canvas::new(&self.device, w, h);
            self.dissolving = false;
            self.canvas_fresh = true;
        }
    }

    /// The view egui samples: the display texture (true frame, or a tween of it).
    pub fn display_view(&self) -> &wgpu::TextureView {
        &self.canvas.display_sampled
    }

    #[allow(clippy::too_many_arguments)] // one argument per binding of the shared group
    fn make_bg0(
        device: &wgpu::Device,
        bgl0: &wgpu::BindGroupLayout,
        params: &wgpu::Buffer,
        tables: &wgpu::Buffer,
        levels: &wgpu::Buffer,
        palette: &wgpu::Buffer,
        cells: &wgpu::Buffer,
        maxbuf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared"),
            layout: bgl0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tables.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: levels.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: palette.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: maxbuf.as_entire_binding() },
            ],
        })
    }

    fn rebuild_bg0(&mut self) {
        self.bg0 = Self::make_bg0(&self.device, &self.bgl0, &self.params_buf, &self.tables_buf, &self.levels_buf, &self.palette_buf, &self.cells_buf, &self.maxbuf);
    }

    /// Largest grid (texels) the device limits allow: the measure fields share one
    /// buffer binding, and a vec4 working buffer must fit in one buffer.
    pub fn max_texels(&self) -> u64 {
        let lim = self.device.limits();
        let by_binding = lim.max_storage_buffer_binding_size / (MAX_MEASURE_SLOTS as u64 * 4);
        let by_buffer = lim.max_buffer_size / 16;
        by_binding.min(by_buffer).min(64_000_000)
    }

    pub fn canvas_size(&self) -> (u32, u32) {
        (self.canvas.frame.width(), self.canvas.frame.height())
    }

    /// The true-frame texture (exports read this). Changes when the canvas is resized.
    pub fn output_texture(&self) -> &wgpu::Texture {
        &self.canvas.frame
    }

    fn ensure_grid(&mut self, grid: &GridSpec, work_elem: u64) {
        let dims_changed = self.grid.map(|g| (g.w, g.h) != (grid.w, grid.h)).unwrap_or(true);
        self.ensure_canvas(grid.canvas_w, grid.canvas_h);
        if dims_changed || work_elem != self.work_elem {
            let texels = grid.texels();
            let stride = round_up(texels, 64) as u32;
            self.stride = stride;
            let d = &self.device;
            if dims_changed {
                self.cells_buf = buffer(d, "cells", texels * 8, STORAGE_RW);
                self.fields_buf = buffer(d, "fields", MAX_MEASURE_SLOTS as u64 * stride as u64 * 4, STORAGE_RW);
                self.scratch_buf = buffer(d, "scratch", AUX_FIELDS * stride as u64 * 4, STORAGE_RW);
                self.tmp_buf = buffer(d, "tmp", stride as u64 * 4, STORAGE_RW);
            }
            self.work_elem = work_elem;
            let wbytes = stride as u64 * work_elem;
            self.work = [buffer(d, "work0", wbytes, STORAGE_RW), buffer(d, "work1", wbytes, STORAGE_RW), buffer(d, "work2", wbytes, STORAGE_RW), buffer(d, "work3", wbytes, STORAGE_RW)];
            self.adapt_buf = buffer(d, "adapt", wbytes, STORAGE_RW);
            self.tween_prev = buffer(d, "tween prev", wbytes, STORAGE_RW);
            self.morphing = false;
            // History: as many frames as fit in a quarter of the binding limit, at most HIST_CAP.
            let limit = self.device.limits().max_storage_buffer_binding_size / 4;
            self.hist_cap = ((limit / wbytes.max(1)) as u32).clamp(2, HIST_CAP);
            self.hist_buf = buffer(d, "history", self.hist_cap as u64 * wbytes, STORAGE_RW);
            self.hist_next = 0;
            self.hist_len = 0;
            self.adapt_valid = false;
            self.rebuild_bg0();
        }
        self.grid = Some(*grid);
    }

    fn ensure_modulus(&mut self, md: &Modulus, spectral: bool) {
        if self.uploaded_m == Some((md.m, spectral)) {
            return;
        }
        if self.uploaded_m.map(|(m, _)| m) != Some(md.m) {
            let data = ModulusGpuData::new(md);
            let bytes = bytemuck::cast_slice::<u32, u8>(&data.tables);
            if (bytes.len() as u64) > self.tables_buf.size() {
                self.tables_buf = buffer(&self.device, "tables", round_up(bytes.len() as u64, 4096) * 2, STORAGE_RW);
                self.rebuild_bg0();
            }
            self.queue.write_buffer(&self.tables_buf, 0, bytes);
            self.modulus_data = Some(data);
        }
        let levels = residue_levels_packed(md.m, spectral);
        let lb = bytemuck::cast_slice::<u32, u8>(&levels);
        if (lb.len() as u64) > self.levels_buf.size() {
            self.levels_buf = buffer(&self.device, "levels", round_up(lb.len() as u64, 4096) * 2, STORAGE_RW);
            self.rebuild_bg0();
        }
        self.queue.write_buffer(&self.levels_buf, 0, lb);
        self.uploaded_m = Some((md.m, spectral));
    }

    fn ensure_palette(&mut self, pal: &Palette) {
        if self.uploaded_scheme == Some(pal.scheme) {
            return;
        }
        self.queue.write_buffer(&self.palette_buf, 0, bytemuck::cast_slice(&pal.gpu_words()));
        self.uploaded_scheme = Some(pal.scheme);
    }

    /// Forget frame history (temporal average, afterimage adaptation).
    pub fn reset_history(&mut self) {
        self.hist_len = 0;
        self.hist_next = 0;
        self.adapt_valid = false;
    }

    /// Auxiliary field `i` (0..AUX_FIELDS), or the temporary for `i == AUX_FIELDS`.
    fn aux(&self, i: u64) -> Bind<'_> {
        let bytes = self.stride as u64 * 4;
        if i >= AUX_FIELDS {
            return Bind { buf: &self.tmp_buf, offset: 0, size: bytes };
        }
        Bind { buf: &self.scratch_buf, offset: i * bytes, size: bytes }
    }

    /// Hex-blur `field` `n` times, ping-ponging with `tmp`; the result ends in `field`.
    fn blur_scalar_n(&self, enc: &mut wgpu::CommandEncoder, passes: &mut Vec<PassParams>, field: Bind<'_>, tmp: Bind<'_>, n: u32) {
        let mut in_field = true;
        for _ in 0..n {
            let (src, dst) = if in_field { (field, tmp) } else { (tmp, field) };
            self.dispatch_scalar(enc, passes, &self.pipe_blur_scalar, [src, dst, tmp, tmp], PassParams::default());
            in_field = !in_field;
        }
        if !in_field {
            enc.copy_buffer_to_buffer(tmp.buf, tmp.offset, field.buf, field.offset, field.size);
        }
    }

    fn slot_bind(&self, slot: u32) -> Bind<'_> {
        let bytes = self.stride as u64 * 4;
        Bind { buf: &self.fields_buf, offset: slot as u64 * bytes, size: bytes }
    }

    fn push_pass(passes: &mut Vec<PassParams>, pp: PassParams) -> u32 {
        let off = (passes.len() as u64 * PASS_STRIDE) as u32;
        passes.push(pp);
        off
    }

    fn groups(w: u32, h: u32) -> (u32, u32) {
        (w.div_ceil(WG), h.div_ceil(WG))
    }

    fn dispatch_scalar(&self, enc: &mut wgpu::CommandEncoder, passes: &mut Vec<PassParams>, pipe: &wgpu::ComputePipeline, binds: [Bind<'_>; 4], pp: PassParams) {
        let off = Self::push_pass(passes, pp);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_scalar,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: binds[0].resource() },
                wgpu::BindGroupEntry { binding: 1, resource: binds[1].resource() },
                wgpu::BindGroupEntry { binding: 2, resource: binds[2].resource() },
                wgpu::BindGroupEntry { binding: 3, resource: binds[3].resource() },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.pass_buf, offset: 0, size: NonZeroU64::new(std::mem::size_of::<PassParams>() as u64) }),
                },
                wgpu::BindGroupEntry { binding: 5, resource: self.counts_buf.as_entire_binding() },
            ],
        });
        let g = self.grid.unwrap();
        let (gx, gy) = Self::groups(g.w, g.h);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &self.bg0, &[]);
        pass.set_bind_group(1, &bg, &[off]);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    #[allow(clippy::too_many_arguments)] // the field passes' bindings, by position
    fn dispatch_field(&self, enc: &mut wgpu::CommandEncoder, passes: &mut Vec<PassParams>, pipe: &wgpu::ComputePipeline, a: Bind<'_>, b: Bind<'_>, c: Bind<'_>, fields: Bind<'_>, pp: PassParams, canvas: bool) {
        let off = Self::push_pass(passes, pp);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_field,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.resource() },
                wgpu::BindGroupEntry { binding: 1, resource: b.resource() },
                wgpu::BindGroupEntry { binding: 2, resource: c.resource() },
                wgpu::BindGroupEntry { binding: 3, resource: fields.resource() },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.pass_buf, offset: 0, size: NonZeroU64::new(std::mem::size_of::<PassParams>() as u64) }),
                },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.canvas.frame_storage) },
            ],
        });
        let g = self.grid.unwrap();
        let (gx, gy) = if canvas { Self::groups(g.canvas_w, g.canvas_h) } else { Self::groups(g.w, g.h) };
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &self.bg0, &[]);
        pass.set_bind_group(1, &bg, &[off]);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    /// Encode the whole frame. Call `queue.submit` with the encoder afterwards (the
    /// renderer writes its uniform buffers through the queue, which land before the
    /// submission).
    pub fn render(&mut self, enc: &mut wgpu::CommandEncoder, input: &FrameInput<'_>) {
        let plan = input.layers.plan();
        let variant = match input.layers.color_mode {
            FieldColorMode::MathRamp => 0usize,
            FieldColorMode::PaletteBlend => 1usize,
        };
        let work_elem = if variant == 0 { 4 } else { 16 };
        self.ensure_grid(&input.grid, work_elem);
        self.ensure_modulus(input.modulus, input.palette.scheme == Scheme::Spectral);
        self.ensure_palette(input.palette);
        self.pass_params.clear();

        let n_measures = plan.measures.len().min(MAX_MEASURE_SLOTS as usize) as u32;
        let g = input.grid;
        let md = self.modulus_data.as_ref().expect("modulus uploaded");
        let mut params = GpuParams {
            grid_w: g.w,
            grid_h: g.h,
            canvas_w: g.canvas_w,
            canvas_h: g.canvas_h,
            n0: g.n0,
            x0i: g.x0i,
            ss: g.ss,
            nfactors: md.nfactors,
            y0f: g.y0f,
            x0f: g.x0f,
            sx: g.sx,
            sy: g.sy,
            cells_per_px: g.cells_per_px,
            rows_per_px: g.rows_per_px,
            m: input.modulus.m,
            p: input.modulus.p,
            dlog_off: md.dlog_off,
            lnf_off: md.lnf_off,
            prank_off: md.prank_off,
            blend_mode: input.layers.blend.code(),
            color_mode: variant as u32,
            wants_value: plan.wants_value as u32,
            n_measures,
            value_style: matches!(input.layers.value_style, ValueStyle::Mask) as u32,
            field_active: plan.field_active as u32,
            field_stride: self.stride,
            signed_mask: plan.measures.iter().take(n_measures as usize).enumerate().fold(0u32, |acc, (i, d)| acc | ((d.signed as u32) << i)),
            res_shift: crate::palette::offset_shift(input.style.offset),
            style_flags: (input.style.outlines as u32 * STYLE_OUTLINES) | (input.style.glow as u32 * STYLE_GLOW),
            glow_rgb: {
                let [r, g, b] = input.palette.glow_color();
                (r as u32) << 16 | (g as u32) << 8 | b as u32
            },
            factors: md.factors,
            ..Default::default()
        };
        for d in crate::layers::REGISTRY {
            if let Some(derive) = d.derive {
                params.dparams[d.dslot as usize] = derive(input.layers, input.modulus);
            }
        }
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&self.maxbuf, 0, &[0u8; (MAXBUF_LEN * 4) as usize]);
        let need_counts = (input.modulus.m as u64 + 1) * 4;
        if plan.measures.iter().any(|d| matches!(d.run, Some(MeasureRun::Histogram { .. }))) && self.counts_buf.size() < need_counts {
            self.counts_buf = buffer(&self.device, "counts", round_up(need_counts, 4096), STORAGE_RW);
        }
        self.last_n_measures = n_measures;

        // Display-only tween: decide morph (field lerp in the resolve pass) or dissolve
        // (color mix in the tween pass). Neither touches the true frame's inputs.
        let same_field = plan.field_active && self.last_field == Some((input.grid, variant));
        let mut capture: Option<f32> = None;
        let e = match input.tween {
            Tween::Off => {
                self.morphing = false;
                self.dissolving = false;
                1.0
            }
            Tween::Start(e) => {
                if same_field {
                    capture = Some(if self.morphing { self.last_e } else { 1.0 });
                    self.morphing = true;
                    self.dissolving = false;
                } else {
                    self.morphing = false;
                    self.dissolving = !self.canvas_fresh;
                    if self.dissolving {
                        let c = &self.canvas;
                        enc.copy_texture_to_texture(c.display.as_image_copy(), c.snap.as_image_copy(), c.display.size());
                    }
                }
                e
            }
            Tween::Continue(e) => {
                if self.morphing && !same_field {
                    self.morphing = false; // the view or pipeline changed under the morph: cut
                }
                e
            }
        };
        let e = e.clamp(0.0, 1.0);
        self.frame_morph = if self.morphing { Some(e) } else { None };
        let dissolve_e: f32 = if self.dissolving { e } else { 1.0 };
        self.queue.write_buffer(&self.tween_buf, 0, bytemuck::cast_slice(&[dissolve_e, 0.0, 0.0, 0.0]));

        let mut passes = Vec::new();
        if let Some(prev_e) = capture {
            // what was on screen: mix(previous start, previous true field, previous e)
            let fp = &self.field_pipes[variant];
            let fields = Bind { buf: &self.fields_buf, offset: 0, size: self.stride as u64 * 4 };
            self.dispatch_field(enc, &mut passes, &fp.capture, Bind::whole(&self.work[self.last_cur]), Bind::whole(&self.tween_prev), Bind::whole(&self.tween_prev), fields, PassParams { f0: prev_e, ..Default::default() }, false);
        }
        let out = self.encode(enc, input, &plan, variant, n_measures, &mut passes);
        self.encode_tween(enc);
        self.last_e = e;
        self.last_field = if plan.field_active { Some((input.grid, variant)) } else { None };
        self.canvas_fresh = false;
        self.pass_params = passes;
        self.last_cur = out.cur;
        if out.adapt_stepped {
            self.adapt_valid = true;
        }
        if out.hist_recorded {
            self.hist_next = (self.hist_next + 1) % self.hist_cap.max(1);
            self.hist_len = (self.hist_len + 1).min(self.hist_cap);
        }
    }

    /// Encode every pass of the frame. Only reads `self`; frame-state updates are
    /// returned so the borrow of the buffers can stay alive throughout.
    fn encode(&self, enc: &mut wgpu::CommandEncoder, input: &FrameInput<'_>, plan: &crate::layers::Plan, variant: usize, n_measures: u32, passes: &mut Vec<PassParams>) -> EncodeOut {
        let g = input.grid;
        let mut out = EncodeOut { cur: 0, adapt_stepped: false, hist_recorded: false };

        // 1. values (the frame's first timestamp is written here)
        {
            let (gx, gy) = Self::groups(g.w, g.h);
            let tw = self.timing.as_ref().filter(|t| t.pending.is_none()).map(|t| wgpu::ComputePassTimestampWrites { query_set: &t.query_set, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: None });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("value"), timestamp_writes: tw });
            pass.set_pipeline(&self.pipe_value);
            pass.set_bind_group(0, &self.bg0, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        if !plan.field_active {
            self.resolve(enc, passes, variant, Bind::whole(&self.work[0]));
            return out;
        }

        // 2. measures → fields[slot], each followed by reduce_max. The recipe, not the
        //    measure's id, decides the passes.
        for (slot, def) in plan.measures.iter().take(n_measures as usize).enumerate() {
            let slot = slot as u32;
            let run = def.run.expect("measure has a run recipe");
            let pipe = |e: &str| self.pipe_measures.get(e).expect("measure pipeline");
            let fb = self.slot_bind(slot);
            let (a0, a1, a2, a3) = (self.aux(0), self.aux(1), self.aux(2), self.aux(3));
            let none = PassParams::default();
            match run {
                MeasureRun::Pointwise { entry } => {
                    self.dispatch_scalar(enc, passes, pipe(entry), [fb, a3, a3, a3], none);
                }
                MeasureRun::Blurred { entry, passes: key } => {
                    self.dispatch_scalar(enc, passes, pipe(entry), [fb, a3, a3, a3], none);
                    self.blur_scalar_n(enc, passes, fb, a3, input.layers.param(key).max(0) as u32);
                }
                MeasureRun::MultiBlur { init, fields, passes: key, combine } => {
                    self.dispatch_scalar(enc, passes, pipe(init), [a0, a1, a2, fb], none);
                    let n = input.layers.param(key).max(0) as u32;
                    for (i, f) in [a0, a1, a2].into_iter().enumerate().take(fields as usize) {
                        let _ = i;
                        self.blur_scalar_n(enc, passes, f, a3, n);
                    }
                    self.dispatch_scalar(enc, passes, pipe(combine), [a0, a1, a2, fb], none);
                }
                MeasureRun::Relax { init, step, finish, cap } => {
                    let iters = cap.min(g.w.max(g.h));
                    self.dispatch_scalar(enc, passes, pipe(init), [fb, a3, a3, a3], PassParams { slot: ZERO_COUNT_SLOT, count: iters, ..none });
                    let mut in_field = true;
                    for _ in 0..iters {
                        let (src, dst) = if in_field { (fb, a3) } else { (a3, fb) };
                        self.dispatch_scalar(enc, passes, pipe(step), [src, dst, a3, a3], none);
                        in_field = !in_field;
                    }
                    if !in_field {
                        enc.copy_buffer_to_buffer(a3.buf, a3.offset, fb.buf, fb.offset, fb.size);
                    }
                    self.dispatch_scalar(enc, passes, pipe(finish), [fb, a3, a3, a3], PassParams { slot: ZERO_COUNT_SLOT, count: iters, ..none });
                }
                MeasureRun::Histogram { count, lookup } => {
                    enc.clear_buffer(&self.counts_buf, 0, None);
                    self.dispatch_scalar(enc, passes, pipe(count), [fb, a3, a3, a3], PassParams { slot: CELL_COUNT_SLOT, ..none });
                    self.dispatch_scalar(enc, passes, pipe(lookup), [fb, a3, a3, a3], PassParams { slot: CELL_COUNT_SLOT, ..none });
                }
            }
            // signed fields are normalized by their largest magnitude
            self.dispatch_scalar(enc, passes, &self.pipe_reduce_max, [fb, a3, a3, a3], PassParams { slot, flag: def.signed as u32, ..none });
        }

        // 3. blend → work[0]
        let fields_all = Bind { buf: &self.fields_buf, offset: 0, size: self.stride as u64 * 4 * n_measures.max(1) as u64 };
        let w = |i: usize| -> Bind<'_> { Bind::whole(&self.work[i]) };
        let fp = &self.field_pipes[variant];
        self.dispatch_field(enc, passes, &fp.blend, w(1), w(0), w(2), fields_all, PassParams::default(), false);
        let mut cur = 0usize;
        let mut free: Vec<usize> = vec![1, 2, 3];

        // Temporal history records the under-field of every new frame.
        let temporal_on = plan.transforms.iter().any(|d| d.id == LayerId::Average) && input.layers.avg_mode == AvgMode::Temporal;
        let wbytes = self.stride as u64 * self.work_elem;
        let (hist_start, hist_len) = if temporal_on {
            let window = input.layers.param("avgAmount").clamp(1, HIST_CAP as i32) as u32;
            let len = self.hist_len.min(window.saturating_sub(1)).min(self.hist_cap);
            let start = (self.hist_next + self.hist_cap - len) % self.hist_cap.max(1);
            (start, len)
        } else {
            (0, 0)
        };

        // 4. transforms, fixed order from the registry.
        for def in &plan.transforms {
            match def.id {
                LayerId::Average => match input.layers.avg_mode {
                    AvgMode::Spatial => {
                        let blur_passes = input.layers.param("avgAmount").clamp(1, 12);
                        for _ in 0..blur_passes {
                            let dst = free.pop().unwrap();
                            self.dispatch_field(enc, passes, &fp.blur, w(cur), w(dst), w(dst), fields_all, PassParams::default(), false);
                            free.insert(0, cur);
                            cur = dst;
                        }
                    }
                    AvgMode::Temporal => {
                        let dst = free.pop().unwrap();
                        let hist = Bind::whole(&self.hist_buf);
                        let pp = PassParams { hist_start, hist_len, hist_cap: self.hist_cap, stride: self.stride, ..Default::default() };
                        self.dispatch_field(enc, passes, &fp.temporal, w(cur), w(dst), hist, fields_all, pp, false);
                        free.insert(0, cur);
                        cur = dst;
                    }
                },
                LayerId::BandPass => {
                    let r_lo = input.layers.param("bandLo").clamp(0, 16);
                    let r_hi = input.layers.param("bandHi").clamp(r_lo + 1, 24);
                    // Multi-pass blur of `src` using buffers from `pool`; returns where it landed.
                    let blur_into = |enc: &mut wgpu::CommandEncoder, passes: &mut Vec<PassParams>, src: usize, n: i32, pool: &mut Vec<usize>| -> usize {
                        if n == 0 {
                            return src;
                        }
                        let a = pool.pop().unwrap();
                        let b = pool.pop().unwrap();
                        let (mut s, mut d) = (src, a);
                        for i in 0..n {
                            self.dispatch_field(enc, passes, &fp.blur, w(s), w(d), w(d), fields_all, PassParams::default(), false);
                            if i + 1 < n {
                                if i == 0 {
                                    s = a;
                                    d = b;
                                } else {
                                    std::mem::swap(&mut s, &mut d);
                                }
                            }
                        }
                        let other = if d == a { b } else { a };
                        pool.push(other);
                        d
                    };
                    let mut pool = free.clone();
                    let lo = blur_into(enc, passes, cur, r_lo, &mut pool);
                    let mut pool2: Vec<usize> = (0..4).filter(|&i| i != cur && i != lo).collect();
                    let hi = blur_into(enc, passes, cur, r_hi, &mut pool2);
                    let dst = (0..4).find(|&i| i != cur && i != lo && i != hi).unwrap();
                    self.dispatch_field(enc, passes, &fp.band_diff, w(lo), w(dst), w(hi), fields_all, PassParams::default(), false);
                    self.dispatch_field(enc, passes, &fp.reduce_max_abs, w(dst), w(dst), w(dst), fields_all, PassParams { slot: BAND_MAX_SLOT, ..Default::default() }, false);
                    self.dispatch_field(enc, passes, &fp.normalize, w(dst), w(dst), w(dst), fields_all, PassParams { slot: BAND_MAX_SLOT, ..Default::default() }, false);
                    cur = dst;
                    free = (0..4).filter(|&i| i != cur).collect();
                }
                LayerId::Afterimage => {
                    let strength = input.layers.param("aiStrength").clamp(0, 100) as f32 / 100.0;
                    let atk = 1.0 / input.layers.param("aiAttack").clamp(1, 64) as f32;
                    let dec = 1.0 / input.layers.param("aiDecay").clamp(1, 64) as f32;
                    let dst = free.pop().unwrap();
                    let flag = (input.advance as u32) | ((self.adapt_valid as u32) << 1);
                    let adapt = Bind::whole(&self.adapt_buf);
                    self.dispatch_field(enc, passes, &fp.afterimage, w(cur), w(dst), adapt, fields_all, PassParams { flag, f0: strength, f1: atk, f2: dec, ..Default::default() }, false);
                    out.adapt_stepped = input.advance;
                    free.insert(0, cur);
                    cur = dst;
                }
                _ => {}
            }
        }

        if temporal_on && input.advance {
            let dst_off = self.hist_next as u64 * wbytes;
            enc.copy_buffer_to_buffer(&self.work[0], 0, &self.hist_buf, dst_off, wbytes);
            out.hist_recorded = true;
        }

        // 5. resolve
        self.resolve(enc, passes, variant, Bind::whole(&self.work[cur]));
        out.cur = cur;
        out
    }

    fn encode_tween(&self, enc: &mut wgpu::CommandEncoder) {
        let c = &self.canvas;
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tween"),
            layout: &self.bgl_tween,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&c.frame_load) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&c.snap_load) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&c.display_storage) },
                wgpu::BindGroupEntry { binding: 3, resource: self.tween_buf.as_entire_binding() },
            ],
        });
        let (gx, gy) = Self::groups(c.frame.width(), c.frame.height());
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("tween"), timestamp_writes: None });
        pass.set_pipeline(&self.pipe_tween);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    fn resolve(&self, enc: &mut wgpu::CommandEncoder, passes: &mut Vec<PassParams>, variant: usize, cur: Bind<'_>) {
        let resolve = &self.field_pipes[variant].resolve;
        let fields = Bind { buf: &self.fields_buf, offset: 0, size: self.stride as u64 * 4 };
        // morph: the resolve pass paints mix(start field, true field, e); the start field is in tween_prev
        let (prev, pp) = match self.frame_morph {
            Some(e) => (Bind::whole(&self.tween_prev), PassParams { flag: 1, f0: e, ..Default::default() }),
            None => (cur, PassParams::default()),
        };
        self.dispatch_field(enc, passes, resolve, cur, cur, prev, fields, pp, true);
        if let Some(t) = self.timing.as_ref().filter(|t| t.pending.is_none()) {
            // Close the timing span with an empty pass, then resolve the two stamps.
            {
                let _pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("timing end"),
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites { query_set: &t.query_set, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(1) }),
                });
            }
            enc.resolve_query_set(&t.query_set, 0..2, &t.resolve_buf, 0);
            enc.copy_buffer_to_buffer(&t.resolve_buf, 0, &t.staging, 0, 16);
        }
    }

    /// After submitting a frame: start reading the timestamps back (non-blocking).
    pub fn begin_timing_readback(&mut self) {
        if let Some(t) = self.timing.as_mut() {
            if t.pending.is_none() {
                let (tx, rx) = std::sync::mpsc::channel();
                t.staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                t.pending = Some(rx);
            }
        }
    }

    /// Upload the per-pass parameters collected during `render`. Call before submitting.
    pub fn flush_pass_params(&mut self) {
        let mut bytes = vec![0u8; self.pass_params.len() * PASS_STRIDE as usize];
        for (i, pp) in self.pass_params.iter().enumerate() {
            let o = i * PASS_STRIDE as usize;
            bytes[o..o + std::mem::size_of::<PassParams>()].copy_from_slice(bytemuck::bytes_of(pp));
        }
        if !bytes.is_empty() {
            self.queue.write_buffer(&self.pass_buf, 0, &bytes);
        }
    }

    /// Render one frame synchronously (tests and export): encode, submit, wait.
    pub fn render_blocking(&mut self, input: &FrameInput<'_>) {
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        self.render(&mut enc, input);
        self.flush_pass_params();
        self.queue.submit(Some(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    // ---- readback ----------------------------------------------------------------

    fn read_buffer(&self, buf: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
        let staging = buffer(&self.device, "staging", size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_buffer_to_buffer(buf, offset, &staging, 0, size);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map callback").expect("map ok");
        let data = slice.get_mapped_range().expect("mapped range").to_vec();
        staging.unmap();
        data
    }

    /// The cell grid: `(residue, packed)` per texel, row-major `grid.w × grid.h`.
    pub fn read_cells(&self) -> Vec<[u32; 2]> {
        let g = self.grid.expect("rendered");
        let bytes = self.read_buffer(&self.cells_buf, 0, g.texels() * 8);
        bytemuck::cast_slice::<u8, [u32; 2]>(&bytes).to_vec()
    }

    /// Measure field `slot` (raw, before normalization).
    pub fn read_field(&self, slot: u32) -> Vec<f32> {
        let g = self.grid.expect("rendered");
        let b = self.slot_bind(slot);
        let bytes = self.read_buffer(&self.fields_buf, b.offset, g.texels() * 4);
        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
    }

    pub fn read_max(&self) -> Vec<f32> {
        let bytes = self.read_buffer(&self.maxbuf, 0, MAXBUF_LEN * 4);
        bytemuck::cast_slice::<u8, u32>(&bytes).iter().map(|&u| f32::from_bits(u)).collect()
    }

    /// The final working field as f32 elements (`vec4` fields come back flattened).
    pub fn read_work(&self) -> Vec<f32> {
        let g = self.grid.expect("rendered");
        let bytes = self.read_buffer(&self.work[self.last_cur], 0, g.texels() * self.work_elem);
        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
    }

    /// The last rendered frame as tightly packed RGBA8. It is the true frame whenever
    /// that render used `Tween::Off` (exports always do); a morph paints into it.
    pub fn read_output_rgba(&self) -> (u32, u32, Vec<u8>) {
        self.read_texture_rgba(&self.canvas.frame)
    }

    /// What the window shows (the frame, or a tween into it).
    pub fn read_display_rgba(&self) -> (u32, u32, Vec<u8>) {
        self.read_texture_rgba(&self.canvas.display)
    }

    fn read_texture_rgba(&self, tex: &wgpu::Texture) -> (u32, u32, Vec<u8>) {
        let (w, h) = (tex.width(), tex.height());
        let bpr = round_up(w as u64 * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64);
        let staging = buffer(&self.device, "staging tex", bpr * h as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback tex") });
        enc.copy_texture_to_buffer(
            tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &staging, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr as u32), rows_per_image: None } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map callback").expect("map ok");
        let data = slice.get_mapped_range().expect("mapped range");
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h as usize {
            let o = y * bpr as usize;
            out.extend_from_slice(&data[o..o + w as usize * 4]);
        }
        drop(data);
        staging.unmap();
        (w, h, out)
    }
}

/// Create a buffer initialized with data (used by tests).
pub fn init_buffer(device: &wgpu::Device, label: &str, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage })
}
