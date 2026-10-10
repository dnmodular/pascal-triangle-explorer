# Pascal triangle explorer

Pascal's triangle modulo *m*, rendered on the GPU. Every pixel computes its own
`C(n,k) mod m` in a shader from a closed form — Lucas's theorem for primes,
Granville's theorem for prime powers, the Chinese remainder theorem for
composites — so there is no row-by-row dependency, no triangle buffer, and the
view can pan and zoom to rows in the hundreds of millions at interactive rates.

![mod 8 in the app](docs/screenshot.png)

The look, palettes, controls and keys follow `reference/pascal-core.html`; the
layer stack (measures blended into a field, transforms on top, two field color
modes) follows the `MAP_MODES` registry of `reference/pascal.html`.

## Build and run

Requirements: a Rust toolchain (stable, via [rustup](https://rustup.rs)) and a
GPU with Vulkan, Metal, DirectX 12 or OpenGL. No other SDKs are needed.

```sh
cargo run --release
```

| Platform | Notes |
| --- | --- |
| **Windows** | Install Rust from rustup.rs (MSVC toolchain). `cargo run --release` uses DirectX 12 or Vulkan. |
| **macOS** | `xcode-select --install` for the linker, then `cargo run --release` (Metal). |
| **Linux** | Needs `libxkbcommon` and a Vulkan or GL driver at runtime, e.g. on Debian/Ubuntu `sudo apt install libxkbcommon-x11-0 mesa-vulkan-drivers`. Wayland and X11 both work. |

Headless rendering (no window), handy for scripts and for machines without a
display:

```sh
cargo run --release -- --export pascal.png --m 30 --rows 256 --palette turbo --size 1920x1080
cargo run --release -- --help
```

`WGPU_BACKEND=vulkan|gl|metal|dx12` forces a backend.

### Troubleshooting

* **Black window, and the driver printed `NVVM compilation failed` (NVIDIA) or a
  similar shader compiler message.** The driver rejected one of the compute
  shaders and lost the device. Run

  ```sh
  cargo run --release -- --check-shaders
  ```

  which compiles every pipeline one at a time and names the ones the driver
  rejects, then retries each failure without naga's runtime-check wrappers
  (index clamps and forced loop bounding, which the shaders do not need because
  they bound every index themselves). If that mode compiles, run the app with
  `PASCAL_UNCHECKED_SHADERS=1`. Please report the output together with the
  driver version. Another backend uses a different shader compiler path:
  `WGPU_BACKEND=gl` on Linux and Windows, or `WGPU_BACKEND=dx12` on Windows.
  Debug info is left out of the SPIR-V by default; `WGPU_DEBUG=1` puts it back
  for graphics debuggers.
* **No adapter found.** Install a Vulkan or GL driver; on Linux without a GPU,
  `mesa-vulkan-drivers` provides a software device that runs the app slowly.
* `RUST_LOG=info` prints the adapter and backend in use.

### Tests

```sh
cargo test
```

`tests/gpu_verify.rs` renders regions of the triangle with the real pipeline,
reads them back and compares every cell with the plain additive rule
(`src/math.rs: triangle_dp`) for primes, prime powers, composites and the edge
moduli 2, 3, 4, 6, 12, 30, 65535, 65536, 65537, 99991, 524288, 999983, 999999,
1000000. It also checks rows near 10⁶, 1.2·10⁸ and 10⁹ against the CPU closed
form, and the measures, blend modes, transforms and resolved colors against CPU
twins. Everything must match exactly (fields to within float rounding).

The tests need a GPU adapter. On a machine without one, Mesa's software Vulkan
driver works: `sudo apt install mesa-vulkan-drivers` (lavapipe) and run
`cargo test`; set `WGPU_BACKEND=vulkan` if another backend is picked up.

## Controls

| Input | Action |
| --- | --- |
| drag | pan |
| wheel / pinch | zoom, centered on the cursor |
| double-click, **F** | fit the chosen number of rows |
| **← / →** | modulus −1 / +1 |
| **PgDn / PgUp** | modulus −10 / +10 |
| **↑ / ↓** | rows to fit +1 / −1, then fit |
| **Shift** | ×10 for any of the above |
| hover | inspector: *n*, *k*, `C(n,k)`, `C mod m`, ν_p for each prime factor, the Lucas digit product |

The panel on the left has the modulus (text, ◀ ▶, log slider), the rows to fit,
supersampling, the palette, the three color presets (residue → palette,
zero/nonzero mask, p-adic valuation heatmap), the layer stack and PNG export.
When cells are at least 18 px wide their residue is printed inside them.

## The math

For each pixel the shader finds its cell `(n, k)` and evaluates:

* **Prime p (Lucas).** Write `n` and `k` in base `p`. Then
  `C(n,k) ≡ ∏ C(n_i, k_i) (mod p)`, and the product is 0 as soon as some
  `k_i > n_i`. The digit binomials come from tables of `i!` and `(i!)⁻¹ mod p`
  built on the CPU and uploaded as a storage buffer.
* **Prime power q = pᵉ (Granville).** Let `(x!)_p` be the product of the
  integers `≤ x` not divisible by `p`, and `N_j = ⌊n/pʲ⌋ mod pᵉ` (likewise `K_j`,
  `R_j` with `r = n − k`). Then

  ```
  C(n,k) / p^ν ≡ (±1)^{c_{≥e−1}} · ∏_j (N_j!)_p / ((K_j!)_p (R_j!)_p)   (mod pᵉ)
  ```

  where `ν` is the number of carries when adding `k` and `n − k` in base `p`
  (Kummer's theorem: `ν = ν_p(C(n,k))`), `c_{≥e−1}` counts the carries at digit
  positions `≥ e − 1`, and `±1` is `−1` except for `p = 2, e ≥ 3`. The tables of
  `(x!)_p` and their inverses for `x < pᵉ` are built on the CPU. For `e = 1` this
  is exactly Lucas, so one shader function serves both. The loop walks the digits
  of `n` and `k` only; the digits of `n − k` and the carries fall out of the
  subtraction with borrow. Once `ν` reaches `e` the residue is 0 and the table
  lookups stop (the carry count continues, because the KUMMER layer wants it).
* **Composite m.** The CPU factors `m`; the shader computes each prime-power
  residue and recombines them with Garner's form of the CRT, where every
  intermediate stays below `m`.
* **32-bit mulmod.** WGSL has no 64-bit integers. Odd moduli use Montgomery
  multiplication with `R = 2³²` (the tables are uploaded in Montgomery form; the
  64-bit product is assembled from 16-bit halves); `2ᵉ` moduli use wrapping
  multiplication and a mask. The CPU side (`src/math.rs`) mirrors the same
  loops so the inspector, the in-cell numbers and the tests agree with the
  shader digit for digit.

Rows are `u32`, so the view reaches a billion rows. The camera is kept in f64
and handed to the GPU split into integer and fractional parts; the shader does
the cell lookup in integer arithmetic, so deep rows stay exact in 32-bit.

Colors reproduce the reference tables bit for bit: residue 0 is void
(`#0a0a0a`), residue 1 is `(236, 236, 226)` in the spectral palette, residues
`2..m−1` are quantized to 254 levels and spread over HSL hue 0–300° at
saturation 0.95 and lightness 0.55; the other palettes interpolate the
reference's stop lists. The residue → level table and the 256-entry color tables
are built on the CPU with JavaScript's rounding and uploaded, so a screenshot
here matches one from the HTML version.

## Architecture

```
src/math.rs      factorization, Montgomery constants, Granville tables, CPU closed form, additive DP
src/palette.rs   the reference color tables
src/view.rs      camera (f64) and the grid: texel ↔ (n, k), pixel → texel
src/layers.rs    the layer registry: kind = value | measure | transform, params, run recipe
src/gpu.rs       wgpu pipeline: value → measures → blend → transforms → resolve
src/export.rs    PNG writer
src/main.rs      winit + egui app
shaders/common.wgsl    mulmod, binomial, grid mapping, lattice neighbours (prepended to every pass)
shaders/value.wgsl     residue + carries + divisibility mask per texel
shaders/measures.wgsl  one entry point per measure, reduce_max, hex blur
shaders/field.wgsl     blend, blur, temporal average, band pass, afterimage, resolve (compiled for f32 and vec4 fields)
```

**The grid.** The whole pipeline runs on a grid of texels that covers the
visible region. Zoomed in, a texel is exactly one cell and the six grid
neighbours are the six lattice neighbours (the half-cell brick offset between
rows is tracked exactly). Zoomed out, a texel is a fraction of a pixel and
samples one cell; the resolve pass averages 2×2 (or 3×3) texels per pixel.

**Layers.** Every visualization is one entry in `REGISTRY` with a `kind`:

* `Value` — the raw residue, painted through the palette (or as a mask).
* `Measure` — a pass that writes a float field per texel (`measures.wgsl`),
  optionally followed by hex-blur passes (DENSITY). Any number can be active;
  `blend_main` normalizes each by its maximum and folds them with the blend
  mode — average, multiply (intersection), max, or screen (union) — plus the
  residue ratio when VALUE is checked.
* `Transform` — reprocesses the blended field, in registry order: AVERAGE
  (spatial hex blur, or temporal mean over a ring of recent frames kept on the
  GPU), BAND PASS (blur(keep) − blur(cut), renormalized around 0.5),
  AFTERIMAGE (complement ghost from an adaptation buffer with attack/decay).

The pipeline is derived from the kinds (`LayerState::plan`), never from id
lists. Two field color modes share the same passes: MATH RAMP runs them on an
`f32` field and paints through the ramp; PALETTE BLEND runs them on the `vec4`
palette image (measures as lightness, transforms per channel). `field.wgsl` is
compiled once per element type.

**Adding a measure** (GRADIENT, DIST→0, MAGNITUDE, VARIANCE, RARITY, MOIRÉ,
PRIMES, STRIPES, LAG DIFF, TEMPORAL, COLOR FLOW, MAGNIFY, TRAIL, CENSUS from the
reference): add a `LayerDef` with `kind: Measure`, its parameters and a `run`
recipe, and one `@compute` entry point in `measures.wgsl` that writes `buf_a`.
Parameters reach the shader through `iparam(slot)` (slots in `gpu::param_slot`);
anything loop-derived (powers of `p`, table sizes) is computed on the CPU by the
entry's `derive` function and read from `P.dparams[dslot]`, so the measure
shaders stay free of data-dependent loops, which NVIDIA's compiler rejected for
LUCAS and P DIGIT. Measures that need extra passes (several blurred fields,
frame history) extend the `MeasureRun` recipe rather than the driver's control
flow.

**Export** reads back the same `rgba8` texture the live view shows, so the two
cannot differ.

## Status

Verified (by `cargo test` on Mesa lavapipe, and by the headless export):

* `C(n,k) mod m` on the GPU equals the additive rule for every modulus listed
  above, and equals the CPU closed form at rows 10⁶, 1.2·10⁸ and 10⁹.
* The packed carry count and per-factor divisibility mask, the seven measures,
  all four blend modes, the three transforms (spatial and temporal AVERAGE,
  BAND PASS, AFTERIMAGE over two frames) and the resolved residue/ramp colors
  match CPU twins.
* The windowed app was driven under Xvfb (keys, wheel zoom, drag, in-cell
  numbers, inspector, GPU timing); the headless export writes valid PNGs.

Not verified here: behavior on a real GPU and display (only the software Vulkan
device was available), light-theme appearance, Windows and macOS builds, and
PALETTE BLEND beyond a smoke test (its colors are not compared against the
reference's JavaScript output).

Deviations from the reference worth knowing: PgUp/PgDn step the modulus by 10
(the reference's "whole step" belongs to its decimal-modulus mode, which is out
of scope for v1); in-cell numbers are drawn by the UI over the live view only,
not into exports; frame history for temporal AVERAGE and AFTERIMAGE lives in
screen space, so panning smears it rather than following cells.
