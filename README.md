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
cargo run --release -- --svg pascal.svg --m 5 --rows 80 --palette ocean --outlines
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

`tests/tetra_verify.rs` checks the tetra stack: every cube's residue against the
additive rule (moduli 2–64, the selection modes, a strided 2–300 and slices up to
4096), that each strided cell lands in exactly one of the two cube lists, and the
ID-buffer pick against a CPU ray cast through the same cubes.

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
| **space** | play / pause the sweep |
| **← / →** | one step back / forward through the sweep's selection mode |
| **PgDn / PgUp** | ten steps back / forward |
| **↑ / ↓** | rows to fit +1 / −1, then fit (with rows following m: the next / previous modulus) |
| **Shift** | ×10 for any of the above |
| hover | inspector: *n*, *k*, `C(n,k)`, `C mod m`, ν_p for each prime factor, the Lucas digit product |
| click a cell | pin its row for the wave overlay (click the row again, or **Esc**, to release) |
| **W** | show / hide the wave overlay (it fades) |
| **F11** | fullscreen: the canvas alone, panel, bars and buttons hidden; **Esc** returns |
| tetra tile | open the 3D tetra stack (drag orbit, right/shift-drag pan, wheel zoom, double-click reset, **G** spacing, **Esc** back) |

Fitting R rows shows exactly rows 0..R−1: the apex on the top edge and row R−1
ending on the bottom edge (a canvas narrower than the triangle's 2 : √3 shows its full
width, and the space below row R−1 stays background while the view is fitted; zoom or
pan and the whole triangle is there again). The status line is a bar of its own
under the canvas, so the canvas is exactly what is in view, and the readouts, the
in-cell numbers and the wave's last-visible-row fallback (the last whole row, never
past R−1 while fitted) agree with it.

The panel on the left has the modulus (text, ◀ ▶, log slider), the rows to fit and
the rows-follow-m link, supersampling, the palette, the three color presets
(residue → palette, zero/nonzero mask, p-adic valuation heatmap), the color offset,
cell outlines and zero glow, the layer stack, PNG and SVG export, and a Reset
button. When cells are at least 18 px wide their residue is printed inside them.

**Rows follow m** (the reference's LINK) parents the row count to the modulus: rows
= m × factor (factor 1 by default: one full period of a prime modulus, whose zero
blocks start at row m), refitted whenever m changes — typed, stepped, slid or swept.
A view you have zoomed or panned keeps its zoom relative to the fit and the same spot
of the triangle as it grows. The link runs both ways: with it on, a row count typed
into the field (applied on Enter) moves the modulus to rows ÷ factor, snapped to the
sweep's selection mode, and ↑/↓ — on the canvas or in the field — step to the next
modulus that way. "Hold rows while sweeping" keeps the count still while the sweep
plays or frame-steps (|◀ ▶|); typed moduli, ◀ ▶ and the arrow keys still move it.
With the rows following, every frame changes the view, so a tween cross-fades instead
of morphing. `--link` turns the link on from the command line.

## From the square scanner

`reference/ChatGPT-version-pascal-square-scanner` is a small canvas page ("Pascal ·
Square Scanner"). These of its features earned a place here:

* **Color offset** (its "fractional offset", −1..1), with its own ▶ / ❚❚: playing,
  it cycles round and round — one full turn per period (4 s by default), wrapping,
  for as long as the button is on. Residue colors slide along
  the palette, cyclically, keeping their spacing — a step of 1/m moves every residue
  one slot, as the scanner's frac(r/m + offset) does, and the top residue waits one
  slot before it wraps to the bottom. Unlike the scanner, the void of 0 and
  spectral's white 1 stay put: zeros are the structure everything here is about (with
  nothing else on the ramp, spectral m = 2 has nothing to move). A single colored
  residue (m = 2) bounces between the ramp's ends. It applies to the residue coloring
  and PALETTE BLEND, on the GPU, so exports carry it.
* **Cell outlines**: the scanner's faint `rgba(235, 238, 250, .12)` outline, as a
  1 px line centred on every cell boundary (zeros included), drawn by coverage so its
  width does not depend on where the boundary falls between pixels.
* **Zero glow** (its glow on residue-0 cells): light in the palette's 0.8 color
  spreads from every zero cell over its neighbours out to min(10 px, 0.8 cell), as the
  scanner's `shadowBlur`. Zero cells receive it from neighbouring zeros too, so a void
  shows a lit lattice, and the light is screen-blended, so it only ever brightens.
  Zeros just off the canvas still cast it (their residues come from the closed form).
* Outlines and glow are drawn by the resolve pass, so exports include them; they
  fade in as cells grow from 3 to 6 px and vanish when zoomed out. Both start off, so
  the default view stays identical to `pascal-core.html`.
* **Palettes** Ember, Ocean and Moss (its four-stop gradients).
* **SVG export** (its "Save SVG"): rows 0..R−1 (the rows setting, up to 1024) as
  vector cells — one path per color, and with outlines on, one stroked path over all
  cells — for the residue, mask and valuation colorings, which the CPU reproduces.
  From the window the valuation is normalized by the max ν of the cells in view, so
  the file matches the screen pixel for pixel (verified against the GPU on a fitted
  view); `--svg` on its own normalizes over the exported rows. The glow is a raster
  effect and stays out.
* **Reset**: every setting back to the app's defaults (outlines and glow off).

Left out: its BigInt coefficients (the closed form here is exact at any row), the
square-cell geometry and cell-scale slider (cells here keep the √3/2 lattice the
neighbour measures rely on; zoom sets the scale), the 2–31 modulus slider, the page
chrome, and its motion modes (only "pulse" animates there — "down" and "up" are
static ramps — and the sweep already moves the picture). Its README names an atlas,
determinant motion and a Lava renderer from the original site that the
reconstruction does not contain, so there was nothing to port.

## Modular wave

A translucent square in a corner of the canvas (top right by default; the View
section of the panel moves it, sets the scroll, turns the fill off) plays one row
of the triangle as a green wave, the tetra stack's wire color:

```
r_k = C(n, k) mod m        w(θ) = Σ_k r_k · cos((n − 2k)·(θ − s(t))),   θ ∈ [−π, π]
```

The row is the pinned one if any, else the row under the pointer, else the last
visible row; small ticks at the canvas edges mark it. **Select cells** (next to
"smooth") plays a block of cells instead: click a start cell (red mark), then an end
cell (blue mark); the overlay plays the average of every cell in the rectangle
between them, (1/N) Σ r_{n,k} cos((n − 2k)(θ − s)), binned by frequency like a row.
It is exact for any selection Pascal's rule can step through in about 24 million
additions — rows up to 2048 directly, deeper ones from their first row (closed form)
down the selection's light cone. Larger selections are summed in 16 exact bands of
consecutive rows spread over them, and wider ones still by whole rows at a stride
coprime with 2m (so no parity or prime-power class is skipped); the note then says
"sampled". "Clear
selection" or Esc ends it, as unpinning does for a pinned row; a selection takes
precedence over the pin. **Smooth** off makes the wave snap from one row (or
selection) to the next instead of easing. Each coefficient sits at
frequency n − 2k (the terms of (2 cos θ)ⁿ, filtered through mod m), so the maps
are keyed by frequency and rows of different lengths line up. On a change of
row or modulus only the target coefficients are rebuilt; the displayed ones ease
toward them (α = 1 − e^(−dt/0.12 s), from the real frame time), so the wave
morphs instead of snapping.

The scroll s(t) slides the whole wave sideways at a steady speed (0.3 rad/s of θ by
default; negative moves left, 0 holds it), wrapping every 2π, the common period of
the integer frequencies. Its shape never changes. It replaces the earlier phase
drift cos(fθ + φ), which made a whole row, being symmetric, swell, flatten and
invert in place. Each term is turned by e^{−ifs} (f·s reduced mod 2π in f64) before
the frequency folding, so sub-sample shifts stay exact at any row. The plot is
normalized by a smoothed running max of the envelope max_θ |Σ r_k e^{i(n−2k)θ}| ≥
|w|, so the peaks never clip, and a whole row fills the box however far it has
scrolled.

* Rows up to 2048 are built with Pascal's rule mod m in `u64` (any m), stepping
  forward or backward from the cached row. Longer rows use the same closed form
  as the GPU (shared with the app, never a full binomial); rows past 16,384
  entries use a window of k centred on the part on screen, shown under the label.
  Deep rows mod a composite are mostly zeros, and the note says so when every
  r_k in the window is 0.
* Rows longer than 2000 draw only their 256 largest coefficients (an
  approximation of the full sum). The current row's terms are ranked by their
  target and the terms still fading out by their value, each group keeping its
  256 largest, so the choice does not change during a morph; ties break by |f|,
  keeping a symmetric row's ±f pairs together.
* Samples: ~2 per pixel of plot width, 256–1024. cos(fθ_j + φ) is reduced exactly
  to a table index, so frequencies near 10⁹ cost no large trig arguments, and
  frequencies are folded by f mod P before the sum (at most P² terms a frame).
* The per-frame path reuses its buffers: the sample, sum, table and point
  vectors, and the fill and line meshes (handed to egui in `Arc`s that egui
  releases after tessellation). The two or three text lines go through egui's
  galley cache each frame (one short string copy each), so they survive font
  atlas rebuilds.

## Tetra stack

A 3D view of many moduli at once. Each selected modulus m becomes one horizontal
slice, rows 0..m−1 of the triangle mod m. Slices stack downward in order of m, so
slice size grows with m and the solid is a tetrahedron, with mod 2 (two rows) as the
apex. Cell (m, n, k) is a cube at x = k − n/2, z = n·√3/2 (the 2D brick layout laid
flat), shifted by its slice's offset, and y = −height. Zero cells get no cube, so the
voids run through the solid as tunnels.

* **Shape.** A slice of side s = m − 1 sits s·√(2/3) below the apex: the height of a
  regular tetrahedron of edge s. The default, **equilateral**, centres every slice on
  one vertical axis, so the stack is a regular tetrahedron: four equilateral faces,
  all equally tilted. Each slice grows a third of a row on every face, and the faces
  are steep (70.5°), so they step in thirds: from the top that shows as thin bars at
  the front and half-cell steps at the sides. **Equilateral, terraced** is the same
  tetrahedron to within a third of a cell, grown one whole row on one face per slice
  in turn (front, right, left; the apex, left or right corner stays put, a lattice
  move each time). Every slice's cells then sit exactly over the next one's, the faces
  step in whole cells, three slices to a terrace, and the top view shows every cell
  whole. The flush variants anchor something else on the axis:
  * a **corner** (apex, left or right end of the last row): the two faces that meet
    there stand vertical, along a vertical edge. The apex corner is the first
    version's right-angled shape.
  * the **middle of an edge** (back = the last rows, left = k = 0, right = k = n):
    that one face stands vertical.

* **Cell** (stack and top face; **C** cycles): what each cell is drawn as. The
  equilateral stack in true spacing is a cannonball pile (face-centred cubic: each
  slice sits in the hollows of the one above, √(2/3) lower, every cell 1 from its
  twelve neighbours; its cell count is the tetrahedral number). Cubes don't share
  the pile's threefold symmetry about its vertical axis, which is why its faces step
  in thirds that look different at the front (bars) and the sides (half-cell steps).
  * *cube* (default): fits every shape and spacing, and tiles cleanly at a pixel or
    two.
  * *rhombic*: the rhombic dodecahedron, the pile's own Voronoi cell, built from the
    lattice's neighbour directions (twelve rhombi at ½ along each). The cells fill
    space with no gap. Each step of a face shows one whole rhombus, the same on all
    three sides, since from above the other two outward rhombi of a surface cell are
    exactly edge-on. Seen square onto a face (the snaps), every cell is a hexagon of
    three shaded rhombi: each face is a gapless honeycomb triangle. Twice a cube's
    triangles (72 indices).
  * *sphere*: the cannonballs, touching. A camera-facing quad per cell; the fragment
    shader finds the sphere with a ray test and writes its depth, in the colour pass
    and the pick pass. All faces alike, but a quarter of the pile is gaps, and small
    on screen they turn into dots.
  The rhombic and sphere cells fit the equilateral shape in true spacing (elsewhere
  the slices don't sit in each other's hollows; the panel says so). The flat views
  keep their boxes, and the preview tile its wire cubes.
* **Entry and exit.** A live preview tile sits in a bottom corner of the 2D canvas,
  in the wave box's style (the corner the wave doesn't use). It turns slowly at about
  30 frames a second, as a green wireframe. Hover it and the palette colors run down
  it from the apex to the base in about a third of a second (ease-out, so it snaps
  in), and back up when the pointer leaves. Click it, or **Open 3D view** in the panel, for the isolated
  space: the view, its options panel, a HUD and a thumbnail of the 2D view. Click the
  thumbnail or press **Esc** to return with all 2D state intact.
* **Options live in the 3D space.** The main panel only has the way in (Open 3D view)
  and the tile's toggles. Everything else is in the space's own left panel: the
  view, the range and selection mode, shape, spacing, highlight, ghost zeros and the
  cube budget. **O** (or *hide ◀* / *options ▶*) hides and shows it. The 2D canvas keeps
  rendering while you are away if the sweep plays, so the thumbnail stays live.
* **Navigation.** In the stack: drag to orbit, right-, middle- or shift-drag to pan,
  wheel or pinch to zoom. In the faces and the diamond (an orthographic camera,
  straight on): drag to pan, wheel to zoom about the pointer. Double-click reframes.
  **V** cycles the views, **F** the faces, **G** toggles the spacing, **U** swings the diamond's hinge
  open again, **P** plays the peel, **1–5** snap the stack's camera.
* **Views.**
  * *Stack*: the tetrahedron.
  * *Face*: one face of the tetrahedron laid flat, chosen under the view buttons:
    * *front*: the face the slices' last rows make. Slice m's last row is row m − 1
      of the triangle mod m; stacked, they form a triangle again (the tetrahedron's
      face is equilateral), whose row r is C(r, k) mod (r + 1). Rows sit at m − 1 in
      true spacing (skipped moduli leave blank rows), one per modulus when packed,
      with m labeled down the left edge.
    * *left* and *right*: the k = 0 and k = n edges of every slice, front corner on
      the right and the left. As they stand they are all 1s; peeled d layers they
      are C(n, d) mod m for every m and n < m (the right face is the left mirrored,
      C(n, n − d) = C(n, d)).
    * *base*: the base slice, the whole triangle mod M; peeled, the slice d moduli up.
    * *top*: the stack itself, turned and seen straight down through the orthographic
      camera: each cell shows the topmost cube over it, so the left, right and front
      faces show from above and the lower slices peek out around the upper ones (in
      the equilateral shape their cells sit a third of a row off the ones above; the
      shape setting applies here). Peeling the top shows the slices below.
    The peel works in every face: the face's own side takes it a layer deeper (the
    face is rebuilt at that depth, the camera left as it is), the other checked sides
    trim its edges, and the top and the base take whole rows (moduli).
  * *Diamond*: the front face with the whole base slice (the last modulus, every
    row, not just the last one it shares with the face) hinged down along that shared
    row: the tetrahedron's net of two faces, a rhombus. When the view is chosen the
    base starts folded where it sits in the tetrahedron (109.47° out of the face's
    plane, i.e. at the dihedral angle arccos(1/3)) and swings flat in about a second;
    each slice's placement is an origin and a row vector, so the swing only rewrites
    one slice's entry on the GPU.
  * Things to look at: a face row alternates 1, m − 1, 1, … exactly when m is prime
    (C(m − 1, k) ≡ (−1)^k mod m; tested for m ≤ 600), so prime rows read as two-tone
    stripes. Zeros on the face are rare: none until m = 30 = 2·3·5, and (tested up to
    1200) only for moduli with three or more distinct prime factors, though not all
    of them (42 has none); they show as dark streaks. Hovering a face row reports
    both facts for that row. The faces and the diamond are drawn in the palette's true
    colors (faces toward the viewer are not shaded), cell against cell: each cell is a
    box one cell wide and one row (√3/2) high, with no gap and no dark edge, as in
    the 2D view (the stack keeps its separate cubes).
* **Look.** In the 3D space every cube is filled in its palette color, through the
  same residue → level table, color offset and palette as the 2D view, with flat
  shading and darker edges. The tile's wire is green on near-black: its edges are
  drawn in the fragment shader (on a cube face the distance to the nearest edge is the
  middle of the three face distances, measured in pixels with `fwidth`, and
  fragments away from the edges are discarded). The fill is a height in the
  uniforms: cubes whose centre lies at or above it are filled, the rest are wire.
  Faint green triangles outline every slice. *Ghost zeros* draws the zero cells as
  faint additive cubes that glow through the solid, which shows the tunnels.
* **Solid zeros on the faces and the diamond** (on by default): their zero cells are
  drawn as dark cells, so the flat shapes have no gaps (the first zeros are row 29
  mod 30), and they can be hovered. The stack keeps its zeros as voids, its tunnels.
* **Hover.** An ID pass renders the nonzero cubes into a 1 × 1 `Rg32Uint` target
  through a projection that maps the pointer's pixel onto it. It returns the
  frontmost cube's packed (n, k, slice, residue) words, read back asynchronously.
  Cubes within the highlight radius of that cube (or, with *whole slice*, the
  hovered slice) light up, with white edges. The zero / nonzero mask lens colors the
  cubes white; other layer setups fall back to residue coloring, and the HUD says
  so. The HUD shows the shape, m and its factorization, n, k and the residue.
* **Peel** (stack and face views): takes the stack apart layer by layer from its sides: left
  (k = 0), right (k = n), front (the last rows), top (slices from the apex down) and
  base (slices from the bottom up). Each side has a checkbox: the checked sides peel
  together, the same number of layers each, and the unchecked ones stay (check one
  side to peel just it, or all but one to keep that one). **P** checks the left side
  (in a face view, that face's own side) if none is checked. A layer is a row or cell of the stride
  on the sides and a drawn slice at the top and the base. **←** / **→** step it a
  layer back or on (**Shift** ten), **Home** / **End** go to nothing / everything
  peeled, and the buttons do the same (|◀ ◀| |▶ ▶|). ▶ (or **P**) plays it, a layer
  every 1/speed s, starting over once nothing is left; ◀ (or **R**) rewinds, putting
  the layers back one by one and starting again from everything peeled. Stepping
  stops the playing; the slider sets the layers by hand. The first layer off the left and right sides is the white
  skin of 1s, and the layer under it is n mod m. Peeling is done while drawing (the
  vertex shader sends peeled cubes out of the clip volume, the pick pass too), so a
  layer costs nothing to take off or put back.
* **Snap** (stack view, keys **1–5**): the camera glides to look straight down onto the
  apex, straight up at the base, or square onto the left, right or front face. The
  directions are the outward normals of the planes from the top slice to the bottom
  one, from the stack's own corners, so they fit the flush shapes and packed spacing
  (whose squat prime stacks tilt the faces far more). A snap sticks: change the
  spacing, range, visit mode or shape and the camera glides to the same face of the
  new shape; orbiting or panning by hand lets go of it.
* **The view stays put.** Rebuilding the stack (a new range, visit mode, spacing,
  shape or level of detail) keeps the camera's angle and its pan and zoom relative to
  the new size; only switching between stack, face and diamond (or to another face),
  or a double-click, frames it afresh.
* **Selection.** A start / end range (2–4096) and the sweep's selection modes: all,
  primes (with their orderings), composites, even, odd, and the interval sequences.
* **Spacing.** *True* (default): slice height follows m. Skipped moduli leave gaps,
  each drawn as a faint dashed outline, and the gap between two slices is labeled
  ("+4" between 7 and 11), so prime gaps read as spacing. The tetrahedron's edges stay
  straight. *Packed*: slice height follows the slice index, no gaps. The HUD notes that the sides
  now curve, because slice size no longer tracks height.
* **Every cell, by default.** The cube budget defaults to *all*: every cell of the
  range is drawn, up to what one scene holds on the GPU (48M cubes, or less when the
  device's largest buffer is smaller: 2–500 is 20.8M, 2–660 about 48M). Cubes live
  in one buffer, nonzero ones from the front and zeros from the back, 8 bytes each;
  when every cube is filled their back faces are culled; and the preview tile draws
  a light stack of its own (1M cubes), so the full one is only drawn in the 3D space.
* **Level of detail, as cubes** (a range past the GPU's limit, or a smaller budget,
  1M–32M, for speed): the smallest stride s that fits: every s-th row and
  cell of a slice, and, walking down from the apex, only the slices at least one cube
  (s cells) below the last one kept; the base is always drawn too (it takes the last
  kept slice's place when they would touch). Anchored at the apex, the top of the stack
  stays the same when the range's end moves (2–487 and 2–488 used to draw different
  slices all the way up). Every cube is a cube, s cells on a
  side, in all three directions, so a wide range no longer turns into stacked slabs,
  and a sparse selection whose slices are already far apart (every 100th modulus,
  say) keeps all of them. The face and the diamond use the same rule on their rows;
  the diamond's base keeps its last row, the hinge, at any stride (its rows run
  M − 1, M − 1 − s, …). The HUD reports the stride and how many slices and cells are
  shown.

How it runs: a compute pass (`shaders/tetra_cells.wgsl`, with `shaders/binom.wgsl`
prepended) takes one thread per cell (a slice is a triangle of rows, or a single face
row). It finds its slice by a 16-step binary search
and its row from an f32 root corrected to exact integers. It then computes C(n,k)
mod m with that slice's own factor descriptors: Granville / Lucas per prime power,
with tables shared between slices with the same prime power, then Garner's CRT, as
in the 2D passes. Every loop is constant-bounded, and all powers and constants come
from the CPU. The pass appends nonzero cells and zeros to two instance buffers, and
the atomic counts are the instance counts of indirect draws, so cubes are drawn
without a CPU round trip. It runs only when the selection or level of detail
changes; colors are looked up while drawing.

## Sweep

The sweep bar under the canvas plays the modulus through a range, ported from the reference's
MODULUS SWEEP group (`playBounds`, `nextModInDir`, `buildSeqMods`). Every control
sits in a fixed-width slot and the explanations are tooltips, so nothing moves
while the modulus or the mode changes:

* **Transport:** jump to the range start, play reverse, play / pause, play forward,
  step back and forward one frame. Steps and typed or slider values always snap to
  the active selection mode.
* **Selection modes:** all, primes (in order, grouped by p mod 4, or only one class),
  composites, even, odd, and interval sequences starting at the range start: fixed
  step, π digits, Fibonacci, primes, squares, powers of 2, 2×Pn, 3×Pn, primes + 2×Pn,
  semiprimes, primes + semiprimes, sphenic numbers.
* **Range, frame time (ms) and loop:** the frame time has a slider (16 ms–1 s) and a
  field that takes any value typed or dragged, 16 ms–10 s. The bar's number fields
  apply typed values on Enter (or when they lose focus), so a half-typed number never
  drives the sweep; dragging them is live. Looping wraps at the range
  ends; with loop off, playback stops on the last frame.
* **Tween (ms):** a display-only transition. When consecutive frames run the same
  field pipeline on the same view, the field morphs (smoothstep lerp, as in the
  reference); otherwise the picture cross-dissolves. The resolve pass still writes
  the true frame, layer history (temporal AVERAGE, AFTERIMAGE) only sees true
  frames, and PNG export re-renders the true frame first.

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
src/wave_overlay.rs  the modular-wave overlay (row cache, easing, sampling, egui shapes)
src/tetra.rs     the tetra stack's selection, layout and stride, camera, CPU ray cast
src/tetra_gpu.rs the tetra stack on the GPU: residue/compaction pass, cubes, ghosts, pick, outlines
src/main.rs      winit + egui app (src/tetra_ui.rs: the stack's panel, tile and 3D space)
shaders/common.wgsl    bindings, grid mapping, lattice neighbours (prepended to every 2D pass, with binom.wgsl)
shaders/binom.wgsl     Montgomery mulmod, Granville / Lucas, Kummer: no bindings, shared with the tetra stack
shaders/tetra_cells.wgsl  the stack's residues, one thread per (m, n, k), appended to two cube lists
shaders/tetra_draw.wgsl   instanced cubes with fragment-shader edges, the ID pass, dashed lines
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

**Measures.** KUMMER, LUCAS, CRT, DLOG, P DIGIT, GRADIENT, DIST→0, MAGNITUDE,
DENSITY, VARIANCE, RARITY, MOIRÉ, PRIMES, STRIPES and LAG DIFF, ported from the
reference's `mapFieldRaw` with the same controls. How each relates to the
reference's whole-triangle computation:

* **Exact at any zoom** (closed form per cell): KUMMER, LUCAS, CRT, DLOG,
  P DIGIT, GRADIENT and LAG DIFF (neighbour residues via `residue_at(n, k)`),
  MAGNITUDE (`ln i!` table below 4096, a cancellation-safe Stirling form above),
  PRIMES (CPU-built prime-index table), STRIPES (walks the line through each cell,
  up to 256 positions each way; longer runs saturate).
* **Over the cells in view** (the reference uses its whole triangle): DENSITY,
  VARIANCE and MOIRÉ blur over the lattice, DIST→0 relaxes up to 256 steps
  (farther distances saturate), RARITY counts residue frequencies. Zoomed in,
  the grid is the cell lattice; zoomed out, these run on the sampled lattice.

The triangle here has no bottom row, so near the reference's last rows (where
its chains and lags run off the edge) the values legitimately differ.

**Adding a measure** (TEMPORAL, COLOR FLOW, MAGNIFY, TRAIL and CENSUS remain
from the reference): add a `LayerDef` with `kind: Measure`, its parameters and a
`run` recipe, and the compute entry points the recipe names in `measures.wgsl`.
Recipes: `Pointwise`, `Blurred` (then N hex blurs), `MultiBlur` (up to three
auxiliary fields blurred, then combined), `Relax` (iterated relaxation
dispatches) and `Histogram` (atomic residue counts, then a lookup). Anything
loop-derived (powers of `p`, per-direction steps, prime pairs) is computed on
the CPU by the entry's `derive` function and read from `P.dparams[DSLOT_<ID>]`
(the constant is generated from the registry). Every shader loop has a constant
bound with an early break, because NVIDIA's compiler rejected data-dependent
power loops in LUCAS and P DIGIT. A `signed: true` measure is normalized by its
largest magnitude and centered on 0.5, as in the reference's blend.

## Status

Verified (by `cargo test` on Mesa lavapipe, and by the headless export):

* `C(n,k) mod m` on the GPU equals the additive rule for every modulus listed
  above, and equals the CPU closed form at rows 10⁶, 1.2·10⁸ and 10⁹.
* The packed carry count and per-factor divisibility mask, all sixteen measures,
  all four blend modes (with a signed measure), the three transforms (spatial
  and temporal AVERAGE, BAND PASS, AFTERIMAGE over two frames) and the resolved
  residue/ramp colors match CPU twins. DIST→0 equals a CPU breadth-first search
  on the same lattice, and STRIPES equals the reference's chain DP, exactly.
* The sweep's selection modes and sequences match the reference's lists; the
  tween shows the previous frame at e = 0 and exactly the true frame at e = 1.
* The windowed app was driven under Xvfb (keys, wheel zoom, drag, in-cell
  numbers, inspector, GPU timing); the headless export writes valid PNGs.
* Wave overlay: Pascal rows mod m equal exact binomials mod m (n ≤ 30 for fourteen
  moduli up to 2⁶⁴ − 59, and n ≤ 125 for the largest), forward and backward;
  closed-form rows equal Pascal rows; the samples equal the direct sum; easing
  converges at 30 and 144 fps alike; m = 1, n = 0, rows near 2³², moduli above
  the cap, zero-sized and resized viewports and NaN/negative frame times stay
  finite; long-row morphs move at most a quarter of full scale per frame (the
  older selection jumped 0.85), ties keep ±f pairs, the normalized peaks stay
  at 1 through a full scroll cycle (no breathing; within the grid's 0.2 %), and tiny
  canvases draw nothing. Under
  Xvfb: hover, a modulus step and a row change morph, **W** fades out and in,
  pinning, corners (the inspector moves out of the box's way), resizes, deep rows
  with their k window.
* Square-scanner features: the resolve pass's color offset, outlines and zero
  glow equal an independent CPU twin (distances to the view's own cell rectangles)
  within one level on ~1.9M pixels — spectral and ember, m = 2, 3, 7, each feature
  alone and all together, with the apex in view, the triangle crossing the left edge
  and the top edge inside row 30 — and vanish on 2 px cells; the SVG export's colors
  equal the GPU's for the residue, mask and valuation colorings on a fitted m = 3
  view whose rows past the export raise the max ν.
* The app is idle when nothing moves (with the wave's scroll at 0), and a redraw
  that changes no GPU input skips the triangle's passes; after a tween the canvas
  equals a fresh render pixel for pixel. The on-screen canvas equals the exported
  PNG byte for byte (checked under Xvfb with AFTERIMAGE on), and Save PNG leaves
  the screen unchanged.
* Tetra stack: strided cubes stay cubes and never overlap (all views, true and
  packed, all moduli and primes to 2000), the base slice is always drawn, and each
  view's cell enumeration is complete. The face is row m − 1 of every slice, centred,
  rows √3/2 apart; the diamond's base meets it along that row at the dihedral angle
  arccos(1/3) when folded, and lies flat as a mirrored triangle when open. The
  face-row facts above. The flat camera frames the views, keeps x to the right,
  casts straight rays and zooms about the pointer. GPU residues match the additive
  rule for the face and the diamond (stride 1 and strided; a prime face is all ±1),
  and GPU picks match the CPU ray cast on the flat face, the open diamond and the
  half-open diamond at a slant. Every face (front, left, right, base, top) peeled
  into: each cell is written once with its residue on the face's own line (row
  m − 1 − d, k = d, n − k = d, the slice d up), the pick sees through the peel, the
  extent clears each face and does not depend on the depth the face is at, and the
  flat views' cells tile a cell apart, centred.
* Tetra stack cells: the rhombic dodecahedron has twelve planar rhombi wound outward,
  each at ½ along its neighbour direction, every corner within ½ along all of them,
  and the volume 1/√2 of one lattice point (so the cells tile space); a slice down is
  one of the lattice's neighbours. GPU picks with rhombic and sphere cells match a CPU
  ray cast through the same shapes in the orbiting stack, through a peel, and in the
  turned top face.
* Tetra stack: the equilateral stack's edges are all equal (a regular tetrahedron),
  every slice's centroid is on the axis, and each flush shape keeps its corner or
  edge midpoint on the axis. Every cube's residue equals the additive rule for moduli 2–64, the
  prime / composite / Fibonacci / square selections, a strided 2–300 and the deep
  slices 2048, 2310, 3125, 3600, 4093, 4095 and 4096; every strided cell is written
  exactly once. The GPU pick equals the CPU ray cast on every decisive test ray
  across four camera views, for the equilateral (true and packed spacing, and
  strided), apex-corner and left-face shapes (rays inside a face, clear of the next cube), and every
  GPU hit on a CPU miss is a grazing ray. The wire, the highlight and the ghosts change
  the image as they should, a full fill colors the stack, and a half fill colors its
  top half. Under Xvfb: the tile, opening, hover with the HUD and
  brush, orbit, zoom, pan, **G**, a strided 2–400 stack and **Esc** back.

Not verified here: behavior on a real GPU and display (only the software Vulkan
device was available), light-theme appearance, Windows and macOS builds, and
PALETTE BLEND beyond a smoke test (its colors are not compared against the
reference's JavaScript output).

Deviations from the reference worth knowing: PgUp/PgDn take ten sweep steps
(the reference's "whole step" belongs to its decimal-modulus mode, which is out
of scope for v1); the tween cross-dissolves in color where the reference fades
the canvas; in-cell numbers are drawn by the UI over the live view only,
not into exports; frame history for temporal AVERAGE and AFTERIMAGE lives in
screen space, so panning smears it rather than following cells.
