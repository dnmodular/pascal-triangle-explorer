//! Pascal's triangle modulo m, rendered on the GPU: every pixel computes its own
//! binomial residue from Lucas / Granville / CRT, with no row-by-row dependency.
//!
//! * [`math`]    — factorization, Montgomery constants, Granville tables, CPU reference
//! * [`palette`] — the color tables of the reference implementation, bit for bit
//! * [`view`]    — camera and the cell ↔ texel geometry
//! * [`sweep`]   — modulus selection modes, interval sequences and playback rules
//! * [`layers`]  — the layer registry (value / measure / transform) and its plan
//! * [`gpu`]     — the wgpu pipeline that runs the plan
//! * [`export`]  — PNG export of the canvas texture
//! * [`tetra`] / [`tetra_gpu`] — the tetra stack: one 3D slice per modulus
//! * [`wave_overlay`] — the "modular wave" overlay: one row of the triangle as a wave
pub mod export;
pub mod gpu;
pub mod layers;
pub mod math;
pub mod palette;
pub mod sweep;
pub mod tetra;
pub mod tetra_gpu;
pub mod view;
pub mod wave_overlay;
