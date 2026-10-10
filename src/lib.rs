//! Pascal's triangle modulo m, rendered on the GPU: every pixel computes its own
//! binomial residue from Lucas / Granville / CRT, with no row-by-row dependency.
//!
//! * [`math`]    — factorization, Montgomery constants, Granville tables, CPU reference
//! * [`palette`] — the color tables of the reference implementation, bit for bit
//! * [`view`]    — camera and the cell ↔ texel geometry
//! * [`layers`]  — the layer registry (value / measure / transform) and its plan
//! * [`gpu`]     — the wgpu pipeline that runs the plan
//! * [`export`]  — PNG export of the canvas texture
pub mod export;
pub mod gpu;
pub mod layers;
pub mod math;
pub mod palette;
pub mod view;
