//! The layer registry: the single description of every visualization.
//!
//! * `Kind::Value`     — the raw residue layer (painted straight through the palette).
//! * `Kind::Measure`   — produces a 0..1 field per cell. Any number can be active; they
//!   are combined by a `BlendMode` (average, multiply, max, screen).
//! * `Kind::Transform` — reprocesses whatever the measures produced, in the fixed order
//!   AVERAGE → BAND PASS → AFTERIMAGE.
//!
//! The GPU pipeline derives what to run from the kinds and from each measure's `run`
//! recipe, never from hard-coded id lists: a new measure is one entry here plus one
//! compute entry point in `shaders/measures.wgsl`.

use std::collections::HashMap;

use crate::math::Modulus;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LayerId {
    Value,
    Kummer,
    Lucas,
    Crt,
    Dlog,
    PDigit,
    Gradient,
    DistZero,
    Magnitude,
    Density,
    Variance,
    Rarity,
    Moire,
    Primes,
    Stripes,
    LagDiff,
    Average,
    BandPass,
    Afterimage,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Value,
    Measure,
    Transform,
}

/// A numeric control of a layer.
#[derive(Clone, Copy, Debug)]
pub struct ParamDef {
    pub key: &'static str,
    pub label: &'static str,
    pub min: i32,
    pub max: i32,
    pub default: i32,
    /// Non-empty for a choice: the value is an index into these labels.
    pub choices: &'static [&'static str],
}

/// How a measure's field is produced on the GPU.
#[derive(Clone, Copy, Debug)]
pub enum MeasureRun {
    /// One compute pass; `entry` is the entry point in `measures.wgsl`.
    Pointwise { entry: &'static str },
    /// One compute pass, then that many hex-blur passes (count from the named parameter).
    Blurred { entry: &'static str, passes: &'static str },
    /// `init` writes `fields` (1..=3) auxiliary fields into buf_a..buf_c, each is
    /// hex-blurred `passes` times, then `combine` reads them and writes the measure to buf_d.
    MultiBlur { init: &'static str, fields: u32, passes: &'static str, combine: &'static str },
    /// `init`, then min(`cap`, larger grid side) passes of `step` (one relaxation
    /// sweep each, ping-ponged), then `finish` in place. Breadth-first distances.
    Relax { init: &'static str, step: &'static str, finish: &'static str, cap: u32 },
    /// `count` builds an atomic histogram of the residues in view, `lookup` reads it.
    Histogram { count: &'static str, lookup: &'static str },
}

impl MeasureRun {
    /// Every compute entry point the recipe dispatches.
    pub fn entries(&self) -> Vec<&'static str> {
        match *self {
            MeasureRun::Pointwise { entry } | MeasureRun::Blurred { entry, .. } => vec![entry],
            MeasureRun::MultiBlur { init, combine, .. } => vec![init, combine],
            MeasureRun::Relax { init, step, finish, .. } => vec![init, step, finish],
            MeasureRun::Histogram { count, lookup } => vec![count, lookup],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LayerDef {
    pub id: LayerId,
    pub label: &'static str,
    pub kind: Kind,
    pub desc: &'static str,
    pub params: &'static [ParamDef],
    /// Measures only.
    pub run: Option<MeasureRun>,
    /// Values the shader needs that are computed on the CPU each frame (powers of p,
    /// anything loop-derived), uploaded as `Params::dparams[dslot]`. Keeps the GPU side
    /// free of data-dependent loops that some driver compilers reject.
    pub derive: Option<Derive>,
    pub dslot: u32,
    /// The field is signed: normalized by its largest magnitude and centered on 0.5.
    pub signed: bool,
    /// Compares frames: reads flat until something changes.
    pub sweep: bool,
}

/// CPU-side derivation of a layer's shader constants (see `LayerDef::derive`).
pub type Derive = fn(&LayerState, &Modulus) -> [u32; 4];

/// `p^e`, or 0 when it does not fit in a u32 (the shaders treat 0 as "digit out of range").
pub fn pow_or_zero(p: u32, e: u32) -> u32 {
    let mut r: u32 = 1;
    for _ in 0..e {
        match r.checked_mul(p) {
            Some(v) => r = v,
            None => return 0,
        }
    }
    r
}

fn derive_lucas(s: &LayerState, md: &Modulus) -> [u32; 4] {
    let l = s.param("lucasScale").max(0) as u32;
    [pow_or_zero(md.p, l), l, 0, 0]
}

fn derive_padic(s: &LayerState, md: &Modulus) -> [u32; 4] {
    let j = s.param("padicDigit").max(0) as u32;
    [pow_or_zero(md.p, j), j, 0, 0]
}

/// Lattice step (rows, columns) of a scan direction choice: ╲ (n+1, k), ╱ (n+1, k+1), — (n, k+1).
pub fn direction_step(choice: i32) -> (u32, u32) {
    match choice {
        1 => (1, 1),
        2 => (0, 1),
        _ => (1, 0),
    }
}

fn derive_stripes(s: &LayerState, md: &Modulus) -> [u32; 4] {
    let (dr, dc) = direction_step(s.param("stripeDir"));
    let period = s.param("stripePeriod").max(1) as u32;
    let tol = (s.param("stripeTol").max(0) as u32).min(md.m);
    [period * dr, period * dc, tol, 0]
}

fn derive_lagdiff(s: &LayerState, _md: &Modulus) -> [u32; 4] {
    let (dr, dc) = direction_step(s.param("lagDir"));
    let lag = s.param("lagLen").max(1) as u32;
    [lag * dr, lag * dc, 0, 0]
}

/// The two smallest distinct primes of m (q = 0 when m is a prime power).
fn derive_moire(_s: &LayerState, md: &Modulus) -> [u32; 4] {
    let q = md.factors.get(1).map(|t| t.f.p).unwrap_or(0);
    [md.p, q, (q != 0) as u32, 0]
}

fn derive_primes(s: &LayerState, _md: &Modulus) -> [u32; 4] {
    [s.param("primesMode").clamp(0, 1) as u32, 0, 0, 0]
}

const DIRECTIONS: &[&str] = &["diag ╲", "diag ╱", "row —"];

/// Measures first (console order), then the transforms in the order they run.
pub const REGISTRY: &[LayerDef] = &[
    LayerDef {
        id: LayerId::Value,
        label: "VALUE",
        kind: Kind::Value,
        desc: "Colors each cell by its residue directly — the raw sieve, and the base layer every other map replaces or blends with.",
        params: &[],
        run: None,
        derive: None,
        dslot: 0,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::Kummer,
        label: "KUMMER",
        kind: Kind::Measure,
        desc: "Carry count when adding k and n−k in base p (p = smallest prime factor of the modulus). By Kummer's theorem that count is exactly the power of p dividing C(n,k) — the depth of every zero, not just its location.",
        params: &[],
        run: Some(MeasureRun::Pointwise { entry: "measure_kummer" }),
        derive: None,
        dslot: 0,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::Lucas,
        label: "LUCAS",
        kind: Kind::Measure,
        desc: "Colors each cell by its base-p digit pair at the chosen scale. Raise SCALE to step through the self-similar block levels; holes (k digit > n digit) show dark.",
        params: &[ParamDef { key: "lucasScale", label: "scale", min: 0, max: 12, default: 0, choices: &[] }],
        run: Some(MeasureRun::Pointwise { entry: "measure_lucas" }),
        derive: Some(derive_lucas),
        dslot: 0,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::Crt,
        label: "CRT",
        kind: Kind::Measure,
        desc: "Splits a composite modulus into its prime-power layers: each cell is colored by which factors divide it. Two primes (6 = 2·3) give four zones — neither, each, or both (the true zeros).",
        params: &[],
        run: Some(MeasureRun::Pointwise { entry: "measure_crt" }),
        derive: None,
        dslot: 0,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::Dlog,
        label: "DLOG",
        kind: Kind::Measure,
        desc: "Recolors each nonzero cell by its discrete log to a primitive root of p. Lucas makes values products across digit scales; in log space those products become sums. Multiples of p stay dark.",
        params: &[],
        run: Some(MeasureRun::Pointwise { entry: "measure_dlog" }),
        derive: None,
        dslot: 0,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::PDigit,
        label: "P DIGIT",
        kind: Kind::Measure,
        desc: "The chosen base-p digit of each residue. Digit 0 is fully Lucas-explained; digits ≥ 1 are what a prime-power modulus knows beyond mod p. Most meaningful on p, p², p³ moduli (try 9, 27, 25).",
        params: &[ParamDef { key: "padicDigit", label: "digit", min: 0, max: 12, default: 1, choices: &[] }],
        run: Some(MeasureRun::Pointwise { entry: "measure_padic" }),
        derive: Some(derive_padic),
        dslot: 1,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::Gradient,
        label: "GRADIENT",
        kind: Kind::Measure,
        desc: "Wrap-distance from each cell to its two parents — how sharply the residue turns over as the triangle grows. The boundaries of the self-similar blocks light up.",
        params: &[],
        run: Some(MeasureRun::Pointwise { entry: "measure_gradient" }),
        derive: None,
        dslot: 0,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::DistZero,
        label: "DIST→0",
        kind: Kind::Measure,
        desc: "Lattice distance from each cell to the nearest zero, so the gasket holes become basins and the ridges sit farthest from any divisible cell. Measured over the cells in view, up to 256 steps.",
        params: &[],
        run: Some(MeasureRun::Relax { init: "distzero_init", step: "distzero_step", finish: "distzero_finish", cap: 256 }),
        derive: None,
        dslot: 0,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::Magnitude,
        label: "MAGNITUDE",
        kind: Kind::Measure,
        desc: "log of the true integer C(n,k), ignoring the modulus — the raw magnitude the residues hide. Peaks at the middle of each row.",
        params: &[],
        run: Some(MeasureRun::Pointwise { entry: "measure_magnitude" }),
        derive: None,
        dslot: 0,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::Density,
        label: "DENSITY",
        kind: Kind::Measure,
        desc: "Local fraction of nonzero cells — the fractal's mass. A larger radius blurs toward broad density structure.",
        params: &[ParamDef { key: "densityR", label: "radius", min: 1, max: 16, default: 4, choices: &[] }],
        run: Some(MeasureRun::Blurred { entry: "measure_density", passes: "densityR" }),
        derive: None,
        dslot: 0,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::Variance,
        label: "VARIANCE",
        kind: Kind::Measure,
        desc: "Circular variance of nearby residues — the wrap-aware dispersion for cyclic values. Dark on uniform blocks, bright where colors churn: DENSITY shows the mass, this shows the gappiness.",
        params: &[ParamDef { key: "varR", label: "radius", min: 1, max: 16, default: 4, choices: &[] }],
        run: Some(MeasureRun::MultiBlur { init: "variance_init", fields: 2, passes: "varR", combine: "variance_combine" }),
        derive: None,
        dslot: 0,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::Rarity,
        label: "RARITY",
        kind: Kind::Measure,
        desc: "Colors each cell by −log₂ of its residue's frequency among the cells in view. The majority mode fades automatically and improbable residues glow — self-normalizing isolation, no residue list needed.",
        params: &[],
        run: Some(MeasureRun::Histogram { count: "rarity_count", lookup: "rarity_lookup" }),
        derive: None,
        dslot: 0,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::Moire,
        label: "MOIRÉ",
        kind: Kind::Measure,
        desc: "Local covariance of the two smallest prime factors' zero sets — bright where the gaskets overlap beyond chance, dark-signed where they avoid each other. Needs two distinct primes (try 6, 10, 15, 30); flat on prime powers. Diverging schemes read best.",
        params: &[ParamDef { key: "moireR", label: "radius", min: 1, max: 16, default: 4, choices: &[] }],
        run: Some(MeasureRun::MultiBlur { init: "moire_init", fields: 3, passes: "moireR", combine: "moire_combine" }),
        derive: Some(derive_moire),
        dslot: 4,
        signed: true,
        sweep: false,
    },
    LayerDef {
        id: LayerId::Primes,
        label: "PRIMES",
        kind: Kind::Measure,
        desc: "Primality of each residue as a field. SPECTRUM spreads the primes by index (a spectral scheme recovers the prime rainbow); MASK is 0/1 — multiply it under another map to see that map on prime cells only.",
        params: &[ParamDef { key: "primesMode", label: "show", min: 0, max: 1, default: 0, choices: &["spectrum", "mask"] }],
        run: Some(MeasureRun::Pointwise { entry: "measure_primes" }),
        derive: Some(derive_primes),
        dslot: 5,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::Stripes,
        label: "STRIPES",
        kind: Kind::Measure,
        desc: "Run length of a periodic A…B…A alternation through each cell. TOLERANCE lets drifting colors count as one band; PERIOD sets band width (1 = A,B,A,B; 2 = A,A,B,B). Runs are followed up to 256 steps each way.",
        params: &[
            ParamDef { key: "stripeDir", label: "direction", min: 0, max: 2, default: 0, choices: DIRECTIONS },
            ParamDef { key: "stripeTol", label: "tolerance", min: 0, max: 1_000_000, default: 0, choices: &[] },
            ParamDef { key: "stripePeriod", label: "period", min: 1, max: 8, default: 1, choices: &[] },
        ],
        run: Some(MeasureRun::Pointwise { entry: "measure_stripes" }),
        derive: Some(derive_stripes),
        dslot: 2,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::LagDiff,
        label: "LAG DIFF",
        kind: Kind::Measure,
        desc: "Wrap-distance between each cell and the cell LAG steps away. Set LAG to a suspected period and the periodic structure cancels to dark — the defects light up. The general form of STRIPES.",
        params: &[
            ParamDef { key: "lagDir", label: "direction", min: 0, max: 2, default: 0, choices: DIRECTIONS },
            ParamDef { key: "lagLen", label: "lag", min: 1, max: 128, default: 1, choices: &[] },
        ],
        run: Some(MeasureRun::Pointwise { entry: "measure_lagdiff" }),
        derive: Some(derive_lagdiff),
        dslot: 3,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::Average,
        label: "AVERAGE",
        kind: Kind::Transform,
        desc: "Blurs the layers beneath it over the triangular lattice, or averages them over recent frames (temporal).",
        params: &[ParamDef { key: "avgAmount", label: "amount", min: 1, max: 64, default: 2, choices: &[] }],
        run: None,
        derive: None,
        dslot: 0,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::BandPass,
        label: "BAND PASS",
        kind: Kind::Transform,
        desc: "blur(KEEP) − blur(CUT) of the layers below. Removes broad trends and pixel speckle; mid-scale structure survives, centered at 0.5 — diverging schemes (cool–warm) read best.",
        params: &[
            ParamDef { key: "bandLo", label: "keep", min: 0, max: 16, default: 1, choices: &[] },
            ParamDef { key: "bandHi", label: "cut", min: 1, max: 24, default: 6, choices: &[] },
        ],
        run: None,
        derive: None,
        dslot: 0,
        signed: false,
        sweep: false,
    },
    LayerDef {
        id: LayerId::Afterimage,
        label: "AFTERIMAGE",
        kind: Kind::Transform,
        desc: "Simulated retinal fatigue: past frames leave a complementary ghost. Under PALETTE BLEND it works per channel, so a red region ghosts cyan. ATTACK and DECAY are frame counts for fatigue to build and release.",
        params: &[
            ParamDef { key: "aiStrength", label: "strength", min: 0, max: 100, default: 45, choices: &[] },
            ParamDef { key: "aiAttack", label: "attack", min: 1, max: 64, default: 3, choices: &[] },
            ParamDef { key: "aiDecay", label: "decay", min: 1, max: 64, default: 16, choices: &[] },
        ],
        run: None,
        derive: None,
        dslot: 0,
        signed: false,
        sweep: true,
    },
];

pub fn def(id: LayerId) -> &'static LayerDef {
    REGISTRY.iter().find(|d| d.id == id).expect("layer in registry")
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlendMode {
    Average,
    Multiply,
    Max,
    Screen,
}

impl BlendMode {
    pub const ALL: [BlendMode; 4] = [BlendMode::Average, BlendMode::Multiply, BlendMode::Max, BlendMode::Screen];
    pub fn label(self) -> &'static str {
        match self {
            BlendMode::Average => "average",
            BlendMode::Multiply => "multiply",
            BlendMode::Max => "max",
            BlendMode::Screen => "screen",
        }
    }
    pub fn note(self) -> &'static str {
        match self {
            BlendMode::Average => "mean of the active measures",
            BlendMode::Multiply => "intersection: product of the measures",
            BlendMode::Max => "brightest measure wins",
            BlendMode::Screen => "union: 1 − ∏(1 − f)",
        }
    }
    pub fn code(self) -> u32 {
        match self {
            BlendMode::Average => 0,
            BlendMode::Multiply => 1,
            BlendMode::Max => 2,
            BlendMode::Screen => 3,
        }
    }
}

/// How a blended field becomes color.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FieldColorMode {
    /// Paint the field through the palette's ramp.
    MathRamp,
    /// Run the measures as brightness on the palette image and the transforms per RGB
    /// channel, so actual colors smear and ghost.
    PaletteBlend,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AvgMode {
    Spatial,
    Temporal,
}

/// What the VALUE layer paints.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ValueStyle {
    /// residue → palette
    Residue,
    /// zero → void, nonzero → near-white
    Mask,
}

/// The user's layer configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerState {
    pub active: Vec<LayerId>,
    pub params: HashMap<&'static str, i32>,
    pub blend: BlendMode,
    pub color_mode: FieldColorMode,
    pub avg_mode: AvgMode,
    pub value_style: ValueStyle,
}

impl Default for LayerState {
    fn default() -> Self {
        let mut params = HashMap::new();
        for d in REGISTRY {
            for p in d.params {
                params.insert(p.key, p.default);
            }
        }
        Self {
            active: vec![LayerId::Value],
            params,
            blend: BlendMode::Average,
            color_mode: FieldColorMode::MathRamp,
            avg_mode: AvgMode::Spatial,
            value_style: ValueStyle::Residue,
        }
    }
}

impl LayerState {
    pub fn is_active(&self, id: LayerId) -> bool {
        self.active.contains(&id)
    }

    pub fn set_active(&mut self, id: LayerId, on: bool) {
        if on {
            if !self.is_active(id) {
                self.active.push(id);
            }
        } else {
            self.active.retain(|&x| x != id);
        }
    }

    /// Parameter value clamped to its declared range.
    pub fn param(&self, key: &str) -> i32 {
        let def = REGISTRY.iter().flat_map(|d| d.params.iter()).find(|p| p.key == key).expect("known param");
        self.params.get(key).copied().unwrap_or(def.default).clamp(def.min, def.max)
    }

    /// Reset to "residue → palette".
    pub fn preset_residue(&mut self) {
        self.active = vec![LayerId::Value];
        self.value_style = ValueStyle::Residue;
    }

    /// "zero / nonzero mask".
    pub fn preset_mask(&mut self) {
        self.active = vec![LayerId::Value];
        self.value_style = ValueStyle::Mask;
    }

    /// "p-adic valuation heatmap" = the KUMMER measure alone through the ramp.
    pub fn preset_valuation(&mut self) {
        self.active = vec![LayerId::Kummer];
        self.color_mode = FieldColorMode::MathRamp;
    }

    /// Which of the three presets the current state is, if any.
    pub fn preset(&self) -> Option<Preset> {
        let mut a = self.active.clone();
        a.sort_by_key(|id| REGISTRY.iter().position(|d| d.id == *id));
        match (a.as_slice(), self.value_style, self.color_mode) {
            ([LayerId::Value], ValueStyle::Residue, _) => Some(Preset::Residue),
            ([LayerId::Value], ValueStyle::Mask, _) => Some(Preset::Mask),
            ([LayerId::Kummer], _, FieldColorMode::MathRamp) => Some(Preset::Valuation),
            _ => None,
        }
    }

    /// The pipeline the GPU runs for this state, derived from the kinds.
    pub fn plan(&self) -> Plan {
        let measures: Vec<&'static LayerDef> = REGISTRY.iter().filter(|d| d.kind == Kind::Measure && self.is_active(d.id)).collect();
        let transforms: Vec<&'static LayerDef> = REGISTRY.iter().filter(|d| d.kind == Kind::Transform && self.is_active(d.id)).collect();
        let wants_value = self.is_active(LayerId::Value);
        let field_active = !measures.is_empty() || !transforms.is_empty();
        Plan { measures, transforms, wants_value, field_active }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preset {
    Residue,
    Mask,
    Valuation,
}

/// What to run this frame.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Active measures in registry order.
    pub measures: Vec<&'static LayerDef>,
    /// Active transforms in registry (= execution) order.
    pub transforms: Vec<&'static LayerDef>,
    /// VALUE is checked: fold the residue ratio into the blend as one more layer.
    pub wants_value: bool,
    /// Something beyond plain residue coloring is on, so the field pipeline runs.
    pub field_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_is_pure_residue() {
        let s = LayerState::default();
        let p = s.plan();
        assert!(!p.field_active);
        assert!(p.measures.is_empty());
        assert_eq!(s.preset(), Some(Preset::Residue));
    }

    #[test]
    fn derived_powers_match_checked_arithmetic() {
        assert_eq!(pow_or_zero(2, 31), 1 << 31);
        assert_eq!(pow_or_zero(2, 32), 0);
        assert_eq!(pow_or_zero(3, 20), 3486784401);
        assert_eq!(pow_or_zero(3, 21), 0);
        assert_eq!(pow_or_zero(7, 0), 1);
        let mut s = LayerState::default();
        s.params.insert("lucasScale", 12);
        s.params.insert("padicDigit", 3);
        let md = Modulus::new(7);
        assert_eq!(derive_lucas(&s, &md), [0, 12, 0, 0]);
        assert_eq!(derive_padic(&s, &md), [343, 3, 0, 0]);
        let dslots: Vec<u32> = REGISTRY.iter().filter(|d| d.derive.is_some()).map(|d| d.dslot).collect();
        let mut sorted = dslots.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), dslots.len(), "derived slots are unique");
        assert!(dslots.iter().all(|&d| d < 8));
    }

    #[test]
    fn plan_is_derived_from_kinds() {
        let mut s = LayerState::default();
        s.set_active(LayerId::Afterimage, true);
        s.set_active(LayerId::Density, true);
        s.set_active(LayerId::Average, true);
        let p = s.plan();
        assert_eq!(p.measures.iter().map(|d| d.id).collect::<Vec<_>>(), vec![LayerId::Density]);
        assert_eq!(p.transforms.iter().map(|d| d.id).collect::<Vec<_>>(), vec![LayerId::Average, LayerId::Afterimage]);
        assert!(p.wants_value && p.field_active);
        assert_eq!(s.param("densityR"), 4);
    }
}
