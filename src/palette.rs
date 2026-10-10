//! Color tables, reproduced from the `<core>` block of `reference/pascal-core.html` so
//! that screenshots are comparable pixel for pixel. All arithmetic is done in f64 with
//! JavaScript's rounding rules, then quantized to 8-bit tables that the GPU indexes.

/// Canvas background and the void color of residue 0.
pub const BG: [u8; 3] = [10, 10, 10];
/// Residue 1 in the spectral palette.
pub const LIGHT: [u8; 3] = [236, 236, 226];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scheme {
    Spectral,
    Viridis,
    Magma,
    Inferno,
    Turbo,
    Cividis,
    CoolWarm,
    Mono,
    /// The three gradients of `reference/ChatGPT-version-pascal-square-scanner`.
    Ember,
    Ocean,
    Moss,
}

impl Scheme {
    pub const ALL: [Scheme; 11] = [
        Scheme::Spectral,
        Scheme::Viridis,
        Scheme::Magma,
        Scheme::Inferno,
        Scheme::Turbo,
        Scheme::Cividis,
        Scheme::CoolWarm,
        Scheme::Mono,
        Scheme::Ember,
        Scheme::Ocean,
        Scheme::Moss,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Scheme::Spectral => "spectral",
            Scheme::Viridis => "viridis",
            Scheme::Magma => "magma",
            Scheme::Inferno => "inferno",
            Scheme::Turbo => "turbo",
            Scheme::Cividis => "cividis",
            Scheme::CoolWarm => "cool–warm",
            Scheme::Mono => "mono",
            Scheme::Ember => "ember",
            Scheme::Ocean => "ocean",
            Scheme::Moss => "moss",
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Scheme::Spectral => "spectral",
            Scheme::Viridis => "viridis",
            Scheme::Magma => "magma",
            Scheme::Inferno => "inferno",
            Scheme::Turbo => "turbo",
            Scheme::Cividis => "cividis",
            Scheme::CoolWarm => "coolwarm",
            Scheme::Mono => "mono",
            Scheme::Ember => "ember",
            Scheme::Ocean => "ocean",
            Scheme::Moss => "moss",
        }
    }

    /// Control-point stops (RGB 0–255), linearly interpolated. `None` for the two
    /// analytic schemes (spectral, mono).
    fn stops(self) -> Option<&'static [[u8; 3]]> {
        const VIRIDIS: &[[u8; 3]] = &[[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37]];
        const MAGMA: &[[u8; 3]] = &[[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]];
        const INFERNO: &[[u8; 3]] = &[[0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85], [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164]];
        const TURBO: &[[u8; 3]] = &[[48, 18, 59], [70, 107, 227], [40, 174, 227], [27, 229, 181], [124, 253, 90], [201, 237, 50], [252, 184, 49], [245, 109, 29], [191, 40, 9], [122, 4, 3]];
        const CIVIDIS: &[[u8; 3]] = &[[0, 32, 76], [0, 42, 102], [39, 63, 108], [75, 82, 107], [110, 101, 105], [146, 120, 99], [185, 141, 85], [223, 164, 65], [253, 190, 44], [255, 233, 69]];
        const COOLWARM: &[[u8; 3]] = &[[59, 76, 192], [98, 130, 234], [141, 176, 254], [184, 208, 249], [221, 221, 221], [245, 196, 173], [244, 154, 123], [222, 96, 77], [180, 4, 38]];
        // the square scanner's `paletteColor` stops
        const EMBER: &[[u8; 3]] = &[[17, 19, 26], [92, 39, 45], [255, 111, 74], [255, 226, 155]];
        const OCEAN: &[[u8; 3]] = &[[10, 19, 31], [22, 76, 109], [54, 190, 201], [197, 255, 239]];
        const MOSS: &[[u8; 3]] = &[[16, 24, 20], [46, 88, 57], [131, 178, 92], [231, 235, 164]];
        match self {
            Scheme::Viridis => Some(VIRIDIS),
            Scheme::Magma => Some(MAGMA),
            Scheme::Inferno => Some(INFERNO),
            Scheme::Turbo => Some(TURBO),
            Scheme::Cividis => Some(CIVIDIS),
            Scheme::CoolWarm => Some(COOLWARM),
            Scheme::Ember => Some(EMBER),
            Scheme::Ocean => Some(OCEAN),
            Scheme::Moss => Some(MOSS),
            Scheme::Spectral | Scheme::Mono => None,
        }
    }
}

/// JavaScript `Math.round` for non-negative values: round half up, without the
/// `floor(x + 0.5)` precision artifact.
pub fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

/// `hslRGB(h, s, l)` from the reference: h in degrees, s and l in 0..1, RGB 0..255.
pub fn hsl_rgb(h: f64, s: f64, l: f64) -> [u8; 3] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = (((h % 360.0) + 360.0) % 360.0) / 60.0;
    let x = c * (1.0 - ((hp % 2.0) - 1.0).abs());
    let (r, g, b) = if hp < 1.0 {
        (c, x, 0.0)
    } else if hp < 2.0 {
        (x, c, 0.0)
    } else if hp < 3.0 {
        (0.0, c, x)
    } else if hp < 4.0 {
        (0.0, x, c)
    } else if hp < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m0 = l - c / 2.0;
    [js_round((r + m0) * 255.0) as u8, js_round((g + m0) * 255.0) as u8, js_round((b + m0) * 255.0) as u8]
}

/// `rampRGB(scheme, t)`: the palette ramp at `t ∈ [0, 1]`.
pub fn ramp_rgb(scheme: Scheme, t: f64) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    if let Some(stops) = scheme.stops() {
        let x = t * (stops.len() - 1) as f64;
        let i = (x.floor() as usize).min(stops.len() - 2);
        let f = x - i as f64;
        let (a, b) = (stops[i], stops[i + 1]);
        let ch = |k: usize| js_round(a[k] as f64 + (b[k] as f64 - a[k] as f64) * f) as u8;
        return [ch(0), ch(1), ch(2)];
    }
    match scheme {
        Scheme::Spectral => hsl_rgb(t * 300.0, 0.95, 0.55),
        _ => {
            let g = js_round(t * 255.0) as u8;
            [g, g, g]
        }
    }
}

/// `residueLevel` at a whole modulus: 0 = zero (void), 1 = exactly 1 (white, spectral
/// only), 2..=255 = palette position. Spectral spreads residues 2..m−1 red → violet; the
/// other schemes spread 0..m−1 over the whole ramp.
pub fn residue_level(r: u32, m: u32, spectral: bool) -> u8 {
    if r == 0 {
        return 0;
    }
    let x = r as f64;
    let mu = m as f64;
    let t = if spectral {
        if r == 1 {
            return 1;
        }
        (x - 2.0) / f64::max(1.0, mu - 3.0)
    } else {
        x / (mu - 1.0)
    };
    2 + js_round(253.0 * t.clamp(0.0, 1.0)) as u8
}

/// Palette positions a color offset rotates: levels 2..=255 (0 is the void of a zero
/// residue and 1 the white of residue 1 in spectral; both stay put).
pub const RAMP_LEVELS: u32 = 254;

/// The level shift for a fractional color offset (the square scanner's "fractional
/// offset": colors move by `offset` of the ramp, wrapping around), in 0..RAMP_LEVELS.
pub fn offset_shift(offset: f32) -> u32 {
    if !offset.is_finite() {
        return 0;
    }
    let s = (offset.rem_euclid(1.0) * RAMP_LEVELS as f32).round() as u32;
    s % RAMP_LEVELS
}

/// `level` moved `shift` positions along the ramp, wrapping (levels 0 and 1 are fixed).
/// The WGSL `shift_level` is the same function.
pub fn shift_level(level: u8, shift: u32) -> u8 {
    if level < 2 {
        return level;
    }
    (2 + (level as u32 - 2 + shift) % RAMP_LEVELS) as u8
}

/// Level of every residue `0..m`, packed four per u32 for the GPU.
pub fn residue_levels_packed(m: u32, spectral: bool) -> Vec<u32> {
    let mut out = vec![0u32; (m as usize).div_ceil(4)];
    for r in 0..m {
        out[(r / 4) as usize] |= (residue_level(r, m, spectral) as u32) << (8 * (r % 4));
    }
    out
}

/// Hue (degrees) and saturation (0..1) of an RGB color, as `rgbToHueSat` in the
/// reference. PALETTE BLEND re-lights the palette color with the field as lightness.
pub fn hue_sat(rgb: [u8; 3]) -> (f64, f64) {
    let r = rgb[0] as f64 / 255.0;
    let g = rgb[1] as f64 / 255.0;
    let b = rgb[2] as f64 / 255.0;
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let d = mx - mn;
    let mut h = 0.0;
    if d > 0.0 {
        h = if mx == r {
            ((g - b) / d) % 6.0
        } else if mx == g {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        };
        h *= 60.0;
        if h < 0.0 {
            h += 360.0;
        }
    }
    let l = (mx + mn) / 2.0;
    let s = if d == 0.0 { 0.0 } else { d / (1.0 - (2.0 * l - 1.0).abs()) };
    (h, s)
}

/// The two 256-entry tables the shaders index, plus hue/saturation per residue level.
#[derive(Clone, Debug)]
pub struct Palette {
    pub scheme: Scheme,
    /// `resLUT`: residue levels → color (0 void, 1 white, 2..255 ramp).
    pub res: [[u8; 3]; 256],
    /// `rampLUT`: field levels 0..255 → color.
    pub ramp: [[u8; 3]; 256],
    /// Hue (degrees) and saturation of each `res` entry.
    pub res_hs: [(f32, f32); 256],
}

impl Palette {
    pub fn new(scheme: Scheme) -> Self {
        let mut res = [[0u8; 3]; 256];
        let mut ramp = [[0u8; 3]; 256];
        let mut res_hs = [(0f32, 0f32); 256];
        for i in 0..256usize {
            res[i] = match i {
                0 => BG,
                1 => LIGHT,
                _ => ramp_rgb(scheme, (i as f64 - 2.0) / 253.0),
            };
            ramp[i] = ramp_rgb(scheme, i as f64 / 255.0);
            let (h, s) = hue_sat(res[i]);
            res_hs[i] = (h as f32, s as f32);
        }
        Self { scheme, res, ramp, res_hs }
    }

    /// GPU layout: 256 packed `0xRRGGBB` residue colors, 256 ramp colors, 256 hues (f32
    /// bits), 256 saturations (f32 bits).
    pub fn gpu_words(&self) -> Vec<u32> {
        let pack = |c: [u8; 3]| (c[0] as u32) << 16 | (c[1] as u32) << 8 | c[2] as u32;
        let mut w = Vec::with_capacity(1024);
        w.extend(self.res.iter().map(|&c| pack(c)));
        w.extend(self.ramp.iter().map(|&c| pack(c)));
        w.extend(self.res_hs.iter().map(|hs| hs.0.to_bits()));
        w.extend(self.res_hs.iter().map(|hs| hs.1.to_bits()));
        w
    }

    /// Color of residue `r` under modulus `m` (what the residue layer paints).
    pub fn residue_color(&self, r: u32, m: u32) -> [u8; 3] {
        self.residue_color_shifted(r, m, 0)
    }

    /// The same with the color offset's level shift (see [`offset_shift`]).
    pub fn residue_color_shifted(&self, r: u32, m: u32, shift: u32) -> [u8; 3] {
        self.res[shift_level(residue_level(r, m, self.scheme == Scheme::Spectral), shift) as usize]
    }

    /// The glow color around zero cells: the ramp at 0.8, as the square scanner's
    /// `paletteColor(.8)`.
    pub fn glow_color(&self) -> [u8; 3] {
        ramp_rgb(self.scheme, 0.8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_match_reference() {
        // from verify-pascal-core.cjs
        assert_eq!(residue_level(0, 7, true), 0);
        assert_eq!(residue_level(1, 7, true), 1);
        assert_eq!(residue_level(2, 7, true), 2);
        assert_eq!(residue_level(6, 7, true), 255);
        assert_eq!(residue_level(6, 7, false), 255);
        assert_eq!(residue_level(1, 2, true), 1);
        assert_eq!(residue_level(2, 3, true), 2);
    }

    #[test]
    fn spectral_anchor_colors() {
        let p = Palette::new(Scheme::Spectral);
        assert_eq!(p.res[0], BG);
        assert_eq!(p.res[1], LIGHT);
        assert_eq!(p.res[2], hsl_rgb(0.0, 0.95, 0.55)); // red
        assert_eq!(p.res[255], hsl_rgb(300.0, 0.95, 0.55)); // violet
        assert_eq!(hsl_rgb(0.0, 0.95, 0.55), [249, 31, 31]);
        assert_eq!(hsl_rgb(120.0, 0.95, 0.55), [31, 249, 31]);
    }

    #[test]
    fn stop_palettes_interpolate() {
        assert_eq!(ramp_rgb(Scheme::Viridis, 0.0), [68, 1, 84]);
        assert_eq!(ramp_rgb(Scheme::Viridis, 1.0), [253, 231, 37]);
        assert_eq!(ramp_rgb(Scheme::Mono, 0.5), [128, 128, 128]);
        // halfway between the first two viridis stops: (68+72)/2 = 70, (1+40)/2 = 20.5 -> 21, (84+120)/2 = 102
        assert_eq!(ramp_rgb(Scheme::Viridis, 0.5 / 9.0), [70, 21, 102]);
    }

    #[test]
    fn scanner_palettes_and_offsets() {
        // the square scanner's stops, end to end
        assert_eq!(ramp_rgb(Scheme::Ember, 0.0), [17, 19, 26]);
        assert_eq!(ramp_rgb(Scheme::Ocean, 1.0), [197, 255, 239]);
        assert_eq!(ramp_rgb(Scheme::Moss, 1.0 / 3.0), [46, 88, 57]);
        // its paletteColor(.8) for ember: 0.8·3 = 2.4 → 0.4 of the way from stop 2 to 3
        assert_eq!(Palette::new(Scheme::Ember).glow_color(), [255, 157, 106]);
        // offsets rotate levels 2..=255 and leave void / white alone
        assert_eq!(offset_shift(0.0), 0);
        assert_eq!(offset_shift(1.0), 0);
        assert_eq!(offset_shift(0.5), 127);
        assert_eq!(offset_shift(-0.25), offset_shift(0.75));
        assert_eq!(offset_shift(f32::NAN), 0);
        for sh in [0u32, 1, 127, 253] {
            assert_eq!(shift_level(0, sh), 0);
            assert_eq!(shift_level(1, sh), 1);
            for lv in 2..=255u8 {
                let s = shift_level(lv, sh);
                assert!(s >= 2);
                assert_eq!(shift_level(s, RAMP_LEVELS - sh), lv, "inverse shift");
            }
        }
        assert_eq!(shift_level(255, 1), 2, "wraps");
        let p = Palette::new(Scheme::Spectral);
        assert_eq!(p.residue_color_shifted(3, 7, 0), p.residue_color(3, 7));
        assert_eq!(p.residue_color_shifted(0, 7, 99), BG);
    }

    #[test]
    fn packed_levels_roundtrip() {
        let m = 1001;
        let packed = residue_levels_packed(m, true);
        for r in 0..m {
            let lv = (packed[(r / 4) as usize] >> (8 * (r % 4))) & 255;
            assert_eq!(lv as u8, residue_level(r, m, true));
        }
    }
}
