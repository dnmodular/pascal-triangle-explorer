//! Modulus sweep: which moduli are visited, in which order, and the playback rules.
//!
//! A port of the reference's MOD_MODES / INTERVAL_SEQS / buildSeqMods / snapMod /
//! nextModInDir / playBounds and the play timer (`reference/pascal.before-exploration.html`,
//! identical in `pascal.html`). Pure logic, no UI: the app calls [`Sweep::step`],
//! [`Sweep::tick`] and [`Sweep::snap`].

use std::sync::OnceLock;

use crate::math::MOD_CAP;

/// Most decimal places between moduli.
pub const MAX_DECIMALS: u32 = 5;

/// Most moduli one sequence enumerates, and the iteration guard while accumulating.
const SEQ_MAX: usize = 5000;
const SEQ_STEP_GUARD: usize = 100_000;

// ---------------------------------------------------------------------------------------
// Prime families up to MOD_CAP (one sieve, built on first use).
// ---------------------------------------------------------------------------------------

pub struct PrimeTables {
    pub primes: Vec<u32>,
    composite: Vec<bool>,
    /// The number of distinct prime factors of v when v is squarefree, else 0xFF.
    sqf_omega: Vec<u8>,
    pub semiprimes: Vec<u32>,
    pub sphenics: Vec<u32>,
}

impl PrimeTables {
    fn build() -> Self {
        let cap = MOD_CAP as usize;
        let mut composite = vec![false; cap + 1];
        composite[0] = true;
        composite[1] = true;
        let mut i = 2;
        while i * i <= cap {
            if !composite[i] {
                let mut j = i * i;
                while j <= cap {
                    composite[j] = true;
                    j += i;
                }
            }
            i += 1;
        }
        let primes: Vec<u32> = (2..=cap).filter(|&v| !composite[v]).map(|v| v as u32).collect();
        // Distinct-pair semiprimes p·q, p < q.
        let mut semiprimes = Vec::new();
        for (a, &p) in primes.iter().enumerate() {
            if (p as u64) * (p as u64) >= MOD_CAP as u64 {
                break;
            }
            for &q in &primes[a + 1..] {
                let v = p as u64 * q as u64;
                if v > MOD_CAP as u64 {
                    break;
                }
                semiprimes.push(v as u32);
            }
        }
        semiprimes.sort_unstable();
        // Sphenic numbers p·q·r of three distinct primes.
        let mut sphenics = Vec::new();
        for (a, &p) in primes.iter().enumerate() {
            if (p as u64).pow(3) > MOD_CAP as u64 {
                break;
            }
            for (b, &q) in primes.iter().enumerate().skip(a + 1) {
                if p as u64 * q as u64 * q as u64 > MOD_CAP as u64 {
                    break;
                }
                for &r in &primes[b + 1..] {
                    let v = p as u64 * q as u64 * r as u64;
                    if v > MOD_CAP as u64 {
                        break;
                    }
                    sphenics.push(v as u32);
                }
            }
        }
        sphenics.sort_unstable();
        let mut sqf_omega = vec![0u8; cap + 1];
        for &p in &primes {
            let p = p as usize;
            for m in (p..=cap).step_by(p) {
                sqf_omega[m] = sqf_omega[m].saturating_add(1);
            }
            if p * p <= cap {
                for m in (p * p..=cap).step_by(p * p) {
                    sqf_omega[m] = 0xFF;
                }
            }
        }
        PrimeTables { primes, composite, sqf_omega, semiprimes, sphenics }
    }

    /// The squarefree products of exactly r distinct primes in [lo, hi], ascending
    /// (r = 1 the primes, 2 the distinct-pair semiprimes, 3 the sphenic numbers).
    pub fn products_of(&self, r: u32, lo: u32, hi: u32) -> Vec<u32> {
        let hi = hi.min(MOD_CAP) as usize;
        (lo.max(2) as usize..=hi).filter(|&v| self.sqf_omega[v] as u32 == r).map(|v| v as u32).take(SEQ_MAX).collect()
    }

    pub fn is_prime(&self, v: u32) -> bool {
        (v as usize) < self.composite.len() && !self.composite[v as usize]
    }

    /// 1-based index of `v` among the primes, 0 when `v` is not prime.
    pub fn rank(&self, v: u32) -> u32 {
        if !self.is_prime(v) {
            return 0;
        }
        self.primes.partition_point(|&p| p < v) as u32 + 1
    }
}

pub fn prime_tables() -> &'static PrimeTables {
    static T: OnceLock<PrimeTables> = OnceLock::new();
    T.get_or_init(PrimeTables::build)
}

// ---------------------------------------------------------------------------------------
// Modes and sequences.
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModMode {
    All,
    Prime,
    Composite,
    Even,
    Odd,
    Interval,
}

impl ModMode {
    pub const ALL: [ModMode; 6] = [ModMode::All, ModMode::Prime, ModMode::Composite, ModMode::Even, ModMode::Odd, ModMode::Interval];
    pub fn label(self) -> &'static str {
        match self {
            ModMode::All => "all",
            ModMode::Prime => "primes",
            ModMode::Composite => "composites",
            ModMode::Even => "even",
            ModMode::Odd => "odd",
            ModMode::Interval => "interval",
        }
    }
    pub fn note(self) -> &'static str {
        match self {
            ModMode::All => "Every modulus from 2 up. Steps move by 1.",
            ModMode::Prime => "Prime moduli only, where Lucas and Kummer apply exactly.",
            ModMode::Composite => "Composites (4 and up), where several prime gaskets interfere. Pairs well with CRT and MOIRÉ.",
            ModMode::Even => "Even moduli only; every frame keeps the mod-2 Sierpinski skeleton.",
            ModMode::Odd => "Odd moduli only. Steps skip by 2.",
            ModMode::Interval => "Walks a sequence of moduli from the range start: fixed steps, or the increment and landing sequences below.",
        }
    }
}

/// Order in which prime mode walks the primes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrimeOrder {
    All,
    ByRem4,
    Rem1,
    Rem3,
}

impl PrimeOrder {
    pub const ALL: [PrimeOrder; 4] = [PrimeOrder::All, PrimeOrder::ByRem4, PrimeOrder::Rem1, PrimeOrder::Rem3];
    pub fn label(self) -> &'static str {
        match self {
            PrimeOrder::All => "all primes",
            PrimeOrder::ByRem4 => "grouped by p mod 4",
            PrimeOrder::Rem1 => "p ≡ 1 (mod 4)",
            PrimeOrder::Rem3 => "p ≡ 3 (mod 4)",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Seq {
    Fixed,
    Pi,
    Fib,
    Prime,
    Square,
    Pow2,
    Semi2p,
    Semi3p,
    PrimeSemi,
    SemiFull,
    PrimeSemiFull,
    Sphenic,
}

impl Seq {
    pub const ALL: [Seq; 12] = [
        Seq::Fixed,
        Seq::Pi,
        Seq::Fib,
        Seq::Prime,
        Seq::Square,
        Seq::Pow2,
        Seq::Semi2p,
        Seq::Semi3p,
        Seq::PrimeSemi,
        Seq::SemiFull,
        Seq::PrimeSemiFull,
        Seq::Sphenic,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Seq::Fixed => "fixed step",
            Seq::Pi => "π digits",
            Seq::Fib => "Fibonacci",
            Seq::Prime => "primes",
            Seq::Square => "squares",
            Seq::Pow2 => "powers of 2",
            Seq::Semi2p => "2×Pn",
            Seq::Semi3p => "3×Pn",
            Seq::PrimeSemi => "primes + 2×Pn",
            Seq::SemiFull => "semiprimes",
            Seq::PrimeSemiFull => "primes + semiprimes",
            Seq::Sphenic => "sphenic",
        }
    }
    /// The sequence's knobs (none, one or two), each with its standard value: the
    /// preset is the family at those values.
    pub fn knobs(self) -> &'static [Knob] {
        const MUL: Knob = Knob { name: "×", tip: "Each increment times this", lo: 1, hi: 100, standard: 1, flag: false };
        const fn k(standard: u32) -> Knob {
            Knob { name: "k", tip: "k × each prime: the walk lands on k·p", lo: 1, hi: 1000, standard, flag: false }
        }
        const fn distinct(standard: u32) -> Knob {
            Knob { name: "distinct", tip: "Skip the primes that divide k, so k and p are distinct factors (3×3 = 9 is a square)", lo: 0, hi: 1, standard, flag: true }
        }
        const fn r(standard: u32) -> Knob {
            Knob { name: "r", tip: "Squarefree products of r distinct primes: 1 the primes, 2 semiprimes, 3 sphenic, up to 7 (510510 = 2·3·5·7·11·13·17)", lo: 1, hi: 7, standard, flag: false }
        }
        const SEMI2: [Knob; 2] = [k(2), distinct(0)];
        const SEMI3: [Knob; 2] = [k(3), distinct(1)];
        const K2: [Knob; 1] = [k(2)];
        const R2: [Knob; 1] = [r(2)];
        const R3: [Knob; 1] = [r(3)];
        match self {
            Seq::Fixed => &[],
            Seq::Pi | Seq::Prime => &[MUL],
            Seq::Fib => &[
                Knob { name: "a", tip: "First seed: the walk lands on b, a + b, a + 2b, … (1, 2 Fibonacci; 2, 1 Lucas)", lo: 0, hi: 10_000, standard: 1, flag: false },
                Knob { name: "b", tip: "Second seed: the first term landed on", lo: 1, hi: 10_000, standard: 2, flag: false },
            ],
            Seq::Square => &[Knob { name: "power", tip: "Increments 1^e, 2^e, 3^e, … (2 squares, 3 cubes)", lo: 1, hi: 6, standard: 2, flag: false }],
            Seq::Pow2 => &[Knob { name: "base", tip: "Increments 1, b, b², b³, …", lo: 2, hi: 16, standard: 2, flag: false }],
            Seq::Semi2p => &SEMI2,
            Seq::Semi3p => &SEMI3,
            Seq::PrimeSemi => &K2,
            Seq::SemiFull | Seq::PrimeSemiFull => &R2,
            Seq::Sphenic => &R3,
        }
    }

    /// The knob values of the preset (0 for knobs it does not have).
    pub fn standard(self) -> [u32; 2] {
        let ks = self.knobs();
        [ks.first().map_or(0, |k| k.standard), ks.get(1).map_or(0, |k| k.standard)]
    }

    /// The family's name at these knob values: the preset's own when standard.
    pub fn describe(self, knobs: [u32; 2]) -> String {
        if knobs == self.standard() {
            return self.label().to_string();
        }
        let [a, b] = knobs;
        match self {
            Seq::Pi => format!("{a} × π digits"),
            Seq::Prime => format!("{a} × primes"),
            Seq::Fib => format!("Fibonacci from {a}, {b}"),
            Seq::Square => format!("powers i^{a}"),
            Seq::Pow2 => format!("powers of {a}"),
            Seq::Semi2p | Seq::Semi3p => format!("{a}×Pn{}", if b == 1 { ", p ∤ k" } else { "" }),
            Seq::PrimeSemi => format!("primes + {a}×Pn"),
            Seq::SemiFull | Seq::Sphenic => format!("products of {a} primes"),
            Seq::PrimeSemiFull => format!("primes + products of {a}"),
            Seq::Fixed => self.label().to_string(),
        }
    }

    /// Where the walk starts when the preset is chosen (or its knobs change): the k×Pn
    /// walks on their first term, k·(the first prime they use).
    pub fn first_term(self, knobs: [u32; 2]) -> Option<u32> {
        match self {
            Seq::Semi2p | Seq::Semi3p => kpn_primes(knobs).next().map(|p| (p as u64 * knobs[0].max(1) as u64).min(MOD_CAP as u64) as u32),
            _ => None,
        }
    }

    /// Sequences whose terms are landed on directly (not added as increments).
    pub fn absolute(self) -> bool {
        matches!(self, Seq::Fib | Seq::PrimeSemi | Seq::SemiFull | Seq::PrimeSemiFull | Seq::Sphenic)
    }
}

const PI_DIGITS: &str = concat!(
    "3",
    "1415926535897932384626433832795028841971693993751058209749445923",
    "0781640628620899862803482534211706798214808651328230664709384460",
    "9550582231725359408128481117450284102701938521105559644622948954",
    "9303819644288109756659334461284756482337867831652712019091456485",
    "6692346034861045432664821339360726024914127372458700660631558817",
);

/// One knob of a sequence: a number in lo..=hi, or a switch (0 / 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Knob {
    pub name: &'static str,
    pub tip: &'static str,
    pub lo: u32,
    pub hi: u32,
    /// The preset's value.
    pub standard: u32,
    /// A switch (shown as a checkbox).
    pub flag: bool,
}

/// The primes a k×Pn walk multiplies: all of them, or (distinct) those not dividing k.
fn kpn_primes(knobs: [u32; 2]) -> impl Iterator<Item = u32> {
    let (k, distinct) = (knobs[0].max(1), knobs[1] == 1);
    prime_tables().primes.iter().copied().filter(move |&p| !(distinct && k.is_multiple_of(p)))
}

/// The increments of an increment sequence (`seqIncrement` in the reference), with
/// its knobs: the walk adds them one by one to the start.
fn seq_increments(seq: Seq, fixed: u32, knobs: [u32; 2]) -> Box<dyn Iterator<Item = u64>> {
    let pr = &prime_tables().primes;
    let [a, _] = knobs;
    let a = a as u64;
    match seq {
        Seq::Pi => Box::new(PI_DIGITS.bytes().cycle().map(move |d| (d - b'0') as u64 * a)),
        Seq::Prime => Box::new(pr.iter().map(move |&p| p as u64 * a).chain(std::iter::repeat(*pr.last().unwrap() as u64 * a))),
        Seq::Square => Box::new((1u64..).map(move |i| i.checked_pow(a as u32).unwrap_or(u64::MAX))),
        Seq::Pow2 => Box::new((0u32..).map(move |i| a.max(2).checked_pow(i).unwrap_or(u64::MAX))),
        Seq::Semi2p | Seq::Semi3p => {
            // k × the gaps between the primes used: from k·p₀ the walk lands on k·p
            let k = a.max(1);
            let ps: Vec<u32> = kpn_primes(knobs).collect();
            Box::new((0..ps.len().saturating_sub(1)).map(move |i| k * (ps[i + 1] - ps[i]) as u64).chain(std::iter::repeat(k)))
        }
        _ => Box::new(std::iter::repeat(fixed.max(1) as u64)),
    }
}

/// The list of moduli a sequence visits in `[start, hi]` (`buildSeqMods`), at the
/// preset's standard knobs.
pub fn seq_mods(start: u32, hi: u32, seq: Seq, fixed: u32) -> Vec<u32> {
    seq_mods_with(start, hi, seq, fixed, seq.standard())
}

/// The same at any knob values.
pub fn seq_mods_with(start: u32, hi: u32, seq: Seq, fixed: u32, knobs: [u32; 2]) -> Vec<u32> {
    let t = prime_tables();
    let mut out = Vec::new();
    if seq.absolute() {
        let lo = start.max(2);
        let merge = |a: &[u32], b: &[u32], out: &mut Vec<u32>| {
            let (mut i, mut j) = (0, 0);
            while out.len() < SEQ_MAX {
                let x = a.get(i).copied();
                let y = b.get(j).copied();
                let v = match (x, y) {
                    (None, None) => break,
                    (Some(p), Some(s)) if p <= s => {
                        i += 1;
                        if p == s {
                            j += 1;
                        }
                        p
                    }
                    (Some(p), None) => {
                        i += 1;
                        p
                    }
                    (_, Some(s)) => {
                        j += 1;
                        s
                    }
                };
                if v > hi {
                    break;
                }
                if v >= lo {
                    out.push(v);
                }
            }
        };
        match seq {
            Seq::Fib => {
                // b, a + b, …: ascending once past the seeds (only rising terms are kept)
                let (mut a, mut b) = (knobs[0] as u64, knobs[1].max(1) as u64);
                while out.len() < SEQ_MAX && b <= hi as u64 {
                    if b >= lo as u64 && out.last().is_none_or(|&l| (l as u64) < b) {
                        out.push(b as u32);
                    }
                    let c = a + b;
                    a = b;
                    b = c;
                }
            }
            Seq::PrimeSemi => {
                let k = knobs[0].max(1);
                let times: Vec<u32> = t.primes.iter().map(|&p| p.saturating_mul(k)).collect();
                merge(&t.primes, &times, &mut out);
            }
            Seq::SemiFull | Seq::Sphenic => out = t.products_of(knobs[0], lo, hi),
            Seq::PrimeSemiFull => merge(&t.primes, &t.products_of(knobs[0], 2, hi), &mut out),
            _ => unreachable!(),
        }
        return out;
    }
    let mut v = start as u64;
    if v >= 2 && v <= hi as u64 {
        out.push(v as u32);
    }
    for inc in seq_increments(seq, fixed, knobs).take(SEQ_STEP_GUARD) {
        if out.len() >= SEQ_MAX {
            break;
        }
        if inc == 0 {
            continue;
        }
        v = v.saturating_add(inc);
        if v > hi as u64 {
            break;
        }
        if v >= 2 {
            out.push(v as u32);
        }
    }
    out
}

fn clamp_mod(v: i64) -> u32 {
    v.clamp(2, MOD_CAP as i64) as u32
}

/// Nearest value of an ascending list to `target`; ties go to the lower one.
fn nearest(arr: &[u32], target: u32) -> u32 {
    if arr.is_empty() {
        return target;
    }
    let lo = arr.partition_point(|&a| a < target).min(arr.len() - 1);
    let above = arr[lo];
    if lo == 0 {
        return above;
    }
    let below = arr[lo - 1];
    if target.abs_diff(below) <= above.abs_diff(target) {
        below
    } else {
        above
    }
}

fn nearest_index(arr: &[u32], target: u32) -> usize {
    let v = nearest(arr, target);
    arr.partition_point(|&a| a < v)
}

// ---------------------------------------------------------------------------------------
// The sweep state.
// ---------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Sweep {
    pub mode: ModMode,
    pub prime_order: PrimeOrder,
    pub seq: Seq,
    /// Step for `Seq::Fixed` (1..=10000).
    pub step: u32,
    /// Knobs set by hand ([`Seq::knobs`]), for the sequence they were set on; any other
    /// sequence (or None) runs at its preset's standard values.
    custom: Option<(Seq, [u32; 2])>,
    /// Playback range; `start` is also the interval anchor.
    pub start: u32,
    pub end: u32,
    /// Milliseconds per frame.
    pub speed_ms: u32,
    pub looping: bool,
    pub playing: bool,
    /// +1 forward, −1 reverse.
    pub dir: i32,
    /// Display-only transition between frames, 0 = hard cuts.
    pub tween_ms: u32,
    /// Decimal places between visited moduli (0 = whole moduli only): a step then moves
    /// 10⁻ᵈ through the gap to the next visited modulus, μ = m + sub·10⁻ᵈ, an analog
    /// glide from modulus to modulus (see [`Sweep::step_fine`]).
    pub decimals: u32,
    /// Cached walk lists, rebuilt when the mode parameters change.
    cache: Option<(PoolKey, Vec<u32>)>,
}

impl Default for Sweep {
    fn default() -> Self {
        Sweep {
            mode: ModMode::All,
            prime_order: PrimeOrder::All,
            seq: Seq::Fixed,
            step: 3,
            custom: None,
            start: 2,
            end: 128,
            speed_ms: 150,
            looping: true,
            playing: false,
            dir: 1,
            tween_ms: 0,
            decimals: 0,
            cache: None,
        }
    }
}

/// What a walk list is built from: mode, prime order, sequence, step, knobs, anchor.
type PoolKey = (ModMode, PrimeOrder, Seq, u32, [u32; 2], u32);

impl Sweep {
    /// Choose sequence `q` at its standard knobs (choosing the current one again resets
    /// it to them); the k×Pn walks start on their first term.
    pub fn choose_seq(&mut self, q: Seq) {
        self.seq = q;
        self.custom = None;
        if let Some(v) = q.first_term(q.standard()) {
            self.start = v;
        }
    }

    /// The current sequence's knob values.
    pub fn knobs(&self) -> [u32; 2] {
        match self.custom {
            Some((q, k)) if q == self.seq => k,
            _ => self.seq.standard(),
        }
    }

    /// Set the knobs by hand (clamped to their ranges). The k×Pn walks move their start
    /// onto their first term again; the start can still be moved off it afterwards,
    /// which shifts the walk.
    pub fn set_knobs(&mut self, mut k: [u32; 2]) {
        if k == self.knobs() {
            return;
        }
        for (v, kn) in k.iter_mut().zip(self.seq.knobs()) {
            *v = (*v).clamp(kn.lo, kn.hi);
        }
        self.custom = Some((self.seq, k));
        if let Some(v) = self.seq.first_term(k) {
            self.start = v;
        }
    }

    /// The knobs are the preset's.
    pub fn standard_knobs(&self) -> bool {
        self.knobs() == self.seq.standard()
    }

    /// The sequence's current name.
    pub fn seq_name(&self) -> String {
        self.seq.describe(self.knobs())
    }

    fn anchor(&self) -> u32 {
        clamp_mod(self.start as i64)
    }

    /// The explicit walk list of list-based modes (primes in walk order, or a sequence).
    fn pool(&mut self) -> Option<&[u32]> {
        let key: PoolKey = (self.mode, self.prime_order, self.seq, self.step, self.knobs(), self.anchor());
        let list_mode = self.mode == ModMode::Prime || (self.mode == ModMode::Interval && self.seq != Seq::Fixed);
        if !list_mode {
            return None;
        }
        let stale = self.cache.as_ref().is_none_or(|(k, _)| *k != key);
        if stale {
            let list = if self.mode == ModMode::Prime {
                let pr = &prime_tables().primes;
                match self.prime_order {
                    PrimeOrder::All => pr.clone(),
                    PrimeOrder::Rem1 => pr.iter().copied().filter(|p| p % 4 == 1).collect(),
                    PrimeOrder::Rem3 => pr.iter().copied().filter(|p| p % 4 == 3).collect(),
                    PrimeOrder::ByRem4 => std::iter::once(2).chain(pr.iter().copied().filter(|p| p % 4 == 1)).chain(pr.iter().copied().filter(|p| p % 4 == 3)).collect(),
                }
            } else {
                seq_mods_with(self.anchor(), MOD_CAP, self.seq, self.step, key.4)
            };
            self.cache = Some((key, list));
        }
        self.cache.as_ref().map(|c| c.1.as_slice())
    }


    /// 10^decimals: the sub-steps per unit of modulus.
    pub fn fine_units(&self) -> u64 {
        10u64.pow(self.decimals.min(MAX_DECIMALS))
    }

    /// The most decimal places the range allows: a decimal modulus μ = P/10ᵈ needs its
    /// numerator P within MOD_CAP (the closed form's tables are built for P).
    pub fn max_decimals(&mut self) -> u32 {
        let (_, hi) = self.bounds();
        let top = hi as u64 + self.gap_after(hi) as u64;
        (0..=MAX_DECIMALS).rev().find(|&d| top * 10u64.pow(d) <= MOD_CAP as u64).unwrap_or(0)
    }

    /// The whole moduli from `m` to the next visited one (1 past the last: its decimals
    /// lead nowhere, so a sub-step there is a whole step).
    pub fn gap_after(&mut self, m: u32) -> u32 {
        let next = self.next_in_dir(m, 1);
        if next > m { next - m } else { 1 }
    }

    /// One step from μ = m + sub·10⁻ᵈ (`pos` = (m, sub)) in direction `dir`, through the
    /// gaps between visited moduli in 10⁻ᵈ steps: forward from m to its next visited
    /// modulus m' takes (m' − m)·10ᵈ steps, the last one landing on m' itself. With no
    /// decimals it is the whole step [`Sweep::next_in_dir`].
    pub fn step_fine(&mut self, pos: (u32, u64), dir: i32) -> (u32, u64) {
        let (m, sub) = pos;
        if self.decimals == 0 {
            return (self.next_in_dir(m, dir), 0);
        }
        let units = self.fine_units();
        if dir > 0 {
            if sub + 1 < self.gap_after(m) as u64 * units {
                return (m, sub + 1);
            }
            return (self.next_in_dir(m, 1), 0);
        }
        if sub > 0 {
            return (m, sub - 1);
        }
        let prev = self.next_in_dir(m, -1);
        if prev < m {
            (prev, (m - prev) as u64 * units - 1)
        } else {
            (prev, 0) // wrapped: land on the whole modulus
        }
    }

    /// `n` fine steps (sign = direction).
    pub fn step_fine_by(&mut self, pos: (u32, u64), n: i32) -> (u32, u64) {
        let mut p = pos;
        for _ in 0..n.unsigned_abs() {
            p = self.step_fine(p, n.signum());
        }
        p
    }

    /// One frame of playback from `pos`: a fine step in the play direction, wrapping at
    /// the range ends (looping) or stopping there; the range ends are whole moduli. With
    /// no decimals it is [`Sweep::tick`].
    pub fn tick_fine(&mut self, pos: (u32, u64)) -> (u32, u64) {
        if self.decimals == 0 {
            return (self.tick(pos.0), 0);
        }
        let (lo, hi) = self.bounds();
        let dir = self.dir;
        let next = self.step_fine(pos, dir);
        let past = if dir > 0 { next.0 > hi || (next.0 == hi && next.1 > 0) || next.0 < pos.0 } else { next.0 < lo || next.0 > pos.0 };
        if !past {
            return next;
        }
        if self.looping {
            (if dir > 0 { lo } else { hi }, 0)
        } else {
            self.playing = false;
            (if dir > 0 { hi } else { lo }, 0)
        }
    }

    /// The modulus nearest to `v` that the active mode allows (`snapMod`).
    pub fn snap(&mut self, v: i64) -> u32 {
        let x = clamp_mod(v);
        let t = prime_tables();
        // snap by value: the grouped-by-mod-4 bank is not ascending, but it holds every prime
        if self.mode == ModMode::Prime && self.prime_order == PrimeOrder::ByRem4 {
            return nearest(&t.primes, x);
        }
        if let Some(pool) = self.pool() {
            if !pool.is_empty() {
                return nearest(pool, x);
            }
        }
        match self.mode {
            ModMode::Interval => {
                let a = self.anchor() as i64;
                let n = self.step.max(1) as i64;
                let k = ((x as i64 - a) as f64 / n as f64).round().max(0.0) as i64;
                clamp_mod(a + k * n)
            }
            ModMode::Even => {
                if x.is_multiple_of(2) {
                    x
                } else if x < MOD_CAP {
                    x + 1
                } else {
                    x - 1
                }
            }
            ModMode::Odd => {
                if x % 2 == 1 {
                    x
                } else if x < MOD_CAP {
                    x + 1
                } else {
                    x - 1
                }
            }
            ModMode::Composite => {
                if !t.is_prime(x) {
                    return x.max(4);
                }
                let (mut up, mut down) = (x + 1, x - 1);
                loop {
                    if down >= 4 && !t.is_prime(down) {
                        return down;
                    }
                    if up <= MOD_CAP && !t.is_prime(up) {
                        return up;
                    }
                    up += 1;
                    down = down.saturating_sub(1);
                }
            }
            _ => x,
        }
    }

    /// Where the modulus lands one step from `cur` in direction `dir` (`nextModInDir`).
    pub fn next_in_dir(&mut self, cur: u32, dir: i32) -> u32 {
        let mode = self.mode;
        let step = self.step.max(1) as i64;
        if let Some(pool) = self.pool() {
            if !pool.is_empty() {
                if mode == ModMode::Prime {
                    let n = pool.len() as i64;
                    let next = match pool.iter().position(|&p| p == cur) {
                        None => 0,
                        Some(i) => (i as i64 + dir as i64).rem_euclid(n),
                    };
                    return pool[next as usize];
                }
                let i = nearest_index(pool, cur) as i64;
                let next = (i + dir as i64).clamp(0, pool.len() as i64 - 1);
                return pool[next as usize];
            }
        }
        let t = prime_tables();
        match mode {
            ModMode::Composite => {
                let mut v = cur as i64 + dir as i64;
                while v >= 4 && v <= MOD_CAP as i64 && t.is_prime(v as u32) {
                    v += dir as i64;
                }
                v.clamp(4, MOD_CAP as i64) as u32
            }
            ModMode::Interval => clamp_mod(cur as i64 + dir as i64 * step),
            ModMode::All => clamp_mod(cur as i64 + dir as i64),
            _ => clamp_mod(cur as i64 + dir as i64 * 2),
        }
    }

    /// `n` steps (sign = direction), stopping at the ends of non-wrapping walks.
    pub fn step_by(&mut self, cur: u32, n: i32) -> u32 {
        let mut v = cur;
        for _ in 0..n.unsigned_abs() {
            v = self.next_in_dir(v, n.signum());
        }
        v
    }

    /// First and last modulus of the playback range, snapped to the mode (`playBounds`).
    pub fn bounds(&mut self) -> (u32, u32) {
        let (mut lo, mut hi) = (clamp_mod(self.start as i64), clamp_mod(self.end as i64));
        if lo > hi {
            std::mem::swap(&mut lo, &mut hi);
        }
        (self.snap(lo as i64), self.snap(hi as i64))
    }

    /// One play tick from `cur`: the next frame, stopping playback at the range edge when
    /// not looping (the reference's play interval).
    pub fn tick(&mut self, cur: u32) -> u32 {
        let (lo, hi) = self.bounds();
        let dir = self.dir;
        let target = self.next_in_dir(cur, dir);
        let at_edge = (dir > 0 && (target >= hi || target == cur)) || (dir < 0 && (target <= lo || target == cur));
        let mut next = target;
        if dir > 0 && target > hi {
            next = if self.looping { lo } else { hi };
        }
        if dir < 0 && target < lo {
            next = if self.looping { hi } else { lo };
        }
        if at_edge && !self.looping {
            next = if dir > 0 { hi } else { lo };
            self.playing = false;
        } else if at_edge && self.looping && target == cur {
            next = if dir > 0 { lo } else { hi };
        }
        next
    }

    /// The first few moduli the active mode visits from the range start (panel preview).
    pub fn preview(&mut self, count: usize) -> Vec<u32> {
        let (lo, _) = self.bounds();
        let mut out = vec![lo];
        let mut v = lo;
        while out.len() < count {
            let n = self.next_in_dir(v, 1);
            if n <= v {
                break;
            }
            out.push(n);
            v = n;
        }
        out
    }

    /// Number of moduli in the range for the active mode (capped at `limit`).
    pub fn frames_in_range(&mut self, limit: usize) -> usize {
        let (lo, hi) = self.bounds();
        let mut v = lo;
        let mut n = 1;
        while n < limit {
            let next = self.next_in_dir(v, 1);
            if next <= v || next > hi {
                break;
            }
            v = next;
            n += 1;
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prime_families() {
        let t = prime_tables();
        assert_eq!(&t.primes[..6], &[2, 3, 5, 7, 11, 13]);
        assert_eq!(t.primes.len(), 78498); // π(10^6)
        assert_eq!(&t.semiprimes[..8], &[6, 10, 14, 15, 21, 22, 26, 33]);
        assert_eq!(&t.sphenics[..6], &[30, 42, 66, 70, 78, 102]);
        assert_eq!(t.rank(2), 1);
        assert_eq!(t.rank(13), 6);
        assert_eq!(t.rank(15), 0);
    }

    #[test]
    fn sequences_match_reference() {
        // increments accumulate from the start
        assert_eq!(&seq_mods(2, 1000, Seq::Fixed, 3)[..4], &[2, 5, 8, 11]);
        assert_eq!(&seq_mods(2, 1000, Seq::Pi, 1)[..6], &[2, 5, 6, 10, 11, 16]);
        assert_eq!(&seq_mods(2, 1000, Seq::Square, 1)[..5], &[2, 3, 7, 16, 32]);
        assert_eq!(&seq_mods(2, 1000, Seq::Pow2, 1)[..5], &[2, 3, 5, 9, 17]);
        assert_eq!(&seq_mods(2, 1000, Seq::Prime, 1)[..5], &[2, 4, 7, 12, 19]);
        // 2×Pn from 4: 4, 6, 10, 14, 22, 26 (2 × consecutive primes)
        assert_eq!(&seq_mods(4, 1000, Seq::Semi2p, 1)[..6], &[4, 6, 10, 14, 22, 26]);
        // 3×Pn from 6: 6, 15, 21, 33, 39 (3 × primes other than 3)
        assert_eq!(&seq_mods(6, 1000, Seq::Semi3p, 1)[..5], &[6, 15, 21, 33, 39]);
        // landing sequences
        assert_eq!(&seq_mods(2, 1000, Seq::Fib, 1)[..6], &[2, 3, 5, 8, 13, 21]);
        assert_eq!(&seq_mods(2, 1000, Seq::PrimeSemi, 1)[..8], &[2, 3, 4, 5, 6, 7, 10, 11]);
        assert_eq!(&seq_mods(2, 1000, Seq::SemiFull, 1)[..4], &[6, 10, 14, 15]);
        assert_eq!(&seq_mods(2, 1000, Seq::PrimeSemiFull, 1)[..8], &[2, 3, 5, 6, 7, 10, 11, 13]);
        assert_eq!(&seq_mods(40, 1000, Seq::Sphenic, 1)[..3], &[42, 66, 70]);
    }

    #[test]
    fn families_open_their_presets_up() {
        let t = prime_tables();
        // products of r distinct primes: r = 2 and 3 are the semiprime and sphenic tables
        // (lists stop at SEQ_MAX terms, as the presets' always did)
        assert_eq!(t.products_of(2, 2, 1_000_000), t.semiprimes[..SEQ_MAX]);
        assert_eq!(t.products_of(3, 2, 1_000_000), t.sphenics[..SEQ_MAX.min(t.sphenics.len())]);
        assert_eq!(&t.products_of(1, 2, 30), &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
        assert_eq!(&t.products_of(4, 2, 1000)[..4], &[210, 330, 390, 462]);
        assert_eq!(&t.products_of(7, 2, 1_000_000)[..3], &[510510, 570570, 690690]);
        // k×Pn: 5 × the primes, and 2 × the odd primes when distinct
        assert_eq!(&seq_mods_with(10, 1000, Seq::Semi2p, 1, [5, 0])[..5], &[10, 15, 25, 35, 55]);
        assert_eq!(&seq_mods_with(6, 1000, Seq::Semi2p, 1, [2, 1])[..4], &[6, 10, 14, 22]);
        // the 3×Pn preset is the same family at k 3, distinct
        assert_eq!(seq_mods_with(6, 1000, Seq::Semi2p, 1, [3, 1]), seq_mods(6, 1000, Seq::Semi3p, 1));
        // a start off the first term shifts the whole walk
        assert_eq!(&seq_mods_with(5, 1000, Seq::Semi2p, 1, [2, 0])[..4], &[5, 7, 11, 15]);
        assert_eq!(&seq_mods_with(2, 100, Seq::PrimeSemi, 1, [3, 0])[..8], &[2, 3, 5, 6, 7, 9, 11, 13]);
        assert_eq!(&seq_mods_with(2, 1000, Seq::SemiFull, 1, [4, 0])[..2], &[210, 330]);
        // Lucas numbers from seeds 2, 1 (the 1 is below every modulus), cubes, base 3
        assert_eq!(&seq_mods_with(2, 1000, Seq::Fib, 1, [2, 1])[..5], &[3, 4, 7, 11, 18]);
        assert_eq!(&seq_mods_with(2, 1000, Seq::Square, 1, [3, 0])[..4], &[2, 3, 11, 38]);
        assert_eq!(&seq_mods_with(2, 1000, Seq::Pow2, 1, [3, 0])[..4], &[2, 3, 6, 15]);
        assert_eq!(&seq_mods_with(2, 1000, Seq::Prime, 1, [2, 0])[..4], &[2, 6, 12, 22]);
        // every preset's standard knobs reproduce it, and every list ascends
        for q in Seq::ALL {
            assert_eq!(seq_mods_with(4, 5000, q, 3, q.standard()), seq_mods(4, 5000, q, 3), "{q:?}");
            assert_eq!(q.describe(q.standard()), q.label());
            for kn in q.knobs() {
                assert!((kn.lo..=kn.hi).contains(&kn.standard), "{q:?} {}", kn.name);
            }
            for knobs in [[1, 0], [7, 1], [3, 1]] {
                let l = seq_mods_with(4, 5000, q, 3, knobs);
                assert!(l.windows(2).all(|w| w[0] < w[1]), "{q:?} {knobs:?}");
            }
        }
    }

    #[test]
    fn choosing_a_preset_again_resets_its_knobs() {
        let mut s = Sweep { mode: ModMode::Interval, ..Default::default() };
        s.choose_seq(Seq::Semi2p);
        assert_eq!((s.knobs(), s.start), ([2, 0], 4));
        s.set_knobs([5, 0]);
        assert_eq!((s.seq_name().as_str(), s.start), ("5×Pn", 10));
        assert_eq!(s.snap(24), 25);
        // the start moved by hand shifts the walk; the knobs stay
        s.start = 11;
        assert_eq!(s.knobs(), [5, 0]);
        // the same preset again: back to the standard 2×Pn, on its first term
        s.choose_seq(Seq::Semi2p);
        assert!(s.standard_knobs());
        assert_eq!((s.seq_name().as_str(), s.start), ("2×Pn", 4));
        assert_eq!(s.snap(25), 26);
        // knobs belong to their sequence: another one runs at its own standard values
        s.set_knobs([7, 1]);
        s.seq = Seq::Sphenic;
        assert_eq!(s.knobs(), [3, 0]);
        // out-of-range knobs are clamped
        s.set_knobs([99, 0]);
        assert_eq!(s.knobs(), [7, 0]);
    }

    #[test]
    fn snapping_and_stepping() {
        let mut s = Sweep { mode: ModMode::Prime, ..Default::default() };
        assert_eq!(s.snap(8), 7);
        assert_eq!(s.snap(9), 7); // tie goes to the lower prime, as in the reference
        assert_eq!(s.snap(10), 11);
        assert_eq!(s.next_in_dir(7, 1), 11);
        assert_eq!(s.next_in_dir(2, -1), *prime_tables().primes.last().unwrap()); // prime bank wraps
        s.mode = ModMode::Composite;
        assert_eq!(s.snap(7), 6);
        assert_eq!(s.next_in_dir(10, 1), 12);
        assert_eq!(s.next_in_dir(4, -1), 4);
        s.mode = ModMode::Even;
        assert_eq!(s.snap(7), 8);
        assert_eq!(s.next_in_dir(8, 1), 10);
        s.mode = ModMode::Interval;
        s.seq = Seq::Fixed;
        s.start = 5;
        s.step = 4;
        assert_eq!(s.snap(12), 13);
        assert_eq!(s.next_in_dir(13, 1), 17);
        s.seq = Seq::Fib;
        assert_eq!(s.snap(20), 21);
        assert_eq!(s.next_in_dir(21, 1), 34);
        assert_eq!(s.step_by(5, 3), 21);
    }

    #[test]
    fn playback_loops_and_stops_at_range_ends() {
        let mut s = Sweep { start: 10, end: 13, looping: true, playing: true, ..Default::default() };
        let mut v = 10;
        let seen: Vec<u32> = (0..6).map(|_| {
            v = s.tick(v);
            v
        }).collect();
        assert_eq!(seen, vec![11, 12, 13, 10, 11, 12]);
        s.looping = false;
        let mut v = 12;
        v = s.tick(v);
        assert_eq!(v, 13);
        assert!(!s.playing, "stops on reaching the edge");
        s.playing = true;
        s.dir = -1;
        let mut v = 11;
        v = s.tick(v);
        assert_eq!((v, s.playing), (10, false));
    }

    #[test]
    fn decimals_glide_through_the_gaps() {
        // primes, one decimal: 7 → 7.1 → … → 10.9 → 11
        let mut s = Sweep { mode: ModMode::Prime, decimals: 1, start: 2, end: 30, ..Default::default() };
        assert_eq!(s.step_fine((7, 0), 1), (7, 1));
        assert_eq!(s.step_fine((7, 39), 1), (11, 0), "7 + 3.9 is the last step before 11");
        assert_eq!(s.step_fine_by((7, 0), 40), (11, 0));
        assert_eq!(s.step_fine((11, 0), -1), (7, 39));
        assert_eq!(s.step_fine((7, 5), -1), (7, 4));
        // playback stops (or loops) at the range end, a whole modulus
        s.looping = false;
        s.playing = true;
        assert_eq!(s.tick_fine((29, 0)), (29, 0));
        assert!(!s.playing);
        s.looping = true;
        assert_eq!(s.tick_fine((29, 0)), (2, 0));
        s.dir = -1;
        assert_eq!(s.tick_fine((2, 0)), (29, 0));
        assert_eq!(s.tick_fine((3, 0)), (2, 9));
        // whole steps without decimals
        s.decimals = 0;
        s.dir = 1;
        assert_eq!(s.step_fine((7, 0), 1), (11, 0));
        // the range bounds the decimal places: P = (hi + gap)·10ᵈ within MOD_CAP
        let mut s = Sweep { end: 128, ..Default::default() };
        assert_eq!(s.max_decimals(), 3);
        s.end = 50_000;
        assert_eq!(s.max_decimals(), 1);
    }

}
