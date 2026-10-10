//! Modulus sweep: which moduli are visited, in which order, and the playback rules.
//!
//! A port of the reference's MOD_MODES / INTERVAL_SEQS / buildSeqMods / snapMod /
//! nextModInDir / playBounds and the play timer (`reference/pascal.before-exploration.html`,
//! identical in `pascal.html`). Pure logic, no UI: the app calls [`Sweep::step`],
//! [`Sweep::tick`] and [`Sweep::snap`].

use std::sync::OnceLock;

use crate::math::MOD_CAP;

/// Most moduli one sequence enumerates, and the iteration guard while accumulating.
const SEQ_MAX: usize = 5000;
const SEQ_STEP_GUARD: usize = 100_000;

// ---------------------------------------------------------------------------------------
// Prime families up to MOD_CAP (one sieve, built on first use).
// ---------------------------------------------------------------------------------------

pub struct PrimeTables {
    pub primes: Vec<u32>,
    composite: Vec<bool>,
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
        PrimeTables { primes, composite, semiprimes, sphenics }
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

/// Increment number `i` of an increment sequence (`seqIncrement` in the reference).
fn seq_increment(seq: Seq, i: usize, fixed: u32) -> u64 {
    let t = prime_tables();
    let pr = &t.primes;
    match seq {
        Seq::Pi => {
            let d = PI_DIGITS.as_bytes();
            (d[i % d.len()] - b'0') as u64
        }
        Seq::Prime => *pr.get(i).unwrap_or(pr.last().unwrap()) as u64,
        Seq::Square => ((i + 1) * (i + 1)) as u64,
        Seq::Pow2 => 1u64.checked_shl(i as u32).unwrap_or(u64::MAX),
        Seq::Semi2p => match (pr.get(i), pr.get(i + 1)) {
            (Some(a), Some(b)) => 2 * (b - a) as u64,
            _ => 2,
        },
        Seq::Semi3p => {
            // the primes with 3 left out (3·3 is a square, not a distinct-pair semiprime)
            let idx = |j: usize| if j == 0 { pr.first() } else { pr.get(j + 1) };
            match (idx(i), idx(i + 1)) {
                (Some(a), Some(b)) => 3 * (b - a) as u64,
                _ => 3,
            }
        }
        _ => fixed.max(1) as u64,
    }
}

/// The list of moduli a sequence visits in `[start, hi]` (`buildSeqMods`).
pub fn seq_mods(start: u32, hi: u32, seq: Seq, fixed: u32) -> Vec<u32> {
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
                let (mut a, mut b) = (1u64, 2u64);
                while out.len() < SEQ_MAX && b <= hi as u64 {
                    if b >= lo as u64 {
                        out.push(b as u32);
                    }
                    let c = a + b;
                    a = b;
                    b = c;
                }
            }
            Seq::PrimeSemi => {
                let doubled: Vec<u32> = t.primes.iter().map(|&p| p.saturating_mul(2)).collect();
                merge(&t.primes, &doubled, &mut out);
            }
            Seq::SemiFull => out.extend(t.semiprimes.iter().copied().filter(|&v| v >= lo && v <= hi).take(SEQ_MAX)),
            Seq::PrimeSemiFull => merge(&t.primes, &t.semiprimes, &mut out),
            Seq::Sphenic => out.extend(t.sphenics.iter().copied().filter(|&v| v >= lo && v <= hi).take(SEQ_MAX)),
            _ => unreachable!(),
        }
        return out;
    }
    let mut v = start as u64;
    if v >= 2 && v <= hi as u64 {
        out.push(v as u32);
    }
    for i in 0..SEQ_STEP_GUARD {
        if out.len() >= SEQ_MAX {
            break;
        }
        let inc = seq_increment(seq, i, fixed);
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
    /// Cached walk lists, rebuilt when the mode parameters change.
    cache: Option<(ModMode, PrimeOrder, Seq, u32, u32, Vec<u32>)>,
}

impl Default for Sweep {
    fn default() -> Self {
        Sweep {
            mode: ModMode::All,
            prime_order: PrimeOrder::All,
            seq: Seq::Fixed,
            step: 3,
            start: 2,
            end: 128,
            speed_ms: 150,
            looping: true,
            playing: false,
            dir: 1,
            tween_ms: 0,
            cache: None,
        }
    }
}

impl Sweep {
    fn anchor(&self) -> u32 {
        clamp_mod(self.start as i64)
    }

    /// The explicit walk list of list-based modes (primes in walk order, or a sequence).
    fn pool(&mut self) -> Option<&[u32]> {
        let key = (self.mode, self.prime_order, self.seq, self.step, self.anchor());
        let list_mode = self.mode == ModMode::Prime || (self.mode == ModMode::Interval && self.seq != Seq::Fixed);
        if !list_mode {
            return None;
        }
        let stale = match &self.cache {
            Some((m, o, s, st, a, _)) => (*m, *o, *s, *st, *a) != key,
            None => true,
        };
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
                seq_mods(self.anchor(), MOD_CAP, self.seq, self.step)
            };
            self.cache = Some((key.0, key.1, key.2, key.3, key.4, list));
        }
        self.cache.as_ref().map(|c| c.5.as_slice())
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
}
