//! Number theory shared by the CPU side and the shaders.
//!
//! The GPU computes `C(n,k) mod m` per pixel from a closed form instead of the additive
//! rule, so nothing depends on the row above:
//!
//! * prime `p`           — Lucas: `C(n,k) ≡ ∏ C(n_i, k_i) (mod p)` over base-`p` digits.
//! * prime power `p^e`   — Granville's generalization: with `(x!)_p` the product of the
//!   integers `≤ x` coprime to `p`, and `N_j = ⌊n/p^j⌋ mod p^e` (same for `K_j`, `R_j`
//!   with `r = n − k`),
//!
//!   ```text
//!   C(n,k) / p^ν ≡ (±1)^{c_{≥e−1}} · ∏_j (N_j!)_p / ((K_j!)_p (R_j!)_p)   (mod p^e)
//!   ```
//!
//!   where `ν` is the number of carries when adding `k` and `n − k` in base `p` (Kummer),
//!   `c_{≥e−1}` counts the carries at digit positions `≥ e − 1`, and `±1 = −1` except for
//!   `p = 2, e ≥ 3` where it is `+1`. For `e = 1` this is exactly Lucas.
//! * composite `m`       — one residue per prime-power factor, recombined with Garner's
//!   form of the Chinese remainder theorem (all intermediate products stay below `m`).
//!
//! This module holds the CPU versions (used by the inspector, the in-cell numbers and
//! the tests), the table builders uploaded to the GPU, and the plain additive reference
//! the tests compare everything against.

/// Largest modulus the app accepts (tables are O(m)).
pub const MOD_CAP: u32 = 1_000_000;

/// Most distinct prime factors a `u32` can have (2·3·5·7·11·13·17·19·23 < 2^32 < ·29).
pub const MAX_FACTORS: usize = 9;

/// One prime-power factor `q = p^e` of the modulus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimePower {
    pub p: u32,
    pub e: u32,
    pub q: u32,
}

/// Prime-power factorization of `m ≥ 1`, ascending in `p`.
pub fn factorize(mut m: u32) -> Vec<PrimePower> {
    let mut out = Vec::new();
    let mut p = 2u32;
    while (p as u64) * (p as u64) <= m as u64 {
        if m.is_multiple_of(p) {
            let mut e = 0;
            let mut q = 1u32;
            while m.is_multiple_of(p) {
                m /= p;
                e += 1;
                q *= p;
            }
            out.push(PrimePower { p, e, q });
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if m > 1 {
        out.push(PrimePower { p: m, e: 1, q: m });
    }
    out
}

pub fn is_prime(n: u32) -> bool {
    n >= 2 && factorize(n).len() == 1 && factorize(n)[0].e == 1
}

/// Next prime above `n` (or `n` itself when `from_self`), capped at `MOD_CAP`.
pub fn next_prime(n: u32, dir: i32) -> Option<u32> {
    let mut x = n as i64 + dir as i64;
    while x >= 2 && x <= MOD_CAP as i64 {
        if is_prime(x as u32) {
            return Some(x as u32);
        }
        x += dir as i64;
    }
    None
}

/// Human form of a factorization, e.g. `2³·3·5`.
pub fn factor_string(fs: &[PrimePower]) -> String {
    const SUP: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    fs.iter()
        .map(|f| {
            if f.e == 1 {
                f.p.to_string()
            } else {
                let sup: String = f.e.to_string().chars().map(|c| SUP[c as usize - '0' as usize]).collect();
                format!("{}{}", f.p, sup)
            }
        })
        .collect::<Vec<_>>()
        .join("·")
}

/// `b^e mod m` in u64 (m < 2^32).
pub fn modpow(mut b: u64, mut e: u64, m: u64) -> u64 {
    let mut r = 1 % m;
    b %= m;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % m;
        }
        b = b * b % m;
        e >>= 1;
    }
    r
}

/// Modular inverse of `a` mod `m` (gcd must be 1), by the extended Euclidean algorithm.
pub fn modinv(a: u64, m: u64) -> u64 {
    let (mut old_r, mut r) = (a as i128 % m as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let qt = old_r / r;
        (old_r, r) = (r, old_r - qt * r);
        (old_s, s) = (s, old_s - qt * s);
    }
    debug_assert_eq!(old_r, 1, "modinv: not invertible");
    ((old_s % m as i128 + m as i128) % m as i128) as u64
}

/// Smallest primitive root of an odd prime `p`.
pub fn primitive_root(p: u32) -> u32 {
    if p == 2 {
        return 1;
    }
    let qs: Vec<u32> = factorize(p - 1).iter().map(|f| f.p).collect();
    (2..p)
        .find(|&g| qs.iter().all(|&q| modpow(g as u64, ((p - 1) / q) as u64, p as u64) != 1))
        .unwrap_or(1)
}

// ---------------------------------------------------------------------------------------
// Montgomery arithmetic (R = 2^32), used by the shader for every odd modulus so that a
// 32-bit GPU never needs a 64-bit product. Even prime powers (2^e) use plain wrapping
// multiplication masked to `e` bits instead.
// ---------------------------------------------------------------------------------------

/// `-q^{-1} mod 2^32` for odd `q` (Newton iteration, 5 steps double the precision each time).
pub fn mont_qinv(q: u32) -> u32 {
    debug_assert!(q & 1 == 1);
    let mut inv = q; // correct to 3 bits (q·q ≡ 1 mod 8 for odd q)
    for _ in 0..5 {
        inv = inv.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(inv)));
    }
    inv.wrapping_neg()
}

/// `x·R mod q` — the Montgomery form of `x` (odd `q`).
pub fn to_mont(x: u32, q: u32) -> u32 {
    (((x as u64) << 32) % q as u64) as u32
}

/// REDC: `a·b·R^{-1} mod q` for odd `q < 2^31`. Mirrors the WGSL `mont_mul` exactly.
pub fn mont_mul(a: u32, b: u32, q: u32, qinv: u32) -> u32 {
    let t = (a as u64) * (b as u64);
    let m = (t as u32).wrapping_mul(qinv);
    let u = ((t + (m as u64) * (q as u64)) >> 32) as u32;
    if u >= q {
        u - q
    } else {
        u
    }
}

// ---------------------------------------------------------------------------------------
// Per-prime-power tables.
// ---------------------------------------------------------------------------------------

/// `(x!)_p mod q` and its inverse for `x < q`, plus everything the shader needs to run
/// Granville's formula and Garner's recombination for this factor.
#[derive(Clone, Debug)]
pub struct PrimePowerTables {
    pub f: PrimePower,
    /// `fp[x] = ∏_{1≤i≤x, p∤i} i  (mod q)`, plain integers.
    pub fp: Vec<u32>,
    /// `ifp[x] = fp[x]^{-1} mod q`, plain integers.
    pub ifp: Vec<u32>,
    /// `(p^e!)_p ≡ −1` unless `p = 2, e ≥ 3`.
    pub minus_one: bool,
    /// Montgomery `−q^{-1} mod 2^32` (0 for `p = 2`).
    pub qinv: u32,
    /// Product of the earlier factors' `q` (Garner's `Q_{f−1}`), 1 for the first factor.
    pub garner_q: u32,
    /// `garner_q^{-1} mod q` (plain), 1 for the first factor.
    pub garner_inv: u32,
}

impl PrimePowerTables {
    pub fn new(f: PrimePower, garner_q: u32) -> Self {
        let q = f.q as u64;
        let p = f.p as u64;
        let mut fp = vec![1u32; f.q as usize];
        for x in 1..f.q as usize {
            fp[x] = if (x as u64).is_multiple_of(p) { fp[x - 1] } else { (fp[x - 1] as u64 * x as u64 % q) as u32 };
        }
        let mut ifp = vec![1u32; f.q as usize];
        ifp[f.q as usize - 1] = modinv(fp[f.q as usize - 1] as u64, q) as u32;
        for x in (1..f.q as usize).rev() {
            ifp[x - 1] = if (x as u64).is_multiple_of(p) { ifp[x] } else { (ifp[x] as u64 * x as u64 % q) as u32 };
        }
        let garner_inv = if garner_q == 1 { 1 } else { modinv(garner_q as u64 % q, q) as u32 };
        Self {
            f,
            fp,
            ifp,
            minus_one: !(f.p == 2 && f.e >= 3),
            qinv: if f.p == 2 { 0 } else { mont_qinv(f.q) },
            garner_q,
            garner_inv,
        }
    }

    /// Plain-integer multiplication mod q (CPU reference path).
    #[inline]
    fn mul(&self, a: u32, b: u32) -> u32 {
        ((a as u64 * b as u64) % self.f.q as u64) as u32
    }

    /// `C(n,k) mod p^e` by Granville's theorem (0 when `k > n`). Same loop structure as
    /// the WGSL `binom_mod_prime_power`, so the two can be compared digit for digit.
    pub fn binom(&self, n: u32, k: u32) -> u32 {
        if k > n {
            return 0;
        }
        let PrimePower { p, e, q } = self.f;
        let mut nn = n;
        let mut kk = k;
        let mut carry = 0u32; // carry into the current digit (= borrow of n − k)
        let mut nu = 0u32; // Kummer: total carries = ν_p(C(n,k))
        let mut sign_carries = 0u32; // carries at positions ≥ e−1
        let mut acc = 1u32;
        let mut j = 0u32;
        while nn > 0 {
            let nj = nn % p;
            let kj = kk % p;
            let nw = nn % q; // N_j: the e-digit window of n starting at digit j
            let kw = kk % q;
            let rw = (nw + q - kw - carry) % q; // R_j = N_j − K_j − carry (mod q)
            acc = self.mul(acc, self.fp[nw as usize]);
            acc = self.mul(acc, self.ifp[kw as usize]);
            acc = self.mul(acc, self.ifp[rw as usize]);
            let c_out = (nj < kj + carry) as u32;
            nu += c_out;
            if j + 1 >= e {
                sign_carries += c_out;
            }
            if nu >= e {
                return 0;
            }
            carry = c_out;
            nn /= p;
            kk /= p;
            j += 1;
        }
        let mut r = acc;
        for _ in 0..nu {
            r = self.mul(r, p);
        }
        if self.minus_one && sign_carries & 1 == 1 {
            r = (q - r) % q;
        }
        r
    }
}

/// Kummer: number of carries adding `k` and `n − k` in base `p` = `ν_p(C(n,k))`.
pub fn carries(n: u32, k: u32, p: u32) -> u32 {
    if k > n {
        return 0;
    }
    let (mut nn, mut kk, mut carry, mut count) = (n, k, 0u32, 0u32);
    while nn > 0 {
        let c = (nn % p < kk % p + carry) as u32;
        count += c;
        carry = c;
        nn /= p;
        kk /= p;
    }
    count
}

/// Everything about one modulus: its factorization, the tables for each prime power and
/// the discrete-log table for the smallest prime (what DLOG colors by).
///
/// A decimal modulus μ = m/q (lowest terms, q a divisor of a power of 10) keeps the
/// whole m's tables: C mod μ = C − μ⌊C/μ⌋ = (q·C mod m)/q, so [`Modulus::scaled`] is
/// q·(C mod m) mod m, exact at any row, and the residue is that over q. q = 1 is the
/// whole modulus.
#[derive(Clone, Debug)]
pub struct Modulus {
    pub m: u32,
    /// The decimal modulus's denominator (μ = m / q); 1 for a whole modulus.
    pub q: u32,
    pub factors: Vec<PrimePowerTables>,
    /// Smallest prime factor.
    pub p: u32,
    /// Primitive root of `p` (1 for p = 2) and `dlog[x] = ind_g(x)` for `0 < x < p` (dlog[0] = 0).
    pub root: u32,
    pub dlog: Vec<u32>,
}

impl Modulus {
    pub fn new(m: u32) -> Self {
        assert!(m >= 2, "modulus must be at least 2");
        let fs = factorize(m);
        let mut factors = Vec::with_capacity(fs.len());
        let mut garner_q = 1u32;
        for f in &fs {
            factors.push(PrimePowerTables::new(*f, garner_q));
            garner_q = garner_q.wrapping_mul(f.q);
        }
        let p = fs[0].p;
        let root = primitive_root(p);
        let mut dlog = vec![0u32; p as usize];
        if p > 2 {
            let mut x = 1u64;
            for j in 0..(p - 1) {
                dlog[x as usize] = j;
                x = x * root as u64 % p as u64;
            }
        }
        Self { m, q: 1, factors, p, root, dlog }
    }

    /// The decimal modulus μ = m / q; reduced to lowest terms (m = 21, q = 2 is 10.5).
    pub fn decimal(m: u32, q: u32) -> Self {
        let g = gcd_u32(m, q.max(1));
        let (m, q) = (m / g, q.max(1) / g);
        Self { q, ..Modulus::new(m) }
    }

    /// The decimal modulus whole + sub·10⁻ᵈ (sub < 10ᵈ·gap), if its numerator fits
    /// `cap`.
    pub fn from_decimal(whole: u32, sub: u64, d: u32, cap: u32) -> Option<Self> {
        let q = 10u64.checked_pow(d)?;
        let units = (whole as u64).checked_mul(q)?.checked_add(sub)?;
        let g = gcd_u64(units, q);
        let (m, q) = (units / g, q / g);
        (m >= 2 && m <= cap as u64).then(|| Modulus::decimal(m as u32, q as u32))
    }

    /// μ itself.
    pub fn mu(&self) -> f64 {
        self.m as f64 / self.q as f64
    }

    /// A decimal modulus (q > 1).
    pub fn is_decimal(&self) -> bool {
        self.q > 1
    }

    /// q·(C(n,k) mod m) mod m: C mod μ times q, a whole number below m (C mod m for a
    /// whole modulus). It is 0 exactly when μ divides C (m and q are coprime).
    pub fn scaled(&self, n: u32, k: u32) -> u32 {
        let r = self.binom(n, k);
        if self.q == 1 {
            r
        } else {
            (r as u64 * (self.q % self.m) as u64 % self.m as u64) as u32
        }
    }

    /// C(n,k) mod μ, the real residue in [0, μ).
    pub fn residue(&self, n: u32, k: u32) -> f64 {
        self.scaled(n, k) as f64 / self.q as f64
    }

    /// μ as text: exact (q divides a power of 10, so the expansion ends).
    pub fn label(&self) -> String {
        decimal_text(self.m as u64, self.q as u64)
    }

    pub fn is_prime(&self) -> bool {
        self.factors.len() == 1 && self.factors[0].f.e == 1
    }

    /// `C(n,k) mod m`: per-factor residues recombined by Garner's algorithm.
    pub fn binom(&self, n: u32, k: u32) -> u32 {
        if k > n {
            return 0;
        }
        let mut x = 0u64; // the residue mod q_1 ⋯ q_f so far
        for t in &self.factors {
            let r = t.binom(n, k) as u64;
            let q = t.f.q as u64;
            if t.garner_q == 1 {
                x = r;
            } else {
                let diff = (r + q - x % q) % q;
                let c = diff * t.garner_inv as u64 % q;
                x += t.garner_q as u64 * c;
            }
        }
        x as u32
    }

    /// `ν_p(C(n,k))` for every prime factor, in factor order.
    pub fn valuations(&self, n: u32, k: u32) -> Vec<u32> {
        self.factors.iter().map(|t| carries(n, k, t.f.p)).collect()
    }
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn gcd_u32(a: u32, b: u32) -> u32 {
    gcd_u64(a as u64, b as u64) as u32
}

/// a / q in decimal, exactly, for q of the form 2^i·5^j (trailing zeros trimmed).
pub fn decimal_text(a: u64, q: u64) -> String {
    let q = q.max(1);
    let (ip, mut rem) = (a / q, a % q);
    if rem == 0 {
        return ip.to_string();
    }
    let mut s = format!("{ip}.");
    for _ in 0..40 {
        if rem == 0 {
            break;
        }
        rem *= 10;
        s.push(char::from(b'0' + (rem / q) as u8));
        rem %= q;
    }
    s
}

// ---------------------------------------------------------------------------------------
// Reference and display helpers.
// ---------------------------------------------------------------------------------------

/// The plain additive triangle, rows `0..rows`, flat index `n(n+1)/2 + k`. This is the
/// ground truth every GPU test compares against.
pub fn triangle_dp(rows: usize, m: u32) -> Vec<u32> {
    let m64 = m as u64;
    let mut t = vec![0u32; rows * (rows + 1) / 2];
    for r in 0..rows {
        let row = r * (r + 1) / 2;
        let prev = row - r;
        t[row] = 1 % m;
        t[row + r] = 1 % m;
        for c in 1..r {
            let s = t[prev + c - 1] as u64 + t[prev + c] as u64;
            t[row + c] = (s % m64) as u32;
        }
    }
    t
}

/// Exact `C(n,k)`, or `None` when the running product would overflow a u128 (never for
/// `n ≤ 125`; C(n,k) itself fits up to n = 130).
pub fn binom_exact(n: u32, k: u32) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut r: u128 = 1;
    for i in 1..=k as u128 {
        r = r.checked_mul(n as u128 - k as u128 + i)? / i;
    }
    Some(r)
}

/// `ln(n!)` by Stirling's series (exact enough for an order-of-magnitude readout).
pub fn ln_factorial(n: u32) -> f64 {
    if n < 2 {
        return 0.0;
    }
    if n < 64 {
        return (2..=n).map(|i| (i as f64).ln()).sum();
    }
    let x = n as f64 + 1.0;
    (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * x) - 1.0 / (360.0 * x * x * x)
}

/// `log10 C(n,k)`.
pub fn log10_binom(n: u32, k: u32) -> f64 {
    (ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)) / std::f64::consts::LN_10
}

/// Lucas expansion text for the inspector: `C(2,1)·C(0,0) ≡ 2 (mod 3)`.
pub fn lucas_string(n: u32, k: u32, p: u32, max_terms: usize) -> Option<String> {
    let mut terms = Vec::new();
    let (mut a, mut b) = (n, k);
    while a > 0 {
        terms.push(format!("C({},{})", a % p, b % p));
        a /= p;
        b /= p;
    }
    if terms.is_empty() || terms.len() > max_terms {
        return None;
    }
    terms.reverse();
    Some(terms.join("·"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorization_basics() {
        assert_eq!(factorize(360), vec![PrimePower { p: 2, e: 3, q: 8 }, PrimePower { p: 3, e: 2, q: 9 }, PrimePower { p: 5, e: 1, q: 5 }]);
        assert_eq!(factorize(65537), vec![PrimePower { p: 65537, e: 1, q: 65537 }]);
        assert_eq!(factorize(65536), vec![PrimePower { p: 2, e: 16, q: 65536 }]);
        assert_eq!(factor_string(&factorize(360)), "2³·3²·5");
    }

    #[test]
    fn montgomery_matches_u64() {
        for &q in &[3u32, 7, 65537, 99991, 999983, 3u32.pow(12), 5u32.pow(8), 1_000_001] {
            let qinv = mont_qinv(q);
            assert_eq!(q.wrapping_mul(qinv.wrapping_neg()), 1, "qinv for {q}");
            let one = to_mont(1, q);
            let mut x = 12345u64 % q as u64;
            for _ in 0..200 {
                let a = (x * 7 + 1) % q as u64;
                let b = (x * 13 + 5) % q as u64;
                let want = a * b % q as u64;
                let got = mont_mul(mont_mul(to_mont(a as u32, q), to_mont(b as u32, q), q, qinv), 1, q, qinv);
                assert_eq!(got as u64, want, "mont_mul {a}·{b} mod {q}");
                assert_eq!(mont_mul(one, to_mont(a as u32, q), q, qinv), to_mont(a as u32, q));
                x = (x * 31 + 17) % q as u64;
            }
        }
    }

    #[test]
    fn granville_matches_additive_rule() {
        let rows = 200usize;
        for &m in &[2u32, 3, 4, 5, 6, 7, 8, 9, 12, 16, 25, 27, 30, 32, 49, 60, 64, 81, 125, 128, 210, 256, 343, 360, 1024, 2310, 4096, 65535, 65536, 65537, 99991] {
            let dp = triangle_dp(rows, m);
            let md = Modulus::new(m);
            for n in 0..rows {
                for k in 0..=n {
                    let want = dp[n * (n + 1) / 2 + k];
                    let got = md.binom(n as u32, k as u32);
                    assert_eq!(got, want, "C({n},{k}) mod {m}");
                }
            }
        }
    }

    #[test]
    fn granville_deep_rows_against_lucas_for_primes() {
        // Beyond the reach of the additive rule: for primes, Lucas is independent ground truth.
        for &p in &[2u32, 3, 7, 101, 65537] {
            let md = Modulus::new(p);
            for &n in &[1_000_000u32, 123_456_789, 4_000_000_000] {
                for k in [0u32, 1, 2, 1000, 65536, 123_457, n / 3, n / 2, n - 1, n] {
                    // Lucas with the digit binomials C(a,b) mod p, a,b < p, taken from the
                    // factorial tables (which the additive-rule test above has already verified).
                    let t = &md.factors[0];
                    let (mut a, mut b, mut prod) = (n, k, 1u64);
                    while a > 0 {
                        let (da, db) = (a % p, b % p);
                        let cab = if db > da { 0 } else { t.fp[da as usize] as u64 * t.ifp[db as usize] as u64 % p as u64 * t.ifp[(da - db) as usize] as u64 % p as u64 };
                        prod = prod * cab % p as u64;
                        a /= p;
                        b /= p;
                    }
                    assert_eq!(md.binom(n, k) as u64, prod, "C({n},{k}) mod {p}");
                }
            }
        }
    }

    #[test]
    fn carries_is_valuation() {
        let rows = 120usize;
        for &p in &[2u32, 3, 5, 7] {
            for n in 0..rows as u32 {
                for k in 0..=n {
                    let b = binom_exact(n, k).unwrap();
                    let mut v = 0;
                    let mut x = b;
                    while x > 0 && x.is_multiple_of(p as u128) {
                        x /= p as u128;
                        v += 1;
                    }
                    assert_eq!(carries(n, k, p), v, "nu_{p} C({n},{k})");
                }
            }
        }
    }

    #[test]
    fn dlog_table_is_consistent() {
        let md = Modulus::new(7);
        assert_eq!(md.root, 3);
        for x in 1..7u32 {
            assert_eq!(modpow(md.root as u64, md.dlog[x as usize] as u64, 7), x as u64);
        }
    }

    #[test]
    fn decimal_moduli_are_exact() {
        // μ = 10.5 = 21/2 in lowest terms; C mod μ = C − μ⌊C/μ⌋ = (2C mod 21)/2
        let md = Modulus::from_decimal(10, 5, 1, MOD_CAP).unwrap();
        assert_eq!((md.m, md.q, md.label().as_str()), (21, 2, "10.5"));
        assert_eq!(Modulus::from_decimal(10, 50, 2, MOD_CAP).unwrap().m, 21, "10.50 is 10.5");
        assert_eq!(Modulus::from_decimal(7, 0, 3, MOD_CAP).unwrap().q, 1, "7.000 is whole");
        assert!(Modulus::from_decimal(5000, 1, 3, MOD_CAP).is_none(), "5000.001 needs 5000001 > the cap");
        for (whole, sub, d) in [(10u32, 5u64, 1u32), (3, 7, 1), (12, 37, 2), (2, 1, 3), (99, 99, 2)] {
            let md = Modulus::from_decimal(whole, sub, d, MOD_CAP).unwrap();
            let (p, q) = (md.m as u128, md.q as u128);
            for n in 0..=60u32 {
                for k in 0..=n {
                    let c = binom_exact(n, k).unwrap();
                    // exact: C − (p/q)·⌊C·q/p⌋, times q
                    let want = c * q - p * (c * q / p);
                    assert_eq!(md.scaled(n, k) as u128, want, "C({n},{k}) mod {}", md.label());
                    let r = md.residue(n, k);
                    assert!((0.0..md.mu()).contains(&r));
                }
            }
        }
        assert_eq!(decimal_text(1037, 100), "10.37");
        assert_eq!(decimal_text(21, 2), "10.5");
        assert_eq!(decimal_text(8, 1), "8");
    }

}
