// ============================================================================
// binom.wgsl — C(n,k) mod p^e with no bindings of its own: Montgomery
// multiplication, Granville's theorem (Lucas when e = 1) and Kummer's carries.
//
// Every module that includes it declares `tables: array<u32>` (the (x!)_p tables
// and p^t powers that `Factor` offsets point into). common.wgsl does, for the 2D
// passes; tetra_cells.wgsl does for the 3D stack, where each slice has its own
// modulus and factors.
// ============================================================================

struct Factor {
  p: u32,          // prime
  e: u32,          // exponent
  q: u32,          // p^e
  qinv: u32,       // Montgomery -q^{-1} mod 2^32 (odd q only)
  one: u32,        // 1 in Montgomery form (R mod q); plain 1 when p = 2
  p_mont: u32,     // p in Montgomery form; plain p when p = 2
  tab_off: u32,    // tables[tab_off .. +q] = (x!)_p, then [+q .. +2q] = its inverse
  minus_one: u32,  // 1 when (p^e !)_p ≡ -1 (every case except p = 2, e ≥ 3)
  garner_q: u32,   // product of the previous factors' q (1 for the first factor)
  garner_inv: u32, // garner_q^{-1} mod q, Montgomery form (plain when p = 2)
  ppow_off: u32,   // tables[ppow_off + t] = p^t mod q for t < e (Montgomery form; plain when p = 2)
  pad1: u32,
};

const MAX_DIGITS: u32 = 32u;     // base-p digits of a u32 (p ≥ 2)

// ---------------------------------------------------------------------------
// 32-bit modular multiplication.
// ---------------------------------------------------------------------------

// Full 64-bit product of two u32 as (lo, hi), from four 16-bit partial products.
fn mul64(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xffffu;
  let a1 = a >> 16u;
  let b0 = b & 0xffffu;
  let b1 = b >> 16u;
  let p00 = a0 * b0;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let p11 = a1 * b1;
  let mid = p01 + (p00 >> 16u);            // < 2^32: (2^16-1)^2 + 2^16 - 1
  let mid2 = (mid & 0xffffu) + p10;        // < 2^32 likewise
  let lo = (mid2 << 16u) | (p00 & 0xffffu);
  let hi = p11 + (mid >> 16u) + (mid2 >> 16u);
  return vec2<u32>(lo, hi);
}

// Montgomery REDC with R = 2^32: a·b·R^{-1} mod q for odd q < 2^31.
// With both inputs in Montgomery form (x·R) the result is in Montgomery form too.
fn mont_mul(a: u32, b: u32, q: u32, qinv: u32) -> u32 {
  let t = mul64(a, b);
  let mq = mul64(t.x * qinv, q);           // t.x * qinv wraps mod 2^32 on purpose
  let lo = t.x + mq.x;                     // ≡ 0 mod 2^32 by construction
  let carry = select(0u, 1u, lo < t.x);
  var u = t.y + mq.y + carry;              // < 2q
  if (u >= q) { u = u - q; }
  return u;
}

// a·b in the arithmetic of factor F: Montgomery for odd q, wrap-and-mask for 2^e.
fn mm(a: u32, b: u32, F: Factor) -> u32 {
  if (F.p == 2u) { return (a * b) & (F.q - 1u); }
  return mont_mul(a, b, F.q, F.qinv);
}

// Montgomery form → plain integer.
fn from_mont(x: u32, F: Factor) -> u32 {
  if (F.p == 2u) { return x; }
  return mont_mul(x, 1u, F.q, F.qinv);
}

// ---------------------------------------------------------------------------
// C(n,k) mod p^e by Granville's theorem; Lucas's theorem is the e = 1 case.
//
// Walk the base-p digits of n and k (the digits of r = n − k and the carries of
// k + r come from the subtraction n − k with borrow, so only two numbers are
// divided). At digit j the e-digit windows N_j, K_j, R_j index the tables of
// (x!)_p and its inverse:
//
//    C(n,k) / p^ν ≡ (±1)^{carries at positions ≥ e−1} · ∏_j (N_j!)_p / ((K_j!)_p (R_j!)_p)
//
// ν = total carries = ν_p(C(n,k)) (Kummer). Once ν reaches e the residue is 0;
// the loop then only keeps counting carries if the caller wants the full ν.
// ---------------------------------------------------------------------------
struct BinomRes { r: u32, nu: u32 };

fn binom_mod_prime_power(n: u32, k: u32, F: Factor, full_nu: bool) -> BinomRes {
  let p = F.p;
  let e = F.e;
  let q = F.q;
  let is2 = (p == 2u);
  let prime = (e == 1u);
  let qm = q - 1u;
  var nn = n;
  var kk = k;
  var carry = 0u;      // carry into digit j (= borrow of n − k)
  var nu = 0u;         // carries so far
  var sc = 0u;         // carries at digit positions ≥ e − 1 (the sign)
  var acc = F.one;     // running product, Montgomery form
  var zero = false;
  var done = false;
  var j = 0u;
  for (var it = 0u; it < MAX_DIGITS; it = it + 1u) {
    if (nn == 0u || done) { break; }
    // One division per number per digit; the digit is the remainder. For p = 2
    // everything is shifts and masks. The e-digit windows N_j, K_j are the digits
    // themselves when e = 1 (Lucas), otherwise one more remainder each.
    var nq = 0u;
    var kq = 0u;
    var nj = 0u;
    var kj = 0u;
    var nw = 0u;
    var kw = 0u;
    if (is2) {
      nq = nn >> 1u; kq = kk >> 1u; nj = nn & 1u; kj = kk & 1u; nw = nn & qm; kw = kk & qm;
    } else {
      nq = nn / p; kq = kk / p; nj = nn - nq * p; kj = kk - kq * p;
      if (prime) { nw = nj; kw = kj; } else { nw = nn % q; kw = kk % q; }
    }
    let c_out = select(0u, 1u, nj < kj + carry);
    if (!zero) {
      let t = nw + q - kw - carry;            // in [1, 2q)
      let rw = select(t, t - q, t >= q);      // R_j = N_j − K_j − carry (mod q)
      acc = mm(acc, tables[F.tab_off + nw], F);
      acc = mm(acc, tables[F.tab_off + q + kw], F);
      acc = mm(acc, tables[F.tab_off + q + rw], F);
      if (j + 1u >= e) { sc = sc + c_out; }
    }
    nu = nu + c_out;
    if (nu >= e) {
      zero = true;
      // The residue is 0; keep walking only when the caller wants the full carry count.
      done = !full_nu;
    }
    carry = c_out;
    nn = nq;
    kk = kq;
    j = j + 1u;
  }
  var r = 0u;
  if (!zero) {
    // ν < e here, and p^ν mod q comes from a CPU-built table: no power loop.
    r = mm(acc, tables[F.ppow_off + nu], F);
    r = from_mont(r, F);
    if (F.minus_one == 1u && (sc & 1u) == 1u) { r = select(q - r, 0u, r == 0u); }
  }
  return BinomRes(r, nu);
}
