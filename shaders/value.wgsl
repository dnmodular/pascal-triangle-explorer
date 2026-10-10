// ============================================================================
// value.wgsl — the VALUE pass: one thread per grid texel computes the residue
// C(n,k) mod m of the cell it samples, with no dependency on any other cell.
//
// Per prime-power factor q = p^e of m, binom_mod_prime_power gives C(n,k) mod q
// and the carry count ν_p. The residues are recombined by Garner's form of the
// Chinese remainder theorem:
//
//   x_1 = r_1
//   x_f = x_{f−1} + Q_{f−1} · ((r_f − x_{f−1}) · Q_{f−1}^{−1} mod q_f),   Q_{f−1} = q_1⋯q_{f−1}
//
// so every intermediate stays below m and fits in a u32.
// ============================================================================

@compute @workgroup_size(16, 16)
fn value_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cell_of(gid.x, gid.y);
  if (!c.valid) {
    cells[idx] = vec2<u32>(0u, 0u);
    return;
  }
  var x = 0u;      // residue mod q_1 ⋯ q_f so far
  var nu0 = 0u;    // ν_p for the smallest prime (what KUMMER shows)
  var mask = 0u;   // bit f set when q_f divides C(n,k) (what CRT shows)
  for (var f = 0u; f < P.nfactors; f = f + 1u) {
    let F = P.factors[f];
    let res = binom_mod_prime_power(c.n, c.k, F, f == 0u);
    if (f == 0u) {
      nu0 = res.nu;
      x = res.r;
    } else {
      let xq = x % F.q;
      let diff = (res.r + F.q - xq) % F.q;
      let cc = mm(diff, F.garner_inv, F);   // plain (r_f − x) · Q^{−1} mod q_f
      x = x + F.garner_q * cc;
    }
    if (res.nu >= F.e) { mask = mask | (1u << f); }
  }
  cells[idx] = vec2<u32>(x, CELL_VALID | (mask << 8u) | min(nu0, 255u));
}
