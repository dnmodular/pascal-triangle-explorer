// ============================================================================
// value.wgsl — the VALUE pass: one thread per grid texel computes the residue
// C(n,k) mod m of the cell it samples, with no dependency on any other cell (for a
// decimal modulus μ = m/q, q times C mod μ).
//
// residue_info (common.wgsl) gives C(n,k) mod m from the per-factor residues
// recombined by Garner's CRT, the carry count ν_p and the divisibility mask.
// ============================================================================

@compute @workgroup_size(16, 16)
fn value_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.grid_w || gid.y >= P.grid_h) { return; }
  let idx = gid.y * P.grid_w + gid.x;
  let c = cell_of(gid.x, gid.y);
  if (c.valid == 0u) {
    cells[idx] = vec2<u32>(0u, 0u);
    return;
  }
  let r = residue_info(c.n, c.k, true);
  // a decimal modulus μ = m/q: C mod μ = (q·C mod m)/q, so the cell holds q·r mod m
  // (zero exactly when r is: q and m are coprime) and the colors read it over q
  var x = r.x;
  if (P.qscale > 1u) { x = mulmod20(r.x, P.qscale, P.m); }
  cells[idx] = vec2<u32>(x, CELL_VALID | (r.mask << 8u) | min(r.nu0, 255u));
  // a morph: the second modulus's residue, in the buffer's second half
  if (P.m_b != 0u) {
    cells[P.grid_w * P.grid_h + idx] = vec2<u32>(residue_b(c.n, c.k), 0u);
  }
}

// a·b mod m for a, b < m ≤ 2^20 in 32-bit steps: b split into 10-bit halves, every
// product below 2^30
fn mulmod20(a: u32, b: u32, m: u32) -> u32 {
  let hi = (a * (b >> 10u)) % m;
  let lo = (a * (b & 1023u)) % m;
  return ((hi << 10u) % m + lo) % m;
}
