// ============================================================================
// value.wgsl — the VALUE pass: one thread per grid texel computes the residue
// C(n,k) mod m of the cell it samples, with no dependency on any other cell.
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
  cells[idx] = vec2<u32>(r.x, CELL_VALID | (r.mask << 8u) | min(r.nu0, 255u));
}
