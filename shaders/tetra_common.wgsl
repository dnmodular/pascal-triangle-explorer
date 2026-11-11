// ============================================================================
// tetra_common.wgsl — what the tetra stack's two modules share: a slice as the
// CPU uploads it (tetra_gpu::GpuSlice, 64 bytes), one per modulus (or per line of
// a face).
// ============================================================================

struct Slice {
  m: u32,
  rows: u32,          // strided rows
  cell_base: u32,     // first cell of the slice in the enumeration
  nfactors: u32,
  factor_off: u32,    // factors[factor_off ..] (garner constants per slice)
  level_off: u32,     // residue → palette level bytes at levels[level_off + r] (draw pass)
  kind: u32,          // 0 rows row_n, row_n + s, … (a triangle); else a line, 1 | dn << 1 | dk << 2
  row_n: u32,
  ox: f32,            // cell (n, k) sits at o + n·nv + (k, 0, 0) (draw pass)
  oy: f32,
  oz: f32,
  k0: u32,            // a line's first k
  nx: f32,
  ny: f32,
  nz: f32,
  peel_id: u32,       // the row the front side's peel counts from | layer << 16
};
