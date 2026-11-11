// ============================================================================
// tween.wgsl — display-only cross-dissolve between sweep frames.
//
// The resolve pass writes the true frame into `frame_tex`. This pass writes what
// the window shows: mix(snapshot of what was on screen, true frame, e). Exports
// and readbacks use `frame_tex`, so a tween can never leak into them. When no
// dissolve is running, e = 1 and the true frame is copied through unchanged.
// (Field morphs happen earlier, in the resolve pass; see field.wgsl.)
// ============================================================================

@group(0) @binding(0) var frame_tex: texture_2d<f32>;
@group(0) @binding(1) var snap_tex: texture_2d<f32>;
@group(0) @binding(2) var display_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> T: vec4<f32>; // x = eased progress e in [0, 1]

@compute @workgroup_size(16, 16)
fn tween_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(frame_tex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  let cur = textureLoad(frame_tex, p, 0);
  var col = cur.rgb;
  if (T.x < 1.0) {
    let old = textureLoad(snap_tex, p, 0);
    col = mix(old.rgb, cur.rgb, T.x);
  }
  textureStore(display_tex, p, vec4<f32>(col, 1.0));
}
