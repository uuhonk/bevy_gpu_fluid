struct Params {
    bounds: vec4<f32>,
    color: vec4<f32>,
    // Radius, reserved, isovalue, kernel support multiplier.
    visual: vec4<f32>,
}
@group(0) @binding(0) var field: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@vertex
fn fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1, -1), vec2<f32>(3, -1), vec2<f32>(-1, 3),
    );
    return vec4<f32>(positions[i], 0, 1);
}

fn sample_field(p: vec2<i32>) -> vec4<f32> {
    let coord = clamp(p, vec2<i32>(0), vec2<i32>(textureDimensions(field)) - 1);
    return min(textureLoad(field, coord, 0), vec4<f32>(60000.0));
}

@fragment
fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(position.xy);
    let density = sample_field(p).x;
    let aa = max(fwidth(density), 0.002);
    let alpha = smoothstep(params.visual.z - aa, params.visual.z + aa, density) * params.color.a;
    if density < params.visual.z - aa { return vec4<f32>(0.0); }
    return vec4<f32>(params.color.rgb, alpha);
}
