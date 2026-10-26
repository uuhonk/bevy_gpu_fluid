struct Particle { position: vec2<f32>, velocity: vec2<f32> }
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> bounds: vec4<f32>;
@group(0) @binding(2) var<storage, read> activity: array<u32>;

struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) speed: f32,
}
@vertex
fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
    if activity[i] == 0u { return Out(vec4<f32>(2.0,2.0,0.0,1.0),vec2<f32>(0.0), 0.0); }
    let corners = array<vec2<f32>, 6>(vec2<f32>(-1, -1), vec2<f32>(1, -1), vec2<f32>(1, 1),
        vec2<f32>(-1, -1), vec2<f32>(1, 1), vec2<f32>(-1, 1));
    let p = particles[i];
    let speed = length(p.velocity);
    // Application-specific stretch along the velocity, and a warm speed palette.
    let direction = normalize(p.velocity + vec2<f32>(0.0001, 0.0001));
    let tangent = vec2<f32>(-direction.y, direction.x);
    let local = corners[v];
    let offset = (direction * local.x * (1.0 + min(speed * 0.12, 2.0)) + tangent * local.y) * 0.075;
    let clip = (p.position + offset - bounds.xy) / (bounds.zw - bounds.xy) * 2.0 - 1.0;
    return Out(vec4<f32>(clip, 0, 1), local, speed);
}
@fragment
fn fragment(in: Out) -> @location(0) vec4<f32> {
    if dot(in.uv, in.uv) > 1.0 { discard; }
    return vec4<f32>(mix(vec3<f32>(0.5, 0.025, 0.12), vec3<f32>(1.0, 0.85, 0.1), clamp(in.speed * 0.1, 0.0, 1.0)), 1.0);
}
