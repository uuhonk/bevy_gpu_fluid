struct Particle { position: vec4<f32>, velocity: vec4<f32> }
struct View { clip_from_world: mat4x4<f32>, world_from_view: mat4x4<f32> }
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var<storage, read> activity: array<u32>;

struct Out { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) speed: f32 }
@vertex
fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
    if activity[i] == 0u { return Out(vec4<f32>(2.0,2.0,0.0,1.0),vec2<f32>(0.0), 0.0); }
    let corners = array<vec2<f32>, 6>(vec2<f32>(-1,-1),vec2<f32>(1,-1),vec2<f32>(1,1),vec2<f32>(-1,-1),vec2<f32>(1,1),vec2<f32>(-1,1));
    let p = particles[i];
    let uv = corners[v];
    let speed = length(p.velocity.xyz);
    let offset = (view.world_from_view[0].xyz * uv.x + view.world_from_view[1].xyz * uv.y * (1.0 + min(speed * 0.1, 2.0))) * 0.09;
    return Out(view.clip_from_world * vec4<f32>(p.position.xyz + offset, 1.0), uv, speed);
}
@fragment
fn fragment(in: Out) -> @location(0) vec4<f32> {
    if dot(in.uv, in.uv) > 1.0 { discard; }
    return vec4<f32>(mix(vec3<f32>(0.8, 0.03, 0.1), vec3<f32>(1.0, 0.8, 0.1), clamp(in.speed * 0.1, 0.0, 1.0)), 1.0);
}
