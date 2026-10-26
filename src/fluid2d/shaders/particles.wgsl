struct Particle { position: vec2<f32>, velocity: vec2<f32> }
struct Params {
    bounds: vec4<f32>,
    color: vec4<f32>,
    visual: vec4<f32>,
}
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;

@group(0) @binding(2) var<storage, read> activity: array<u32>;

struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
    if activity[i] == 0u { return Out(vec4<f32>(2.0,2.0,0.0,1.0),vec2<f32>(0.0)); }
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1, -1), vec2<f32>(1, -1), vec2<f32>(1, 1),
        vec2<f32>(-1, -1), vec2<f32>(1, 1), vec2<f32>(-1, 1),
    );
    let p = particles[i];
    let local = corners[v];
    let world = p.position + local * params.visual.x * params.visual.w;
    let clip = (world - params.bounds.xy) / (params.bounds.zw - params.bounds.xy) * 2.0 - 1.0;
    return Out(vec4<f32>(clip, 0, 1), local);
}

@fragment
fn fragment(in: Out) -> @location(0) vec4<f32> {
    let q = max(1.0 - dot(in.local, in.local), 0.0);
    let density = q * q * q;
    return vec4<f32>(density, 0.0, 0.0, 0.0);
}
