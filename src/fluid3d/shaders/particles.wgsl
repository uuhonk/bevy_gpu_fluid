struct Particle { position: vec4<f32>, velocity: vec4<f32> }
struct Params {
    color: vec4<f32>,
    visual: vec4<f32>,
    optics: vec4<f32>,
    size: vec4<f32>,
}
struct View {
    clip_from_world: mat4x4<f32>,
    world_from_view: mat4x4<f32>,
    view_from_world: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
    viewport: vec4<f32>,
}
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;
@group(1) @binding(0) var<uniform> view: View;
@group(2) @binding(0) var front_depth: texture_2d<f32>;

@group(0) @binding(2) var<storage, read> activity: array<u32>;

struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) center: vec3<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
    if activity[i] == 0u { return Out(vec4<f32>(2.0,2.0,0.0,1.0),vec2<f32>(0.0), vec3<f32>(0.0)); }
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1, -1), vec2<f32>(1, -1), vec2<f32>(1, 1),
        vec2<f32>(-1, -1), vec2<f32>(1, 1), vec2<f32>(-1, 1),
    );
    let p = particles[i];
    let uv = corners[v];
    let offset = normalize(view.world_from_view[0].xyz) * uv.x
        + normalize(view.world_from_view[1].xyz) * uv.y;
    let world = p.position.xyz + offset * params.visual.x * params.visual.w;
    return Out(view.clip_from_world * vec4<f32>(world, 1), uv, p.position.xyz);
}

struct DepthOut {
    @location(0) linear_depth: f32,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn depth(in: Out) -> DepthOut {
    let q = 1.0 - dot(in.uv, in.uv);
    // Invisible kernel outskirts must not hide a valid surface in a deeper layer.
    if q * q * q < params.visual.z { discard; }
    // Match the depth profile to the same isovalue/radius as the density contour.
    // A fixed small hemisphere can let scene geometry cut a rim into a wider kernel.
    let support = params.visual.x * params.visual.w;
    let height = support * sqrt(max(q - pow(params.visual.z, 1.0 / 3.0), 0.0));
    let world = in.center + normalize(view.world_from_view[2].xyz) * height;
    let clip = view.clip_from_world * vec4<f32>(world, 1);
    if clip.w <= 0.0 || clip.z < 0.0 || clip.z > clip.w { discard; }
    return DepthOut(-(view.view_from_world * vec4<f32>(world, 1)).z, clip.z / clip.w);
}

@fragment
fn density(in: Out) -> @location(0) vec4<f32> {
    let q = 1.0 - dot(in.uv, in.uv);
    if q <= 0.0 { discard; }
    let front = textureLoad(front_depth, vec2<i32>(in.position.xy), 0).x;
    let center = -(view.view_from_world * vec4<f32>(in.center, 1)).z;
    // Far-away projected layers must not inflate the front surface's density.
    let depth_range = params.visual.x * max(3.0, params.visual.w);
    if front <= 0.0 || abs(center - front) > depth_range { discard; }
    let value = q * q * q;
    return vec4<f32>(value, 0.0, 0.0, 0.0);
}
