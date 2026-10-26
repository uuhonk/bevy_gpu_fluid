struct Params {
    color: vec4<f32>,
    // Radius, reserved, isovalue, kernel support multiplier.
    visual: vec4<f32>,
    // Reserved, reserved, filter radius, resolution scale.
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
@group(0) @binding(0) var surface_depth: texture_2d<f32>;
@group(0) @binding(1) var density: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<uniform> view: View;

@vertex
fn fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1, -1), vec2<f32>(3, -1), vec2<f32>(-1, 3),
    );
    return vec4<f32>(positions[i], 0, 1);
}

fn coord(p: vec2<i32>) -> vec2<i32> {
    return clamp(p, vec2<i32>(0), vec2<i32>(textureDimensions(surface_depth)) - 1);
}

fn field(p: vec2<i32>) -> vec4<f32> {
    // Half-float additive accumulation can overflow for extreme overdraw.
    return min(textureLoad(density, coord(p), 0), vec4<f32>(60000.0));
}

fn bilateral(p: vec2<i32>, axis: vec2<i32>) -> f32 {
    var center = textureLoad(surface_depth, coord(p), 0).x;
    let hole = center <= 0.0;
    let radius = i32(round(params.optics.z));
    // Extend depth into density bridges. Their contour still comes from the scalar field.
    // Ignore invisible kernel outskirts in the original splat, then borrow only nearby
    // valid core depths instead of letting a low-density foreground splat punch a hole.
    let search_radius = select(radius, 8, hole);
    if hole {
        var closest = 9;
        for (var k = -8; k <= 8; k++) {
            let sample = textureLoad(surface_depth, coord(p + axis * k), 0).x;
            if sample > 0.0 && (abs(k) < closest || (abs(k) == closest && sample < center)) {
                center = sample;
                closest = abs(k);
            }
        }
        if center <= 0.0 { return 0.0; }
    }
    let sigma = max(f32(search_radius) * 0.5, 0.5);
    let range = max(params.visual.x * 0.8, 0.0001);
    var total = 0.0;
    var weight = 0.0;
    for (var k = -8; k <= 8; k++) {
        if abs(k) > search_radius { continue; }
        let sample = textureLoad(surface_depth, coord(p + axis * k), 0).x;
        if sample <= 0.0 { continue; }
        let dz = (sample - center) / range;
        let w = exp(-0.5 * (f32(k * k) / (sigma * sigma) + dz * dz));
        total += sample * w;
        weight += w;
    }
    return total / max(weight, 0.00001);
}

@fragment
fn horizontal(@builtin(position) p: vec4<f32>) -> @location(0) f32 {
    return bilateral(vec2<i32>(p.xy), vec2<i32>(1, 0));
}

@fragment
fn vertical(@builtin(position) p: vec4<f32>) -> @location(0) f32 {
    return bilateral(vec2<i32>(p.xy), vec2<i32>(0, 1));
}

// Interpolate the scalar field before thresholding, including when upscaling.
// R32Float is unfilterable on baseline devices, so interpolate depth manually.
fn sample_density(p: vec2<f32>) -> f32 {
    let base = vec2<i32>(floor(p));
    let t = fract(p);
    let a = mix(field(base).x, field(base + vec2<i32>(1, 0)).x, t.x);
    let b = mix(field(base + vec2<i32>(0, 1)).x, field(base + vec2<i32>(1, 1)).x, t.x);
    return mix(a, b, t.y);
}

// Integrate the isocontour over the output pixel instead of using fwidth(density).
// Density can jump between depth layers *inside* the water. Derivative-based AA
// treated those internal changes as silhouettes and drew dark circular seams.
fn surface_coverage(p: vec2<f32>, step: vec2<f32>) -> f32 {
    let lo = vec2<i32>(floor(p - step * 0.5));
    let hi = vec2<i32>(ceil(p + step * 0.5));
    var low = 60000.0;
    var high = 0.0;
    for (var y = 0; y < 3; y++) {
        for (var x = 0; x < 3; x++) {
            let sample = field(min(lo + vec2<i32>(x, y), hi)).x;
            low = min(low, sample);
            high = max(high, sample);
        }
    }
    if low >= params.visual.z { return 1.0; }
    if high < params.visual.z { return 0.0; }
    // Only contour pixels need supersampling; uniform interior/background exits above.
    var coverage = 0.0;
    for (var y = 0; y < 4; y++) {
        for (var x = 0; x < 4; x++) {
            let offset = (vec2<f32>(f32(x), f32(y)) + 0.5) * 0.25 - 0.5;
            coverage += select(0.0, 0.0625, sample_density(p + offset * step) >= params.visual.z);
        }
    }
    return coverage;
}

fn sample_depth(p: vec2<f32>) -> f32 {
    let base = vec2<i32>(floor(p));
    let t = fract(p);
    let depths = vec4<f32>(
        textureLoad(surface_depth, coord(base), 0).x,
        textureLoad(surface_depth, coord(base + vec2<i32>(1, 0)), 0).x,
        textureLoad(surface_depth, coord(base + vec2<i32>(0, 1)), 0).x,
        textureLoad(surface_depth, coord(base + vec2<i32>(1, 1)), 0).x,
    );
    var weights = vec4<f32>((1.0 - t.x) * (1.0 - t.y), t.x * (1.0 - t.y),
        (1.0 - t.x) * t.y, t.x * t.y);
    // Empty pixels must not pull the silhouette's depth toward zero/the camera.
    weights *= select(vec4<f32>(0.0), vec4<f32>(1.0), depths > vec4<f32>(0.0));
    return dot(depths, weights) / max(dot(weights, vec4<f32>(1.0)), 0.00001);
}

fn view_position(p: vec2<f32>, z: f32) -> vec3<f32> {
    let uv = (vec2<f32>(p) + 0.5) / params.size.xy;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // Two finite points on the view ray support perspective and orthographic cameras.
    let a = view.view_from_clip * vec4<f32>(ndc, 1.0, 1.0);
    let b = view.view_from_clip * vec4<f32>(ndc, 0.001, 1.0);
    let start = a.xyz / a.w;
    let direction = b.xyz / b.w - start;
    return start + direction * ((-z - start.z) / direction.z);
}

struct Fragment {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn composite(@builtin(position) position: vec4<f32>) -> Fragment {
    let uv = (position.xy - view.viewport.xy) / view.viewport.zw;
    let p = uv * params.size.xy - 0.5;
    let coverage = surface_coverage(p, params.size.xy / view.viewport.zw);
    let alpha = coverage * params.color.a;
    let z = sample_depth(p);
    if alpha <= 0.0001 || z <= 0.0 { discard; }
    let center = view_position(p, z);
    let world = view.world_from_view * vec4<f32>(center, 1);
    let clip = view.clip_from_world * world;
    if clip.w <= 0.0 || clip.z < 0.0 || clip.z > clip.w { discard; }
    // Premultiplication preserves the same water hue at partial edge coverage.
    return Fragment(vec4<f32>(params.color.rgb * alpha, alpha), clip.z / clip.w);
}
