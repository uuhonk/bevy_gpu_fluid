// Independent 3D GPU SPH. Particle stride 32, density stride 8, acceleration stride 16.
struct Particle { position: vec4<f32>, velocity: vec4<f32> }
struct Collider { shape: vec4<f32>, min: vec4<f32>, max: vec4<f32>, rotation: vec4<f32> }
struct Params {
    bounds_min: vec4<f32>, bounds_max: vec4<f32>, kernel: vec4<f32>, forces: vec4<f32>,
    integration: vec4<f32>, counts: vec4<u32>, outlet: vec4<f32>, outlet_extra: vec4<f32>,
    velocity: vec4<f32>, environment: vec4<u32>, colliders: array<Collider, 16>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> heads: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> next: array<u32>;
@group(0) @binding(4) var<storage, read_write> densities: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> accelerations: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> activity: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> live_count: atomic<u32>;
const END: u32 = 0xffffffffu;
const PI: f32 = 3.141592653589793;
fn cell(p: vec3<f32>) -> vec3<i32> {
    return clamp(vec3<i32>(floor((p - params.bounds_min.xyz) / params.kernel.x)),
        vec3<i32>(0), vec3<i32>(params.counts.yzw) - vec3<i32>(1));
}
fn cell_index(c: vec3<i32>) -> u32 { return (u32(c.z) * params.counts.z + u32(c.y)) * params.counts.y + u32(c.x); }
fn valid_cell(c: vec3<i32>) -> bool { return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(params.counts.yzw)); }
// 315/(64 pi h^9) * (h^2-r^2)^3, normalized in three dimensions.
fn poly6(r2: f32) -> f32 {
    let h = params.kernel.x;
    let q = max(1.0 - r2 / (h * h), 0.0);
    return 315.0 / (64.0 * PI * h * h * h) * q * q * q;
}
fn limited(v: vec3<f32>, maximum: f32) -> vec3<f32> { return v * min(1.0, maximum / max(length(v), 1e-12)); }
@compute @workgroup_size(64)
fn emit_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = params.environment.x + id.x;
    if i >= params.counts.x { return; }
    let nx = max(params.environment.y, 1u);
    let nz = max(params.environment.z, 1u);
    let x = ((f32(id.x % nx) + 0.5) / f32(nx) - 0.5) * params.outlet.w;
    let z = ((f32((id.x / nx) % nz) + 0.5) / f32(nz) - 0.5) * params.outlet_extra.x;
    particles[i] = Particle(vec4<f32>(params.outlet.xyz + vec3<f32>(x, 0.0, z), 0.0), params.velocity);
    atomicStore(&activity[i],1u);
    atomicAdd(&live_count,1u);
}
@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < params.counts.y * params.counts.z * params.counts.w { atomicStore(&heads[id.x], END); }
}

// Contact time is a fraction of displacement; values greater than one miss.
struct Hit { time: f32, normal: vec3<f32> }
fn miss() -> Hit { return Hit(2.0, vec3<f32>(0.0)); }
fn earlier(a: Hit, b: Hit) -> Hit { if a.time < b.time { return a; } return b; }
fn reflected(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - (1.0 + params.outlet_extra.y) * min(dot(v, n), 0.0) * n;
}
fn sweep_sphere(p: vec3<f32>, d: vec3<f32>, center: vec3<f32>, radius: f32) -> Hit {
    let q = p - center;
    let a = dot(d, d);
    let b = dot(q, d);
    let c = dot(q, q) - radius * radius;
    if a < 1e-16 || b >= 0.0 { return miss(); }
    let discriminant = b * b - a * c;
    if discriminant < 0.0 { return miss(); }
    let t = max(0.0, (-b - sqrt(discriminant)) / a);
    if t > 1.0 { return miss(); }
    return Hit(t, normalize(q + d * t));
}
// Body of an axis-aligned edge capsule. Corner spheres handle its two caps.
fn sweep_edge(p: vec3<f32>, d: vec3<f32>, start: vec3<f32>, finish: f32, axis: u32, r: f32) -> Hit {
    var q = p - start;
    var radial_d = d;
    q[axis] = 0.0;
    radial_d[axis] = 0.0;
    let a = dot(radial_d, radial_d);
    let b = dot(q, radial_d);
    let c = dot(q, q) - r * r;
    if a < 1e-16 || b >= 0.0 { return miss(); }
    let discriminant = b * b - a * c;
    if discriminant < 0.0 { return miss(); }
    let t = (-b - sqrt(discriminant)) / a;
    if t < 0.0 || t > 1.0 { return miss(); }
    let axial = p[axis] + d[axis] * t;
    if axial < start[axis] || axial > finish { return miss(); }
    return Hit(t, normalize(q + radial_d * t));
}
// Exact sphere/cuboid Minkowski boundary: 6 faces, 12 edge cylinders, 8 spheres.
fn sweep_axis_collider(p: vec3<f32>, d: vec3<f32>, c: Collider, r: f32) -> Hit {
    if c.shape.x == 2.0 { return sweep_sphere(p, d, c.min.xyz, c.shape.y + r); }
    let lo = c.min.xyz;
    let hi = c.max.xyz;
    var hit = miss();
    for (var axis = 0u; axis < 3u; axis++) {
        let a = (axis + 1u) % 3u;
        let b = (axis + 2u) % 3u;
        if abs(d[axis]) > 1e-12 {
            for (var side = 0u; side < 2u; side++) {
                let sign_n = select(-1.0, 1.0, side == 1u);
                if d[axis] * sign_n >= 0.0 { continue; }
                let plane = select(lo[axis] - r, hi[axis] + r, side == 1u);
                let t = (plane - p[axis]) / d[axis];
                let point = p + d * t;
                if t >= 0.0 && t <= 1.0 && point[a] >= lo[a] && point[a] <= hi[a] && point[b] >= lo[b] && point[b] <= hi[b] {
                    var n = vec3<f32>(0.0); n[axis] = sign_n;
                    hit = earlier(hit, Hit(t, n));
                }
            }
        }
        for (var edge = 0u; edge < 4u; edge++) {
            var start = lo;
            start[a] = select(lo[a], hi[a], (edge & 1u) != 0u);
            start[b] = select(lo[b], hi[b], (edge & 2u) != 0u);
            hit = earlier(hit, sweep_edge(p, d, start, hi[axis], axis, r));
        }
    }
    for (var corner = 0u; corner < 8u; corner++) {
        let center = vec3<f32>(select(lo.x, hi.x, (corner & 1u) != 0u),
            select(lo.y, hi.y, (corner & 2u) != 0u), select(lo.z, hi.z, (corner & 4u) != 0u));
        hit = earlier(hit, sweep_sphere(p, d, center, r));
    }
    return hit;
}
fn axis_penetration(p: vec3<f32>, c: Collider, r: f32) -> Hit {
    if c.shape.x == 2.0 {
        let delta = p - c.min.xyz;
        let distance = length(delta);
        var normal = vec3<f32>(0.0, 1.0, 0.0);
        if distance > 1e-8 { normal = delta / distance; }
        return Hit(distance - c.shape.y - r, normal);
    }
    let nearest = clamp(p, c.min.xyz, c.max.xyz);
    let delta = p - nearest;
    let distance = length(delta);
    if distance > 1e-8 { return Hit(distance - r, delta / distance); }
    var depth = 1e30;
    var normal = vec3<f32>(-1.0, 0.0, 0.0);
    for (var axis = 0u; axis < 3u; axis++) {
        if p[axis] - c.min[axis] < depth {
            depth = p[axis] - c.min[axis]; normal = vec3<f32>(0.0); normal[axis] = -1.0;
        }
        if c.max[axis] - p[axis] < depth {
            depth = c.max[axis] - p[axis]; normal = vec3<f32>(0.0); normal[axis] = 1.0;
        }
    }
    return Hit(-depth - r, normal);
}

fn rotate(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz,v) + q.w*v);
}
fn sweep_collider(p: vec3<f32>, d: vec3<f32>, c: Collider, r: f32) -> Hit {
    if c.shape.x != 3.0 { return sweep_axis_collider(p,d,c,r); }
    let inverse = vec4<f32>(-c.rotation.xyz,c.rotation.w);
    let local_collider = Collider(vec4<f32>(1.0,0.0,0.0,0.0),-c.max,c.max,c.rotation);
    let hit = sweep_axis_collider(rotate(p-c.min.xyz,inverse),rotate(d,inverse),local_collider,r);
    return Hit(hit.time,rotate(hit.normal,c.rotation));
}
fn penetration(p: vec3<f32>, c: Collider, r: f32) -> Hit {
    if c.shape.x != 3.0 { return axis_penetration(p,c,r); }
    let inverse = vec4<f32>(-c.rotation.xyz,c.rotation.w);
    let local_collider = Collider(vec4<f32>(1.0,0.0,0.0,0.0),-c.max,c.max,c.rotation);
    let hit = axis_penetration(rotate(p-c.min.xyz,inverse),local_collider,r);
    return Hit(hit.time,rotate(hit.normal,c.rotation));
}

@compute @workgroup_size(64)
fn build_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.counts.x { return; }
    if atomicLoad(&activity[id.x]) == 0u { return; }
    let bucket = cell_index(cell(particles[id.x].position.xyz));
    // Every particle owns exactly one next slot. Traversal starts in the next
    // pass, after all next writes have completed; no fixed bucket capacity.
    next[id.x] = atomicExchange(&heads[bucket], id.x);
}

@compute @workgroup_size(64)
fn density_pressure(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.counts.x { return; }
    if atomicLoad(&activity[i]) == 0u { return; }
    let p = particles[i].position.xyz;
    let c = cell(p);
    var density = 0.0;
    for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = c + vec3<i32>(x, y, z);
            if !valid_cell(neighbor) { continue; }
            var j = atomicLoad(&heads[cell_index(neighbor)]);
            loop {
                if j == END { break; }
                let delta = p - particles[j].position.xyz;
                density += params.kernel.y * poly6(dot(delta, delta));
                j = next[j];
            }
        }
    }
    }
    // Includes self contribution. Positive pressure only avoids tensile clumping.
    density = max(density, params.kernel.z * 0.01);
    densities[i] = vec2<f32>(density, params.kernel.w * max(density - params.kernel.z, 0.0));
}

@compute @workgroup_size(64)
fn acceleration(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.counts.x { return; }
    if atomicLoad(&activity[i]) == 0u { return; }
    let p = particles[i];
    let c = cell(p.position.xyz);
    let h = params.kernel.x;
    let mass = params.kernel.y;
    let rho = densities[i].x;
    let pressure = densities[i].y;
    var a = params.forces.xyz;
    for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = c + vec3<i32>(x, y, z);
            if !valid_cell(neighbor) { continue; }
            var j = atomicLoad(&heads[cell_index(neighbor)]);
            loop {
                if j == END { break; }
                if j != i {
                    let delta = p.position.xyz - particles[j].position.xyz;
                    let r = length(delta);
                    if r > 1e-6 && r < h {
                        let q = 1.0 - r / h;
                        // 3D spiky gradient -45/(pi h^6) * (h-r)^2 * direction.
                        let gradient = -45.0 / (PI * h * h * h * h) * q * q * delta / r;
                        let rho_j = densities[j].x;
                        a -= mass * (pressure / (rho * rho) + densities[j].y / (rho_j * rho_j)) * gradient;
                        // 3D viscosity Laplacian 45/(pi h^6) * (h-r).
                        let laplacian = 45.0 / (PI * h * h * h * h * h) * q;
                        a += params.forces.w * mass / rho_j * (particles[j].velocity.xyz - p.velocity.xyz) * laplacian;
                    }
                }
                j = next[j];
            }
        }
    }
    }
    // Separate acceleration buffer: no velocity write/read races between neighbors.
    accelerations[i] = vec4<f32>(limited(a, params.integration.w), 0.0);
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.counts.x { return; }
    if atomicLoad(&activity[i]) == 0u { return; }
    let dt = params.integration.x;
    var velocity = limited(particles[i].velocity.xyz + accelerations[i].xyz * dt, params.integration.z);
    var position = particles[i].position.xyz;
    let radius = params.integration.y;
    let epsilon = radius * 0.0001;
    // Bounded projection handles existing penetration, including exact centers.
    for (var iteration = 0u; iteration < 4u; iteration++) {
        var moved = false;
        for (var j = 0u; j < params.environment.w; j++) {
            let contact = penetration(position, params.colliders[j], radius);
            if contact.time < 0.0 {
                position += contact.normal * (-contact.time + epsilon);
                velocity = reflected(velocity, contact.normal);
                moved = true;
            }
        }
        if !moved { break; }
    }
    // Continuous sweep prevents crossing thin obstacles at high velocity.
    // Stop safely at the last contact if four impacts exhaust this step's budget.
    var remaining = dt;
    for (var iteration = 0u; iteration < 4u; iteration++) {
        let displacement = velocity * remaining;
        var hit = miss();
        for (var j = 0u; j < params.environment.w; j++) {
            hit = earlier(hit, sweep_collider(position, displacement, params.colliders[j], radius));
        }
        if hit.time > 1.0 { position += displacement; break; }
        position += displacement * hit.time + hit.normal * epsilon;
        velocity = reflected(velocity, hit.normal);
        remaining *= 1.0 - hit.time;
    }
    let lo = params.bounds_min.xyz + vec3<f32>(params.integration.y);
    let hi = params.bounds_max.xyz - vec3<f32>(params.integration.y);
    for (var axis = 0u; axis < 3u; axis++) {
        if position[axis] < lo[axis] {
            position[axis] = lo[axis];
            velocity[axis] = abs(velocity[axis]) * params.outlet_extra.y;
        }
        if position[axis] > hi[axis] {
            position[axis] = hi[axis];
            velocity[axis] = -abs(velocity[axis]) * params.outlet_extra.y;
        }
    }
    particles[i] = Particle(vec4<f32>(position, 0.0), vec4<f32>(velocity, 0.0));
}
