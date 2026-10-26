// Independent 2D SPH implementation. All positions and velocities are GPU resident.
struct Particle { position: vec2<f32>, velocity: vec2<f32> }
struct Collider { shape: vec4<f32>, bounds: vec4<f32>, rotation: vec4<f32> }
struct Params {
    bounds: vec4<f32>,       // min.xy, max.xy
    kernel: vec4<f32>,       // h, mass, rest density, pressure stiffness
    forces: vec4<f32>,       // gravity.xy, viscosity, restitution
    integration: vec4<f32>,  // dt, collision radius, max speed, max acceleration
    counts: vec4<u32>,       // active particles, grid x, grid y, previous active count
    outlet: vec4<f32>,       // center.xy, width, unused
    velocity: vec4<f32>,     // initial velocity.xy, unused
    environment: vec4<u32>,  // outlet lanes, collider count, unused
    colliders: array<Collider, 16>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> heads: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> next: array<u32>;
@group(0) @binding(4) var<storage, read_write> densities: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> accelerations: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> activity: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> live_count: atomic<u32>;
const END: u32 = 0xffffffffu;
const PI: f32 = 3.141592653589793;

fn cell(position: vec2<f32>) -> vec2<i32> {
    return clamp(vec2<i32>(floor((position - params.bounds.xy) / params.kernel.x)),
        vec2<i32>(0), vec2<i32>(params.counts.yz) - vec2<i32>(1));
}
fn cell_index(c: vec2<i32>) -> u32 { return u32(c.y) * params.counts.y + u32(c.x); }
fn valid_cell(c: vec2<i32>) -> bool {
    return all(c >= vec2<i32>(0)) && all(c < vec2<i32>(params.counts.yz));
}
// Normalized 2D poly6: 4/(pi h^8) * (h^2-r^2)^3, written in q=r/h
// form to avoid large intermediate h powers.
fn poly6(r2: f32) -> f32 {
    let h = params.kernel.x;
    let q = max(1.0 - r2 / (h * h), 0.0);
    return 4.0 / (PI * h * h) * q * q * q;
}
fn limited(v: vec2<f32>, maximum: f32) -> vec2<f32> {
    return v * min(1.0, maximum / max(length(v), 1e-12));
}

@compute @workgroup_size(64)
fn emit_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = params.counts.w + id.x;
    if i >= params.counts.x { return; }
    let lanes = max(params.environment.x, 1u);
    let x = ((f32(id.x % lanes) + 0.5) / f32(lanes) - 0.5) * params.outlet.z;
    particles[i] = Particle(params.outlet.xy + vec2<f32>(x, 0.0), params.velocity.xy);
    atomicStore(&activity[i],1u);
    atomicAdd(&live_count,1u);
}

// Contact time is a fraction of this step's displacement. Values above one miss.
struct Hit { time: f32, normal: vec2<f32> }
fn miss() -> Hit { return Hit(2.0, vec2<f32>(0.0)); }
fn earlier(a: Hit, b: Hit) -> Hit { if a.time < b.time { return a; } return b; }

fn sweep_circle(p: vec2<f32>, d: vec2<f32>, center: vec2<f32>, radius: f32) -> Hit {
    let relative = p - center;
    let a = dot(d, d);
    let b = dot(relative, d);
    let c = dot(relative, relative) - radius * radius;
    if a < 1e-16 || b >= 0.0 { return miss(); }
    let discriminant = b * b - a * c;
    if discriminant < 0.0 { return miss(); }
    let t = max(0.0, (-b - sqrt(discriminant)) / a);
    if t > 1.0 { return miss(); }
    return Hit(t, normalize(relative + d * t));
}

fn sweep_box(p: vec2<f32>, d: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>) -> Hit {
    var entry = -1e20;
    var exit = 1e20;
    var normal = vec2<f32>(0.0);
    for (var axis = 0u; axis < 2u; axis++) {
        if abs(d[axis]) < 1e-12 {
            if p[axis] < lo[axis] || p[axis] > hi[axis] { return miss(); }
        } else {
            let a = (lo[axis] - p[axis]) / d[axis];
            let b = (hi[axis] - p[axis]) / d[axis];
            let near = min(a, b);
            if near > entry {
                entry = near;
                normal = vec2<f32>(0.0);
                normal[axis] = -sign(d[axis]);
            }
            exit = min(exit, max(a, b));
        }
    }
    if entry < 0.0 || entry > exit || entry > 1.0 { return miss(); }
    return Hit(entry, normal);
}

// Exact particle-circle vs axis-aligned rectangle sweep: the expanded shape
// is the union of two strips and four circular corners (a rounded rectangle).
fn sweep_axis_collider(p: vec2<f32>, d: vec2<f32>, c: Collider, r: f32) -> Hit {
    if c.shape.x == 2.0 { return sweep_circle(p, d, c.shape.yz, c.shape.w + r); }
    let lo = c.bounds.xy;
    let hi = c.bounds.zw;
    var hit = sweep_box(p, d, lo - vec2<f32>(r, 0.0), hi + vec2<f32>(r, 0.0));
    hit = earlier(hit, sweep_box(p, d, lo - vec2<f32>(0.0, r), hi + vec2<f32>(0.0, r)));
    hit = earlier(hit, sweep_circle(p, d, lo, r));
    hit = earlier(hit, sweep_circle(p, d, hi, r));
    hit = earlier(hit, sweep_circle(p, d, vec2<f32>(lo.x, hi.y), r));
    return earlier(hit, sweep_circle(p, d, vec2<f32>(hi.x, lo.y), r));
}

// Signed distance/normal supports seeds inside solids and moved colliders.
fn axis_penetration(p: vec2<f32>, c: Collider, r: f32) -> Hit {
    if c.shape.x == 2.0 {
        let delta = p - c.shape.yz;
        let distance = length(delta);
        var normal = vec2<f32>(0.0, 1.0);
        if distance > 1e-8 { normal = delta / distance; }
        return Hit(distance - c.shape.w - r, normal);
    }
    let nearest = clamp(p, c.bounds.xy, c.bounds.zw);
    let delta = p - nearest;
    let distance = length(delta);
    if distance > 1e-8 { return Hit(distance - r, delta / distance); }
    let edges = vec4<f32>(p - c.bounds.xy, c.bounds.zw - p);
    var depth = edges.x;
    var normal = vec2<f32>(-1.0, 0.0);
    if edges.y < depth { depth = edges.y; normal = vec2<f32>(0.0, -1.0); }
    if edges.z < depth { depth = edges.z; normal = vec2<f32>(1.0, 0.0); }
    if edges.w < depth { depth = edges.w; normal = vec2<f32>(0.0, 1.0); }
    return Hit(-depth - r, normal);
}

fn rotate(v: vec2<f32>, rotation: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(rotation.x*v.x-rotation.y*v.y, rotation.y*v.x+rotation.x*v.y);
}
fn sweep_collider(p: vec2<f32>, d: vec2<f32>, c: Collider, r: f32) -> Hit {
    if c.shape.x != 3.0 { return sweep_axis_collider(p,d,c,r); }
    let inverse = vec4<f32>(c.rotation.x,-c.rotation.y,0.0,0.0);
    let local_collider = Collider(vec4<f32>(1.0,0.0,0.0,0.0),c.bounds,c.rotation);
    let hit = sweep_axis_collider(rotate(p-c.shape.yz,inverse),rotate(d,inverse),local_collider,r);
    return Hit(hit.time,rotate(hit.normal,c.rotation));
}
fn penetration(p: vec2<f32>, c: Collider, r: f32) -> Hit {
    if c.shape.x != 3.0 { return axis_penetration(p,c,r); }
    let inverse = vec4<f32>(c.rotation.x,-c.rotation.y,0.0,0.0);
    let local_collider = Collider(vec4<f32>(1.0,0.0,0.0,0.0),c.bounds,c.rotation);
    let hit = axis_penetration(rotate(p-c.shape.yz,inverse),local_collider,r);
    return Hit(hit.time,rotate(hit.normal,c.rotation));
}

fn reflected(velocity: vec2<f32>, normal: vec2<f32>) -> vec2<f32> {
    return velocity - (1.0 + params.forces.w) * min(dot(velocity, normal), 0.0) * normal;
}

@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < params.counts.y * params.counts.z { atomicStore(&heads[id.x], END); }
}

@compute @workgroup_size(64)
fn build_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.counts.x { return; }
    if atomicLoad(&activity[id.x]) == 0u { return; }
    let bucket = cell_index(cell(particles[id.x].position));
    // Every particle owns exactly one next slot. Traversal starts in the next
    // pass, after all next writes have completed; no fixed bucket capacity.
    next[id.x] = atomicExchange(&heads[bucket], id.x);
}

@compute @workgroup_size(64)
fn density_pressure(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.counts.x { return; }
    if atomicLoad(&activity[i]) == 0u { return; }
    let p = particles[i].position;
    let c = cell(p);
    var density = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = c + vec2<i32>(x, y);
            if !valid_cell(neighbor) { continue; }
            var j = atomicLoad(&heads[cell_index(neighbor)]);
            loop {
                if j == END { break; }
                let delta = p - particles[j].position;
                density += params.kernel.y * poly6(dot(delta, delta));
                j = next[j];
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
    let c = cell(p.position);
    let h = params.kernel.x;
    let mass = params.kernel.y;
    let rho = densities[i].x;
    let pressure = densities[i].y;
    var a = params.forces.xy;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = c + vec2<i32>(x, y);
            if !valid_cell(neighbor) { continue; }
            var j = atomicLoad(&heads[cell_index(neighbor)]);
            loop {
                if j == END { break; }
                if j != i {
                    let delta = p.position - particles[j].position;
                    let r = length(delta);
                    if r > 1e-6 && r < h {
                        let q = 1.0 - r / h;
                        // 2D spiky gradient -30/(pi h^5) * (h-r)^2 * direction.
                        let gradient = -30.0 / (PI * h * h * h) * q * q * delta / r;
                        let rho_j = densities[j].x;
                        a -= mass * (pressure / (rho * rho) + densities[j].y / (rho_j * rho_j)) * gradient;
                        // 2D viscosity Laplacian 40/(pi h^5) * (h-r).
                        let laplacian = 40.0 / (PI * h * h * h * h) * q;
                        a += params.forces.z * mass / rho_j * (particles[j].velocity - p.velocity) * laplacian;
                    }
                }
                j = next[j];
            }
        }
    }
    // Separate acceleration buffer: no velocity write/read races between neighbors.
    accelerations[i] = limited(a, params.integration.w);
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.counts.x { return; }
    if atomicLoad(&activity[i]) == 0u { return; }
    let dt = params.integration.x;
    var velocity = limited(particles[i].velocity + accelerations[i] * dt, params.integration.z);
    var position = particles[i].position;
    let radius = params.integration.y;
    let epsilon = radius * 0.0001;
    // Bounded projection handles existing penetration, including exact centers.
    for (var iteration = 0u; iteration < 4u; iteration++) {
        var moved = false;
        for (var j = 0u; j < params.environment.y; j++) {
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
        for (var j = 0u; j < params.environment.y; j++) {
            hit = earlier(hit, sweep_collider(position, displacement, params.colliders[j], radius));
        }
        if hit.time > 1.0 { position += displacement; break; }
        position += displacement * hit.time + hit.normal * epsilon;
        velocity = reflected(velocity, hit.normal);
        remaining *= 1.0 - hit.time;
    }
    let lo = params.bounds.xy + vec2<f32>(params.integration.y);
    let hi = params.bounds.zw - vec2<f32>(params.integration.y);
    for (var axis = 0u; axis < 2u; axis++) {
        if position[axis] < lo[axis] {
            position[axis] = lo[axis];
            velocity[axis] = abs(velocity[axis]) * params.forces.w;
        }
        if position[axis] > hi[axis] {
            position[axis] = hi[axis];
            velocity[axis] = -abs(velocity[axis]) * params.forces.w;
        }
    }
    particles[i] = Particle(position, velocity);
}
