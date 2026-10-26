// Optional GPU extension: center/segment triggers, not solid collision response.
struct Particle { position: vec4<f32>, velocity: vec4<f32> }
struct Collider { shape: vec4<f32>, min: vec4<f32>, max: vec4<f32>, rotation: vec4<f32> }
struct Params { counts: vec4<u32>, regions: array<Collider,16> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage,read> particles: array<Particle>;
@group(0) @binding(2) var<storage,read_write> activity: array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> live_count: atomic<u32>;
@group(0) @binding(4) var<storage,read_write> previous: array<vec4<f32>>;

fn intersects_box(p: vec3<f32>, d: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> bool {
    var entry=0.0; var exit=1.0;
    for(var axis=0u;axis<3u;axis++) {
        if abs(d[axis])<1e-12 { if p[axis]<lo[axis] || p[axis]>hi[axis] {return false;} }
        else {
            let a=(lo[axis]-p[axis])/d[axis];let b=(hi[axis]-p[axis])/d[axis];
            entry=max(entry,min(a,b));exit=min(exit,max(a,b));
        }
    }
    return entry<=exit;
}
fn local(v: vec3<f32>, rotation: vec4<f32>) -> vec3<f32> {
    let q=vec4<f32>(-rotation.xyz,rotation.w);
    return v+2.0*cross(q.xyz,cross(q.xyz,v)+q.w*v);
}
fn intersects(start: vec3<f32>, finish: vec3<f32>, c: Collider) -> bool {
    let displacement=finish-start;
    if c.shape.x==2.0 {
        let delta=start-c.min.xyz;
        let t=clamp(-dot(delta,displacement)/max(dot(displacement,displacement),1e-20),0.0,1.0);
        let closest=delta+t*displacement;
        return dot(closest,closest)<=c.shape.y*c.shape.y;
    }
    if c.shape.x==3.0 {return intersects_box(local(start-c.min.xyz,c.rotation),local(displacement,c.rotation),-c.max.xyz,c.max.xyz);}
    return intersects_box(start,displacement,c.min.xyz,c.max.xyz);
}
fn retire(i: u32) {
    if atomicExchange(&activity[i],0u)==1u {atomicSub(&live_count,1u);}
}
@compute @workgroup_size(64)
fn snapshot(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i>=params.counts.x {return;}
    if atomicLoad(&activity[i])==0u {return;}
    previous[i]=particles[i].position;
    let p=particles[i].position.xyz;
    for(var j=0u;j<params.counts.y;j++) {if intersects(p,p,params.regions[j]) {retire(i);return;}}
}
@compute @workgroup_size(64)
fn collect(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i>=params.counts.x {return;}
    if atomicLoad(&activity[i])==0u {return;}
    for(var j=0u;j<params.counts.y;j++) {if intersects(previous[i].xyz,particles[i].position.xyz,params.regions[j]) {retire(i);return;}}
}
