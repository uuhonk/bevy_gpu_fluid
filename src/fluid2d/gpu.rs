use std::{collections::HashMap, sync::Arc};

use super::{Fluid2dComputeContext, Fluid2dComputePasses};
use crate::compute::FluidComputeStage;
use bevy::{
    asset::embedded_asset,
    core_pipeline::schedule::camera_driver,
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
    },
    shader::ShaderCacheError,
};
use bytemuck::{Pod, Zeroable};

use super::{
    Fluid2d, Fluid2dCollider, Fluid2dFaucet, Fluid2dSettings, FluidParticle2d, MAX_COLLIDERS_2D,
};

/// Public ordering contract in `RenderGraph`. Custom passes go after `Simulate`
/// and before `camera_driver` (or in `Render` to replace the built-in fluid pass).
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fluid2dSystems {
    /// Render: GPU allocation and settings synchronization.
    Prepare,
    /// Render: prepare extension bind groups, after allocation and before step planning.
    PrepareCompute,
    /// Render: fixed-substep planning, after all extensions are prepared.
    Plan,
    Simulate,
    Render,
}

/// Simulation only. Requires Bevy's asset and render plugins to be installed first.
pub struct Fluid2dPlugin;

impl Plugin for Fluid2dPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/sph.wgsl");
        let render_app = app
            .get_sub_app_mut(RenderApp)
            .expect("Fluid2dPlugin requires RenderPlugin; add it after DefaultPlugins");
        render_app
            .init_resource::<ExtractedFluids>()
            .init_resource::<GpuFluids2d>()
            .init_resource::<Fluid2dComputePasses>()
            .add_systems(ExtractSchedule, extract_fluids)
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                prepare_fluids
                    .in_set(RenderSystems::PrepareResources)
                    .in_set(Fluid2dSystems::Prepare),
            )
            .configure_sets(
                Render,
                (Fluid2dSystems::PrepareCompute, Fluid2dSystems::Plan)
                    .chain()
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(Render, plan_fluids.in_set(Fluid2dSystems::Plan))
            .configure_sets(
                RenderGraph,
                (Fluid2dSystems::Simulate, Fluid2dSystems::Render)
                    .chain()
                    .in_set(RenderGraphSystems::Render)
                    .before(camera_driver),
            )
            .add_systems(RenderGraph, simulate.in_set(Fluid2dSystems::Simulate));
    }
}

#[derive(Resource, Default)]
struct ExtractedFluids {
    fluids: HashMap<Entity, Fluid2d>,
    delta: f64,
}

fn extract_fluids(
    mut extracted: ResMut<ExtractedFluids>,
    query: Extract<Query<(Entity, &Fluid2d)>>,
    time: Extract<Res<Time>>,
) {
    extracted.fluids.clear();
    extracted
        .fluids
        .extend(query.iter().map(|(e, f)| (e, f.clone())));
    extracted.delta = time.delta_secs_f64();
}

/// GPU simulations keyed by **main-world** entity. Exists only in `RenderApp`.
#[derive(Resource, Default)]
pub struct GpuFluids2d {
    fluids: HashMap<Entity, GpuFluid2d>,
    failures: HashMap<Entity, String>,
}

impl GpuFluids2d {
    pub fn get(&self, entity: Entity) -> Option<&GpuFluid2d> {
        self.fluids.get(&entity)
    }
    pub fn iter(&self) -> impl Iterator<Item = (Entity, &GpuFluid2d)> {
        self.fluids.iter().map(|(&e, f)| (e, f))
    }
    /// A device limit failure prevents allocation; logged once and available here.
    pub fn error(&self, entity: Entity) -> Option<&str> {
        self.failures.get(&entity).map(String::as_str)
    }
}

/// GPU data for custom render passes. Positions and velocities stay on the GPU.
///
/// `particle_buffer()` is a storage/vertex/COPY_SRC buffer of [`FluidParticle2d`].
/// Its allocation can change on reset or seed replacement: rebuild bind groups
/// when the buffer ID changes, or build them each frame. Only substep extensions may mutate it.
pub struct GpuFluid2d {
    particles: Buffer,
    activity: Buffer,
    live_count: Buffer,
    _heads: Buffer,
    _next: Buffer,
    _density: Buffer,
    _acceleration: Buffer,
    uniform: Buffer,
    bind_group: BindGroup,
    count: u32,
    capacity: u32,
    faucet: Option<Fluid2dFaucet>,
    emission_time: f64,
    uniform_stride: u64,
    dispatches: Vec<StepDispatch>,
    grid: UVec2,
    settings: Fluid2dSettings,
    seed: Arc<[FluidParticle2d]>,
    revision: u64,
    paused: bool,
    accumulator: f64,
    simulated_steps: u64,
}

impl GpuFluid2d {
    pub fn particle_buffer(&self) -> &Buffer {
        &self.particles
    }
    /// Released slot prefix (including retired slots). Renderers must check activity.
    pub fn particle_count(&self) -> u32 {
        self.count
    }
    /// Per-slot u32: 1 is alive, 0 is inactive. Shader atomic storage in compute, read-only in rendering.
    pub fn particle_activity_buffer(&self) -> &Buffer {
        &self.activity
    }
    /// One GPU atomic<u32>, actual live count. STORAGE | COPY_SRC; no automatic CPU readback.
    pub fn live_count_buffer(&self) -> &Buffer {
        &self.live_count
    }
    /// Per-slot vec2<f32>: physical density and pressure. Read-only for extensions.
    /// This auxiliary buffer may be replaced when the grid is resized.
    pub fn density_pressure_buffer(&self) -> &Buffer {
        &self._density
    }
    /// Per-slot vec2<f32> (8 bytes). Modify in BeforeIntegrate to add custom forces.
    /// This auxiliary buffer may be replaced when the grid is resized.
    pub fn acceleration_buffer(&self) -> &Buffer {
        &self._acceleration
    }
    pub fn particle_capacity(&self) -> u32 {
        self.capacity
    }
    /// Emission budget exhausted. Live particles continue simulating.
    pub fn is_exhausted(&self) -> bool {
        self.count == self.capacity
    }
    pub fn settings(&self) -> &Fluid2dSettings {
        &self.settings
    }
    /// Substeps encoded since creation/reset. GPU execution is asynchronous.
    pub fn simulated_steps(&self) -> u64 {
        self.simulated_steps
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ColliderParams {
    shape: [f32; 4],
    bounds: [f32; 4],
    rotation: [f32; 4],
}

impl From<&Fluid2dCollider> for ColliderParams {
    fn from(collider: &Fluid2dCollider) -> Self {
        match collider {
            Fluid2dCollider::Rectangle(r) => ColliderParams {
                shape: [1.0, 0.0, 0.0, 0.0],
                bounds: [r.min.x, r.min.y, r.max.x, r.max.y],
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
            Fluid2dCollider::Circle { center, radius } => ColliderParams {
                shape: [2.0, center.x, center.y, *radius],
                bounds: [0.0; 4],
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
            Fluid2dCollider::OrientedRectangle {
                center,
                half_extents,
                rotation,
            } => ColliderParams {
                shape: [3.0, center.x, center.y, 0.0],
                bounds: [
                    -half_extents.x,
                    -half_extents.y,
                    half_extents.x,
                    half_extents.y,
                ],
                rotation: [rotation.cos(), rotation.sin(), 0.0, 0.0],
            },
        }
    }
}

// All members use vec4-sized blocks, including the fixed collider array.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SimParams {
    bounds: [f32; 4],
    kernel: [f32; 4],
    forces: [f32; 4],
    integration: [f32; 4],
    counts: [u32; 4],
    outlet: [f32; 4],
    velocity: [f32; 4],
    environment: [u32; 4],
    colliders: [ColliderParams; MAX_COLLIDERS_2D],
}

impl SimParams {
    pub(crate) fn new(s: &Fluid2dSettings, count: u32) -> Self {
        let grid = s.grid_size();
        Self {
            bounds: [
                s.bounds.min.x,
                s.bounds.min.y,
                s.bounds.max.x,
                s.bounds.max.y,
            ],
            kernel: [
                s.smoothing_radius,
                s.particle_mass,
                s.rest_density,
                s.stiffness,
            ],
            forces: [s.gravity.x, s.gravity.y, s.viscosity, s.restitution],
            integration: [
                s.time_step,
                s.particle_radius,
                s.max_speed,
                s.max_acceleration,
            ],
            counts: [count, grid.x, grid.y, count],
            outlet: [0.0; 4],
            velocity: [0.0; 4],
            environment: [0; 4],
            colliders: [ColliderParams::zeroed(); MAX_COLLIDERS_2D],
        }
    }
    pub(crate) fn with_colliders(mut self, colliders: &[Fluid2dCollider]) -> Self {
        self.environment[1] = colliders.len() as u32;
        for (gpu, collider) in self.colliders.iter_mut().zip(colliders) {
            *gpu = ColliderParams::from(collider);
        }
        self
    }
}

struct StepDispatch {
    offset: u32,
    count: u32,
    emitted: u32,
}

#[derive(Resource)]
struct SimPipeline {
    layout: BindGroupLayoutDescriptor,
    passes: [CachedComputePipelineId; 6],
}

fn init_pipeline(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let layout = BindGroupLayoutDescriptor::new(
        "fluid2d simulation",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(true, BufferSize::new(size_of::<SimParams>() as u64)),
                storage_buffer_sized(false, BufferSize::new(16)),
                storage_buffer_sized(false, BufferSize::new(4)),
                storage_buffer_sized(false, BufferSize::new(4)),
                storage_buffer_sized(false, BufferSize::new(8)),
                storage_buffer_sized(false, BufferSize::new(8)),
                storage_buffer_sized(false, BufferSize::new(4)),
                storage_buffer_sized(false, BufferSize::new(4)),
            ),
        ),
    );
    let shader = assets.load("embedded://bevy_gpu_fluid/fluid2d/shaders/sph.wgsl");
    let passes = [
        "emit_particles",
        "clear_grid",
        "build_grid",
        "density_pressure",
        "acceleration",
        "integrate",
    ]
    .map(|entry| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("fluid2d {entry}").into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            entry_point: Some(entry.into()),
            ..default()
        })
    });
    commands.insert_resource(SimPipeline { layout, passes });
}

fn device_check(device: &RenderDevice, fluid: &Fluid2d) -> Result<(), String> {
    let limits = device.limits();
    let grid = fluid.settings.grid_size();
    let count = fluid.capacity.max(1) as u64;
    let sizes = [count * 16, count * 8, grid.x as u64 * grid.y as u64 * 4];
    if sizes
        .iter()
        .any(|&n| n > limits.max_buffer_size || n > limits.max_storage_buffer_binding_size)
    {
        return Err("fluid buffer exceeds this adapter's storage buffer limits".into());
    }
    if (grid.x * grid.y).div_ceil(64) > limits.max_compute_workgroups_per_dimension
        || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension as u64
    {
        return Err("fluid exceeds this adapter's compute dispatch limits".into());
    }
    if limits.max_storage_buffers_per_shader_stage < 7
        || limits.max_compute_invocations_per_workgroup < 64
    {
        return Err("fluid requires 7 storage buffers and workgroups of 64 threads".into());
    }
    Ok(())
}

fn allocate(
    device: &RenderDevice,
    cache: &PipelineCache,
    pipeline: &SimPipeline,
    fluid: &Fluid2d,
    previous_particles: Option<(Buffer, Buffer, Buffer)>,
) -> GpuFluid2d {
    let count = fluid.seed.len() as u32;
    let grid = fluid.settings.grid_size();
    let storage = |label: &'static str, size| {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let (previous_particles, previous_activity, previous_live_count) = match previous_particles {
        Some((particles, activity, live_count)) => {
            (Some(particles), Some(activity), Some(live_count))
        }
        None => (None, None, None),
    };
    let particles = previous_particles.unwrap_or_else(|| {
        if fluid.seed.is_empty() {
            return device.create_buffer(&BufferDescriptor {
                label: Some("fluid2d finite particle pool"),
                size: fluid.capacity.max(1) as u64 * 16,
                usage: BufferUsages::STORAGE | BufferUsages::VERTEX | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
        }
        device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("fluid2d particles"),
            contents: bytemuck::cast_slice(&fluid.seed),
            usage: BufferUsages::STORAGE | BufferUsages::VERTEX | BufferUsages::COPY_SRC,
        })
    });
    let n = fluid.capacity.max(1) as u64;
    let activity = previous_activity.unwrap_or_else(|| {
        let mut flags = vec![0u32; n as usize];
        flags[..count as usize].fill(1);
        device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("fluid2d activity"),
            contents: bytemuck::cast_slice(&flags),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        })
    });
    let live_count = previous_live_count.unwrap_or_else(|| {
        device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("fluid2d live count"),
            contents: bytemuck::bytes_of(&count),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        })
    });
    let heads = storage("fluid2d grid heads", grid.x as u64 * grid.y as u64 * 4);
    let next = storage("fluid2d next", n * 4);
    let density = storage("fluid2d density + pressure", n * 8);
    let acceleration = storage("fluid2d acceleration", n * 8);
    let uniform_stride = (size_of::<SimParams>() as u64)
        .div_ceil(device.limits().min_uniform_buffer_offset_alignment as u64)
        * device.limits().min_uniform_buffer_offset_alignment as u64;
    let uniform = device.create_buffer(&BufferDescriptor {
        label: Some("fluid2d parameters"),
        size: uniform_stride * 64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(
        "fluid2d simulation",
        &cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            BindingResource::Buffer(BufferBinding {
                buffer: &uniform,
                offset: 0,
                size: BufferSize::new(size_of::<SimParams>() as u64),
            }),
            particles.as_entire_binding(),
            heads.as_entire_binding(),
            next.as_entire_binding(),
            density.as_entire_binding(),
            acceleration.as_entire_binding(),
            activity.as_entire_binding(),
            live_count.as_entire_binding(),
        )),
    );
    GpuFluid2d {
        particles,
        activity,
        live_count,
        _heads: heads,
        _next: next,
        _density: density,
        _acceleration: acceleration,
        uniform,
        bind_group,
        count,
        capacity: fluid.capacity,
        faucet: fluid.faucet,
        emission_time: 0.0,
        uniform_stride,
        dispatches: Vec::new(),
        grid,
        settings: fluid.settings.clone(),
        seed: fluid.seed.clone(),
        revision: fluid.revision,
        paused: fluid.paused,
        accumulator: 0.0,
        simulated_steps: 0,
    }
}

fn prepare_fluids(
    extracted: Res<ExtractedFluids>,
    mut gpu: ResMut<GpuFluids2d>,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    pipeline: Res<SimPipeline>,
) {
    gpu.fluids.retain(|e, _| extracted.fluids.contains_key(e));
    gpu.failures.retain(|e, _| extracted.fluids.contains_key(e));
    for (&entity, fluid) in &extracted.fluids {
        if let Err(message) = device_check(&device, fluid) {
            gpu.fluids.remove(&entity);
            if gpu.failures.get(&entity) != Some(&message) {
                error!("Fluid2d {entity:?}: {message}");
            }
            gpu.failures.insert(entity, message);
            continue;
        }
        gpu.failures.remove(&entity);
        let old = gpu.fluids.get(&entity);
        let reset = old.is_none_or(|g| {
            g.revision != fluid.revision
                || g.capacity != fluid.capacity
                || !Arc::ptr_eq(&g.seed, &fluid.seed)
        });
        let grid_changed = old.is_some_and(|g| g.grid != fluid.settings.grid_size());
        if reset || grid_changed {
            let previous = if reset {
                None
            } else {
                old.map(|g| {
                    (
                        g.particles.clone(),
                        g.activity.clone(),
                        g.live_count.clone(),
                    )
                })
            };
            let mut allocated = allocate(&device, &cache, &pipeline, fluid, previous);
            if !reset && let Some(old) = old {
                allocated.simulated_steps = old.simulated_steps;
                allocated.count = old.count;
                allocated.emission_time = old.emission_time;
                allocated.accumulator = old.accumulator;
            }
            gpu.fluids.insert(entity, allocated);
        }
        let g = gpu.fluids.get_mut(&entity).unwrap();
        if g.settings.time_step != fluid.settings.time_step || fluid.paused {
            g.accumulator = 0.0;
        }
        g.settings = fluid.settings.clone();
        g.paused = fluid.paused;
        if g.faucet.map(|f| (f.is_open, f.rows_per_second))
            != fluid.faucet.map(|f| (f.is_open, f.rows_per_second))
            || g.paused
        {
            g.emission_time = 0.0;
        }
        g.faucet = fluid.faucet;
    }
}

fn plan_fluids(
    extracted: Res<ExtractedFluids>,
    mut gpu: ResMut<GpuFluids2d>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    pipeline: Res<SimPipeline>,
    extensions: Res<Fluid2dComputePasses>,
) {
    for (&entity, g) in &mut gpu.fluids {
        let Some(fluid) = extracted.fluids.get(&entity) else {
            continue;
        };
        g.dispatches.clear();
        if g.paused
            || !extensions.ready(entity, &cache)
            || !pipeline
                .passes
                .iter()
                .all(|id| cache.get_compute_pipeline(*id).is_some())
        {
            g.accumulator = 0.0;
            continue;
        }
        let steps = fixed_steps(
            &mut g.accumulator,
            extracted.delta,
            g.settings.time_step as f64,
            g.settings.max_substeps,
        );
        let mut bytes = vec![0u8; g.uniform_stride as usize * steps as usize];
        for step in 0..steps {
            let old_count = g.count;
            let emitted = emit_count(
                &mut g.emission_time,
                g.faucet.as_ref(),
                g.settings.time_step as f64,
                g.capacity - g.count,
            );
            g.count += emitted;
            let mut params = SimParams::new(&g.settings, g.count).with_colliders(&fluid.colliders);
            params.counts[3] = old_count;
            if let Some(f) = g.faucet {
                params.outlet = [f.position.x, f.position.y, f.width, 0.0];
                params.velocity = [f.velocity.x, f.velocity.y, 0.0, 0.0];
                params.environment[0] = f.lanes();
            }
            let offset = step as u64 * g.uniform_stride;
            bytes[offset as usize..offset as usize + size_of::<SimParams>()]
                .copy_from_slice(bytemuck::bytes_of(&params));
            g.dispatches.push(StepDispatch {
                offset: offset as u32,
                count: g.count,
                emitted,
            });
        }
        if !bytes.is_empty() {
            queue.write_buffer(&g.uniform, 0, &bytes);
        }
    }
}

fn simulate(
    mut context: RenderContext,
    mut gpu: ResMut<GpuFluids2d>,
    pipeline: Res<SimPipeline>,
    cache: Res<PipelineCache>,
    extensions: Res<Fluid2dComputePasses>,
    mut reported_error: Local<bool>,
) {
    let pipelines: Option<Vec<_>> = pipeline
        .passes
        .iter()
        .map(|&id| cache.get_compute_pipeline(id))
        .collect();
    let Some(pipelines) = pipelines else {
        for id in pipeline.passes {
            if let CachedPipelineState::Err(err) = cache.get_compute_pipeline_state(id)
                && !matches!(
                    err,
                    ShaderCacheError::ShaderNotLoaded(_)
                        | ShaderCacheError::ShaderImportNotYetAvailable
                )
                && !*reported_error
            {
                error!("Fluid2d compute pipeline: {err}");
                *reported_error = true;
            }
        }
        return;
    };
    for (&entity, g) in &mut gpu.fluids {
        for (step_index, step) in g.dispatches.iter().enumerate() {
            // Separate dispatches/passes provide global visibility between all grid,
            // density, force and integration writes. Never use a workgroup barrier
            // as a substitute for a device-wide dependency.
            for (i, pipeline) in pipelines.iter().enumerate() {
                if i == 1 {
                    extensions.encode(
                        FluidComputeStage::BeforeSubstep,
                        &mut Fluid2dComputeContext {
                            entity,
                            fluid: g,
                            particle_count: step.count,
                            substep_index: g.simulated_steps + step_index as u64,
                            encoder: context.command_encoder(),
                            pipelines: &cache,
                        },
                    );
                }
                if i == 5 {
                    extensions.encode(
                        FluidComputeStage::BeforeIntegrate,
                        &mut Fluid2dComputeContext {
                            entity,
                            fluid: g,
                            particle_count: step.count,
                            substep_index: g.simulated_steps + step_index as u64,
                            encoder: context.command_encoder(),
                            pipelines: &cache,
                        },
                    );
                }
                let invocations = match i {
                    0 => step.emitted,
                    1 => g.grid.x * g.grid.y,
                    _ => step.count,
                };
                if invocations == 0 {
                    continue;
                }
                let mut pass =
                    context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("fluid2d SPH"),
                            ..default()
                        });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &g.bind_group, &[step.offset]);
                pass.dispatch_workgroups(invocations.div_ceil(64), 1, 1);
            }
            extensions.encode(
                FluidComputeStage::AfterSubstep,
                &mut Fluid2dComputeContext {
                    entity,
                    fluid: g,
                    particle_count: step.count,
                    substep_index: g.simulated_steps + step_index as u64,
                    encoder: context.command_encoder(),
                    pipelines: &cache,
                },
            );
        }
        g.simulated_steps += g.dispatches.len() as u64;
    }
}

fn emit_count(time: &mut f64, faucet: Option<&Fluid2dFaucet>, dt: f64, remaining: u32) -> u32 {
    let Some(f) = faucet.filter(|f| f.is_open && remaining > 0) else {
        *time = 0.0;
        return 0;
    };
    *time += dt;
    let interval = 1.0 / f.rows_per_second as f64;
    if *time + interval * 1e-6 < interval {
        return 0;
    }
    *time = (*time - interval).max(0.0);
    f.lanes().min(remaining)
}

fn fixed_steps(accumulator: &mut f64, delta: f64, dt: f64, max: u32) -> u32 {
    *accumulator += delta.max(0.0).min(dt * max as f64);
    let steps = (*accumulator / dt).floor().min(max as f64) as u32;
    *accumulator -= steps as f64 * dt;
    steps
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_step_carries_fraction_and_caps_stalls() {
        let mut acc = 0.0;
        assert_eq!(fixed_steps(&mut acc, 0.004, 0.01, 4), 0);
        assert_eq!(fixed_steps(&mut acc, 0.007, 0.01, 4), 1);
        assert!((acc - 0.001).abs() < 1e-12);
        assert_eq!(fixed_steps(&mut acc, 10.0, 0.01, 4), 4);
        assert!(acc < 0.01);
    }
    #[test]
    fn finite_emission_clock_closes_without_catchup_and_caps_last_row() {
        let mut f = Fluid2dFaucet {
            width: 0.8,
            particle_spacing: 0.2,
            rows_per_second: 10.0,
            ..default()
        };
        let mut time = 0.0;
        assert_eq!(emit_count(&mut time, Some(&f), 0.04, 7), 0);
        assert_eq!(emit_count(&mut time, Some(&f), 0.06, 7), 4);
        f.is_open = false;
        assert_eq!(emit_count(&mut time, Some(&f), 10.0, 3), 0);
        assert_eq!(time, 0.0);
        f.is_open = true;
        assert_eq!(emit_count(&mut time, Some(&f), 0.1, 3), 3);
        assert_eq!(emit_count(&mut time, Some(&f), 0.1, 0), 0);
    }
}
