//! Optional recycling extension, implemented entirely through the public substep API.
use super::{gpu::ColliderParams, *};
use crate::compute::FluidComputeStage;
use bevy::{
    asset::embedded_asset,
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup,
        render_resource::{binding_types::*, *},
        renderer::{RenderDevice, RenderQueue},
    },
};
use bytemuck::{Pod, Zeroable};
use std::{collections::HashMap, sync::Arc};

pub const MAX_RECYCLING_REGIONS_3D: usize = 16;
/// Trigger regions, separate from solid collision obstacles. Particle centers/segments
/// inside these shapes are retired on the GPU. Add Fluid3dRecyclingPlugin to enable.
#[derive(Component, Clone)]
pub struct Fluid3dRecycling {
    regions: Arc<[Fluid3dCollider]>,
    pub enabled: bool,
}
impl Fluid3dRecycling {
    pub fn new(regions: Vec<Fluid3dCollider>) -> Result<Self, Fluid3dError> {
        let mut result = Self {
            regions: Arc::from([]),
            enabled: true,
        };
        result.set_regions(regions)?;
        Ok(result)
    }
    pub fn regions(&self) -> &[Fluid3dCollider] {
        &self.regions
    }
    /// Transactional: invalid regions preserve the previous list. Empty disables collection.
    pub fn set_regions(&mut self, regions: Vec<Fluid3dCollider>) -> Result<(), Fluid3dError> {
        if regions.len() > MAX_RECYCLING_REGIONS_3D {
            return Err(Fluid3dError("maximum 16 recycling regions per fluid"));
        }
        for region in &regions {
            region.validate()?;
        }
        self.regions = regions.into();
        Ok(())
    }
}

/// Opt-in example of a GPU compute extension. Not included in Fluid3dPlugins.
pub struct Fluid3dRecyclingPlugin;
impl Plugin for Fluid3dRecyclingPlugin {
    fn build(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<Fluid3dPlugin>(),
            "add Fluid3dPlugin first"
        );
        embedded_asset!(app, "shaders/recycle.wgsl");
        app.sub_app_mut(RenderApp)
            .init_resource::<ExtractedRegions>()
            .add_systems(ExtractSchedule, extract)
            .add_systems(RenderStartup, init)
            .add_systems(Render, prepare.in_set(Fluid3dSystems::PrepareCompute));
    }
}
#[derive(Resource, Default)]
struct ExtractedRegions(HashMap<Entity, Fluid3dRecycling>);
fn extract(
    mut regions: ResMut<ExtractedRegions>,
    query: Extract<Query<(Entity, &Fluid3dRecycling), With<Fluid3d>>>,
) {
    regions.0.clear();
    regions.0.extend(
        query
            .iter()
            .filter(|(_, r)| r.enabled && !r.regions.is_empty())
            .map(|(e, r)| (e, r.clone())),
    );
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    counts: [u32; 4],
    regions: [ColliderParams; 16],
}
#[derive(Clone)]
struct Entry {
    particle_id: BufferId,
    uniform: Buffer,
    // Bind groups retain their buffers; keep the history handle explicit for readability.
    _history: Buffer,
    group: BindGroup,
}
#[derive(Resource)]
struct Prepared {
    layout: BindGroupLayoutDescriptor,
    pipelines: [CachedComputePipelineId; 2],
    entries: HashMap<Entity, Entry>,
}
fn init(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let layout = BindGroupLayoutDescriptor::new(
        "fluid3d recycling",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, BufferSize::new(size_of::<Params>() as u64)),
                storage_buffer_read_only_sized(false, BufferSize::new(32)),
                storage_buffer_sized(false, BufferSize::new(4)),
                storage_buffer_sized(false, BufferSize::new(4)),
                storage_buffer_sized(false, BufferSize::new(16)),
            ),
        ),
    );
    let shader = assets.load("embedded://bevy_gpu_fluid/fluid3d/shaders/recycle.wgsl");
    let pipelines = ["snapshot", "collect"].map(|entry| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("fluid3d recycle {entry}").into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            entry_point: Some(entry.into()),
            ..default()
        })
    });
    commands.insert_resource(Prepared {
        layout,
        pipelines,
        entries: default(),
    });
}
fn prepare(
    mut prepared: ResMut<Prepared>,
    regions: Res<ExtractedRegions>,
    fluids: Res<GpuFluids3d>,
    (device, queue): (Res<RenderDevice>, Res<RenderQueue>),
    cache: Res<PipelineCache>,
    mut passes: ResMut<Fluid3dComputePasses>,
    mut reported: Local<bool>,
) {
    prepared
        .entries
        .retain(|e, _| regions.0.contains_key(e) && fluids.get(*e).is_some());
    for (&entity, regions) in &regions.0 {
        let Some(fluid) = fluids.get(entity) else {
            continue;
        };
        let mut params = Params::zeroed();
        params.counts = [
            fluid.particle_capacity(),
            regions.regions.len() as u32,
            0,
            0,
        ];
        for (target, region) in params.regions.iter_mut().zip(regions.regions.iter()) {
            *target = ColliderParams::from(region);
        }
        if prepared
            .entries
            .get(&entity)
            .is_none_or(|e| e.particle_id != fluid.particle_buffer().id())
        {
            let uniform = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("recycling regions"),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });
            let history = device.create_buffer(&BufferDescriptor {
                label: Some("recycling previous positions"),
                size: fluid.particle_capacity().max(1) as u64 * 16,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let group = device.create_bind_group(
                "fluid3d recycling",
                &cache.get_bind_group_layout(&prepared.layout),
                &BindGroupEntries::sequential((
                    uniform.as_entire_binding(),
                    fluid.particle_buffer().as_entire_binding(),
                    fluid.particle_activity_buffer().as_entire_binding(),
                    fluid.live_count_buffer().as_entire_binding(),
                    history.as_entire_binding(),
                )),
            );
            prepared.entries.insert(
                entity,
                Entry {
                    particle_id: fluid.particle_buffer().id(),
                    uniform,
                    _history: history,
                    group,
                },
            );
        }
        queue.write_buffer(
            &prepared.entries[&entity].uniform,
            0,
            bytemuck::bytes_of(&params),
        );
    }
    for id in prepared.pipelines {
        if let CachedPipelineState::Err(err) = cache.get_compute_pipeline_state(id)
            && !matches!(
                err,
                bevy::shader::ShaderCacheError::ShaderNotLoaded(_)
                    | bevy::shader::ShaderCacheError::ShaderImportNotYetAvailable
            )
            && !*reported
        {
            error!("fluid3d recycling pipeline: {err}");
            *reported = true;
        }
    }
    passes.insert(RecyclingPass {
        pipelines: prepared.pipelines,
        entries: prepared.entries.clone(),
    });
}
struct RecyclingPass {
    pipelines: [CachedComputePipelineId; 2],
    entries: HashMap<Entity, Entry>,
}
impl Fluid3dComputePass for RecyclingPass {
    fn is_ready(&self, entity: Entity, cache: &PipelineCache) -> bool {
        !self.entries.contains_key(&entity)
            || self
                .pipelines
                .iter()
                .all(|id| cache.get_compute_pipeline(*id).is_some())
    }
    fn encode(&self, stage: FluidComputeStage, context: &mut Fluid3dComputeContext<'_>) {
        let Some(entry) = self.entries.get(&context.entity) else {
            return;
        };
        if context.particle_count == 0 {
            return;
        }
        let index = match stage {
            FluidComputeStage::BeforeSubstep => 0,
            FluidComputeStage::AfterSubstep => 1,
            FluidComputeStage::BeforeIntegrate => return,
        };
        let Some(pipeline) = context
            .pipelines
            .get_compute_pipeline(self.pipelines[index])
        else {
            return;
        };
        let mut pass = context.encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("fluid3d recycling extension"),
            ..default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &entry.group, &[]);
        pass.dispatch_workgroups(context.particle_count.div_ceil(64), 1, 1);
    }
}
