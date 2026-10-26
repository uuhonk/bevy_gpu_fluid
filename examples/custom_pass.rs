//! Replaces the built-in renderer with an application-owned instanced render pass.
//! Only Fluid2dPlugin is installed; no compute or grid code is copied.
use std::collections::HashMap;

use bevy::{
    camera::ScalingMode,
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderGraph},
        texture::GpuImage,
    },
};
use bevy_gpu_fluid::{
    fluid2d::{Fluid2dSystems, GpuFluids2d},
    prelude::*,
};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.018, 0.025, 0.045)))
        .add_plugins((DefaultPlugins, Fluid2dPlugin, CustomFluidRenderer))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    commands.spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 10.0,
            },
            ..OrthographicProjection::default_2d()
        }),
    ));
    let fluid = Fluid2d::new(
        Fluid2dSettings::default(),
        FluidParticle2d::lattice(Vec2::new(-7.5, -3.9), UVec2::new(40, 48), 0.16),
    )
    .unwrap();
    // Reuse the convenience target/sprite bundle, but supply our own render pass.
    commands.spawn(Fluid2dBundle::new(fluid, &mut images, UVec2::new(1280, 720)).unwrap());
}

pub(crate) struct CustomFluidRenderer;
impl Plugin for CustomFluidRenderer {
    fn build(&self, app: &mut App) {
        // Assets<Shader> retains a strong handle in the render world. The application
        // owns this shader; the library's embedded compute shader loads independently.
        let shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(
                include_str!("shaders/custom.wgsl"),
                "custom-fluid.wgsl",
            ));
        app.sub_app_mut(RenderApp)
            .insert_resource(CustomShader(shader))
            .init_resource::<Targets>()
            .init_resource::<Draws>()
            .add_systems(ExtractSchedule, extract_targets)
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                prepare
                    .in_set(RenderSystems::PrepareBindGroups)
                    .after(Fluid2dSystems::Plan),
            )
            // This set is ordered after simulation and before Bevy cameras.
            .add_systems(RenderGraph, draw.in_set(Fluid2dSystems::Render));
    }
}
#[derive(Resource)]
struct CustomShader(Handle<Shader>);
#[derive(Resource, Default)]
struct Targets(HashMap<Entity, Handle<Image>>);
#[derive(Resource, Default)]
struct Draws(Vec<(Handle<Image>, BindGroup, u32)>);
#[derive(Resource)]
struct CustomPipeline {
    layout: BindGroupLayoutDescriptor,
    id: CachedRenderPipelineId,
}

fn extract_targets(mut targets: ResMut<Targets>, query: Extract<Query<(Entity, &Fluid2dSurface)>>) {
    targets.0 = query.iter().map(|(e, s)| (e, s.image.clone())).collect();
}

fn init_pipeline(mut commands: Commands, cache: Res<PipelineCache>, shader: Res<CustomShader>) {
    let layout = BindGroupLayoutDescriptor::new(
        "custom fluid",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (
                storage_buffer_read_only_sized(false, BufferSize::new(16)),
                uniform_buffer_sized(false, BufferSize::new(16)),
                storage_buffer_read_only_sized(false, BufferSize::new(4)),
            ),
        ),
    );
    let id = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("application fluid renderer".into()),
        layout: vec![layout.clone()],
        vertex: VertexState {
            shader: shader.0.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader: shader.0.clone(),
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });
    commands.insert_resource(CustomPipeline { layout, id });
}

fn prepare(
    mut draws: ResMut<Draws>,
    targets: Res<Targets>,
    fluids: Res<GpuFluids2d>,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    pipeline: Res<CustomPipeline>,
) {
    draws.0.clear();
    for (&entity, target) in &targets.0 {
        let Some(fluid) = fluids.get(entity) else {
            continue;
        };
        let b = fluid.settings().bounds;
        let uniform = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("custom bounds"),
            contents: bytemuck::cast_slice(&[b.min.x, b.min.y, b.max.x, b.max.y]),
            usage: BufferUsages::UNIFORM,
        });
        // Rebuilding here also handles resets that replace the particle buffer.
        let group = device.create_bind_group(
            "custom fluid",
            &cache.get_bind_group_layout(&pipeline.layout),
            &BindGroupEntries::sequential((
                fluid.particle_buffer().as_entire_binding(),
                uniform.as_entire_binding(),
                fluid.particle_activity_buffer().as_entire_binding(),
            )),
        );
        draws
            .0
            .push((target.clone(), group, fluid.particle_count()));
    }
}

fn draw(
    mut context: RenderContext,
    draws: Res<Draws>,
    images: Res<RenderAssets<GpuImage>>,
    cache: Res<PipelineCache>,
    pipeline: Res<CustomPipeline>,
) {
    let Some(pipeline) = cache.get_render_pipeline(pipeline.id) else {
        return;
    };
    for (target, group, count) in &draws.0 {
        let Some(image) = images.get(target) else {
            continue;
        };
        let attachments = [Some(RenderPassColorAttachment {
            view: &image.texture_view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(default()),
                store: StoreOp::Store,
            },
        })];
        let mut pass = context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("application-owned fluid pass"),
                color_attachments: &attachments,
                ..default()
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..6, 0..*count);
    }
}
