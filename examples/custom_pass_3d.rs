//! Application-owned Camera3d render pass. No library render plugin or SPH code copied.
use bevy::{
    core_pipeline::{
        Core3dSystems,
        core_3d::{CORE_3D_DEPTH_FORMAT, main_opaque_pass_3d, main_transparent_pass_3d},
        schedule::Core3d,
    },
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        camera::ExtractedCamera,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, ViewQuery},
        view::{ExtractedView, ViewDepthStencilTexture, ViewTarget},
    },
};
use bevy_gpu_fluid::{
    fluid3d::{Fluid3dSystems, GpuFluids3d},
    prelude::*,
};
use std::collections::HashMap;
#[path = "support/scene3d.rs"]
pub(crate) mod scene3d;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.018, 0.025, 0.045)))
        .add_plugins((DefaultPlugins, Fluid3dPlugin, CustomFluid3dRenderer))
        .add_systems(Startup, setup)
        .add_systems(Update, scene3d::controls)
        .run();
}
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let obstacles = scene3d::scene(&mut commands, &mut meshes, &mut materials);
    let mut fluid =
        Fluid3d::from_faucet(Fluid3dSettings::default(), 8000, Fluid3dFaucet::default()).unwrap();
    fluid.set_colliders(obstacles).unwrap();
    commands.spawn(fluid);
}

pub(crate) struct CustomFluid3dRenderer;
impl Plugin for CustomFluid3dRenderer {
    fn build(&self, app: &mut App) {
        let shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(
                include_str!("shaders/custom_3d.wgsl"),
                "custom-fluid-3d.wgsl",
            ));
        let layout = BindGroupLayoutDescriptor::new(
            "application fluid3d",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (
                    storage_buffer_read_only_sized(false, BufferSize::new(32)),
                    uniform_buffer_sized(false, BufferSize::new(128)),
                    storage_buffer_read_only_sized(false, BufferSize::new(4)),
                ),
            ),
        );
        app.sub_app_mut(RenderApp)
            .insert_resource(CustomPipeline {
                shader,
                layout,
                variants: default(),
            })
            .add_systems(
                Render,
                prepare
                    .in_set(RenderSystems::PrepareBindGroups)
                    .after(Fluid3dSystems::Plan),
            )
            .configure_sets(
                Core3d,
                Fluid3dSystems::Render
                    .in_set(Core3dSystems::MainPass)
                    .after(main_opaque_pass_3d)
                    .before(main_transparent_pass_3d),
            )
            .add_systems(Core3d, draw.in_set(Fluid3dSystems::Render));
    }
}
#[derive(Resource)]
struct CustomPipeline {
    shader: Handle<Shader>,
    layout: BindGroupLayoutDescriptor,
    variants: HashMap<(TextureFormat, u32), CachedRenderPipelineId>,
}
#[derive(Component)]
struct Draws {
    pipeline: CachedRenderPipelineId,
    groups: Vec<(BindGroup, u32)>,
}

fn prepare(
    mut commands: Commands,
    fluids: Res<GpuFluids3d>,
    views: Query<(Entity, &ExtractedView, &ViewTarget, &Msaa), With<Camera3d>>,
    mut pipeline: ResMut<CustomPipeline>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
) {
    for (entity, view, target, msaa) in &views {
        let key = (target.main_texture_format(), msaa.samples());
        let id = if let Some(&id) = pipeline.variants.get(&key) {
            id
        } else {
            let shader = pipeline.shader.clone();
            let id = cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("application fluid3d".into()),
                layout: vec![pipeline.layout.clone()],
                vertex: VertexState {
                    shader: shader.clone(),
                    entry_point: Some("vertex".into()),
                    ..default()
                },
                fragment: Some(FragmentState {
                    shader,
                    entry_point: Some("fragment".into()),
                    targets: vec![Some(ColorTargetState {
                        format: key.0,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                    ..default()
                }),
                depth_stencil: Some(DepthStencilState {
                    format: CORE_3D_DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(CompareFunction::GreaterEqual),
                    stencil: default(),
                    bias: default(),
                }),
                multisample: MultisampleState {
                    count: key.1,
                    ..default()
                },
                ..default()
            });
            pipeline.variants.insert(key, id);
            id
        };
        let world = view.world_from_view.to_matrix();
        let clip = view
            .clip_from_world
            .unwrap_or_else(|| view.clip_from_view * world.inverse());
        let matrices = [clip.to_cols_array_2d(), world.to_cols_array_2d()];
        let uniform = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("application camera"),
            contents: bytemuck::cast_slice(&matrices),
            usage: BufferUsages::UNIFORM,
        });
        let groups = fluids
            .iter()
            .map(|(_, fluid)| {
                // Rebuilding bind groups here automatically handles reset/reallocation.
                let group = device.create_bind_group(
                    "application fluid3d",
                    &cache.get_bind_group_layout(&pipeline.layout),
                    &BindGroupEntries::sequential((
                        fluid.particle_buffer().as_entire_binding(),
                        uniform.as_entire_binding(),
                        fluid.particle_activity_buffer().as_entire_binding(),
                    )),
                );
                (group, fluid.particle_count())
            })
            .collect();
        commands.entity(entity).insert(Draws {
            pipeline: id,
            groups,
        });
    }
}
fn draw(
    view: ViewQuery<(
        &ViewTarget,
        &ViewDepthStencilTexture,
        &ExtractedCamera,
        &Draws,
    )>,
    cache: Res<PipelineCache>,
    mut context: RenderContext,
) {
    let (target, depth, camera, draws) = view.into_inner();
    let Some(pipeline) = cache.get_render_pipeline(draws.pipeline) else {
        return;
    };
    let attachments = [Some(target.get_color_attachment())];
    let mut pass = context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("application-owned fluid3d pass"),
            color_attachments: &attachments,
            depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
            ..default()
        });
    if let Some(v) = &camera.viewport {
        pass.set_viewport(
            v.physical_position.x as f32,
            v.physical_position.y as f32,
            v.physical_size.x as f32,
            v.physical_size.y as f32,
            v.depth.start,
            v.depth.end,
        );
    }
    pass.set_pipeline(pipeline);
    for (group, count) in &draws.groups {
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..6, 0..*count);
    }
}
