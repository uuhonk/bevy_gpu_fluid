use super::{Fluid3d, Fluid3dSystems, GpuFluids3d};
use bevy::{
    asset::embedded_asset,
    camera::{MainPassResolutionOverride, Viewport, visibility::RenderLayers},
    core_pipeline::{
        Core3dSystems,
        core_3d::{CORE_3D_DEPTH_FORMAT, main_opaque_pass_3d, main_transparent_pass_3d},
        schedule::Core3d,
    },
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
        camera::ExtractedCamera,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        view::{ExtractedView, ViewDepthStencilTexture, ViewTarget},
    },
    shader::ShaderCacheError,
};
use bytemuck::{Pod, Zeroable};
use std::collections::{HashMap, HashSet};

/// Solid-color screen-space liquid surface, with density merging and depth filtering.
/// Attach beside Fluid3d; positions are world coordinates. RenderLayers is supported.
#[derive(Component, Clone, Debug)]
pub struct Fluid3dRender {
    /// Linear RGBA, alpha blended once per reconstructed surface. Does not write scene depth.
    pub color: LinearRgba,
    pub particle_radius: f32,
    /// Retained for source compatibility; ignored by the solid-color renderer.
    pub speed_color_scale: f32,
    pub enabled: bool,
    /// Isovalue of the summed render kernels, independent of physical SPH density.
    pub surface_threshold: f32,
    /// Kernel support radius relative to particle_radius.
    pub smoothing_factor: f32,
    /// Bilateral filter radius in field pixels, 0..=8.
    pub blur_radius: f32,
    /// Retained for source compatibility; ignored by the solid-color renderer.
    pub absorption: f32,
    /// Retained for source compatibility; ignored by the solid-color renderer.
    pub roughness: f32,
    /// Internal screen-space field size relative to the camera viewport, 0.25..=1.
    pub resolution_scale: f32,
}
impl Default for Fluid3dRender {
    fn default() -> Self {
        Self {
            color: LinearRgba::new(0.04, 0.36, 0.56, 0.78),
            particle_radius: 0.09,
            speed_color_scale: 0.0,
            enabled: true,
            surface_threshold: 0.18,
            smoothing_factor: 2.5,
            blur_radius: 3.0,
            absorption: 0.0,
            roughness: 0.16,
            resolution_scale: 1.0,
        }
    }
}
impl Fluid3dRender {
    fn valid(&self) -> bool {
        self.enabled
            && self.particle_radius.is_finite()
            && self.particle_radius > 0.0
            && self.surface_threshold.is_finite()
            && self.surface_threshold > 0.0
            && self.surface_threshold < 1.0
            && self.smoothing_factor.is_finite()
            && (1.0..=8.0).contains(&self.smoothing_factor)
            && self.blur_radius.is_finite()
            && (0.0..=8.0).contains(&self.blur_radius)
            && self.resolution_scale.is_finite()
            && (0.25..=1.0).contains(&self.resolution_scale)
            && self.color.to_f32_array().iter().all(|v| v.is_finite())
            && self.color.alpha > 0.0
            && self.color.alpha <= 1.0
    }
}
/// Optional Camera3d liquid renderer; simulation and external GPU buffer API are unchanged.
pub struct Fluid3dRenderPlugin;
impl Plugin for Fluid3dRenderPlugin {
    fn build(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<super::Fluid3dPlugin>(),
            "add Fluid3dPlugin first"
        );
        embedded_asset!(app, "shaders/particles.wgsl");
        embedded_asset!(app, "shaders/surface.wgsl");
        app.sub_app_mut(RenderApp)
            .init_resource::<ExtractedStyles>()
            .init_resource::<PreparedSurfaces>()
            .add_systems(ExtractSchedule, extract_styles)
            .add_systems(RenderStartup, init_pipeline)
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
#[derive(Resource, Default)]
struct ExtractedStyles(HashMap<Entity, (Fluid3dRender, RenderLayers)>);
type StyleSources<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Fluid3dRender,
        Option<&'static RenderLayers>,
    ),
    With<Fluid3d>,
>;
fn extract_styles(mut styles: ResMut<ExtractedStyles>, query: Extract<StyleSources>) {
    styles.0.clear();
    styles.0.extend(
        query
            .iter()
            .filter(|(_, s, _)| s.valid())
            .map(|(e, s, l)| (e, (s.clone(), l.cloned().unwrap_or_default()))),
    );
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SurfaceParams {
    color: [f32; 4],
    visual: [f32; 4],
    optics: [f32; 4],
    size: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewParams {
    clip_from_world: [[f32; 4]; 4],
    world_from_view: [[f32; 4]; 4],
    view_from_world: [[f32; 4]; 4],
    view_from_clip: [[f32; 4]; 4],
    viewport: [f32; 4],
}
#[derive(Resource)]
struct Pipelines {
    particles: BindGroupLayoutDescriptor,
    view: BindGroupLayoutDescriptor,
    front: BindGroupLayoutDescriptor,
    surface: BindGroupLayoutDescriptor,
    depth: CachedRenderPipelineId,
    density: CachedRenderPipelineId,
    horizontal: CachedRenderPipelineId,
    vertical: CachedRenderPipelineId,
    shader: Handle<Shader>,
    variants: HashMap<(TextureFormat, u32), CachedRenderPipelineId>,
}
fn depth_state() -> DepthStencilState {
    DepthStencilState {
        format: CORE_3D_DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: Some(CompareFunction::GreaterEqual),
        stencil: default(),
        bias: default(),
    }
}
fn additive_blend() -> BlendState {
    let component = BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    };
    BlendState {
        color: component,
        alpha: component,
    }
}
fn init_pipeline(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let particles = BindGroupLayoutDescriptor::new(
        "fluid3d splats",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                storage_buffer_read_only_sized(false, BufferSize::new(32))
                    .visibility(ShaderStages::VERTEX),
                uniform_buffer_sized(false, BufferSize::new(64)),
                storage_buffer_read_only_sized(false, BufferSize::new(4))
                    .visibility(ShaderStages::VERTEX),
            ),
        ),
    );
    let view = BindGroupLayoutDescriptor::new(
        "fluid3d camera",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (uniform_buffer_sized(false, BufferSize::new(272)),),
        ),
    );
    let front = BindGroupLayoutDescriptor::new(
        "fluid3d front depth",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (texture_2d(TextureSampleType::Float { filterable: false }),),
        ),
    );
    let surface = BindGroupLayoutDescriptor::new(
        "fluid3d surface fields",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                uniform_buffer_sized(false, BufferSize::new(64)),
                uniform_buffer_sized(false, BufferSize::new(272)),
            ),
        ),
    );
    let splats = assets.load("embedded://bevy_gpu_fluid/fluid3d/shaders/particles.wgsl");
    let shader = assets.load("embedded://bevy_gpu_fluid/fluid3d/shaders/surface.wgsl");
    let depth = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("fluid3d nearest surface".into()),
        layout: vec![particles.clone(), view.clone()],
        vertex: VertexState {
            shader: splats.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader: splats.clone(),
            entry_point: Some("depth".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::R32Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        depth_stencil: Some(depth_state()),
        ..default()
    });
    let density = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("fluid3d density".into()),
        layout: vec![particles.clone(), view.clone(), front.clone()],
        vertex: VertexState {
            shader: splats.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader: splats,
            entry_point: Some("density".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: Some(additive_blend()),
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });
    let [horizontal, vertical] = ["horizontal", "vertical"].map(|entry| {
        cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some(format!("fluid3d bilateral {entry}").into()),
            layout: vec![surface.clone()],
            vertex: VertexState {
                shader: shader.clone(),
                entry_point: Some("fullscreen".into()),
                ..default()
            },
            fragment: Some(FragmentState {
                shader: shader.clone(),
                entry_point: Some(entry.into()),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::R32Float,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        })
    });
    commands.insert_resource(Pipelines {
        particles,
        view,
        front,
        surface,
        depth,
        density,
        horizontal,
        vertical,
        shader,
        variants: default(),
    });
}
struct CameraState {
    uniform: Buffer,
    group: BindGroup,
    pipeline: CachedRenderPipelineId,
}
struct Surface {
    _textures: [Texture; 5],
    nearest: TextureView,
    depth: TextureView,
    density: TextureView,
    horizontal: TextureView,
    vertical: TextureView,
    uniform: Buffer,
    particles: BindGroup,
    front: BindGroup,
    filter_h: BindGroup,
    filter_v: BindGroup,
    composite: BindGroup,
    size: UVec2,
    buffer_id: BufferId,
    count: u32,
    sort_depth: f32,
}
#[derive(Resource, Default)]
struct PreparedSurfaces {
    cameras: HashMap<Entity, CameraState>,
    surfaces: HashMap<(Entity, Entity), Surface>,
}
type Cameras<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static ExtractedView,
        &'static ViewTarget,
        &'static Msaa,
        &'static ExtractedCamera,
        Option<&'static RenderLayers>,
        Option<&'static MainPassResolutionOverride>,
    ),
    With<Camera3d>,
>;
fn texture(
    device: &RenderDevice,
    label: &'static str,
    size: UVec2,
    format: TextureFormat,
) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}
fn create_surface(
    device: &RenderDevice,
    cache: &PipelineCache,
    pipelines: &Pipelines,
    params: &SurfaceParams,
    particle_buffers: (&Buffer, &Buffer),
    camera: &Buffer,
    count: u32,
) -> Surface {
    let (particle_buffer, activity) = particle_buffers;
    let size = UVec2::new(params.size[0] as u32, params.size[1] as u32);
    let textures = [
        texture(device, "fluid3d nearest", size, TextureFormat::R32Float),
        texture(device, "fluid3d splat depth", size, CORE_3D_DEPTH_FORMAT),
        texture(device, "fluid3d density", size, TextureFormat::Rgba16Float),
        texture(device, "fluid3d horizontal", size, TextureFormat::R32Float),
        texture(device, "fluid3d vertical", size, TextureFormat::R32Float),
    ];
    let [nearest, depth, density, horizontal, vertical] =
        textures.each_ref().map(|t| t.create_view(&default()));
    let uniform = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("fluid3d surface parameters"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });
    let particles = device.create_bind_group(
        "fluid3d splats",
        &cache.get_bind_group_layout(&pipelines.particles),
        &BindGroupEntries::sequential((
            particle_buffer.as_entire_binding(),
            uniform.as_entire_binding(),
            activity.as_entire_binding(),
        )),
    );
    let front = device.create_bind_group(
        "fluid3d front",
        &cache.get_bind_group_layout(&pipelines.front),
        &BindGroupEntries::single(&vertical),
    );
    let group = |input: &TextureView| {
        device.create_bind_group(
            "fluid3d surface",
            &cache.get_bind_group_layout(&pipelines.surface),
            &BindGroupEntries::sequential((
                input,
                &density,
                uniform.as_entire_binding(),
                camera.as_entire_binding(),
            )),
        )
    };
    let filter_h = group(&nearest);
    let filter_v = group(&horizontal);
    let composite = group(&vertical);
    Surface {
        _textures: textures,
        nearest,
        depth,
        density,
        horizontal,
        vertical,
        uniform,
        particles,
        front,
        filter_h,
        filter_v,
        composite,
        size,
        buffer_id: particle_buffer.id(),
        count,
        sort_depth: 0.0,
    }
}
fn prepare(
    mut prepared: ResMut<PreparedSurfaces>,
    styles: Res<ExtractedStyles>,
    fluids: Res<GpuFluids3d>,
    cameras: Cameras,
    mut pipelines: ResMut<Pipelines>,
    cache: Res<PipelineCache>,
    (device, queue): (Res<RenderDevice>, Res<RenderQueue>),
) {
    let mut alive_cameras = HashSet::new();
    let mut alive_surfaces = HashSet::new();
    for (entity, view, target, msaa, camera, layers, resolution) in &cameras {
        alive_cameras.insert(entity);
        let key = (target.main_texture_format(), msaa.samples());
        let id = if let Some(&id) = pipelines.variants.get(&key) {
            id
        } else {
            let shader = pipelines.shader.clone();
            let id = cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("fluid3d liquid composite".into()),
                layout: vec![pipelines.surface.clone()],
                vertex: VertexState {
                    shader: shader.clone(),
                    entry_point: Some("fullscreen".into()),
                    ..default()
                },
                fragment: Some(FragmentState {
                    shader,
                    entry_point: Some("composite".into()),
                    targets: vec![Some(ColorTargetState {
                        format: key.0,
                        blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                    ..default()
                }),
                depth_stencil: Some(DepthStencilState {
                    depth_write_enabled: Some(false),
                    ..depth_state()
                }),
                multisample: MultisampleState {
                    count: key.1,
                    ..default()
                },
                ..default()
            });
            pipelines.variants.insert(key, id);
            id
        };
        let viewport = Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution);
        let size = viewport
            .as_ref()
            .map(|v| v.physical_size)
            .unwrap_or_else(|| {
                let s = target.main_texture().size();
                UVec2::new(s.width, s.height)
            });
        let origin = viewport
            .as_ref()
            .map(|v| v.physical_position)
            .unwrap_or(UVec2::ZERO);
        let world = view.world_from_view.to_matrix();
        let inverse = world.inverse();
        let clip = view
            .clip_from_world
            .unwrap_or_else(|| view.clip_from_view * inverse);
        let params = ViewParams {
            clip_from_world: clip.to_cols_array_2d(),
            world_from_view: world.to_cols_array_2d(),
            view_from_world: inverse.to_cols_array_2d(),
            view_from_clip: (clip * world).inverse().to_cols_array_2d(),
            viewport: [
                origin.x as f32,
                origin.y as f32,
                size.x as f32,
                size.y as f32,
            ],
        };
        if let Some(c) = prepared.cameras.get_mut(&entity) {
            queue.write_buffer(&c.uniform, 0, bytemuck::bytes_of(&params));
            c.pipeline = id;
        } else {
            let uniform = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("fluid3d camera"),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });
            let group = device.create_bind_group(
                "fluid3d camera",
                &cache.get_bind_group_layout(&pipelines.view),
                &BindGroupEntries::single(uniform.as_entire_binding()),
            );
            prepared.cameras.insert(
                entity,
                CameraState {
                    uniform,
                    group,
                    pipeline: id,
                },
            );
        }
        let layers = layers.cloned().unwrap_or_default();
        for (&fluid_entity, (style, fluid_layers)) in &styles.0 {
            let Some(fluid) = fluids.get(fluid_entity) else {
                continue;
            };
            if fluid.particle_count() == 0 || !fluid_layers.intersects(&layers) {
                continue;
            }
            let key = (entity, fluid_entity);
            alive_surfaces.insert(key);
            let field_size = (size.as_vec2() * style.resolution_scale)
                .ceil()
                .as_uvec2()
                .max(UVec2::ONE);
            let params = SurfaceParams {
                color: style.color.to_f32_array(),
                visual: [
                    style.particle_radius,
                    0.0,
                    style.surface_threshold,
                    style.smoothing_factor,
                ],
                optics: [0.0, 0.0, style.blur_radius, style.resolution_scale],
                size: [field_size.x as f32, field_size.y as f32, 0.0, 0.0],
            };
            if prepared
                .surfaces
                .get(&key)
                .is_none_or(|s| s.size != field_size)
            {
                let surface = create_surface(
                    &device,
                    &cache,
                    &pipelines,
                    &params,
                    (fluid.particle_buffer(), fluid.particle_activity_buffer()),
                    &prepared.cameras[&entity].uniform,
                    fluid.particle_count(),
                );
                prepared.surfaces.insert(key, surface);
            }
            let surface = prepared.surfaces.get_mut(&key).unwrap();
            queue.write_buffer(&surface.uniform, 0, bytemuck::bytes_of(&params));
            surface.count = fluid.particle_count();
            // No readback: approximate inter-instance ordering by simulation bounds.
            surface.sort_depth = -inverse.transform_point3(fluid.settings().bounds.center()).z;
            if surface.buffer_id != fluid.particle_buffer().id() {
                surface.particles = device.create_bind_group(
                    "fluid3d rebound splats",
                    &cache.get_bind_group_layout(&pipelines.particles),
                    &BindGroupEntries::sequential((
                        fluid.particle_buffer().as_entire_binding(),
                        surface.uniform.as_entire_binding(),
                        fluid.particle_activity_buffer().as_entire_binding(),
                    )),
                );
                surface.buffer_id = fluid.particle_buffer().id();
            }
        }
    }
    prepared.cameras.retain(|e, _| alive_cameras.contains(e));
    prepared.surfaces.retain(|k, _| alive_surfaces.contains(k));
}
type FluidCamera = (
    &'static ViewTarget,
    &'static ViewDepthStencilTexture,
    &'static ExtractedCamera,
    Option<&'static MainPassResolutionOverride>,
);
fn draw(
    view: ViewQuery<FluidCamera>,
    prepared: Res<PreparedSurfaces>,
    pipelines: Res<Pipelines>,
    cache: Res<PipelineCache>,
    mut context: RenderContext,
    mut reported: Local<bool>,
) {
    let entity = view.entity();
    let (target, depth, camera, resolution) = view.into_inner();
    let Some(camera_state) = prepared.cameras.get(&entity) else {
        return;
    };
    let ids = [
        pipelines.depth,
        pipelines.density,
        pipelines.horizontal,
        pipelines.vertical,
        camera_state.pipeline,
    ];
    let passes: Option<Vec<_>> = ids
        .iter()
        .map(|id| cache.get_render_pipeline(*id))
        .collect();
    let Some(passes) = passes else {
        for id in ids {
            if let CachedPipelineState::Err(err) = cache.get_render_pipeline_state(id)
                && !matches!(
                    err,
                    ShaderCacheError::ShaderNotLoaded(_)
                        | ShaderCacheError::ShaderImportNotYetAvailable
                )
                && !*reported
            {
                error!("Fluid3d surface pipeline: {err}");
                *reported = true;
            }
        }
        return;
    };
    let mut surfaces: Vec<_> = prepared
        .surfaces
        .iter()
        .filter(|((camera_entity, _), _)| *camera_entity == entity)
        .collect();
    surfaces.sort_by(|((_, a_entity), a), ((_, b_entity), b)| {
        b.sort_depth
            .total_cmp(&a.sort_depth)
            .then_with(|| a_entity.to_bits().cmp(&b_entity.to_bits()))
    });
    for (_, s) in surfaces {
        let attachment = |view| {
            Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(default()),
                    store: StoreOp::Store,
                },
            })
        };
        let colors = [attachment(&s.nearest)];
        {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("fluid3d front depth"),
                    color_attachments: &colors,
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &s.depth,
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(0.0),
                            store: StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..default()
                });
            pass.set_pipeline(passes[0]);
            pass.set_bind_group(0, &s.particles, &[]);
            pass.set_bind_group(1, &camera_state.group, &[]);
            pass.draw(0..6, 0..s.count);
        }
        for (index, output, group) in [
            (2, &s.horizontal, &s.filter_h),
            (3, &s.vertical, &s.filter_v),
        ] {
            let colors = [attachment(output)];
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("fluid3d bilateral depth"),
                    color_attachments: &colors,
                    ..default()
                });
            pass.set_pipeline(passes[index]);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        let colors = [attachment(&s.density)];
        {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("fluid3d density"),
                    color_attachments: &colors,
                    ..default()
                });
            pass.set_pipeline(passes[1]);
            pass.set_bind_group(0, &s.particles, &[]);
            pass.set_bind_group(1, &camera_state.group, &[]);
            pass.set_bind_group(2, &s.front, &[]);
            pass.draw(0..6, 0..s.count);
        }
        let colors = [Some(target.get_color_attachment())];
        let mut pass = context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("fluid3d liquid shading"),
                color_attachments: &colors,
                depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
                ..default()
            });
        if let Some(v) = Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution)
        {
            pass.set_viewport(
                v.physical_position.x as f32,
                v.physical_position.y as f32,
                v.physical_size.x as f32,
                v.physical_size.y as f32,
                v.depth.start,
                v.depth.end,
            );
        }
        pass.set_pipeline(passes[4]);
        pass.set_bind_group(0, &s.composite, &[]);
        pass.draw(0..3, 0..1);
    }
}
