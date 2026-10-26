use std::collections::HashMap;

use bevy::{
    asset::{RenderAssetUsages, embedded_asset},
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},
        texture::GpuImage,
    },
    shader::ShaderCacheError,
};
use bytemuck::{Pod, Zeroable};

use super::{Fluid2d, Fluid2dError, Fluid2dSystems, GpuFluids2d};

/// Appearance of the default solid-color density-field liquid surface.
#[derive(Clone, Debug)]
pub struct Fluid2dStyle {
    pub color: LinearRgba,
    pub particle_radius: f32,
    /// Retained for source compatibility; ignored by the solid-color renderer.
    pub speed_color_scale: f32,
    /// Isovalue of the summed render kernels. Lower values connect more easily.
    pub surface_threshold: f32,
    /// Kernel support radius relative to particle_radius; independent from SPH.
    pub smoothing_factor: f32,
}

impl Default for Fluid2dStyle {
    fn default() -> Self {
        Self {
            color: LinearRgba::new(0.03, 0.35, 0.9, 1.0),
            particle_radius: 0.095,
            speed_color_scale: 0.0,
            surface_threshold: 0.18,
            smoothing_factor: 2.5,
        }
    }
}

/// Offscreen render target attached to the simulation entity. Reuse `image` in
/// sprites, UI or your own materials. Removing this component disables its render pass.
/// Custom renderers may also use this type without installing `Fluid2dRenderPlugin`.
#[derive(Component, Clone)]
pub struct Fluid2dSurface {
    pub image: Handle<Image>,
    pub style: Fluid2dStyle,
}

impl Fluid2dSurface {
    pub fn new(images: &mut Assets<Image>, resolution: UVec2) -> Result<Self, Fluid2dError> {
        if resolution.min_element() == 0 || resolution.max_element() > 8192 {
            return Err(Fluid2dError(
                "surface resolution must be between 1 and 8192 on each axis",
            ));
        }
        let mut image =
            Image::new_target_texture(resolution.x, resolution.y, TextureFormat::Rgba8Unorm, None);
        image.asset_usage = RenderAssetUsages::RENDER_WORLD;
        image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
            | TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::COPY_SRC;
        Ok(Self {
            image: images.add(image),
            style: default(),
        })
    }

    pub fn sprite(&self, size: Vec2) -> Sprite {
        Sprite {
            image: self.image.clone(),
            custom_size: Some(size),
            ..default()
        }
    }
}

/// Ready-to-display simulation and sprite. Spawn a `Camera2d` separately.
/// Transform affects display only; physics uses `Fluid2dSettings::bounds`.
#[derive(Bundle)]
pub struct Fluid2dBundle {
    pub fluid: Fluid2d,
    pub surface: Fluid2dSurface,
    pub sprite: Sprite,
    pub transform: Transform,
}

impl Fluid2dBundle {
    pub fn new(
        fluid: Fluid2d,
        images: &mut Assets<Image>,
        resolution: UVec2,
    ) -> Result<Self, Fluid2dError> {
        let surface = Fluid2dSurface::new(images, resolution)?;
        let bounds = fluid.settings().bounds;
        let sprite = surface.sprite(bounds.size());
        Ok(Self {
            fluid,
            surface,
            sprite,
            transform: Transform::from_translation(bounds.center().extend(0.0)),
        })
    }
}

/// Optional default render pass. Add after `Fluid2dPlugin`, or use `Fluid2dPlugins`.
pub struct Fluid2dRenderPlugin;

impl Plugin for Fluid2dRenderPlugin {
    fn build(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<super::Fluid2dPlugin>(),
            "add Fluid2dPlugin before Fluid2dRenderPlugin"
        );
        embedded_asset!(app, "shaders/particles.wgsl");
        embedded_asset!(app, "shaders/surface.wgsl");
        app.sub_app_mut(RenderApp)
            .init_resource::<ExtractedSurfaces>()
            .init_resource::<PreparedSurfaces>()
            .add_systems(ExtractSchedule, extract_surfaces)
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                prepare_surfaces
                    .in_set(RenderSystems::PrepareBindGroups)
                    .after(Fluid2dSystems::Plan),
            )
            .add_systems(RenderGraph, render_surfaces.in_set(Fluid2dSystems::Render));
    }
}

#[derive(Resource, Default)]
struct ExtractedSurfaces(HashMap<Entity, Fluid2dSurface>);

fn extract_surfaces(
    mut surfaces: ResMut<ExtractedSurfaces>,
    query: Extract<Query<(Entity, &Fluid2dSurface), With<Fluid2d>>>,
) {
    surfaces.0.clear();
    surfaces.0.extend(query.iter().map(|(e, s)| (e, s.clone())));
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawParams {
    bounds: [f32; 4],
    color: [f32; 4],
    visual: [f32; 4],
}
#[derive(Resource)]
struct DrawPipeline {
    layout: BindGroupLayoutDescriptor,
    field_layout: BindGroupLayoutDescriptor,
    accumulate: CachedRenderPipelineId,
    composite: CachedRenderPipelineId,
}
fn init_pipeline(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let layout = BindGroupLayoutDescriptor::new(
        "fluid2d density splats",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                storage_buffer_read_only_sized(false, BufferSize::new(16))
                    .visibility(ShaderStages::VERTEX),
                uniform_buffer_sized(false, BufferSize::new(48)),
                storage_buffer_read_only_sized(false, BufferSize::new(4))
                    .visibility(ShaderStages::VERTEX),
            ),
        ),
    );
    let field_layout = BindGroupLayoutDescriptor::new(
        "fluid2d isosurface",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                uniform_buffer_sized(false, BufferSize::new(48)),
            ),
        ),
    );
    let splat = assets.load("embedded://bevy_gpu_fluid/fluid2d/shaders/particles.wgsl");
    let surface = assets.load("embedded://bevy_gpu_fluid/fluid2d/shaders/surface.wgsl");
    let accumulate = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("fluid2d density accumulation".into()),
        layout: vec![layout.clone()],
        vertex: VertexState {
            shader: splat.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader: splat,
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: Some(additive_blend()),
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });
    let composite = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("fluid2d liquid surface".into()),
        layout: vec![field_layout.clone()],
        vertex: VertexState {
            shader: surface.clone(),
            entry_point: Some("fullscreen".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader: surface,
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
    commands.insert_resource(DrawPipeline {
        layout,
        field_layout,
        accumulate,
        composite,
    });
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
struct DensityField {
    _texture: Texture,
    view: TextureView,
    group: BindGroup,
    size: UVec2,
}
struct PreparedSurface {
    image: Handle<Image>,
    uniform: Buffer,
    particle_buffer_id: BufferId,
    bind_group: BindGroup,
    count: u32,
    field: Option<DensityField>,
}
#[derive(Resource, Default)]
struct PreparedSurfaces(HashMap<Entity, PreparedSurface>);
fn prepare_surfaces(
    mut prepared: ResMut<PreparedSurfaces>,
    surfaces: Res<ExtractedSurfaces>,
    fluids: Res<GpuFluids2d>,
    images: Res<RenderAssets<GpuImage>>,
    (device, queue): (Res<RenderDevice>, Res<RenderQueue>),
    cache: Res<PipelineCache>,
    pipeline: Res<DrawPipeline>,
) {
    prepared
        .0
        .retain(|e, _| surfaces.0.contains_key(e) && fluids.get(*e).is_some());
    for (&entity, surface) in &surfaces.0 {
        let Some(fluid) = fluids.get(entity) else {
            continue;
        };
        let style = &surface.style;
        if !style.particle_radius.is_finite()
            || style.particle_radius <= 0.0
            || !style.surface_threshold.is_finite()
            || !(0.0..1.0).contains(&style.surface_threshold)
            || style.surface_threshold == 0.0
            || !style.smoothing_factor.is_finite()
            || !(1.0..=8.0).contains(&style.smoothing_factor)
            || !style.color.to_f32_array().iter().all(|v| v.is_finite())
        {
            prepared.0.remove(&entity);
            continue;
        }
        let b = fluid.settings().bounds;
        let params = DrawParams {
            bounds: [b.min.x, b.min.y, b.max.x, b.max.y],
            color: style.color.to_f32_array(),
            visual: [
                style.particle_radius,
                0.0,
                style.surface_threshold,
                style.smoothing_factor,
            ],
        };
        let needs_buffer = prepared
            .0
            .get(&entity)
            .is_none_or(|p| p.particle_buffer_id != fluid.particle_buffer().id());
        if needs_buffer {
            let uniform = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("fluid2d surface params"),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });
            let bind_group = device.create_bind_group(
                "fluid2d splats",
                &cache.get_bind_group_layout(&pipeline.layout),
                &BindGroupEntries::sequential((
                    fluid.particle_buffer().as_entire_binding(),
                    uniform.as_entire_binding(),
                    fluid.particle_activity_buffer().as_entire_binding(),
                )),
            );
            prepared.0.insert(
                entity,
                PreparedSurface {
                    image: surface.image.clone(),
                    uniform,
                    particle_buffer_id: fluid.particle_buffer().id(),
                    bind_group,
                    count: fluid.particle_count(),
                    field: None,
                },
            );
        }
        let p = prepared.0.get_mut(&entity).unwrap();
        queue.write_buffer(&p.uniform, 0, bytemuck::bytes_of(&params));
        p.image = surface.image.clone();
        p.count = fluid.particle_count();
        let Some(image) = images.get(&surface.image) else {
            continue;
        };
        let size = UVec2::new(
            image.texture_descriptor.size.width,
            image.texture_descriptor.size.height,
        );
        if p.field.as_ref().is_some_and(|f| f.size == size) {
            continue;
        }
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("fluid2d density field"),
            size: Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&default());
        let group = device.create_bind_group(
            "fluid2d surface",
            &cache.get_bind_group_layout(&pipeline.field_layout),
            &BindGroupEntries::sequential((&view, p.uniform.as_entire_binding())),
        );
        p.field = Some(DensityField {
            _texture: texture,
            view,
            group,
            size,
        });
    }
}
fn render_surfaces(
    mut context: RenderContext,
    prepared: Res<PreparedSurfaces>,
    images: Res<RenderAssets<GpuImage>>,
    pipeline: Res<DrawPipeline>,
    cache: Res<PipelineCache>,
    mut reported_error: Local<bool>,
) {
    let (Some(accumulate), Some(composite)) = (
        cache.get_render_pipeline(pipeline.accumulate),
        cache.get_render_pipeline(pipeline.composite),
    ) else {
        for id in [pipeline.accumulate, pipeline.composite] {
            if let CachedPipelineState::Err(err) = cache.get_render_pipeline_state(id)
                && !matches!(
                    err,
                    ShaderCacheError::ShaderNotLoaded(_)
                        | ShaderCacheError::ShaderImportNotYetAvailable
                )
                && !*reported_error
            {
                error!("Fluid2d surface pipeline: {err}");
                *reported_error = true;
            }
        }
        return;
    };
    for p in prepared.0.values() {
        let (Some(image), Some(field)) = (images.get(&p.image), &p.field) else {
            continue;
        };
        if image.texture_descriptor.format != TextureFormat::Rgba8Unorm
            || !image
                .texture_descriptor
                .usage
                .contains(TextureUsages::RENDER_ATTACHMENT)
        {
            continue;
        }
        let attachments = [Some(RenderPassColorAttachment {
            view: &field.view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(default()),
                store: StoreOp::Store,
            },
        })];
        {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("fluid2d density field"),
                    color_attachments: &attachments,
                    ..default()
                });
            pass.set_pipeline(accumulate);
            pass.set_bind_group(0, &p.bind_group, &[]);
            pass.draw(0..6, 0..p.count);
        }
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
                label: Some("fluid2d isosurface shading"),
                color_attachments: &attachments,
                ..default()
            });
        pass.set_pipeline(composite);
        pass.set_bind_group(0, &field.group, &[]);
        pass.draw(0..3, 0..1);
    }
}
