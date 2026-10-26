use bevy::{
    app::PluginsState,
    camera::{RenderTarget, ScalingMode, visibility::RenderLayers},
    prelude::*,
    render::{
        RenderApp, RenderPlugin,
        pipelined_rendering::PipelinedRenderingPlugin,
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        texture::GpuImage,
        view::Tonemapping,
    },
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_gpu_fluid::{fluid3d::GpuFluids3d, prelude::*};
#[path = "support/capture.rs"]
mod capture;
use std::time::Duration;
#[allow(dead_code)]
#[path = "../examples/custom_pass_3d.rs"]
mod custom_example;
use custom_example::scene3d as example_scene;

fn headless_app<M>(renderer: impl bevy::app::Plugins<M>) -> App {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .set(RenderPlugin {
                synchronous_pipeline_compilation: true,
                ..default()
            })
            .disable::<bevy::log::LogPlugin>()
            .disable::<WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>(),
    )
    .add_plugins((Fluid2dPlugins, Fluid3dPlugin, renderer))
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
        1.0 / 60.0,
    )));
    while app.plugins_state() != PluginsState::Ready {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        std::thread::sleep(Duration::from_millis(5));
    }
    app.finish();
    app.cleanup();
    app
}
fn frames(app: &mut App, count: usize) {
    for _ in 0..count {
        app.update();
        std::thread::sleep(Duration::from_millis(3));
    }
}
fn camera_target(app: &mut App, samples: Msaa, layer: usize) -> (Entity, Handle<Image>) {
    let mut image = Image::new_target_texture(256, 256, TextureFormat::Rgba8UnormSrgb, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let image = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    let camera = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            Camera {
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            RenderTarget::Image(image.clone().into()),
            samples,
            Tonemapping::None,
            RenderLayers::layer(layer),
            Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::FixedVertical {
                    viewport_height: 4.0,
                },
                near: 0.1,
                far: 100.0,
                ..OrthographicProjection::default_3d()
            }),
        ))
        .id();
    (camera, image)
}
fn pixels(app: &App, image: &Handle<Image>) -> Vec<u8> {
    let render = app.sub_app(RenderApp).world();
    let image = render
        .resource::<RenderAssets<GpuImage>>()
        .get(image)
        .unwrap();
    let device = render.resource::<RenderDevice>();
    let size = image.texture_descriptor.size;
    let row_bytes = (size.width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: row_bytes as u64 * size.height as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&default());
    encoder.copy_texture_to_buffer(
        image.texture.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(size.height),
            },
        },
        size,
    );
    render.resource::<RenderQueue>().submit([encoder.finish()]);
    map_buffer(device, &buffer)
        .chunks_exact(row_bytes as usize)
        .flat_map(|row| row[..size.width as usize * 4].iter().copied())
        .collect()
}
fn map_buffer(device: &RenderDevice, buffer: &Buffer) -> Vec<u8> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    buffer.slice(..).get_mapped_range().unwrap().to_vec()
}
fn particles(app: &App, entity: Entity) -> Vec<FluidParticle3d> {
    let render = app.sub_app(RenderApp).world();
    let gpu = render.resource::<GpuFluids3d>().get(entity).unwrap();
    let device = render.resource::<RenderDevice>();
    let size = gpu.particle_capacity().max(1) as u64 * 32;
    let buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&default());
    encoder.copy_buffer_to_buffer(gpu.particle_buffer(), 0, &buffer, 0, size);
    render.resource::<RenderQueue>().submit([encoder.finish()]);
    let bytes = map_buffer(device, &buffer);
    bytemuck::cast_slice::<u8, FluidParticle3d>(&bytes)[..gpu.particle_count() as usize].to_vec()
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn finite_3d_faucet_budget_reset_live_colliders_and_2d_coexistence() {
    let mut app = headless_app(Fluid3dRenderPlugin);
    let other = app
        .world_mut()
        .spawn(
            Fluid2d::new(
                Fluid2dSettings::default(),
                vec![FluidParticle2d::new(Vec2::ZERO)],
            )
            .unwrap(),
        )
        .id();
    let settings = Fluid3dSettings {
        gravity: Vec3::ZERO,
        stiffness: 0.0,
        viscosity: 0.0,
        ..default()
    };
    let faucet = Fluid3dFaucet {
        position: Vec3::new(0.0, 1.0, 0.0),
        width: 0.8,
        depth: 0.4,
        particle_spacing: 0.2,
        layers_per_second: 30.0,
        velocity: Vec3::ZERO,
        is_open: false,
    };
    let entity = app
        .world_mut()
        .spawn(Fluid3d::from_faucet(settings, 19, faucet).unwrap())
        .id();
    frames(&mut app, 30);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    assert!(state.simulated_steps() > 0);
    assert_eq!(state.particle_count(), 0);
    assert_eq!(state.particle_capacity(), 19);
    let buffer_id = state.particle_buffer().id();
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_faucet_open(true)
        .unwrap();
    for _ in 0..20 {
        app.update();
        if !particles(&app, entity).is_empty() {
            break;
        }
    }
    let released = particles(&app, entity);
    assert_eq!(released.len(), 8);
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_faucet_open(false)
        .unwrap();
    frames(&mut app, 12);
    assert_eq!(particles(&app, entity), released);
    let mut settings = app
        .world()
        .get::<Fluid3d>(entity)
        .unwrap()
        .settings()
        .clone();
    settings.smoothing_radius *= 1.5;
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_settings(settings)
        .unwrap();
    let changed = Fluid3dFaucet {
        position: Vec3::splat(1.5),
        width: 0.4,
        depth: 0.4,
        is_open: true,
        ..faucet
    };
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_faucet(Some(changed))
        .unwrap();
    frames(&mut app, 25);
    let all = particles(&app, entity);
    assert_eq!(all.len(), 19);
    assert_eq!(&all[..8], &released);
    assert!(
        all[8..]
            .iter()
            .all(|p| p.position_vec3().distance(changed.position) < 0.2)
    );
    let gpu = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    assert!(gpu.is_exhausted());
    assert_eq!(gpu.particle_buffer().id(), buffer_id);
    for _ in 0..3 {
        app.world_mut()
            .get_mut::<Fluid3d>(entity)
            .unwrap()
            .set_faucet_open(false)
            .unwrap();
        app.update();
        app.world_mut()
            .get_mut::<Fluid3d>(entity)
            .unwrap()
            .set_faucet_open(true)
            .unwrap();
        app.update();
    }
    assert_eq!(particles(&app, entity), all);
    let mut shapes = vec![
        Fluid3dCollider::Sphere {
            center: Vec3::splat(100.0),
            radius: 0.1
        };
        15
    ];
    shapes.push(Fluid3dCollider::Sphere {
        center: faucet.position,
        radius: 0.8,
    });
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_colliders(shapes)
        .unwrap();
    app.update();
    assert!(
        particles(&app, entity)[..8]
            .iter()
            .all(|p| p.position_vec3().distance(faucet.position) >= 0.87 - 1e-5)
    );
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_paused(true);
    app.world_mut().get_mut::<Fluid3d>(entity).unwrap().reset();
    app.update();
    assert!(particles(&app, entity).is_empty());
    assert_eq!(
        app.sub_app(RenderApp)
            .world()
            .resource::<GpuFluids3d>()
            .get(entity)
            .unwrap()
            .particle_capacity(),
        19
    );
    app.world_mut().despawn(entity);
    app.world_mut().despawn(other);
    app.update();
    assert_eq!(
        app.sub_app(RenderApp)
            .world()
            .resource::<GpuFluids3d>()
            .iter()
            .count(),
        0
    );
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn camera3d_depth_msaa_layers_and_reset_rebinding() {
    let mut app = headless_app(Fluid3dRenderPlugin);
    let (_, image) = camera_target(&mut app, Msaa::Off, 0);
    let (_, msaa_image) = camera_target(&mut app, Msaa::Sample4, 0);
    let (hdr_camera, hdr_image) = camera_target(&mut app, Msaa::Off, 0);
    app.world_mut()
        .entity_mut(hdr_camera)
        .insert(bevy::camera::Hdr);
    let (perspective_camera, perspective_image) = camera_target(&mut app, Msaa::Off, 0);
    app.world_mut()
        .entity_mut(perspective_camera)
        .insert(Projection::Perspective(PerspectiveProjection::default()));
    let (_, empty_image) = camera_target(&mut app, Msaa::Off, 1);
    let mut fluid = Fluid3d::new(
        Fluid3dSettings::default(),
        vec![FluidParticle3d::new(Vec3::ZERO)],
    )
    .unwrap();
    fluid.set_paused(true);
    let entity = app
        .world_mut()
        .spawn((
            fluid,
            Fluid3dRender {
                particle_radius: 0.5,
                ..default()
            },
        ))
        .id();
    frames(&mut app, 60);
    let center = (128 * 256 + 128) * 4;
    for image in [&image, &msaa_image, &hdr_image, &perspective_image] {
        let bytes = pixels(&app, image);
        assert!(
            bytes[center + 2] > bytes[center] + 30,
            "fluid did not render blue"
        );
    }
    let empty = pixels(&app, &empty_image);
    assert_eq!(&empty[center..center + 3], &[0, 0, 0]);
    // An opaque red box in front of the particles must occlude them.
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Cuboid::new(1.3, 1.3, 0.1));
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.02, 0.02),
            unlit: true,
            ..default()
        });
    let obstacle = app
        .world_mut()
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_xyz(0.0, 0.0, 1.0),
        ))
        .id();
    frames(&mut app, 30);
    let red = pixels(&app, &image);
    assert!(
        red[center] > red[center + 2] + 30,
        "mesh should occlude fluid"
    );
    // Move the same box behind the front hemisphere: particles should occlude it.
    app.world_mut()
        .get_mut::<Transform>(obstacle)
        .unwrap()
        .translation
        .z = -1.0;
    frames(&mut app, 10);
    let blue = pixels(&app, &image);
    assert!(
        blue[center + 2] > blue[center] + 30,
        "fluid should occlude mesh"
    );
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .reset_with_particles(vec![])
        .unwrap();
    frames(&mut app, 3);
    let red = pixels(&app, &image);
    assert!(
        red[center] > red[center + 2] + 30,
        "zero-particle reset must remove fluid"
    );
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn application_owned_3d_render_pass() {
    let mut app = headless_app(custom_example::CustomFluid3dRenderer);
    let (_, image) = camera_target(&mut app, Msaa::Off, 0);
    let mut fluid = Fluid3d::new(
        Fluid3dSettings::default(),
        vec![FluidParticle3d::new(Vec3::ZERO)],
    )
    .unwrap();
    fluid.set_paused(true);
    app.world_mut().spawn(fluid);
    frames(&mut app, 60);
    let bytes = pixels(&app, &image);
    assert!(
        bytes
            .chunks_exact(4)
            .any(|p| p[0] > 150 && p[0] > p[2] + 30)
    );
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn screen_space_surface_merges_neighbors_without_connecting_distant_depth_layers() {
    let mut app = headless_app(Fluid3dRenderPlugin);
    let (camera, target) = camera_target(&mut app, Msaa::Off, 0);
    let seed = vec![
        FluidParticle3d::new(Vec3::new(-0.4, 0.0, 0.0)),
        FluidParticle3d::new(Vec3::new(0.4, 0.0, 0.0)),
    ];
    let mut fluid = Fluid3d::new(Fluid3dSettings::default(), seed).unwrap();
    fluid.set_paused(true);
    let entity = app
        .world_mut()
        .spawn((
            fluid,
            Fluid3dRender {
                particle_radius: 0.25,
                ..default()
            },
        ))
        .id();
    frames(&mut app, 70);
    let center = (128 * 256 + 128) * 4;
    let joined = pixels(&app, &target);
    assert!(
        joined[center + 2] > joined[center] + 30,
        "nearby particles must share a blue liquid bridge"
    );
    capture::capture("liquid_3d_joined.png", &joined, false);
    app.world_mut()
        .get_mut::<Fluid3dRender>(entity)
        .unwrap()
        .surface_threshold = 0.8;
    frames(&mut app, 3);
    let separate = pixels(&app, &target);
    assert_eq!(&separate[center..center + 3], &[0, 0, 0]);
    capture::capture("liquid_3d_separate.png", &separate, false);
    app.world_mut()
        .get_mut::<Fluid3dRender>(entity)
        .unwrap()
        .surface_threshold = 0.35;
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .reset_with_particles(vec![
            FluidParticle3d::new(Vec3::new(-0.4, 0.0, 0.0)),
            FluidParticle3d::new(Vec3::new(0.4, 0.0, -2.0)),
        ])
        .unwrap();
    frames(&mut app, 3);
    let deep = pixels(&app, &target);
    assert_eq!(
        &deep[center..center + 3],
        &[0, 0, 0],
        "unrelated depth layers must not form a bridge"
    );
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .reset_with_particles(vec![
            FluidParticle3d::new(Vec3::new(-0.4, 0.0, 0.0)),
            FluidParticle3d::new(Vec3::new(0.4, 0.0, 0.0)),
        ])
        .unwrap();
    app.world_mut().get_mut::<Camera>(camera).unwrap().viewport = Some(bevy::camera::Viewport {
        physical_position: UVec2::new(64, 32),
        physical_size: UVec2::splat(128),
        ..default()
    });
    frames(&mut app, 5);
    let resized = pixels(&app, &target);
    let shifted = (96 * 256 + 128) * 4;
    assert!(
        resized[shifted + 2] > resized[shifted] + 30,
        "viewport origin/resize must recreate and map the surface fields"
    );
    app.world_mut()
        .get_mut::<Fluid3dRender>(entity)
        .unwrap()
        .enabled = false;
    frames(&mut app, 3);
    let cleared = pixels(&app, &target);
    assert_eq!(&cleared[shifted..shifted + 3], &[0, 0, 0]);
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn liquid_edges_preserve_water_hue_and_alpha_at_full_and_half_resolution() {
    let mut app = headless_app(Fluid3dRenderPlugin);
    let (camera, target) = camera_target(&mut app, Msaa::Off, 0);
    let (msaa_camera, msaa_target) = camera_target(&mut app, Msaa::Sample4, 0);
    for camera in [camera, msaa_camera] {
        app.world_mut()
            .get_mut::<Camera>(camera)
            .unwrap()
            .clear_color = ClearColorConfig::Custom(Color::linear_rgb(0.25, 0.25, 0.25));
    }
    // A fractional screen position exposes staircase edges and stale field pixels.
    let mut fluid = Fluid3d::new(
        Fluid3dSettings::default(),
        vec![FluidParticle3d::new(Vec3::new(0.013, -0.019, 0.0))],
    )
    .unwrap();
    fluid.set_paused(true);
    let style = Fluid3dRender {
        particle_radius: 0.38,
        ..default()
    };
    let water = style.color.to_f32_array();
    let entity = app.world_mut().spawn((fluid, style)).id();
    frames(&mut app, 60);
    let linear = |byte: u8| {
        let value = byte as f32 / 255.0;
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    for scale in [1.0, 0.5] {
        app.world_mut()
            .get_mut::<Fluid3dRender>(entity)
            .unwrap()
            .resolution_scale = scale;
        frames(&mut app, 4);
        for (name, image) in [("single", &target), ("msaa", &msaa_target)] {
            let bytes = pixels(&app, image);
            let mut partial_edges = 0;
            let mut max_alpha = 0.0_f32;
            for pixel in bytes.chunks_exact(4) {
                let rgb = [linear(pixel[0]), linear(pixel[1]), linear(pixel[2])];
                let alpha = ((rgb[2] - 0.25) / (water[2] - 0.25)).max(0.0);
                assert!(alpha <= water[3] + 0.02, "opacity exceeded color.a");
                max_alpha = max_alpha.max(alpha);
                // Every edge pixel must lie on the same water/background blend.
                // This catches gray/black fringes and straight/premultiplied alpha mismatch.
                for channel in 0..3 {
                    let expected = 0.25 + alpha * (water[channel] - 0.25);
                    assert!((rgb[channel] - expected).abs() < 0.012, "edge changed hue");
                }
                if alpha > 0.02 && alpha < water[3] - 0.02 {
                    partial_edges += 1;
                }
            }
            assert!(max_alpha > water[3] - 0.02, "surface interior is missing");
            assert!(
                partial_edges > 30,
                "contour must have fractional coverage without MSAA"
            );
            capture::capture(
                &format!("water_edges_{name}_{}.png", (scale * 100.0) as u32),
                &bytes,
                false,
            );
        }
    }
    app.world_mut()
        .get_mut::<Fluid3dRender>(entity)
        .unwrap()
        .color
        .alpha = 0.0;
    frames(&mut app, 3);
    for image in [&target, &msaa_target] {
        let bytes = pixels(&app, image);
        assert!(bytes.chunks_exact(4).all(|pixel| pixel[..3] == bytes[..3]));
    }
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn default_water_renders_while_falling_and_spreading_over_example_colliders() {
    let mut app = headless_app(Fluid3dRenderPlugin);
    app.insert_resource(ClearColor(Color::srgb(0.018, 0.025, 0.045)))
        .add_systems(
            Startup,
            |mut commands: Commands,
             mut meshes: ResMut<Assets<Mesh>>,
             mut materials: ResMut<Assets<StandardMaterial>>| {
                let obstacles = example_scene::scene(&mut commands, &mut meshes, &mut materials);
                let mut fluid = Fluid3d::from_faucet(
                    Fluid3dSettings::default(),
                    8000,
                    Fluid3dFaucet::default(),
                )
                .unwrap();
                fluid.set_colliders(obstacles).unwrap();
                commands.spawn((fluid, Fluid3dRender::default()));
            },
        );
    app.update();
    let camera = {
        let world = app.world_mut();
        world
            .query_filtered::<Entity, With<Camera3d>>()
            .single(world)
            .unwrap()
    };
    let size = UVec2::new(960, 640);
    let mut image = Image::new_target_texture(size.x, size.y, TextureFormat::Rgba8UnormSrgb, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    app.world_mut()
        .entity_mut(camera)
        .insert((RenderTarget::Image(target.clone().into()), Msaa::Sample4));
    for (frame, steps) in [(45, 45), (90, 45), (150, 60)] {
        frames(&mut app, steps);
        let bytes = pixels(&app, &target);
        assert!(
            bytes
                .chunks_exact(4)
                .filter(|p| p[2] > p[0].saturating_add(25) && p[1] > p[0].saturating_add(25))
                .count()
                > 100,
            "default liquid disappeared during emission/collision"
        );
        capture::capture_sized(&format!("water_faucet_{frame}.png"), &bytes, size, false);
    }
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn invisible_foreground_kernel_does_not_cut_holes_in_deeper_water() {
    let mut app = headless_app(Fluid3dRenderPlugin);
    let (camera, target) = camera_target(&mut app, Msaa::Off, 0);
    app.world_mut()
        .get_mut::<Camera>(camera)
        .unwrap()
        .clear_color = ClearColorConfig::Custom(Color::linear_rgb(0.25, 0.25, 0.25));
    // At x=0.5 the near particle contributes less than the isovalue, but its
    // expanded kernel still overlaps the fully visible far particle's center.
    let mut fluid = Fluid3d::new(
        Fluid3dSettings::default(),
        vec![
            FluidParticle3d::new(Vec3::new(0.0, 0.0, 1.0)),
            FluidParticle3d::new(Vec3::new(0.5, 0.0, -1.0)),
        ],
    )
    .unwrap();
    fluid.set_paused(true);
    app.world_mut().spawn((
        fluid,
        Fluid3dRender {
            particle_radius: 0.25,
            ..default()
        },
    ));
    frames(&mut app, 60);
    let bytes = pixels(&app, &target);
    let far_center = (128 * 256 + 160) * 4;
    assert!(
        bytes[far_center + 2] > bytes[far_center] + 30,
        "an invisible foreground kernel cut a gray hole in the background liquid"
    );
    // This rectangle lies fully inside the far droplet, across the near core's rim.
    // A density derivative there is a layer transition, not an exterior contour:
    // it must not reduce alpha and draw a dark circular seam inside the water.
    for y in 122..135 {
        for x in 140..176 {
            let pixel = (y * 256 + x) * 4;
            for channel in 0..3 {
                assert!(
                    bytes[pixel + channel].abs_diff(bytes[far_center + channel]) <= 4,
                    "internal density change drew a circular seam"
                );
            }
        }
    }
    capture::capture("water_depth_layers.png", &bytes, false);
}

#[path = "support/lifecycle.rs"]
mod lifecycle;

#[test]
#[ignore = "requires a compute-capable GPU"]
fn recycling_is_optional_sweeps_thin_rotated_regions_and_preserves_finite_budget() {
    let settings = Fluid3dSettings {
        gravity: Vec3::ZERO,
        viscosity: 0.0,
        ..default()
    };
    let regions = Fluid3dRecycling::new(vec![
        Fluid3dCollider::Sphere {
            center: Vec3::ZERO,
            radius: 0.2,
        },
        Fluid3dCollider::OrientedCuboid {
            center: Vec3::new(0.0, 0.8, 0.0),
            half_extents: Vec3::new(0.001, 0.2, 0.2),
            rotation: Quat::from_rotation_z(0.4),
        },
        Fluid3dCollider::Sphere {
            center: Vec3::ZERO,
            radius: 0.2,
        },
    ])
    .unwrap();
    // Component alone does nothing: installing the extension is an application choice.
    let mut without = headless_app(Fluid3dRenderPlugin);
    let control = without
        .world_mut()
        .spawn((
            Fluid3d::new(settings.clone(), vec![FluidParticle3d::new(Vec3::ZERO)]).unwrap(),
            regions.clone(),
        ))
        .id();
    lifecycle::frames(&mut without, 35);
    let state = without
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(control)
        .unwrap();
    assert_eq!(
        lifecycle::read_u32(&without, state.live_count_buffer()),
        vec![1]
    );
    drop(without);

    let mut app = headless_app((Fluid3dRenderPlugin, Fluid3dRecyclingPlugin));
    let (_, target) = camera_target(&mut app, Msaa::Off, 0);
    let mut seed = vec![FluidParticle3d::new(Vec3::ZERO); 129];
    seed.push(FluidParticle3d::with_velocity(
        Vec3::new(-0.1, 0.8, 0.0),
        Vec3::X * 50.0,
    ));
    seed.push(FluidParticle3d::new(Vec3::new(0.3, 0.0, 0.0)));
    let mut fluid = Fluid3d::new(settings.clone(), seed).unwrap();
    fluid.set_paused(true);
    let entity = app.world_mut().spawn((fluid, regions.clone())).id();
    lifecycle::frames(&mut app, 40);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![131],
        "pause must pause extensions"
    );
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_paused(false);
    lifecycle::frames(&mut app, 6);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    let flags = lifecycle::read_u32(&app, state.particle_activity_buffer());
    assert!(
        flags[..130].iter().all(|&flag| flag == 0),
        "inside seeds and a full thin-region crossing must retire"
    );
    assert_eq!(flags[130], 1);
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![1],
        "overlapping triggers must decrement once"
    );
    let particles = lifecycle::read_u32(&app, state.particle_buffer());
    assert!(
        (f32::from_bits(particles[130 * 8]) - 0.3).abs() < 0.00001,
        "retired particles must not contribute SPH pressure"
    );
    let activity_id = state.particle_activity_buffer().id();
    let mut changed = settings.clone();
    changed.smoothing_radius *= 1.5;
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_settings(changed)
        .unwrap();
    lifecycle::frames(&mut app, 4);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    assert_eq!(state.particle_activity_buffer().id(), activity_id);
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![1]
    );
    app.world_mut()
        .get_mut::<Fluid3dRecycling>(entity)
        .unwrap()
        .enabled = false;
    {
        let mut fluid = app.world_mut().get_mut::<Fluid3d>(entity).unwrap();
        fluid.set_paused(true);
        fluid.reset();
    }
    lifecycle::frames(&mut app, 3);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    assert_ne!(state.particle_activity_buffer().id(), activity_id);
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![131]
    );
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_paused(false);
    lifecycle::frames(&mut app, 3);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![131],
        "disabled region must not collect"
    );
    app.world_mut()
        .entity_mut(entity)
        .remove::<Fluid3dRecycling>();
    lifecycle::frames(&mut app, 2);
    app.world_mut().entity_mut(entity).insert(regions);
    app.world_mut().get_mut::<Fluid3d>(entity).unwrap().reset();
    lifecycle::frames(&mut app, 3);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![1]
    );
    app.world_mut().despawn(entity);

    let faucet = Fluid3dFaucet {
        position: Vec3::ZERO,
        velocity: Vec3::ZERO,
        layers_per_second: 240.0,
        ..default()
    };
    let fluid = Fluid3d::from_faucet(settings, 7, faucet).unwrap();
    let region = Fluid3dRecycling::new(vec![Fluid3dCollider::Cuboid(Fluid3dBounds::new(
        -Vec3::ONE,
        Vec3::ONE,
    ))])
    .unwrap();
    let entity = app
        .world_mut()
        .spawn((fluid, region, Fluid3dRender::default()))
        .id();
    lifecycle::frames(&mut app, 10);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    assert_eq!(state.particle_count(), 7);
    assert!(state.is_exhausted());
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![0]
    );
    assert!(
        pixels(&app, &target)
            .chunks_exact(4)
            .all(|p| p[..3] == [0, 0, 0]),
        "default renderer must skip retired slots"
    );
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_faucet_open(false)
        .unwrap();
    lifecycle::frames(&mut app, 2);
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_faucet_open(true)
        .unwrap();
    lifecycle::frames(&mut app, 8);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    assert_eq!(state.particle_count(), 7);
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![0]
    );
    {
        let mut fluid = app.world_mut().get_mut::<Fluid3d>(entity).unwrap();
        fluid.set_paused(true);
        fluid.reset();
    }
    lifecycle::frames(&mut app, 2);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    assert_eq!(state.particle_count(), 0);
    assert!(!state.is_exhausted());
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_paused(false);
    lifecycle::frames(&mut app, 5);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    assert_eq!(state.particle_count(), 7);
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![0]
    );
    app.world_mut().despawn(entity);
    lifecycle::frames(&mut app, 2);
    assert!(
        app.sub_app(RenderApp)
            .world()
            .resource::<GpuFluids3d>()
            .get(entity)
            .is_none()
    );
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn application_compute_passes_gate_readiness_run_each_substep_and_add_forces() {
    use bevy::render::render_resource::binding_types::*;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };
    type Trace = Arc<Mutex<Vec<(u64, FluidComputeStage, u8)>>>;
    struct Impulse {
        id: CachedComputePipelineId,
        group: BindGroup,
        ready: Arc<AtomicBool>,
        trace: Trace,
    }
    impl Fluid3dComputePass for Impulse {
        fn is_ready(&self, _: Entity, cache: &PipelineCache) -> bool {
            self.ready.load(Ordering::Relaxed) && cache.get_compute_pipeline(self.id).is_some()
        }
        fn encode(&self, stage: FluidComputeStage, ctx: &mut Fluid3dComputeContext<'_>) {
            self.trace
                .lock()
                .unwrap()
                .push((ctx.substep_index, stage, 0));
            if stage != FluidComputeStage::BeforeIntegrate {
                return;
            }
            let mut pass = ctx.encoder.begin_compute_pass(&default());
            pass.set_pipeline(ctx.pipelines.get_compute_pipeline(self.id).unwrap());
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(ctx.particle_count.div_ceil(64), 1, 1);
        }
    }
    struct Probe(Trace);
    impl Fluid3dComputePass for Probe {
        fn encode(&self, stage: FluidComputeStage, ctx: &mut Fluid3dComputeContext<'_>) {
            self.0.lock().unwrap().push((ctx.substep_index, stage, 1));
        }
    }
    let mut app = headless_app(Fluid3dRenderPlugin);
    let settings = Fluid3dSettings {
        gravity: Vec3::ZERO,
        stiffness: 0.0,
        viscosity: 0.0,
        ..default()
    };
    let dt = settings.time_step;
    let mut fluid = Fluid3d::new(settings, vec![FluidParticle3d::new(Vec3::ZERO)]).unwrap();
    fluid.set_paused(true);
    let entity = app.world_mut().spawn(fluid).id();
    lifecycle::frames(&mut app, 35);
    let shader = app
        .world_mut()
        .resource_mut::<Assets<Shader>>()
        .add(Shader::from_wgsl(
            r#"
        @group(0) @binding(0) var<storage,read_write> acceleration:array<vec4<f32>>;
        @group(0) @binding(1) var<storage,read> activity:array<u32>;
        @compute @workgroup_size(64) fn force(@builtin(global_invocation_id) id:vec3<u32>) {
            if id.x>=arrayLength(&activity) {return;}
            if activity[id.x]==0u {return;}
            acceleration[id.x]+=vec4<f32>(2.0,0.0,0.0,0.0);
        }
    "#,
            "test-external-force.wgsl",
        ));
    let (id, group) = {
        let world = app.sub_app(RenderApp).world();
        let device = world.resource::<RenderDevice>();
        let cache = world.resource::<PipelineCache>();
        let layout = BindGroupLayoutDescriptor::new(
            "external test force",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_sized(false, BufferSize::new(16)),
                    storage_buffer_read_only_sized(false, BufferSize::new(4)),
                ),
            ),
        );
        let id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("external force".into()),
            layout: vec![layout.clone()],
            shader,
            entry_point: Some("force".into()),
            ..default()
        });
        let state = world.resource::<GpuFluids3d>().get(entity).unwrap();
        let group = device.create_bind_group(
            "external test force",
            &cache.get_bind_group_layout(&layout),
            &BindGroupEntries::sequential((
                state.acceleration_buffer().as_entire_binding(),
                state.particle_activity_buffer().as_entire_binding(),
            )),
        );
        (id, group)
    };
    let ready = Arc::new(AtomicBool::new(false));
    let trace: Trace = default();
    {
        let mut passes = app
            .sub_app_mut(RenderApp)
            .world_mut()
            .resource_mut::<Fluid3dComputePasses>();
        passes.insert(Impulse {
            id,
            group,
            ready: ready.clone(),
            trace: trace.clone(),
        });
        passes.insert(Probe(trace.clone()));
    }
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_paused(false);
    lifecycle::frames(&mut app, 5);
    assert!(trace.lock().unwrap().is_empty());
    assert_eq!(
        app.sub_app(RenderApp)
            .world()
            .resource::<GpuFluids3d>()
            .get(entity)
            .unwrap()
            .simulated_steps(),
        0
    );
    ready.store(true, Ordering::Relaxed);
    lifecycle::frames(&mut app, 7);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    let steps = state.simulated_steps();
    assert!(steps > 1);
    let output = lifecycle::read_u32(&app, state.particle_buffer());
    let velocity = f32::from_bits(output[4]);
    assert!(
        (velocity - 2.0 * dt * steps as f32).abs() < 0.0001,
        "external force was not applied each fixed substep"
    );
    let log = trace.lock().unwrap();
    assert_eq!(log.len(), steps as usize * 6);
    for (step, entries) in log.chunks_exact(6).enumerate() {
        assert_eq!(
            entries,
            &[
                (step as u64, FluidComputeStage::BeforeSubstep, 0),
                (step as u64, FluidComputeStage::BeforeSubstep, 1),
                (step as u64, FluidComputeStage::BeforeIntegrate, 0),
                (step as u64, FluidComputeStage::BeforeIntegrate, 1),
                (step as u64, FluidComputeStage::AfterSubstep, 0),
                (step as u64, FluidComputeStage::AfterSubstep, 1),
            ]
        );
    }
    drop(log);
    app.sub_app_mut(RenderApp)
        .world_mut()
        .resource_mut::<Fluid3dComputePasses>()
        .remove::<Impulse>();
    lifecycle::frames(&mut app, 4);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    assert_eq!(
        f32::from_bits(lifecycle::read_u32(&app, state.particle_buffer())[4]),
        velocity
    );
}

#[path = "../examples/support/sink3d.rs"]
mod sink_scene;

#[test]
#[ignore = "requires a compute-capable GPU"]
fn sink_bottom_hole_and_countertop_spill_both_reach_the_collector() {
    let mut app = headless_app((Fluid3dRenderPlugin, Fluid3dRecyclingPlugin));
    let geometry = sink_scene::SinkGeometry::default();
    let settings = Fluid3dSettings {
        stiffness: 0.0,
        viscosity: 0.0,
        ..default()
    };
    let mut fluid = Fluid3d::new(
        settings,
        vec![
            // Straight through the hole, with no hidden full-width bottom collider.
            FluidParticle3d::new(Vec3::new(0.0, 1.5, 0.0)),
            // Above an actual bottom tile: must remain supported instead of draining.
            FluidParticle3d::new(Vec3::new(1.2, 0.5, 0.0)),
            // First land on the desktop, then run off the outside edge.
            FluidParticle3d::with_velocity(Vec3::new(3.0, 0.93, 0.8), Vec3::X * 0.8),
        ],
    )
    .unwrap();
    fluid.set_colliders(geometry.colliders()).unwrap();
    fluid.set_paused(true);
    let entity = app
        .world_mut()
        .spawn((
            fluid,
            Fluid3dRecycling::new(vec![Fluid3dCollider::Cuboid(geometry.collector)]).unwrap(),
        ))
        .id();
    frames(&mut app, 40);
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_paused(false);
    frames(&mut app, 5);
    let on_table = particles(&app, entity);
    assert!(on_table[2].position_vec3().y >= 0.85 + 0.07 - 0.0001);
    assert!(on_table[2].position_vec3().x < 3.25);
    frames(&mut app, 120);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .get(entity)
        .unwrap();
    assert_eq!(
        lifecycle::read_u32(&app, state.particle_activity_buffer()),
        vec![0, 1, 0],
        "hole drainage and table-edge spills must collect; solid bottom tiles must hold water"
    );
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![1]
    );
    let supported = particles(&app, entity)[1].position_vec3();
    assert!(supported.y >= 0.07 - 0.0001 && supported.y < 0.1);
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn sink_example_renders_and_recycles_with_default_faucet() {
    let mut app = headless_app((Fluid3dRenderPlugin, Fluid3dRecyclingPlugin));
    app.insert_resource(ClearColor(Color::srgb(0.018, 0.025, 0.045)))
        .insert_resource(sink_scene::SinkDemo {
            recycling_enabled: true,
        })
        .add_systems(Startup, sink_scene::setup)
        .add_systems(Update, sink_scene::sync_faucet);
    app.update();
    let camera = {
        let world = app.world_mut();
        world
            .query_filtered::<Entity, With<Camera3d>>()
            .single(world)
            .unwrap()
    };
    let size = UVec2::new(960, 640);
    let mut image = Image::new_target_texture(size.x, size.y, TextureFormat::Rgba8UnormSrgb, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    app.world_mut()
        .entity_mut(camera)
        .insert((RenderTarget::Image(target.clone().into()), Msaa::Sample4));
    for (frame, steps) in [(90, 90), (180, 90), (300, 120)] {
        frames(&mut app, steps);
        let bytes = pixels(&app, &target);
        assert!(
            bytes
                .chunks_exact(4)
                .filter(|p| p[2] > p[0].saturating_add(25) && p[1] > p[0].saturating_add(25))
                .count()
                > 100,
            "sink scene lost its water rendering"
        );
        capture::capture_sized(&format!("sink_faucet_{frame}.png"), &bytes, size, false);
    }
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .iter()
        .next()
        .unwrap()
        .1;
    let live = lifecycle::read_u32(&app, state.live_count_buffer())[0];
    assert!(
        live > 0 && live < state.particle_count(),
        "default sink must both emit and collect"
    );
    assert_eq!(state.particle_capacity(), 8000);

    // The metal faucet must track changes made by the keyboard controls.
    let entity = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids3d>()
        .iter()
        .next()
        .unwrap()
        .0;
    let mut outlet = *app
        .world()
        .get::<Fluid3d>(entity)
        .unwrap()
        .faucet()
        .unwrap();
    outlet.position.x = 3.0;
    outlet.width = 0.8;
    outlet.depth = 0.8;
    app.world_mut()
        .get_mut::<Fluid3d>(entity)
        .unwrap()
        .set_faucet(Some(outlet))
        .unwrap();
    frames(&mut app, 2);
    let world = app.world_mut();
    let visual = world
        .query_filtered::<&Transform, With<sink_scene::FaucetVisual>>()
        .single(world)
        .unwrap();
    assert_eq!(visual.translation, outlet.position);
    let mouth = world
        .query_filtered::<&Transform, With<sink_scene::OutletMouth>>()
        .single(world)
        .unwrap();
    assert_eq!(mouth.scale, Vec3::new(0.92, 0.18, 0.92));
}
