//! End-to-end asset, extraction, scheduling and offscreen rendering test.
use std::time::Duration;

use bevy::{
    app::PluginsState,
    prelude::*,
    render::{
        RenderApp, RenderPlugin,
        pipelined_rendering::PipelinedRenderingPlugin,
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        texture::GpuImage,
    },
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_gpu_fluid::{fluid2d::GpuFluids2d, prelude::*};
#[path = "support/capture.rs"]
mod capture;

#[allow(dead_code)]
#[path = "../examples/custom_pass.rs"]
mod custom_example;

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
    .add_plugins((Fluid2dPlugin, renderer))
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

#[test]
#[ignore = "requires a compute-capable GPU; cargo test --test bevy_smoke -- --ignored"]
fn embedded_assets_simulation_render_reset_and_cleanup() {
    let mut app = headless_app(Fluid2dRenderPlugin);
    let fluid = Fluid2d::new(
        Fluid2dSettings::default(),
        FluidParticle2d::lattice(Vec2::new(-1.0, -1.0), UVec2::new(12, 12), 0.16),
    )
    .unwrap();
    let bundle = Fluid2dBundle::new(
        fluid,
        &mut app.world_mut().resource_mut::<Assets<Image>>(),
        UVec2::splat(256),
    )
    .unwrap();
    let image = bundle.surface.image.clone();
    let entity = app.world_mut().spawn(bundle).id();
    let other = app
        .world_mut()
        .spawn(Fluid2d::new(Fluid2dSettings::default(), vec![]).unwrap())
        .id();
    for _ in 0..200 {
        app.update();
        let gpu = app.sub_app(RenderApp).world().resource::<GpuFluids2d>();
        assert!(gpu.error(entity).is_none());
        if gpu.get(entity).is_some_and(|g| g.simulated_steps() >= 20) {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let render = app.sub_app(RenderApp).world();
    assert!(
        render
            .resource::<GpuFluids2d>()
            .get(entity)
            .unwrap()
            .simulated_steps()
            >= 20,
        "embedded compute shader never became ready"
    );
    assert_eq!(
        render
            .resource::<GpuFluids2d>()
            .get(other)
            .unwrap()
            .particle_count(),
        0
    );

    // Verify the default render pass actually wrote nontransparent blue texels.
    let device = render.resource::<RenderDevice>();
    let queue = render.resource::<RenderQueue>();
    let gpu_image = render
        .resource::<RenderAssets<GpuImage>>()
        .get(&image)
        .unwrap();
    let buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 256 * 256 * 4,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&default());
    encoder.copy_texture_to_buffer(
        gpu_image.texture.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(1024),
                rows_per_image: Some(256),
            },
        },
        Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let pixels = buffer.slice(..).get_mapped_range().unwrap();
    assert!(
        pixels.chunks_exact(4).any(|p| p[3] > 200 && p[2] > p[0]),
        "default render pass produced no particles"
    );
    assert!(
        pixels.chunks_exact(4).any(|p| p[3] == 0),
        "background must be transparent"
    );
    drop(pixels);
    buffer.unmap();

    // Pause keeps state fixed. A reset while paused replaces the GPU buffer and
    // the renderer must rebind it. A settings/grid change preserves positions.
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_paused(true);
    app.update();
    let steps = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap()
        .simulated_steps();
    app.update();
    assert_eq!(
        app.sub_app(RenderApp)
            .world()
            .resource::<GpuFluids2d>()
            .get(entity)
            .unwrap()
            .simulated_steps(),
        steps
    );
    let old_buffer = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap()
        .particle_buffer()
        .id();
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .reset_with_particles(vec![FluidParticle2d::new(Vec2::ZERO)])
        .unwrap();
    app.update();
    let gpu = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap();
    assert_eq!(gpu.particle_count(), 1);
    assert_eq!(gpu.simulated_steps(), 0);
    assert_ne!(gpu.particle_buffer().id(), old_buffer);
    let reset_buffer = gpu.particle_buffer().id();
    let mut settings = app
        .world()
        .get::<Fluid2d>(entity)
        .unwrap()
        .settings()
        .clone();
    settings.smoothing_radius *= 2.0;
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_settings(settings)
        .unwrap();
    app.update();
    assert_eq!(
        app.sub_app(RenderApp)
            .world()
            .resource::<GpuFluids2d>()
            .get(entity)
            .unwrap()
            .particle_buffer()
            .id(),
        reset_buffer
    );
    app.world_mut().despawn(entity);
    app.world_mut().despawn(other);
    app.update();
    assert_eq!(
        app.sub_app(RenderApp)
            .world()
            .resource::<GpuFluids2d>()
            .iter()
            .count(),
        0
    );
}

#[test]
#[ignore = "requires a compute-capable GPU; cargo test --test bevy_smoke -- --ignored"]
fn application_owned_render_pass() {
    let mut app = headless_app(custom_example::CustomFluidRenderer);
    let fluid = Fluid2d::new(
        Fluid2dSettings::default(),
        vec![FluidParticle2d::new(Vec2::ZERO)],
    )
    .unwrap();
    let bundle = Fluid2dBundle::new(
        fluid,
        &mut app.world_mut().resource_mut::<Assets<Image>>(),
        UVec2::splat(256),
    )
    .unwrap();
    let target = bundle.surface.image.clone();
    let entity = app.world_mut().spawn(bundle).id();
    for _ in 0..200 {
        app.update();
        if app
            .sub_app(RenderApp)
            .world()
            .resource::<GpuFluids2d>()
            .get(entity)
            .is_some_and(|g| g.simulated_steps() > 20)
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let render = app.sub_app(RenderApp).world();
    assert!(
        render
            .resource::<GpuFluids2d>()
            .get(entity)
            .unwrap()
            .simulated_steps()
            > 20
    );
    let device = render.resource::<RenderDevice>();
    let image = render
        .resource::<RenderAssets<GpuImage>>()
        .get(&target)
        .unwrap();
    let buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 256 * 256 * 4,
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
                bytes_per_row: Some(1024),
                rows_per_image: Some(256),
            },
        },
        Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
    );
    render.resource::<RenderQueue>().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let pixels = buffer.slice(..).get_mapped_range().unwrap();
    assert!(
        pixels.chunks_exact(4).any(|p| p[3] == 255 && p[0] > p[2]),
        "application-owned pass produced no warm-colored pixels"
    );
}

fn read_particles(app: &App, entity: Entity) -> Vec<FluidParticle2d> {
    let render = app.sub_app(RenderApp).world();
    let gpu = render.resource::<GpuFluids2d>().get(entity).unwrap();
    let device = render.resource::<RenderDevice>();
    let size = gpu.particle_capacity().max(1) as u64 * 16;
    let buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&default());
    encoder.copy_buffer_to_buffer(gpu.particle_buffer(), 0, &buffer, 0, size);
    render.resource::<RenderQueue>().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let data = buffer.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice::<u8, FluidParticle2d>(&data)[..gpu.particle_count() as usize].to_vec()
}

fn surface_pixels(app: &App, target: &Handle<Image>) -> Vec<u8> {
    let render = app.sub_app(RenderApp).world();
    let device = render.resource::<RenderDevice>();
    let image = render
        .resource::<RenderAssets<GpuImage>>()
        .get(target)
        .unwrap();
    let buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 256 * 256 * 4,
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
                bytes_per_row: Some(1024),
                rows_per_image: Some(256),
            },
        },
        Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
    );
    render.resource::<RenderQueue>().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    buffer.slice(..).get_mapped_range().unwrap().to_vec()
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn density_surface_connects_neighbors_and_threshold_controls_separation() {
    let mut app = headless_app(Fluid2dRenderPlugin);
    let settings = Fluid2dSettings {
        bounds: Rect::new(-2.0, -2.0, 2.0, 2.0),
        ..default()
    };
    // Radius 0.25 disks have a 0.3-unit gap; the previous defaults did not bridge it. The density isosurface should bridge it.
    let seed = vec![
        FluidParticle2d::new(Vec2::new(-0.4, 0.0)),
        FluidParticle2d::new(Vec2::new(0.4, 0.0)),
    ];
    let mut fluid = Fluid2d::new(settings, seed).unwrap();
    fluid.set_paused(true);
    let mut bundle = Fluid2dBundle::new(
        fluid,
        &mut app.world_mut().resource_mut::<Assets<Image>>(),
        UVec2::splat(256),
    )
    .unwrap();
    bundle.surface.style.particle_radius = 0.25;
    let target = bundle.surface.image.clone();
    let entity = app.world_mut().spawn(bundle).id();
    for _ in 0..70 {
        app.update();
        std::thread::sleep(Duration::from_millis(3));
    }
    let center = (128 * 256 + 128) * 4;
    let joined = surface_pixels(&app, &target);
    assert!(
        joined[center + 3] > 240,
        "nearby particles must share a liquid bridge"
    );
    assert_eq!(&joined[..4], &[0, 0, 0, 0]);
    capture::capture("liquid_2d_joined.png", &joined, true);
    app.world_mut()
        .get_mut::<Fluid2dSurface>(entity)
        .unwrap()
        .style
        .surface_threshold = 0.8;
    for _ in 0..3 {
        app.update();
    }
    let separate = surface_pixels(&app, &target);
    assert_eq!(
        separate[center + 3],
        0,
        "raising the isovalue must break the bridge"
    );
    capture::capture("liquid_2d_separate.png", &separate, true);
    app.world_mut()
        .get_mut::<Fluid2dSurface>(entity)
        .unwrap()
        .style
        .surface_threshold = 0.35;
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .reset_with_particles(vec![
            FluidParticle2d::new(Vec2::new(-0.8, 0.0)),
            FluidParticle2d::new(Vec2::new(0.8, 0.0)),
        ])
        .unwrap();
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(
        surface_pixels(&app, &target)[center + 3],
        0,
        "distant droplets must stay separate"
    );
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .reset_with_particles(vec![])
        .unwrap();
    for _ in 0..3 {
        app.update();
    }
    assert!(
        surface_pixels(&app, &target).iter().all(|p| *p == 0),
        "empty fields must clear the old surface"
    );
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn finite_faucet_preserves_budget_on_close_move_resize_and_exhaustion() {
    let mut app = headless_app(Fluid2dRenderPlugin);
    let settings = Fluid2dSettings {
        gravity: Vec2::ZERO,
        stiffness: 0.0,
        viscosity: 0.0,
        ..default()
    };
    let faucet = Fluid2dFaucet {
        position: Vec2::new(0.0, 2.0),
        width: 0.8,
        particle_spacing: 0.2,
        rows_per_second: 30.0,
        velocity: Vec2::ZERO,
        is_open: false,
    };
    let fluid = Fluid2d::from_faucet(settings, 13, faucet).unwrap();
    let bundle = Fluid2dBundle::new(
        fluid,
        &mut app.world_mut().resource_mut::<Assets<Image>>(),
        UVec2::splat(256),
    )
    .unwrap();
    let entity = app.world_mut().spawn(bundle).id();
    for _ in 0..200 {
        app.update();
        if app
            .sub_app(RenderApp)
            .world()
            .resource::<GpuFluids2d>()
            .get(entity)
            .is_some_and(|g| g.simulated_steps() > 0)
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap();
    assert!(state.simulated_steps() > 0);
    assert_eq!(state.particle_capacity(), 13);
    assert_eq!(state.particle_count(), 0);
    let buffer_id = state.particle_buffer().id();
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_faucet_open(true)
        .unwrap();
    for _ in 0..20 {
        app.update();
        if app
            .sub_app(RenderApp)
            .world()
            .resource::<GpuFluids2d>()
            .get(entity)
            .unwrap()
            .particle_count()
            > 0
        {
            break;
        }
    }
    let released = read_particles(&app, entity);
    assert_eq!(released.len(), 4);
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_faucet_open(false)
        .unwrap();
    for _ in 0..20 {
        app.update();
    }
    assert_eq!(read_particles(&app, entity), released);
    // Changing the grid must preserve released count and unused capacity.
    let mut settings = app
        .world()
        .get::<Fluid2d>(entity)
        .unwrap()
        .settings()
        .clone();
    settings.smoothing_radius *= 1.5;
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_settings(settings)
        .unwrap();
    let changed = Fluid2dFaucet {
        position: Vec2::new(1.5, 2.5),
        width: 0.4,
        is_open: true,
        ..faucet
    };
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_faucet(Some(changed))
        .unwrap();
    for _ in 0..40 {
        app.update();
    }
    let particles = read_particles(&app, entity);
    assert_eq!(particles.len(), 13);
    assert_eq!(&particles[..4], &released);
    assert!(
        particles[4..]
            .iter()
            .all(|p| (1.3..=1.7).contains(&p.position[0]) && (p.position[1] - 2.5).abs() < 1e-5)
    );
    let gpu = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap();
    assert!(gpu.is_exhausted());
    assert_eq!(gpu.particle_buffer().id(), buffer_id);
    for _ in 0..4 {
        app.world_mut()
            .get_mut::<Fluid2d>(entity)
            .unwrap()
            .set_faucet_open(false)
            .unwrap();
        app.update();
        app.world_mut()
            .get_mut::<Fluid2d>(entity)
            .unwrap()
            .set_faucet_open(true)
            .unwrap();
        app.update();
    }
    assert_eq!(read_particles(&app, entity), particles);
    // Exercise live collider uploads and the last slot of the shared 16-shape array.
    let mut colliders = vec![
        Fluid2dCollider::Circle {
            center: Vec2::splat(100.0),
            radius: 0.1
        };
        15
    ];
    colliders.push(Fluid2dCollider::Circle {
        center: Vec2::new(0.0, 2.0),
        radius: 0.7,
    });
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_colliders(colliders)
        .unwrap();
    app.update();
    let projected = read_particles(&app, entity);
    assert_eq!(projected.len(), 13);
    assert!(
        projected[..4]
            .iter()
            .all(|p| Vec2::from_array(p.position).distance(Vec2::new(0.0, 2.0)) >= 0.77 - 1e-5)
    );
    assert_eq!(
        app.sub_app(RenderApp)
            .world()
            .resource::<GpuFluids2d>()
            .get(entity)
            .unwrap()
            .particle_buffer()
            .id(),
        buffer_id
    );
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_paused(true);
    app.world_mut().get_mut::<Fluid2d>(entity).unwrap().reset();
    app.update();
    let gpu = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap();
    assert_eq!(gpu.particle_capacity(), 13);
    assert_eq!(gpu.particle_count(), 0);
    assert!(!gpu.is_exhausted());
}

#[path = "support/lifecycle.rs"]
mod lifecycle;

#[test]
#[ignore = "requires a compute-capable GPU"]
fn recycling_is_optional_sweeps_thin_rotated_regions_and_preserves_finite_budget() {
    let settings = Fluid2dSettings {
        gravity: Vec2::ZERO,
        viscosity: 0.0,
        ..default()
    };
    let regions = Fluid2dRecycling::new(vec![
        Fluid2dCollider::Circle {
            center: Vec2::ZERO,
            radius: 0.2,
        },
        Fluid2dCollider::OrientedRectangle {
            center: Vec2::new(0.0, 0.8),
            half_extents: Vec2::new(0.001, 0.2),
            rotation: 0.4,
        },
        Fluid2dCollider::Circle {
            center: Vec2::ZERO,
            radius: 0.2,
        },
    ])
    .unwrap();
    // Component alone does nothing: installing the extension is an application choice.
    let mut without = headless_app(Fluid2dRenderPlugin);
    let control = without
        .world_mut()
        .spawn((
            Fluid2d::new(settings.clone(), vec![FluidParticle2d::new(Vec2::ZERO)]).unwrap(),
            regions.clone(),
        ))
        .id();
    lifecycle::frames(&mut without, 35);
    let state = without
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(control)
        .unwrap();
    assert_eq!(
        lifecycle::read_u32(&without, state.live_count_buffer()),
        vec![1]
    );
    drop(without);

    let mut app = headless_app((Fluid2dRenderPlugin, Fluid2dRecyclingPlugin));
    let mut seed = vec![FluidParticle2d::new(Vec2::ZERO); 129];
    seed.push(FluidParticle2d {
        position: [-0.1, 0.8],
        velocity: [50.0, 0.0],
    });
    seed.push(FluidParticle2d::new(Vec2::new(0.3, 0.0)));
    let mut fluid = Fluid2d::new(settings.clone(), seed).unwrap();
    fluid.set_paused(true);
    let entity = app.world_mut().spawn((fluid, regions.clone())).id();
    lifecycle::frames(&mut app, 40);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap();
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![131],
        "pause must pause extensions"
    );
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_paused(false);
    lifecycle::frames(&mut app, 6);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
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
        (f32::from_bits(particles[130 * 4]) - 0.3).abs() < 0.00001,
        "retired particles must not contribute SPH pressure"
    );
    let activity_id = state.particle_activity_buffer().id();
    let mut changed = settings.clone();
    changed.smoothing_radius *= 1.5;
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_settings(changed)
        .unwrap();
    lifecycle::frames(&mut app, 4);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap();
    assert_eq!(state.particle_activity_buffer().id(), activity_id);
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![1]
    );
    app.world_mut()
        .get_mut::<Fluid2dRecycling>(entity)
        .unwrap()
        .enabled = false;
    {
        let mut fluid = app.world_mut().get_mut::<Fluid2d>(entity).unwrap();
        fluid.set_paused(true);
        fluid.reset();
    }
    lifecycle::frames(&mut app, 3);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap();
    assert_ne!(state.particle_activity_buffer().id(), activity_id);
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![131]
    );
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_paused(false);
    lifecycle::frames(&mut app, 3);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap();
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![131],
        "disabled region must not collect"
    );
    app.world_mut()
        .entity_mut(entity)
        .remove::<Fluid2dRecycling>();
    lifecycle::frames(&mut app, 2);
    app.world_mut().entity_mut(entity).insert(regions);
    app.world_mut().get_mut::<Fluid2d>(entity).unwrap().reset();
    lifecycle::frames(&mut app, 3);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap();
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![1]
    );
    app.world_mut().despawn(entity);

    let faucet = Fluid2dFaucet {
        position: Vec2::ZERO,
        velocity: Vec2::ZERO,
        rows_per_second: 240.0,
        ..default()
    };
    let fluid = Fluid2d::from_faucet(settings, 7, faucet).unwrap();
    let bundle = Fluid2dBundle::new(
        fluid,
        &mut app.world_mut().resource_mut::<Assets<Image>>(),
        UVec2::splat(256),
    )
    .unwrap();
    let target = bundle.surface.image.clone();
    let region = Fluid2dRecycling::new(vec![Fluid2dCollider::Rectangle(Rect::new(
        -1.0, -1.0, 1.0, 1.0,
    ))])
    .unwrap();
    let entity = app.world_mut().spawn((bundle, region)).id();
    lifecycle::frames(&mut app, 10);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap();
    assert_eq!(state.particle_count(), 7);
    assert!(state.is_exhausted());
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![0]
    );
    assert!(
        surface_pixels(&app, &target).iter().all(|&v| v == 0),
        "default renderer must skip retired slots"
    );
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_faucet_open(false)
        .unwrap();
    lifecycle::frames(&mut app, 2);
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_faucet_open(true)
        .unwrap();
    lifecycle::frames(&mut app, 8);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap();
    assert_eq!(state.particle_count(), 7);
    assert_eq!(
        lifecycle::read_u32(&app, state.live_count_buffer()),
        vec![0]
    );
    {
        let mut fluid = app.world_mut().get_mut::<Fluid2d>(entity).unwrap();
        fluid.set_paused(true);
        fluid.reset();
    }
    lifecycle::frames(&mut app, 2);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap();
    assert_eq!(state.particle_count(), 0);
    assert!(!state.is_exhausted());
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_paused(false);
    lifecycle::frames(&mut app, 5);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
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
            .resource::<GpuFluids2d>()
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
    impl Fluid2dComputePass for Impulse {
        fn is_ready(&self, _: Entity, cache: &PipelineCache) -> bool {
            self.ready.load(Ordering::Relaxed) && cache.get_compute_pipeline(self.id).is_some()
        }
        fn encode(&self, stage: FluidComputeStage, ctx: &mut Fluid2dComputeContext<'_>) {
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
    impl Fluid2dComputePass for Probe {
        fn encode(&self, stage: FluidComputeStage, ctx: &mut Fluid2dComputeContext<'_>) {
            self.0.lock().unwrap().push((ctx.substep_index, stage, 1));
        }
    }
    let mut app = headless_app(Fluid2dRenderPlugin);
    let settings = Fluid2dSettings {
        gravity: Vec2::ZERO,
        stiffness: 0.0,
        viscosity: 0.0,
        ..default()
    };
    let dt = settings.time_step;
    let mut fluid = Fluid2d::new(settings, vec![FluidParticle2d::new(Vec2::ZERO)]).unwrap();
    fluid.set_paused(true);
    let entity = app.world_mut().spawn(fluid).id();
    lifecycle::frames(&mut app, 35);
    let shader = app
        .world_mut()
        .resource_mut::<Assets<Shader>>()
        .add(Shader::from_wgsl(
            r#"
        @group(0) @binding(0) var<storage,read_write> acceleration:array<vec2<f32>>;
        @group(0) @binding(1) var<storage,read> activity:array<u32>;
        @compute @workgroup_size(64) fn force(@builtin(global_invocation_id) id:vec3<u32>) {
            if id.x>=arrayLength(&activity) {return;}
            if activity[id.x]==0u {return;}
            acceleration[id.x]+=vec2<f32>(2.0,0.0);
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
                    storage_buffer_sized(false, BufferSize::new(8)),
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
        let state = world.resource::<GpuFluids2d>().get(entity).unwrap();
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
            .resource_mut::<Fluid2dComputePasses>();
        passes.insert(Impulse {
            id,
            group,
            ready: ready.clone(),
            trace: trace.clone(),
        });
        passes.insert(Probe(trace.clone()));
    }
    app.world_mut()
        .get_mut::<Fluid2d>(entity)
        .unwrap()
        .set_paused(false);
    lifecycle::frames(&mut app, 5);
    assert!(trace.lock().unwrap().is_empty());
    assert_eq!(
        app.sub_app(RenderApp)
            .world()
            .resource::<GpuFluids2d>()
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
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap();
    let steps = state.simulated_steps();
    assert!(steps > 1);
    let output = lifecycle::read_u32(&app, state.particle_buffer());
    let velocity = f32::from_bits(output[2]);
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
        .resource_mut::<Fluid2dComputePasses>()
        .remove::<Impulse>();
    lifecycle::frames(&mut app, 4);
    let state = app
        .sub_app(RenderApp)
        .world()
        .resource::<GpuFluids2d>()
        .get(entity)
        .unwrap();
    assert_eq!(
        f32::from_bits(lifecycle::read_u32(&app, state.particle_buffer())[2]),
        velocity
    );
}
