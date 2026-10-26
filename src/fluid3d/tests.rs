use super::{gpu::SimParams, *};
use bevy::{prelude::*, tasks::block_on};
use wgpu::util::DeviceExt;

const SPH: &str = include_str!("shaders/sph.wgsl");

#[test]
fn configuration_and_finite_budget_validation() {
    let settings = Fluid3dSettings::default();
    assert!(settings.validate().is_ok());
    assert_eq!(settings.grid_size(), UVec3::new(25, 19, 19));
    let mut fluid = Fluid3d::from_faucet(settings, 129, Fluid3dFaucet::default()).unwrap();
    assert_eq!(fluid.particle_count(), 129);
    assert!(fluid.seed.is_empty());
    let sphere = Fluid3dCollider::Sphere {
        center: Vec3::ZERO,
        radius: 0.5,
    };
    fluid.set_colliders(vec![sphere; MAX_COLLIDERS_3D]).unwrap();
    assert!(
        fluid
            .set_colliders(vec![sphere; MAX_COLLIDERS_3D + 1])
            .is_err()
    );
    assert_eq!(fluid.colliders().len(), 16);
    assert!(
        fluid
            .set_colliders(vec![Fluid3dCollider::Cuboid(Fluid3dBounds::new(
                Vec3::ZERO,
                Vec3::ZERO
            ))])
            .is_err()
    );
    assert!(
        fluid
            .set_faucet(Some(Fluid3dFaucet {
                depth: f32::NAN,
                ..default()
            }))
            .is_err()
    );
    let mut bad = fluid.settings().clone();
    bad.smoothing_radius = 1e-9;
    assert!(fluid.set_settings(bad).is_err());
    fluid.set_faucet_open(false).unwrap();
    fluid.reset();
    assert!(!fluid.faucet().unwrap().is_open);
    assert_eq!(fluid.particle_count(), 129);
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn gpu_matches_3d_reference_and_handles_dense_cells() {
    let gpu = Harness::new();
    let settings = Fluid3dSettings::default();
    let mut seed = FluidParticle3d::lattice(Vec3::new(-0.4, -0.4, -0.4), UVec3::new(5, 5, 6), 0.13);
    seed.truncate(129);
    seed[0] = FluidParticle3d::with_velocity(Vec3::new(3.92, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
    for (i, p) in seed.iter_mut().enumerate().skip(1) {
        p.velocity = [i as f32 * 0.01, -0.2, (i % 7) as f32 * 0.1, 0.0];
    }
    let (expected, states) = reference(&seed, &settings);
    let (actual, density) = gpu.run(&seed, &settings, 1);
    for i in 0..seed.len() {
        assert!(
            actual[i]
                .position_vec3()
                .distance(expected[i].position_vec3())
                < 0.0001,
            "position {i}"
        );
        assert!(
            actual[i]
                .velocity_vec3()
                .distance(expected[i].velocity_vec3())
                < 0.001,
            "velocity {i}"
        );
        assert!(
            (density[i][0] - states[i][0]).abs() / states[i][0] < 0.0001,
            "density {i}"
        );
    }
    assert!(actual[0].velocity[0] < 0.0);
    let dense = vec![FluidParticle3d::new(Vec3::ZERO); 129];
    let (_, density) = gpu.run(&dense, &settings, 1);
    let expected = 129.0 * settings.particle_mass * 315.0
        / (64.0 * std::f32::consts::PI * settings.smoothing_radius.powi(3));
    assert!((density[0][0] - expected).abs() / expected < 0.0001);
    let (evolved, _) = gpu.run(&seed, &settings, 600);
    for p in evolved {
        let pos = p.position_vec3();
        assert!(pos.is_finite() && p.velocity_vec3().is_finite());
        assert!(
            pos.cmpge(settings.bounds.min + Vec3::splat(settings.particle_radius))
                .all()
        );
        assert!(
            pos.cmple(settings.bounds.max - Vec3::splat(settings.particle_radius))
                .all()
        );
    }
    gpu.run(&[], &settings, 1);
    let (single, _) = gpu.run(&[FluidParticle3d::new(Vec3::ZERO)], &settings, 1);
    assert!((single[0].velocity[1] - settings.gravity.y * settings.time_step).abs() < 1e-6);
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn gpu_sweeps_spheres_cuboids_edges_and_corners() {
    let gpu = Harness::new();
    let settings = Fluid3dSettings {
        gravity: Vec3::ZERO,
        stiffness: 0.0,
        viscosity: 0.0,
        time_step: 0.02,
        max_speed: 200.0,
        restitution: 0.5,
        ..default()
    };
    let box_shape = Fluid3dCollider::Cuboid(Fluid3dBounds::new(
        Vec3::new(-0.01, -1.0, -1.0),
        Vec3::new(0.01, 1.0, 1.0),
    ));
    for collider in [
        box_shape,
        Fluid3dCollider::Sphere {
            center: Vec3::ZERO,
            radius: 0.25,
        },
    ] {
        for axis in 0..3 {
            let mut start = Vec3::ZERO;
            start[axis] = -2.0;
            let mut velocity = Vec3::ZERO;
            velocity[axis] = 200.0;
            let (out, _) = gpu.run_with_colliders(
                &[FluidParticle3d::with_velocity(start, velocity)],
                &settings,
                1,
                &[collider],
            );
            assert!(out[0].position[axis] < 0.0, "axis {axis}");
            assert!((out[0].velocity[axis] + 100.0).abs() < 1e-4);
        }
        let (out, _) = gpu.run_with_colliders(
            &[FluidParticle3d::new(Vec3::ZERO)],
            &settings,
            1,
            &[collider],
        );
        let p = out[0].position_vec3();
        assert!(p.is_finite());
        match collider {
            Fluid3dCollider::Sphere { center, radius } => {
                assert!(p.distance(center) >= radius + settings.particle_radius - 1e-5)
            }
            Fluid3dCollider::OrientedCuboid { .. } => unreachable!(),
            Fluid3dCollider::Cuboid(b) => {
                assert!(p.distance(p.clamp(b.min, b.max)) >= settings.particle_radius - 1e-5)
            }
        }
    }
    let rounded = Fluid3dCollider::Cuboid(Fluid3dBounds::new(Vec3::ZERO, Vec3::ONE));
    // Diagonal path through a rounded edge (outside both adjacent face rectangles).
    let (out, _) = gpu.run_with_colliders(
        &[FluidParticle3d::with_velocity(
            Vec3::new(-0.2, -0.2, 0.5),
            Vec3::new(20.0, 20.0, 0.0),
        )],
        &settings,
        1,
        &[rounded],
    );
    assert!(out[0].velocity[0] < 0.0 && out[0].velocity[1] < 0.0);
    // A corner miss would hit a naively square-expanded cuboid.
    let (out, _) = gpu.run_with_colliders(
        &[FluidParticle3d::with_velocity(
            Vec3::new(-0.1, -0.06, -0.06),
            Vec3::new(2.0, 0.0, 0.0),
        )],
        &settings,
        1,
        &[rounded],
    );
    assert!((out[0].velocity[0] - 2.0).abs() < 1e-5);
}

#[test]
fn shaders_validate() {
    for source in [
        SPH,
        include_str!("shaders/recycle.wgsl"),
        include_str!("shaders/particles.wgsl"),
        include_str!("shaders/surface.wgsl"),
        include_str!("../../examples/shaders/custom_3d.wgsl"),
    ] {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
    assert_eq!(size_of::<FluidParticle3d>(), 32);
    assert_eq!(size_of::<SimParams>(), 1184);
}

// Independent O(N^2) reference, with the normalized 3D kernels.
fn reference(
    seed: &[FluidParticle3d],
    s: &Fluid3dSettings,
) -> (Vec<FluidParticle3d>, Vec<[f32; 2]>) {
    let pi = std::f32::consts::PI;
    let h = s.smoothing_radius;
    let states: Vec<_> = seed
        .iter()
        .map(|p| {
            let rho = seed
                .iter()
                .map(|q| {
                    let q = (1.0
                        - p.position_vec3().distance_squared(q.position_vec3()) / h.powi(2))
                    .max(0.0);
                    s.particle_mass * 315.0 / (64.0 * pi * h.powi(3)) * q.powi(3)
                })
                .sum::<f32>()
                .max(s.rest_density * 0.01);
            [rho, s.stiffness * (rho - s.rest_density).max(0.0)]
        })
        .collect();
    let result = seed
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let mut a = s.gravity;
            for (j, n) in seed.iter().enumerate() {
                let delta = p.position_vec3() - n.position_vec3();
                let r = delta.length();
                if i == j || r <= 1e-6 || r >= h {
                    continue;
                }
                let q = 1.0 - r / h;
                let gradient = -45.0 / (pi * h.powi(4)) * q * q * delta / r;
                a -= s.particle_mass
                    * (states[i][1] / states[i][0].powi(2) + states[j][1] / states[j][0].powi(2))
                    * gradient;
                a += s.viscosity * s.particle_mass / states[j][0]
                    * (n.velocity_vec3() - p.velocity_vec3())
                    * (45.0 / (pi * h.powi(5)) * q);
            }
            a = a.clamp_length_max(s.max_acceleration);
            let mut velocity = (p.velocity_vec3() + a * s.time_step).clamp_length_max(s.max_speed);
            let mut position = p.position_vec3() + velocity * s.time_step;
            for axis in 0..3 {
                let lo = s.bounds.min[axis] + s.particle_radius;
                let hi = s.bounds.max[axis] - s.particle_radius;
                if position[axis] < lo {
                    position[axis] = lo;
                    velocity[axis] = velocity[axis].abs() * s.restitution;
                }
                if position[axis] > hi {
                    position[axis] = hi;
                    velocity[axis] = -velocity[axis].abs() * s.restitution;
                }
            }
            FluidParticle3d::with_velocity(position, velocity)
        })
        .collect();
    (result, states)
}

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
}
impl Harness {
    fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter required");
        let (device, queue) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        device.set_device_lost_callback(|reason, message| {
            eprintln!("GPU device lost: {reason:?}: {message}");
        });
        Self { device, queue }
    }

    fn run(
        &self,
        seed: &[FluidParticle3d],
        s: &Fluid3dSettings,
        steps: u32,
    ) -> (Vec<FluidParticle3d>, Vec<[f32; 2]>) {
        self.run_with_colliders(seed, s, steps, &[])
    }
    fn run_with_colliders(
        &self,
        seed: &[FluidParticle3d],
        s: &Fluid3dSettings,
        steps: u32,
        colliders: &[Fluid3dCollider],
    ) -> (Vec<FluidParticle3d>, Vec<[f32; 2]>) {
        let device = &self.device;
        let grid = s.grid_size();
        let n = seed.len().max(1) as u64;
        let zero = [FluidParticle3d::new(Vec3::ZERO)];
        let buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(
                    &SimParams::new(s, seed.len() as u32).with_colliders(colliders),
                ),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(if seed.is_empty() { &zero } else { seed }),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }),
            self.storage(grid.x as u64 * grid.y as u64 * grid.z as u64 * 4),
            self.storage(n * 4),
            self.storage(n * 8),
            self.storage(n * 16),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&vec![1u32; n as usize]),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&(seed.len() as u32)),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        ];
        let entries: Vec<_> = (0..8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: false }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SPH.into()),
        });
        let pipelines = [
            "clear_grid",
            "build_grid",
            "density_pressure",
            "acceleration",
            "integrate",
        ]
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: default(),
                cache: None,
            })
        });
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let mut encoder = device.create_command_encoder(&default());
        for step in 0..steps {
            for (i, pipeline) in pipelines.iter().enumerate() {
                let mut pass = encoder.begin_compute_pass(&default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &group, &[]);
                let count = if i == 0 {
                    grid.x * grid.y * grid.z
                } else {
                    seed.len() as u32
                };
                if count != 0 {
                    pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
                }
            }
            // Model frame-sized submissions rather than thousands of Metal
            // encoders in one command buffer; keep GPU work bounded like the plugin.
            if (step + 1) % s.max_substeps == 0 {
                self.queue.submit([encoder.finish()]);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                encoder = device.create_command_encoder(&default());
            }
        }
        // Copy all outputs in one readback, including lists for structural validation.
        let sizes = [
            n * 32,
            grid.x as u64 * grid.y as u64 * grid.z as u64 * 4,
            n * 4,
            n * 8,
        ];
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: sizes.iter().sum(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut offset = 0;
        for (i, size) in sizes.iter().enumerate() {
            encoder.copy_buffer_to_buffer(&buffers[i + 1], 0, &staging, offset, *size);
            offset += size;
        }
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap_or_else(|e| {
            panic!(
                "readback failed for {} particles / {steps} steps: {e}",
                seed.len()
            )
        });
        let bytes = staging.slice(..).get_mapped_range().unwrap();
        let particles = bytemuck::cast_slice::<u8, FluidParticle3d>(&bytes[..sizes[0] as usize])
            [..seed.len()]
            .to_vec();
        let head_end = (sizes[0] + sizes[1]) as usize;
        let next_end = head_end + sizes[2] as usize;
        let heads: &[u32] = bytemuck::cast_slice(&bytes[sizes[0] as usize..head_end]);
        let next: &[u32] = bytemuck::cast_slice(&bytes[head_end..next_end]);
        let mut visited = vec![false; seed.len()];
        for &head in heads {
            let mut j = head;
            while j != u32::MAX {
                assert!((j as usize) < seed.len(), "invalid linked-list index");
                assert!(
                    !visited[j as usize],
                    "cycle or duplicate particle in linked lists"
                );
                visited[j as usize] = true;
                j = next[j as usize];
            }
        }
        assert!(visited.iter().all(|v| *v), "grid dropped a particle");
        let density =
            bytemuck::cast_slice::<u8, [f32; 2]>(&bytes[next_end..])[..seed.len()].to_vec();
        (particles, density)
    }
    fn storage(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn oriented_cuboids_preserve_face_edge_corner_sweeps_and_inside_projection() {
    let gpu = Harness::new();
    let settings = Fluid3dSettings {
        bounds: Fluid3dBounds::new(Vec3::splat(-5.0), Vec3::splat(5.0)),
        gravity: Vec3::ZERO,
        stiffness: 0.0,
        viscosity: 0.0,
        time_step: 0.1,
        max_speed: 100.0,
        restitution: 0.5,
        ..default()
    };
    let local = vec![
        FluidParticle3d::with_velocity(Vec3::new(-1.0, 0.0, 0.0), Vec3::X * 20.0),
        FluidParticle3d::new(Vec3::ZERO),
        FluidParticle3d::with_velocity(Vec3::new(-1.0, 1.05, 0.25), Vec3::X * 20.0),
        FluidParticle3d::with_velocity(Vec3::new(-1.0, 1.035, 0.535), Vec3::X * 20.0),
        FluidParticle3d::with_velocity(Vec3::new(-1.0, 1.12, 0.62), Vec3::X * 20.0),
    ];
    let half = Vec3::new(0.02, 1.0, 0.5);
    let (reference, _) = gpu.run_with_colliders(
        &local,
        &settings,
        1,
        &[Fluid3dCollider::Cuboid(Fluid3dBounds::new(-half, half))],
    );
    for rotation in [
        Quat::IDENTITY,
        Quat::from_euler(EulerRot::XYZ, 0.4, 0.7, 0.2),
        Quat::from_rotation_y(1.5),
    ] {
        let center = Vec3::new(-0.4, 0.2, 0.3);
        let seed: Vec<_> = local
            .iter()
            .map(|p| {
                FluidParticle3d::with_velocity(
                    center + rotation * p.position_vec3(),
                    rotation * p.velocity_vec3(),
                )
            })
            .collect();
        let (actual, _) = gpu.run_with_colliders(
            &seed,
            &settings,
            1,
            &[Fluid3dCollider::OrientedCuboid {
                center,
                half_extents: half,
                rotation,
            }],
        );
        for (actual, expected) in actual.iter().zip(&reference) {
            assert!(
                actual
                    .position_vec3()
                    .distance(center + rotation * expected.position_vec3())
                    < 0.0005
            );
            assert!(
                actual
                    .velocity_vec3()
                    .distance(rotation * expected.velocity_vec3())
                    < 0.002
            );
        }
    }
}

#[test]
fn oriented_cuboid_and_recycling_validation_is_transactional() {
    let good = Fluid3dCollider::OrientedCuboid {
        center: Vec3::ZERO,
        half_extents: Vec3::ONE,
        rotation: Quat::from_rotation_x(0.4),
    };
    let bad = Fluid3dCollider::OrientedCuboid {
        center: Vec3::ZERO,
        half_extents: Vec3::ONE,
        rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
    };
    let mut fluid = Fluid3d::new(Fluid3dSettings::default(), vec![]).unwrap();
    fluid.set_colliders(vec![good]).unwrap();
    assert!(fluid.set_colliders(vec![bad]).is_err());
    assert_eq!(fluid.colliders(), &[good]);
    let mut regions = Fluid3dRecycling::new(vec![good]).unwrap();
    assert!(regions.set_regions(vec![good; 17]).is_err());
    assert!(regions.set_regions(vec![bad]).is_err());
    assert_eq!(regions.regions(), &[good]);
}
