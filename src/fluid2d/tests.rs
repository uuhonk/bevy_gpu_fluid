use super::{gpu::SimParams, *};
use bevy::{prelude::*, tasks::block_on};
use wgpu::util::DeviceExt;

const SPH: &str = include_str!("shaders/sph.wgsl");

#[test]
fn shaders_validate() {
    for source in [
        SPH,
        include_str!("shaders/recycle.wgsl"),
        include_str!("shaders/particles.wgsl"),
        include_str!("shaders/surface.wgsl"),
        include_str!("../../examples/shaders/custom.wgsl"),
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
    assert_eq!(size_of::<FluidParticle2d>(), 16);
    assert_eq!(size_of::<SimParams>(), 896);
}

// Independent all-pairs reference: deliberately has no grid or linked-list logic.
fn reference(
    seed: &[FluidParticle2d],
    s: &Fluid2dSettings,
) -> (Vec<FluidParticle2d>, Vec<[f32; 2]>) {
    let pi = std::f32::consts::PI;
    let h = s.smoothing_radius;
    let states: Vec<_> = seed
        .iter()
        .map(|p| {
            let rho: f32 = seed
                .iter()
                .map(|n| {
                    let r2 =
                        Vec2::from_array(p.position).distance_squared(Vec2::from_array(n.position));
                    let q = (1.0 - r2 / (h * h)).max(0.0);
                    s.particle_mass * 4.0 / (pi * h * h) * q.powi(3)
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
            let pos = Vec2::from_array(p.position);
            let vel = Vec2::from_array(p.velocity);
            let [rho, pressure] = states[i];
            let mut a = s.gravity;
            for (j, n) in seed.iter().enumerate() {
                let delta = pos - Vec2::from_array(n.position);
                let r = delta.length();
                if i == j || r <= 1e-6 || r >= h {
                    continue;
                }
                let q = 1.0 - r / h;
                let grad = -30.0 / (pi * h.powi(3)) * q * q * delta / r;
                a -= s.particle_mass
                    * (pressure / rho.powi(2) + states[j][1] / states[j][0].powi(2))
                    * grad;
                a += s.viscosity * s.particle_mass / states[j][0]
                    * (Vec2::from_array(n.velocity) - vel)
                    * (40.0 / (pi * h.powi(4)) * q);
            }
            a = a.clamp_length_max(s.max_acceleration);
            let mut velocity = (vel + a * s.time_step).clamp_length_max(s.max_speed);
            let mut position = pos + velocity * s.time_step;
            for axis in 0..2 {
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
            FluidParticle2d {
                position: position.to_array(),
                velocity: velocity.to_array(),
            }
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
        seed: &[FluidParticle2d],
        s: &Fluid2dSettings,
        steps: u32,
    ) -> (Vec<FluidParticle2d>, Vec<[f32; 2]>) {
        self.run_with_colliders(seed, s, steps, &[])
    }
    fn run_with_colliders(
        &self,
        seed: &[FluidParticle2d],
        s: &Fluid2dSettings,
        steps: u32,
        colliders: &[Fluid2dCollider],
    ) -> (Vec<FluidParticle2d>, Vec<[f32; 2]>) {
        let device = &self.device;
        let grid = s.grid_size();
        let n = seed.len().max(1) as u64;
        let zero = [FluidParticle2d::new(Vec2::ZERO)];
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
            self.storage(grid.x as u64 * grid.y as u64 * 4),
            self.storage(n * 4),
            self.storage(n * 8),
            self.storage(n * 8),
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
                    grid.x * grid.y
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
        let sizes = [n * 16, grid.x as u64 * grid.y as u64 * 4, n * 4, n * 8];
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
        let particles = bytemuck::cast_slice::<u8, FluidParticle2d>(&bytes[..sizes[0] as usize])
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
#[ignore = "requires a compute-capable GPU; cargo test --lib -- --include-ignored"]
fn gpu_matches_all_pairs_reference_and_remains_bounded() {
    let gpu = Harness::new();
    let settings = Fluid2dSettings {
        bounds: Rect::new(-2.0, -2.0, 2.0, 2.0),
        ..default()
    };
    let mut seed = FluidParticle2d::lattice(Vec2::new(-1.5, -1.0), UVec2::new(17, 8), 0.13);
    seed.truncate(129); // partial last workgroup
    seed[0] = FluidParticle2d {
        position: [1.92, 0.0],
        velocity: [10.0, 0.0],
    };
    for (i, p) in seed.iter_mut().enumerate().skip(1) {
        p.velocity = [(i % 3) as f32, -((i % 7) as f32) * 0.1];
    }
    let (expected, expected_density) = reference(&seed, &settings);
    let (actual, density) = gpu.run(&seed, &settings, 1);
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            Vec2::from_array(a.position).distance(Vec2::from_array(e.position)) < 0.0001,
            "position mismatch {i}"
        );
        assert!(
            Vec2::from_array(a.velocity).distance(Vec2::from_array(e.velocity)) < 0.001,
            "velocity mismatch {i}: {a:?} vs {e:?}"
        );
        assert!((density[i][0] - expected_density[i][0]).abs() / expected_density[i][0] < 0.0001);
    }
    assert!(
        actual[0].velocity[0] < 0.0,
        "wall must reflect outgoing velocity"
    );
    // High-occupancy cell: the grid must neither truncate nor cycle.
    let dense = vec![FluidParticle2d::new(Vec2::ZERO); 129];
    let (_, density) = gpu.run(&dense, &settings, 1);
    let expected = 129.0 * settings.particle_mass * 4.0
        / (std::f32::consts::PI * settings.smoothing_radius.powi(2));
    assert!((density[0][0] - expected).abs() / expected < 0.0001);
    let (evolved, _) = gpu.run(&seed, &settings, 600);
    for p in evolved {
        let pos = Vec2::from_array(p.position);
        assert!(pos.is_finite() && Vec2::from_array(p.velocity).is_finite());
        assert!(
            pos.cmpge(settings.bounds.min + Vec2::splat(settings.particle_radius))
                .all()
        );
        assert!(
            pos.cmple(settings.bounds.max - Vec2::splat(settings.particle_radius))
                .all()
        );
    }
    gpu.run(&[], &settings, 1);
    let (single, _) = gpu.run(&[FluidParticle2d::new(Vec2::ZERO)], &settings, 1);
    assert!((single[0].velocity[1] - settings.gravity.y * settings.time_step).abs() < 1e-6);
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn gpu_sweeps_thin_rectangles_circles_and_projects_interior_seeds() {
    let gpu = Harness::new();
    let settings = Fluid2dSettings {
        gravity: Vec2::ZERO,
        stiffness: 0.0,
        viscosity: 0.0,
        time_step: 0.02,
        max_speed: 200.0,
        restitution: 0.5,
        ..default()
    };
    let rectangle = Fluid2dCollider::Rectangle(Rect::new(-0.01, -1.0, 0.01, 1.0));
    let circle = Fluid2dCollider::Circle {
        center: Vec2::ZERO,
        radius: 0.25,
    };
    for collider in [rectangle, circle] {
        // End position is beyond the solid, so endpoint-only tests would miss it.
        let seed = [FluidParticle2d {
            position: [-1.0, 0.0],
            velocity: [100.0, 0.0],
        }];
        let (out, _) = gpu.run_with_colliders(&seed, &settings, 1, &[collider]);
        assert!(out[0].position[0] < -settings.particle_radius);
        assert!((out[0].velocity[0] + 50.0).abs() < 1e-4);
        let (inside, _) = gpu.run_with_colliders(
            &[FluidParticle2d::new(Vec2::ZERO)],
            &settings,
            1,
            &[collider],
        );
        let p = Vec2::from_array(inside[0].position);
        assert!(p.is_finite());
        match collider {
            Fluid2dCollider::Circle { center, radius } => {
                assert!(p.distance(center) >= radius + settings.particle_radius - 1e-5)
            }
            Fluid2dCollider::OrientedRectangle { .. } => unreachable!(),
            Fluid2dCollider::Rectangle(rect) => {
                assert!(p.distance(p.clamp(rect.min, rect.max)) >= settings.particle_radius - 1e-5)
            }
        }
    }
    // Two nearby walls require more than one impact within the same step.
    let walls = [
        rectangle,
        Fluid2dCollider::Rectangle(Rect::new(-1.5, -1.0, -1.4, 1.0)),
    ];
    let (out, _) = gpu.run_with_colliders(
        &[FluidParticle2d {
            position: [-1.0, 0.0],
            velocity: [200.0, 0.0],
        }],
        &settings,
        1,
        &walls,
    );
    assert!(out[0].position[0] > -1.4 + settings.particle_radius - 1e-5);
    assert!(out[0].position[0] < -0.01 - settings.particle_radius + 1e-5);
    // The rounded corner must not behave as a square-expanded box.
    let corner = Fluid2dCollider::Rectangle(Rect::new(0.0, 0.0, 1.0, 1.0));
    let (out, _) = gpu.run_with_colliders(
        &[FluidParticle2d {
            position: [-0.1, -0.06],
            velocity: [2.0, 0.0],
        }],
        &settings,
        1,
        &[corner],
    );
    assert!((out[0].velocity[0] - 2.0).abs() < 1e-5);
}

#[test]
#[ignore = "requires a compute-capable GPU"]
fn oriented_rectangles_preserve_sweeps_corner_normals_and_interior_projection() {
    let gpu = Harness::new();
    let settings = Fluid2dSettings {
        bounds: Rect::new(-5.0, -5.0, 5.0, 5.0),
        gravity: Vec2::ZERO,
        stiffness: 0.0,
        viscosity: 0.0,
        time_step: 0.1,
        max_speed: 100.0,
        restitution: 0.5,
        ..default()
    };
    let local = vec![
        FluidParticle2d {
            position: [-1.0, 0.0],
            velocity: [20.0, 0.0],
        },
        FluidParticle2d::new(Vec2::ZERO),
        FluidParticle2d {
            position: [-1.0, 1.05],
            velocity: [20.0, 0.0],
        },
        FluidParticle2d {
            position: [-1.0, 1.15],
            velocity: [20.0, 0.0],
        },
    ];
    let half = Vec2::new(0.02, 1.0);
    let (reference, _) = gpu.run_with_colliders(
        &local,
        &settings,
        1,
        &[Fluid2dCollider::Rectangle(Rect::from_center_size(
            Vec2::ZERO,
            half * 2.0,
        ))],
    );
    for angle in [0.0, 0.63, 1.7, std::f32::consts::PI] {
        let rotation = Mat2::from_angle(angle);
        let center = Vec2::new(0.3, -0.2);
        let seed: Vec<_> = local
            .iter()
            .map(|p| FluidParticle2d {
                position: (center + rotation * Vec2::from_array(p.position)).to_array(),
                velocity: (rotation * Vec2::from_array(p.velocity)).to_array(),
            })
            .collect();
        let (actual, _) = gpu.run_with_colliders(
            &seed,
            &settings,
            1,
            &[Fluid2dCollider::OrientedRectangle {
                center,
                half_extents: half,
                rotation: angle,
            }],
        );
        for (actual, expected) in actual.iter().zip(&reference) {
            assert!(
                Vec2::from_array(actual.position)
                    .distance(center + rotation * Vec2::from_array(expected.position))
                    < 0.0005
            );
            assert!(
                Vec2::from_array(actual.velocity)
                    .distance(rotation * Vec2::from_array(expected.velocity))
                    < 0.002
            );
        }
    }
}

#[test]
fn oriented_rectangle_and_recycling_validation_is_transactional() {
    let good = Fluid2dCollider::OrientedRectangle {
        center: Vec2::ZERO,
        half_extents: Vec2::ONE,
        rotation: 0.4,
    };
    let mut fluid = Fluid2d::new(Fluid2dSettings::default(), vec![]).unwrap();
    fluid.set_colliders(vec![good]).unwrap();
    let bad = Fluid2dCollider::OrientedRectangle {
        rotation: f32::NAN,
        center: Vec2::ZERO,
        half_extents: Vec2::ONE,
    };
    assert!(fluid.set_colliders(vec![bad]).is_err());
    assert_eq!(fluid.colliders(), &[good]);
    let mut regions = Fluid2dRecycling::new(vec![good]).unwrap();
    assert!(regions.set_regions(vec![good; 17]).is_err());
    assert!(regions.set_regions(vec![bad]).is_err());
    assert_eq!(regions.regions(), &[good]);
}
