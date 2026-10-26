//! Primitive-only sink geometry shared by the runnable example and GPU regression tests.
use bevy::prelude::*;
use bevy_gpu_fluid::prelude::*;

#[derive(Resource)]
pub struct SinkDemo {
    pub recycling_enabled: bool,
}

pub struct SinkGeometry {
    pub walls: [Fluid3dBounds; 4],
    pub bottom: [Fluid3dBounds; 4],
    pub countertop: [Fluid3dBounds; 4],
    pub collector: Fluid3dBounds,
}

fn bounds(min: [f32; 3], max: [f32; 3]) -> Fluid3dBounds {
    Fluid3dBounds::new(Vec3::from_array(min), Vec3::from_array(max))
}

impl Default for SinkGeometry {
    fn default() -> Self {
        Self {
            // Four walls, sharing edges with the bottom and the countertop.
            walls: [
                bounds([-2.25, -0.28, -1.6], [-2.05, 0.85, 1.6]),
                bounds([2.05, -0.28, -1.6], [2.25, 0.85, 1.6]),
                bounds([-2.05, -0.28, -1.6], [2.05, 0.85, -1.4]),
                bounds([-2.05, -0.28, 1.4], [2.05, 0.85, 1.6]),
            ],
            // Four separate tiles. The central 0.72 x 0.72 opening has no collider.
            bottom: [
                bounds([-2.05, -0.28, -1.4], [-0.36, 0.0, 1.4]),
                bounds([0.36, -0.28, -1.4], [2.05, 0.0, 1.4]),
                bounds([-0.36, -0.28, -1.4], [0.36, 0.0, -0.36]),
                bounds([-0.36, -0.28, 0.36], [0.36, 0.0, 1.4]),
            ],
            // A second open ring at rim height: spills can run off its outer edge.
            countertop: [
                bounds([-3.25, 0.65, -2.3], [-2.25, 0.85, 2.3]),
                bounds([2.25, 0.65, -2.3], [3.25, 0.85, 2.3]),
                bounds([-2.25, 0.65, -2.3], [2.25, 0.85, -1.6]),
                bounds([-2.25, 0.65, 1.6], [2.25, 0.85, 2.3]),
            ],
            // Span the entire simulation XZ footprint, below both the hole and table.
            // Collect before the container's global bottom wall can trap particles.
            collector: bounds([-4.05, -3.1, -3.05], [4.05, -2.65, 3.05]),
        }
    }
}

impl SinkGeometry {
    pub fn colliders(&self) -> Vec<Fluid3dCollider> {
        self.walls
            .iter()
            .chain(&self.bottom)
            .chain(&self.countertop)
            .copied()
            .map(Fluid3dCollider::Cuboid)
            .collect()
    }
}

pub fn faucet() -> Fluid3dFaucet {
    Fluid3dFaucet {
        // Offset from the hole so the jet wets the basin before draining.
        position: Vec3::new(0.75, 2.3, -0.4),
        width: 0.64,
        depth: 0.64,
        ..default()
    }
}

#[derive(Component)]
pub struct FaucetVisual;
#[derive(Component)]
pub struct OutletMouth;

fn blocks(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: &Handle<StandardMaterial>,
    pieces: &[Fluid3dBounds],
) {
    for piece in pieces {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::from_size(piece.size()))),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(piece.center()),
        ));
    }
}

pub fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    demo: Res<SinkDemo>,
) {
    let geometry = SinkGeometry::default();
    let outlet = faucet();
    let mut fluid = Fluid3d::from_faucet(Fluid3dSettings::default(), 8000, outlet).unwrap();
    fluid.set_colliders(geometry.colliders()).unwrap();
    commands.spawn((
        fluid,
        Fluid3dRender::default(),
        Fluid3dRecycling::new(vec![Fluid3dCollider::Cuboid(geometry.collector)]).unwrap(),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(7.8, 5.8, 9.2).looking_at(Vec3::new(0.0, -0.3, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            ..default()
        },
        Transform::from_xyz(3.0, 8.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    let enamel = materials.add(StandardMaterial {
        base_color: Color::srgb(0.72, 0.77, 0.8),
        perceptual_roughness: 0.65,
        ..default()
    });
    let tabletop = materials.add(StandardMaterial {
        base_color: Color::srgb(0.25, 0.17, 0.11),
        perceptual_roughness: 0.8,
        ..default()
    });
    let tray = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 0.25, 0.2),
        perceptual_roughness: 0.8,
        ..default()
    });
    blocks(&mut commands, &mut meshes, &enamel, &geometry.walls);
    blocks(&mut commands, &mut meshes, &enamel, &geometry.bottom);
    blocks(&mut commands, &mut meshes, &tabletop, &geometry.countertop);

    // Visual catch box: the trigger handles collection; these meshes do not seal the drain.
    // Low front wall leaves the falling stream and the inside of the tray visible.
    let tray_pieces = [
        bounds([-4.2, -3.18, -3.2], [4.2, -3.0, 3.2]),
        bounds([-4.2, -3.0, -3.2], [-4.05, -2.05, 3.2]),
        bounds([4.05, -3.0, -3.2], [4.2, -2.05, 3.2]),
        bounds([-4.05, -3.0, -3.2], [4.05, -2.05, -3.05]),
        bounds([-4.05, -3.0, 3.05], [4.05, -2.65, 3.2]),
    ];
    blocks(&mut commands, &mut meshes, &tray, &tray_pieces);
    if demo.recycling_enabled {
        let active = materials.add(StandardMaterial {
            base_color: Color::srgb(0.12, 0.48, 0.3),
            unlit: true,
            ..default()
        });
        blocks(
            &mut commands,
            &mut meshes,
            &active,
            &[bounds([-4.02, -3.0, -3.02], [4.02, -2.98, 3.02])],
        );
    }

    let metal = materials.add(StandardMaterial {
        base_color: Color::srgb(0.45, 0.52, 0.57),
        metallic: 0.65,
        perceptual_roughness: 0.28,
        ..default()
    });
    let mouth = meshes.add(Cuboid::default());
    let horizontal = meshes.add(Cuboid::new(0.28, 0.28, 1.2));
    let vertical = meshes.add(Cuboid::new(0.28, 1.85, 0.28));
    commands
        .spawn((
            FaucetVisual,
            Transform::from_translation(outlet.position),
            Visibility::default(),
        ))
        .with_children(|parent| {
            parent.spawn((
                OutletMouth,
                Mesh3d(mouth),
                MeshMaterial3d(metal.clone()),
                Transform::from_xyz(0.0, 0.16, 0.0).with_scale(Vec3::new(
                    outlet.width + 0.12,
                    0.18,
                    outlet.depth + 0.12,
                )),
            ));
            parent.spawn((
                Mesh3d(horizontal),
                MeshMaterial3d(metal.clone()),
                Transform::from_xyz(0.0, 0.36, -0.55),
            ));
            parent.spawn((
                Mesh3d(vertical),
                MeshMaterial3d(metal),
                Transform::from_xyz(0.0, -0.525, -1.1),
            ));
        });
}

pub fn sync_faucet(
    fluids: Query<&Fluid3d>,
    mut roots: Query<&mut Transform, (With<FaucetVisual>, Without<OutletMouth>)>,
    mut mouths: Query<&mut Transform, (With<OutletMouth>, Without<FaucetVisual>)>,
) {
    let Some(outlet) = fluids.iter().find_map(Fluid3d::faucet) else {
        return;
    };
    for mut transform in &mut roots {
        transform.translation = outlet.position;
    }
    for mut transform in &mut mouths {
        transform.scale = Vec3::new(outlet.width + 0.12, 0.18, outlet.depth + 0.12);
    }
}
