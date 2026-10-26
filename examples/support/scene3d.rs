use bevy::prelude::*;
use bevy_gpu_fluid::prelude::*;

pub fn scene(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) -> Vec<Fluid3dCollider> {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(8.0, 5.5, 10.0).looking_at(Vec3::new(0.0, -0.3, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.22, 0.27, 0.34),
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(8.2, 0.2, 6.2))),
        MeshMaterial3d(material.clone()),
        Transform::from_xyz(0.0, -3.1, 0.0),
    ));
    let obstacles = vec![
        Fluid3dCollider::Sphere {
            center: Vec3::new(0.0, 0.2, 0.0),
            radius: 0.65,
        },
        Fluid3dCollider::OrientedCuboid {
            center: Vec3::new(1.9, -1.25, 0.0),
            half_extents: Vec3::new(1.1, 0.15, 1.5),
            rotation: Quat::from_euler(EulerRot::XYZ, 0.12, 0.18, -0.22),
        },
    ];
    for obstacle in &obstacles {
        match *obstacle {
            Fluid3dCollider::Sphere { center, radius } => {
                commands.spawn((
                    Mesh3d(meshes.add(Sphere::new(radius))),
                    MeshMaterial3d(material.clone()),
                    Transform::from_translation(center),
                ));
            }
            Fluid3dCollider::OrientedCuboid {
                center,
                half_extents,
                rotation,
            } => {
                commands.spawn((
                    Mesh3d(meshes.add(Cuboid::from_size(half_extents * 2.0))),
                    MeshMaterial3d(material.clone()),
                    Transform::from_translation(center).with_rotation(rotation),
                ));
            }
            Fluid3dCollider::Cuboid(b) => {
                commands.spawn((
                    Mesh3d(meshes.add(Cuboid::from_size(b.size()))),
                    MeshMaterial3d(material.clone()),
                    Transform::from_translation(b.center()),
                ));
            }
        }
    }
    obstacles
}

pub fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut fluids: Query<&mut Fluid3d>,
    mut cameras: Query<&mut Transform, With<Camera3d>>,
) {
    let dt = time.delta_secs();
    for mut fluid in &mut fluids {
        if keys.just_pressed(KeyCode::KeyR) {
            fluid.reset();
        }
        if keys.just_pressed(KeyCode::KeyP)
            || (fluid.faucet().is_none() && keys.just_pressed(KeyCode::Space))
        {
            let paused = fluid.is_paused();
            fluid.set_paused(!paused);
        }
        if let Some(mut f) = fluid.faucet().copied() {
            if keys.just_pressed(KeyCode::Space) {
                f.is_open = !f.is_open;
            }
            let x = i32::from(keys.pressed(KeyCode::KeyD)) - i32::from(keys.pressed(KeyCode::KeyA));
            let z = i32::from(keys.pressed(KeyCode::KeyE)) - i32::from(keys.pressed(KeyCode::KeyQ));
            let width =
                i32::from(keys.pressed(KeyCode::KeyW)) - i32::from(keys.pressed(KeyCode::KeyS));
            f.width = (f.width + width as f32 * dt).clamp(0.16, 2.8);
            f.depth = f.width;
            f.position.x = (f.position.x + x as f32 * dt * 2.0)
                .clamp(-3.8 + f.width * 0.5, 3.8 - f.width * 0.5);
            f.position.z = (f.position.z + z as f32 * dt * 2.0)
                .clamp(-2.8 + f.depth * 0.5, 2.8 - f.depth * 0.5);
            fluid.set_faucet(Some(f)).unwrap();
        }
    }
    let rotation =
        i32::from(keys.pressed(KeyCode::ArrowRight)) - i32::from(keys.pressed(KeyCode::ArrowLeft));
    let elevation =
        i32::from(keys.pressed(KeyCode::ArrowUp)) - i32::from(keys.pressed(KeyCode::ArrowDown));
    for mut camera in &mut cameras {
        camera.translation = Quat::from_rotation_y(rotation as f32 * dt) * camera.translation;
        camera.translation.y =
            (camera.translation.y + elevation as f32 * dt * 4.0).clamp(0.0, 12.0);
        camera.look_at(Vec3::new(0.0, -0.3, 0.0), Vec3::Y);
    }
}
