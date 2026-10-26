//! Finite faucet and mixed obstacles. Space: valve, R: refill/reset,
//! P: pause, A/D: move outlet, W/S: change width. Pass --recycle to install the optional drain plugin.
use bevy::{camera::ScalingMode, prelude::*};
use bevy_gpu_fluid::prelude::*;

#[derive(Component)]
struct OutletVisual;

#[derive(Resource)]
struct RecyclingDemo(bool);

fn main() {
    let recycle = std::env::args().any(|arg| arg == "--recycle");
    let mut app = App::new();
    let windows = WindowPlugin {
        primary_window: Some(Window {
            title: "Finite faucet | Space: valve | R: reset | P: pause | A/D: move | W/S: width"
                .into(),
            resolution: (1280, 800).into(),
            ..default()
        }),
        ..default()
    };
    app.insert_resource(ClearColor(Color::srgb(0.018, 0.025, 0.045)))
        .add_plugins((DefaultPlugins.set(windows), Fluid2dPlugins));
    if recycle {
        app.add_plugins(Fluid2dRecyclingPlugin);
    }
    app.insert_resource(RecyclingDemo(recycle))
        .add_systems(Startup, setup)
        .add_systems(Update, controls)
        .run();
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    recycling: Res<RecyclingDemo>,
) {
    commands.spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 10.0,
            },
            ..OrthographicProjection::default_2d()
        }),
    ));
    let mut fluid =
        Fluid2d::from_faucet(Fluid2dSettings::default(), 1920, Fluid2dFaucet::default()).unwrap();
    let obstacles = vec![
        Fluid2dCollider::Circle {
            center: Vec2::new(0.0, 0.8),
            radius: 0.8,
        },
        Fluid2dCollider::OrientedRectangle {
            center: Vec2::new(-2.1, -0.85),
            half_extents: Vec2::new(1.4, 0.15),
            rotation: -0.28,
        },
        Fluid2dCollider::Rectangle(Rect::new(0.7, -2.0, 3.5, -1.7)),
    ];
    for collider in &obstacles {
        let color = Color::srgb(0.28, 0.35, 0.43);
        match *collider {
            Fluid2dCollider::Rectangle(rect) => {
                commands.spawn((
                    Sprite::from_color(color, rect.size()),
                    Transform::from_translation(rect.center().extend(1.0)),
                ));
            }
            Fluid2dCollider::OrientedRectangle {
                center,
                half_extents,
                rotation,
            } => {
                commands.spawn((
                    Sprite::from_color(color, half_extents * 2.0),
                    Transform::from_translation(center.extend(1.0))
                        .with_rotation(Quat::from_rotation_z(rotation)),
                ));
            }
            Fluid2dCollider::Circle { center, radius } => {
                commands.spawn((
                    Mesh2d(meshes.add(Circle::new(radius))),
                    MeshMaterial2d(materials.add(color)),
                    Transform::from_translation(center.extend(1.0)),
                ));
            }
        }
    }
    fluid.set_colliders(obstacles).unwrap();
    let drain = Rect::new(-7.8, -4.3, 7.8, -3.9);
    commands.spawn((
        Fluid2dBundle::new(fluid, &mut images, UVec2::new(1280, 720)).unwrap(),
        Fluid2dRecycling::new(vec![Fluid2dCollider::Rectangle(drain)]).unwrap(),
    ));
    if recycling.0 {
        commands.spawn((
            Sprite::from_color(Color::srgba(0.1, 0.85, 0.35, 0.35), drain.size()),
            Transform::from_translation(drain.center().extend(-0.1)),
        ));
    }
    commands.spawn((
        OutletVisual,
        Sprite::from_color(Color::srgb(0.2, 0.9, 0.4), Vec2::new(1.6, 0.12)),
        Transform::from_xyz(0.0, 3.65, 2.0),
    ));
}

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut fluids: Query<&mut Fluid2d>,
    mut visual: Single<(&mut Sprite, &mut Transform), With<OutletVisual>>,
) {
    let mut outlet = None;
    for mut fluid in &mut fluids {
        if keys.just_pressed(KeyCode::KeyR) {
            fluid.reset();
        }
        if keys.just_pressed(KeyCode::KeyP) {
            let paused = fluid.is_paused();
            fluid.set_paused(!paused);
        }
        let Some(mut f) = fluid.faucet().copied() else {
            continue;
        };
        if keys.just_pressed(KeyCode::Space) {
            f.is_open = !f.is_open;
        }
        let dx = i32::from(keys.pressed(KeyCode::KeyD)) - i32::from(keys.pressed(KeyCode::KeyA));
        let dw = i32::from(keys.pressed(KeyCode::KeyW)) - i32::from(keys.pressed(KeyCode::KeyS));
        f.width = (f.width + dw as f32 * time.delta_secs()).clamp(0.16, 5.0);
        f.position.x = (f.position.x + dx as f32 * time.delta_secs() * 3.0)
            .clamp(-7.8 + f.width * 0.5, 7.8 - f.width * 0.5);
        fluid.set_faucet(Some(f)).unwrap();
        outlet = Some(f);
    }
    if let Some(f) = outlet {
        visual.0.custom_size = Some(Vec2::new(f.width, 0.12));
        visual.0.color = if f.is_open {
            Color::srgb(0.2, 0.9, 0.4)
        } else {
            Color::srgb(0.9, 0.2, 0.2)
        };
        visual.1.translation = (f.position + Vec2::new(0.0, 0.15)).extend(2.0);
    }
}
