//! Run with `cargo run --example dam_break`. Space: pause, R: reset, arrows: tilt gravity.
use bevy::{camera::ScalingMode, prelude::*};
use bevy_gpu_fluid::prelude::*;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.018, 0.025, 0.045)))
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "GPU SPH 2D | Space: pause | R: reset | arrows: gravity".into(),
                    resolution: (1280, 800).into(),
                    ..default()
                }),
                ..default()
            }),
            Fluid2dPlugins,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, controls)
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    commands.spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 10.0,
            },
            ..OrthographicProjection::default_2d()
        }),
    ));
    let settings = Fluid2dSettings::default();
    let particles = FluidParticle2d::lattice(Vec2::new(-7.5, -3.9), UVec2::new(40, 48), 0.16);
    let fluid = Fluid2d::new(settings, particles).unwrap();
    commands.spawn(Fluid2dBundle::new(fluid, &mut images, UVec2::new(1280, 720)).unwrap());
}

fn controls(keys: Res<ButtonInput<KeyCode>>, mut fluids: Query<&mut Fluid2d>) {
    for mut fluid in &mut fluids {
        if keys.just_pressed(KeyCode::Space) {
            let paused = fluid.is_paused();
            fluid.set_paused(!paused);
        }
        if keys.just_pressed(KeyCode::KeyR) {
            fluid.reset();
        }
        let x = i32::from(keys.pressed(KeyCode::ArrowRight))
            - i32::from(keys.pressed(KeyCode::ArrowLeft));
        let y =
            i32::from(keys.pressed(KeyCode::ArrowUp)) - i32::from(keys.pressed(KeyCode::ArrowDown));
        let mut settings = fluid.settings().clone();
        settings.gravity = Vec2::new(x as f32 * 15.0, -9.81 + y as f32 * 20.0);
        if &settings != fluid.settings() {
            fluid.set_settings(settings).unwrap();
        }
    }
}
