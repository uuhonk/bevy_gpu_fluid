//! Sink: four walls, four bottom tiles with a drain hole, and a large collector below.
//! Recycling is installed by default; use --no-recycle to compare without it.
//! Space: valve, P: pause, R: refill, A/D: X, Q/E: Z, W/S: outlet size, arrows: camera.
use bevy::prelude::*;
use bevy_gpu_fluid::prelude::*;
// Reuse the shared controls; the old obstacle scene is intentionally unused here.
#[allow(dead_code)]
#[path = "support/scene3d.rs"]
mod scene3d;
#[path = "support/sink3d.rs"]
mod sink3d;

fn main() {
    let recycle = !std::env::args().any(|arg| arg == "--no-recycle");
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::srgb(0.018, 0.025, 0.045)))
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Sink faucet | Space: valve | P: pause | R: refill | A/D, Q/E: move | W/S: size".into(),
                    resolution: (1280, 800).into(),
                    ..default()
                }),
                ..default()
            }),
            Fluid3dPlugins,
        ));
    if recycle {
        app.add_plugins(Fluid3dRecyclingPlugin);
    }
    app.insert_resource(sink3d::SinkDemo {
        recycling_enabled: recycle,
    })
    .add_systems(Startup, sink3d::setup)
    .add_systems(
        Update,
        (
            scene3d::controls,
            sink3d::sync_faucet.after(scene3d::controls),
        ),
    )
    .run();
}
