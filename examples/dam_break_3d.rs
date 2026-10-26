//! Independent 3D SPH. Arrows: orbit camera, Space/P: pause, R: reset.
use bevy::prelude::*;
use bevy_gpu_fluid::prelude::*;
#[path = "support/scene3d.rs"]
mod scene3d;
fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.018, 0.025, 0.045)))
        .add_plugins((DefaultPlugins, Fluid3dPlugins))
        .add_systems(Startup, setup)
        .add_systems(Update, scene3d::controls)
        .run();
}
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let obstacles = scene3d::scene(&mut commands, &mut meshes, &mut materials);
    let particles =
        FluidParticle3d::lattice(Vec3::new(-3.6, -2.5, -1.2), UVec3::new(16, 24, 16), 0.16);
    let mut fluid = Fluid3d::new(Fluid3dSettings::default(), particles).unwrap();
    fluid.set_colliders(obstacles).unwrap();
    commands.spawn((fluid, Fluid3dRender::default()));
}
