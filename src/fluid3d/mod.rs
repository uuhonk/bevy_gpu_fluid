//! Independent 3D SPH: three-dimensional kernels, grid, collisions and particle ABI.
#![doc = include_str!("../../docs/3d.md")]
mod compute;
mod environment;
mod gpu;
mod recycling;
mod render;
mod settings;
#[cfg(test)]
mod tests;

use bevy::{app::PluginGroupBuilder, prelude::*};
pub use compute::{Fluid3dComputeContext, Fluid3dComputePass, Fluid3dComputePasses};
pub use environment::{Fluid3dCollider, Fluid3dFaucet, MAX_COLLIDERS_3D};
pub use gpu::{Fluid3dPlugin, Fluid3dSystems, GpuFluid3d, GpuFluids3d};
pub use recycling::{Fluid3dRecycling, Fluid3dRecyclingPlugin, MAX_RECYCLING_REGIONS_3D};
pub use render::{Fluid3dRender, Fluid3dRenderPlugin};
pub use settings::{Fluid3d, Fluid3dBounds, Fluid3dError, Fluid3dSettings, FluidParticle3d};

/// Simulation plus depth-aware default rendering. Add after DefaultPlugins.
pub struct Fluid3dPlugins;
impl PluginGroup for Fluid3dPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(Fluid3dPlugin)
            .add(Fluid3dRenderPlugin)
    }
}
