//! Independent 2D implementation. GPU buffer layouts here are not shared with 3D.

mod compute;
mod environment;
mod gpu;
mod recycling;
mod render;
mod settings;

#[cfg(test)]
mod tests;

pub use compute::{Fluid2dComputeContext, Fluid2dComputePass, Fluid2dComputePasses};
pub use environment::{Fluid2dCollider, Fluid2dFaucet, MAX_COLLIDERS_2D};
pub use gpu::{Fluid2dPlugin, Fluid2dSystems, GpuFluid2d, GpuFluids2d};
pub use recycling::{Fluid2dRecycling, Fluid2dRecyclingPlugin, MAX_RECYCLING_REGIONS_2D};
pub use render::{Fluid2dBundle, Fluid2dRenderPlugin, Fluid2dStyle, Fluid2dSurface};
pub use settings::{Fluid2d, Fluid2dError, Fluid2dSettings, FluidParticle2d};

use bevy::{app::PluginGroupBuilder, prelude::*};

/// Compute and default rendering plugins. Add after Bevy's `DefaultPlugins`.
pub struct Fluid2dPlugins;

impl PluginGroup for Fluid2dPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(Fluid2dPlugin)
            .add(Fluid2dRenderPlugin)
    }
}
