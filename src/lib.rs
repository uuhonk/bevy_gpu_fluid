//! Independent 2D and 3D GPU SPH fluids for Bevy 0.20.
//!
//! Add [`prelude::Fluid2dPlugins`] after Bevy's `DefaultPlugins`, then spawn a
//! [`prelude::Fluid2dBundle`]. Shaders are embedded; no external assets are needed.
//! For a custom renderer, add only [`prelude::Fluid2dPlugin`] and access
//! [`fluid2d::GpuFluids2d`] in the render world, after [`fluid2d::Fluid2dSystems::Simulate`].
//!
//! For 3D, add [`prelude::Fluid3dPlugins`], create a `Camera3d`, and spawn
//! [`prelude::Fluid3d`] with [`prelude::Fluid3dRender`]. Initialize particles with
//! [`prelude::FluidParticle3d`] or use [`prelude::Fluid3d::from_faucet`].

#![doc = include_str!("../README.md")]

pub mod compute;
pub mod fluid2d;
pub mod fluid3d;

/// Types needed by most applications.
pub mod prelude {
    pub use crate::compute::FluidComputeStage;
    pub use crate::fluid2d::{
        Fluid2d, Fluid2dBundle, Fluid2dCollider, Fluid2dComputeContext, Fluid2dComputePass,
        Fluid2dComputePasses, Fluid2dError, Fluid2dFaucet, Fluid2dPlugin, Fluid2dPlugins,
        Fluid2dRecycling, Fluid2dRecyclingPlugin, Fluid2dRenderPlugin, Fluid2dSettings,
        Fluid2dStyle, Fluid2dSurface, FluidParticle2d, MAX_COLLIDERS_2D, MAX_RECYCLING_REGIONS_2D,
    };
    pub use crate::fluid3d::{
        Fluid3d, Fluid3dBounds, Fluid3dCollider, Fluid3dComputeContext, Fluid3dComputePass,
        Fluid3dComputePasses, Fluid3dError, Fluid3dFaucet, Fluid3dPlugin, Fluid3dPlugins,
        Fluid3dRecycling, Fluid3dRecyclingPlugin, Fluid3dRender, Fluid3dRenderPlugin,
        Fluid3dSettings, FluidParticle3d, MAX_COLLIDERS_3D, MAX_RECYCLING_REGIONS_3D,
    };
}
