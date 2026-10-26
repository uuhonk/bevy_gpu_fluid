//! Shared scheduling stages for optional, GPU-resident simulation extensions.
#![doc = include_str!("../docs/compute.md")]

/// Called inside each fixed substep, with a GPU pass boundary between phases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FluidComputeStage {
    /// Newly emitted particles exist; the neighbor grid has not been built yet.
    BeforeSubstep,
    /// SPH accelerations are ready; extensions may add forces before integration.
    BeforeIntegrate,
    /// Integration/collision is complete; changes are visible to rendering and the next substep.
    AfterSubstep,
}
