//! Custom GPU compute passes. Prepare/register in Fluid3dSystems::PrepareCompute.
use super::GpuFluid3d;
use crate::compute::FluidComputeStage;
use bevy::{prelude::*, render::render_resource::*};
use std::any::TypeId;

/// Scoped access during a fixed GPU substep. Particle writes are allowed only here.
/// Retiring a particle requires atomicExchange(activity[i],0) and, if previously alive,
/// atomicSub(live_count,1). Do not activate slots or replenish emission budgets.
pub struct Fluid3dComputeContext<'a> {
    pub entity: Entity,
    pub fluid: &'a GpuFluid3d,
    /// Released slot prefix for this substep, including retired slots. Always test activity.
    pub particle_count: u32,
    pub substep_index: u64,
    pub encoder: &'a mut CommandEncoder,
    pub pipelines: &'a PipelineCache,
}

/// Own pipelines/bind groups in an implementation; no private SPH bindings are exposed.
pub trait Fluid3dComputePass: Send + Sync + 'static {
    /// Returning false stalls this instance without accumulating catch-up time.
    fn is_ready(&self, _entity: Entity, _pipelines: &PipelineCache) -> bool {
        true
    }
    fn encode(&self, stage: FluidComputeStage, context: &mut Fluid3dComputeContext<'_>);
}

/// Render-world pass registry, executed in insertion order at each substep stage.
/// One entry per Rust type; insert replaces its prepared state without reordering.
#[derive(Resource, Default)]
pub struct Fluid3dComputePasses(Vec<(TypeId, Box<dyn Fluid3dComputePass>)>);
impl Fluid3dComputePasses {
    pub fn insert<P: Fluid3dComputePass>(&mut self, pass: P) {
        let id = TypeId::of::<P>();
        if let Some((_, old)) = self.0.iter_mut().find(|(key, _)| *key == id) {
            *old = Box::new(pass);
        } else {
            self.0.push((id, Box::new(pass)));
        }
    }
    pub fn remove<P: Fluid3dComputePass>(&mut self) {
        self.0.retain(|(key, _)| *key != TypeId::of::<P>());
    }
    pub(crate) fn ready(&self, entity: Entity, pipelines: &PipelineCache) -> bool {
        self.0
            .iter()
            .all(|(_, pass)| pass.is_ready(entity, pipelines))
    }
    pub(crate) fn encode(&self, stage: FluidComputeStage, context: &mut Fluid3dComputeContext<'_>) {
        for (_, pass) in &self.0 {
            pass.encode(stage, context);
        }
    }
}
