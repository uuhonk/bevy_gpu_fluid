use std::{fmt, sync::Arc};

use super::{Fluid3dCollider, Fluid3dFaucet, MAX_COLLIDERS_3D};
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

/// GPU particle ABI: position.xyz at byte 0, velocity.xyz at byte 16, stride 32 bytes.
/// The fourth component of both vectors is reserved and initialized to zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct FluidParticle3d {
    pub position: [f32; 4],
    pub velocity: [f32; 4],
}

impl FluidParticle3d {
    pub fn new(position: Vec3) -> Self {
        Self::with_velocity(position, Vec3::ZERO)
    }
    pub fn with_velocity(position: Vec3, velocity: Vec3) -> Self {
        Self {
            position: position.extend(0.0).to_array(),
            velocity: velocity.extend(0.0).to_array(),
        }
    }
    pub fn position_vec3(&self) -> Vec3 {
        Vec4::from_array(self.position).truncate()
    }
    pub fn velocity_vec3(&self) -> Vec3 {
        Vec4::from_array(self.velocity).truncate()
    }
    /// Cubic lattice, X fastest, then Y, then Z. Origin is the first center.
    pub fn lattice(origin: Vec3, size: UVec3, spacing: f32) -> Vec<Self> {
        (0..size.z)
            .flat_map(|z| {
                (0..size.y).flat_map(move |y| {
                    (0..size.x).map(move |x| {
                        Self::new(origin + Vec3::new(x as f32, y as f32, z as f32) * spacing)
                    })
                })
            })
            .collect()
    }
}

/// Axis-aligned volume in simulation/world coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fluid3dBounds {
    pub min: Vec3,
    pub max: Vec3,
}
impl Fluid3dBounds {
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
}

/// Weakly compressible SPH settings, in simulation/world units.
#[derive(Clone, Debug, PartialEq)]
pub struct Fluid3dSettings {
    pub bounds: Fluid3dBounds,
    /// Kernel support radius, also the grid cell width.
    pub smoothing_radius: f32,
    /// Wall collision radius. Independent from the visual particle radius.
    pub particle_radius: f32,
    pub particle_mass: f32,
    pub rest_density: f32,
    /// Linear equation of state: p = stiffness * max(density - rest_density, 0).
    pub stiffness: f32,
    pub viscosity: f32,
    pub gravity: Vec3,
    /// Normal velocity retained on wall collision, in [0, 1].
    pub restitution: f32,
    pub time_step: f32,
    /// Maximum fixed substeps per rendered frame. Excess time is dropped.
    pub max_substeps: u32,
    /// Numerical safety caps; these do not replace a stable time step.
    pub max_speed: f32,
    pub max_acceleration: f32,
}

impl Default for Fluid3dSettings {
    fn default() -> Self {
        Self {
            bounds: Fluid3dBounds::new(Vec3::new(-4.0, -3.0, -3.0), Vec3::new(4.0, 3.0, 3.0)),
            smoothing_radius: 0.32,
            particle_radius: 0.07,
            particle_mass: 4.096,
            rest_density: 1000.0,
            stiffness: 200.0,
            viscosity: 0.1,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            restitution: 0.25,
            time_step: 1.0 / 240.0,
            max_substeps: 8,
            max_speed: 50.0,
            max_acceleration: 10_000.0,
        }
    }
}

/// Invalid configuration or seed data. GPU-specific limits are checked on allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fluid3dError(pub(crate) &'static str);

impl fmt::Display for Fluid3dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Fluid3dError {}

impl Fluid3dSettings {
    pub fn validate(&self) -> Result<(), Fluid3dError> {
        let positive = [
            self.smoothing_radius,
            self.particle_radius,
            self.particle_mass,
            self.rest_density,
            self.time_step,
            self.max_speed,
            self.max_acceleration,
        ];
        if positive.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(Fluid3dError(
                "radii, mass, density, time step and safety caps must be finite and positive",
            ));
        }
        if !self.bounds.min.is_finite()
            || !self.bounds.max.is_finite()
            || !self.bounds.size().is_finite()
            || self.bounds.size().min_element() <= 2.0 * self.particle_radius
        {
            return Err(Fluid3dError(
                "bounds must be finite and wider than two collision radii",
            ));
        }
        if !self.gravity.is_finite()
            || !self.stiffness.is_finite()
            || self.stiffness < 0.0
            || !self.viscosity.is_finite()
            || self.viscosity < 0.0
            || !self.restitution.is_finite()
            || !(0.0..=1.0).contains(&self.restitution)
            || !(1..=64).contains(&self.max_substeps)
        {
            return Err(Fluid3dError(
                "invalid gravity, stiffness, viscosity, restitution or substep count (1..=64)",
            ));
        }
        let cells = (self.bounds.size() / self.smoothing_radius)
            .ceil()
            .max(Vec3::ONE);
        // Bound dense-grid memory as well as float-to-integer conversion.
        if !cells.is_finite()
            || cells.max_element() > 1_048_576.0
            || cells.x as u64 * cells.y as u64 * cells.z as u64 > 16_777_216
        {
            return Err(Fluid3dError(
                "grid exceeds 16M cells; increase smoothing radius or reduce bounds",
            ));
        }
        Ok(())
    }

    pub(crate) fn grid_size(&self) -> UVec3 {
        (self.bounds.size() / self.smoothing_radius)
            .ceil()
            .max(Vec3::ONE)
            .as_uvec3()
    }
}

/// One independent simulation. Multiple entities are supported.
///
/// Seed particles are uploaded only on creation/reset, never read back each frame.
/// Change settings through [`Self::set_settings`], pause with [`Self::set_paused`].
#[derive(Component, Clone)]
pub struct Fluid3d {
    pub(crate) settings: Fluid3dSettings,
    pub(crate) seed: Arc<[FluidParticle3d]>,
    pub(crate) revision: u64,
    pub(crate) paused: bool,
    pub(crate) capacity: u32,
    pub(crate) faucet: Option<Fluid3dFaucet>,
    pub(crate) colliders: Arc<[Fluid3dCollider]>,
}

impl Fluid3d {
    pub fn new(
        settings: Fluid3dSettings,
        particles: Vec<FluidParticle3d>,
    ) -> Result<Self, Fluid3dError> {
        settings.validate()?;
        validate_particles(&settings, &particles)?;
        Ok(Self {
            settings,
            capacity: particles.len() as u32,
            seed: particles.into(),
            revision: 0,
            paused: false,
            faucet: None,
            colliders: Arc::from([]),
        })
    }
    /// Starts empty and allocates exactly `particle_limit` GPU slots. Each slot
    /// is released at most once until `reset()`; particles are never recycled.
    pub fn from_faucet(
        settings: Fluid3dSettings,
        particle_limit: u32,
        faucet: Fluid3dFaucet,
    ) -> Result<Self, Fluid3dError> {
        settings.validate()?;
        faucet.validate(&settings)?;
        if particle_limit == u32::MAX {
            return Err(Fluid3dError(
                "particle limit reserves u32::MAX for linked-list sentinel",
            ));
        }
        Ok(Self {
            settings,
            capacity: particle_limit,
            seed: Arc::from([]),
            faucet: Some(faucet),
            colliders: Arc::from([]),
            revision: 0,
            paused: false,
        })
    }
    pub fn settings(&self) -> &Fluid3dSettings {
        &self.settings
    }
    /// Total particle budget. Active count is available from render-world GpuFluid3d.
    pub fn particle_count(&self) -> usize {
        self.capacity as usize
    }
    pub fn faucet(&self) -> Option<&Fluid3dFaucet> {
        self.faucet.as_ref()
    }
    /// Updates/removes the outlet without resetting particles or refilling its budget.
    pub fn set_faucet(&mut self, faucet: Option<Fluid3dFaucet>) -> Result<(), Fluid3dError> {
        if let Some(f) = faucet {
            f.validate(&self.settings)?;
        }
        self.faucet = faucet;
        Ok(())
    }
    pub fn set_faucet_open(&mut self, open: bool) -> Result<(), Fluid3dError> {
        self.faucet
            .as_mut()
            .ok_or(Fluid3dError("simulation has no faucet"))?
            .is_open = open;
        Ok(())
    }
    pub fn colliders(&self) -> &[Fluid3dCollider] {
        &self.colliders
    }
    /// Replaces all solid obstacles transactionally. Shapes share the limit of 16.
    pub fn set_colliders(&mut self, colliders: Vec<Fluid3dCollider>) -> Result<(), Fluid3dError> {
        if colliders.len() > MAX_COLLIDERS_3D {
            return Err(Fluid3dError("at most 16 colliders per simulation"));
        }
        for c in &colliders {
            c.validate()?;
        }
        self.colliders = colliders.into();
        Ok(())
    }
    pub fn is_paused(&self) -> bool {
        self.paused
    }
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Applies settings without resetting positions. A new grid is allocated if needed;
    /// moved walls clamp particles on the next integration step.
    pub fn set_settings(&mut self, settings: Fluid3dSettings) -> Result<(), Fluid3dError> {
        settings.validate()?;
        if let Some(f) = &self.faucet {
            f.validate(&settings)?;
        }
        self.settings = settings;
        Ok(())
    }
    /// Reset to the original seed and clear the accumulator, preserving pause state.
    pub fn reset(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    /// Replaces the seed and requests a GPU reset. Empty seeds are supported.
    pub fn reset_with_particles(
        &mut self,
        particles: Vec<FluidParticle3d>,
    ) -> Result<(), Fluid3dError> {
        validate_particles(&self.settings, &particles)?;
        self.capacity = particles.len() as u32;
        self.seed = particles.into();
        self.reset();
        Ok(())
    }
}

fn validate_particles(
    settings: &Fluid3dSettings,
    particles: &[FluidParticle3d],
) -> Result<(), Fluid3dError> {
    if particles.len() > u32::MAX as usize - 1 {
        return Err(Fluid3dError(
            "too many particles for u32 linked-list indices",
        ));
    }
    let min = settings.bounds.min + Vec3::splat(settings.particle_radius);
    let max = settings.bounds.max - Vec3::splat(settings.particle_radius);
    if particles.iter().any(|p| {
        let pos = p.position_vec3();
        !pos.is_finite()
            || !Vec4::from_array(p.position).is_finite()
            || !Vec4::from_array(p.velocity).is_finite()
            || !p.velocity_vec3().is_finite()
            || pos.cmplt(min).any()
            || pos.cmpgt(max).any()
    }) {
        return Err(Fluid3dError(
            "seed particles must be finite and inside the collision bounds",
        ));
    }
    Ok(())
}
